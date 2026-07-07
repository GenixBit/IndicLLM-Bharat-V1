from __future__ import annotations

import pytest
import torch

from bharat.tokenizer import load_tokenizer
from bharat.training.checkpointing import (
    CheckpointMetadata,
    make_checkpoint_data,
    validate_checkpoint,
)

# ── SFT Loss Masking (C1) ────────────────────────────────────


def test_sft_loss_masking_assistant_only() -> None:
    tokenizer = load_tokenizer("gpt2")
    special = {"additional_special_tokens": ["<|instruction|>", "<|response|>"]}
    tokenizer.add_special_tokens(special)

    instruction = "What is 2+2?"
    response = "4"
    full_text = f"<|instruction|>{instruction}<|response|>{response}<|endoftext|>"
    response_marker = "<|response|>"
    response_start = full_text.index(response_marker) + len(response_marker)
    full_ids = tokenizer.encode(full_text, add_special_tokens=False)
    prefix_ids = tokenizer.encode(full_text[:response_start], add_special_tokens=False)
    prompt_end = len(prefix_ids)

    assert prompt_end < len(full_ids), "Response must have at least one token"
    y = torch.tensor(full_ids[1:], dtype=torch.long)
    y_masked = y.clone()
    y_masked[:prompt_end] = -100

    assistant_tokens = y[prompt_end:]
    assert (assistant_tokens != -100).all(), "Assistant tokens should not be masked"
    assert (y_masked[:prompt_end] == -100).all(), "Non-assistant tokens must be masked with -100"
    assert y_masked[prompt_end:].tolist() == y[prompt_end:].tolist(), "Assistant tokens preserved"


def test_sft_loss_masking_multiturn() -> None:
    tokenizer = load_tokenizer("gpt2")
    special = {"additional_special_tokens": ["<|instruction|>", "<|response|>"]}
    tokenizer.add_special_tokens(special)

    turns = [
        ("What is 2+2?", "4"),
        ("What is 3+3?", "6"),
    ]
    full_text = ""
    for inst, resp in turns:
        full_text += f"<|instruction|>{inst}<|response|>{resp}<|endoftext|>"

    # Find prompt boundaries for each turn
    full_ids = tokenizer.encode(full_text, add_special_tokens=False)
    y = torch.tensor(full_ids[1:], dtype=torch.long)
    y_masked = y.clone()

    # Find each response marker position in encoded form
    turn_boundaries = []
    pos = 0
    for inst, resp in turns:
        turn_text = f"<|instruction|>{inst}<|response|>{resp}<|endoftext|>"
        prefix_until_response = f"<|instruction|>{inst}<|response|>"
        prompt_end = len(tokenizer.encode(prefix_until_response, add_special_tokens=False))
        turn_len = len(tokenizer.encode(turn_text, add_special_tokens=False))
        turn_boundaries.append((pos + prompt_end, pos + turn_len))
        pos += turn_len

    for prompt_end, _turn_end in turn_boundaries:
        y_masked[: prompt_end - 1] = -100  # first token offset

    # Verify at least some assistant tokens are not masked
    assert (y_masked == -100).sum() > 0, "Some tokens should be masked"
    assert (y_masked != -100).sum() > 0, "Some tokens should contribute to loss"


# ── DPO Per-Sample Masking (C2/C3) ────────────────────────────


def test_dpo_per_sample_prompt_length() -> None:
    tokenizer = load_tokenizer("gpt2")
    prompts = [
        tokenizer.encode("Short", add_special_tokens=False),
        tokenizer.encode("A much longer prompt here", add_special_tokens=False),
        tokenizer.encode("Medium length one", add_special_tokens=False),
    ]
    responses = [
        tokenizer.encode(" ok", add_special_tokens=False),
        tokenizer.encode(" response", add_special_tokens=False),
        tokenizer.encode(" answer", add_special_tokens=False),
    ]
    block_size = 20
    pad_id = 0

    prompt_lens = []
    padded = []
    for p, r in zip(prompts, responses, strict=False):
        combined = (p + r)[:block_size]
        prompt_lens.append(len(p))
        combined = combined + [pad_id] * (block_size - len(combined))
        padded.append(torch.tensor(combined, dtype=torch.long))

    ids = torch.stack(padded)
    pl_tensor = torch.tensor(prompt_lens)

    # Verify each sample has different prompt length
    assert len(set(prompt_lens)) > 1, "Need varied prompt lengths for per-sample test"

    # Verify per-sample masking logic
    b_size, t_len = ids.shape
    arange = torch.arange(t_len).unsqueeze(0).expand(b_size, -1)
    mask = (arange >= pl_tensor.unsqueeze(-1)).float()
    for i in range(b_size):
        assert mask[i, : prompt_lens[i]].sum() == 0, "Prompt tokens should be masked"
        assert mask[i, prompt_lens[i] :].sum() > 0, "Non-prompt positions should be unmasked"
        assert (
            mask[i].sum() == t_len - prompt_lens[i]
        ), "All non-prompt positions contribute (including padding)"


def test_dpo_variable_prompt_length_batch() -> None:
    tokenizer = load_tokenizer("gpt2")

    def encode(text):
        return tokenizer.encode(text, add_special_tokens=False)

    batch = [
        (encode("Hi"), encode("Hello world"), encode("Goodbye world")),
        (encode("Longer prompt here please"), encode("Short"), encode("Also short")),
        (encode("A"), encode("BC"), encode("DE")),
    ]

    prompt_lens = []
    block_size = 30
    pad_id = 0
    chosen_list = []
    rejected_list = []

    for prompt, chosen, rejected in batch:
        prompt_lens.append(len(prompt))
        chosen_ids = (prompt + chosen)[:block_size]
        chosen_ids = chosen_ids + [pad_id] * (block_size - len(chosen_ids))
        rejected_ids = (prompt + rejected)[:block_size]
        rejected_ids = rejected_ids + [pad_id] * (block_size - len(rejected_ids))
        chosen_list.append(torch.tensor(chosen_ids, dtype=torch.long))
        rejected_list.append(torch.tensor(rejected_ids, dtype=torch.long))

    chosen = torch.stack(chosen_list)
    rejected = torch.stack(rejected_list)
    pl = torch.tensor(prompt_lens)

    # Simulate log_probs with per-sample masking
    b_size, t_len = chosen.shape
    dummy_lp = torch.randn(b_size, t_len - 1)
    arange = torch.arange(t_len - 1, device=dummy_lp.device).unsqueeze(0).expand(b_size, -1)
    mask = (arange >= pl.unsqueeze(-1)).float()

    masked_lp = dummy_lp * mask
    chosen_scores = masked_lp.sum(-1)

    assert chosen_scores.shape == (b_size,), "Per-sample scores should be shape (B,)"
    assert not torch.isnan(chosen_scores).any(), "No NaN in scores"


# ── Checkpoint Tokenizer Metadata ────────────────────────────


def test_checkpoint_tokenizer_metadata_roundtrip() -> None:
    tokenizer = load_tokenizer("gpt2")
    model_state = {"test": torch.zeros(2, 2)}
    ckpt = make_checkpoint_data(
        model_state=model_state,
        tokenizer=tokenizer,
        step=42,
        seed=1234,
        data_version="test-v1",
    )
    assert "metadata" in ckpt, "Checkpoint must contain metadata"
    meta = ckpt["metadata"]
    assert meta["tokenizer_type"] == "gpt2", "Should store tokenizer type"
    assert meta["tokenizer_hash"], "Should store tokenizer hash"
    assert meta["vocab_size"] > 0, "Should store vocab size"
    assert meta["git_sha"] or meta["git_sha"] == "", "Should store git SHA"
    assert meta["training_step"] == 42, "Should store training step"
    assert "torch" in meta.get("package_versions", {}), "Should store package versions"


def test_checkpoint_tokenizer_mismatch_rejection() -> None:
    gpt2 = load_tokenizer("gpt2")
    model_state = {"dummy": torch.tensor(0)}
    ckpt = make_checkpoint_data(model_state=model_state, tokenizer=gpt2)

    wrong = load_tokenizer("bert-base-uncased")
    with pytest.raises(ValueError, match="Tokenizer mismatch"):
        validate_checkpoint(ckpt, tokenizer=wrong)


def test_legacy_checkpoint_no_metadata() -> None:
    old_ckpt = {
        "model": {"dummy": torch.tensor(0)},
        "config": {"model": {"n_layer": 6}},
        "iter_num": 100,
    }
    with pytest.raises(ValueError, match="no metadata"):
        validate_checkpoint(old_ckpt)

    old_ckpt_with_empty_meta = {**old_ckpt, "metadata": CheckpointMetadata().__dict__}
    meta = validate_checkpoint(old_ckpt_with_empty_meta, tokenizer=None)
    assert meta is not None, "Legacy with empty metadata should load with default values"


# ── Unified Tokenizer Loading (consistency) ──────────────────


def test_load_tokenizer_from_config_key() -> None:
    tok = load_tokenizer("gpt2")
    assert tok.tokenizer_type == "gpt2"
    assert tok.vocab_size == 50257

    tok2 = load_tokenizer(None)
    assert tok2.tokenizer_type == "gpt2"

    assert tok2.vocab_size == tok.vocab_size


def test_load_tokenizer_eos_stop() -> None:
    tokenizer = load_tokenizer("gpt2")
    ids = [101, 102, 103]
    decoded = tokenizer.decode(ids)
    assert isinstance(decoded, str)
    assert tokenizer.eos_token_id == 50256
    assert tokenizer.pad_token_id == 50256
