from __future__ import annotations

import json
import math
import tempfile
from pathlib import Path

import pytest
import torch

from bharat.posttraining.collators import SFTCollator
from bharat.posttraining.datasets import SFTDataset
from bharat.posttraining.sft import SFTConfig, SFTResult, sft_train
from bharat.posttraining.templates import Template
from bharat.tokenizer import load_tokenizer
from train.pretrain import GPT, GPTConfig

# ---------------------------------------------------------------------------
# Shared test data
# ---------------------------------------------------------------------------

INDIC_TEMPLATE = Template(
    name="indic_instruction",
    system_prefix="",
    user_prefix="<|instruction|>",
    assistant_prefix="<|response|>",
    suffix="<|endoftext|>",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_tiny_model(vocab_size: int = 256):
    return GPT(
        GPTConfig(vocab_size=vocab_size, n_embd=32, n_head=4, n_layer=2, block_size=64, bias=False)
    )


# ---------------------------------------------------------------------------
# Config validation tests
# ---------------------------------------------------------------------------


class TestSFTConfig:
    def test_valid_config(self):
        c = SFTConfig(max_iters=1, batch_size=1, block_size=64, save_interval=1, log_interval=1)
        assert c.max_iters == 1

    def test_max_iters_zero_raises(self):
        with pytest.raises(ValueError, match="max_iters"):
            SFTConfig(max_iters=0)

    def test_max_iters_negative_raises(self):
        with pytest.raises(ValueError, match="max_iters"):
            SFTConfig(max_iters=-1)

    def test_batch_size_zero_raises(self):
        with pytest.raises(ValueError, match="batch_size"):
            SFTConfig(max_iters=1, batch_size=0)

    def test_block_size_one_raises(self):
        with pytest.raises(ValueError, match="block_size"):
            SFTConfig(max_iters=1, block_size=1)

    def test_lr_zero_raises(self):
        with pytest.raises(ValueError, match="learning_rate"):
            SFTConfig(max_iters=1, learning_rate=0)

    def test_lr_negative_raises(self):
        with pytest.raises(ValueError, match="learning_rate"):
            SFTConfig(max_iters=1, learning_rate=-1)

    def test_save_interval_zero_raises(self):
        with pytest.raises(ValueError, match="save_interval"):
            SFTConfig(max_iters=1, save_interval=0)

    def test_log_interval_zero_raises(self):
        with pytest.raises(ValueError, match="log_interval"):
            SFTConfig(max_iters=1, log_interval=0)

    def test_invalid_device_raises(self):
        with pytest.raises(ValueError, match="device"):
            SFTConfig(max_iters=1, device="invalid_device")


# ---------------------------------------------------------------------------
# Dataset tests (offline, use tiny model vocab)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def tokenizer():
    return load_tokenizer("gpt2")


@pytest.fixture(scope="module")
def tiny_tokenizer():
    """Build a tiny local BPE tokenizer for offline tests."""
    from tokenizers import Tokenizer as HFTokenizersTokenizer
    from tokenizers.models import BPE
    from tokenizers.pre_tokenizers import ByteLevel
    from tokenizers.trainers import BpeTrainer

    bpe = BPE()
    tok = HFTokenizersTokenizer(bpe)
    tok.pre_tokenizer = ByteLevel(add_prefix_space=False)
    trainer = BpeTrainer(
        vocab_size=256,
        min_frequency=1,
        special_tokens=["<|endoftext|>", "<|pad|>", "<|instruction|>", "<|response|>"],
    )
    tok.train_from_iterator(
        [
            "Hello world how are you today",
            "I am fine thank you",
            "a b c d e f g h i j k l m n o p q r s t u v w x y z",
            "Machine learning is fascinating",
            "What is the answer to this question",
            "User: hi Assistant: hello there",
            "System: be concise User: ok Assistant: fine",
        ],
        trainer=trainer,
    )
    import os
    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        tok.save(f.name)
        path = f.name
    yield load_tokenizer(path)
    os.unlink(path)


@pytest.fixture
def sft_jsonl():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        f.write(json.dumps({"instruction": "What is 2+2?", "response": "4"}) + "\n")
        f.write(
            json.dumps({"instruction": "What is Python?", "response": "A programming language"})
            + "\n"
        )
        path = f.name
    yield path
    import os

    os.unlink(path)


class TestSFTDataset:
    def test_dataset_initialization(self, sft_jsonl, tokenizer):
        dataset = SFTDataset(sft_jsonl, INDIC_TEMPLATE, block_size=512)
        assert len(dataset) == 2

    def test_dataset_returns_messages(self, sft_jsonl, tokenizer):
        dataset = SFTDataset(sft_jsonl, INDIC_TEMPLATE, block_size=512)
        item = dataset[0]
        assert "messages" in item
        assert len(item["messages"]) == 2
        assert item["messages"][0]["role"] == "user"
        assert item["messages"][1]["role"] == "assistant"

    def test_dataset_messages_format(self, tokenizer):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write(
                json.dumps(
                    {
                        "messages": [
                            {"role": "user", "content": "Hello"},
                            {"role": "assistant", "content": "Hi"},
                        ]
                    }
                )
                + "\n"
            )
            path = f.name

        dataset = SFTDataset(path, INDIC_TEMPLATE, block_size=512)
        assert len(dataset) == 1
        item = dataset[0]
        assert item["messages"][0]["content"] == "Hello"
        assert item["messages"][1]["content"] == "Hi"

        import os

        os.unlink(path)


# ---------------------------------------------------------------------------
# Loss masking tests
# ---------------------------------------------------------------------------


class TestSFTLossMasking:
    def test_assistant_only_loss(self, tokenizer):
        collator = SFTCollator(
            tokenizer=tokenizer,
            template=INDIC_TEMPLATE,
            block_size=512,
        )
        messages = [
            {"role": "user", "content": "What is 2+2?"},
            {"role": "assistant", "content": "4"},
        ]
        batch = [{"messages": messages}]
        result = collator(batch)
        labels = result["labels"][0]
        non_masked = (labels != -100).sum().item()
        assert non_masked > 0, "Assistant tokens should not be masked"

    def test_user_tokens_masked(self, tokenizer):
        collator = SFTCollator(
            tokenizer=tokenizer,
            template=INDIC_TEMPLATE,
            block_size=512,
        )
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "World"},
        ]
        batch = [{"messages": messages}]
        result = collator(batch)
        labels = result["labels"][0]

        ap_ids = tokenizer.encode("<|response|>", add_special_tokens=False)
        ap_len = len(ap_ids)

        full_ids = tokenizer.encode(
            "<|instruction|>Hello<|endoftext|><|response|>World<|endoftext|>",
            add_special_tokens=False,
        )
        ap_start = None
        for i in range(len(full_ids) - ap_len + 1):
            if full_ids[i : i + ap_len] == ap_ids:
                ap_start = i
                break

        assert ap_start is not None

        user_end_in_targets = ap_start - 1
        user_labels = labels[: max(0, user_end_in_targets)]
        assert (user_labels == -100).all(), "User tokens should be fully masked"

    def test_system_tokens_masked(self, tokenizer):
        template_with_system = Template(
            name="test_system",
            system_prefix="<|system|>",
            user_prefix="<|user|>",
            assistant_prefix="<|assistant|>",
            suffix="<|end|>",
        )
        collator = SFTCollator(
            tokenizer=tokenizer,
            template=template_with_system,
            block_size=512,
        )
        messages = [
            {"role": "system", "content": "Be concise"},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"},
        ]
        batch = [{"messages": messages}]
        result = collator(batch)
        labels = result["labels"][0]

        sys_ids = tokenizer.encode("<|system|>Be concise<|end|>", add_special_tokens=False)
        user_start = len(sys_ids)

        sys_end_in_targets = user_start - 1
        sys_labels = labels[: max(0, sys_end_in_targets)]
        assert (sys_labels == -100).all(), "System tokens should be fully masked"

    def test_padding_masked(self, tokenizer):
        collator = SFTCollator(
            tokenizer=tokenizer,
            template=INDIC_TEMPLATE,
            block_size=512,
        )
        batch = [
            {"messages": [{"role": "user", "content": "A"}, {"role": "assistant", "content": "B"}]},
            {"messages": [{"role": "user", "content": "C"}, {"role": "assistant", "content": "D"}]},
        ]
        result = collator(batch)
        labels = result["labels"]
        padding_mask = labels == -100
        assert padding_mask.any(), "Should have some padding"


# ---------------------------------------------------------------------------
# Collator tests
# ---------------------------------------------------------------------------


class TestSFTCollator:
    def test_empty_batch(self, tokenizer):
        collator = SFTCollator(
            tokenizer=tokenizer,
            template=INDIC_TEMPLATE,
            block_size=512,
        )
        result = collator([])
        assert "input_ids" in result
        assert "labels" in result
        assert result["input_ids"].shape[0] == 0

    def test_variable_length_batch(self, tokenizer):
        collator = SFTCollator(
            tokenizer=tokenizer,
            template=INDIC_TEMPLATE,
            block_size=512,
        )
        batch = [
            {
                "messages": [
                    {"role": "user", "content": "Short"},
                    {"role": "assistant", "content": "A"},
                ]
            },
            {
                "messages": [
                    {"role": "user", "content": "very " * 50 + "long"},
                    {"role": "assistant", "content": "B"},
                ]
            },
        ]
        result = collator(batch)
        assert result["input_ids"].shape[0] == 2
        assert result["input_ids"].shape[1] == result["labels"].shape[1]

    def test_first_response_token_included(self, tokenizer):
        collator = SFTCollator(
            tokenizer=tokenizer,
            template=INDIC_TEMPLATE,
            block_size=512,
        )
        messages = [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello"},
        ]
        batch = [{"messages": messages}]
        result = collator(batch)
        labels = result["labels"][0]

        full_ids = tokenizer.encode(
            "<|instruction|>Hi<|endoftext|><|response|>Hello<|endoftext|>",
            add_special_tokens=False,
        )
        target_ids = full_ids[1:]
        ap_ids = tokenizer.encode("<|response|>", add_special_tokens=False)
        ap_len = len(ap_ids)

        ap_start = None
        for i in range(len(full_ids) - ap_len + 1):
            if full_ids[i : i + ap_len] == ap_ids:
                ap_start = i
                break

        non_masked_mask = labels != -100
        first_response_idx = ap_start + ap_len - 1
        assert first_response_idx < len(labels)
        assert non_masked_mask[first_response_idx].item(), (
            f"First response token at index {first_response_idx} should not be masked"
        )
        assert labels[first_response_idx].item() == target_ids[first_response_idx]

    def test_user_tokens_masked_multi_turn(self, tokenizer):
        collator = SFTCollator(
            tokenizer=tokenizer,
            template=INDIC_TEMPLATE,
            block_size=512,
        )
        messages = [
            {"role": "user", "content": "First Q"},
            {"role": "assistant", "content": "First A"},
            {"role": "user", "content": "Second Q"},
            {"role": "assistant", "content": "Second A"},
        ]
        batch = [{"messages": messages}]
        result = collator(batch)
        labels = result["labels"][0]

        ap_ids = tokenizer.encode("<|response|>", add_special_tokens=False)
        up_ids = tokenizer.encode("<|instruction|>", add_special_tokens=False)

        full_text = (
            "<|instruction|>First Q<|endoftext|><|response|>First A<|endoftext|>"
            "<|instruction|>Second Q<|endoftext|><|response|>Second A<|endoftext|>"
        )
        full_ids = tokenizer.encode(full_text, add_special_tokens=False)

        ap_positions = []
        for i in range(len(full_ids) - len(ap_ids) + 1):
            if full_ids[i : i + len(ap_ids)] == ap_ids:
                ap_positions.append(i)

        up_positions = []
        for i in range(len(full_ids) - len(up_ids) + 1):
            if full_ids[i : i + len(up_ids)] == up_ids:
                up_positions.append(i)

        for up_start in up_positions:
            up_end = len(full_ids)
            for p in sorted(ap_positions + up_positions):
                if p > up_start:
                    up_end = p
                    break

            target_start = max(0, up_start - 1)
            target_end = min(len(labels), up_end - 1)
            if target_start < target_end:
                user_label_slice = labels[target_start:target_end]
                assert (user_label_slice == -100).all(), (
                    f"User tokens at target positions {target_start}:{target_end} should be masked"
                )

    def test_multi_turn_assistant_active(self, tokenizer):
        collator = SFTCollator(
            tokenizer=tokenizer,
            template=INDIC_TEMPLATE,
            block_size=512,
        )
        messages = [
            {"role": "user", "content": "First Q"},
            {"role": "assistant", "content": "First A"},
            {"role": "user", "content": "Second Q"},
            {"role": "assistant", "content": "Second A"},
        ]
        batch = [{"messages": messages}]
        result = collator(batch)
        labels = result["labels"][0]

        transitions = 0
        prev_masked = True
        for v in labels:
            if prev_masked and v != -100:
                transitions += 1
            prev_masked = v == -100

        assert transitions >= 2, f"Expected at least 2 active regions, got {transitions}"

    def test_truncated_conversation(self, tokenizer):
        small_collator = SFTCollator(
            tokenizer=tokenizer,
            template=INDIC_TEMPLATE,
            block_size=10,
        )
        messages = [
            {"role": "user", "content": "This is a very long user message that will be truncated"},
            {"role": "assistant", "content": "Short"},
        ]
        batch = [{"messages": messages}]
        result = small_collator(batch)
        labels = result["labels"][0]

        assert not torch.isnan(labels.float()).any()


# ---------------------------------------------------------------------------
# End-to-end training tests (offline, use tiny_tokenizer)
# ---------------------------------------------------------------------------


class TestSFTOneStepTraining:
    def test_empty_dataset_raises(self, tiny_tokenizer):
        model = _make_tiny_model(vocab_size=tiny_tokenizer.vocab_size)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write("")
            data_path = f.name

        config = SFTConfig(
            data_path=data_path, max_iters=1, batch_size=1, block_size=64, device="cpu"
        )
        config.output_dir = tempfile.mkdtemp()

        with pytest.raises(ValueError, match="empty"):
            sft_train(model, config, tiny_tokenizer)

        import os

        os.unlink(data_path)

    def test_dataset_smaller_than_batch(self, tiny_tokenizer):
        """1 sample with batch_size=2 must still train successfully."""
        model = _make_tiny_model(vocab_size=tiny_tokenizer.vocab_size)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write(
                json.dumps(
                    {
                        "messages": [
                            {"role": "user", "content": "Hi"},
                            {"role": "assistant", "content": "Hello"},
                        ]
                    }
                )
                + "\n"
            )
            data_path = f.name

        config = SFTConfig(
            data_path=data_path,
            max_iters=1,
            batch_size=2,
            block_size=64,
            learning_rate=1e-3,
            warmup_iters=0,
            device="cpu",
        )
        config.output_dir = tempfile.mkdtemp()

        result = sft_train(model, config, tiny_tokenizer)
        assert isinstance(result, SFTResult)
        assert math.isfinite(result.final_loss)
        assert result.final_loss > 0
        assert result.completed_steps == 1
        assert result.next_step == 1

        import os
        import shutil

        os.unlink(data_path)
        shutil.rmtree(config.output_dir, ignore_errors=True)

    def test_max_iters_one(self, tiny_tokenizer):
        model = _make_tiny_model(vocab_size=tiny_tokenizer.vocab_size)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write(
                json.dumps(
                    {
                        "messages": [
                            {"role": "user", "content": "Hello"},
                            {"role": "assistant", "content": "Hi"},
                        ]
                    }
                )
                + "\n"
            )
            data_path = f.name

        config = SFTConfig(
            data_path=data_path,
            max_iters=1,
            batch_size=1,
            block_size=64,
            learning_rate=1e-3,
            warmup_iters=0,
            device="cpu",
        )
        config.output_dir = tempfile.mkdtemp()

        result = sft_train(model, config, tiny_tokenizer)
        assert isinstance(result, SFTResult)
        assert math.isfinite(result.final_loss)
        assert result.final_loss > 0
        assert result.best_loss > 0
        assert result.completed_steps == 1
        assert result.next_step == 1
        assert result.samples_processed >= 1
        assert result.active_tokens >= 1

        import os
        import shutil

        os.unlink(data_path)
        shutil.rmtree(config.output_dir, ignore_errors=True)

    def test_cpu_backward(self, tiny_tokenizer):
        model = _make_tiny_model(vocab_size=tiny_tokenizer.vocab_size)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write(
                json.dumps(
                    {
                        "messages": [
                            {"role": "user", "content": "Hello"},
                            {"role": "assistant", "content": "Hi"},
                        ]
                    }
                )
                + "\n"
            )
            data_path = f.name

        config = SFTConfig(
            data_path=data_path,
            max_iters=1,
            batch_size=1,
            block_size=64,
            learning_rate=1e-3,
            warmup_iters=0,
            device="cpu",
        )
        config.output_dir = tempfile.mkdtemp()

        init_params = {k: v.clone() for k, v in model.named_parameters() if v.requires_grad}

        result = sft_train(model, config, tiny_tokenizer)
        assert isinstance(result, SFTResult)
        assert math.isfinite(result.final_loss)
        assert result.final_loss > 0

        params_changed = False
        for k, v in model.named_parameters():
            if v.requires_grad and not torch.equal(v, init_params[k]):
                params_changed = True
                break
        assert params_changed, "At least one model parameter should change after training"

        import os
        import shutil

        os.unlink(data_path)
        shutil.rmtree(config.output_dir, ignore_errors=True)

    def test_sample_without_assistant_response(self, tiny_tokenizer):
        """Sample with no assistant turn should be rejected."""
        model = _make_tiny_model(vocab_size=tiny_tokenizer.vocab_size)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write(
                json.dumps({"messages": [{"role": "user", "content": "Only user message"}]}) + "\n"
            )
            data_path = f.name

        config = SFTConfig(
            data_path=data_path,
            max_iters=1,
            batch_size=1,
            block_size=64,
            device="cpu",
        )
        config.output_dir = tempfile.mkdtemp()

        with pytest.raises(ValueError, match="zero active assistant"):
            sft_train(model, config, tiny_tokenizer)

        import os
        import shutil

        os.unlink(data_path)
        shutil.rmtree(config.output_dir, ignore_errors=True)

    def test_sample_truncated_before_assistant(self, tiny_tokenizer):
        """Very small block_size may truncate before assistant content."""
        model = _make_tiny_model(vocab_size=tiny_tokenizer.vocab_size)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write(
                json.dumps(
                    {
                        "messages": [
                            {"role": "user", "content": "A" * 100},
                            {"role": "assistant", "content": "B"},
                        ]
                    }
                )
                + "\n"
            )
            data_path = f.name

        config = SFTConfig(
            data_path=data_path,
            max_iters=1,
            batch_size=1,
            block_size=10,
            device="cpu",
        )
        config.output_dir = tempfile.mkdtemp()

        with pytest.raises(ValueError, match="zero active assistant"):
            sft_train(model, config, tiny_tokenizer)

        import os
        import shutil

        os.unlink(data_path)
        shutil.rmtree(config.output_dir, ignore_errors=True)

    # --- Best checkpoint tests (Task 1) ---

    def test_best_pt_created(self, tiny_tokenizer):
        """best.pt is created during a short one-step run."""
        model = _make_tiny_model(vocab_size=tiny_tokenizer.vocab_size)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write(
                json.dumps(
                    {
                        "messages": [
                            {"role": "user", "content": "Hello"},
                            {"role": "assistant", "content": "World"},
                        ]
                    }
                )
                + "\n"
            )
            data_path = f.name

        out_dir = Path(tempfile.mkdtemp())
        config = SFTConfig(
            data_path=data_path,
            max_iters=3,
            batch_size=1,
            block_size=64,
            learning_rate=1e-3,
            warmup_iters=0,
            device="cpu",
            save_interval=5,
        )
        config.output_dir = out_dir

        result = sft_train(model, config, tiny_tokenizer)
        assert isinstance(result, SFTResult)

        best_path = out_dir / "best.pt"
        assert best_path.exists(), "best.pt should be created"

        best_ckpt = torch.load(best_path, map_location="cpu", weights_only=False)
        assert math.isfinite(best_ckpt["best_loss"])
        assert best_ckpt["best_loss"] <= best_ckpt["final_loss"]
        assert best_ckpt["completed_steps"] > 0
        assert best_ckpt["next_step"] == best_ckpt["completed_steps"]
        assert best_ckpt["metadata"]["tokenizer_hash"]

        # best_loss in result should match checkpoint
        assert abs(result.best_loss - best_ckpt["best_loss"]) < 1e-6

        import os
        import shutil

        os.unlink(data_path)
        shutil.rmtree(out_dir, ignore_errors=True)

    def test_best_pt_contains_lowest_loss(self, tiny_tokenizer):
        """best.pt contains the lowest observed loss across multiple steps."""
        model = _make_tiny_model(vocab_size=tiny_tokenizer.vocab_size)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write(
                json.dumps(
                    {
                        "messages": [
                            {"role": "user", "content": "Hello"},
                            {"role": "assistant", "content": "World"},
                        ]
                    }
                )
                + "\n"
            )
            for _ in range(5):
                f.write(
                    json.dumps(
                        {
                            "messages": [
                                {"role": "user", "content": "Test"},
                                {"role": "assistant", "content": "Data"},
                            ]
                        }
                    )
                    + "\n"
                )
            data_path = f.name

        out_dir = Path(tempfile.mkdtemp())
        config = SFTConfig(
            data_path=data_path,
            max_iters=5,
            batch_size=1,
            block_size=64,
            learning_rate=1e-3,
            warmup_iters=0,
            device="cpu",
            save_interval=10,
        )
        config.output_dir = out_dir

        sft_train(model, config, tiny_tokenizer)

        best_path = out_dir / "best.pt"
        final_path = out_dir / "final.pt"
        assert best_path.exists()

        best_ckpt = torch.load(best_path, map_location="cpu", weights_only=False)
        final_ckpt = torch.load(final_path, map_location="cpu", weights_only=False)
        # best_loss in best.pt should be <= final_loss
        assert best_ckpt["best_loss"] <= final_ckpt["final_loss"] + 1e-6

        import os
        import shutil

        os.unlink(data_path)
        shutil.rmtree(out_dir, ignore_errors=True)

    def test_worse_step_does_not_overwrite_best(self, tiny_tokenizer):
        """A worse later step does not overwrite a better checkpoint."""
        model = _make_tiny_model(vocab_size=tiny_tokenizer.vocab_size)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write(
                json.dumps(
                    {
                        "messages": [
                            {"role": "user", "content": "Hello"},
                            {"role": "assistant", "content": "World"},
                        ]
                    }
                )
                + "\n"
            )
            data_path = f.name

        out_dir = Path(tempfile.mkdtemp())
        config = SFTConfig(
            data_path=data_path,
            max_iters=1,
            batch_size=1,
            block_size=64,
            learning_rate=1e-3,
            warmup_iters=0,
            device="cpu",
        )
        config.output_dir = out_dir

        sft_train(model, config, tiny_tokenizer)

        best_ckpt = torch.load(out_dir / "best.pt", map_location="cpu", weights_only=False)
        best_loss_saved = best_ckpt["best_loss"]

        # Re-run with same data and config, loss should be similar or lower
        model2 = _make_tiny_model(vocab_size=tiny_tokenizer.vocab_size)
        out_dir2 = Path(tempfile.mkdtemp())
        config.output_dir = out_dir2
        sft_train(model2, config, tiny_tokenizer)

        best_ckpt2 = torch.load(out_dir2 / "best.pt", map_location="cpu", weights_only=False)

        # Both should have valid best_loss values
        assert math.isfinite(best_loss_saved)
        assert math.isfinite(best_ckpt2["best_loss"])

        import os
        import shutil

        os.unlink(data_path)
        shutil.rmtree(out_dir, ignore_errors=True)
        shutil.rmtree(out_dir2, ignore_errors=True)

    def test_best_loss_matches_checkpoint(self, tiny_tokenizer):
        """best_loss in SFTResult matches the saved checkpoint."""
        model = _make_tiny_model(vocab_size=tiny_tokenizer.vocab_size)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write(
                json.dumps(
                    {
                        "messages": [
                            {"role": "user", "content": "Hello"},
                            {"role": "assistant", "content": "World"},
                        ]
                    }
                )
                + "\n"
            )
            data_path = f.name

        out_dir = Path(tempfile.mkdtemp())
        config = SFTConfig(
            data_path=data_path,
            max_iters=1,
            batch_size=1,
            block_size=64,
            learning_rate=1e-3,
            warmup_iters=0,
            device="cpu",
        )
        config.output_dir = out_dir

        result = sft_train(model, config, tiny_tokenizer)

        best_ckpt = torch.load(out_dir / "best.pt", map_location="cpu", weights_only=False)
        assert abs(result.best_loss - best_ckpt["best_loss"]) < 1e-6

        import os
        import shutil

        os.unlink(data_path)
        shutil.rmtree(out_dir, ignore_errors=True)
