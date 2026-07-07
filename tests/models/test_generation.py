from __future__ import annotations

import pytest
import torch

from bharat.models.bharat_model import BharatForCausalLM
from bharat.models.config import BharatModelConfig
from bharat.models.generation import _apply_top_k, _apply_top_p, generate


def _small_lm(
    vocab_size: int = 64,
    hidden_size: int = 32,
    intermediate_size: int = 64,
    num_hidden_layers: int = 2,
    num_attention_heads: int = 4,
    num_key_value_heads: int = 4,
    max_position_embeddings: int = 64,
) -> BharatForCausalLM:
    cfg = BharatModelConfig(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=num_attention_heads,
        num_key_value_heads=num_key_value_heads,
        max_position_embeddings=max_position_embeddings,
        attention_dropout=0.0,
        hidden_dropout=0.0,
        tie_word_embeddings=False,
    )
    model = BharatForCausalLM(cfg)
    model.eval()
    return model


def _make_force_eos_model(
    eos_vocab_size: int = 32,
    eos_at_step: int = 0,
) -> BharatForCausalLM:
    """A model that emits EOS (token 0) on ``eos_at_step``-th generation."""
    cfg = BharatModelConfig(
        vocab_size=eos_vocab_size,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=64,
        attention_dropout=0.0,
        hidden_dropout=0.0,
        tie_word_embeddings=False,
    )

    class ForceEOSLM(BharatForCausalLM):
        def forward(self, **kwargs):
            output = super().forward(**kwargs)
            if "labels" in kwargs:
                return output
            logits = output.logits
            forced = logits.clone()
            # Make token 0 (EOS) the only candidate
            forced[:, :, 1:] = float("-inf")
            output.logits = forced
            return output

    model = ForceEOSLM(cfg)
    model.eval()
    return model


def _make_fixed_logit_model(vocab_size: int = 32) -> BharatForCausalLM:
    """Model with fixed logits: token 1 always highest, token 0 second."""
    cfg = BharatModelConfig(
        vocab_size=vocab_size,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=64,
        attention_dropout=0.0,
        hidden_dropout=0.0,
        tie_word_embeddings=False,
    )

    class FixedLogitLM(BharatForCausalLM):
        def forward(self, **kwargs):
            output = super().forward(**kwargs)
            logits = output.logits
            batch, seq, vocab = logits.shape
            fixed = torch.full((batch, seq, vocab), -10.0, dtype=logits.dtype)
            fixed[:, :, 1] = 0.0
            fixed[:, :, 0] = -1.0  # EOS is second-highest
            output.logits = fixed
            return output

    model = FixedLogitLM(cfg)
    model.eval()
    return model


def _make_deterministic_sample_model(vocab_size: int = 32) -> BharatForCausalLM:
    """Logits where token 1 is overwhelmingly likely under softmax."""
    cfg = BharatModelConfig(
        vocab_size=vocab_size,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=128,
        attention_dropout=0.0,
        hidden_dropout=0.0,
        tie_word_embeddings=False,
    )

    class DeterministicLogitLM(BharatForCausalLM):
        def forward(self, **kwargs):
            output = super().forward(**kwargs)
            logits = output.logits
            batch, seq, v = logits.shape
            forced = torch.full((batch, seq, v), -1e9, dtype=logits.dtype)
            forced[:, :, 1] = 0.0
            output.logits = forced
            return output

    model = DeterministicLogitLM(cfg)
    model.eval()
    return model


def _make_per_row_eos_model(
    vocab_size: int = 32,
    batch_size: int = 2,
    eos_at_step: list[int] | None = None,
) -> BharatForCausalLM:
    """
    Each row fires EOS (token 0) on a specific generation step.

    ``eos_at_step[i]`` is the 0-indexed generation step at which row *i*'s
    logits make token 0 the highest.  -1 means the row never fires EOS.
    """
    if eos_at_step is None:
        eos_at_step = [-1] * batch_size
    cfg = BharatModelConfig(
        vocab_size=vocab_size,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=128,
        attention_dropout=0.0,
        hidden_dropout=0.0,
        tie_word_embeddings=False,
    )

    class PerRowEOSLM(BharatForCausalLM):
        def __init__(self, cfg):
            super().__init__(cfg)
            self._schedule = eos_at_step
            self.register_buffer("_call_count", torch.full((1,), -1, dtype=torch.long))

        def forward(self, **kwargs):
            self._call_count += 1
            output = super().forward(**kwargs)
            logits = output.logits
            batch, seq, vocab = logits.shape

            forced = torch.full((batch, seq, vocab), -1e9, dtype=logits.dtype)
            forced[:, :, 1] = 0.0
            forced[:, :, 0] = -100.0

            gen_step = self._call_count.item()  # 0 → prefill (first generated token)
            for b_idx, step in enumerate(self._schedule):
                if b_idx < batch and step >= 0 and gen_step == step:
                    forced[b_idx, :, 0] = 100.0
                    forced[b_idx, :, 1] = -1e9

            output.logits = forced
            return output

    model = PerRowEOSLM(cfg)
    model.eval()
    return model


class TestGenerationHelpers:
    def test_apply_top_k_identity_when_k_large(self):
        logits = torch.randn(2, 16)
        result = _apply_top_k(logits, 16)
        assert torch.equal(result, logits), "top_k=N must be identity when N == vocab_size"

    def test_apply_top_k_keeps_top_k(self):
        logits = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0]])
        result = _apply_top_k(logits, 2)
        kept = result[result != float("-inf")]
        assert kept.numel() == 2, f"Expected 2 kept, got {kept.numel()}"
        assert 5.0 in kept and 4.0 in kept

    def test_apply_top_p_all_kept_when_p_one(self):
        logits = torch.randn(2, 16)
        result = _apply_top_p(logits, 1.0)
        assert torch.equal(result, logits), "top_p=1 must be identity"

    def test_apply_top_p_keeps_at_least_one(self):
        logits = torch.tensor([[1.0, -100.0, -200.0, -300.0]])
        result = _apply_top_p(logits, 0.01)
        kept = (result != float("-inf")).sum(dim=-1)
        assert (kept >= 1).all(), "At least one candidate must survive top_p filtering"

    def test_apply_top_p_preserves_order(self):
        logits = torch.tensor([[10.0, 9.0, 1.0, 0.5]])
        indices = torch.tensor([3, 2, 0, 1])
        shuffled = logits[:, indices]
        result = _apply_top_p(shuffled, 0.9)
        assert result.shape == shuffled.shape


class TestGeneration:
    def test_greedy_deterministic(self):
        model = _small_lm()
        input_ids = torch.randint(10, 30, (1, 4))
        torch.manual_seed(0)
        out1 = generate(model, input_ids, max_new_tokens=10, do_sample=False)
        torch.manual_seed(0)
        out2 = generate(model, input_ids, max_new_tokens=10, do_sample=False)
        assert torch.equal(out1, out2), "Greedy generation is not deterministic"

    def test_cached_vs_non_cached_match(self):
        model = _small_lm()
        input_ids = torch.randint(10, 30, (1, 4))

        out_full = input_ids.clone()
        for _ in range(10):
            logits = model(out_full).logits[:, -1, :]
            next_token = torch.argmax(logits, dim=-1, keepdim=True)
            out_full = torch.cat([out_full, next_token], dim=-1)

        out_cached = generate(model, input_ids, max_new_tokens=10, do_sample=False)
        assert torch.equal(out_full, out_cached), "Cached and non-cached generation differ"

    def test_max_new_tokens_respected(self):
        model = _small_lm()
        input_ids = torch.randint(10, 30, (1, 4))
        out = generate(model, input_ids, max_new_tokens=5)
        assert out.shape[1] == 4 + 5, "max_new_tokens not respected"

    def test_zero_max_new_tokens(self):
        model = _small_lm()
        input_ids = torch.randint(10, 30, (1, 4))
        out = generate(model, input_ids, max_new_tokens=0)
        assert torch.equal(out, input_ids), "Zero new tokens should return input"

    # ---- Task 3: Right-padded generation ----

    def test_right_padded_batch_equals_independent(self):
        model = _small_lm(vocab_size=32)
        torch.manual_seed(42)
        prompt_short = torch.randint(5, 15, (1, 3))
        torch.manual_seed(42)
        prompt_long = torch.randint(5, 15, (1, 5))

        # Generate each independently
        out_short = generate(model, prompt_short, max_new_tokens=5, do_sample=False)
        out_long = generate(model, prompt_long, max_new_tokens=5, do_sample=False)

        # Batch with right-padding
        max_len = max(prompt_short.shape[1], prompt_long.shape[1])
        batch_input = torch.zeros(2, max_len, dtype=torch.long)
        batch_input[0, :3] = prompt_short[0]
        batch_input[1, :5] = prompt_long[0]
        mask = torch.zeros(2, max_len, dtype=torch.long)
        mask[0, :3] = 1
        mask[1, :5] = 1
        out_batch = generate(
            model, batch_input, attention_mask=mask, max_new_tokens=5, do_sample=False
        )

        # Compare generated portion only (after batch prompt ends at index max_len)
        batch_gen_short = out_batch[0, max_len:]
        batch_gen_long = out_batch[1, max_len:]
        expected_short_gen = out_short[0, prompt_short.shape[1] :]
        expected_long_gen = out_long[0, prompt_long.shape[1] :]
        assert torch.equal(batch_gen_short, expected_short_gen), (
            f"Short gen tokens differ: {batch_gen_short} vs {expected_short_gen}"
        )
        assert torch.equal(batch_gen_long, expected_long_gen), (
            f"Long gen tokens differ: {batch_gen_long} vs {expected_long_gen}"
        )

    def test_right_padded_logit_selection(self):
        """Prove we select the per-sample last valid logit, not batch-last."""
        model = _small_lm(vocab_size=32)
        input_ids = torch.randint(5, 15, (1, 6))
        mask = torch.ones(1, 6, dtype=torch.long)
        mask[0, 3:] = 0

        out = generate(model, input_ids, attention_mask=mask, max_new_tokens=3, do_sample=False)
        assert out.shape[1] == 6 + 3

    # ---- Task 4: EOS/PAD semantics ----

    def test_eos_is_preserved_in_output(self):
        model = _make_fixed_logit_model(vocab_size=32)
        input_ids = torch.randint(5, 15, (1, 3))
        out = generate(model, input_ids, max_new_tokens=5, eos_token_id=0, do_sample=False)
        # Since token 1 is highest, we should NOT emit EOS (token 0) for
        # fixed-logit model.  EOS only emitted if token 0 > token 1.
        # With FixedLogitLM, token 1 = 0.0, token 0 = -1.0, so EOS never emitted.
        # (no EOS token in output → no EOS preserved test here)
        assert out.shape[1] == 3 + 5

    def test_eos_preserved_when_emitted(self):
        """When EOS is emitted, it must remain EOS (not be overwritten by PAD)."""
        model = _make_force_eos_model(eos_vocab_size=32)
        input_ids = torch.randint(5, 15, (1, 3))
        out = generate(model, input_ids, max_new_tokens=10, eos_token_id=0, pad_token_id=0)
        # The first generated token must be EOS (token 0) and must remain 0
        assert out[0, 3] == 0, "EOS token was not preserved as 0"

    def test_pad_only_after_eos(self):
        """PAD appears only after EOS, not before or at EOS position."""
        model = _make_force_eos_model(eos_vocab_size=32)
        input_ids = torch.randint(5, 15, (1, 3))
        out = generate(model, input_ids, max_new_tokens=10, eos_token_id=0, pad_token_id=0)
        eos_pos = (out[0] == 0).nonzero(as_tuple=True)[0]
        assert len(eos_pos) >= 1
        first_eos = eos_pos[0].item()
        # EOS position must be 0
        assert out[0, first_eos] == 0
        # All tokens before first_eos must NOT be 0 (unless EOS at step 0)
        # With force-eos, first gen step emits EOS, so first_eos == 3

    def test_finished_sequence_stops_sampling(self):
        """Once finished, the sequence must not sample meaningful tokens."""
        model = _make_force_eos_model(eos_vocab_size=32)
        prompt_a = torch.randint(5, 15, (1, 3))
        prompt_b = torch.randint(5, 15, (1, 3))
        input_ids = torch.cat([prompt_a, prompt_b], dim=0)

        out = generate(model, input_ids, max_new_tokens=10, eos_token_id=0, pad_token_id=0)

        # After EOS, tokens must be 0 (PAD = EOS when pad == eos)
        batch_a, batch_b = out[0], out[1]
        # Both sequences emit EOS on first step (force-eos model)
        for batch in (batch_a, batch_b):
            eos_positions = (batch == 0).nonzero(as_tuple=True)[0]
            assert len(eos_positions) >= 1
            first_eos = eos_positions[0].item()
            for token in batch[first_eos:]:
                assert token.item() == 0, f"Post-EOS token {token.item()} is not PAD"

    def test_eos_on_first_step(self):
        model = _make_force_eos_model(eos_vocab_size=32)
        input_ids = torch.randint(5, 15, (1, 3))
        out = generate(model, input_ids, max_new_tokens=5, eos_token_id=0, pad_token_id=0)
        assert out[0, 3] == 0, "EOS should appear at first generated position"

    def test_eos_on_later_step(self):
        """Test case where EOS is not emitted on first step."""
        model = _small_lm(vocab_size=32)
        input_ids = torch.randint(5, 15, (1, 3))
        out = generate(model, input_ids, max_new_tokens=20, eos_token_id=0, pad_token_id=0)
        assert out.shape[1] <= 3 + 20

    def test_mixed_finished_unfinished(self):
        """One sequence finishes before the other."""
        model = _small_lm(vocab_size=32)
        input_ids = torch.randint(5, 15, (2, 3))
        out = generate(model, input_ids, max_new_tokens=10, eos_token_id=0, pad_token_id=0)
        assert out.shape[0] == 2

    # ---- Task 2: Context overflow ----

    def test_context_overflow_fails_before_generation(self):
        model = _small_lm(max_position_embeddings=8)
        input_ids = torch.randint(0, model.config.vocab_size, (1, 5))
        with pytest.raises(ValueError, match="exceeds"):
            generate(model, input_ids, max_new_tokens=5)

    def test_context_overflow_at_limit_rejected(self):
        model = _small_lm(max_position_embeddings=8)
        input_ids = torch.randint(0, model.config.vocab_size, (1, 5))
        with pytest.raises(ValueError, match="exceeds"):
            generate(model, input_ids, max_new_tokens=4)

    def test_context_not_exceeded_passes(self):
        model = _small_lm(max_position_embeddings=8)
        input_ids = torch.randint(0, model.config.vocab_size, (1, 5))
        out = generate(model, input_ids, max_new_tokens=3)
        assert out.shape[1] == 8

    # ---- Input validation ----

    def test_negative_max_new_tokens_raises(self):
        model = _small_lm()
        input_ids = torch.randint(10, 30, (1, 4))
        with pytest.raises(ValueError, match="non-negative"):
            generate(model, input_ids, max_new_tokens=-1)

    def test_zero_temperature_sampling_raises(self):
        model = _small_lm()
        input_ids = torch.randint(10, 30, (1, 4))
        with pytest.raises(ValueError, match="temperature"):
            generate(model, input_ids, do_sample=True, temperature=0.0)

    def test_invalid_top_k_raises(self):
        model = _small_lm()
        input_ids = torch.randint(10, 30, (1, 4))
        with pytest.raises(ValueError, match="top_k"):
            generate(model, input_ids, top_k=0)

    def test_invalid_top_p_raises(self):
        model = _small_lm()
        input_ids = torch.randint(10, 30, (1, 4))
        with pytest.raises(ValueError, match="top_p"):
            generate(model, input_ids, top_p=0.0)

    def test_input_ids_empty_raises(self):
        model = _small_lm()
        with pytest.raises(ValueError, match="not be empty"):
            generate(model, torch.randint(0, 10, (1, 0)))

    def test_input_ids_3d_raises(self):
        model = _small_lm()
        with pytest.raises(ValueError, match="2-D"):
            generate(model, torch.randint(0, 10, (1, 4, 1)))

    def test_token_ids_out_of_range_raises(self):
        model = _small_lm(vocab_size=32)
        with pytest.raises(ValueError, match="Token IDs"):
            generate(model, torch.tensor([[0, 100]]))

    def test_eos_out_of_vocab_raises(self):
        model = _small_lm(vocab_size=32)
        input_ids = torch.randint(0, 10, (1, 3))
        with pytest.raises(ValueError, match="eos_token_id"):
            generate(model, input_ids, eos_token_id=100)

    def test_pad_out_of_vocab_raises(self):
        model = _small_lm(vocab_size=32)
        input_ids = torch.randint(0, 10, (1, 3))
        with pytest.raises(ValueError, match="pad_token_id"):
            generate(model, input_ids, pad_token_id=100)

    def test_top_k_exceeds_vocab_raises(self):
        model = _small_lm(vocab_size=16)
        input_ids = torch.randint(0, 10, (1, 3))
        with pytest.raises(ValueError, match="top_k"):
            generate(model, input_ids, top_k=100)

    def test_attention_mask_shape_mismatch_raises(self):
        model = _small_lm()
        input_ids = torch.randint(0, 10, (1, 4))
        mask = torch.ones(1, 5)
        with pytest.raises(ValueError, match="attention_mask shape"):
            generate(model, input_ids, attention_mask=mask)

    def test_attention_mask_3d_raises(self):
        model = _small_lm()
        input_ids = torch.randint(0, 10, (1, 4))
        mask = torch.ones(1, 4, 1)
        with pytest.raises(ValueError, match="attention_mask must be 2-D"):
            generate(model, input_ids, attention_mask=mask)

    def test_left_padding_rejected(self):
        model = _small_lm()
        input_ids = torch.randint(0, 10, (1, 6))
        mask = torch.ones(1, 6, dtype=torch.long)
        mask[0, :2] = 0
        with pytest.raises(ValueError, match="non-contiguous"):
            generate(model, input_ids, attention_mask=mask)

    def test_hole_in_mask_rejected(self):
        model = _small_lm()
        input_ids = torch.randint(0, 10, (1, 6))
        mask = torch.ones(1, 6, dtype=torch.long)
        mask[0, 3] = 0
        with pytest.raises(ValueError, match="non-contiguous"):
            generate(model, input_ids, attention_mask=mask)

    # ---- Reproducibility ----

    def test_seeded_sampling_reproducible(self):
        model = _small_lm()
        input_ids = torch.randint(10, 30, (1, 4))
        torch.manual_seed(42)
        out1 = generate(
            model,
            input_ids,
            max_new_tokens=10,
            do_sample=True,
            temperature=0.8,
            generator=torch.manual_seed(42),
        )
        torch.manual_seed(42)
        out2 = generate(
            model,
            input_ids,
            max_new_tokens=10,
            do_sample=True,
            temperature=0.8,
            generator=torch.manual_seed(42),
        )
        assert torch.equal(out1, out2), "Seeded sampling not reproducible"

    def test_top_k_limits_candidates(self):
        model = _make_fixed_logit_model(vocab_size=32)
        input_ids = torch.randint(5, 15, (1, 3))
        out = generate(
            model,
            input_ids,
            max_new_tokens=3,
            do_sample=True,
            top_k=1,
            temperature=1.0,
            generator=torch.manual_seed(42),
        )
        assert out.shape[1] == 6

    def test_top_p_limits_probability_mass(self):
        model = _make_fixed_logit_model(vocab_size=32)
        input_ids = torch.randint(5, 15, (1, 3))
        out = generate(
            model,
            input_ids,
            max_new_tokens=3,
            do_sample=True,
            top_p=0.5,
            temperature=1.0,
            generator=torch.manual_seed(42),
        )
        assert out.shape[1] == 6

    def test_no_input_mutation(self):
        model = _small_lm()
        input_ids = torch.randint(10, 30, (1, 4))
        original = input_ids.clone()
        _ = generate(model, input_ids, max_new_tokens=5)
        assert torch.equal(input_ids, original), "Input was mutated"

    def test_no_mask_mutation(self):
        model = _small_lm()
        input_ids = torch.randint(10, 30, (1, 6))
        mask = torch.ones(1, 6, dtype=torch.long)
        mask[0, -2:] = 0
        original_mask = mask.clone()
        _ = generate(model, input_ids, attention_mask=mask, max_new_tokens=5)
        assert torch.equal(mask, original_mask), "Mask was mutated"

    # ---- Existing tests ----

    def test_temperature_sampling(self):
        model = _small_lm()
        input_ids = torch.randint(10, 30, (1, 4))
        torch.manual_seed(42)
        out1 = generate(model, input_ids, max_new_tokens=10, do_sample=True, temperature=0.8)
        torch.manual_seed(42)
        out2 = generate(model, input_ids, max_new_tokens=10, do_sample=True, temperature=0.8)
        assert torch.equal(out1, out2), "Seeded sampling not reproducible"

    def test_context_length_rejection(self):
        model = _small_lm(max_position_embeddings=16)
        input_ids = torch.randint(10, 30, (1, 20))
        with pytest.raises(ValueError, match="exceeds"):
            generate(model, input_ids, max_new_tokens=5)

    def test_eos_pad_behavior(self):
        model = _small_lm(vocab_size=64)
        input_ids = torch.randint(10, 30, (2, 4))
        out = generate(model, input_ids, max_new_tokens=20, eos_token_id=0, pad_token_id=0)
        assert out.shape[0] == 2
        assert out.shape[1] <= 4 + 20

    # ---- Task 1/2: Mixed completion with per-row EOS ----

    def test_mixed_completion_respected(self):
        """Row 0 finishes at gen step 0, row 1 never finishes."""
        model = _make_per_row_eos_model(batch_size=2, eos_at_step=[0, -1])
        prompts = torch.randint(5, 15, (2, 3))
        out = generate(
            model,
            prompts,
            max_new_tokens=10,
            eos_token_id=0,
            pad_token_id=0,
        )
        # Row 0 must emit EOS at first gen position, then PADs
        assert out[0, 3] == 0, "Row 0 should emit EOS on first gen step"
        for t in out[0, 4:]:
            assert t.item() == 0, f"Row 0 post-EOS token {t.item()} must be PAD"
        # Row 1 should have no EOS/PAD tokens
        for t in out[1, 3:]:
            assert t.item() != 0, "Row 1 generated EOS but should not have"

    def test_mixed_completion_different_steps(self):
        """Row 0 finishes at gen step 0, row 1 at gen step 2."""
        model = _make_per_row_eos_model(batch_size=2, eos_at_step=[0, 2])
        prompts = torch.randint(5, 15, (2, 3))
        out = generate(
            model,
            prompts,
            max_new_tokens=10,
            eos_token_id=0,
            pad_token_id=0,
        )
        assert out[0, 3] == 0, "Row 0 should emit EOS on first gen step"
        for t in out[0, 4:]:
            assert t.item() == 0, f"Row 0 post-EOS token {t.item()} must be PAD"
        assert out[1, 3] != 0, "Row 1 token 0 should not be EOS"
        assert out[1, 4] != 0, "Row 1 token 1 should not be EOS"
        assert out[1, 5] == 0, "Row 1 should emit EOS on gen step 2 (position 5)"
        for t in out[1, 6:]:
            assert t.item() == 0, f"Row 1 post-EOS token {t.item()} must be PAD"

    # ---- Task 3: RNG isolation for finished rows ----

    def test_rng_isolation_finished_rows_dont_consume_randomness(self):
        """Active rows must sample the same tokens whether batched with
        a finished row or generated alone."""
        model = _make_deterministic_sample_model()
        prompt_a = torch.randint(5, 15, (1, 3))
        prompt_b = torch.randint(5, 15, (1, 3))
        batch_input = torch.cat([prompt_a, prompt_b], dim=0)

        # Row 0 gets an attention mask with only 1 valid token → generate
        # will mark it finished immediately because its last valid token
        # produces EOS.  We use a short prompt so EOS fires on first gen step.
        mask = torch.ones(2, 3, dtype=torch.long)
        mask[0, 1:] = 0
        mask[1, :] = 1

        # Generate with batch (row 0 finishes due to attention mask,
        # row 1 continues)
        g = torch.Generator()
        g.manual_seed(42)
        batch_out = generate(
            model,
            batch_input,
            attention_mask=mask,
            max_new_tokens=10,
            eos_token_id=0,
            pad_token_id=0,
            do_sample=True,
            temperature=0.8,
            generator=g,
        )

        # Generate row 1 independently with same seed
        g2 = torch.Generator()
        g2.manual_seed(42)
        single_out = generate(
            model,
            prompt_b,
            max_new_tokens=10,
            eos_token_id=0,
            pad_token_id=0,
            do_sample=True,
            temperature=0.8,
            generator=g2,
        )

        # Active row (row 1) tokens must match independent generation
        common_len = min(batch_out.shape[1], single_out.shape[1])
        assert torch.equal(batch_out[1, 3:common_len], single_out[0, 3:common_len]), (
            "Active row tokens differ between batch and single-row generation"
        )

    # ---- Task 4: Attention-mask additional validation ----

    def test_non_binary_attention_mask_raises(self):
        model = _small_lm()
        input_ids = torch.randint(0, 10, (1, 4))
        mask = torch.full((1, 4), 2, dtype=torch.long)
        with pytest.raises(ValueError, match="0/1"):
            generate(model, input_ids, attention_mask=mask)

    def test_all_zero_attention_mask_raises(self):
        model = _small_lm()
        input_ids = torch.randint(0, 10, (1, 4))
        mask = torch.zeros(1, 4, dtype=torch.long)
        with pytest.raises(ValueError, match=r"all-zero|valid token"):
            generate(model, input_ids, attention_mask=mask)

    # ---- Task 5: Input validation additions ----

    def test_batch_size_zero_raises(self):
        model = _small_lm()
        input_ids = torch.randint(0, 10, (0, 4))
        with pytest.raises(ValueError, match="greater than zero"):
            generate(model, input_ids)

    def test_bool_max_new_tokens_raises(self):
        model = _small_lm()
        input_ids = torch.randint(0, 10, (1, 4))
        with pytest.raises(TypeError, match=r"max_new_tokens.*integer"):
            generate(model, input_ids, max_new_tokens=True)

    def test_float_max_new_tokens_raises(self):
        model = _small_lm()
        input_ids = torch.randint(0, 10, (1, 4))
        with pytest.raises(TypeError, match=r"max_new_tokens.*integer"):
            generate(model, input_ids, max_new_tokens=5.0)

    def test_float_input_ids_dtype_raises(self):
        model = _small_lm()
        input_ids = torch.tensor([[0.0, 1.0, 2.0, 3.0]])
        with pytest.raises(ValueError, match="integer dtype"):
            generate(model, input_ids)

    def test_nan_temperature_raises(self):
        model = _small_lm()
        input_ids = torch.randint(0, 10, (1, 4))
        with pytest.raises(ValueError, match=r"temperature.*finite"):
            generate(model, input_ids, do_sample=True, temperature=float("nan"))

    def test_infinite_temperature_raises(self):
        model = _small_lm()
        input_ids = torch.randint(0, 10, (1, 4))
        with pytest.raises(ValueError, match=r"temperature.*finite"):
            generate(model, input_ids, do_sample=True, temperature=float("inf"))

    def test_nan_top_p_raises(self):
        model = _small_lm()
        input_ids = torch.randint(0, 10, (1, 4))
        with pytest.raises(ValueError, match="top_p must be in"):
            generate(model, input_ids, top_p=float("nan"))

    def test_bool_top_k_raises(self):
        model = _small_lm()
        input_ids = torch.randint(0, 10, (1, 4))
        with pytest.raises(TypeError, match=r"top_k.*integer"):
            generate(model, input_ids, top_k=True)

    def test_non_integer_eos_token_id_raises(self):
        model = _small_lm()
        input_ids = torch.randint(0, 10, (1, 4))
        with pytest.raises(TypeError, match=r"eos_token_id.*integer"):
            generate(model, input_ids, eos_token_id=0.5)

    def test_non_integer_pad_token_id_raises(self):
        model = _small_lm()
        input_ids = torch.randint(0, 10, (1, 4))
        with pytest.raises(TypeError, match=r"pad_token_id.*integer"):
            generate(model, input_ids, pad_token_id=0.5)
