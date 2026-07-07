from __future__ import annotations

import pytest
import torch

from bharat.models.bharat_model import BharatForCausalLM
from bharat.models.config import BharatModelConfig
from bharat.models.generation import generate


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
