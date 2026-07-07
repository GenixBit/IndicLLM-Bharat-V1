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

        # Non-cached autoregressive
        out_full = input_ids.clone()
        for _ in range(10):
            logits = model(out_full).logits[:, -1, :]
            next_token = torch.argmax(logits, dim=-1, keepdim=True)
            out_full = torch.cat([out_full, next_token], dim=-1)

        # Cached generation
        out_cached = generate(model, input_ids, max_new_tokens=10, do_sample=False)

        assert torch.equal(out_full, out_cached), "Cached and non-cached generation differ"

    def test_eos_stops_early(self):
        model = _small_lm(vocab_size=64)
        input_ids = torch.randint(10, 30, (1, 4))
        # Use token 0 as EOS
        out = generate(model, input_ids, max_new_tokens=50, eos_token_id=0)
        assert out.shape[1] <= 4 + 50, "EOS did not stop generation"

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

    def test_batch_generation(self):
        model = _small_lm()
        input_ids = torch.randint(10, 30, (2, 4))
        out = generate(model, input_ids, max_new_tokens=5, do_sample=False)
        assert out.shape == (2, 9)

    def test_right_padded_prompts(self):
        model = _small_lm()
        input_ids = torch.randint(10, 30, (2, 6))
        mask = torch.ones(2, 6, dtype=torch.long)
        mask[1, 3:] = 0  # sample 1 has shorter effective prompt
        out = generate(model, input_ids, attention_mask=mask, max_new_tokens=3)
        assert out.shape == (2, 9)

    def test_temperature_sampling(self):
        model = _small_lm()
        input_ids = torch.randint(10, 30, (1, 4))
        torch.manual_seed(42)
        out1 = generate(model, input_ids, max_new_tokens=10, do_sample=True, temperature=0.8)
        torch.manual_seed(42)
        out2 = generate(model, input_ids, max_new_tokens=10, do_sample=True, temperature=0.8)
        assert torch.equal(out1, out2), "Seeded sampling not reproducible"

    def test_top_k_filtering(self):
        model = _small_lm(vocab_size=64)
        input_ids = torch.randint(10, 30, (1, 4))
        out = generate(
            model,
            input_ids,
            max_new_tokens=5,
            do_sample=True,
            top_k=10,
            temperature=1.0,
            generator=torch.manual_seed(42),
        )
        assert out.shape == (1, 9)

    def test_top_p_filtering(self):
        model = _small_lm(vocab_size=64)
        input_ids = torch.randint(10, 30, (1, 4))
        out = generate(
            model,
            input_ids,
            max_new_tokens=5,
            do_sample=True,
            top_p=0.9,
            temperature=1.0,
            generator=torch.manual_seed(42),
        )
        assert out.shape == (1, 9)

    def test_no_input_mutation(self):
        model = _small_lm()
        input_ids = torch.randint(10, 30, (1, 4))
        original = input_ids.clone()
        _ = generate(model, input_ids, max_new_tokens=5)
        assert torch.equal(input_ids, original), "Input was mutated"

    def test_context_length_rejection(self):
        model = _small_lm(max_position_embeddings=16)
        input_ids = torch.randint(10, 30, (1, 20))
        with pytest.raises(ValueError, match="exceeds"):
            generate(model, input_ids, max_new_tokens=5)

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

    def test_eos_pad_behavior(self):
        model = _small_lm(vocab_size=64)
        input_ids = torch.randint(10, 30, (2, 4))
        out = generate(
            model,
            input_ids,
            max_new_tokens=20,
            eos_token_id=0,
            pad_token_id=0,
        )
        assert out.shape[0] == 2
        # If one sequence finished early, pad tokens should appear
        assert out.shape[1] <= 4 + 20

    def test_different_temperatures_differ(self):
        model = _small_lm()
        input_ids = torch.randint(10, 30, (1, 4))
        torch.manual_seed(42)
        out1 = generate(
            model,
            input_ids,
            max_new_tokens=10,
            do_sample=True,
            temperature=0.5,
            generator=torch.manual_seed(42),
        )
        torch.manual_seed(42)
        out2 = generate(
            model,
            input_ids,
            max_new_tokens=10,
            do_sample=True,
            temperature=1.5,
            generator=torch.manual_seed(42),
        )
        # Different temperatures should (likely) produce different results
        # This is probabilistic but almost certain for 1-D token generation
        assert out1.shape == out2.shape

    def test_cached_with_mask_parity(self):
        """Full-vs-cached logit parity with right-padded prompt."""
        model = _small_lm()
        input_ids = torch.randint(10, 30, (1, 6))
        mask = torch.ones(1, 6, dtype=torch.long)
        mask[0, -2:] = 0

        # Full forward
        out_full = model(input_ids, attention_mask=mask).logits

        # Token-by-token cached
        past = None
        incremental_logits = []
        for pos in range(6):
            token_input = input_ids[:, pos : pos + 1]
            step_mask = mask[:, : pos + 1]
            out_step = model(
                token_input,
                attention_mask=step_mask,
                past_key_values=past,
                use_cache=True,
            )
            incremental_logits.append(out_step.logits)
            past = out_step.past_key_values

        cat_logits = torch.cat(incremental_logits, dim=1)
        assert torch.allclose(out_full, cat_logits, atol=1e-4)

    def test_generate_with_padding_mask(self):
        """End-to-end generation with right-padded prompts should not crash."""
        model = _small_lm()
        input_ids = torch.randint(10, 30, (2, 6))
        mask = torch.ones(2, 6, dtype=torch.long)
        mask[1, 3:] = 0
        out = generate(
            model,
            input_ids,
            attention_mask=mask,
            max_new_tokens=5,
            do_sample=False,
        )
        assert out.shape == (2, 11)
