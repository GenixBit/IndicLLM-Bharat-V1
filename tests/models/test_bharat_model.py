from __future__ import annotations

import pytest
import torch

from bharat.models.bharat_model import BharatDecoderLayer, BharatForCausalLM, BharatModel
from bharat.models.config import BharatModelConfig
from bharat.models.outputs import BharatCausalLMOutput, BharatModelOutput


def _small_config(
    vocab_size: int = 128,
    hidden_size: int = 64,
    intermediate_size: int = 256,
    num_hidden_layers: int = 2,
    num_attention_heads: int = 4,
    num_key_value_heads: int = 4,
    max_position_embeddings: int = 64,
    tie_word_embeddings: bool = True,
) -> BharatModelConfig:
    return BharatModelConfig(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=num_attention_heads,
        num_key_value_heads=num_key_value_heads,
        max_position_embeddings=max_position_embeddings,
        attention_dropout=0.0,
        hidden_dropout=0.0,
        tie_word_embeddings=tie_word_embeddings,
    )


class TestBharatDecoderLayer:
    def test_forward_shape(self):
        cfg = _small_config()
        layer = BharatDecoderLayer(cfg)
        x = torch.randn(2, 8, 64)
        out, _cache = layer(x)
        assert out.shape == (2, 8, 64)

    def test_forward_finite(self):
        cfg = _small_config()
        layer = BharatDecoderLayer(cfg)
        x = torch.randn(2, 8, 64)
        out, _cache = layer(x)
        assert torch.isfinite(out).all()

    def test_backward(self):
        cfg = _small_config()
        layer = BharatDecoderLayer(cfg)
        x = torch.randn(2, 8, 64, requires_grad=True)
        out, _cache = layer(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None
        assert torch.isfinite(x.grad).all()

    def test_all_params_grad(self):
        cfg = _small_config()
        layer = BharatDecoderLayer(cfg)
        x = torch.randn(2, 8, 64, requires_grad=True)
        out, _cache = layer(x)
        loss = out.sum()
        loss.backward()
        for name, param in layer.named_parameters():
            assert param.grad is not None, f"{name} has no gradient"
            assert torch.isfinite(param.grad).all(), f"{name} has non-finite gradient"

    def test_cache_shape(self):
        cfg = _small_config()
        layer = BharatDecoderLayer(cfg)
        x = torch.randn(2, 4, 64)
        _, cache = layer(x, use_cache=True)
        assert cache is not None
        k, v = cache
        assert k.shape == (2, 4, 4, 16)
        assert v.shape == (2, 4, 4, 16)

    def test_deterministic_eval(self):
        cfg = _small_config()
        layer = BharatDecoderLayer(cfg)
        layer.eval()
        x = torch.randn(2, 8, 64)
        out1, _ = layer(x)
        out2, _ = layer(x)
        assert torch.allclose(out1, out2, atol=1e-5)


class TestBharatModel:
    def test_forward_shape(self):
        cfg = _small_config()
        model = BharatModel(cfg)
        input_ids = torch.randint(0, cfg.vocab_size, (2, 8))
        output = model(input_ids)
        assert isinstance(output, BharatModelOutput)
        assert output.last_hidden_state.shape == (2, 8, cfg.hidden_size)

    def test_forward_finite(self):
        cfg = _small_config()
        model = BharatModel(cfg)
        input_ids = torch.randint(0, cfg.vocab_size, (2, 8))
        output = model(input_ids)
        assert torch.isfinite(output.last_hidden_state).all()

    def test_backward(self):
        cfg = _small_config()
        model = BharatModel(cfg)
        input_ids = torch.randint(0, cfg.vocab_size, (2, 8))
        output = model(input_ids)
        loss = output.last_hidden_state.sum()
        loss.backward()
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"{name} has no gradient"

    def test_inputs_embeds(self):
        cfg = _small_config()
        model = BharatModel(cfg)
        embeds = torch.randn(2, 8, cfg.hidden_size)
        output = model(inputs_embeds=embeds)
        assert output.last_hidden_state.shape == (2, 8, cfg.hidden_size)

    def test_both_input_ids_and_embeds_raises(self):
        cfg = _small_config()
        model = BharatModel(cfg)
        with pytest.raises(ValueError, match="Only one"):
            model(
                input_ids=torch.randint(0, cfg.vocab_size, (2, 8)),
                inputs_embeds=torch.randn(2, 8, cfg.hidden_size),
            )

    def test_neither_input_ids_nor_embeds_raises(self):
        cfg = _small_config()
        model = BharatModel(cfg)
        with pytest.raises(ValueError, match="must be supplied"):
            model()

    def test_seq_too_long_raises(self):
        cfg = _small_config(max_position_embeddings=16)
        model = BharatModel(cfg)
        with pytest.raises(ValueError, match="exceeds"):
            model(input_ids=torch.randint(0, cfg.vocab_size, (2, 20)))

    def test_invalid_token_ids_raises(self):
        cfg = _small_config(vocab_size=16)
        model = BharatModel(cfg)
        with pytest.raises(ValueError, match="Token IDs"):
            model(input_ids=torch.tensor([[0, 100]]))

    def test_use_cache(self):
        cfg = _small_config()
        model = BharatModel(cfg)
        input_ids = torch.randint(0, cfg.vocab_size, (2, 8))
        output = model(input_ids, use_cache=True)
        assert output.past_key_values is not None
        assert len(output.past_key_values) == cfg.num_hidden_layers

    def test_cache_not_returned_when_use_cache_false(self):
        cfg = _small_config()
        model = BharatModel(cfg)
        input_ids = torch.randint(0, cfg.vocab_size, (2, 8))
        output = model(input_ids, use_cache=False)
        assert output.past_key_values is None

    def test_attention_mask(self):
        cfg = _small_config()
        model = BharatModel(cfg)
        input_ids = torch.randint(0, cfg.vocab_size, (2, 8))
        mask = torch.ones(2, 8, dtype=torch.long)
        mask[0, -2:] = 0
        output = model(input_ids, attention_mask=mask)
        assert torch.isfinite(output.last_hidden_state).all()

    def test_position_ids(self):
        cfg = _small_config()
        model = BharatModel(cfg)
        input_ids = torch.randint(0, cfg.vocab_size, (2, 8))
        pos_ids = torch.arange(0, 8).unsqueeze(0).expand(2, -1)
        output = model(input_ids, position_ids=pos_ids)
        assert output.last_hidden_state.shape == (2, 8, cfg.hidden_size)

    def test_state_dict_roundtrip(self):
        cfg = _small_config()
        model = BharatModel(cfg)
        state = model.state_dict()
        loaded = BharatModel(cfg)
        loaded.load_state_dict(state)
        input_ids = torch.randint(0, cfg.vocab_size, (2, 8))
        out1 = model(input_ids)
        out2 = loaded(input_ids)
        assert torch.allclose(out1.last_hidden_state, out2.last_hidden_state, atol=1e-5)

    def test_deterministic_eval(self):
        cfg = _small_config()
        model = BharatModel(cfg)
        model.eval()
        input_ids = torch.randint(0, cfg.vocab_size, (2, 8))
        out1 = model(input_ids)
        out2 = model(input_ids)
        assert torch.allclose(out1.last_hidden_state, out2.last_hidden_state, atol=1e-5)

    def test_float32(self):
        cfg = _small_config()
        model = BharatModel(cfg)
        input_ids = torch.randint(0, cfg.vocab_size, (2, 8))
        output = model(input_ids)
        assert output.last_hidden_state.dtype == torch.float32

    def test_bfloat16(self):
        cfg = _small_config()
        model = BharatModel(cfg).to(dtype=torch.bfloat16)
        input_ids = torch.randint(0, cfg.vocab_size, (2, 8))
        output = model(input_ids)
        assert output.last_hidden_state.dtype == torch.bfloat16
        assert torch.isfinite(output.last_hidden_state.float()).all()

    def test_cached_forward(self):
        cfg = _small_config()
        model = BharatModel(cfg)
        model.eval()
        input_ids = torch.randint(0, cfg.vocab_size, (2, 8))

        # Full forward
        out_full = model(input_ids, use_cache=False)

        # Token-by-token
        past = None
        out_tokens = []
        for pos in range(8):
            token_input = input_ids[:, pos : pos + 1]
            out_step = model(token_input, past_key_values=past, use_cache=True)
            out_tokens.append(out_step.last_hidden_state)
            past = out_step.past_key_values

        out_cached = torch.cat(out_tokens, dim=1)
        assert torch.allclose(out_full.last_hidden_state, out_cached, atol=1e-4)


class TestBharatForCausalLM:
    def test_logits_shape(self):
        cfg = _small_config()
        model = BharatForCausalLM(cfg)
        input_ids = torch.randint(0, cfg.vocab_size, (2, 8))
        output = model(input_ids)
        assert isinstance(output, BharatCausalLMOutput)
        assert output.logits.shape == (2, 8, cfg.vocab_size)

    def test_forward_finite(self):
        cfg = _small_config()
        model = BharatForCausalLM(cfg)
        input_ids = torch.randint(0, cfg.vocab_size, (2, 8))
        output = model(input_ids)
        assert torch.isfinite(output.logits).all()

    def test_backward(self):
        cfg = _small_config()
        model = BharatForCausalLM(cfg)
        input_ids = torch.randint(0, cfg.vocab_size, (2, 8))
        output = model(input_ids)
        loss = output.logits.sum()
        loss.backward()
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"{name} has no gradient"

    def test_loss_none_without_labels(self):
        cfg = _small_config()
        model = BharatForCausalLM(cfg)
        input_ids = torch.randint(0, cfg.vocab_size, (2, 8))
        output = model(input_ids)
        assert output.loss is None

    def test_loss_shape(self):
        cfg = _small_config()
        model = BharatForCausalLM(cfg)
        input_ids = torch.randint(0, cfg.vocab_size, (2, 8))
        labels = input_ids.clone()
        output = model(input_ids, labels=labels)
        assert output.loss is not None
        assert output.loss.ndim == 0

    def test_loss_is_finite(self):
        cfg = _small_config()
        model = BharatForCausalLM(cfg)
        input_ids = torch.randint(0, cfg.vocab_size, (2, 8))
        output = model(input_ids, labels=input_ids)
        assert torch.isfinite(output.loss).all()

    def test_causal_loss_reference(self):
        cfg = _small_config(
            vocab_size=16,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=4,
            max_position_embeddings=32,
        )
        torch.manual_seed(42)
        model = BharatForCausalLM(cfg)
        input_ids = torch.randint(0, cfg.vocab_size, (2, 6))
        labels = input_ids.clone()

        output = model(input_ids, labels=labels)

        # Manual reference loss
        logits = model.model(input_ids).last_hidden_state
        logits = model.lm_head(logits)
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        manual_loss = torch.nn.functional.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100,
        )
        assert torch.allclose(output.loss, manual_loss, atol=1e-5), (
            f"Model loss {output.loss.item()} != manual loss {manual_loss.item()}"
        )

    def test_loss_ignores_masked_labels(self):
        cfg = _small_config(
            vocab_size=16,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=4,
            max_position_embeddings=32,
        )
        model = BharatForCausalLM(cfg)
        input_ids = torch.randint(0, cfg.vocab_size, (2, 6))

        labels_all_valid = input_ids.clone()
        loss_all = model(input_ids, labels=labels_all_valid).loss

        labels_masked = input_ids.clone()
        labels_masked[0, -2:] = -100
        loss_masked = model(input_ids, labels=labels_masked).loss

        assert loss_masked.item() != loss_all.item(), "Masked loss should differ"

    def test_label_shape_mismatch_raises(self):
        cfg = _small_config()
        model = BharatForCausalLM(cfg)
        input_ids = torch.randint(0, cfg.vocab_size, (2, 8))
        with pytest.raises(ValueError, match="labels shape"):
            model(input_ids, labels=torch.randint(0, cfg.vocab_size, (2, 5)))

    def test_weight_tying_default(self):
        cfg = _small_config(tie_word_embeddings=True)
        model = BharatForCausalLM(cfg)
        assert model.lm_head.weight is model.model.embed_tokens.weight

    def test_weight_untied(self):
        cfg = _small_config(tie_word_embeddings=False)
        model = BharatForCausalLM(cfg)
        assert model.lm_head.weight is not model.model.embed_tokens.weight

    def test_use_cache(self):
        cfg = _small_config()
        model = BharatForCausalLM(cfg)
        input_ids = torch.randint(0, cfg.vocab_size, (2, 8))
        output = model(input_ids, use_cache=True)
        assert output.past_key_values is not None
        assert len(output.past_key_values) == cfg.num_hidden_layers

    def test_state_dict_roundtrip(self):
        cfg = _small_config()
        model = BharatForCausalLM(cfg)
        state = model.state_dict()
        loaded = BharatForCausalLM(cfg)
        loaded.load_state_dict(state)
        input_ids = torch.randint(0, cfg.vocab_size, (2, 8))
        out1 = model(input_ids)
        out2 = loaded(input_ids)
        assert torch.allclose(out1.logits, out2.logits, atol=1e-5)

    def test_tied_after_state_dict_load(self):
        cfg = _small_config(tie_word_embeddings=True)
        model = BharatForCausalLM(cfg)
        state = model.state_dict()
        loaded = BharatForCausalLM(cfg)
        loaded.load_state_dict(state)
        assert loaded.lm_head.weight is loaded.model.embed_tokens.weight

    # ---------- Full vs cached logit parity ----------

    def test_mha_full_vs_cached_parity(self):
        cfg = _small_config(num_attention_heads=4, num_key_value_heads=4)
        model = BharatForCausalLM(cfg)
        model.eval()
        input_ids = torch.randint(0, cfg.vocab_size, (1, 6))
        self._check_cache_parity(model, input_ids)

    def test_gqa_full_vs_cached_parity(self):
        cfg = _small_config(num_attention_heads=4, num_key_value_heads=2)
        model = BharatForCausalLM(cfg)
        model.eval()
        input_ids = torch.randint(0, cfg.vocab_size, (1, 6))
        self._check_cache_parity(model, input_ids)

    def test_mqa_full_vs_cached_parity(self):
        cfg = _small_config(num_attention_heads=4, num_key_value_heads=1)
        model = BharatForCausalLM(cfg)
        model.eval()
        input_ids = torch.randint(0, cfg.vocab_size, (1, 6))
        self._check_cache_parity(model, input_ids)

    def test_batch_full_vs_cached_parity(self):
        cfg = _small_config(num_attention_heads=4, num_key_value_heads=4)
        model = BharatForCausalLM(cfg)
        model.eval()
        input_ids = torch.randint(0, cfg.vocab_size, (2, 6))
        self._check_cache_parity(model, input_ids)

    def test_masked_full_vs_cached_parity(self):
        cfg = _small_config(num_attention_heads=4, num_key_value_heads=4)
        model = BharatForCausalLM(cfg)
        model.eval()
        input_ids = torch.randint(0, cfg.vocab_size, (1, 6))
        mask = torch.ones(1, 6, dtype=torch.long)
        mask[0, -2:] = 0
        self._check_cache_parity(model, input_ids, attention_mask=mask)

    def test_diff_prompt_lengths_parity(self):
        cfg = _small_config(num_attention_heads=4, num_key_value_heads=4)
        model = BharatForCausalLM(cfg)
        model.eval()
        input_ids = torch.randint(0, cfg.vocab_size, (2, 8))
        mask = torch.ones(2, 8, dtype=torch.long)
        mask[1, 2:4] = 0  # sample 1 has a shorter effective prompt
        mask[1, 4:] = 0
        self._check_cache_parity(model, input_ids, attention_mask=mask)

    def _check_cache_parity(
        self,
        model: BharatForCausalLM,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ):
        seq_len = input_ids.shape[1]
        out_full = model(input_ids, attention_mask=attention_mask, use_cache=False)

        past = None
        incremental_logits = []
        for pos in range(seq_len):
            token_input = input_ids[:, pos : pos + 1]
            step_mask = attention_mask[:, : pos + 1] if attention_mask is not None else None
            out_step = model(
                token_input,
                attention_mask=step_mask,
                past_key_values=past,
                use_cache=True,
            )
            incremental_logits.append(out_step.logits)
            past = out_step.past_key_values

        cat_logits = torch.cat(incremental_logits, dim=1)
        assert torch.allclose(out_full.logits, cat_logits, atol=1e-4), (
            "Full vs cached logit mismatch"
        )

    # ---------- Cache growth checks ----------

    def test_cache_grows_by_one_per_step(self):
        cfg = _small_config()
        model = BharatForCausalLM(cfg)
        model.eval()
        input_ids = torch.randint(0, cfg.vocab_size, (1, 4))

        past = None
        for pos in range(4):
            token_input = input_ids[:, pos : pos + 1]
            out_step = model(token_input, past_key_values=past, use_cache=True)
            if past is not None:
                expected_len = pos  # cached length before this step
                for k, v in past:
                    assert k.shape[-2] == expected_len, (
                        f"Expected cached length {expected_len}, got {k.shape[-2]}"
                    )
                    assert v.shape[-2] == expected_len
            past = out_step.past_key_values

    def test_cache_stores_kv_heads(self):
        cfg = _small_config(num_attention_heads=8, num_key_value_heads=2)
        model = BharatForCausalLM(cfg)
        model.eval()
        input_ids = torch.randint(0, cfg.vocab_size, (1, 4))
        out = model(input_ids, use_cache=True)
        for k, v in out.past_key_values:
            assert k.shape[1] == 2, f"Cache has {k.shape[1]} heads, expected 2"
            assert v.shape[1] == 2

    # ---------- Future-token leakage ----------

    def test_cached_attention_is_causal(self):
        cfg = _small_config(num_attention_heads=4, num_key_value_heads=4)
        model = BharatForCausalLM(cfg)
        model.eval()
        batch, seq = 1, 5
        input_ids = torch.randint(0, cfg.vocab_size, (batch, seq))

        out_full = model(input_ids).logits

        past = None
        for pos in range(seq):
            token_input = input_ids[:, pos : pos + 1]
            output = model(
                token_input,
                past_key_values=past,
                use_cache=True,
            )
            token_logit = output.logits[:, 0, :]
            past = output.past_key_values

            expected = out_full[:, pos, :]
            assert torch.allclose(token_logit, expected, atol=1e-4), f"Cached position {pos} leaked"

    # ---------- Weight initialization ----------

    def test_initialization_deterministic(self):
        cfg = _small_config()
        torch.manual_seed(42)
        model1 = BharatForCausalLM(cfg)
        torch.manual_seed(42)
        model2 = BharatForCausalLM(cfg)
        for p1, p2 in zip(model1.parameters(), model2.parameters(), strict=True):
            assert torch.equal(p1.data, p2.data), "Deterministic init failed"

    def test_embed_init_stats(self):
        cfg = _small_config(vocab_size=256, hidden_size=128)
        model = BharatForCausalLM(cfg)
        embed_mean = model.model.embed_tokens.weight.data.mean().item()
        embed_std = model.model.embed_tokens.weight.data.std().item()
        assert abs(embed_mean) < 0.1, f"Embed mean {embed_mean} too far from 0"
        assert abs(embed_std - cfg.initializer_range) < 0.02, (
            f"Embed std {embed_std} != {cfg.initializer_range}"
        )

    # ---------- Save/load ----------

    def test_save_load_equality(self, tmp_path):
        cfg = _small_config()
        model = BharatForCausalLM(cfg)
        model.save_pretrained(str(tmp_path))
        loaded = BharatForCausalLM.from_pretrained(str(tmp_path))

        input_ids = torch.randint(0, cfg.vocab_size, (2, 8))
        out1 = model(input_ids)
        out2 = loaded(input_ids)
        assert torch.allclose(out1.logits, out2.logits, atol=1e-5)

    def test_save_load_tied_weights_preserved(self, tmp_path):
        cfg = _small_config(tie_word_embeddings=True)
        model = BharatForCausalLM(cfg)
        model.save_pretrained(str(tmp_path))
        loaded = BharatForCausalLM.from_pretrained(str(tmp_path))
        assert loaded.lm_head.weight is loaded.model.embed_tokens.weight

    def test_config_roundtrip(self, tmp_path):
        cfg = _small_config()
        model = BharatForCausalLM(cfg)
        model.save_pretrained(str(tmp_path))
        loaded = BharatForCausalLM.from_pretrained(str(tmp_path))
        assert loaded.config == cfg

    def test_load_missing_dir_raises(self):
        with pytest.raises((FileNotFoundError, RuntimeError)):
            BharatForCausalLM.from_pretrained("/nonexistent/path")

    def test_load_incompatible_config_raises(self, tmp_path):
        import json
        import os

        cfg = _small_config()
        model = BharatForCausalLM(cfg)
        model.save_pretrained(str(tmp_path))
        bad_config = cfg.to_dict()
        bad_config["model_format_version"] = "gpt2-v1"
        with open(os.path.join(str(tmp_path), "config.json"), "w") as f:
            json.dump(bad_config, f)
        with pytest.raises(ValueError, match="model format"):
            BharatForCausalLM.from_pretrained(str(tmp_path))
