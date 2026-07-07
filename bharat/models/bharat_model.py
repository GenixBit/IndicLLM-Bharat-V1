from __future__ import annotations

import json
import os
from typing import Any

import torch
import torch.nn as nn

from bharat.models.attention import GroupedQueryAttention, KeyValueCache
from bharat.models.cache import PastKeyValues, past_length, validate_cache
from bharat.models.config import BharatModelConfig
from bharat.models.mlp import SwiGLU
from bharat.models.normalization import RMSNorm
from bharat.models.outputs import BharatCausalLMOutput, BharatModelOutput

_MODEL_FORMAT_VERSION = "bharat-v1"


class BharatDecoderLayer(nn.Module):
    """
    A single Bharat decoder layer with pre-normalisation.

    Architecture (pre-normalisation):
        residual → RMSNorm → GQA attention → dropout → + residual
        → RMSNorm → SwiGLU MLP → dropout → + residual

    Args:
        config: Model configuration.
    """

    def __init__(self, config: BharatModelConfig) -> None:
        super().__init__()
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.self_attn = GroupedQueryAttention(config)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = SwiGLU(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            bias=config.mlp_bias,
            dropout=config.hidden_dropout,
        )
        self.hidden_dropout = config.hidden_dropout

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        past_key_value: KeyValueCache | None = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, KeyValueCache | None]:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        attn_output, cache = self.self_attn(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            use_cache=use_cache,
        )
        if self.hidden_dropout > 0 and self.training:
            attn_output = nn.functional.dropout(attn_output, p=self.hidden_dropout)
        hidden_states = residual + attn_output

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        mlp_output = self.mlp(hidden_states)
        hidden_states = residual + mlp_output

        return hidden_states, cache


class BharatModel(nn.Module):
    """
    Base Bharat decoder-only model without a language modelling head.

    Architecture:
        token embeddings → decoder layers → final RMSNorm

    No learned absolute position embeddings are used; position information
    is injected via RoPE inside ``GroupedQueryAttention``.

    When padding is present, padded hidden-state rows receive masked attention
    (their values are determined only by padding tokens) and should be discarded
    before loss computation or generation.

    Args:
        config: Model configuration.
    """

    def __init__(self, config: BharatModelConfig) -> None:
        super().__init__()
        self.config = config

        self.embed_tokens = nn.Embedding(
            config.vocab_size,
            config.hidden_size,
        )
        self.layers = nn.ModuleList(
            [BharatDecoderLayer(config) for _ in range(config.num_hidden_layers)]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # Initialise weights
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        past_key_values: PastKeyValues | None = None,
        use_cache: bool = False,
    ) -> BharatModelOutput:
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("Only one of input_ids or inputs_embeds may be supplied")
        if input_ids is None and inputs_embeds is None:
            raise ValueError("One of input_ids or inputs_embeds must be supplied")

        if input_ids is not None:
            batch_size, seq_len = input_ids.shape
            if seq_len > self.config.max_position_embeddings:
                raise ValueError(
                    f"Sequence length {seq_len} exceeds "
                    f"max_position_embeddings ({self.config.max_position_embeddings})"
                )
            if input_ids.min() < 0 or input_ids.max() >= self.config.vocab_size:
                raise ValueError(
                    f"Token IDs must be in [0, {self.config.vocab_size - 1}], "
                    f"got range [{input_ids.min().item()}, {input_ids.max().item()}]"
                )
            hidden_states = self.embed_tokens(input_ids)
        else:
            assert inputs_embeds is not None
            batch_size, seq_len = inputs_embeds.shape[:2]
            if seq_len > self.config.max_position_embeddings:
                raise ValueError(
                    f"Sequence length {seq_len} exceeds "
                    f"max_position_embeddings ({self.config.max_position_embeddings})"
                )
            hidden_states = inputs_embeds

        device = hidden_states.device
        dtype = hidden_states.dtype

        # Auto-create position IDs if not supplied
        if position_ids is None:
            past_len = past_length(past_key_values)
            if attention_mask is not None:
                # Padding-aware cumulative positions (works for both first pass and cached)
                position_ids = (attention_mask.cumsum(dim=-1).long() - 1).clamp(min=0)[:, -seq_len:]
            else:
                # Simple offset (no mask) — first pass or cached
                position_ids = (
                    torch.arange(
                        past_len,
                        past_len + seq_len,
                        dtype=torch.long,
                        device=device,
                    )
                    .unsqueeze(0)
                    .expand(batch_size, -1)
                )

        # Validate cache if provided
        if past_key_values is not None:
            validate_cache(
                past_key_values,
                expected_layers=self.config.num_hidden_layers,
                batch_size=batch_size,
                kv_heads=self.config.num_key_value_heads,
                head_dim=self.config.head_dim,
                device=device,
                dtype=dtype,
            )

        new_past_key_values: list[KeyValueCache] = []
        for i, layer in enumerate(self.layers):
            layer_past = past_key_values[i] if past_key_values is not None else None
            hidden_states, cache = layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=layer_past,
                use_cache=use_cache,
            )
            if use_cache and cache is not None:
                new_past_key_values.append(cache)

        hidden_states = self.norm(hidden_states)

        pkv: PastKeyValues | None = tuple(new_past_key_values) if use_cache else None
        return BharatModelOutput(
            last_hidden_state=hidden_states,
            past_key_values=pkv,
        )

    def save_pretrained(self, path: str) -> None:
        """
        Save model configuration and weights to a local directory.

        Stores ``config.json`` and ``model.pt`` using an atomic
        temporary-file-and-rename pattern.
        """
        os.makedirs(path, exist_ok=True)

        config_path = os.path.join(path, "config.json")
        tmp_config = config_path + ".tmp"
        config_dict: dict[str, Any] = self.config.to_dict()
        config_dict["model_format_version"] = _MODEL_FORMAT_VERSION
        with open(tmp_config, "w") as f:
            json.dump(config_dict, f)
        os.replace(tmp_config, config_path)

        model_path = os.path.join(path, "model.pt")
        tmp_model = model_path + ".tmp"
        torch.save(self.state_dict(), tmp_model)
        os.replace(tmp_model, model_path)

    @classmethod
    def from_pretrained(
        cls,
        path: str,
        map_location: str | torch.device | None = None,
    ) -> BharatModel:
        """
        Load a Bharat model from a local directory.

        Expects ``config.json`` and ``model.pt`` in the directory.
        """
        config_path = os.path.join(path, "config.json")
        if not os.path.isfile(config_path):
            raise FileNotFoundError(f"Config not found: {config_path}")

        with open(config_path) as f:
            config_dict = json.load(f)

        model_format = config_dict.pop("model_format_version", None)
        if model_format != _MODEL_FORMAT_VERSION:
            raise ValueError(
                f"Unsupported model format version '{model_format}'. "
                f"Expected '{_MODEL_FORMAT_VERSION}'."
            )

        config = BharatModelConfig.from_dict(config_dict)

        model_path = os.path.join(path, "model.pt")
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"Model weights not found: {model_path}")

        model = cls(config)
        state = torch.load(model_path, map_location=map_location, weights_only=True)
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing:
            raise RuntimeError(f"Missing keys: {missing}")
        if unexpected:
            raise RuntimeError(f"Unexpected keys: {unexpected}")
        return model


class BharatForCausalLM(nn.Module):
    """
    Bharat decoder-only language model with a causal language modelling head.

    Components:
        model: The base ``BharatModel``.
        lm_head: Linear projection from ``hidden_size`` to ``vocab_size``
            (no bias).

    Weight tying:
        When ``config.tie_word_embeddings`` is ``True``, ``lm_head.weight``
        is the same parameter object as ``model.embed_tokens.weight``.

    Loss:
        Standard next-token causal cross-entropy with ``ignore_index=-100``
        for masked positions.
    """

    def __init__(self, config: BharatModelConfig) -> None:
        super().__init__()
        self.config = config
        self.model = BharatModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            if module.weight is not self.model.embed_tokens.weight:
                module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        past_key_values: PastKeyValues | None = None,
        use_cache: bool = False,
        labels: torch.Tensor | None = None,
    ) -> BharatCausalLMOutput:
        model_output = self.model(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
        )

        logits = self.lm_head(model_output.last_hidden_state)

        loss: torch.Tensor | None = None
        if labels is not None:
            if input_ids is not None and labels.shape != input_ids.shape:
                raise ValueError(
                    f"labels shape {labels.shape} must match input_ids shape {input_ids.shape}"
                )
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = nn.functional.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )

        return BharatCausalLMOutput(
            logits=logits,
            loss=loss,
            past_key_values=model_output.past_key_values,
        )

    def save_pretrained(self, path: str) -> None:
        """
        Save model configuration and weights to a local directory.

        Stores ``config.json`` and ``model.pt``.
        """
        os.makedirs(path, exist_ok=True)

        config_path = os.path.join(path, "config.json")
        tmp_config = config_path + ".tmp"
        config_dict: dict[str, Any] = self.config.to_dict()
        config_dict["model_format_version"] = _MODEL_FORMAT_VERSION
        with open(tmp_config, "w") as f:
            json.dump(config_dict, f)
        os.replace(tmp_config, config_path)

        model_path = os.path.join(path, "model.pt")
        tmp_model = model_path + ".tmp"
        torch.save(self.state_dict(), tmp_model)
        os.replace(tmp_model, model_path)

    @classmethod
    def from_pretrained(
        cls,
        path: str,
        map_location: str | torch.device | None = None,
    ) -> BharatForCausalLM:
        """
        Load a BharatForCausalLM from a local directory.

        Expects ``config.json`` and ``model.pt`` in the directory.
        """
        config_path = os.path.join(path, "config.json")
        if not os.path.isfile(config_path):
            raise FileNotFoundError(f"Config not found: {config_path}")

        with open(config_path) as f:
            config_dict = json.load(f)

        model_format = config_dict.pop("model_format_version", None)
        if model_format != _MODEL_FORMAT_VERSION:
            raise ValueError(
                f"Unsupported model format version '{model_format}'. "
                f"Expected '{_MODEL_FORMAT_VERSION}'."
            )

        config = BharatModelConfig.from_dict(config_dict)

        model_path = os.path.join(path, "model.pt")
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"Model weights not found: {model_path}")

        model = cls(config)
        state = torch.load(model_path, map_location=map_location, weights_only=True)
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing:
            raise RuntimeError(f"Missing keys: {missing}")
        if unexpected:
            raise RuntimeError(f"Unexpected keys: {unexpected}")
        return model
