from __future__ import annotations

import torch
import torch.nn.functional as F

from bharat.models.bharat_model import BharatForCausalLM
from bharat.models.cache import PastKeyValues, past_length


@torch.no_grad()
def generate(
    model: BharatForCausalLM,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
    max_new_tokens: int = 20,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
    do_sample: bool = False,
    eos_token_id: int | None = None,
    pad_token_id: int | None = None,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """
    Generate tokens using the Bharat model with KV caching.

    Args:
        model: The causal LM model.
        input_ids: Prompt token IDs of shape ``(batch_size, sequence_length)``.
        attention_mask: Optional attention mask of shape
            ``(batch_size, sequence_length)`` where 1 = valid, 0 = padding.
        max_new_tokens: Maximum number of tokens to generate.
        temperature: Sampling temperature. Values > 0 required when sampling.
        top_k: If set, only the top-k highest-probability tokens are considered.
        top_p: If set, nucleus (top-p) filtering is applied.
        do_sample: If ``True``, sample from the distribution; otherwise greedy.
        eos_token_id: Token ID that stops generation.
        pad_token_id: Token ID used to pad finished sequences.
        generator: Optional ``torch.Generator`` for reproducible sampling.

    Returns:
        Generated token IDs of shape ``(batch_size, sequence_length + generated)``.
    """
    if max_new_tokens < 0:
        raise ValueError(f"max_new_tokens must be non-negative, got {max_new_tokens}")
    if do_sample and temperature <= 0.0:
        raise ValueError(f"temperature must be positive when sampling, got {temperature}")
    if top_k is not None and top_k < 1:
        raise ValueError(f"top_k must be at least 1, got {top_k}")
    if top_p is not None and not (0 < top_p <= 1.0):
        raise ValueError(f"top_p must be in (0, 1], got {top_p}")

    input_ids = input_ids.clone()
    batch_size, prompt_len = input_ids.shape
    device = input_ids.device
    _dtype = torch.long

    if attention_mask is not None:
        attention_mask = attention_mask.clone()

    if prompt_len > model.config.max_position_embeddings:
        raise ValueError(
            f"Prompt length {prompt_len} exceeds "
            f"max_position_embeddings ({model.config.max_position_embeddings})"
        )

    if eos_token_id is not None:
        eos_token_id_tensor = torch.tensor(eos_token_id, device=device)
    finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

    generated = input_ids
    past_key_values: PastKeyValues | None = None
    next_token = torch.empty((batch_size, 1), dtype=_dtype, device=device)

    for step in range(max_new_tokens):
        if step == 0:
            model_output = model(
                input_ids=generated,
                attention_mask=attention_mask,
                use_cache=True,
                past_key_values=None,
            )
            next_token_logits = model_output.logits[:, -1, :]
            past_key_values = model_output.past_key_values
        else:
            pos_kwargs: dict[str, torch.Tensor] = {}
            if attention_mask is None:
                pos_kwargs["position_ids"] = torch.full(
                    (batch_size, 1),
                    past_length(past_key_values),
                    dtype=_dtype,
                    device=device,
                )
            model_output = model(
                input_ids=next_token,
                attention_mask=attention_mask,
                use_cache=True,
                past_key_values=past_key_values,
                **pos_kwargs,
            )
            next_token_logits = model_output.logits[:, -1, :]
            past_key_values = model_output.past_key_values

        # Apply temperature
        logits = next_token_logits
        if do_sample:
            logits = logits / temperature

        # Top-k filtering
        if top_k is not None:
            top_k_values, _ = torch.topk(logits, min(top_k, logits.size(-1)), dim=-1)
            threshold = top_k_values[:, -1].unsqueeze(-1)
            logits = torch.where(logits < threshold, float("-inf"), logits)

        # Top-p filtering
        if top_p is not None:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_mask = cumulative_probs - F.softmax(sorted_logits, dim=-1) > top_p
            sorted_logits[sorted_mask] = float("-inf")
            logits = sorted_logits.scatter(1, sorted_indices, sorted_logits)

        # Sample or greedy
        if do_sample:
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1, generator=generator)
        else:
            next_token = torch.argmax(logits, dim=-1, keepdim=True)

        # Check for EOS
        if eos_token_id is not None:
            just_finished = next_token.squeeze(-1) == eos_token_id_tensor
            finished = finished | just_finished

        # Replace finished tokens with pad token
        if pad_token_id is not None:
            next_token = next_token.clone()
            next_token[finished.unsqueeze(-1)] = pad_token_id

        # Append to generated sequence
        generated = torch.cat([generated, next_token], dim=-1)

        # Extend attention mask for the next step
        if attention_mask is not None:
            new_col = torch.ones((batch_size, 1), dtype=attention_mask.dtype, device=device)
            if pad_token_id is not None:
                new_col[finished.unsqueeze(-1)] = 0
            attention_mask = torch.cat([attention_mask, new_col], dim=-1)

        # Stop early if all sequences are finished
        if finished.all():
            break

    return generated
