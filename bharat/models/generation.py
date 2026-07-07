from __future__ import annotations

import torch
import torch.nn.functional as F

from bharat.models.bharat_model import BharatForCausalLM
from bharat.models.cache import PastKeyValues, past_length


def _apply_top_k(logits: torch.Tensor, top_k: int) -> torch.Tensor:
    """Zero out all logits except the top-k highest for each row.

    The threshold is the k-th highest logit value.  Ties are resolved
    by ``torch.topk`` which returns an arbitrary subset when multiple
    candidates share the same value.

    Args:
        logits: ``(batch_size, vocab_size)`` logits.
        top_k: Number of candidates to keep.

    Returns:
        Filtered logits with excluded entries set to ``-inf``.
    """
    top_k_values, _ = torch.topk(logits, min(top_k, logits.size(-1)), dim=-1)
    threshold = top_k_values[:, -1].unsqueeze(-1)
    return torch.where(logits < threshold, float("-inf"), logits)


def _apply_top_p(logits: torch.Tensor, top_p: float) -> torch.Tensor:
    """Zero out logits whose cumulative probability exceeds ``top_p``.

    Logits are sorted descending; the first candidate that pushes the
    cumulative softmax above ``top_p`` is also kept.  At least one
    candidate always remains.  Original vocabulary order is restored.

    Args:
        logits: ``(batch_size, vocab_size)`` logits.
        top_p: Cumulative probability threshold in ``(0, 1]``.

    Returns:
        Filtered logits with excluded entries set to ``-inf``.
    """
    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    sorted_mask = cumulative_probs - F.softmax(sorted_logits, dim=-1) > top_p
    sorted_logits[sorted_mask] = float("-inf")
    return sorted_logits.scatter(1, sorted_indices, sorted_logits)


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
    # ---- Task 5: Input validation ----

    if not isinstance(max_new_tokens, int) or isinstance(max_new_tokens, bool):
        raise TypeError(f"max_new_tokens must be an integer, got {type(max_new_tokens).__name__}")
    if max_new_tokens < 0:
        raise ValueError(f"max_new_tokens must be non-negative, got {max_new_tokens}")

    if input_ids.dim() != 2:
        raise ValueError(
            f"input_ids must be 2-D (batch_size, sequence_length), got {input_ids.dim()}-D"
        )
    batch_size, prompt_len = input_ids.shape
    if batch_size == 0:
        raise ValueError("batch_size must be greater than zero")
    if prompt_len == 0:
        raise ValueError("input_ids must not be empty")

    dtype = input_ids.dtype
    if dtype not in (torch.long, torch.int, torch.int32, torch.int64):
        raise ValueError(f"input_ids must be an integer dtype, got {dtype}")

    if input_ids.min() < 0 or input_ids.max() >= model.config.vocab_size:
        raise ValueError(
            f"Token IDs must be in [0, {model.config.vocab_size - 1}], "
            f"got range [{input_ids.min().item()}, {input_ids.max().item()}]"
        )

    if do_sample:
        if not isinstance(temperature, (int, float)):
            raise TypeError(f"temperature must be a number, got {type(temperature).__name__}")
        if temperature <= 0.0:
            raise ValueError(f"temperature must be positive when sampling, got {temperature}")
        if not torch.isfinite(torch.tensor(temperature)):
            raise ValueError(f"temperature must be finite, got {temperature}")

    if top_k is not None:
        if isinstance(top_k, bool) or not isinstance(top_k, int):
            raise TypeError(f"top_k must be an integer, got {type(top_k).__name__}")
        if top_k < 1:
            raise ValueError(f"top_k must be at least 1, got {top_k}")
        if top_k > model.config.vocab_size:
            raise ValueError(
                f"top_k ({top_k}) must not exceed vocab_size ({model.config.vocab_size})"
            )

    if top_p is not None:
        if not isinstance(top_p, (int, float)):
            raise TypeError(f"top_p must be a number, got {type(top_p).__name__}")
        if not (0 < top_p <= 1.0):
            raise ValueError(f"top_p must be in (0, 1], got {top_p}")
        if not torch.isfinite(torch.tensor(top_p)):
            raise ValueError(f"top_p must be finite, got {top_p}")

    # ---- Task 4: Attention-mask validation ----

    if attention_mask is not None:
        if attention_mask.dim() != 2:
            raise ValueError(f"attention_mask must be 2-D, got {attention_mask.dim()}-D")
        if attention_mask.shape != input_ids.shape:
            raise ValueError(
                f"attention_mask shape {attention_mask.shape} must match "
                f"input_ids shape {input_ids.shape}"
            )

        mask_dtype = attention_mask.dtype
        if mask_dtype not in (
            torch.bool,
            torch.uint8,
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
            torch.float16,
            torch.float32,
            torch.float64,
        ):
            raise ValueError(f"attention_mask has unsupported dtype {mask_dtype}")

        if not ((attention_mask == 0) | (attention_mask == 1)).all():
            raise ValueError(
                "attention_mask must contain only 0/1 or True/False values, "
                f"got values in [{attention_mask.min().item()}, {attention_mask.max().item()}]"
            )

        # Each row must have at least one valid token
        valid_counts = attention_mask.sum(dim=-1)
        if (valid_counts == 0).any():
            raise ValueError(
                "Every attention_mask row must contain at least one valid token (1). "
                "Found all-zero row(s)."
            )

        # Validate right-padding contiguity (vectorized)
        cumsum = attention_mask.cumsum(dim=-1)
        for i in range(batch_size):
            n_valid = int(valid_counts[i].item())
            if n_valid > 0 and int(cumsum[i, n_valid - 1].item()) != n_valid:
                raise ValueError(
                    f"attention_mask row {i} has non-contiguous padding. "
                    f"Valid tokens must be left-aligned (right-padding only)."
                )

    # ---- EOS / PAD validation ----

    if eos_token_id is not None:
        if isinstance(eos_token_id, bool) or not isinstance(eos_token_id, int):
            raise TypeError(f"eos_token_id must be an integer, got {type(eos_token_id).__name__}")
        if not (0 <= eos_token_id < model.config.vocab_size):
            raise ValueError(
                f"eos_token_id ({eos_token_id}) must be in [0, {model.config.vocab_size - 1}]"
            )
    if pad_token_id is not None:
        if isinstance(pad_token_id, bool) or not isinstance(pad_token_id, int):
            raise TypeError(f"pad_token_id must be an integer, got {type(pad_token_id).__name__}")
        if not (0 <= pad_token_id < model.config.vocab_size):
            raise ValueError(
                f"pad_token_id ({pad_token_id}) must be in [0, {model.config.vocab_size - 1}]"
            )

    input_ids = input_ids.clone()
    device = input_ids.device

    if attention_mask is not None:
        attention_mask = attention_mask.clone()

    if prompt_len > model.config.max_position_embeddings:
        raise ValueError(
            f"Prompt length {prompt_len} exceeds "
            f"max_position_embeddings ({model.config.max_position_embeddings})"
        )
    if prompt_len + max_new_tokens > model.config.max_position_embeddings:
        raise ValueError(
            f"Prompt length ({prompt_len}) + max_new_tokens ({max_new_tokens}) exceeds "
            f"max_position_embeddings ({model.config.max_position_embeddings}). "
            f"The generated sequence would exceed the configured context window."
        )

    if eos_token_id is not None:
        eos_token_id_tensor = torch.tensor(eos_token_id, device=device)
    finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

    # Task 1: deterministic fill token for finished rows
    if pad_token_id is not None:
        fill_token = pad_token_id
    elif eos_token_id is not None:
        fill_token = eos_token_id
    else:
        fill_token = 0

    generated = input_ids
    past_key_values: PastKeyValues | None = None
    next_token = torch.empty(0, 1, dtype=torch.long, device=device)

    for step in range(max_new_tokens):
        was_finished = finished.clone()

        # ---- Model forward ----
        if step == 0:
            model_output = model(
                input_ids=generated,
                attention_mask=attention_mask,
                use_cache=True,
                past_key_values=None,
            )
            if attention_mask is not None:
                valid_lengths = attention_mask.sum(dim=-1)
                last_valid_indices = valid_lengths - 1
                next_token_logits = model_output.logits[
                    torch.arange(batch_size, device=device),
                    last_valid_indices,
                ]
            else:
                next_token_logits = model_output.logits[:, -1, :]
            past_key_values = model_output.past_key_values
        else:
            pos_kwargs: dict[str, torch.Tensor] = {}
            if attention_mask is None:
                pos_kwargs["position_ids"] = torch.full(
                    (batch_size, 1),
                    past_length(past_key_values),
                    dtype=torch.long,
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

        # ---- Task 1: Only process active (not previously finished) rows ----
        logits = next_token_logits
        active = ~was_finished

        if active.any():
            active_logits = logits.index_select(0, active.nonzero(as_tuple=True)[0])

            # Apply temperature
            if do_sample:
                active_logits = active_logits / temperature

            # Top-k filtering
            if top_k is not None:
                active_logits = _apply_top_k(active_logits, top_k)

            # Top-p filtering
            if top_p is not None:
                active_logits = _apply_top_p(active_logits, top_p)

            # Sample or greedy
            if do_sample:
                active_probs = F.softmax(active_logits, dim=-1)
                active_tokens = torch.multinomial(active_probs, num_samples=1, generator=generator)
            else:
                active_tokens = torch.argmax(active_logits, dim=-1, keepdim=True)

        # ---- Build next_token from active results and fill tokens ----
        next_token = torch.full((batch_size, 1), fill_token, dtype=torch.long, device=device)
        if active.any():
            active_indices = active.nonzero(as_tuple=True)[0]
            next_token[active_indices] = active_tokens

        # ---- Update finished state ----
        if eos_token_id is not None:
            just_finished = next_token.squeeze(-1) == eos_token_id_tensor
            finished = was_finished | just_finished
        else:
            just_finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

        # ---- Append to generated sequence ----
        generated = torch.cat([generated, next_token], dim=-1)

        # ---- Extend attention mask ----
        if attention_mask is not None:
            new_col = torch.ones((batch_size, 1), dtype=attention_mask.dtype, device=device)
            new_col[was_finished.unsqueeze(-1)] = 0
            attention_mask = torch.cat([attention_mask, new_col], dim=-1)

        if finished.all():
            break

    return generated
