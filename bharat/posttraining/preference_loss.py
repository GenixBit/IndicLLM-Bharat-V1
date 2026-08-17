from __future__ import annotations

import contextlib
from typing import Any

import torch
import torch.nn.functional as F


def per_sample_log_probs(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    response_masks: torch.Tensor,
    ctx: contextlib.AbstractContextManager[Any],
) -> torch.Tensor:
    """Compute per-sample log-probabilities for response tokens only.

    response_masks should be a boolean tensor of shape (B, T) where True
    indicates a response token position (aligned to target_ids, i.e.
    input_ids[:, 1:]).

    The mask must account for the causal LM shift: model output at
    position i predicts input_ids[:, i+1], so the mask at position i
    should be True if input_ids[:, i+1] is a response token.
    """
    with ctx:
        out = model(input_ids)
        if hasattr(out, "logits"):
            logits = out.logits
        elif isinstance(out, tuple | list):
            logits = out[0]
        else:
            logits = out

    log_p = F.log_softmax(logits, dim=-1)

    _batch_size, _seq_len, _vocab = log_p.shape
    # logits[:, i] predicts input_ids[:, i+1]
    # Gather log-probs for the predicted targets
    targets = input_ids[:, 1:].unsqueeze(-1)
    per_token_lp = log_p[:, :-1].gather(-1, targets).squeeze(-1)

    # response_masks is (B, T-1) aligned to targets
    if response_masks.dtype != torch.bool:
        response_masks = response_masks.bool()

    if response_masks.shape != per_token_lp.shape:
        # Handle if mask was provided for full sequence length
        response_masks = response_masks[:, : per_token_lp.shape[1]]

    return (per_token_lp * response_masks.to(per_token_lp.dtype)).sum(-1)


def dpo_loss(
    policy_chosen_lp: torch.Tensor,
    policy_rejected_lp: torch.Tensor,
    ref_chosen_lp: torch.Tensor,
    ref_rejected_lp: torch.Tensor,
    beta: float = 0.1,
) -> torch.Tensor:
    """Standard DPO loss with per-sample log-probabilities."""
    chosen_ratio = policy_chosen_lp - ref_chosen_lp
    rejected_ratio = policy_rejected_lp - ref_rejected_lp
    return -F.logsigmoid(beta * (chosen_ratio - rejected_ratio)).mean()


def reward_accuracy(
    policy_chosen_lp: torch.Tensor,
    policy_rejected_lp: torch.Tensor,
    ref_chosen_lp: torch.Tensor,
    ref_rejected_lp: torch.Tensor,
) -> torch.Tensor:
    """Accuracy based on implicit reward: beta * (policy_logprob - ref_logprob)."""
    chosen_implicit_reward = policy_chosen_lp - ref_chosen_lp
    rejected_implicit_reward = policy_rejected_lp - ref_rejected_lp
    return (chosen_implicit_reward > rejected_implicit_reward).float().mean()


def approximate_kl_divergence(
    policy_chosen_lp: torch.Tensor,
    ref_chosen_lp: torch.Tensor,
    policy_rejected_lp: torch.Tensor,
    ref_rejected_lp: torch.Tensor,
) -> torch.Tensor:
    """Compute an approximate KL divergence between policy and reference.

    Uses the DPO implicit reward formulation:
    KL_approx = mean(policy_logprob - ref_logprob)
    This is an unbiased estimate of KL(policy || ref) under the
    DPO preference model.
    """
    chosen_kl = (policy_chosen_lp - ref_chosen_lp).mean()
    rejected_kl = (policy_rejected_lp - ref_rejected_lp).mean()
    return (chosen_kl + rejected_kl) * 0.5
