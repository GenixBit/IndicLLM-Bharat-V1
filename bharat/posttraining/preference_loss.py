from __future__ import annotations

import torch
import torch.nn.functional as F


def per_sample_log_probs(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    prompt_lengths: torch.Tensor,
    ctx: object,
) -> torch.Tensor:
    """Compute per-sample log-probs for response tokens only.

    Each sample in the batch has its own prompt length, so masking
    is done per-sample rather than with a single batch-level value.
    """
    with ctx:
        logits, _ = model(input_ids)

    log_p = F.log_softmax(logits, dim=-1)
    tokens = input_ids[:, 1:].unsqueeze(-1)
    per_token_lp = log_p[:, :-1].gather(-1, tokens).squeeze(-1)

    batch_size, seq_len = per_token_lp.shape
    arange = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)
    prompt_lens = prompt_lengths.unsqueeze(1).expand(batch_size, seq_len)
    mask = (arange >= prompt_lens).float()

    return (per_token_lp * mask).sum(-1)


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
) -> torch.Tensor:
    return (policy_chosen_lp > policy_rejected_lp).float().mean()


def kl_divergence(
    policy_chosen_lp: torch.Tensor,
    ref_chosen_lp: torch.Tensor,
    policy_rejected_lp: torch.Tensor,
    ref_rejected_lp: torch.Tensor,
) -> torch.Tensor:
    chosen_kl = (policy_chosen_lp - ref_chosen_lp).mean()
    rejected_kl = (policy_rejected_lp - ref_rejected_lp).mean()
    return (chosen_kl + rejected_kl).abs()
