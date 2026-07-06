from __future__ import annotations

import contextlib
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from bharat.posttraining.preference_dataset import PreferenceDataset, dpo_collate
from bharat.posttraining.preference_loss import (
    approximate_kl_divergence,
    dpo_loss,
    per_sample_log_probs,
    reward_accuracy,
)
from bharat.posttraining.templates import get_template
from bharat.tokenizer import BharatTokenizer, load_tokenizer


@dataclass
class DPOConfig:
    data_path: str | Path = "data/dpo/preferences.jsonl"
    sft_checkpoint: str | Path = "checkpoints/gpt2-10m-sft/final.pt"
    output_dir: str | Path = "checkpoints/dpo"
    template_name: str = "indic_instruction"
    max_iters: int = 2000
    batch_size: int = 4
    learning_rate: float = 5e-6
    beta: float = 0.1
    block_size: int = 512
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    seed: int = 42
    log_interval: int = 50
    save_interval: int = 500
    device: str = "cpu"


def dpo_train(
    policy_model: torch.nn.Module,
    ref_model: torch.nn.Module,
    config: DPOConfig,
    tokenizer: BharatTokenizer | None = None,
) -> float:
    if tokenizer is None:
        tokenizer = load_tokenizer("gpt2")

    template = get_template(config.template_name)
    dataset = PreferenceDataset(
        config.data_path, template, config.block_size, tokenizer
    )
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=True,
        collate_fn=lambda b: dpo_collate(b, pad_token_id=tokenizer.pad_token_id),
    )

    policy_model = policy_model.to(config.device)
    ref_model = ref_model.to(config.device)
    for p in ref_model.parameters():
        p.requires_grad = False
    ref_model.eval()

    policy_model.train()
    optimizer = torch.optim.AdamW(
        policy_model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    if config.device == "cuda":
        autocast_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        train_ctx = torch.amp.autocast(device_type="cuda", dtype=autocast_dtype)
    else:
        train_ctx = contextlib.nullcontext()

    ref_ctx = torch.no_grad()

    step = 0
    config.output_dir = Path(config.output_dir)
    config.output_dir.mkdir(parents=True, exist_ok=True)

    for _epoch in range(100):
        for batch in loader:
            if step >= config.max_iters:
                break

            chosen = batch["chosen_ids"].to(config.device)
            rejected = batch["rejected_ids"].to(config.device)
            chosen_mask = batch["chosen_response_mask"].to(config.device)
            rejected_mask = batch["rejected_response_mask"].to(config.device)

            # Policy forward with gradients
            policy_chosen_lp = per_sample_log_probs(policy_model, chosen, chosen_mask, train_ctx)
            policy_rejected_lp = per_sample_log_probs(policy_model, rejected, rejected_mask, train_ctx)

            # Reference forward without gradients
            ref_chosen_lp = per_sample_log_probs(ref_model, chosen, chosen_mask, ref_ctx)
            ref_rejected_lp = per_sample_log_probs(ref_model, rejected, rejected_mask, ref_ctx)

            loss = dpo_loss(
                policy_chosen_lp, policy_rejected_lp,
                ref_chosen_lp, ref_rejected_lp,
                config.beta,
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy_model.parameters(), config.grad_clip)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            if step % config.log_interval == 0:
                r_acc = reward_accuracy(
                    policy_chosen_lp, policy_rejected_lp,
                    ref_chosen_lp, ref_rejected_lp,
                )
                kl = approximate_kl_divergence(
                    policy_chosen_lp, ref_chosen_lp,
                    policy_rejected_lp, ref_rejected_lp,
                )
                print(f"  step {step:>4}: loss {loss.item():.4f}  reward_acc {r_acc:.2%}  kl {kl:.4f}")

            if step % config.save_interval == 0 and step > 0:
                torch.save(
                    {"model": policy_model.state_dict(), "step": step},
                    config.output_dir / "ckpt.pt",
                )

            step += 1

        if step >= config.max_iters:
            break

    torch.save({"model": policy_model.state_dict()}, config.output_dir / "final.pt")
    return loss.item()
