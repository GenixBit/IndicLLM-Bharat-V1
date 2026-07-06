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
from bharat.tokenizer.metadata import tokenizer_hash

_VALID_DEVICES = {"cpu", "cuda", "mps"}


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

    def __post_init__(self) -> None:
        errors: list[str] = []
        if self.max_iters <= 0:
            errors.append(f"max_iters must be > 0, got {self.max_iters}")
        if self.batch_size <= 0:
            errors.append(f"batch_size must be > 0, got {self.batch_size}")
        if self.block_size <= 1:
            errors.append(f"block_size must be > 1, got {self.block_size}")
        if self.learning_rate <= 0:
            errors.append(f"learning_rate must be > 0, got {self.learning_rate}")
        if self.save_interval <= 0:
            errors.append(f"save_interval must be > 0, got {self.save_interval}")
        if self.log_interval <= 0:
            errors.append(f"log_interval must be > 0, got {self.log_interval}")
        dev = self.device.split(":")[0]
        if dev not in _VALID_DEVICES:
            errors.append(f"device must be one of {_VALID_DEVICES}, got {self.device}")
        if errors:
            raise ValueError("DPOConfig validation failed:\n" + "\n".join(errors))


@dataclass
class DPOResult:
    final_loss: float
    best_loss: float
    completed_steps: int
    next_step: int
    samples_processed: int
    chosen_tokens: int
    rejected_tokens: int
    final_reward_accuracy: float
    final_kl: float
    checkpoint_path: str


def dpo_train(
    policy_model: torch.nn.Module,
    ref_model: torch.nn.Module,
    config: DPOConfig,
    tokenizer: BharatTokenizer | None = None,
) -> DPOResult:
    if tokenizer is None:
        tokenizer = load_tokenizer("gpt2")

    template = get_template(config.template_name)
    dataset = PreferenceDataset(config.data_path, template, config.block_size, tokenizer)

    if len(dataset) == 0:
        raise ValueError(
            f"DPO dataset at '{config.data_path}' is empty. "
            "Provide at least one valid preference sample."
        )

    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=False,
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
        train_ctx = torch.cuda.amp.autocast(dtype=autocast_dtype)
    else:
        train_ctx = contextlib.nullcontext()

    ref_ctx = torch.no_grad()

    step = 0
    best_loss = float("inf")
    final_loss = float("inf")
    samples_processed = 0
    total_chosen_tokens = 0
    total_rejected_tokens = 0
    final_reward_acc = 0.0
    final_kl_val = 0.0
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

            chosen_active = chosen_mask.sum().item()
            rejected_active = rejected_mask.sum().item()

            if chosen_active == 0 or rejected_active == 0:
                raise ValueError(
                    f"Batch at step {step} has zero chosen ({chosen_active}) or "
                    f"rejected ({rejected_active}) response tokens. "
                    "Every preference batch must have active response tokens in both."
                )

            # Policy forward with gradients
            policy_chosen_lp = per_sample_log_probs(policy_model, chosen, chosen_mask, train_ctx)
            policy_rejected_lp = per_sample_log_probs(
                policy_model, rejected, rejected_mask, train_ctx
            )

            # Reference forward without gradients
            ref_chosen_lp = per_sample_log_probs(ref_model, chosen, chosen_mask, ref_ctx)
            ref_rejected_lp = per_sample_log_probs(ref_model, rejected, rejected_mask, ref_ctx)

            loss = dpo_loss(
                policy_chosen_lp,
                policy_rejected_lp,
                ref_chosen_lp,
                ref_rejected_lp,
                config.beta,
            )

            loss.backward()  # type: ignore[no-untyped-call]
            torch.nn.utils.clip_grad_norm_(policy_model.parameters(), config.grad_clip)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            completed_steps = step + 1
            final_loss = loss.item()
            is_best = final_loss < best_loss
            if is_best:
                best_loss = final_loss

            samples_processed += chosen.size(0)
            total_chosen_tokens += chosen_active
            total_rejected_tokens += rejected_active
            final_reward_acc = reward_accuracy(
                policy_chosen_lp,
                policy_rejected_lp,
                ref_chosen_lp,
                ref_rejected_lp,
            ).item()
            final_kl_val = approximate_kl_divergence(
                policy_chosen_lp,
                ref_chosen_lp,
                policy_rejected_lp,
                ref_rejected_lp,
            ).item()

            if step % config.log_interval == 0:
                print(
                    f"  step {step:>4}: loss {final_loss:.4f}  "
                    f"reward_acc {final_reward_acc:.2%}  kl {final_kl_val:.4f}"
                )

            ckpt = _build_dpo_ckpt(
                policy_model,
                optimizer,
                completed_steps,
                config,
                tokenizer,
                final_loss,
                best_loss,
                samples_processed,
                total_chosen_tokens,
                total_rejected_tokens,
                final_reward_acc,
                final_kl_val,
            )

            if step % config.save_interval == 0 and step > 0:
                torch.save(ckpt, config.output_dir / "ckpt.pt")

            if is_best:
                torch.save(ckpt, config.output_dir / "best.pt")

            step += 1

        if step >= config.max_iters:
            break

    final_ckpt = _build_dpo_ckpt(
        policy_model,
        optimizer,
        step,
        config,
        tokenizer,
        final_loss,
        best_loss,
        samples_processed,
        total_chosen_tokens,
        total_rejected_tokens,
        final_reward_acc,
        final_kl_val,
    )
    torch.save(final_ckpt, config.output_dir / "final.pt")

    result = DPOResult(
        final_loss=final_loss,
        best_loss=best_loss if best_loss < float("inf") else final_loss,
        completed_steps=step,
        next_step=step,
        samples_processed=samples_processed,
        chosen_tokens=total_chosen_tokens,
        rejected_tokens=total_rejected_tokens,
        final_reward_accuracy=final_reward_acc,
        final_kl=final_kl_val,
        checkpoint_path=str(config.output_dir),
    )
    return result


def _build_dpo_ckpt(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    completed_steps: int,
    config: DPOConfig,
    tokenizer: BharatTokenizer,
    final_loss: float,
    best_loss: float,
    samples_processed: int,
    chosen_tokens: int,
    rejected_tokens: int,
    reward_accuracy_val: float,
    kl_val: float,
) -> dict[str, object]:
    return {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "completed_steps": completed_steps,
        "next_step": completed_steps,
        "config": config,
        "final_loss": final_loss,
        "best_loss": best_loss,
        "samples_processed": samples_processed,
        "chosen_tokens": chosen_tokens,
        "rejected_tokens": rejected_tokens,
        "reward_accuracy": reward_accuracy_val,
        "kl": kl_val,
        "metadata": {
            "tokenizer_type": tokenizer.tokenizer_type,
            "tokenizer_hash": tokenizer_hash(tokenizer),
            "vocab_size": tokenizer.vocab_size,
        },
    }
