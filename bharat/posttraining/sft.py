from __future__ import annotations

import contextlib
import math
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from bharat.posttraining.collators import SFTCollator
from bharat.posttraining.datasets import SFTDataset
from bharat.posttraining.templates import get_template
from bharat.tokenizer import BharatTokenizer, load_tokenizer
from bharat.tokenizer.metadata import tokenizer_hash

_VALID_DEVICES = {"cpu", "cuda", "mps"}


@dataclass
class SFTConfig:
    data_path: str | Path = "data/sft/train.jsonl"
    checkpoint_path: str | Path = "checkpoints/gpt2-10m/ckpt.pt"
    output_dir: str | Path = "checkpoints/sft"
    config_path: str | Path = "configs/gpt2-10m.yaml"
    template_name: str = "indic_instruction"
    max_iters: int = 5000
    batch_size: int = 8
    learning_rate: float = 2e-5
    warmup_iters: int = 200
    block_size: int = 512
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    seed: int = 42
    log_interval: int = 100
    save_interval: int = 500
    device: str = "cpu"
    compile_model: bool = False

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
            raise ValueError("SFTConfig validation failed:\n" + "\n".join(errors))


@dataclass
class SFTResult:
    final_loss: float
    best_loss: float
    completed_steps: int
    next_step: int
    samples_processed: int
    active_tokens: int
    checkpoint_path: str


def get_lr(it: int, config: SFTConfig) -> float:
    if it < config.warmup_iters:
        return config.learning_rate * it / config.warmup_iters
    if it > config.max_iters:
        return config.learning_rate * 0.1
    ratio = (it - config.warmup_iters) / (config.max_iters - config.warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * ratio))
    min_lr = config.learning_rate * 0.1
    return min_lr + coeff * (config.learning_rate - min_lr)


def sft_train(
    model: torch.nn.Module,
    config: SFTConfig,
    tokenizer: BharatTokenizer | None = None,
) -> SFTResult:
    if tokenizer is None:
        tokenizer = load_tokenizer("gpt2")

    template = get_template(config.template_name)
    dataset = SFTDataset(config.data_path, template, config.block_size)

    if len(dataset) == 0:
        raise ValueError(
            f"SFT dataset at '{config.data_path}' is empty. "
            "Provide at least one valid training sample."
        )

    collator = SFTCollator(
        tokenizer=tokenizer,
        template=template,
        block_size=config.block_size,
        pad_token_id=tokenizer.pad_token_id,
    )
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=False,
        collate_fn=collator,
    )

    model = model.to(config.device)
    model.train()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        betas=(config.beta1, config.beta2),
        weight_decay=config.weight_decay,
    )

    if config.device == "cuda":
        autocast_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        ctx = torch.cuda.amp.autocast(dtype=autocast_dtype)
    else:
        ctx = contextlib.nullcontext()

    step = 0
    best_loss = float("inf")
    final_loss = float("inf")
    samples_processed = 0
    total_active_tokens = 0
    config.output_dir = Path(config.output_dir)
    config.output_dir.mkdir(parents=True, exist_ok=True)

    for _epoch in range(100):
        for batch in loader:
            if step >= config.max_iters:
                break

            input_ids = batch["input_ids"].to(config.device)
            labels = batch["labels"].to(config.device)

            active_in_batch = (labels != -100).sum().item()
            if active_in_batch == 0:
                raise ValueError(
                    f"Batch at step {step} has zero active assistant tokens. "
                    "Every batch must contain at least one assistant response target."
                )

            lr = get_lr(step, config)
            for pg in optimizer.param_groups:
                pg["lr"] = lr

            with ctx:
                _, loss = model(input_ids, labels)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            completed_steps = step + 1
            final_loss = loss.item()
            is_best = final_loss < best_loss
            if is_best:
                best_loss = final_loss

            samples_processed += input_ids.size(0)
            total_active_tokens += active_in_batch

            if step % config.log_interval == 0:
                nz = (labels != -100).sum().item()
                print(f"  step {step:>5}: loss {final_loss:.4f}  lr {lr:.2e}  active_tokens {nz}")

            ckpt = _build_sft_ckpt(
                model,
                optimizer,
                completed_steps,
                config,
                tokenizer,
                final_loss,
                best_loss,
                samples_processed,
                total_active_tokens,
            )

            if step % config.save_interval == 0 and step > 0:
                torch.save(ckpt, config.output_dir / "ckpt.pt")

            if is_best:
                torch.save(ckpt, config.output_dir / "best.pt")

            step += 1

        if step >= config.max_iters:
            break

    final_ckpt = _build_sft_ckpt(
        model,
        optimizer,
        step,
        config,
        tokenizer,
        final_loss,
        best_loss,
        samples_processed,
        total_active_tokens,
    )
    torch.save(final_ckpt, config.output_dir / "final.pt")

    result = SFTResult(
        final_loss=final_loss,
        best_loss=best_loss if best_loss < float("inf") else final_loss,
        completed_steps=step,
        next_step=step,
        samples_processed=samples_processed,
        active_tokens=total_active_tokens,
        checkpoint_path=str(config.output_dir),
    )
    return result


def _build_sft_ckpt(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    completed_steps: int,
    config: SFTConfig,
    tokenizer: BharatTokenizer,
    final_loss: float,
    best_loss: float,
    samples_processed: int,
    active_tokens: int,
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
        "active_tokens": active_tokens,
        "metadata": {
            "tokenizer_type": tokenizer.tokenizer_type,
            "tokenizer_hash": tokenizer_hash(tokenizer),
            "vocab_size": tokenizer.vocab_size,
        },
    }
