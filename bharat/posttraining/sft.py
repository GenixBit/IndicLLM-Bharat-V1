from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from bharat.posttraining.collators import SFTCollator
from bharat.posttraining.datasets import SFTDataset
from bharat.posttraining.templates import get_template
from bharat.tokenizer import BharatTokenizer, load_tokenizer


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
) -> float:
    if tokenizer is None:
        tokenizer = load_tokenizer("gpt2")

    template = get_template(config.template_name)
    dataset = SFTDataset(config.data_path, template, config.block_size)
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
        drop_last=True,
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

    ctx = torch.amp.autocast(device_type=config.device, dtype=torch.bfloat16) \
        if config.device == "cuda" else torch.no_grad()

    step = 0
    best_loss = float("inf")
    config.output_dir = Path(config.output_dir)
    config.output_dir.mkdir(parents=True, exist_ok=True)

    for _epoch in range(100):
        for batch in loader:
            if step >= config.max_iters:
                break

            lr = get_lr(step, config)
            for pg in optimizer.param_groups:
                pg["lr"] = lr

            input_ids = batch["input_ids"].to(config.device)
            labels = batch["labels"].to(config.device)

            with ctx:
                _, loss = model(input_ids, labels)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            if step % config.log_interval == 0:
                nz = (labels != -100).sum().item()
                print(f"  step {step:>5}: loss {loss.item():.4f}  lr {lr:.2e}  active_tokens {nz}")

            if step % config.save_interval == 0 and step > 0:
                ckpt = {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "step": step,
                    "config": config,
                }
                torch.save(ckpt, config.output_dir / "ckpt.pt")
                if loss.item() < best_loss:
                    best_loss = loss.item()
                    torch.save(ckpt, config.output_dir / "best.pt")

            step += 1

        if step >= config.max_iters:
            break

    torch.save({"model": model.state_dict()}, config.output_dir / "final.pt")
    return best_loss
