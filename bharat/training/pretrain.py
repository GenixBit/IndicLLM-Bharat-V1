from __future__ import annotations

import contextlib
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import yaml

from bharat.models.bharat_model import BharatForCausalLM
from bharat.models.config import BharatModelConfig
from bharat.tokenizer.base import BharatTokenizer
from bharat.training.checkpointing import load_checkpoint, save_checkpoint

_VALID_DEVICES = {"cpu", "cuda", "mps"}
_VALID_DTYPES = {"float32", "bfloat16", "float16"}


@dataclass
class PretrainConfig:
    """Configuration for Bharat model pretraining."""

    model_config_path: str | Path | None = None
    model_config: BharatModelConfig | None = None
    data_path: str | Path | None = None
    val_data_path: str | Path | None = None
    synthetic_data: bool = False
    output_dir: str | Path = "checkpoints/pretrain"
    max_iters: int = 1000
    batch_size: int = 4
    seq_len: int = 512
    gradient_accumulation_steps: int = 1
    learning_rate: float = 3e-4
    min_lr: float | None = None
    warmup_iters: int = 100
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    seed: int = 42
    log_interval: int = 10
    eval_interval: int = 100
    eval_iters: int = 20
    save_interval: int = 500
    device: str = "cpu"
    dtype: str = "float32"
    resume_checkpoint: str | Path | None = None

    def __post_init__(self) -> None:
        errors: list[str] = []
        if self.max_iters <= 0:
            errors.append(f"max_iters must be > 0, got {self.max_iters}")
        if self.batch_size <= 0:
            errors.append(f"batch_size must be > 0, got {self.batch_size}")
        if self.seq_len <= 1:
            errors.append(f"seq_len must be > 1, got {self.seq_len}")
        if self.gradient_accumulation_steps <= 0:
            errors.append(
                f"gradient_accumulation_steps must be > 0, got {self.gradient_accumulation_steps}"
            )
        if self.learning_rate <= 0:
            errors.append(f"learning_rate must be > 0, got {self.learning_rate}")
        if self.warmup_iters < 0:
            errors.append(f"warmup_iters must be >= 0, got {self.warmup_iters}")
        if self.log_interval <= 0:
            errors.append(f"log_interval must be > 0, got {self.log_interval}")
        if self.eval_interval <= 0:
            errors.append(f"eval_interval must be > 0, got {self.eval_interval}")
        if self.save_interval <= 0:
            errors.append(f"save_interval must be > 0, got {self.save_interval}")

        dev = self.device.split(":")[0]
        if dev not in _VALID_DEVICES:
            errors.append(f"device must be one of {_VALID_DEVICES}, got {self.device}")
        if self.dtype not in _VALID_DTYPES:
            errors.append(f"dtype must be one of {_VALID_DTYPES}, got {self.dtype}")

        if self.model_config is None and self.model_config_path is None:
            errors.append("Either model_config or model_config_path must be provided")

        if self.min_lr is None:
            self.min_lr = self.learning_rate * 0.1

        if errors:
            raise ValueError("PretrainConfig validation failed:\n" + "\n".join(errors))


@dataclass
class PretrainResult:
    final_loss: float
    best_loss: float
    val_loss: float | None
    completed_steps: int
    total_tokens_processed: int
    checkpoint_path: str | None
    step_losses: list[float] = field(default_factory=list)


def load_model_config_from_yaml(config_path: str | Path) -> BharatModelConfig:
    """Load a BharatModelConfig from a YAML model configuration file."""
    path = Path(config_path)
    if not path.is_file():
        raise FileNotFoundError(f"Model config file not found: {path}")

    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if not isinstance(data, dict):
        raise ValueError(f"Invalid YAML config format in {path}: expected dict")

    arch = data.get("architecture", data)
    return BharatModelConfig.from_dict(arch)


def configure_optimizers(
    model: nn.Module,
    weight_decay: float,
    learning_rate: float,
    betas: tuple[float, float],
    device_type: str = "cpu",
) -> torch.optim.AdamW:
    """
    Separate parameters into weight decay (2D weight tensors) and non-decay
    (1D biases, layernorms, embedding scales).
    """
    decay_params: list[nn.Parameter] = []
    nodecay_params: list[nn.Parameter] = []

    for _name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.dim() >= 2:
            decay_params.append(param)
        else:
            nodecay_params.append(param)

    optim_groups = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": nodecay_params, "weight_decay": 0.0},
    ]

    use_fused = device_type == "cuda" and hasattr(torch.optim.AdamW, "fused")
    extra_args = {"fused": True} if use_fused else {}
    optimizer = torch.optim.AdamW(
        optim_groups,
        lr=learning_rate,
        betas=betas,
        **extra_args,
    )
    return optimizer


def get_cosine_lr(it: int, config: PretrainConfig) -> float:
    """Compute learning rate with linear warmup and cosine decay."""
    min_lr = config.min_lr if config.min_lr is not None else config.learning_rate * 0.1
    if it < config.warmup_iters:
        if config.warmup_iters == 0:
            return config.learning_rate
        return config.learning_rate * (it + 1) / config.warmup_iters
    if it >= config.max_iters:
        return min_lr
    decay_ratio = (it - config.warmup_iters) / (config.max_iters - config.warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (config.learning_rate - min_lr)


class SyntheticTokenBatchIterator:
    """Deterministic synthetic token batch iterator for smoke testing and validation."""

    def __init__(
        self,
        batch_size: int,
        seq_len: int,
        vocab_size: int,
        seed: int = 42,
        fixed_batch: bool = False,
    ) -> None:
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.rng = np.random.RandomState(seed)
        self.fixed_batch = fixed_batch
        self._cached_batch: tuple[torch.Tensor, torch.Tensor] | None = None

    def get_batch(self, device: str) -> tuple[torch.Tensor, torch.Tensor]:
        if self.fixed_batch and self._cached_batch is not None:
            x, y = self._cached_batch
            return x.to(device), y.to(device)

        data = self.rng.randint(0, self.vocab_size, size=(self.batch_size, self.seq_len + 1))
        x = torch.from_numpy(data[:, :-1].astype(np.int64)).to(device)
        y = torch.from_numpy(data[:, 1:].astype(np.int64)).to(device)

        if self.fixed_batch:
            self._cached_batch = (x.cpu(), y.cpu())

        return x, y


class BinaryShardTokenBatchIterator:
    """Reads batches from memory-mapped numpy/binary token arrays."""

    def __init__(
        self,
        data_path: str | Path,
        batch_size: int,
        seq_len: int,
        seed: int = 42,
    ) -> None:
        path = Path(data_path)
        if not path.exists():
            raise FileNotFoundError(f"Dataset file not found: {path}")

        dtype = np.uint32 if path.stat().st_size > 4 * 1024 * 1024 * 1024 else np.uint16
        self.data: Any
        try:
            self.data = np.memmap(path, dtype=dtype, mode="r")
        except Exception:
            self.data = np.fromfile(path, dtype=dtype)

        self.batch_size = batch_size
        self.seq_len = seq_len
        self.rng = np.random.RandomState(seed)
        self.num_tokens = len(self.data)
        if self.num_tokens <= seq_len + 1:
            raise ValueError(
                f"Data length ({self.num_tokens}) must be greater than seq_len + 1 ({seq_len + 1})"
            )

    def get_batch(self, device: str) -> tuple[torch.Tensor, torch.Tensor]:
        max_idx = self.num_tokens - self.seq_len - 1
        ix = self.rng.randint(0, max_idx, size=(self.batch_size,))
        x = np.stack([self.data[i : i + self.seq_len] for i in ix]).astype(np.int64)
        y = np.stack([self.data[i + 1 : i + 1 + self.seq_len] for i in ix]).astype(np.int64)
        return torch.from_numpy(x).to(device), torch.from_numpy(y).to(device)


@torch.no_grad()
def estimate_loss(
    model: BharatForCausalLM,
    data_iter: SyntheticTokenBatchIterator | BinaryShardTokenBatchIterator,
    eval_iters: int,
    device: str,
    amp_ctx: Any,
) -> float:
    """Estimate model loss across multiple evaluation batches."""
    model.eval()
    losses = torch.zeros(eval_iters)
    for k in range(eval_iters):
        x, y = data_iter.get_batch(device)
        with amp_ctx:
            output = model(input_ids=x, labels=y)
            loss = output.loss
            assert loss is not None
            losses[k] = loss.item()
    model.train()
    return float(losses.mean().item())


def pretrain(
    config: PretrainConfig,
    model: BharatForCausalLM | None = None,
    tokenizer: BharatTokenizer | None = None,
) -> PretrainResult:
    """
    Execute pretraining loop for BharatForCausalLM.
    """
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    # 1. Resolve Model Configuration
    if model is None:
        if config.model_config is not None:
            model_cfg = config.model_config
        else:
            assert config.model_config_path is not None
            model_cfg = load_model_config_from_yaml(config.model_config_path)
        model = BharatForCausalLM(model_cfg)
    else:
        model_cfg = model.config

    if config.seq_len > model_cfg.max_position_embeddings:
        raise ValueError(
            f"PretrainConfig.seq_len ({config.seq_len}) exceeds model max_position_embeddings "
            f"({model_cfg.max_position_embeddings})"
        )

    device = config.device
    model = model.to(device)
    model.train()

    # 2. Setup Optimizer
    optimizer = configure_optimizers(
        model,
        weight_decay=config.weight_decay,
        learning_rate=config.learning_rate,
        betas=(config.beta1, config.beta2),
        device_type=device.split(":")[0],
    )

    # 3. Setup Mixed Precision Context & Scaler
    dev_type = device.split(":")[0]
    use_amp = config.dtype in {"bfloat16", "float16"}
    ptdtype = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }[config.dtype]

    if dev_type == "cuda" and use_amp:
        amp_ctx = torch.amp.autocast(device_type="cuda", dtype=ptdtype)  # type: ignore[attr-defined]
        scaler = torch.amp.GradScaler("cuda", enabled=(config.dtype == "float16"))  # type: ignore[attr-defined]
    elif dev_type == "mps" and use_amp:
        amp_ctx = torch.amp.autocast(device_type="mps", dtype=ptdtype)  # type: ignore[attr-defined]
        scaler = torch.amp.GradScaler("mps", enabled=False)  # type: ignore[attr-defined]
    elif dev_type == "cpu" and use_amp:
        amp_ctx = torch.amp.autocast(device_type="cpu", dtype=ptdtype)  # type: ignore[attr-defined]
        scaler = torch.amp.GradScaler("cpu", enabled=False)  # type: ignore[attr-defined]
    else:
        amp_ctx = contextlib.nullcontext()
        scaler = torch.amp.GradScaler("cpu", enabled=False)  # type: ignore[attr-defined]

    # 4. Resume from Checkpoint if Requested
    start_step = 0
    if config.resume_checkpoint is not None:
        ckpt_meta = load_checkpoint(
            config.resume_checkpoint,
            model,
            optimizer=optimizer,
            tokenizer=tokenizer,
            device=device,
        )
        start_step = ckpt_meta.get("step", 0)

    # 5. Setup Data Iterators
    if config.synthetic_data or config.data_path is None:
        train_iter: SyntheticTokenBatchIterator | BinaryShardTokenBatchIterator = (
            SyntheticTokenBatchIterator(
                batch_size=config.batch_size,
                seq_len=config.seq_len,
                vocab_size=model_cfg.vocab_size,
                seed=config.seed,
            )
        )
    else:
        train_iter = BinaryShardTokenBatchIterator(
            data_path=config.data_path,
            batch_size=config.batch_size,
            seq_len=config.seq_len,
            seed=config.seed,
        )

    val_iter: SyntheticTokenBatchIterator | BinaryShardTokenBatchIterator | None = None
    if config.val_data_path is not None:
        val_iter = BinaryShardTokenBatchIterator(
            data_path=config.val_data_path,
            batch_size=config.batch_size,
            seq_len=config.seq_len,
            seed=config.seed + 1,
        )

    # 6. Training Loop Variables
    out_dir = Path(config.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    best_loss = float("inf")
    final_loss = float("inf")
    val_loss: float | None = None
    latest_checkpoint: str | None = None
    step_losses: list[float] = []
    tokens_per_step = config.batch_size * config.seq_len * config.gradient_accumulation_steps
    total_tokens_processed = 0

    # 7. Main Training Loop
    optimizer.zero_grad(set_to_none=True)

    for step in range(start_step, config.max_iters):
        lr = get_cosine_lr(step, config)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        accum_loss = 0.0

        for _micro_step in range(config.gradient_accumulation_steps):
            x, y = train_iter.get_batch(device)
            with amp_ctx:
                output = model(input_ids=x, labels=y)
                loss = output.loss
                assert loss is not None
                loss = loss / config.gradient_accumulation_steps

            if scaler.is_enabled():
                scaler.scale(loss).backward()
            else:
                loss.backward()

            accum_loss += loss.item() * config.gradient_accumulation_steps

        if config.grad_clip > 0.0:
            if scaler.is_enabled():
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)

        if scaler.is_enabled():
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()

        optimizer.zero_grad(set_to_none=True)

        final_loss = accum_loss
        step_losses.append(accum_loss)
        total_tokens_processed += tokens_per_step

        if accum_loss < best_loss:
            best_loss = accum_loss

        # Evaluation
        if (step + 1) % config.eval_interval == 0 or (step + 1) == config.max_iters:
            eval_target_iter = val_iter if val_iter is not None else train_iter
            val_loss = estimate_loss(
                model=model,
                data_iter=eval_target_iter,
                eval_iters=config.eval_iters,
                device=device,
                amp_ctx=amp_ctx,
            )

        # Checkpointing
        if (step + 1) % config.save_interval == 0 or (step + 1) == config.max_iters:
            ckpt_path = out_dir / f"ckpt_step_{step + 1}.pt"
            save_checkpoint(
                path=ckpt_path,
                model=model,
                optimizer=optimizer,
                config=model_cfg.to_dict(),
                tokenizer=tokenizer,
                step=step + 1,
                seed=config.seed,
                loss=accum_loss,
            )
            latest_checkpoint = str(ckpt_path)

    return PretrainResult(
        final_loss=final_loss,
        best_loss=best_loss,
        val_loss=val_loss,
        completed_steps=config.max_iters - start_step,
        total_tokens_processed=total_tokens_processed,
        checkpoint_path=latest_checkpoint,
        step_losses=step_losses,
    )
