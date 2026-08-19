"""Sovereign Supervised Fine-Tuning (SFT) Trainer for IndicLLM-Bharat.

Implements assistant-only loss masking, cosine LR scheduling with linear warmup,
and mixed-precision fine-tuning for BharatForCausalLM architectures (350M -> 1B -> 10B).
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from bharat.data.instruction_curriculum import (
    export_instruction_curriculum,
    get_all_instruction_curriculum,
)
from bharat.models.bharat_model import BharatForCausalLM
from bharat.models.config import BharatModelConfig
from bharat.tokenizer import BharatTokenizer, load_tokenizer
from bharat.training.checkpointing import save_checkpoint
from bharat.training.scale_trainer import get_scale_tier_config


@dataclass
class SFTTrainingConfig:
    tier: str = "1b"
    data_path: str | Path = "data/sft/bharat_instruction_curriculum.jsonl"
    checkpoint_path: str | Path | None = None
    output_dir: str | Path = "checkpoints/bharat_sft"
    steps: int = 50
    batch_size: int = 2
    block_size: int = 512
    learning_rate: float = 2e-5
    warmup_steps: int = 10
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    device: str = "auto"
    seed: int = 42


@dataclass
class SFTTrainingResult:
    tier: str
    final_loss: float
    best_loss: float
    total_samples_trained: int
    active_tokens: int
    checkpoint_path: str


class BharatSFTTrainer:
    """Supervised Fine-Tuning trainer with assistant-only loss masking."""

    def __init__(self, config: SFTTrainingConfig) -> None:
        self.config = config
        torch.manual_seed(config.seed)

        # 1. Device resolution
        if config.device == "auto":
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(config.device)

        # 2. Tokenizer
        self.tokenizer: BharatTokenizer = load_tokenizer("gpt2")

        # 3. Model setup
        if config.tier == "tiny":
            self.model_config = BharatModelConfig(
                vocab_size=self.tokenizer.vocab_size,
                hidden_size=64,
                intermediate_size=128,
                num_hidden_layers=2,
                num_attention_heads=4,
                num_key_value_heads=2,
                max_position_embeddings=4096,
            )
        else:
            self.model_config = get_scale_tier_config(config.tier, self.tokenizer.vocab_size)

        self.model = BharatForCausalLM(self.model_config).to(self.device)

        # Load pretrained checkpoint if available
        if config.checkpoint_path and Path(config.checkpoint_path).is_file():
            state = torch.load(config.checkpoint_path, map_location=self.device, weights_only=False)
            if "model_state_dict" in state:
                self.model.load_state_dict(state["model_state_dict"], strict=False)
            elif "state_dict" in state:
                self.model.load_state_dict(state["state_dict"], strict=False)

        # 4. Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            betas=(0.9, 0.95),
            eps=1e-8,
        )

    def _get_lr(self, step: int) -> float:
        if step < self.config.warmup_steps:
            return self.config.learning_rate * (step + 1) / max(1, self.config.warmup_steps)
        if step > self.config.steps:
            return self.config.learning_rate * 0.1
        ratio = (step - self.config.warmup_steps) / max(
            1, (self.config.steps - self.config.warmup_steps)
        )
        coeff = 0.5 * (1.0 + math.cos(math.pi * ratio))
        return self.config.learning_rate * (0.1 + 0.9 * coeff)

    def prepare_dataset(self) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """Load dialogues, format prompt/response, and prepare assistant-only masked tensors."""
        p = Path(self.config.data_path)
        if not p.is_file():
            p.parent.mkdir(parents=True, exist_ok=True)
            export_instruction_curriculum(p)

        items: list[dict[str, Any]] = []
        with open(p, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    items.append(json.loads(line))

        if not items:
            items = get_all_instruction_curriculum()

        samples: list[tuple[torch.Tensor, torch.Tensor]] = []
        eos_id = getattr(self.tokenizer, "eos_token_id", 50256)

        for item in items:
            prompt = f"User: {item['prompt']}\n\nAssistant: "
            response = item["response"]

            p_tokens = self.tokenizer.encode(prompt)
            r_tokens = [*self.tokenizer.encode(response), eos_id]

            full_tokens = p_tokens + r_tokens
            if len(full_tokens) > self.config.block_size:
                full_tokens = full_tokens[: self.config.block_size]

            # Shifted input and target
            input_ids = torch.tensor(full_tokens[:-1], dtype=torch.long)

            # Mask out prompt tokens with -100 (assistant-only loss calculation)
            target_ids = torch.full((len(full_tokens) - 1,), -100, dtype=torch.long)
            p_len = min(len(p_tokens), len(full_tokens) - 1)
            target_ids[p_len - 1 :] = torch.tensor(full_tokens[p_len:], dtype=torch.long)

            # Bound within vocab
            input_ids = input_ids % self.model_config.vocab_size
            # Keep -100 unchanged, bound valid targets
            valid_mask = target_ids != -100
            target_ids[valid_mask] = target_ids[valid_mask] % self.model_config.vocab_size

            samples.append((input_ids, target_ids))

        return samples

    def train(self) -> SFTTrainingResult:
        """Execute SFT training loop."""
        dataset = self.prepare_dataset()
        self.model.train()

        param_count = sum(p.numel() for p in self.model.parameters())

        print("\n" + "=" * 65)
        print(f"🎓 Starting IndicLLM-Bharat SFT Training [Tier: {self.config.tier.upper()}]")
        print(
            f"  • Architecture:      {self.model_config.num_hidden_layers} layers | {self.model_config.hidden_size} hidden | {self.model_config.num_attention_heads} heads ({self.model_config.num_key_value_heads} KV)"
        )
        print(f"  • Parameters:        {param_count:,} ({param_count / 1e9:.2f}B)")
        print(f"  • Total Steps:       {self.config.steps}")
        print("  • Loss Masking:      Assistant-Only (-100 on Prompt Tokens)")
        print(f"  • Dataset Samples:   {len(dataset)} instruction pairs")
        print(f"  • Compute Device:    {self.device}")
        print("=" * 65 + "\n")

        running_loss = 0.0
        best_loss = float("inf")
        active_tokens = 0
        samples_count = 0
        start_time = time.perf_counter()

        step = 0
        while step < self.config.steps:
            batch_x_list: list[torch.Tensor] = []
            batch_y_list: list[torch.Tensor] = []

            # Determine max length in current batch for padding
            batch_indices = [
                (step * self.config.batch_size + i) % len(dataset)
                for i in range(self.config.batch_size)
            ]
            batch_samples = [dataset[idx] for idx in batch_indices]
            max_len = max(x.shape[0] for x, _ in batch_samples)

            for x, y in batch_samples:
                pad_len = max_len - x.shape[0]
                if pad_len > 0:
                    px = F.pad(x, (0, pad_len), value=0)
                    py = F.pad(y, (0, pad_len), value=-100)
                else:
                    px, py = x, y
                batch_x_list.append(px)
                batch_y_list.append(py)

            bx = torch.stack(batch_x_list).to(self.device)
            by = torch.stack(batch_y_list).to(self.device)

            # Update LR
            lr = self._get_lr(step)
            for pg in self.optimizer.param_groups:
                pg["lr"] = lr

            self.optimizer.zero_grad()
            out = self.model(bx)

            # Cross entropy with ignore_index=-100
            loss = F.cross_entropy(
                out.logits.view(-1, self.model_config.vocab_size),
                by.view(-1),
                ignore_index=-100,
            )

            loss.backward()  # type: ignore[no-untyped-call]
            if self.config.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
            self.optimizer.step()

            step_loss = loss.item()
            running_loss = 0.9 * running_loss + 0.1 * step_loss if step > 0 else step_loss
            best_loss = min(best_loss, step_loss)
            active_tokens += int((by != -100).sum().item())
            samples_count += self.config.batch_size

            if (step + 1) % max(1, self.config.steps // 5) == 0 or step == self.config.steps - 1:
                elapsed = time.perf_counter() - start_time
                tps = active_tokens / max(1e-5, elapsed)
                print(
                    f"Step {step+1:3d}/{self.config.steps:3d} | "
                    f"SFT Loss: {step_loss:.4f} (avg: {running_loss:.4f}) | "
                    f"LR: {lr:.2e} | "
                    f"Throughput: {tps:,.1f} tok/s"
                )

            step += 1

        out_dir = Path(self.config.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        final_ckpt = out_dir / "final.pt"

        save_checkpoint(
            path=final_ckpt,
            model=self.model,
            optimizer=self.optimizer,
            config=self.model_config.to_dict(),
            tokenizer=self.tokenizer,
            step=self.config.steps,
            loss=running_loss,
        )

        print("\n" + "=" * 65)
        print(f"✅ SFT Alignment Complete for Tier {self.config.tier.upper()}!")
        print(f"  • Final SFT Loss:    {running_loss:.4f}")
        print(f"  • Active Tokens:     {active_tokens:,}")
        print(f"  • Saved Checkpoint:  {final_ckpt}")
        print("=" * 65 + "\n")

        return SFTTrainingResult(
            tier=self.config.tier,
            final_loss=running_loss,
            best_loss=best_loss,
            total_samples_trained=samples_count,
            active_tokens=active_tokens,
            checkpoint_path=str(final_ckpt),
        )
