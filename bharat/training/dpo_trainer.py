"""Sovereign Direct Preference Optimization (DPO) Alignment Trainer for IndicLLM-Bharat.

Aligns policy models against reference checkpoints using direct log-ratio optimization
across multilingual Indian language and global knowledge preference pairs.
"""

from __future__ import annotations

import copy
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from bharat.data.preference_curriculum import export_preference_curriculum
from bharat.models.bharat_model import BharatForCausalLM
from bharat.models.config import BharatModelConfig
from bharat.tokenizer import BharatTokenizer
from bharat.tokenizer import load_tokenizer as load_bharat_tokenizer
from bharat.training.checkpointing import save_checkpoint


@dataclass
class DPOTrainerConfig:
    sft_checkpoint: str | Path = "checkpoints/bharat_smart/final.pt"
    preference_data: str | Path = "data/preferences/bharat_dpo_curriculum.jsonl"
    output_dir: str | Path = "checkpoints/bharat_dpo"
    model_tier: str = "small"
    max_iters: int = 60
    batch_size: int = 2
    block_size: int = 512
    learning_rate: float = 5e-5
    beta: float = 0.1
    warmup_iters: int = 10
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    device: str = "auto"
    seed: int = 42


@dataclass
class DPOTrainerResult:
    final_loss: float
    final_reward_accuracy: float
    final_reward_margin: float
    checkpoint_path: str
    total_samples: int
    completed_steps: int


def _get_log_probs(
    logits: torch.Tensor,
    labels: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Compute sum of log probabilities of target tokens given logits."""
    # Shift logits and labels for autoregressive prediction: logits[:, :-1], labels[:, 1:]
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    shift_mask = mask[:, 1:].contiguous()

    log_probs = F.log_softmax(shift_logits, dim=-1)
    per_token_log_probs = torch.gather(log_probs, dim=-1, index=shift_labels.unsqueeze(-1)).squeeze(
        -1
    )

    # Apply mask and sum over sequence dimension
    masked_log_probs = per_token_log_probs * shift_mask.float()
    return masked_log_probs.sum(dim=-1)


def build_dpo_sequence(
    tokenizer: BharatTokenizer,
    prompt: str,
    response: str,
    block_size: int,
) -> tuple[list[int], list[bool]]:
    """Format instruction dialogue and build binary response mask."""
    full_prompt = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
    prompt_ids = tokenizer.encode(full_prompt, add_special_tokens=False)
    response_ids = tokenizer.encode(f"{response}<|im_end|>\n", add_special_tokens=False)

    input_ids = (prompt_ids + response_ids)[:block_size]
    mask = [False] * min(len(prompt_ids), block_size)
    rem = block_size - len(mask)
    if rem > 0:
        mask.extend([True] * min(len(response_ids), rem))

    return input_ids, mask


class BharatDPOTrainer:
    """End-to-end sovereign Direct Preference Optimization trainer."""

    def __init__(self, config: DPOTrainerConfig) -> None:
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
        self.tokenizer = self._load_tokenizer()

        # 3. Policy and Reference Models
        self.policy_model, self.model_config = self._init_policy_model()
        self.ref_model = copy.deepcopy(self.policy_model).to(self.device)
        self.ref_model.eval()
        for p in self.ref_model.parameters():
            p.requires_grad = False

        # 4. Optimizer & Schedule
        self.optimizer = torch.optim.AdamW(
            [p for p in self.policy_model.parameters() if p.requires_grad],
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            betas=(0.9, 0.95),
            eps=1e-8,
        )

    def _load_tokenizer(self) -> BharatTokenizer:
        try:
            return load_bharat_tokenizer("gpt2")
        except Exception:
            return load_bharat_tokenizer()

    def _init_policy_model(self) -> tuple[BharatForCausalLM, BharatModelConfig]:
        ckpt_p = Path(self.config.sft_checkpoint)
        if ckpt_p.is_file():
            ckpt = torch.load(ckpt_p, map_location=self.device, weights_only=False)
            if "metadata" in ckpt and hasattr(ckpt["metadata"], "model_config"):
                cfg = BharatModelConfig.from_dict(ckpt["metadata"].model_config)
            elif "model_config" in ckpt:
                cfg = BharatModelConfig.from_dict(ckpt["model_config"])
            elif "config" in ckpt and isinstance(ckpt["config"], dict):
                cfg = BharatModelConfig.from_dict(ckpt["config"])
            else:
                cfg = BharatModelConfig(
                    vocab_size=self.tokenizer.vocab_size,
                    hidden_size=128,
                    intermediate_size=256,
                    num_hidden_layers=2,
                    num_attention_heads=4,
                    num_key_value_heads=2,
                    max_position_embeddings=4096,
                )
            model = BharatForCausalLM(cfg).to(self.device)
            if "model" in ckpt:
                model.load_state_dict(ckpt["model"], strict=False)
            print(
                f"Loaded SFT Checkpoint from {ckpt_p} ({sum(p.numel() for p in model.parameters()):,} params)"
            )
            return model, cfg

        cfg = BharatModelConfig(
            vocab_size=self.tokenizer.vocab_size,
            hidden_size=128,
            intermediate_size=256,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=4096,
        )
        model = BharatForCausalLM(cfg).to(self.device)
        return model, cfg

    def _get_lr(self, step: int) -> float:
        """Cosine annealing learning rate schedule with linear warmup."""
        if step < self.config.warmup_iters:
            return self.config.learning_rate * (step + 1) / self.config.warmup_iters
        if step > self.config.max_iters:
            return self.config.learning_rate * 0.1
        decay_ratio = (step - self.config.warmup_iters) / (
            self.config.max_iters - self.config.warmup_iters
        )
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return self.config.learning_rate * (0.1 + 0.9 * coeff)

    def load_preference_dataset(self) -> list[dict[str, Any]]:
        p_path = Path(self.config.preference_data)
        if not p_path.is_file():
            print(f"Preference dataset not found at {p_path}. Synthesizing from curriculum...")
            export_preference_curriculum(p_path)

        samples: list[dict[str, Any]] = []
        with open(p_path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    samples.append(json.loads(line))
        return samples

    def train(self) -> DPOTrainerResult:
        """Execute full DPO training loop."""
        dataset = self.load_preference_dataset()
        if not dataset:
            raise ValueError("Preference dataset is empty!")

        self.policy_model.train()
        total_steps = self.config.max_iters
        vocab_size = self.model_config.vocab_size

        print("\n" + "=" * 60)
        print("🚀 Starting IndicLLM-Bharat DPO Preference Alignment")
        print(f"  • Device:          {self.device}")
        print(f"  • Beta:            {self.config.beta}")
        print(f"  • Total Steps:     {total_steps}")
        print(f"  • Preference Pairs:{len(dataset)}")
        print("=" * 60 + "\n")

        running_loss = 0.0
        running_acc = 0.0
        running_margin = 0.0
        step = 0

        while step < total_steps:
            # Batch creation
            batch_indices = [
                (step * self.config.batch_size + i) % len(dataset)
                for i in range(self.config.batch_size)
            ]
            batch = [dataset[idx] for idx in batch_indices]

            chosen_ids_list: list[list[int]] = []
            chosen_mask_list: list[list[bool]] = []
            rejected_ids_list: list[list[int]] = []
            rejected_mask_list: list[list[bool]] = []

            for item in batch:
                c_ids, c_mask = build_dpo_sequence(
                    self.tokenizer, item["prompt"], item["chosen"], self.config.block_size
                )
                r_ids, r_mask = build_dpo_sequence(
                    self.tokenizer, item["prompt"], item["rejected"], self.config.block_size
                )
                chosen_ids_list.append([t % vocab_size for t in c_ids])
                chosen_mask_list.append(c_mask)
                rejected_ids_list.append([t % vocab_size for t in r_ids])
                rejected_mask_list.append(r_mask)

            # Pad batches
            max_c_len = max(len(ids) for ids in chosen_ids_list)
            max_r_len = max(len(ids) for ids in rejected_ids_list)

            c_ids_padded = [ids + [0] * (max_c_len - len(ids)) for ids in chosen_ids_list]
            c_mask_padded = [m + [False] * (max_c_len - len(m)) for m in chosen_mask_list]
            r_ids_padded = [ids + [0] * (max_r_len - len(ids)) for ids in rejected_ids_list]
            r_mask_padded = [m + [False] * (max_r_len - len(m)) for m in rejected_mask_list]

            c_ids_t = torch.tensor(c_ids_padded, dtype=torch.long, device=self.device)
            c_mask_t = torch.tensor(c_mask_padded, dtype=torch.bool, device=self.device)
            r_ids_t = torch.tensor(r_ids_padded, dtype=torch.long, device=self.device)
            r_mask_t = torch.tensor(r_mask_padded, dtype=torch.bool, device=self.device)

            # LR schedule
            lr = self._get_lr(step)
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = lr

            # 1. Policy forward
            self.optimizer.zero_grad()
            c_policy_logits = self.policy_model(c_ids_t).logits
            r_policy_logits = self.policy_model(r_ids_t).logits

            c_policy_log_probs = _get_log_probs(c_policy_logits, c_ids_t, c_mask_t)
            r_policy_log_probs = _get_log_probs(r_policy_logits, r_ids_t, r_mask_t)

            # 2. Reference forward (frozen)
            with torch.no_grad():
                c_ref_logits = self.ref_model(c_ids_t).logits
                r_ref_logits = self.ref_model(r_ids_t).logits
                c_ref_log_probs = _get_log_probs(c_ref_logits, c_ids_t, c_mask_t)
                r_ref_log_probs = _get_log_probs(r_ref_logits, r_ids_t, r_mask_t)

            # 3. DPO Loss & Implicit Rewards
            pi_log_ratios = c_policy_log_probs - r_policy_log_probs
            ref_log_ratios = c_ref_log_probs - r_ref_log_probs

            chosen_rewards = self.config.beta * (c_policy_log_probs - c_ref_log_probs)
            rejected_rewards = self.config.beta * (r_policy_log_probs - r_ref_log_probs)
            reward_margins = chosen_rewards - rejected_rewards

            logits = self.config.beta * (pi_log_ratios - ref_log_ratios)
            loss = -F.logsigmoid(logits).mean()

            # Optimization step
            loss.backward()  # type: ignore[no-untyped-call]
            if self.config.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.policy_model.parameters(), self.config.grad_clip
                )
            self.optimizer.step()

            # Metrics
            acc = (reward_margins > 0).float().mean().item()
            margin = reward_margins.mean().item()

            running_loss = 0.9 * running_loss + 0.1 * loss.item() if step > 0 else loss.item()
            running_acc = 0.9 * running_acc + 0.1 * acc if step > 0 else acc
            running_margin = 0.9 * running_margin + 0.1 * margin if step > 0 else margin

            if (step + 1) % 10 == 0 or step == total_steps - 1:
                print(
                    f"Step {step+1:3d}/{total_steps:3d} | "
                    f"Loss: {loss.item():.4f} (avg: {running_loss:.4f}) | "
                    f"Reward Acc: {running_acc*100:5.1f}% | "
                    f"Reward Margin: {running_margin:+.4f} | "
                    f"LR: {lr:.2e}"
                )

            step += 1

        # 4. Save Final Aligned Checkpoint
        out_dir = Path(self.config.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        final_ckpt = out_dir / "final.pt"

        save_checkpoint(
            path=final_ckpt,
            model=self.policy_model,
            optimizer=self.optimizer,
            config=self.model_config.to_dict(),
            tokenizer=self.tokenizer,
            step=total_steps,
            loss=running_loss,
        )

        print("\n" + "=" * 60)
        print("✅ DPO Preference Alignment Complete!")
        print(f"  • Final Loss:          {running_loss:.4f}")
        print(f"  • Final Reward Acc:    {running_acc*100:.1f}%")
        print(f"  • Final Reward Margin: {running_margin:+.4f}")
        print(f"  • Saved Checkpoint:    {final_ckpt}")
        print("=" * 60 + "\n")

        return DPOTrainerResult(
            final_loss=running_loss,
            final_reward_accuracy=running_acc,
            final_reward_margin=running_margin,
            checkpoint_path=str(final_ckpt),
            total_samples=len(dataset),
            completed_steps=total_steps,
        )
