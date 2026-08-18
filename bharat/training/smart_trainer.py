"""Smart Multi-Tier Training Pipeline for IndicLLM-Bharat.

Orchestrates native pretraining on Indic + Worldwide Knowledge Curriculum and
SFT instruction-tuning for models from tiny up to 10B parameters.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F

from bharat.data.synthetic_curriculum import export_curriculum_datasets
from bharat.models.bharat_model import BharatForCausalLM
from bharat.models.config import BharatModelConfig
from bharat.tokenizer import BharatTokenizer
from bharat.tokenizer import load_tokenizer as load_bharat_tokenizer
from bharat.training.checkpointing import save_checkpoint


@dataclass
class SmartTrainerConfig:
    model_tier: str = "small"
    curriculum_dir: str | Path = "data/curriculum"
    output_dir: str | Path = "checkpoints/bharat_smart"
    num_samples: int = 500
    pretrain_iters: int = 100
    sft_iters: int = 50
    batch_size: int = 2
    block_size: int = 256
    learning_rate: float = 5e-4
    warmup_iters: int = 20
    weight_decay: float = 0.05
    grad_clip: float = 1.0
    device: str = "auto"
    seed: int = 42


@dataclass
class SmartTrainerResult:
    final_pretrain_loss: float
    final_sft_loss: float
    checkpoint_path: str
    model_tier: str
    total_tokens: int


def get_tier_config(tier: str) -> BharatModelConfig:
    """Return architecture configuration for specified model tier."""
    t = tier.lower()
    if t == "tiny":
        return BharatModelConfig(
            vocab_size=64000,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=4096,
        )
    if t == "small":
        return BharatModelConfig(
            vocab_size=64000,
            hidden_size=256,
            intermediate_size=512,
            num_hidden_layers=4,
            num_attention_heads=8,
            num_key_value_heads=4,
            max_position_embeddings=4096,
        )
    if t == "350m":
        return BharatModelConfig(
            vocab_size=64000,
            hidden_size=1024,
            intermediate_size=2816,
            num_hidden_layers=24,
            num_attention_heads=16,
            num_key_value_heads=4,
            max_position_embeddings=4096,
        )
    if t == "1b":
        return BharatModelConfig(
            vocab_size=64000,
            hidden_size=2048,
            intermediate_size=5504,
            num_hidden_layers=24,
            num_attention_heads=16,
            num_key_value_heads=4,
            max_position_embeddings=4096,
        )
    if t == "3b":
        return BharatModelConfig(
            vocab_size=64000,
            hidden_size=3072,
            intermediate_size=8192,
            num_hidden_layers=28,
            num_attention_heads=24,
            num_key_value_heads=8,
            max_position_embeddings=4096,
        )
    if t == "7b":
        return BharatModelConfig(
            vocab_size=64000,
            hidden_size=4096,
            intermediate_size=13824,
            num_hidden_layers=32,
            num_attention_heads=32,
            num_key_value_heads=8,
            max_position_embeddings=4096,
        )
    if t == "10b":
        return BharatModelConfig(
            vocab_size=64000,
            hidden_size=4096,
            intermediate_size=14336,
            num_hidden_layers=44,
            num_attention_heads=32,
            num_key_value_heads=8,
            max_position_embeddings=4096,
            tie_word_embeddings=False,
        )
    raise ValueError(f"Unknown model tier '{tier}'. Supported: tiny, small, 350m, 1b, 3b, 7b, 10b")


def resolve_device(dev_str: str) -> torch.device:
    if dev_str == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(dev_str)


def get_default_tokenizer() -> BharatTokenizer:
    try:
        return load_bharat_tokenizer("gpt2")
    except Exception:
        from bharat.tokenizer.bpe import BPETokenizer
        from bharat.tokenizer.bpe_adapter import BharatBPETokenizer

        return BharatBPETokenizer(BPETokenizer())


def train_smart_bharat(config: SmartTrainerConfig) -> SmartTrainerResult:
    """Execute end-to-end multi-phase training on Indic & Worldwide curriculum."""
    torch.manual_seed(config.seed)
    dev = resolve_device(config.device)
    out_dir = Path(config.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Curriculum Dataset Preparation
    curr_dir = Path(config.curriculum_dir)
    pretrain_file = curr_dir / "pretrain_corpus.txt"
    sft_file = curr_dir / "sft_instruct.jsonl"
    if not pretrain_file.is_file() or not sft_file.is_file():
        print(f"Generating curriculum dataset in {curr_dir}...")
        pretrain_file, sft_file = export_curriculum_datasets(curr_dir, config.num_samples)

    # 2. Tokenizer & Model Initialization
    tokenizer = get_default_tokenizer()
    model_cfg = get_tier_config(config.model_tier)
    model = BharatForCausalLM(model_cfg).to(dev)

    total_params = sum(p.numel() for p in model.parameters())
    print(
        f"\n🚀 Initialized Bharat-{config.model_tier.upper()} ({total_params / 1e6:.1f}M params) on {dev}"
    )

    # 3. Read & Tokenize Pretrain Corpus
    with open(pretrain_file, encoding="utf-8") as f:
        pretrain_text = f.read()

    tokens = [t % model_cfg.vocab_size for t in tokenizer.encode(pretrain_text)]
    if len(tokens) < config.block_size + 1:
        tokens = tokens * ((config.block_size + 1) // len(tokens) + 1)
    data_tensor = torch.tensor(tokens, dtype=torch.long)

    # Optimizer with Cosine LR Warmup
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        betas=(0.9, 0.95),
    )

    def get_lr(step: int, max_steps: int) -> float:
        if step < config.warmup_iters:
            return config.learning_rate * (step + 1) / max(1, config.warmup_iters)
        decay_ratio = (step - config.warmup_iters) / max(1, max_steps - config.warmup_iters)
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return config.learning_rate * 0.1 + coeff * (config.learning_rate * 0.9)

    # Phase 1: Pretraining
    print(f"\n📚 Phase 1: Native Pretraining ({config.pretrain_iters} iterations)...")
    model.train()
    pretrain_loss = 0.0
    total_tokens_processed = 0

    for step in range(config.pretrain_iters):
        lr = get_lr(step, config.pretrain_iters)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        # Random batch extraction
        ix = torch.randint(len(data_tensor) - config.block_size, (config.batch_size,))
        x = torch.stack([data_tensor[i : i + config.block_size] for i in ix]).to(dev)
        y = torch.stack([data_tensor[i + 1 : i + 1 + config.block_size] for i in ix]).to(dev)

        optimizer.zero_grad()
        out = model(x)
        loss = F.cross_entropy(out.logits.view(-1, model_cfg.vocab_size), y.view(-1))
        loss.backward()  # type: ignore[no-untyped-call]
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        optimizer.step()

        pretrain_loss = loss.item()
        total_tokens_processed += config.batch_size * config.block_size

        if (step + 1) % max(
            1, config.pretrain_iters // 5
        ) == 0 or step == config.pretrain_iters - 1:
            print(
                f"  Step {step + 1:4d}/{config.pretrain_iters} | Loss: {pretrain_loss:.4f} | LR: {lr:.6f}"
            )

    # Phase 2: SFT Instruction Tuning
    print(f"\n🎯 Phase 2: Native SFT Instruction Alignment ({config.sft_iters} iterations)...")
    sft_samples: list[str] = []
    with open(sft_file, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                sample = json.loads(line)
                dialogue = ""
                for msg in sample.get("messages", []):
                    dialogue += f"<|im_start|>{msg['role']}\n{msg['content']}<|im_end|>\n"
                sft_samples.append(dialogue)

    sft_loss = 0.0
    if sft_samples and config.sft_iters > 0:
        for step in range(config.sft_iters):
            chosen = sft_samples[step % len(sft_samples)]
            sft_tokens = [t % model_cfg.vocab_size for t in tokenizer.encode(chosen)]
            if len(sft_tokens) > config.block_size:
                sft_tokens = sft_tokens[: config.block_size]
            elif len(sft_tokens) < 4:
                continue

            sx = torch.tensor([sft_tokens[:-1]], dtype=torch.long, device=dev)
            sy = torch.tensor([sft_tokens[1:]], dtype=torch.long, device=dev)

            optimizer.zero_grad()
            out = model(sx)
            loss = F.cross_entropy(out.logits.view(-1, model_cfg.vocab_size), sy.view(-1))
            loss.backward()  # type: ignore[no-untyped-call]
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimizer.step()

            sft_loss = loss.item()
            if (step + 1) % max(1, config.sft_iters // 3) == 0 or step == config.sft_iters - 1:
                print(f"  SFT Step {step + 1:3d}/{config.sft_iters} | Loss: {sft_loss:.4f}")

    # 4. Save Final Production Checkpoint
    final_ckpt_p = out_dir / "final.pt"
    save_checkpoint(
        path=final_ckpt_p,
        model=model,
        optimizer=optimizer,
        config=model_cfg.to_dict(),
        tokenizer=tokenizer,
        step=config.pretrain_iters + config.sft_iters,
        loss=sft_loss or pretrain_loss,
    )
    print(f"\n✅ Production checkpoint successfully saved to: {final_ckpt_p}")

    return SmartTrainerResult(
        final_pretrain_loss=pretrain_loss,
        final_sft_loss=sft_loss,
        checkpoint_path=str(final_ckpt_p),
        model_tier=config.model_tier,
        total_tokens=total_tokens_processed,
    )
