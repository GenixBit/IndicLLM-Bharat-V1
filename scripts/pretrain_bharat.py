from __future__ import annotations

import argparse
import json
import sys

from bharat.training.pretrain import PretrainConfig, load_model_config_from_yaml, pretrain


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Pretrain a Bharat architecture language model (e.g. Bharat-350M, Bharat-1B)"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/models/bharat-350m.yaml",
        help="Path to YAML model configuration file",
    )
    parser.add_argument(
        "--data",
        type=str,
        default=None,
        help="Path to binary training dataset (.bin)",
    )
    parser.add_argument(
        "--val-data",
        type=str,
        default=None,
        help="Path to binary validation dataset (.bin)",
    )
    parser.add_argument(
        "--synthetic-data",
        action="store_true",
        help="Use deterministic synthetic data for smoke/overfit testing",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="checkpoints/bharat-350m",
        help="Directory to save checkpoints",
    )
    parser.add_argument(
        "--max-iters",
        type=int,
        default=1000,
        help="Total training iterations",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size per step",
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=512,
        help="Sequence length (context window)",
    )
    parser.add_argument(
        "--grad-accum",
        type=int,
        default=1,
        help="Gradient accumulation steps",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=3e-4,
        help="Peak learning rate",
    )
    parser.add_argument(
        "--min-lr",
        type=float,
        default=None,
        help="Minimum learning rate after cosine decay",
    )
    parser.add_argument(
        "--warmup-iters",
        type=int,
        default=100,
        help="Number of linear warmup iterations",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.1,
        help="Weight decay coefficient for 2D parameters",
    )
    parser.add_argument(
        "--grad-clip",
        type=float,
        default=1.0,
        help="Max gradient norm for clipping",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda", "mps"],
        help="Training compute device",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="float32",
        choices=["float32", "bfloat16", "float16"],
        help="Computation data type / precision",
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=10,
        help="Log progress every N steps",
    )
    parser.add_argument(
        "--eval-interval",
        type=int,
        default=100,
        help="Evaluate model every N steps",
    )
    parser.add_argument(
        "--eval-iters",
        type=int,
        default=10,
        help="Number of batches to evaluate on",
    )
    parser.add_argument(
        "--save-interval",
        type=int,
        default=500,
        help="Save checkpoint every N steps",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume training from",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output final training result as JSON",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        model_cfg = load_model_config_from_yaml(args.config)
    except Exception as e:
        print(f"Error loading model configuration: {e}", file=sys.stderr)
        return 1

    pretrain_cfg = PretrainConfig(
        model_config=model_cfg,
        data_path=args.data,
        val_data_path=args.val_data,
        synthetic_data=args.synthetic_data or (args.data is None),
        output_dir=args.output_dir,
        max_iters=args.max_iters,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        min_lr=args.min_lr,
        warmup_iters=args.warmup_iters,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        device=args.device,
        dtype=args.dtype,
        log_interval=args.log_interval,
        eval_interval=args.eval_interval,
        eval_iters=args.eval_iters,
        save_interval=args.save_interval,
        seed=args.seed,
        resume_checkpoint=args.resume,
    )

    if not args.json:
        print("=" * 60)
        print("🇮🇳 Bharat AI Pretraining Engine")
        print(f"Model Configuration: {args.config}")
        print(f"Vocab Size: {model_cfg.vocab_size:,} | Hidden Size: {model_cfg.hidden_size}")
        print(
            f"Layers: {model_cfg.num_hidden_layers} | Q Heads: {model_cfg.num_attention_heads} | KV Heads: {model_cfg.num_key_value_heads}"
        )
        print(f"Device: {args.device} | Precision: {args.dtype} | Max Steps: {args.max_iters}")
        print("=" * 60)

    try:
        result = pretrain(pretrain_cfg)
    except Exception as e:
        print(f"Pretraining failed with error: {e}", file=sys.stderr)
        return 1

    summary = {
        "status": "success",
        "completed_steps": result.completed_steps,
        "final_loss": round(result.final_loss, 4),
        "best_loss": round(result.best_loss, 4),
        "val_loss": round(result.val_loss, 4) if result.val_loss is not None else None,
        "total_tokens_processed": result.total_tokens_processed,
        "checkpoint_path": result.checkpoint_path,
    }

    if args.json:
        print(json.dumps(summary, indent=2))
    else:
        print("\n" + "=" * 60)
        print("Training Completed Successfully!")
        print(f"Final Loss: {summary['final_loss']}")
        print(f"Best Loss:  {summary['best_loss']}")
        if summary["val_loss"] is not None:
            print(f"Val Loss:   {summary['val_loss']}")
        print(f"Total Tokens: {summary['total_tokens_processed']:,}")
        print(f"Checkpoint: {summary['checkpoint_path']}")
        print("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
