#!/usr/bin/env python3
"""
Bharat model parameter and memory calculator.

Usage:
    scripts/calculate_params.py configs/models/bharat-350m.yaml
    scripts/calculate_params.py configs/models/bharat-7b.yaml --weight-dtype bf16 --batch-size 1 --sequence-length 4096
    scripts/calculate_params.py --all --weight-dtype bf16 --optimizer adamw_fp32 --gradient-dtype bf16 --fp32-master-weights
    scripts/calculate_params.py configs/models/bharat-1b.yaml --json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from bharat.models.sizing import (
    KVCacheMemoryReport,
    ParameterCount,
    StaticMemoryReport,
    calculate_kv_cache_memory,
    calculate_parameter_count,
    calculate_static_memory,
)
from bharat.models.spec import load_model_spec

ROOT = Path(__file__).resolve().parent.parent
CONFIGS_DIR = ROOT / "configs" / "models"

PRODUCTION_CONFIGS: list[str] = [
    "bharat-350m.yaml",
    "bharat-1b.yaml",
    "bharat-3b.yaml",
    "bharat-7b.yaml",
]

ONE_PERCENT = 0.01


def _gib(bytes_val: int) -> str:
    return f"{bytes_val / (1024**3):.4f} GiB"


def _mib(bytes_val: int) -> str:
    return f"{bytes_val / (1024**2):.4f} MiB"


def _fail(msg: str) -> None:
    print(f"error: {msg}", file=sys.stderr)
    sys.exit(1)


def human_report(
    spec, params: ParameterCount, memory: StaticMemoryReport | None, kv: KVCacheMemoryReport | None
) -> None:
    arch = spec.architecture
    head_dim = arch.hidden_size // arch.num_attention_heads
    gqa_groups = arch.num_attention_heads // arch.num_key_value_heads

    print(f"Model: {spec.model_name} ({spec.size_label})")
    print(f"  Target parameters: {spec.target_parameter_count:,}")
    print(f"  Expected parameters: {spec.expected_parameter_count:,}")
    diff_pct = (params.total - spec.target_parameter_count) / spec.target_parameter_count * 100
    print(f"  Actual parameters: {params.total:,}")
    print(f"  Difference from target: {diff_pct:+.4f}%")
    print()
    print("Architecture:")
    print(f"  vocab_size:             {arch.vocab_size}")
    print(f"  hidden_size:            {arch.hidden_size}")
    print(f"  intermediate_size:      {arch.intermediate_size}")
    print(f"  num_hidden_layers:      {arch.num_hidden_layers}")
    print(f"  num_attention_heads:    {arch.num_attention_heads}")
    print(f"  num_key_value_heads:    {arch.num_key_value_heads}")
    print(f"  head_dim:               {head_dim}")
    print(f"  GQA groups:             {gqa_groups}")
    print(f"  max_position_embeddings: {arch.max_position_embeddings}")
    print(f"  rope_theta:             {arch.rope_theta}")
    print(f"  tie_word_embeddings:    {arch.tie_word_embeddings}")
    print(f"  attention_bias:         {arch.attention_bias}")
    print(f"  mlp_bias:               {arch.mlp_bias}")
    print()
    print("Parameter breakdown:")
    print(f"  token_embeddings:       {params.token_embeddings:>12,}")
    print(f"  attention_per_layer:    {params.attention_per_layer:>12,}")
    print(f"  mlp_per_layer:          {params.mlp_per_layer:>12,}")
    print(f"  norms_per_layer:        {params.norms_per_layer:>12,}")
    print(f"  transformer_layers:     {params.transformer_layers:>12,}")
    print(f"  final_norm:             {params.final_norm:>12,}")
    print(f"  lm_head:                {params.lm_head:>12,}")
    print(f"  total:                  {params.total:>12,}")
    print()

    if memory is not None:
        print("Weight memory:")
        print(
            f"  weight:                 {memory.weight_bytes:>12,}  ({_gib(memory.weight_bytes)})"
        )
        if memory.gradient_bytes > 0:
            print(
                f"  gradients:              {memory.gradient_bytes:>12,}  ({_gib(memory.gradient_bytes)})"
            )
        if memory.master_weight_bytes > 0:
            print(
                f"  master weights (fp32):  {memory.master_weight_bytes:>12,}  ({_gib(memory.master_weight_bytes)})"
            )
        if memory.optimizer_state_bytes > 0:
            print(
                f"  optimizer state:        {memory.optimizer_state_bytes:>12,}  ({_gib(memory.optimizer_state_bytes)})"
            )
        print(
            f"  total training state:   {memory.total_training_state_bytes:>12,}  ({_gib(memory.total_training_state_bytes)})"
        )
        print()
        print("  Note: Activation memory and framework overhead are excluded.")
        print()

    if kv is not None:
        print(f"KV cache ({kv.bytes_per_token_per_batch_item} bytes/token/batch item):")
        print(f"  total:                  {kv.total_bytes:>12,}  ({_mib(kv.total_bytes)})")
        print()


def json_report(spec, params, memory, kv) -> dict:
    arch = spec.architecture
    head_dim = arch.hidden_size // arch.num_attention_heads
    gqa_groups = arch.num_attention_heads // arch.num_key_value_heads
    diff_pct = (params.total - spec.target_parameter_count) / spec.target_parameter_count * 100

    data: dict = {
        "model_name": spec.model_name,
        "size_label": spec.size_label,
        "target_parameter_count": spec.target_parameter_count,
        "expected_parameter_count": spec.expected_parameter_count,
        "actual_parameter_count": params.total,
        "difference_percent": round(diff_pct, 4),
        "architecture": {
            "vocab_size": arch.vocab_size,
            "hidden_size": arch.hidden_size,
            "intermediate_size": arch.intermediate_size,
            "num_hidden_layers": arch.num_hidden_layers,
            "num_attention_heads": arch.num_attention_heads,
            "num_key_value_heads": arch.num_key_value_heads,
            "head_dim": head_dim,
            "gqa_groups": gqa_groups,
            "max_position_embeddings": arch.max_position_embeddings,
            "rope_theta": arch.rope_theta,
            "tie_word_embeddings": arch.tie_word_embeddings,
            "attention_bias": arch.attention_bias,
            "mlp_bias": arch.mlp_bias,
        },
        "parameter_breakdown": {
            "token_embeddings": params.token_embeddings,
            "attention_per_layer": params.attention_per_layer,
            "mlp_per_layer": params.mlp_per_layer,
            "norms_per_layer": params.norms_per_layer,
            "transformer_layers": params.transformer_layers,
            "final_norm": params.final_norm,
            "lm_head": params.lm_head,
            "total": params.total,
        },
    }

    if memory is not None:
        data["weight_memory"] = {
            "weight_bytes": memory.weight_bytes,
            "gradient_bytes": memory.gradient_bytes,
            "master_weight_bytes": memory.master_weight_bytes,
            "optimizer_state_bytes": memory.optimizer_state_bytes,
            "total_training_state_bytes": memory.total_training_state_bytes,
        }

    if kv is not None:
        data["kv_cache"] = {
            "bytes_per_token_per_batch_item": kv.bytes_per_token_per_batch_item,
            "total_bytes": kv.total_bytes,
        }

    return data


def _validate_config(spec) -> None:
    """Validate analytical parameters against the spec's expected and target values."""
    params = calculate_parameter_count(spec.architecture)

    if params.total != spec.expected_parameter_count:
        _fail(
            f"{spec.model_name}: analytical parameter count {params.total} "
            f"differs from expected_parameter_count {spec.expected_parameter_count}"
        )

    diff_pct = abs(params.total - spec.target_parameter_count) / spec.target_parameter_count
    if diff_pct >= ONE_PERCENT:
        _fail(
            f"{spec.model_name}: analytical parameter count {params.total} "
            f"is {diff_pct * 100:.4f}% from target {spec.target_parameter_count}, "
            "exceeding the 1% threshold"
        )

    return params


def main() -> None:
    parser = argparse.ArgumentParser(description="Bharat model parameter calculator")
    parser.add_argument("config", nargs="?", help="Path to model YAML config")
    parser.add_argument("--all", action="store_true", help="Calculate for all production configs")
    parser.add_argument("--json", action="store_true", help="Output JSON")
    parser.add_argument(
        "--weight-dtype", default=None, help="Weight dtype (fp32, bf16, fp16, int8, int4)"
    )
    parser.add_argument("--gradient-dtype", default=None, help="Gradient dtype")
    parser.add_argument("--optimizer", default=None, help="Optimizer (none, adamw_fp32)")
    parser.add_argument(
        "--fp32-master-weights", action="store_true", help="Use fp32 master weights"
    )
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size for KV cache")
    parser.add_argument(
        "--sequence-length", type=int, default=None, help="Sequence length for KV cache"
    )
    parser.add_argument("--kv-dtype", default=None, help="KV cache dtype")

    args = parser.parse_args()

    if not args.config and not args.all:
        parser.print_help()
        sys.exit(1)

    if args.config and args.all:
        parser.print_help()
        _fail("specify either a config file or --all, not both")

    if (
        args.gradient_dtype is not None or args.optimizer is not None or args.fp32_master_weights
    ) and args.weight_dtype is None:
        _fail("--gradient-dtype, --optimizer, and --fp32-master-weights require --weight-dtype")

    if args.optimizer is not None and args.optimizer not in ("none", "adamw_fp32"):
        _fail(f"unsupported optimizer '{args.optimizer}'; supported: none, adamw_fp32")

    if args.optimizer == "none":
        args.optimizer = None

    if (args.batch_size is not None) != (args.sequence_length is not None):
        _fail("--batch-size and --sequence-length must be supplied together")

    if args.kv_dtype is not None and (args.batch_size is None or args.sequence_length is None):
        _fail("--kv-dtype requires --batch-size and --sequence-length")

    if args.config:
        config_path = Path(args.config)
        if not config_path.exists():
            _fail(f"file not found: {config_path}")
        configs_to_run = [config_path]
    else:
        configs_to_run = [CONFIGS_DIR / name for name in PRODUCTION_CONFIGS]

    results = []
    for config_path in configs_to_run:
        try:
            spec = load_model_spec(config_path)
        except Exception as e:
            _fail(str(e))

        params = _validate_config(spec)

        memory: StaticMemoryReport | None = None
        if args.weight_dtype is not None:
            try:
                memory = calculate_static_memory(
                    parameter_count=params.total,
                    weight_dtype=args.weight_dtype,
                    gradient_dtype=args.gradient_dtype,
                    optimizer=args.optimizer,
                    use_fp32_master_weights=args.fp32_master_weights,
                )
            except (TypeError, ValueError) as e:
                _fail(str(e))

        kv: KVCacheMemoryReport | None = None
        if args.batch_size is not None and args.sequence_length is not None:
            kv_dtype = args.kv_dtype or args.weight_dtype or "bf16"
            try:
                kv = calculate_kv_cache_memory(
                    config=spec.architecture,
                    batch_size=args.batch_size,
                    sequence_length=args.sequence_length,
                    dtype=kv_dtype,
                )
            except (TypeError, ValueError) as e:
                _fail(str(e))

        if args.json:
            results.append(json_report(spec, params, memory, kv))
        else:
            human_report(spec, params, memory, kv)

    if args.json:
        if len(results) == 1:
            print(json.dumps(results[0], indent=2))
        else:
            print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
