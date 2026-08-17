from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
import yaml

from bharat.models.config import BharatModelConfig
from bharat.serving.gguf_preflight import GGUFMetadataEntry, GGUFPreflightResult
from bharat.serving.gguf_tensor_writer import (
    write_gguf_f32_tensors,
    write_gguf_q8_0_tensors,
)
from bharat.serving.safetensors_writer import write_safetensors_checkpoint


def compute_sha256(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(65536):
            sha.update(chunk)
    return sha.hexdigest()


def _load_state_dict(checkpoint_path: Path) -> dict[str, torch.Tensor]:
    target_file = checkpoint_path
    if checkpoint_path.is_dir():
        for candidate in ["model.pt", "final.pt", "best.pt", "ckpt.pt"]:
            cand_path = checkpoint_path / candidate
            if cand_path.is_file():
                target_file = cand_path
                break
    loaded = torch.load(target_file, map_location="cpu", weights_only=False)
    if isinstance(loaded, dict) and "model" in loaded:
        state = loaded["model"]
    elif isinstance(loaded, dict):
        state = loaded
    else:
        raise ValueError(f"Unsupported checkpoint format in {target_file}")
    return {str(k): v for k, v in state.items() if isinstance(v, torch.Tensor)}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Package trained IndicLLM-Bharat checkpoints into verified production release bundles"
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to trained checkpoint file (.pt) or checkpoint directory",
    )
    parser.add_argument(
        "--model-config",
        default=None,
        help="Path to YAML model configuration (e.g. configs/models/bharat-350m.yaml)",
    )
    parser.add_argument(
        "--tokenizer",
        default=None,
        help="Path to tokenizer.json file",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where release bundle will be written",
    )
    parser.add_argument(
        "--model-name",
        default="Bharat-350M",
        help="Name of the model release (e.g. Bharat-350M, Bharat-1B)",
    )
    parser.add_argument(
        "--version",
        default="1.0.0",
        help="Semantic version string for this release bundle",
    )
    parser.add_argument(
        "--include-gguf",
        action="store_true",
        help="Export GGUF artifact in addition to Safetensors",
    )
    parser.add_argument(
        "--gguf-type",
        choices=["Q8_0", "F32"],
        default="Q8_0",
        help="GGUF tensor quantization type (default: Q8_0)",
    )
    parser.add_argument(
        "--model-card",
        default="docs/MODEL_CARD.md",
        help="Path to MODEL_CARD.md to bundle with release",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print final release manifest JSON to stdout",
    )
    return parser


def build_release_bundle(
    checkpoint_path: str | Path,
    output_dir: str | Path,
    model_config_path: str | Path | None = None,
    tokenizer_path: str | Path | None = None,
    model_name: str = "Bharat-350M",
    version: str = "1.0.0",
    include_gguf: bool = False,
    gguf_type: str = "Q8_0",
    model_card_path: str | Path | None = None,
) -> dict[str, Any]:
    cp_path = Path(checkpoint_path).resolve()
    out_dir = Path(output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Resolve model configuration
    model_config_dict: dict[str, Any] = {}
    if model_config_path and Path(model_config_path).is_file():
        with open(model_config_path, encoding="utf-8") as f:
            model_config_dict = yaml.safe_load(f)
    else:
        # Fallback to default 350M config
        model_config_dict = BharatModelConfig(
            vocab_size=64000,
            hidden_size=1024,
            intermediate_size=2816,
            num_hidden_layers=16,
            num_attention_heads=16,
            num_key_value_heads=4,
            max_position_embeddings=2048,
        ).to_dict()

    config_json_path = out_dir / "config.json"
    with config_json_path.open("w", encoding="utf-8") as f:
        json.dump(model_config_dict, f, indent=2)

    # 2. Export Safetensors
    safetensors_path = out_dir / "model.safetensors"
    write_safetensors_checkpoint(
        checkpoint_path=cp_path,
        output_path=safetensors_path,
        model_name=model_name,
    )

    # 3. Export GGUF (optional)
    if include_gguf:
        normalized_gguf_type = gguf_type.lower()
        gguf_path = out_dir / f"model-{normalized_gguf_type}.gguf"
        state_dict = _load_state_dict(cp_path)
        preflight = GGUFPreflightResult(
            schema_version=1,
            architecture="bharat",
            alignment=32,
            tensor_count=len(state_dict),
            output_file=gguf_path.name,
            metadata=(
                GGUFMetadataEntry(key="general.architecture", value_type="string", value="bharat"),
                GGUFMetadataEntry(key="general.name", value_type="string", value=model_name),
                GGUFMetadataEntry(key="general.version", value_type="string", value=version),
            ),
            gguf_tensor_type=normalized_gguf_type,
        )
        if normalized_gguf_type == "q8_0":
            write_gguf_q8_0_tensors(preflight, state_dict, gguf_path)
        else:
            write_gguf_f32_tensors(preflight, state_dict, gguf_path)

    # 4. Copy Tokenizer files
    if tokenizer_path and Path(tokenizer_path).is_file():
        tok_src = Path(tokenizer_path)
        shutil.copy2(tok_src, out_dir / "tokenizer.json")
    else:
        tok_data = {
            "model_type": "bharat_bpe",
            "vocab_size": model_config_dict.get("vocab_size", 64000),
            "pad_token": "<|pad|>",
            "eos_token": "<|im_end|>",
            "bos_token": "<|im_start|>",
        }
        with (out_dir / "tokenizer_config.json").open("w", encoding="utf-8") as f:
            json.dump(tok_data, f, indent=2)

    # 5. Copy Model Card
    if model_card_path and Path(model_card_path).is_file():
        shutil.copy2(Path(model_card_path), out_dir / "MODEL_CARD.md")

    # 6. Build Manifest
    files_manifest: list[dict[str, Any]] = []
    for item in sorted(out_dir.iterdir()):
        if item.name == "release_manifest.json":
            continue
        if item.is_file():
            files_manifest.append(
                {
                    "filename": item.name,
                    "size_bytes": item.stat().st_size,
                    "sha256": compute_sha256(item),
                }
            )

    release_manifest = {
        "manifest_version": "1.0",
        "model_name": model_name,
        "version": version,
        "created_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "files": files_manifest,
        "total_files": len(files_manifest),
        "total_bytes": sum(f["size_bytes"] for f in files_manifest),
    }

    manifest_path = out_dir / "release_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(release_manifest, f, indent=2)

    return release_manifest


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        manifest = build_release_bundle(
            checkpoint_path=args.checkpoint,
            output_dir=args.output_dir,
            model_config_path=args.model_config,
            tokenizer_path=args.tokenizer,
            model_name=args.model_name,
            version=args.version,
            include_gguf=args.include_gguf,
            gguf_type=args.gguf_type,
            model_card_path=args.model_card,
        )
    except Exception as e:
        print(f"error building release bundle: {e}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(manifest, indent=2))
    else:
        print("=" * 64)
        print("📦 IndicLLM-Bharat Production Release Bundle Created")
        print(f"Model:       {manifest['model_name']} (v{manifest['version']})")
        print(f"Output Dir:  {args.output_dir}")
        print(f"Total Files: {manifest['total_files']} ({manifest['total_bytes']:,} bytes)")
        print("-" * 64)
        for f in manifest["files"]:
            print(f"  • {f['filename']:<30} {f['size_bytes']:>10,} bytes  [{f['sha256'][:10]}...]")
        print("=" * 64)

    return 0


if __name__ == "__main__":
    sys.exit(main())
