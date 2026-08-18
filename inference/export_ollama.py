#!/usr/bin/env python3
"""IndicLLM-Bharat-V1 — GGUF Export & Ollama Integration CLI.

Exports trained BharatForCausalLM checkpoints to native GGUF format and generates
an optimized Ollama Modelfile for seamless local inference with Ollama, llama.cpp,
and LM Studio.

Usage:
  # Export checkpoint to Q8_0 GGUF + Ollama Modelfile
  python inference/export_ollama.py \
    --checkpoint checkpoints/bharat-350m/final.pt \
    --name bharat-350m \
    --output-dir dist/ollama/bharat-350m

  # Auto-register with local Ollama runtime
  python inference/export_ollama.py \
    --checkpoint checkpoints/bharat-350m/final.pt \
    --name bharat-350m \
    --register
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch

from bharat.models.config import BharatModelConfig
from bharat.serving.gguf_preflight import GGUFMetadataEntry, GGUFPreflightResult
from bharat.serving.gguf_tensor_writer import (
    write_gguf_f32_tensors,
    write_gguf_q8_0_tensors,
)

DEFAULT_SYSTEM_PROMPT = (
    "You are Bharat, a state-of-the-art multilingual Indian AI assistant. "
    "You understand and generate high-quality text in Hindi, Bengali, Tamil, "
    "Telugu, Marathi, Gujarati, Kannada, Malayalam, and other Indian languages."
)

MODELFILE_TEMPLATE = """FROM ./{gguf_filename}

# Model Parameters
PARAMETER temperature 0.8
PARAMETER top_p 0.95
PARAMETER top_k 50
PARAMETER num_ctx {context_length}

# Stop Sequences
PARAMETER stop "<|endoftext|>"
PARAMETER stop "<|instruction|>"
PARAMETER stop "<|response|>"

# System Prompt
SYSTEM \"\"\"{system_prompt}\"\"\"

# Indic Chat Template
TEMPLATE \"\"\"{{{{ if .System }}}}<|system|>
{{{{ .System }}}}
{{{{ end }}}}{{{{ if .Prompt }}}}<|instruction|>
{{{{ .Prompt }}}}
{{{{ end }}}}<|response|>
\"\"\"
"""


def generate_modelfile(
    gguf_filename: str,
    output_dir: Path,
    context_length: int = 4096,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
) -> Path:
    """Generate an optimized Ollama Modelfile."""
    modelfile_content = MODELFILE_TEMPLATE.format(
        gguf_filename=gguf_filename,
        context_length=context_length,
        system_prompt=system_prompt,
    )
    modelfile_path = output_dir / "Modelfile"
    modelfile_path.write_text(modelfile_content, encoding="utf-8")
    return modelfile_path


def export_ollama(
    checkpoint_path: str | Path,
    output_dir: str | Path,
    name: str = "bharat",
    quant: str = "q8_0",
    context_length: int = 4096,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    register: bool = False,
) -> dict[str, Any]:
    cp_path = Path(checkpoint_path).resolve()
    if not cp_path.is_file() and not cp_path.is_dir():
        raise FileNotFoundError(f"Checkpoint not found at: {cp_path}")

    out_dir = Path(output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 60}")
    print("  IndicLLM-Bharat — Native GGUF Export & Ollama Integration")
    print(f"  Checkpoint : {cp_path}")
    print(f"  Target Name: {name}")
    print(f"  Quant Type : {quant.upper()}")
    print(f"  Output Dir : {out_dir}")
    print(f"{'=' * 60}\n")

    ckpt_file = cp_path if cp_path.is_file() else cp_path / "final.pt"
    if not ckpt_file.exists():
        ckpt_file = cp_path / "ckpt.pt"

    ckpt_data = torch.load(ckpt_file, map_location="cpu", weights_only=False)
    raw_state_dict = ckpt_data.get("model", ckpt_data)
    state_dict = {
        k.replace("_orig_mod.", ""): v
        for k, v in raw_state_dict.items()
        if not k.endswith((".inv_freq", ".cos_cached", ".sin_cached"))
    }

    # Extract model config
    if "metadata" in ckpt_data and hasattr(ckpt_data["metadata"], "model_config"):
        cfg_obj = BharatModelConfig.from_dict(ckpt_data["metadata"].model_config)
        context_length = cfg_obj.max_position_embeddings
    elif "model_config" in ckpt_data:
        m_cfg = ckpt_data["model_config"]
        cfg_obj = BharatModelConfig.from_dict(m_cfg if isinstance(m_cfg, dict) else m_cfg.__dict__)
        context_length = cfg_obj.max_position_embeddings

    # Perform preflight and native GGUF export
    quant_lower = quant.lower()
    gguf_filename = f"{name}-{quant_lower}.gguf"
    gguf_out_path = out_dir / gguf_filename

    print(f"  [1/2] Writing native GGUF ({quant_lower.upper()})...")
    preflight = GGUFPreflightResult(
        schema_version=1,
        architecture="bharat",
        alignment=32,
        tensor_count=len(state_dict),
        output_file=gguf_filename,
        metadata=(
            GGUFMetadataEntry(key="general.architecture", value_type="string", value="bharat"),
            GGUFMetadataEntry(key="general.name", value_type="string", value=name),
            GGUFMetadataEntry(
                key="general.quantization_version", value_type="string", value=quant_lower
            ),
        ),
        gguf_tensor_type=quant_lower,
    )

    if quant_lower in ("q8_0", "q8"):
        write_gguf_q8_0_tensors(preflight, state_dict, gguf_out_path)
    elif quant_lower in ("f32", "fp32"):
        write_gguf_f32_tensors(preflight, state_dict, gguf_out_path)
    else:
        raise ValueError(f"Unsupported native quantization type: {quant}. Choose 'q8_0' or 'f32'.")

    print(f"        ✓ Saved GGUF ({gguf_out_path.stat().st_size:,} bytes) to {gguf_out_path.name}")

    # Generate Ollama Modelfile
    print("  [2/2] Generating Ollama Modelfile...")
    modelfile = generate_modelfile(
        gguf_filename=gguf_filename,
        output_dir=out_dir,
        context_length=context_length,
        system_prompt=system_prompt,
    )
    print(f"        ✓ Generated {modelfile}")

    registered = False
    if register:
        ollama_bin = shutil.which("ollama")
        if ollama_bin:
            print(f"\n  Registering with Ollama runtime as '{name}'...")
            res = subprocess.run(
                [ollama_bin, "create", name, "-f", str(modelfile)],
                capture_output=True,
                text=True,
                cwd=str(out_dir),
            )
            if res.returncode == 0:
                print(f"  ✅ Successfully registered with Ollama! Run with: ollama run {name}")
                registered = True
            else:
                print(f"  ⚠️ Ollama registration failed: {res.stderr[:200]}")
        else:
            print("\n  [INFO] 'ollama' binary not found on PATH. Run manually:")
            print(f"     cd {out_dir} && ollama create {name} -f Modelfile")

    return {
        "name": name,
        "gguf_path": str(gguf_out_path),
        "modelfile_path": str(modelfile),
        "registered": registered,
        "quant": quant_lower,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="IndicLLM-Bharat Native GGUF Export & Ollama Integration CLI"
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to checkpoint file (.pt)",
    )
    parser.add_argument(
        "--name",
        default="bharat",
        help="Target model name for Ollama",
    )
    parser.add_argument(
        "--quant",
        default="q8_0",
        choices=["q8_0", "f32"],
        help="Native GGUF quantization format (default: q8_0)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for GGUF and Modelfile (default: dist/ollama/<name>)",
    )
    parser.add_argument(
        "--system-prompt",
        default=DEFAULT_SYSTEM_PROMPT,
        help="Custom default system prompt for Modelfile",
    )
    parser.add_argument(
        "--register",
        action="store_true",
        help="Automatically register model with local Ollama runtime",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    out_dir = args.output_dir or Path("dist") / "ollama" / args.name

    try:
        export_ollama(
            checkpoint_path=args.checkpoint,
            output_dir=out_dir,
            name=args.name,
            quant=args.quant,
            system_prompt=args.system_prompt,
            register=args.register,
        )
    except Exception as e:
        print(f"error exporting to ollama: {e}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
