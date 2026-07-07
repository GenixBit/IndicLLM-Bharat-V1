#!/usr/bin/env python3
"""
IndicLLM-Bharat-V1 — GGUF Export + Ollama Integration

Converts a trained IndicLLM checkpoint to GGUF format for use
with llama.cpp, Ollama, LM Studio, and other local inference tools.

Pipeline:
  1. Convert our GPT checkpoint → HuggingFace format (safetensors)
  2. Use llama.cpp's convert script to create GGUF
  3. Quantize (Q4_K_M, Q5_K_M, Q8_0, etc.)
  4. Generate Ollama Modelfile
  5. Register with Ollama

Usage:
  # Full pipeline: checkpoint → GGUF → Ollama
  python inference/export_ollama.py \
    --checkpoint checkpoints/gpt2-124m/final.pt \
    --name indicllm-124m \
    --quant q4_k_m

  # Just export to HF format (for push_to_hub.py)
  python inference/export_ollama.py \
    --checkpoint checkpoints/gpt2-10m/ckpt.pt \
    --name indicllm-10m \
    --hf-only

  # Skip quantization (full precision GGUF)
  python inference/export_ollama.py \
    --checkpoint checkpoints/gpt2-124m/final.pt \
    --name indicllm-124m \
    --no-quant

Requirements:
  pip install transformers safetensors
  # For GGUF conversion:
  git clone https://github.com/ggerganov/llama.cpp && cd llama.cpp && make
  # For Ollama:
  curl -fsSL https://ollama.com/install.sh | sh
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


# ── Step 1: Convert to HuggingFace Format ────────────────────


def convert_to_hf(ckpt_path: Path, output_dir: Path) -> dict:
    """Convert our GPT checkpoint to HuggingFace GPT2LMHeadModel format."""
    from transformers import GPT2Config, GPT2LMHeadModel, GPT2TokenizerFast

    print("\n  [1/4] Converting checkpoint to HuggingFace format...")
    print(f"        Input : {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ckpt.get("config", {})
    model_cfg = cfg.get("model", {})

    if not model_cfg:
        raise ValueError("Checkpoint missing model config section")

    # Create HF config
    hf_config = GPT2Config(
        vocab_size=model_cfg.get("vocab_size", 50257),
        n_layer=model_cfg["n_layer"],
        n_head=model_cfg["n_head"],
        n_embd=model_cfg["n_embd"],
        n_positions=model_cfg.get("block_size", 1024),
        n_inner=4 * model_cfg["n_embd"],
        activation_function="gelu_new",
        resid_pdrop=0.0,
        embd_pdrop=0.0,
        attn_pdrop=0.0,
        use_cache=True,
    )

    hf_model = GPT2LMHeadModel(hf_config)

    # Map our weights to HF format
    state = ckpt["model"]
    # Strip _orig_mod prefix if present (from torch.compile)
    state = {k.replace("_orig_mod.", ""): v for k, v in state.items()}

    # Our model uses nn.Linear [out_features, in_features]
    # HF GPT2 uses Conv1D [in_features, out_features] — need transpose
    # Keys that need transposing: attn.c_attn, attn.c_proj, mlp.c_fc, mlp.c_proj
    transpose_keys = {"c_attn.weight", "c_proj.weight", "c_fc.weight"}

    mapped = {}
    for k, v in state.items():
        hf_key = (
            k if k.startswith("transformer.") or k.startswith("lm_head.") else f"transformer.{k}"
        )

        # Transpose 2D weight matrices for Conv1D compatibility
        if v.ndim == 2 and any(tk in k for tk in transpose_keys):
            v = v.t()

        mapped[hf_key] = v

    # Load with strict=False (lm_head may be tied to wte)
    missing, _unexpected = hf_model.load_state_dict(mapped, strict=False)
    if missing:
        print(f"        Warning: {len(missing)} missing keys (may be OK for tied weights)")

    # Save
    output_dir.mkdir(parents=True, exist_ok=True)
    hf_model.save_pretrained(output_dir, safe_serialization=True)

    # Save tokenizer
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    tokenizer.save_pretrained(output_dir)

    params = sum(p.numel() for p in hf_model.parameters()) / 1e6
    iter_num = ckpt.get("iter_num", "?")

    print(f"        Output: {output_dir}")
    print(f"        Model : {params:.1f}M params (iter {iter_num})")

    return {
        "params_m": params,
        "iter_num": iter_num,
        "model_cfg": model_cfg,
        "hf_dir": output_dir,
    }


# ── Step 2: Convert HF → GGUF ───────────────────────────────


def convert_to_gguf(hf_dir: Path, gguf_path: Path, llama_cpp_dir: Path | None = None) -> bool:
    """Convert HF model to GGUF using llama.cpp's convert script."""
    print("\n  [2/4] Converting to GGUF format...")

    # Find convert script
    convert_script = None
    search_paths = [
        llama_cpp_dir / "convert_hf_to_gguf.py" if llama_cpp_dir else None,
        Path.home() / "llama.cpp" / "convert_hf_to_gguf.py",
        Path("/usr/local/bin/convert_hf_to_gguf.py"),
        ROOT / "tools" / "llama.cpp" / "convert_hf_to_gguf.py",
    ]
    for p in search_paths:
        if p and p.exists():
            convert_script = p
            break

    if not convert_script:
        print("        ⚠ llama.cpp convert script not found.")
        print("        Install: git clone https://github.com/ggerganov/llama.cpp")
        print("        Then re-run with: --llama-cpp ~/llama.cpp")
        print("        Skipping GGUF conversion (HF format still available)")
        return False

    cmd = [
        sys.executable,
        str(convert_script),
        str(hf_dir),
        "--outfile",
        str(gguf_path),
        "--outtype",
        "f16",
    ]
    print(f"        Running: {' '.join(cmd[-4:])}")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"        ✗ Convert failed: {result.stderr[:200]}")
        return False

    size_mb = gguf_path.stat().st_size / 1e6
    print(f"        ✓ {gguf_path.name} ({size_mb:.0f} MB)")
    return True


# ── Step 3: Quantize GGUF ────────────────────────────────────


def quantize_gguf(
    gguf_input: Path, gguf_output: Path, quant: str, llama_cpp_dir: Path | None = None
) -> bool:
    """Quantize GGUF using llama.cpp's llama-quantize."""
    print(f"\n  [3/4] Quantizing to {quant.upper()}...")

    quantize_bin = None
    search_paths = [
        llama_cpp_dir / "build" / "bin" / "llama-quantize" if llama_cpp_dir else None,
        llama_cpp_dir / "llama-quantize" if llama_cpp_dir else None,
        Path.home() / "llama.cpp" / "build" / "bin" / "llama-quantize",
        Path("/usr/local/bin/llama-quantize"),
    ]
    for p in search_paths:
        if p and p.exists():
            quantize_bin = p
            break

    if not quantize_bin:
        # Try finding it via which
        result = subprocess.run(["which", "llama-quantize"], capture_output=True, text=True)
        if result.returncode == 0:
            quantize_bin = Path(result.stdout.strip())

    if not quantize_bin:
        print("        ⚠ llama-quantize not found, skipping quantization")
        print("        Build: cd ~/llama.cpp && make llama-quantize")
        return False

    cmd = [str(quantize_bin), str(gguf_input), str(gguf_output), quant]
    print(f"        Running: llama-quantize → {quant}")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"        ✗ Quantize failed: {result.stderr[:200]}")
        return False

    size_mb = gguf_output.stat().st_size / 1e6
    ratio = gguf_output.stat().st_size / gguf_input.stat().st_size
    print(f"        ✓ {gguf_output.name} ({size_mb:.0f} MB, {ratio:.1%} of F16)")
    return True


# ── Step 4: Generate Ollama Modelfile ─────────────────────────


def generate_modelfile(model_path: Path, name: str, output_dir: Path, info: dict) -> Path:
    """Generate Ollama Modelfile for the exported model."""
    print("\n  [4/4] Generating Ollama Modelfile...")

    modelfile_path = output_dir / "Modelfile"
    model_cfg = info.get("model_cfg", {})

    modelfile_content = f"""# IndicLLM-Bharat — Ollama Modelfile
# Model: {name} ({info.get("params_m", "?")}M params)
# Trained: iter {info.get("iter_num", "?")}

FROM {model_path.resolve()}

# Sampling parameters
PARAMETER temperature 0.7
PARAMETER top_k 50
PARAMETER top_p 0.9
PARAMETER repeat_penalty 1.1
PARAMETER num_ctx {model_cfg.get("block_size", 1024)}

# Stop tokens
PARAMETER stop "<|endoftext|>"

# System prompt
SYSTEM \"\"\"You are IndicLLM-Bharat, a multilingual Indian language model.
You can understand and generate text in Hindi, Bengali, Tamil, Telugu,
Marathi, Gujarati, Kannada, Malayalam, and other Indic languages.
Respond helpfully and accurately.\"\"\"

# Chat template
TEMPLATE \"\"\"{{{{ if .System }}}}System: {{{{ .System }}}}
{{{{ end }}}}{{{{ range .Messages }}}}{{{{ if eq .Role "user" }}}}User: {{{{ .Content }}}}
{{{{ else if eq .Role "assistant" }}}}Assistant: {{{{ .Content }}}}
{{{{ end }}}}{{{{ end }}}}Assistant: \"\"\"
"""

    modelfile_path.write_text(modelfile_content)
    print(f"        ✓ {modelfile_path}")
    return modelfile_path


def register_ollama(modelfile: Path, name: str) -> bool:
    """Register model with Ollama."""
    result = subprocess.run(["which", "ollama"], capture_output=True, text=True)
    if result.returncode != 0:
        print("\n  Ollama not installed. Install: curl -fsSL https://ollama.com/install.sh | sh")
        print(f"  Then register manually: ollama create {name} -f {modelfile}")
        return False

    print(f"\n  Registering with Ollama as '{name}'...")
    result = subprocess.run(
        ["ollama", "create", name, "-f", str(modelfile)], capture_output=True, text=True
    )
    if result.returncode == 0:
        print(f"  ✓ Registered! Run: ollama run {name}")
        return True
    else:
        print(f"  ✗ Registration failed: {result.stderr[:200]}")
        print(f"  Manual: ollama create {name} -f {modelfile}")
        return False


# ── Main ─────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="IndicLLM → GGUF → Ollama export pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full pipeline
  python inference/export_ollama.py --checkpoint checkpoints/gpt2-10m/ckpt.pt --name indicllm-10m

  # Just HuggingFace format
  python inference/export_ollama.py --checkpoint checkpoints/gpt2-10m/ckpt.pt --hf-only

  # Custom quantization
  python inference/export_ollama.py --checkpoint checkpoints/gpt2-124m/final.pt --quant q5_k_m
""",
    )
    parser.add_argument(
        "--checkpoint", type=Path, required=True, help="Path to ckpt.pt or final.pt"
    )
    parser.add_argument("--name", default="indicllm", help="Model name for Ollama")
    parser.add_argument(
        "--quant", default="q4_k_m", help="Quantization level (q4_k_m, q5_k_m, q8_0, f16)"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: inference/ollama/<name>)",
    )
    parser.add_argument("--llama-cpp", type=Path, default=None, help="Path to llama.cpp directory")
    parser.add_argument("--hf-only", action="store_true", help="Only export to HuggingFace format")
    parser.add_argument("--no-quant", action="store_true", help="Skip quantization (F16 GGUF)")
    parser.add_argument("--no-ollama", action="store_true", help="Don't register with Ollama")
    args = parser.parse_args()

    out_dir = args.output_dir or ROOT / "inference" / "ollama" / args.name
    hf_dir = out_dir / "hf"

    print(f"\n{'=' * 60}")
    print("  IndicLLM-Bharat → GGUF → Ollama Export Pipeline")
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"  Name      : {args.name}")
    print(f"  Quant     : {args.quant}")
    print(f"  Output    : {out_dir}")
    print(f"{'=' * 60}")

    # Step 1: Convert to HF
    info = convert_to_hf(args.checkpoint, hf_dir)

    if args.hf_only:
        print(f"\n  ✅ HuggingFace export complete: {hf_dir}")
        print(
            f"  Push to hub: python scripts/push_to_hub.py --checkpoint {args.checkpoint} --repo GenixBit/IndicLLM-Bharat"
        )
        return

    # Step 2: GGUF
    gguf_f16 = out_dir / f"{args.name}-f16.gguf"
    has_gguf = convert_to_gguf(hf_dir, gguf_f16, args.llama_cpp)

    # Step 3: Quantize
    gguf_final = gguf_f16
    if has_gguf and not args.no_quant and args.quant != "f16":
        gguf_quant = out_dir / f"{args.name}-{args.quant}.gguf"
        if quantize_gguf(gguf_f16, gguf_quant, args.quant, args.llama_cpp):
            gguf_final = gguf_quant

    # Step 4: Ollama Modelfile
    model_for_ollama = gguf_final if has_gguf else hf_dir
    modelfile = generate_modelfile(model_for_ollama, args.name, out_dir, info)

    # Step 5: Register (optional)
    if not args.no_ollama and has_gguf:
        register_ollama(modelfile, args.name)

    print(f"\n{'=' * 60}")
    print("  ✅ Export complete!")
    print(f"  HF model  : {hf_dir}")
    if has_gguf:
        print(f"  GGUF      : {gguf_final}")
    print(f"  Modelfile : {modelfile}")
    print("\n  Quick start:")
    if has_gguf:
        print(f"    ollama create {args.name} -f {modelfile}")
        print(f"    ollama run {args.name}")
    else:
        print(f"    python inference/generate.py --checkpoint {args.checkpoint}")
        print(f"    python inference/api.py --checkpoint {args.checkpoint}")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
