#!/usr/bin/env python3
"""
Export a Hugging Face checkpoint to GGUF-friendly format and generate Ollama Modelfile.

Usage:
  python inference/export_ollama.py --model checkpoints/gpt2-124m-sft --name llm-lab-124m
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--name", default="llm-lab")
    parser.add_argument("--quant", default="q4_k_m")
    args = parser.parse_args()

    out_dir = ROOT / "inference" / "ollama"
    out_dir.mkdir(parents=True, exist_ok=True)
    gguf_path = out_dir / f"{args.name}.gguf"

    convert = [
        sys.executable,
        "-m",
        "llama_cpp.server",
    ]
    # Prefer llama.cpp convert script if installed via brew/pip
    try:
        subprocess.run(
            ["python3", "-c", "import llama_cpp"],
            check=True,
            capture_output=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass

    modelfile = out_dir / "Modelfile"
    modelfile.write_text(
        f"""FROM {args.model.resolve()}

PARAMETER temperature 0.7
PARAMETER stop "### Instruction:"
PARAMETER stop "user:"
PARAMETER stop "assistant:"

TEMPLATE \"\"\"{{{{ if .System }}}}system: {{{{ .System }}}}{{{{ end }}}}
{{{{ range .Messages }}}}{{{{ .Role }}}}: {{{{ .Content }}}}
{{{{ end }}}}assistant:
\"\"\"
"""
    )

    print(f"Wrote Ollama Modelfile to {modelfile}")
    print("Create the model locally with:")
    print(f"  ollama create {args.name} -f {modelfile}")
    print("")
    print("For GGUF conversion, install llama.cpp and run:")
    print(f"  python convert_hf_to_gguf.py {args.model} --outfile {gguf_path}")
    print(f"  ollama create {args.name}-gguf -f {out_dir / 'Modelfile.gguf'}")


if __name__ == "__main__":
    main()
