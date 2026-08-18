#!/usr/bin/env python3
"""IndicLLM-Bharat-V1 — Q8_0 GGUF Compatibility Verification CLI.

Validates independent structural, numerical, and byte-level compatibility
of exported Q8_0 GGUF models against the official GGML Python specification.

Usage:
  python scripts/validate_q8_0_compatibility.py
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
COMPAT_TEST = REPO_ROOT / "tests" / "compatibility" / "test_q8_0_external_gguf.py"


def main(_argv: list[str] | None = None) -> int:
    print("=" * 60)
    print("  🇮🇳 IndicLLM-Bharat — Q8_0 GGUF Compatibility Verification")
    print("=" * 60)

    try:
        import gguf  # noqa: F401
    except ImportError:
        print("FAIL: gguf package not installed.")
        print("Install: pip install gguf==0.19.0")
        return 1

    if not COMPAT_TEST.is_file():
        print(f"FAIL: Compatibility test file not found at {COMPAT_TEST}")
        return 1

    print("Running independent GGML Q8_0 compatibility test harness...")
    result = subprocess.run(
        [sys.executable, "-m", "pytest", str(COMPAT_TEST), "-v", "--tb=short", "-x"],
        capture_output=True,
        text=True,
    )
    print(result.stdout)
    if result.stderr:
        print(result.stderr[:500])
    if result.returncode != 0:
        print("\nFAIL: Compatibility tests failed.")
        return 1

    print("=" * 60)
    print("  ✅ All Q8_0 compatibility checks passed successfully!")
    print("  • Independent GGML format validation: OK")
    print("  • Numeric & dequantization bounds   : OK")
    print("  • Byte-level tensor alignment       : OK")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
