#!/usr/bin/env python3

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
CLI_SCRIPT = REPO_ROOT / "run_export_plan.py"
COMPAT_TEST = REPO_ROOT.parent / "tests" / "compatibility" / "test_q8_0_external_gguf.py"


def main() -> None:
    print("=" * 60)
    print("Q8_0 GGUF Independent Compatibility Validation")
    print("=" * 60)

    try:
        import gguf
    except ImportError:
        print("FAIL: gguf package not installed.")
        print("Install: pip install gguf==0.19.0")
        sys.exit(1)

    print(f"\ngguf version: {gguf.__name__} (from {gguf.__file__})")
    print(f"gguf supported versions: {gguf.READER_SUPPORTED_VERSIONS}")

    print("\n--- Step 1: CLI Export ---")
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
        sys.exit(1)

    print("\n--- Results ---")
    print("All Q8_0 compatibility checks passed.")
    print("  Independent implementation: gguf (official GGML Python package)")
    print("  Structural validation: OK")
    print("  Numeric validation: OK")
    print("  Byte-level validation: OK")
    print("  F32 control: OK")
    print("  Corruption detection: OK")
    print("=" * 60)


if __name__ == "__main__":
    main()
