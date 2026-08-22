"""CLI script to start the IndicLLM-Bharat Universal Hybrid AI Operating Gateway."""

from __future__ import annotations

import argparse
import sys

import uvicorn

from bharat.gateway.server import create_app


def parse_args(args: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Start IndicLLM-Bharat Universal Hybrid AI Operating Gateway",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--tier", type=str, default="1b", help="Model tier")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to weights (.pt)")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host interface")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind")
    parser.add_argument("--workers", type=int, default=1, help="Worker count")
    return parser.parse_args(args)


def main(args: list[str] | None = None) -> int:
    parsed = parse_args(args)
    app = create_app(tier=parsed.tier, checkpoint_path=parsed.checkpoint)

    print("\n" + "=" * 65)
    print("🚀 IndicLLM-Bharat Universal Hybrid AI Operating Gateway")
    print(f"  • Host: http://{parsed.host}:{parsed.port}")
    print(f"  • Model Tier: {parsed.tier.upper()}")
    print("=" * 65 + "\n")

    uvicorn.run(app, host=parsed.host, port=parsed.port, log_level="info")
    return 0


if __name__ == "__main__":
    sys.exit(main())
