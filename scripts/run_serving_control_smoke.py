#!/usr/bin/env python3
"""IndicLLM-Bharat-V1 — Serving Engine Control & Smoke Verification CLI.

Runs local end-to-end streaming serving verification with token authentication,
sliding-window rate limiting, and Prometheus metrics recording.

Usage:
  python scripts/run_serving_control_smoke.py --prompt "नमस्ते भारत" --json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from bharat.serving.auth import AuthConfig
from bharat.serving.controller import ServingController
from bharat.serving.metrics import ServingMetrics
from bharat.serving.rate_limit import RateLimitConfig
from bharat.serving.streaming import (
    StreamRequest,
    stream_events_to_json,
    stream_events_to_jsonl,
)

_URL_RE = re.compile(r"^(https?|ftp|s3|gs)://", re.IGNORECASE)


def _is_remote_url(path: str) -> bool:
    return bool(_URL_RE.match(path))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run a local serving control smoke test with auth, rate limiting, and metrics"
    )
    parser.add_argument("--prompt", default="Hello", help="Input prompt")
    parser.add_argument("--api-key", default="test-key-001", help="API key for auth")
    parser.add_argument("--max-tokens", type=int, default=256, help="Maximum tokens")
    parser.add_argument(
        "--output",
        default="",
        help="Output file path for streaming events",
    )
    parser.add_argument(
        "--format",
        choices=["json", "jsonl"],
        default="jsonl",
        help="Output format (default: jsonl)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output machine-readable JSON result",
    )

    args = parser.parse_args(argv)

    if args.output and _is_remote_url(args.output):
        print(f"error: Remote output path rejected: {args.output}", file=sys.stderr)
        return 1

    config = AuthConfig(valid_api_keys=("test-key-001", "test-key-002"))
    rate_config = RateLimitConfig(max_requests=10, window_seconds=60.0)
    metrics = ServingMetrics()

    controller = ServingController(
        auth_config=config,
        rate_limit_config=rate_config,
        metrics=metrics,
    )

    request = StreamRequest(
        prompt=args.prompt,
        max_tokens=args.max_tokens,
    )

    try:
        events = controller.handle_request(
            request=request,
            api_key=args.api_key,
        )
    except Exception as e:
        print(f"error: serving request failed: {e}", file=sys.stderr)
        return 1

    if args.format == "json":
        output = stream_events_to_json(events)
    else:
        output = stream_events_to_jsonl(events)

    if args.output:
        output_path = Path(args.output)
        output_path.write_text(output, encoding="utf-8")

    metrics_output = metrics.snapshot().to_json()

    if args.json:
        result = {
            "status": "success",
            "event_count": len(events),
            "format": args.format,
            "metrics": metrics.snapshot().to_dict(),
        }
        if args.output:
            result["output_path"] = str(args.output)
        print(json.dumps(result, sort_keys=True, indent=2))
    else:
        print("=" * 60)
        print("  🇮🇳 IndicLLM-Bharat — Serving Smoke Test")
        print("=" * 60)
        print(f"  Events ({len(events)}):")
        print(output)
        print("  Metrics:")
        print(metrics_output)
        print("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
