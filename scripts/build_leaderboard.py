#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from bharat.eval.leaderboard import load_leaderboard

_URL_RE = re.compile(r"^(https?|ftp|s3|gs)://", re.IGNORECASE)


def _is_remote_url(path: str) -> bool:
    return bool(_URL_RE.match(path))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a BharatBench checkpoint leaderboard from local evaluation report files"
    )
    parser.add_argument(
        "--reports-dir",
        required=True,
        help="Directory containing BharatBench report JSON files",
    )
    parser.add_argument(
        "--output",
        default="",
        help="Output file path (prints to stdout if omitted)",
    )
    parser.add_argument(
        "--benchmark-id",
        default="",
        help="Filter leaderboard to a specific benchmark",
    )
    parser.add_argument(
        "--category",
        default="",
        help="Filter leaderboard to a specific category",
    )
    parser.add_argument(
        "--format",
        choices=["json", "markdown"],
        default="json",
        help="Output format (default: json)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output JSON machine-readable result",
    )

    args = parser.parse_args()

    reports_dir = Path(args.reports_dir)

    if _is_remote_url(str(reports_dir)):
        print(f"error: Remote reports-dir path rejected: {reports_dir}", file=sys.stderr)
        sys.exit(1)

    if not reports_dir.exists():
        print(f"error: Reports directory not found: {reports_dir}", file=sys.stderr)
        sys.exit(1)
    if not reports_dir.is_dir():
        print(f"error: Not a directory: {reports_dir}", file=sys.stderr)
        sys.exit(1)

    benchmark_id: str | None = args.benchmark_id or None
    category: str | None = args.category or None

    try:
        leaderboard = load_leaderboard(reports_dir)
    except (ValueError, json.JSONDecodeError) as e:
        print(f"error: {e}", file=sys.stderr)
        sys.exit(1)

    if args.format == "markdown":
        output = leaderboard.to_markdown(benchmark_id=benchmark_id, category=category)
    else:
        output = leaderboard.to_json()

    if args.output:
        output_path = Path(args.output)
        output_path.write_text(output, encoding="utf-8")
        if args.json:
            result = {
                "status": "success",
                "entry_count": leaderboard.entry_count,
                "output_path": str(output_path),
                "format": args.format,
            }
            print(json.dumps(result))
        else:
            print(f"Leaderboard written to {output_path}")
            print(f"  Entries: {leaderboard.entry_count}")
            print(f"  Format:  {args.format}")
    else:
        print(output)


if __name__ == "__main__":
    main()
