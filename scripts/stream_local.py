#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from bharat.serving.streaming import (
    FunctionSpec,
    LocalStreamer,
    StreamRequest,
    stream_events_to_json,
    stream_events_to_jsonl,
)


def _parse_functions(files: list[str]) -> list[FunctionSpec]:
    specs: list[FunctionSpec] = []
    for path_str in files:
        path = Path(path_str)
        if not path.exists():
            print(f"error: Function spec file not found: {path}", file=sys.stderr)
            sys.exit(1)
        try:
            data = json.loads(path.read_text())
        except json.JSONDecodeError as e:
            print(f"error: Invalid JSON in {path}: {e}", file=sys.stderr)
            sys.exit(1)
        if isinstance(data, list):
            for item in data:
                specs.append(FunctionSpec.from_dict(item))
        elif isinstance(data, dict):
            specs.append(FunctionSpec.from_dict(data))
        else:
            print(f"error: Function spec must be a JSON object or array: {path}", file=sys.stderr)
            sys.exit(1)
    return specs


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate local deterministic streaming event output"
    )
    parser.add_argument("--prompt", default="Hello", help="Input prompt")
    parser.add_argument("--max-tokens", type=int, default=256, help="Maximum tokens")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature")
    parser.add_argument(
        "--function",
        "-f",
        action="append",
        default=[],
        dest="function_files",
        help="Path to a JSON function spec file (may be repeated)",
    )
    parser.add_argument(
        "--format",
        choices=["json", "jsonl"],
        default="jsonl",
        help="Output format (default: jsonl)",
    )
    parser.add_argument("--output", default="", help="Output file path")
    parser.add_argument("--json", action="store_true", help="Output machine-readable JSON result")

    args = parser.parse_args()

    functions = _parse_functions(args.function_files)

    request = StreamRequest(
        prompt=args.prompt,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        functions=tuple(functions),
        stream=True,
    )

    streamer = LocalStreamer(request)
    events = streamer.generate()

    if args.format == "json":
        output = stream_events_to_json(events)
    else:
        output = stream_events_to_jsonl(events)

    if args.output:
        output_path = Path(args.output)
        output_path.write_text(output, encoding="utf-8")
        if args.json:
            result = {
                "status": "success",
                "event_count": len(events),
                "output_path": str(output_path),
                "format": args.format,
            }
            print(json.dumps(result, sort_keys=True))
        else:
            print(f"Stream output written to {output_path}")
            print(f"  Events: {len(events)}")
            print(f"  Format: {args.format}")
    else:
        print(output, end="")

    if not args.output and args.json:
        result = {
            "status": "success",
            "event_count": len(events),
            "format": args.format,
        }
        print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
