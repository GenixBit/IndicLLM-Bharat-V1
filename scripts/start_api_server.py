"""CLI tool to launch the OpenAI-Compatible Sovereign REST API Server for IndicLLM-Bharat."""

from __future__ import annotations

import argparse
import sys

from bharat.serving.openai_server import ServerConfig, run_api_server


def parse_args(args: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Launch OpenAI-Compatible REST & Streaming API Server for IndicLLM-Bharat",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host address to bind",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to listen on",
    )
    parser.add_argument(
        "--tier",
        type=str,
        default="1b",
        choices=["tiny", "small", "350m", "1b", "3b", "7b", "10b"],
        help="Model architecture tier",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to model checkpoint (.pt)",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="Optional secret Bearer API key for authorization",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "mps", "cuda"],
        help="Compute device",
    )
    return parser.parse_args(args)


def main(args: list[str] | None = None) -> int:
    parsed = parse_args(args)

    config = ServerConfig(
        host=parsed.host,
        port=parsed.port,
        tier=parsed.tier,
        checkpoint_path=parsed.checkpoint,
        api_key=parsed.api_key,
        device=parsed.device,
    )

    server = run_api_server(config)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping API server...")
        server.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
