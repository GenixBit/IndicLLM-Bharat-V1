"""CLI tool to launch the Interactive Sovereign Terminal Chat for IndicLLM-Bharat."""

from __future__ import annotations

import argparse
import sys

from bharat.inference.chat import (
    InteractiveChatSession,
    run_interactive_terminal,
)


def parse_args(args: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Interactive Sovereign Terminal Chat Client for IndicLLM-Bharat",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
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
        "--prompt",
        type=str,
        default=None,
        help="Single prompt to execute and exit",
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

    session = InteractiveChatSession(
        tier=parsed.tier,
        checkpoint_path=parsed.checkpoint,
        device=parsed.device,
    )

    if parsed.prompt:
        session.send_message(parsed.prompt, stream=True)
        return 0

    run_interactive_terminal(session)
    return 0


if __name__ == "__main__":
    sys.exit(main())
