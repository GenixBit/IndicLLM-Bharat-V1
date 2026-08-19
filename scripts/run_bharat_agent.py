"""CLI tool to run interactive sovereign agent sessions with IndicLLM-Bharat."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from bharat.agent.runtime import AgentStep, BharatAgent


def parse_args(args: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Sovereign Multi-Turn Agent with Python, Math & Knowledge Tools",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--query",
        type=str,
        default=None,
        help="Optional single query to execute. If omitted, launches interactive REPL.",
    )
    parser.add_argument(
        "--tier",
        type=str,
        default="1b",
        choices=["tiny", "small", "350m", "1b", "3b", "7b", "10b"],
        help="Model tier for the agent backbone",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/bharat_dpo/final.pt",
        help="Model checkpoint path",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "mps", "cuda"],
        help="Compute device",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=5,
        help="Maximum agent reasoning & tool-call steps per turn",
    )
    return parser.parse_args(args)


def on_agent_step(step: AgentStep) -> None:
    print(f"\n[Step {step.iteration}] 🧠 Thought / Model Output:")
    print(f"  {step.thought}")
    if step.tool_calls:
        print("\n🔧 Invoking Sovereign Tools:")
        for call in step.tool_calls:
            print(f"  • Tool: {call.get('name')}")
            print(f"    Args: {call.get('arguments')}")
    if step.tool_results:
        print("\n📥 Tool Results Received:")
        for res in step.tool_results:
            status = "✅ Success" if res.get("success") else "❌ Error"
            print(f"  • [{status}] Output: {res.get('output') or res.get('error')}")


def main(args: list[str] | None = None) -> int:
    parsed = parse_args(args)

    device = parsed.device
    if device == "auto":
        import torch

        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    print("\n" + "=" * 65)
    print("🤖 IndicLLM-Bharat Autonomous Multi-Turn Sovereign Agent")
    print(f"  • Model Tier:       Bharat-{parsed.tier.upper()}")
    print("  • Active Tools:     Python Interpreter, Math, Knowledge, 22-Lang Indic")
    print(f"  • Compute Device:   {device}")
    print(f"  • Max Iterations:   {parsed.max_iterations}")
    print("=" * 65 + "\n")

    agent = BharatAgent(
        tier=parsed.tier,
        checkpoint_path=parsed.checkpoint if Path(parsed.checkpoint).is_file() else None,
        device=device,
        max_iterations=parsed.max_iterations,
    )

    if parsed.query:
        print(f"User Query: {parsed.query}\n")
        response = agent.run(parsed.query, step_callback=on_agent_step)
        print("\n" + "=" * 65)
        print("🎯 Final Agent Answer:")
        print(response.final_answer)
        print("=" * 65 + "\n")
        return 0

    print("Type your message (or 'exit' / 'quit' to stop):\n")
    while True:
        try:
            user_input = input("User > ").strip()
            if not user_input or user_input.lower() in ("exit", "quit", "q"):
                break

            response = agent.run(user_input, step_callback=on_agent_step)
            print(f"\nAssistant > {response.final_answer}\n")
        except (KeyboardInterrupt, EOFError):
            break

    print("\nSession ended. Bharat Agent closed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
