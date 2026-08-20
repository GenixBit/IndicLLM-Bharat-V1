"""CLI tool to run the Autonomous Sovereign ReAct Tool Agent for IndicLLM-Bharat."""

from __future__ import annotations

import argparse
import json
import sys

from bharat.tools.executor import SovereignToolAgent


def parse_args(args: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Autonomous Sovereign ReAct Tool Agent CLI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--task",
        type=str,
        required=True,
        help="Task for the ReAct agent to solve with tools",
    )
    parser.add_argument(
        "--tier",
        type=str,
        default="1b",
        choices=["tiny", "small", "350m", "1b", "3b", "7b", "10b"],
        help="Model tier",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=3,
        help="Maximum ReAct reasoning steps",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to model checkpoint (.pt)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "mps", "cuda"],
        help="Compute device",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output raw JSON format",
    )
    return parser.parse_args(args)


def main(args: list[str] | None = None) -> int:
    parsed = parse_args(args)

    agent = SovereignToolAgent(
        tier=parsed.tier,
        checkpoint_path=parsed.checkpoint,
        device=parsed.device,
    )

    result = agent.run(parsed.task, max_steps=parsed.max_steps)

    if parsed.json:
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0

    print("\n" + "=" * 65)
    print("🛠️ IndicLLM-Bharat Sovereign ReAct Tool Agent")
    print(f"  • Task:        {result['task']}")
    print(f"  • Steps Taken: {result['steps_taken']}")
    print("=" * 65 + "\n")

    for i, step in enumerate(result["steps"], 1):
        print(f"--- Step {i} ---")
        if step["tool_call"]:
            print(
                f"🔧 Tool Call:   {step['tool_call']['name']}({step['tool_call'].get('arguments', {})})"
            )
            print(f"👁️ Observation: {step['observation']}")
        print()

    print("🎯 Final Answer:")
    print(result["final_answer"])
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
