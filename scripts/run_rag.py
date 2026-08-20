"""CLI tool to execute grounded Sovereign RAG queries with citations using IndicLLM-Bharat."""

from __future__ import annotations

import argparse
import json
import sys

from bharat.rag.engine import SovereignRAGPipeline


def parse_args(args: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sovereign RAG Vector Retrieval & Grounded Generation CLI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--query",
        type=str,
        required=True,
        help="Query to search knowledge base and answer",
    )
    parser.add_argument(
        "--tier",
        type=str,
        default="1b",
        choices=["tiny", "small", "350m", "1b", "3b", "7b", "10b"],
        help="Model tier",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=2,
        help="Number of document chunks to retrieve",
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

    pipeline = SovereignRAGPipeline(
        tier=parsed.tier,
        checkpoint_path=parsed.checkpoint,
        device=parsed.device,
    )

    result = pipeline.query(parsed.query, top_k=parsed.top_k)

    if parsed.json:
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0

    print("\n" + "=" * 65)
    print("📚 Sovereign RAG Grounded Retrieval & Generation")
    print(f"  • Query:     {result['query']}")
    print(f"  • Retrieved: {result['documents_retrieved']} documents")
    print("=" * 65 + "\n")

    print("📄 Grounded Sources / Citations:")
    for cit in result["citations"]:
        print(f"  [{cit['citation_id']}] {cit['title']} (Score: {cit['score']})")
        print(f"      {cit['snippet']}")

    print("\n🤖 Grounded Response:")
    print(result["response"])
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
