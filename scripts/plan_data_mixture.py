from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import yaml

from bharat.data.manifest import load_manifest
from bharat.data.mixture import MixtureConstraint, MixturePlan, MixturePlanner


def load_recipe(recipe_path: str | Path) -> MixtureConstraint:
    path = Path(recipe_path)
    if not path.is_file():
        raise FileNotFoundError(f"Recipe file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if not isinstance(data, dict):
        raise ValueError(f"Recipe file {path} must contain a dictionary mapping")

    lang_weights = data.get("language_weights", {})
    domain_weights = data.get("domain_weights", {})
    max_pct = float(data.get("max_pct_per_source", 0.5))
    min_thresh = int(data.get("min_record_threshold", 0))

    return MixtureConstraint(
        language_weights=lang_weights,
        domain_weights=domain_weights,
        max_pct_per_source=max_pct,
        min_record_threshold=min_thresh,
    )


def collect_manifests(
    manifests_dir: str | Path | None = None,
    manifest_paths: list[str] | None = None,
) -> list[Any]:
    manifests: list[Any] = []
    seen_paths: set[Path] = set()

    if manifest_paths:
        for p_str in manifest_paths:
            p = Path(p_str)
            if p.is_file() and p not in seen_paths:
                manifests.append(load_manifest(p))
                seen_paths.add(p)

    if manifests_dir:
        mdir = Path(manifests_dir)
        if mdir.is_dir():
            for p in sorted(mdir.rglob("manifest.json")):
                if p not in seen_paths:
                    manifests.append(load_manifest(p))
                    seen_paths.add(p)

    return manifests


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plan governed pretraining and SFT data mixtures across Indic datasets"
    )
    parser.add_argument(
        "--recipe",
        required=True,
        help="Path to mixture recipe YAML/JSON (e.g. configs/data/mixture_pretrain_indic_1b.yaml)",
    )
    parser.add_argument(
        "--manifests-dir",
        default=None,
        help="Directory containing manifest.json files to sample from",
    )
    parser.add_argument(
        "--manifests",
        nargs="*",
        default=None,
        help="Individual manifest.json file paths",
    )
    parser.add_argument(
        "--target-records",
        type=int,
        default=None,
        help="Target total record count to sample across mixture",
    )
    parser.add_argument(
        "--target-tokens",
        type=int,
        default=None,
        help="Target total token count (assumes ~512 tokens/record if per-source stats not given)",
    )
    parser.add_argument(
        "--domain-mapping",
        default=None,
        help="Optional JSON string or file path mapping source_id -> domain",
    )
    parser.add_argument(
        "--allow-split-fallback",
        action="store_true",
        help="Allow using split name as domain if domain is not specified in manifest",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional path to save planned mixture as JSON",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output plan summary as JSON",
    )
    return parser


def format_plans_summary(plans: tuple[MixturePlan, ...]) -> str:
    lines: list[str] = [
        "=" * 78,
        "🇮🇳 IndicLLM-Bharat Data Mixture Plan",
        "=" * 78,
        f"{'Source ID':<24} {'Lang':<6} {'Domain':<12} {'Weight (%)':<12} {'Est Records':<12}",
        "-" * 78,
    ]
    total_recs = 0
    total_w = 0.0
    for p in plans:
        lines.append(
            f"{p.source_id:<24} {p.language:<6} {p.domain:<12} {p.weight * 100:>8.2f}%    {p.estimated_records:>10,}"
        )
        total_recs += p.estimated_records
        total_w += p.weight
    lines.append("-" * 78)
    lines.append(f"{'TOTAL':<24} {'--':<6} {'--':<12} {total_w * 100:>8.2f}%    {total_recs:>10,}")
    lines.append("=" * 78)
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        constraint = load_recipe(args.recipe)
    except Exception as e:
        print(f"error loading recipe '{args.recipe}': {e}", file=sys.stderr)
        return 1

    manifests = collect_manifests(
        manifests_dir=args.manifests_dir,
        manifest_paths=args.manifests,
    )

    if not manifests:
        print(
            "error: no manifests found. Provide --manifests-dir or --manifests",
            file=sys.stderr,
        )
        return 1

    domain_mapping: dict[str, str] | None = None
    if args.domain_mapping:
        dm_path = Path(args.domain_mapping)
        if dm_path.is_file():
            with dm_path.open("r", encoding="utf-8") as f:
                domain_mapping = json.load(f)
        else:
            try:
                domain_mapping = json.loads(args.domain_mapping)
            except Exception as e:
                print(f"error parsing --domain-mapping: {e}", file=sys.stderr)
                return 1

    target_records = args.target_records
    if target_records is None and args.target_tokens is not None:
        target_records = max(1, args.target_tokens // 512)

    planner = MixturePlanner()
    try:
        plans = planner.plan(
            manifests=manifests,
            constraint=constraint,
            domain_mapping=domain_mapping,
            allow_split_fallback=args.allow_split_fallback,
            target_records=target_records,
        )
    except Exception as e:
        print(f"error during mixture planning: {e}", file=sys.stderr)
        return 1

    plan_dicts = [
        {
            "source_id": p.source_id,
            "weight": round(p.weight, 6),
            "estimated_records": p.estimated_records,
            "language": p.language,
            "domain": p.domain,
            "note": p.note,
        }
        for p in plans
    ]

    summary_data = {
        "recipe": str(args.recipe),
        "total_sources": len(plans),
        "total_estimated_records": sum(p.estimated_records for p in plans),
        "plans": plan_dicts,
    }

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(summary_data, f, indent=2)

    if args.json:
        print(json.dumps(summary_data, indent=2))
    else:
        print(format_plans_summary(plans))

    return 0


if __name__ == "__main__":
    sys.exit(main())
