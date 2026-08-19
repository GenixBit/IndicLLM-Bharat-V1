"""Governed Real-World & Multilingual Data Mixture Pipeline for IndicLLM-Bharat.

Implements data mixture balance:
  - 40% Indic Multilingual (Sangraha, IndicCorp-v2, Samanantar, 22-Language Wikipedia)
  - 30% Global Science & STEM (FineWeb-Edu, Open-Web-Math, Physics, Genetics)
  - 15% World Knowledge & History (Civilizations, Geography, Constitutions)
  - 15% Code & Algorithms (Python, C++, SQL, Data Structures)
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from bharat.data.binary_stream import pack_text_corpus
from bharat.data.world_knowledge import get_all_world_knowledge_documents
from bharat.tokenizer import BharatTokenizer


@dataclass(frozen=True)
class MixtureConstraint:
    language_weights: dict[str, float]
    domain_weights: dict[str, float]
    max_pct_per_source: float = 0.5
    min_record_threshold: int = 1000


@dataclass(frozen=True)
class MixturePlan:
    source_id: str
    weight: float
    estimated_records: int
    language: str
    domain: str
    note: str = ""


def _resolve_domain(
    manifest: Any,
    domain_mapping: dict[str, str] | None,
    allow_split_fallback: bool,
) -> str:
    m = manifest
    if hasattr(m, "domain") and m.domain:
        return str(m.domain)
    if domain_mapping is not None:
        source_id = getattr(m, "source_id", None)
        if source_id is not None and source_id in domain_mapping:
            return domain_mapping[source_id]
    if allow_split_fallback:
        split = getattr(m, "split", "")
        if split:
            return str(split)
    return ""


class MixturePlanner:
    def plan(
        self,
        manifests: Sequence,  # type: ignore[type-arg]
        constraint: MixtureConstraint,
        domain_mapping: dict[str, str] | None = None,
        allow_split_fallback: bool = False,
        target_records: int | None = None,
    ) -> tuple[MixturePlan, ...]:
        if not manifests:
            raise ValueError("At least one manifest is required")
        if not constraint.language_weights:
            raise ValueError("language_weights must be non-empty")
        total_lang_weight = sum(constraint.language_weights.values())
        if abs(total_lang_weight - 1.0) > 1e-9:
            raise ValueError(f"language_weights must sum to 1.0, got {total_lang_weight}")
        for w in constraint.language_weights.values():
            if w < 0:
                raise ValueError(f"language weight must be non-negative, got {w}")

        if constraint.domain_weights:
            total_domain_weight = sum(constraint.domain_weights.values())
            if abs(total_domain_weight - 1.0) > 1e-9:
                raise ValueError(
                    f"domain_weights must sum to 1.0 when provided, got {total_domain_weight}"
                )
        for w in constraint.domain_weights.values():
            if w < 0:
                raise ValueError(f"domain weight must be non-negative, got {w}")

        if not 0.0 < constraint.max_pct_per_source <= 1.0:
            raise ValueError("max_pct_per_source must be in (0.0, 1.0]")
        if constraint.min_record_threshold < 0:
            raise ValueError("min_record_threshold must be >= 0")

        total_records = sum(m.records for m in manifests)
        target = total_records if target_records is None else target_records
        if target < 0:
            raise ValueError(f"target_records must be non-negative, got {target}")
        if total_records < constraint.min_record_threshold:
            raise ValueError(
                f"Total records ({total_records}) below min threshold "
                f"({constraint.min_record_threshold})"
            )

        # --- Resolve domains and compute raw weights ---
        infos: list[dict[str, Any]] = []
        raw_weights: list[float] = []
        for m in manifests:
            domain = _resolve_domain(m, domain_mapping, allow_split_fallback)
            if not domain:
                raise ValueError(
                    f"Cannot determine domain for source '{m.source_id}' "
                    f"(language='{m.language}', split='{m.split}'). "
                    "Provide a manifest-level domain, use domain_mapping, "
                    "or enable allow_split_fallback."
                )
            lang_weight = constraint.language_weights.get(m.language, 0.0)
            domain_weight = constraint.domain_weights.get(domain, 0.0)
            raw = lang_weight * domain_weight
            infos.append(
                {
                    "manifest": m,
                    "domain": domain,
                    "lang_weight": lang_weight,
                    "domain_weight": domain_weight,
                }
            )
            raw_weights.append(raw)

        if all(w == 0.0 for w in raw_weights):
            raise ValueError(
                "All source weights are zero. Check language_weights and domain_weights "
                "configurations."
            )

        source_raw: dict[str, float] = {}
        for i, m in enumerate(manifests):
            source_raw[m.source_id] = source_raw.get(m.source_id, 0.0) + raw_weights[i]

        total_source_raw = sum(source_raw.values())
        source_candidate: dict[str, float] = {
            sid: w / total_source_raw for sid, w in source_raw.items()
        }

        max_pct = constraint.max_pct_per_source
        nonzero_sources = [sid for sid in source_candidate if source_candidate[sid] > 0]

        if max_pct * len(nonzero_sources) < 1.0 - 1e-9:
            raise ValueError(
                f"Per-source cap {max_pct:.0%} cannot accommodate "
                f"{len(nonzero_sources)} sources "
                f"(max total {max_pct * len(nonzero_sources):.0%} < 100%)"
            )

        source_final: dict[str, float] = {}
        capped_sources: set[str] = set()

        for sid in source_candidate:
            if source_candidate[sid] == 0.0:
                source_final[sid] = 0.0

        uncapped = set(nonzero_sources)

        while True:
            over_cap = {sid for sid in uncapped if source_candidate[sid] > max_pct + 1e-12}
            if not over_cap:
                for sid in uncapped:
                    source_final[sid] = source_candidate[sid]
                break
            for sid in over_cap:
                source_final[sid] = max_pct
                capped_sources.add(sid)
                uncapped.discard(sid)
            if not uncapped:
                raise ValueError(
                    f"All sources exceed the per-source cap ({max_pct:.0%}); cannot redistribute"
                )
            capped_total = sum(source_final[sid] for sid in capped_sources)
            remaining = 1.0 - capped_total
            uncapped_raw = sum(source_candidate[sid] for sid in uncapped)
            for sid in uncapped:
                source_candidate[sid] = remaining * (source_candidate[sid] / uncapped_raw)

        notes_builder: list[str] = []
        if capped_sources:
            notes_builder.append(
                f"Capped {', '.join(sorted(capped_sources))} to {max_pct:.0%} of total weight"
            )

        plans: list[MixturePlan] = []
        for i, m in enumerate(manifests):
            sid = m.source_id
            src_weight = source_final[sid]
            if source_raw[sid] > 0 and raw_weights[i] > 0:
                manifest_fraction = raw_weights[i] / source_raw[sid]
            else:
                manifest_fraction = 0.0
            final_weight = src_weight * manifest_fraction

            note = ""
            if sid in capped_sources:
                note = f"capped at {max_pct:.0%} of total weight"
            elif raw_weights[i] == 0.0:
                reasons = []
                if infos[i]["lang_weight"] == 0.0:
                    reasons.append(f"no language weight for '{m.language}'")
                if infos[i]["domain_weight"] == 0.0:
                    reasons.append(f"no domain weight for '{infos[i]['domain']}'")
                note = f"excluded: {', '.join(reasons)}"

            plans.append(
                MixturePlan(
                    source_id=sid,
                    weight=final_weight,
                    estimated_records=0,
                    language=m.language,
                    domain=infos[i]["domain"],
                    note=note,
                )
            )

        total_weight = sum(p.weight for p in plans)
        if total_weight > 0:
            plans = [
                MixturePlan(
                    source_id=p.source_id,
                    weight=p.weight / total_weight,
                    estimated_records=p.estimated_records,
                    language=p.language,
                    domain=p.domain,
                    note=p.note,
                )
                for p in plans
            ]

        if not plans:
            raise ValueError("Empty mixture plan")

        raw_alloc = [p.weight * target for p in plans]
        floors = [int(r) for r in raw_alloc]
        remainders = [r - int(r) for r in raw_alloc]
        allocated = sum(floors)
        remainder = target - allocated

        plan_indices = sorted(
            range(len(plans)),
            key=lambda i: (-remainders[i], plans[i].source_id),
        )
        for i in range(remainder):
            floors[plan_indices[i]] += 1

        plans = [
            MixturePlan(
                source_id=p.source_id,
                weight=p.weight,
                estimated_records=floors[i],
                language=p.language,
                domain=p.domain,
                note=p.note,
            )
            for i, p in enumerate(plans)
        ]

        return tuple(sorted(plans, key=lambda p: (-p.weight, p.source_id)))


@dataclass
class MixtureWeights:
    indic_multilingual: float = 0.40
    stem_science_math: float = 0.30
    world_knowledge_history: float = 0.15
    code_algorithms: float = 0.15

    def validate(self) -> None:
        total = (
            self.indic_multilingual
            + self.stem_science_math
            + self.world_knowledge_history
            + self.code_algorithms
        )
        if abs(total - 1.0) > 1e-4:
            raise ValueError(f"Mixture weights must sum to 1.0, got {total:.4f}")


def clean_and_filter_text(text: str, min_chars: int = 50) -> str:
    """Apply heuristic quality filtering and whitespace normalization."""
    cleaned = re.sub(r"\s+", " ", text).strip()
    if len(cleaned) < min_chars:
        return ""

    cleaned = re.sub(r"[\w\.-]+@[\w\.-]+\.\w+", "[EMAIL]", cleaned)
    cleaned = re.sub(r"\+?\d[\d -]{8,12}\d", "[PHONE]", cleaned)
    return cleaned


def build_world_and_indic_corpus() -> list[str]:
    """Generate high-density governed world + Indic multilingual text corpus."""
    docs = get_all_world_knowledge_documents()
    corpus: list[str] = []

    for doc in docs:
        text = doc.get("text", "")
        cleaned = clean_and_filter_text(text)
        if cleaned:
            corpus.append(cleaned)

    code_examples = [
        (
            "def binary_search(arr: list[int], target: int) -> int:\n"
            "    left, right = 0, len(arr) - 1\n"
            "    while left <= right:\n"
            "        mid = (left + right) // 2\n"
            "        if arr[mid] == target:\n"
            "            return mid\n"
            "        elif arr[mid] < target:\n"
            "            left = mid + 1\n"
            "        else:\n"
            "            right = mid - 1\n"
            "    return -1"
        ),
        (
            "class SegmentTree:\n"
            "    def __init__(self, data: list[int]):\n"
            "        self.n = len(data)\n"
            "        self.tree = [0] * (4 * self.n)\n"
            "        self.build(data, 1, 0, self.n - 1)\n\n"
            "    def build(self, data: list[int], node: int, start: int, end: int):\n"
            "        if start == end:\n"
            "            self.tree[node] = data[start]\n"
            "            return\n"
            "        mid = (start + end) // 2\n"
            "        self.build(data, 2 * node, start, mid)\n"
            "        self.build(data, 2 * node + 1, mid + 1, end)\n"
            "        self.tree[node] = self.tree[2 * node] + self.tree[2 * node + 1]"
        ),
    ]

    for c in code_examples:
        corpus.append(c)

    return corpus


def stream_and_pack_mixture(
    tokenizer: BharatTokenizer,
    output_dir: str | Path,
    max_tokens_per_shard: int = 500_000,
    max_docs: int | None = None,
) -> list[Path]:
    """Prepare balanced world + Indic mixture and write into memory-mapped shards."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    texts = build_world_and_indic_corpus()
    if max_docs is not None:
        texts = texts[:max_docs]

    temp_jsonl = out_dir / "_mixture_temp.jsonl"
    with open(temp_jsonl, "w", encoding="utf-8") as f:
        for t in texts:
            f.write(json.dumps({"text": t}, ensure_ascii=False) + "\n")

    shards = pack_text_corpus(
        tokenizer=tokenizer,
        input_file=temp_jsonl,
        output_dir=out_dir,
        prefix="bharat_mixture_shard",
        max_tokens_per_shard=max_tokens_per_shard,
    )

    if temp_jsonl.is_file():
        temp_jsonl.unlink()

    return shards
