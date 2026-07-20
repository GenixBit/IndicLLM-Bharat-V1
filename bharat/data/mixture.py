from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass


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


class MixturePlanner:
    def plan(
        self,
        manifests: Sequence,  # type: ignore[type-arg]
        constraint: MixtureConstraint,
        domain_mapping: dict[str, str] | None = None,
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
                    f"domain_weights must sum to 1.0 when provided, "
                    f"got {total_domain_weight}"
                )
        for w in constraint.domain_weights.values():
            if w < 0:
                raise ValueError(f"domain weight must be non-negative, got {w}")

        if not 0.0 < constraint.max_pct_per_source <= 1.0:
            raise ValueError("max_pct_per_source must be in (0.0, 1.0]")
        if constraint.min_record_threshold < 0:
            raise ValueError("min_record_threshold must be >= 0")

        total_records = sum(m.records for m in manifests)
        if total_records < constraint.min_record_threshold:
            raise ValueError(
                f"Total records ({total_records}) below min threshold "
                f"({constraint.min_record_threshold})"
            )

        source_record_map: dict[str, int] = {}
        for m in manifests:
            source_record_map[m.source_id] = source_record_map.get(m.source_id, 0) + m.records

        max_records_per_source = total_records * constraint.max_pct_per_source
        capped_sources: set[str] = set()
        uncapped_sources: set[str] = set()
        total_capped_records = 0
        for sid, recs in sorted(source_record_map.items()):
            if recs > max_records_per_source:
                capped_sources.add(sid)
                total_capped_records += max_records_per_source
            else:
                uncapped_sources.add(sid)

        if capped_sources and not uncapped_sources:
            raise ValueError(
                "All sources exceed the per-source cap "
                f"({constraint.max_pct_per_source:.0%}); "
                "cannot redistribute"
            )

        notes: list[str] = []
        excess = total_records - total_capped_records
        if capped_sources:
            notes.append(
                f"Capped {', '.join(sorted(capped_sources))} "
                f"to {constraint.max_pct_per_source:.0%} each; "
                f"redistributing {excess} records to uncapped sources"
            )

        uncapped_total_records = sum(
            source_record_map[sid] for sid in uncapped_sources
        ) or 1

        plans: list[MixturePlan] = []
        for m in manifests:
            lang = m.language
            lang_weight = constraint.language_weights.get(lang, 0.0)

            if domain_mapping is not None:
                domain = domain_mapping.get(m.source_id, "general")
            else:
                domain = m.split
            domain_weight = constraint.domain_weights.get(
                domain, 1.0 / max(len(constraint.domain_weights), 1)
            )

            weight = lang_weight * domain_weight
            note = ""
            raw_recs = m.records
            if m.source_id in capped_sources:
                capped = max_records_per_source
                effective_recs = int(capped * weight)
                note = f"capped at {constraint.max_pct_per_source:.0%} of total ({capped:.0f} records)"
            else:
                boost = 1.0 + excess / uncapped_total_records
                effective_recs = int(raw_recs * weight * boost)
                weight = weight * boost

            plans.append(
                MixturePlan(
                    source_id=m.source_id,
                    weight=weight,
                    estimated_records=effective_recs,
                    language=lang,
                    domain=domain,
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

        return tuple(sorted(plans, key=lambda p: (-p.weight, p.source_id)))
