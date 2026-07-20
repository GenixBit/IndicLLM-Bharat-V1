from __future__ import annotations

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


class MixturePlanner:
    def plan(
        self,
        manifests: list,
        constraint: MixtureConstraint,
    ) -> tuple[MixturePlan, ...]:
        if not manifests:
            raise ValueError("At least one manifest is required")
        if not constraint.language_weights:
            raise ValueError("language_weights must be non-empty")
        total_lang_weight = sum(constraint.language_weights.values())
        if abs(total_lang_weight - 1.0) > 1e-9:
            raise ValueError(
                f"language_weights must sum to 1.0, got {total_lang_weight}"
            )
        for w in constraint.language_weights.values():
            if w < 0:
                raise ValueError(
                    f"language weight must be non-negative, got {w}"
                )
        for w in constraint.domain_weights.values():
            if w < 0:
                raise ValueError(
                    f"domain weight must be non-negative, got {w}"
                )
        if not 0.0 < constraint.max_pct_per_source <= 1.0:
            raise ValueError(
                "max_pct_per_source must be in (0.0, 1.0]"
            )
        if constraint.min_record_threshold < 0:
            raise ValueError(
                "min_record_threshold must be >= 0"
            )

        total_records = sum(m.records for m in manifests)
        if total_records < constraint.min_record_threshold:
            raise ValueError(
                f"Total records ({total_records}) below min threshold "
                f"({constraint.min_record_threshold})"
            )

        source_record_map: dict[str, int] = {}
        for m in manifests:
            source_record_map[m.source_id] = (
                source_record_map.get(m.source_id, 0) + m.records
            )

        max_records_per_source = total_records * constraint.max_pct_per_source
        for sid, recs in source_record_map.items():
            if recs > max_records_per_source:
                raise ValueError(
                    f"Source '{sid}' has {recs} records "
                    f"({recs/total_records:.1%}) exceeding max "
                    f"{constraint.max_pct_per_source:.0%} cap "
                    f"({max_records_per_source:.0f})"
                )

        plans: list[MixturePlan] = []
        for m in manifests:
            lang = m.language
            lang_weight = constraint.language_weights.get(lang, 0.0)
            domain = m.split
            domain_weight = constraint.domain_weights.get(
                domain, 1.0 / max(len(constraint.domain_weights), 1)
            )
            weight = lang_weight * domain_weight
            plans.append(
                MixturePlan(
                    source_id=m.source_id,
                    weight=weight,
                    estimated_records=int(m.records * weight),
                    language=lang,
                    domain=domain,
                )
            )

        if not plans:
            raise ValueError("Empty mixture plan")

        return tuple(sorted(plans, key=lambda p: (-p.weight, p.source_id)))
