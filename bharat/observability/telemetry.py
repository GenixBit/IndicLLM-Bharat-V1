"""Telemetry, Metrics & Observability Collector for IndicLLM-Bharat.

Tracks:
  - TTFT (Time To First Token) & TPS (Tokens Per Second)
  - Routing destination breakdown (Local, Cloud, Web, Tools, Cache)
  - Token counts and cloud cost estimation ($)
  - Cache hit/miss rates and system health
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class RequestMetric:
    request_id: str
    destination: str
    ttft_ms: float
    total_latency_ms: float
    prompt_tokens: int
    completion_tokens: int
    estimated_cost_usd: float
    timestamp: float = field(default_factory=time.time)


class TelemetryCollector:
    """Aggregates metrics for CloudWatch, OpenTelemetry, and system dashboards."""

    def __init__(self) -> None:
        self.metrics: list[RequestMetric] = []
        self.cache_hits = 0
        self.cache_misses = 0

    def record_request(
        self,
        request_id: str,
        destination: str,
        ttft_ms: float,
        total_latency_ms: float,
        prompt_tokens: int,
        completion_tokens: int,
        estimated_cost_usd: float = 0.0,
    ) -> None:
        self.metrics.append(
            RequestMetric(
                request_id=request_id,
                destination=destination,
                ttft_ms=ttft_ms,
                total_latency_ms=total_latency_ms,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                estimated_cost_usd=estimated_cost_usd,
            )
        )

    def record_cache_hit(self) -> None:
        self.cache_hits += 1

    def record_cache_miss(self) -> None:
        self.cache_misses += 1

    def get_summary(self) -> dict[str, Any]:
        total_reqs = len(self.metrics)
        if total_reqs == 0:
            return {
                "total_requests": 0,
                "avg_ttft_ms": 0.0,
                "avg_tps": 0.0,
                "cache_hit_rate_pct": 0.0,
                "total_estimated_cloud_cost_usd": 0.0,
            }

        avg_ttft = sum(m.ttft_ms for m in self.metrics) / total_reqs
        total_tokens = sum(m.completion_tokens for m in self.metrics)
        total_gen_time_s = sum(m.total_latency_ms for m in self.metrics) / 1000.0
        avg_tps = total_tokens / max(1e-5, total_gen_time_s)

        total_cache_ops = self.cache_hits + self.cache_misses
        hit_rate = (self.cache_hits / max(1, total_cache_ops)) * 100.0

        total_cost = sum(m.estimated_cost_usd for m in self.metrics)

        # Breakdown by destination
        dest_counts: dict[str, int] = {}
        for m in self.metrics:
            dest_counts[m.destination] = dest_counts.get(m.destination, 0) + 1

        return {
            "total_requests": total_reqs,
            "avg_ttft_ms": round(avg_ttft, 2),
            "avg_tps": round(avg_tps, 2),
            "total_prompt_tokens": sum(m.prompt_tokens for m in self.metrics),
            "total_completion_tokens": total_tokens,
            "cache_hit_rate_pct": round(hit_rate, 2),
            "total_estimated_cloud_cost_usd": round(total_cost, 6),
            "destination_breakdown": dest_counts,
        }
