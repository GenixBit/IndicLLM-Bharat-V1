"""Intelligent Multi-Tier Model Router for IndicLLM-Bharat.

Decides dynamically whether to route queries to:
  1. CACHE: Exact or semantic query match
  2. LOCAL: Sovereign local foundation model (low latency, private, simple)
  3. AWS_BEDROCK: Cloud models (complex reasoning, high concurrency, long context)
  4. WEB: Live web intelligence (latest 2026 news, current prices, documentation)
  5. TOOLS: Deterministic calculation, code execution, unit conversion, GitHub/DB
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import ClassVar


class RouteDestination(str, Enum):
    CACHE = "CACHE"
    LOCAL_MODEL = "LOCAL_MODEL"
    AWS_BEDROCK = "AWS_BEDROCK"
    LIVE_WEB = "LIVE_WEB"
    TOOLS_MCP = "TOOLS_MCP"


@dataclass
class QueryAnalysis:
    raw_query: str
    intent: str
    is_freshness_required: bool
    is_computation_required: bool
    is_private_data: bool
    complexity_score: float
    detected_language: str


@dataclass
class RoutingDecision:
    destination: RouteDestination
    target_model_tier: str
    reason: str
    estimated_latency_ms: float
    estimated_cost_usd: float
    requires_retrieval: bool
    requires_tools: bool


class IntelligentModelRouter:
    """Classifies queries and dynamically routes across local, cloud, web, and tools."""

    # Keywords signaling real-time / web freshness
    FRESHNESS_TRIGGERS: ClassVar[list[str]] = [
        "latest",
        "today",
        "current",
        "recent",
        "2026",
        "this week",
        "news",
        "stock price",
        "weather",
        "updates",
        "breaking",
        "right now",
        "हालिया",
        "आज",
        "वर्तमान",
    ]

    # Keywords signaling math calculation / tools
    COMPUTATION_TRIGGERS: ClassVar[list[str]] = [
        "calculate",
        "compute",
        "solve",
        "math",
        "evaluate",
        "+",
        "-",
        "*",
        "/",
        "^",
        "convert",
        "celsius to",
        "km to",
        "miles to",
        "gb to",
        "गणना",
        "बदलें",
    ]

    # Keywords signaling complex reasoning / proofs
    COMPLEX_REASONING_TRIGGERS: ClassVar[list[str]] = [
        "derive",
        "prove",
        "step by step proof",
        "formal proof",
        "calculus",
        "differential equation",
        "quantum algorithm",
        "theorem",
        "सिद्ध कीजिए",
        "प्रमेय",
    ]

    def analyze_query(self, query: str, user_privacy_flag: bool = False) -> QueryAnalysis:
        """Extract multi-dimensional intent, freshness, and complexity metrics."""
        q_lower = query.lower()

        # Check freshness
        is_fresh = any(kw in q_lower for kw in self.FRESHNESS_TRIGGERS)

        # Check computation / tool need
        is_compute = any(f" {kw} " in f" {q_lower} " for kw in self.COMPUTATION_TRIGGERS)
        if re.search(r"\b\d+\s*[\+\*\/]\s*\d+\b|\b\d+\s+-\s+\d+\b", query):
            is_compute = True

        # Check complex reasoning
        is_complex = any(kw in q_lower for kw in self.COMPLEX_REASONING_TRIGGERS)

        complexity = 0.8 if is_complex else (0.5 if len(query.split()) > 40 else 0.2)

        # Language detection heuristic
        indic_chars = sum(1 for c in query if 0x0900 <= ord(c) <= 0x0D7F)
        detected_lang = "indic" if indic_chars > 5 else "en"

        intent = "GENERAL"
        if is_fresh:
            intent = "REAL_TIME_WEB"
        elif is_compute:
            intent = "COMPUTATION"
        elif is_complex:
            intent = "COMPLEX_REASONING"

        return QueryAnalysis(
            raw_query=query,
            intent=intent,
            is_freshness_required=is_fresh,
            is_computation_required=is_compute,
            is_private_data=user_privacy_flag,
            complexity_score=complexity,
            detected_language=detected_lang,
        )

    def route(
        self,
        query: str,
        user_privacy_flag: bool = False,
        local_gpu_utilization: float = 0.0,
        context_tokens: int = 0,
    ) -> RoutingDecision:
        """Determine the optimal compute, retrieval, or tool destination."""
        analysis = self.analyze_query(query, user_privacy_flag)

        # 1. Private data or explicit local preference
        if analysis.is_private_data:
            return RoutingDecision(
                destination=RouteDestination.LOCAL_MODEL,
                target_model_tier="1b",
                reason="User data is marked private/confidential. Enforcing 100% sovereign local processing.",
                estimated_latency_ms=25.0,
                estimated_cost_usd=0.0,
                requires_retrieval=True,
                requires_tools=False,
            )

        # 2. Computation / Tool needed
        if analysis.is_computation_required:
            return RoutingDecision(
                destination=RouteDestination.TOOLS_MCP,
                target_model_tier="local_mcp",
                reason="Query contains mathematical expressions or conversions requiring deterministic tool execution.",
                estimated_latency_ms=10.0,
                estimated_cost_usd=0.0,
                requires_retrieval=False,
                requires_tools=True,
            )

        # 3. Live Web Freshness required
        if analysis.is_freshness_required:
            return RoutingDecision(
                destination=RouteDestination.LIVE_WEB,
                target_model_tier="1b",
                reason="Query demands real-time or latest 2026 information beyond static training cutoffs.",
                estimated_latency_ms=180.0,
                estimated_cost_usd=0.0001,
                requires_retrieval=True,
                requires_tools=True,
            )

        # 4. Cloud Escalation (High GPU overload, extreme context > 8k, or extreme complexity > 0.75)
        if (
            local_gpu_utilization > 0.85
            or context_tokens > 8192
            or analysis.complexity_score > 0.75
        ):
            return RoutingDecision(
                destination=RouteDestination.AWS_BEDROCK,
                target_model_tier="bedrock-claude-or-llama",
                reason="Task requires heavy reasoning or long context exceeding local GPU saturation threshold.",
                estimated_latency_ms=350.0,
                estimated_cost_usd=0.0015,
                requires_retrieval=True,
                requires_tools=False,
            )

        # 5. Default Local Fast Path
        return RoutingDecision(
            destination=RouteDestination.LOCAL_MODEL,
            target_model_tier="1b",
            reason="Standard query comfortably serviced by local sovereign model with low latency and zero cost.",
            estimated_latency_ms=20.0,
            estimated_cost_usd=0.0,
            requires_retrieval=False,
            requires_tools=False,
        )
