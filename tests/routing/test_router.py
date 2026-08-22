from __future__ import annotations

from bharat.routing.router import IntelligentModelRouter, RouteDestination


class TestModelRouter:
    def test_routing_freshness(self):
        router = IntelligentModelRouter()
        decision = router.route("What are the latest 2026 updates on ISRO Chandrayaan-4?")
        assert decision.destination == RouteDestination.LIVE_WEB
        assert decision.requires_retrieval

    def test_routing_computation(self):
        router = IntelligentModelRouter()
        decision = router.route("Calculate 1729 * 3 + 45")
        assert decision.destination == RouteDestination.TOOLS_MCP
        assert decision.requires_tools

    def test_routing_private(self):
        router = IntelligentModelRouter()
        decision = router.route(
            "Summarize our internal corporate financials", user_privacy_flag=True
        )
        assert decision.destination == RouteDestination.LOCAL_MODEL

    def test_routing_cloud_overload(self):
        router = IntelligentModelRouter()
        decision = router.route("Standard query", local_gpu_utilization=0.92)
        assert decision.destination == RouteDestination.AWS_BEDROCK
