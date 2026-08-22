from __future__ import annotations

from bharat.web.intelligence import LiveWebIntelligenceEngine, WebSourcePassage


class TestWebIntelligence:
    def test_robots_allowed(self):
        engine = LiveWebIntelligenceEngine()
        assert engine.is_url_allowed("https://www.isro.gov.in/chandrayaan")
        assert not engine.is_url_allowed("https://paywall-news.example.com/secret")

    def test_retrieve_live_passages(self):
        engine = LiveWebIntelligenceEngine()
        passages = engine.retrieve_live_passages("ISRO Chandrayaan-4 update 2026", max_results=2)
        assert len(passages) > 0
        p = passages[0]
        assert isinstance(p, WebSourcePassage)
        assert p.authority_score > 0.5
        assert p.freshness_score > 0.5
        assert len(p.extracted_text) > 0
