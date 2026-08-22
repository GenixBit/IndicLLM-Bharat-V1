from __future__ import annotations

from bharat.caching.multi_cache import MultiTierCache


class TestMultiCache:
    def test_exact_cache_hit_and_miss(self):
        cache = MultiTierCache(default_ttl_seconds=60.0)
        q = "What is the Constitution of India?"
        ans = "The supreme legal document of the Republic of India."

        assert cache.get_exact(q) is None
        cache.set_exact(q, ans)
        assert cache.get_exact(q) == ans

    def test_tool_cache(self):
        cache = MultiTierCache()
        args = {"expression": "sqrt(16)"}
        cache.set_tool_result("calculator", args, 4.0)
        assert cache.get_tool_result("calculator", args) == 4.0
