from __future__ import annotations

from bharat.rag.hybrid_search import BM25LexicalIndex, SovereignHybridSearchEngine
from bharat.rag.knowledge_graph import KnowledgeGraph


class TestHybridSearchAndGraph:
    def test_bm25_lexical_search(self):
        bm25 = BM25LexicalIndex()
        corpus = [
            "The Constitution of India guarantees fundamental rights.",
            "ISRO successfully launched the Chandrayaan-3 lunar mission.",
            "Python dynamic programming algorithms with Kadane approach.",
        ]
        bm25.fit(corpus)
        results = bm25.search("Chandrayaan-3 mission", top_k=1)
        assert len(results) == 1
        assert results[0][0] == 1  # Index 1 matched

    def test_knowledge_graph_traversal(self):
        kg = KnowledgeGraph()
        facts = kg.search_subgraph("ISRO Chandrayaan-3", max_hops=2)
        assert len(facts) > 0
        assert any("ISRO" in f for f in facts)

    def test_hybrid_search_engine(self):
        engine = SovereignHybridSearchEngine(tier="tiny", device="cpu")
        res = engine.search_hybrid("Constitution Fundamental Rights", top_k=2)
        assert len(res) > 0
        assert res[0].hybrid_score > 0.0

        rag_out = engine.query_with_hybrid_rag("Fundamental Rights", top_k=1)
        assert "response" in rag_out
        assert len(rag_out["citations"]) > 0
