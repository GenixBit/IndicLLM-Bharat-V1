from __future__ import annotations

from bharat.rag.engine import (
    SOVEREIGN_KNOWLEDGE_DOCS,
    SovereignEmbeddingModel,
    SovereignRAGPipeline,
    SovereignVectorIndex,
)
from bharat.serving.openai_server import BharatInferenceEngine
from scripts.run_rag import main as rag_main
from scripts.run_rag import parse_args


class TestSovereignRAG:
    def test_embedding_model_encoding(self):
        engine = BharatInferenceEngine(tier="tiny", device="cpu")
        embedder = SovereignEmbeddingModel(engine)
        embeds = embedder.encode(["Constitution of India", "ISRO Chandrayaan-3"])

        assert embeds.shape[0] == 2
        assert embeds.shape[1] == engine.config.hidden_size

    def test_vector_index_add_and_search(self):
        engine = BharatInferenceEngine(tier="tiny", device="cpu")
        embedder = SovereignEmbeddingModel(engine)
        index = SovereignVectorIndex(embedder)

        num_chunks = index.add_documents(SOVEREIGN_KNOWLEDGE_DOCS[:2], chunk_size=32)
        assert num_chunks > 0

        results = index.search("Who landed near the lunar south pole?", top_k=2)
        assert len(results) > 0
        assert results[0].similarity_score is not None

    def test_rag_pipeline_query(self):
        pipeline = SovereignRAGPipeline(tier="tiny", device="cpu")
        res = pipeline.query("What is the Constitution of India?", top_k=2, max_new_tokens=5)

        assert "query" in res
        assert "response" in res
        assert "citations" in res
        assert len(res["citations"]) > 0

    def test_cli_parse_args(self):
        args = parse_args(["--query", "ISRO missions", "--top-k", "3", "--tier", "350m"])
        assert args.query == "ISRO missions"
        assert args.top_k == 3
        assert args.tier == "350m"

    def test_cli_main(self):
        code = rag_main(["--query", "What is ISRO?", "--tier", "tiny", "--device", "cpu", "--json"])
        assert code == 0
