"""Multi-Stage Hybrid Search & Context Compression Engine for IndicLLM-Bharat.

Combines:
  1. Dense Vector Embeddings (Semantic Cosine Similarity)
  2. BM25 Lexical Search (Keyword & Acronym Precision)
  3. Knowledge Graph Multi-Hop Traversal
  4. Reciprocal Rank Fusion (RRF) & Context Compression
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from bharat.rag.engine import (
    SOVEREIGN_KNOWLEDGE_DOCS,
    DocumentChunk,
    SovereignEmbeddingModel,
    SovereignVectorIndex,
)
from bharat.rag.knowledge_graph import KnowledgeGraph
from bharat.serving.openai_server import BharatInferenceEngine


class BM25LexicalIndex:
    """Lightweight sovereign BM25 lexical keyword retriever."""

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.corpus_size = 0
        self.avgdl = 0.0
        self.doc_freqs: list[dict[str, int]] = []
        self.idf: dict[str, float] = {}
        self.doc_lens: list[int] = []

    def tokenize(self, text: str) -> list[str]:
        return re.findall(r"\w+", text.lower())

    def fit(self, documents: list[str]) -> None:
        self.corpus_size = len(documents)
        if self.corpus_size == 0:
            return

        total_len = 0
        df: Counter[str] = Counter()
        self.doc_freqs = []
        self.doc_lens = []

        for doc in documents:
            tokens = self.tokenize(doc)
            self.doc_lens.append(len(tokens))
            total_len += len(tokens)
            freq = Counter(tokens)
            self.doc_freqs.append(freq)
            for word in freq:
                df[word] += 1

        self.avgdl = total_len / self.corpus_size
        self.idf = {}
        for word, count in df.items():
            self.idf[word] = math.log((self.corpus_size - count + 0.5) / (count + 0.5) + 1.0)

    def search(self, query: str, top_k: int = 5) -> list[tuple[int, float]]:
        query_tokens = self.tokenize(query)
        scores: list[tuple[int, float]] = []

        for idx, (freq, doc_len) in enumerate(zip(self.doc_freqs, self.doc_lens, strict=False)):
            score = 0.0
            for token in query_tokens:
                if token in freq:
                    tf = freq[token]
                    idf_val = self.idf.get(token, 0.0)
                    numerator = idf_val * tf * (self.k1 + 1.0)
                    denominator = tf + self.k1 * (
                        1.0 - self.b + self.b * (doc_len / max(1e-5, self.avgdl))
                    )
                    score += numerator / max(1e-5, denominator)
            scores.append((idx, score))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]


@dataclass
class HybridSearchResult:
    chunk: DocumentChunk
    hybrid_score: float
    dense_rank: int | None = None
    bm25_rank: int | None = None


class SovereignHybridSearchEngine:
    """Full 8-stage hybrid retrieval engine."""

    def __init__(
        self,
        tier: str = "tiny",
        checkpoint_path: str | Path | None = None,
        device: str = "auto",
        initial_docs: list[dict[str, str]] | None = None,
    ) -> None:
        self.engine = BharatInferenceEngine(
            tier=tier, checkpoint_path=checkpoint_path, device=device
        )
        self.embedder = SovereignEmbeddingModel(self.engine)
        self.vector_index = SovereignVectorIndex(self.embedder)
        self.bm25_index = BM25LexicalIndex()
        self.knowledge_graph = KnowledgeGraph()

        docs_to_index = initial_docs if initial_docs is not None else SOVEREIGN_KNOWLEDGE_DOCS
        self.index_documents(docs_to_index)

    def index_documents(self, docs: list[dict[str, str]]) -> None:
        """Populate both dense vector and BM25 lexical indices."""
        self.vector_index.add_documents(docs, chunk_size=64, chunk_overlap=16)
        corpus_texts = [f"{c.title} {c.text}" for c in self.vector_index.chunks]
        self.bm25_index.fit(corpus_texts)

    def search_hybrid(
        self, query: str, top_k: int = 3, rrf_k: int = 60
    ) -> list[HybridSearchResult]:
        """Perform Reciprocal Rank Fusion (RRF) over Dense + BM25."""
        if not self.vector_index.chunks:
            return []

        # 1. Dense Vector Search
        dense_results = self.vector_index.search(query, top_k=len(self.vector_index.chunks))
        chunk_to_dense_rank: dict[int, int] = {}
        for rank, res in enumerate(dense_results, start=1):
            idx = res.chunk.chunk_index
            chunk_to_dense_rank[idx] = rank

        # 2. BM25 Lexical Search
        bm25_results = self.bm25_index.search(query, top_k=len(self.vector_index.chunks))
        chunk_to_bm25_rank: dict[int, int] = {}
        for rank, (doc_idx, score) in enumerate(bm25_results, start=1):
            if score > 0.0:
                chunk_to_bm25_rank[doc_idx] = rank

        # 3. Reciprocal Rank Fusion
        all_chunk_indices = set(chunk_to_dense_rank.keys()).union(set(chunk_to_bm25_rank.keys()))
        fused_scores: list[tuple[int, float]] = []

        for idx in all_chunk_indices:
            dense_r = chunk_to_dense_rank.get(idx, 1000)
            bm25_r = chunk_to_bm25_rank.get(idx, 1000)

            score = (1.0 / (rrf_k + dense_r)) + (1.0 / (rrf_k + bm25_r))
            fused_scores.append((idx, score))

        fused_scores.sort(key=lambda x: x[1], reverse=True)

        results: list[HybridSearchResult] = []
        for idx, score in fused_scores[:top_k]:
            if idx < len(self.vector_index.chunks):
                results.append(
                    HybridSearchResult(
                        chunk=self.vector_index.chunks[idx],
                        hybrid_score=round(score, 5),
                        dense_rank=chunk_to_dense_rank.get(idx),
                        bm25_rank=chunk_to_bm25_rank.get(idx),
                    )
                )

        return results

    def query_with_hybrid_rag(self, query: str, top_k: int = 2) -> dict[str, Any]:
        """Complete RAG pipeline: Hybrid Search + Graph + Context Compression + Answer."""
        hybrid_docs = self.search_hybrid(query, top_k=top_k)
        graph_facts = self.knowledge_graph.search_subgraph(query, max_hops=2)

        # Context Compression
        context_snippets: list[str] = []
        for idx, h in enumerate(hybrid_docs, 1):
            context_snippets.append(f"[{idx}] {h.chunk.title}:\n{h.chunk.text}")

        if graph_facts:
            context_snippets.append("[KG Facts]:\n" + "\n".join(graph_facts))

        compressed_context = "\n\n".join(context_snippets)

        prompt = (
            f"You are IndicLLM-Bharat, a sovereign hybrid AI system. "
            f"Synthesize an accurate answer using the verified hybrid evidence below. Cite sources accurately.\n\n"
            f"--- Evidence ---\n{compressed_context}\n----------------\n\n"
            f"User: {query}\n\n"
            f"Assistant: "
        )

        response = self.engine.generate(prompt, max_new_tokens=128, temperature=0.3)

        return {
            "query": query,
            "response": response,
            "hybrid_documents_retrieved": len(hybrid_docs),
            "graph_relationships_found": len(graph_facts),
            "citations": [
                {
                    "title": h.chunk.title,
                    "hybrid_score": h.hybrid_score,
                    "dense_rank": h.dense_rank,
                    "bm25_rank": h.bm25_rank,
                    "snippet": h.chunk.text[:120] + "...",
                }
                for h in hybrid_docs
            ],
            "graph_facts": graph_facts,
        }
