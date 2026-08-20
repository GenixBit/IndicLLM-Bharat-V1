"""Sovereign Retrieval-Augmented Generation (RAG) Vector Retrieval Engine for IndicLLM-Bharat.

Provides multilingual document chunking, dense vector indexing, cosine similarity retrieval,
and grounded generation across all 22 Scheduled Indian Languages and English.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from bharat.serving.openai_server import BharatInferenceEngine

SOVEREIGN_KNOWLEDGE_DOCS: list[dict[str, str]] = [
    {
        "id": "doc_constitution_01",
        "title": "Constitution of India - Preamble & Fundamental Rights",
        "text": (
            "The Constitution of India is the supreme law of India. It declares India a sovereign, "
            "socialist, secular, democratic republic, assuring its citizens justice, equality, and liberty, "
            "and endeavours to promote fraternity. Part III (Articles 12 to 35) guarantees Fundamental Rights "
            "including the Right to Equality (Articles 14-18), Right to Freedom (Articles 19-22), "
            "Right against Exploitation (Articles 23-24), Right to Freedom of Religion (Articles 25-28), "
            "and Right to Constitutional Remedies (Article 32)."
        ),
    },
    {
        "id": "doc_isro_01",
        "title": "ISRO Lunar and Planetary Exploration",
        "text": (
            "The Indian Space Research Organisation (ISRO) achieved global acclaim with the Chandrayaan-3 mission, "
            "becoming the first nation to successfully soft-land near the lunar south pole on August 23, 2023. "
            "The mission comprised the Vikram lander and Pragyan rover. Other landmark missions include "
            "Mars Orbiter Mission (Mangalyaan) in 2013, Aditya-L1 solar observatory in 2023, and Gaganyaan human spaceflight."
        ),
    },
    {
        "id": "doc_indic_lang_01",
        "title": "Eighth Schedule to the Constitution of India",
        "text": (
            "The Eighth Schedule to the Constitution of India lists the 22 official scheduled languages: "
            "Assamese, Bengali, Bodo, Dogri, Gujarati, Hindi, Kannada, Kashmiri, Konkani, Maithili, "
            "Malayalam, Manipuri, Marathi, Nepali, Odia, Punjabi, Sanskrit, Santali, Sindhi, Tamil, Telugu, and Urdu. "
            "Classical language status is granted to ancient Indian languages with high antiquity and valuable heritage."
        ),
    },
    {
        "id": "doc_quantum_01",
        "title": "National Quantum Mission of India",
        "text": (
            "The National Quantum Mission (NQM) was approved by the Union Cabinet of India to seed, nurture, "
            "and scale scientific and industrial R&D in Quantum Technology. The mission targets developing "
            "intermediate-scale quantum computers with 50-1000 physical qubits in 8 years, secure quantum communications, "
            "magnetometers, and quantum materials."
        ),
    },
]


@dataclass
class DocumentChunk:
    doc_id: str
    chunk_index: int
    title: str
    text: str
    embedding: torch.Tensor | None = None


@dataclass
class RetrievalResult:
    chunk: DocumentChunk
    similarity_score: float


class SovereignEmbeddingModel:
    """Extracts normalized mean-pooled dense embeddings from Bharat foundation model."""

    def __init__(self, engine: BharatInferenceEngine) -> None:
        self.engine = engine
        self.tokenizer = engine.tokenizer
        self.model = engine.model
        self.device = engine.device

    def encode(self, texts: list[str]) -> torch.Tensor:
        """Compute normalized dense embeddings for input texts."""
        embeddings: list[torch.Tensor] = []

        with torch.no_grad():
            for text in texts:
                input_ids = self.tokenizer.encode(text)
                if not input_ids:
                    input_ids = [0]
                tokens_tensor = torch.tensor([input_ids], dtype=torch.long, device=self.device)
                tokens_tensor = tokens_tensor % self.engine.config.vocab_size

                # Extract token embeddings and mean pool
                token_embeds = self.model.model.embed_tokens(tokens_tensor)
                mean_pooled = token_embeds.mean(dim=1)
                normalized = F.normalize(mean_pooled, p=2, dim=-1)
                embeddings.append(normalized.squeeze(0).cpu())

        return torch.stack(embeddings, dim=0)


class SovereignVectorIndex:
    """In-memory dense vector index with cosine similarity search."""

    def __init__(self, embedder: SovereignEmbeddingModel) -> None:
        self.embedder = embedder
        self.chunks: list[DocumentChunk] = []
        self.embeddings: torch.Tensor | None = None

    def add_documents(
        self,
        docs: list[dict[str, str]],
        chunk_size: int = 64,
        chunk_overlap: int = 16,
    ) -> int:
        """Chunk documents and index their dense representations."""
        new_chunks: list[DocumentChunk] = []

        for doc in docs:
            text = doc.get("text", "")
            words = text.split()
            if not words:
                continue

            step = max(1, chunk_size - chunk_overlap)
            for i in range(0, len(words), step):
                chunk_words = words[i : i + chunk_size]
                chunk_str = " ".join(chunk_words)
                new_chunks.append(
                    DocumentChunk(
                        doc_id=doc.get("id", f"doc_{len(self.chunks)}"),
                        chunk_index=len(new_chunks),
                        title=doc.get("title", "Untitled Document"),
                        text=chunk_str,
                    )
                )

        if not new_chunks:
            return 0

        texts_to_embed = [f"{c.title}: {c.text}" for c in new_chunks]
        dense_embeds = self.embedder.encode(texts_to_embed)

        for chunk, emb in zip(new_chunks, dense_embeds, strict=False):
            chunk.embedding = emb

        if self.embeddings is None:
            self.embeddings = dense_embeds
        else:
            self.embeddings = torch.cat([self.embeddings, dense_embeds], dim=0)

        self.chunks.extend(new_chunks)
        return len(new_chunks)

    def search(self, query: str, top_k: int = 3) -> list[RetrievalResult]:
        """Search vector index for most relevant document chunks."""
        if not self.chunks or self.embeddings is None:
            return []

        query_emb = self.embedder.encode([query])  # (1, hidden_dim)
        # Cosine similarity between query and all chunks
        scores = torch.mm(query_emb, self.embeddings.t()).squeeze(0)  # (N,)

        k = min(top_k, len(self.chunks))
        top_scores, top_indices = torch.topk(scores, k=k)

        results: list[RetrievalResult] = []
        for score, idx in zip(top_scores.tolist(), top_indices.tolist(), strict=False):
            results.append(
                RetrievalResult(
                    chunk=self.chunks[idx],
                    similarity_score=float(score),
                )
            )
        return results


class SovereignRAGPipeline:
    """End-to-end grounded RAG generation pipeline."""

    def __init__(
        self,
        tier: str = "1b",
        checkpoint_path: str | Path | None = None,
        device: str = "auto",
        initial_docs: list[dict[str, str]] | None = None,
    ) -> None:
        self.engine = BharatInferenceEngine(
            tier=tier,
            checkpoint_path=checkpoint_path,
            device=device,
        )
        self.embedder = SovereignEmbeddingModel(self.engine)
        self.index = SovereignVectorIndex(self.embedder)

        docs_to_load = initial_docs if initial_docs is not None else SOVEREIGN_KNOWLEDGE_DOCS
        self.index.add_documents(docs_to_load)

    def query(
        self,
        user_query: str,
        top_k: int = 2,
        max_new_tokens: int = 128,
        temperature: float = 0.7,
    ) -> dict[str, Any]:
        """Retrieve grounded context and generate answer with citations."""
        results = self.index.search(user_query, top_k=top_k)

        context_blocks: list[str] = []
        citations: list[dict[str, Any]] = []

        for idx, res in enumerate(results, start=1):
            context_blocks.append(f"[{idx}] {res.chunk.title}:\n{res.chunk.text}")
            citations.append(
                {
                    "citation_id": idx,
                    "doc_id": res.chunk.doc_id,
                    "title": res.chunk.title,
                    "score": round(res.similarity_score, 4),
                    "snippet": res.chunk.text[:120] + "...",
                }
            )

        grounded_context = "\n\n".join(context_blocks)
        prompt = (
            f"You are IndicLLM-Bharat, a sovereign AI assistant. "
            f"Use the following retrieved official documents to answer the user request accurately. "
            f"Cite your sources using [1], [2], etc.\n\n"
            f"--- Context Documents ---\n{grounded_context}\n-------------------------\n\n"
            f"User: {user_query}\n\n"
            f"Assistant: "
        )

        response = self.engine.generate(
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )

        return {
            "query": user_query,
            "response": response,
            "citations": citations,
            "documents_retrieved": len(results),
        }
