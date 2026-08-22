# Sovereign Multi-Stage Hybrid RAG Architecture

This document specifies the 8-stage hybrid retrieval architecture combining Dense Vector Search, BM25 Lexical Matching, Knowledge Graph Entity Traversals, Reciprocal Rank Fusion, and Context Compression.

---

## 1. Multi-Stage Retrieval Pipeline

```mermaid
graph TD
    Query[User Query] --> Understand[Stage 1: Query Understanding & Entity Extraction]
    Understand --> Dense[Stage 2: Dense Vector Embeddings Cosine Search]
    Understand --> BM25[Stage 3: BM25 Lexical Keyword Search]
    Understand --> KG[Stage 4: Knowledge Graph Multi-Hop Traversal]
    
    Dense & BM25 --> RRF["Stage 5: Reciprocal Rank Fusion (RRF: Dense + BM25)"]
    RRF & KG --> Rerank[Stage 6: Cross-Encoder Reranking & Filtering]
    Rerank --> Compress[Stage 7: Context Compression & Token Pruning]
    Compress --> LLM[Stage 8: Grounded Generation with Traceable Citations]
```

---

## 2. Mathematical Formulation of Reciprocal Rank Fusion (RRF)

For each candidate document chunk $d$ across retrieval models $M = \{\text{dense}, \text{bm25}\}$:

$$RRF(d) = \sum_{m \in M} \frac{1}{k + r_m(d)}$$

where $k = 60$ is the smoothing constant, and $r_m(d)$ is the 1-based rank assigned by retrieval channel $m$.
