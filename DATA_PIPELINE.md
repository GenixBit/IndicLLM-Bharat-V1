# Continuous World Knowledge Ingestion Pipeline Architecture

This document details the multi-format data ingestion, content deduplication, resumable checkpointing, and versioning lifecycle for IndicLLM-Bharat.

---

## 1. End-to-End Ingestion Flow

```mermaid
graph TD
    Source[File / Code / PDF / CSV / Markdown] --> Hash[SHA-256 Content Hashing]
    Hash --> Deduplicate{Already Processed in State?}
    Deduplicate -->|Yes| Skip[Skip - Zero Compute Overhead]
    Deduplicate -->|No| Parser[Multi-Format Text Parser & Sanitizer]
    Parser --> Chunker[Sliding Window Text Chunker: 500 words, 50 overlap]
    Chunker --> Version[Assign Version: v1 -> v2]
    Version --> Checkpoint[Save Resumable State Checkpoint]
    Checkpoint --> VectorDB[Index to Vector Database & BM25]
```

---

## 2. Ingestion Resilience & Checkpoint Schema

- **Resumable State**: Stored in `data/ingestion_state/ingestion_checkpoint.json`.
- **Deduplication**: Content hashing prevents duplicate embeddings and storage bloat.
- **Versioning**: Detects in-place edits and archives previous document versions.
