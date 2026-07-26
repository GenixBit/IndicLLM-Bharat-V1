# Milestone 5.3: Local GGUF compatibility reader

This slice adds a deterministic, offline reader for the exact GGUF v3 subset emitted by the repository's local F32 writer.

## Acceptance covered

- validates GGUF magic and version
- reads scalar bool, float64, signed int64, and UTF-8 string metadata
- requires deterministic metadata and tensor-name ordering
- reads F32 tensor descriptors and restores logical tensor shapes
- validates aligned, increasing tensor offsets
- verifies zero-filled header alignment padding
- verifies every declared tensor payload remains within local file bounds
- returns stable JSON-serializable metadata without interpreting tensor values
- includes focused local fixtures for valid output and corruption cases

## Safety boundary

The reader performs local file reads only. It does not download models, datasets, or benchmarks; call external APIs; scrape; upload files or artifacts; invoke subprocesses or external GGUF tools; train or quantize models; or access the network.

This is compatibility validation for the repository's supported GGUF subset. It is not a claim of support for every GGUF metadata type, quantization type, architecture, or third-party implementation.

## Remaining Milestone 5.3 work

Architecture-complete Bharat metadata and independent trusted-reader validation remain required before the GGUF export can be described as broadly runnable or the milestone can be closed.
