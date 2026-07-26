# Milestone 5.3 — GGUF Header Writer Foundation

This slice adds a deterministic, local GGUF v3 header and scalar-metadata writer.

## Included

- GGUF v3 magic, version, tensor-count, and metadata-count fields
- deterministic metadata ordering by key
- scalar `bool`, `float`, `int`, and `string` encoding
- signed 64-bit integer range validation
- local same-directory temporary file creation
- atomic no-overwrite publication using a hard link
- temporary-file cleanup after success or failure
- focused offline tests using tiny in-memory fixtures

## Deliberate boundary

The writer only accepts `tensor_count == 0`. It does not write tensor descriptors, tensor payloads, quantized data, or a runnable model. This is a format foundation rather than completion of GGUF model export.

No training, checkpoint loading, model/dataset/benchmark downloads, external APIs, scraping, uploads, workflow artifacts, external GGUF tools, quantization, or network access are introduced.

## Milestone status

Milestone 5.3 remains incomplete until real local GGUF tensor descriptors and model-weight payload serialization are implemented and verified alongside the existing safetensors export path.
