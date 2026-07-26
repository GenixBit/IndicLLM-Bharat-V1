# Milestone 5.3 — GGUF F32 Tensor Payload Writing

**Status:** Implemented as a local deterministic F32-only writer slice

## Objective

Write complete GGUF v3 files for explicitly supplied in-memory dense `torch.float32`
tensors, building on the existing deterministic metadata and tensor-descriptor
encoders.

## Supported behavior

- local-only execution
- GGUF v3 little-endian output
- deterministic tensor-name ordering
- dense, unquantized F32 tensors only
- CPU, detached, contiguous tensor normalization
- descriptor offsets aligned to the validated GGUF alignment
- zero-filled alignment gaps between tensor payloads
- existing-output rejection
- same-directory temporary file and atomic no-overwrite publication
- deterministic result metadata

## Deliberate exclusions

This slice does not add:

- training or fine-tuning
- checkpoint or model downloads
- dataset or benchmark downloads
- network access, external APIs, or scraping
- external GGUF tools
- quantization
- automatic checkpoint loading
- export registry or CLI integration
- uploads or workflow artifacts

Only tensors explicitly provided by the local caller are serialized.

## Acceptance evidence

Offline tests cover:

1. deterministic ordering independent of mapping insertion order
2. correct aligned F32 payload placement
3. successful local file writing and result metadata
4. tensor-count mismatch rejection
5. non-F32 and sparse tensor rejection
6. no-overwrite behavior and input normalization without mutation

## Remaining Milestone 5.3 work

- independent compatibility validation against a trusted local GGUF reader
- integration into the export registry and CLI behind explicit execution mode
- model architecture metadata completeness for runnable Bharat checkpoints
- broader GGML tensor types or quantization only in a separately approved scope
