# Milestone 5.4 — GGUF Q8_0 Closure

**Status:** Complete
**Date:** 2026-07-27

## Scope completed

Milestone 5.4 adds explicit, deterministic, local, CPU-only GGUF Q8_0 export while preserving the existing F32 path as the default.

The completed implementation includes:

- standalone Q8_0 quantization and dequantization
- GGUF Q8_0 tensor descriptors and payload serialization
- repository-local reader support for Q8_0 files
- registry selection between F32 and Q8_0 writers
- explicit CLI selection through `--gguf-tensor-type q8_0`
- manifest reporting for tensor type and per-type tensor counts
- mismatch rejection across export plan, preflight, registry, and concrete writer
- deterministic local compatibility fixtures
- independent validation with the official `gguf==0.19.0` Python package

## Acceptance evidence

The closure is based on the merged implementation and compatibility work through PR #56.

Verified properties include:

1. repeated Q8_0 exports are byte-identical
2. Q8_0 descriptors preserve tensor names, shapes, types, and aligned offsets
3. local and independent readers parse repository-generated files
4. independent dequantization meets the documented numeric thresholds for the deterministic fixtures
5. F32 remains the default and continues to parse independently
6. unsupported tensor types and cross-layer mismatches fail before output publication
7. NaN, Inf, empty, partial-block, existing-output, and corrupted-file cases are rejected
8. atomic no-overwrite publication and temporary-file cleanup remain in force
9. checkpoint loading remains CPU-only with `torch.load(..., weights_only=True, map_location="cpu")`
10. CI run #311 completed successfully for the independent compatibility-validation head

## Compatibility boundary

The verified compatibility boundary is the repository-generated GGUF v3 F32/Q8_0 subset exercised by the deterministic fixtures and validated with `gguf==0.19.0`.

This closure does not claim:

- universal compatibility with every GGUF producer or consumer
- architectural inference readiness for an unvalidated Bharat checkpoint
- Q4_0, K-quant, IQ, F16, or BF16 export
- mixed-precision per-tensor quantization
- GPU or streaming quantization
- remote checkpoint loading

## Safety boundary

No training, model downloads, dataset downloads, benchmark downloads, external APIs, scraping, uploads, workflow artifact uploads, or remote checkpoint access are part of the production export path.

The compatibility dependency is exactly pinned and used only by deterministic local validation tests and the local validation script.

## Result

Milestone 5.4 is complete. The next roadmap work must be selected separately and must not silently introduce training, downloads, remote services, or broader quantization formats.
