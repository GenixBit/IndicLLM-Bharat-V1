# Milestone 5.4 — Q8_0 GGUF Independent Compatibility Validation

**Status:** Complete
**Date:** 2026-07-27
**Branch:** `test/q8-0-independent-compatibility`

## Independent Implementation

| Field | Value |
|-------|-------|
| **Project** | gguf — official GGML Python package |
| **Repository** | [github.com/ggml-org/ggml](https://github.com/ggml-org/ggml) (Python bindings published to PyPI) |
| **Version** | 0.19.0 |
| **Installation** | `pip install gguf==0.19.0` |
| **Entry point** | `gguf.GGUFReader`, `gguf.dequantize()` |
| **Supported GGUF versions** | 2, 3 |
| **Q8_0 support** | Yes: `GGMLQuantizationType.Q8_0` (value 8), `GGML_QUANT_SIZES[Q8_0] = (32, 34)` |
| **Why trusted** | Official companion package of the reference ggml/llama.cpp implementation; maintained by GGML authors |

## Setup Instructions

```bash
pip install gguf==0.19.0
```

This is the only additional dependency. All other dependencies (torch, numpy, etc.) are already present in the project.

## Fixture Description

A deterministic tiny checkpoint with four F32 tensors:

| Tensor | Shape | Elements | Blocks | Description |
|--------|-------|----------|--------|-------------|
| `tensor_32` | (32,) | 32 | 1 | Mixed positive/negative values |
| `tensor_64` | (64,) | 64 | 2 | First block all zeros (zero-block test) |
| `tensor_96` | (96,) | 96 | 3 | Oscillating values across blocks |
| `tensor_2d` | (1, 32) | 32 | 1 | Multidimensional tensor |

Values are deterministic, with no NaN or infinity.

## CLI Export Command

```bash
python scripts/run_export_plan.py \
  --checkpoint-path <checkpoint> \
  --output-path <output.gguf> \
  --format gguf \
  --model-name compatibility-fixture \
  --gguf-metadata-path <metadata.json> \
  --gguf-tensor-type q8_0 \
  --execute
```

The same command with `--gguf-tensor-type f32` produces the F32 control file.

## Q8_0 Export Result

All four exports succeed:
- Process exits with code 0
- Output file exists and is non-empty
- Result JSON reports `gguf_tensor_type: "q8_0"`
- `bytes_written` matches actual file size
- No network access (tested with `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`)
- CPU-only (no GPU required)

## Structural Results

All tensors parsed correctly by the independent `gguf` reader:

| Check | Result |
|-------|--------|
| GGUF magic (0x46554747) | Valid |
| GGUF version (3) | Supported |
| Tensor count (4) | Matches |
| Tensor names | `tensor_32`, `tensor_64`, `tensor_96`, `tensor_2d` |
| Tensor shapes | `[32]`, `[64]`, `[96]`, `[32, 1]` (raw file order) |
| Tensor type | All `GGMLQuantizationType.Q8_0` (8) |
| Payload sizes | Positive, offsets within file |
| Alignment | Power of two (32 or 256) |
| End-to-end parse | Succeeds without error |

## Numerical Results

Dequantization via `gguf.dequantize()` compared against source F32 values:

| Tensor | Max Error | Mean Error | RMSE | Cosine Similarity |
|--------|-----------|------------|------|-------------------|
| `tensor_32` | < 1.0 | < 0.5 | < 0.5 | > 0.99 |
| `tensor_64` | < 1.0 | < 0.5 | < 0.5 | > 0.99 |
| `tensor_96` | < 1.0 | < 0.5 | < 0.5 | > 0.99 |
| `tensor_2d` | < 1.0 | < 0.5 | < 0.5 | > 0.99 |

- Max absolute error: < 1.0 for all tensors.
- Cosine similarity: > 0.99 for all tensors (except all-zero block where it is 1.0).

Thresholds defined in `docs/GGUF_QUANTIZATION_ARCHITECTURE.md` are satisfied.

## Byte-Level Block Results

For the first block of `tensor_32`:
- Scale (float16 LE): correctly parsed from first 2 bytes
- 32 quantized values (int8): correctly extracted
- Dequantization formula `value = quant * scale` verified against `gguf.dequantize()`
- Scale encoding: little-endian float16 verified
- Quantized values: signed int8 verified

## F32 Control Results

| Check | Result |
|-------|--------|
| F32 GGUF through gguf reader | Parses successfully |
| Tensor count (4) | Matches |
| Tensor types | All `GGMLQuantizationType.F32` (0) |
| CL defaults still produce F32 | Default f32, explicit required for q8_0 |

## Size Comparison

Q8_0 file is smaller than F32 file for the same checkpoint.

## Determinism

Two exports with identical inputs produce byte-identical output files.

## Corruption-Test Results

| Test | Result |
|------|--------|
| Truncated Q8_0 payload | Rejected by gguf reader |
| Corrupted tensor data byte | Dequantized values differ from original |
| Corrupted type identifier (8 → 99) | Rejected by gguf reader |
| Malformed magic (GGUF → BADD) | Rejected by gguf reader |

## Offline and CPU Results

All tests pass with:
- `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`, `WANDB_MODE=disabled`, `TOKENIZERS_PARALLELISM=false`
- No model download, dataset download, or external API calls
- CPU-only execution
- Tiny temporary files (cleaned up by pytest)

## Compatibility Limitations

1. **Multidimensional tensors**: The gguf reader requires the last stored dimension (fastest-varying in memory) to be a multiple of the Q8_0 block size (32). Tensors whose last original dimension < 32 are read correctly only by the repository-local reader. This is consistent with how real LLM weight tensors are shaped (e.g., `(4096, 4096)`, `(11008, 4096)`) where the last dimension is always ≥ 32.

2. **Corrupted offset detection**: The gguf reader uses lazy `np.memmap` and does not validate tensor offsets during initialization. An invalid offset is only detected when tensor data is accessed, and the behavior depends on the OS memory-mapping implementation.

3. **Incomplete final block**: Detected during tensor info parsing (element count must be a multiple of block size).

4. **Scope**: This validation covers Q8_0 only. Q4_0, K-quants, IQ formats, and F16/BF16 are not tested.

5. **No architectural inference**: This test validates file format compatibility only, not inference correctness.

## Conclusion

Validated with gguf v0.19.0 (official GGML Python package) for the tested repository-generated Q8_0 GGUF subset.

**Not claimed:**
- Universal GGUF compatibility with every llama.cpp version
- Architecture-complete inference compatibility
- Support for Q4_0, K-quants, or other formats

## Acceptance Criteria Review

| # | Criterion | Status |
|---|-----------|--------|
| 1 | Q8_0 writer produces deterministic output | ✅ Byte-identical repeated exports |
| 2 | Q8_0 round-trip fidelity | ✅ Max error < 1.0, cosine sim > 0.99 |
| 3 | Reader correctly parses Q8_0 descriptors | ✅ All names, shapes, types match |
| 4 | Reader rejects unsupported types | ✅ Validated via independent reader |
| 5 | CLI `--gguf-tensor-type q8_0` writes valid GGUF | ✅ gguf reader parses successfully |
| 6 | CLI default f32 equals existing f32 path | ✅ F32 control parses |
| 7 | Unsupported values rejected before tensor loading | ✅ Validated in unit tests |
| 8 | All existing F32 export tests pass | ✅ 1850+ tests pass |
| 9 | No overwrite, atomic publication, temp cleanup | ✅ Existing tests verify |
| 10 | `torch.load` uses `weights_only=True` | ✅ Existing grep confirms |
| 11 | Dry-run is default | ✅ CLI test confirms |
| 12 | GGUF v3 compliance | ✅ Magic, version, alignment validated |
| 13 | Lint + typecheck clean | ✅ ruff/mypy pass |
| 14 | `general.quantization_version` metadata | ✅ Reader validates |
| 15 | Independent Q8_0 compatibility | ✅ Validated with gguf v0.19.0 (this document) |

**Milestone 5.4 Decision: READY TO CLOSE**

All acceptance criteria are met. Independent structural validation, numeric validation, and byte-level validation all succeed.
