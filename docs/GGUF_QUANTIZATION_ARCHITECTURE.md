# GGUF Quantization Architecture

**Status:** Draft — Planning / Milestone 5.4  
**Date:** 2026-07-27  
**Branch:** `docs/gguf-quantization-architecture`

## 1. Objective

Add deterministic, local, CPU-only quantization to the existing GGUF F32 export pipeline. All existing safety guarantees (`weights_only=True`, `map_location="cpu"`, no overwrite, atomic publication, temp cleanup, no network) must be preserved.

### Guiding principles

1. **Quantization is explicit** — dry-run remains default; quantization requires `--quantize {none,q8_0}` (or similar). Unsupported formats fail before checkpoint loading.
2. **First format: Q8_0** — the smallest, safest boundary. Q4_0 and all other formats are deferred.
3. **No K-quants, no IQ variants** — deferred until after Q4_0.
4. **Deterministic** — same input + same format = identical binary output on any platform.
5. **No architecture awareness** — quantization operates per-tensor on the flat float32 buffer without inspecting tensor roles. Attention, embedding, output, norm tensors are all quantized identically (no mixed-precision schemes in the first slice).

## 2. Current State (Milestone 5.3)

| Component | File | Role |
|-----------|------|------|
| GGUF writer | `bharat/serving/gguf_writer.py` | GGUF v3 header + tensor descriptors + scalar metadata. F32-only (`GGML_TYPE_F32 = 0`; `_SUPPORTED_GGML_TYPES = frozenset({GGML_TYPE_F32})`). |
| F32 tensor writer | `bharat/serving/gguf_tensor_writer.py` | `build_gguf_f32_payload()` — serialises `torch.float32` tensors via `struct.pack("<f", ...)`. |
| GGUF reader | `bharat/serving/gguf_reader.py` | Validates and reads the F32-only subset. Rejects `ggml_type != GGML_TYPE_F32`. |
| Export writer registry | `bharat/serving/export_writer.py` | `LocalGGUFF32ExportWriter` — `_load_f32_state_dict()` → `write_gguf_f32_tensors()`. |
| CLI | `scripts/run_export_plan.py` | `--format {safetensors,gguf}`; quantization does not exist as a parameter. |
| Preflight | `bharat/serving/gguf_preflight.py` | Validates metadata JSON; no quantization field exists. |

### Limits (per-closure doc)

- Quantized GGUF (Q4_0, Q8_0, etc.) — out of scope
- F16/BF16 GGUF tensors — rejected
- Only `torch.float32` accepted as input

## 3. Authoritative Sources

### 3.1 GGUF specification

Source: [ggml-org/ggml — gguf.md](https://raw.githubusercontent.com/ggml-org/ggml/master/docs/gguf.md)

Key types:

```
GGML_TYPE_F32  = 0
GGML_TYPE_F16  = 1
GGML_TYPE_Q4_0 = 2
GGML_TYPE_Q8_0 = 8
```

Tensor descriptor stores `ggml_type` (uint32). The type determines both the block size and the dequantisation formula used by the inference engine.

### 3.2 ggml-quants.c reference implementation

Source: [ggml-org/ggml — ggml-quants.c](https://raw.githubusercontent.com/ggml-org/ggml/master/src/ggml-quants.c)

#### Q8_0 block structure (`block_q8_0`)

```c
#define QK8_0 32

typedef struct {
    ggml_fp16_t d;           // scale (float16)
    int8_t      qs[QK8_0];  // quants (32 × int8)
} block_q8_0;
```

- **Block size:** 32 elements
- **Block bytes:** `sizeof(ggml_fp16_t) + 32 × sizeof(int8_t)` = 2 + 32 = 34 bytes
- **Compression ratio:** 32 × 4 = 128 bytes (F32) → 34 bytes = 3.76× (73.4% reduction)

#### Quantization formula (`quantize_row_q8_0_ref`)

```c
float amax = 0.0f;
for (int j = 0; j < QK8_0; j++) {
    amax = MAX(amax, fabsf(x[j]));
}
const float d  = amax / ((1 << 7) - 1);   // d = amax / 127
const float id = d ? 1.0f / d : 0.0f;

y[i].d = GGML_FP32_TO_FP16(d);

for (int j = 0; j < QK8_0; j++) {
    y[i].qs[j] = roundf(x[j] * id);
}
```

#### Dequantization formula (`dequantize_row_q8_0`)

```c
const float d = GGML_FP16_TO_FP32(x[i].d);
for (int j = 0; j < QK8_0; j++) {
    y[j] = x[i].qs[j] * d;
}
```

## 4. GGML Type Table (Relevant Subset)

| Name | Enum | Block size | Block bytes | Ratio vs F32 |
|------|------|-----------|-------------|--------------|
| F32  | 0    | 1         | 4           | 1.0×         |
| F16  | 1    | 1         | 2           | 2.0×         |
| Q4_0 | 2    | 32        | 18          | 7.1×         |
| Q8_0 | 8    | 32        | 34          | 3.76×        |

**First implementation target: Q8_0** (type 8). Lowest complexity, integer-only quantisation with a single float16 scale per block.

## 5. Block Layout (Q8_0)

```
Block:   [d: fp16][qs[0]: int8][qs[1]: int8] ... [qs[31]: int8]
Offset:  0        2          3                  33
Size:    34 bytes total
```

- Tensors are padded at the end to `ALIGNMENT` (256 bytes) — same as F32 path.
- Within a tensor, blocks are contiguous (no intra-tensor alignment).
- The total tensor byte size is: `ceil(element_count / 32) × 34` padded to alignment.

## 6. Tensor-Selection Policy

**Every tensor is quantized.** The first slice applies Q8_0 uniformly to all tensors. No mixed-precision, no architecture-specific handling.

Rationale:
- Simplicity of implementation and testing.
- The reader/consumer decides which tensors need higher precision. When mixed-precision is added later, the metadata will record per-tensor types.
- Inference engines (llama.cpp) can handle uniform Q8_0 for all tensors without special cases.

## 7. Numeric Thresholds and Edge Cases

| Condition | Behaviour |
|-----------|-----------|
| Zero block (`amax == 0`) | `d = 0`, all `qs[j] = 0` |
| All-equal positive block | `amax = value`, `d = value/127`, all `qs[j] = 127` |
| All-equal negative block | `amax = |value|`, `d = |value|/127`, all `qs[j] = -128` |
| Mixed signs | Standard round-to-nearest with ties going away from zero (`roundf`) |
| NaN in input | Rejected at tensor-normalisation layer |
| Inf in input | Rejected at tensor-normalisation layer |
| Empty tensor | Rejected (must have at least one dimension, all dimensions positive) |

## 8. Determinism

The reference `quantize_row_q8_0_ref` uses `roundf` and a reduction over a fixed 32-element block, which is deterministic across Python invocations on the same platform. To guarantee cross-platform determinism:

- `roundf` rounds halfway cases away from zero (IEEE 754-2008).
- Floating-point addition order in the amax reduction is sequential (no parallel reduction) — same order as the C reference.
- No parallelism, no atomics, no non-deterministic floating-point contractions.

**Python implementation** will replicate the C algorithm exactly: iterate blocks, compute max, compute scale, quantise.

## 9. Memory Design

- Input tensors are loaded via `torch.load(weights_only=True, map_location="cpu")`.
- Each tensor is quantised block-by-block into a pre-allocated `bytearray`.
- The quantised payload is written to a temp file via the same atomic `os.link()` path as F32.
- Peak memory: one F32 tensor at a time (~4 bytes/element) plus one quantized tensor at a time (~34/32 = 1.0625 bytes/element).
- No streaming to disk during quantization (the entire quantised payload is built in memory then written). This is acceptable for the target model sizes (350M–1B parameters; ~1.4–4 GB F32, ~370 MB–1.1 GB Q8_0).

## 10. CPU-Only Design

- `torch.load(..., map_location="cpu")` — enforced by the existing `_load_f32_state_dict`.
- All quantization arithmetic is pure Python (no `torch` ops needed for the block quantisation itself). This avoids GPU dependencies and cuBLAS/cuDNN path differences.
- No `torch.Tensor` quantization functions are used (no `torch.quantize_per_tensor`, no `torch.int8` conversion).

## 11. Writer Integration

### New module: `bharat/serving/gguf_quant_writer.py`

```
GGUFQuantTensorWriter
├── quantize_q8_0(tensor: torch.Tensor) -> bytes
├── build_gguf_quant_payload(preflight, tensors, ggml_type) -> bytes
└── write_gguf_quant_tensors(preflight, tensors, output_path, ggml_type) -> GGUFQuantTensorWriteResult
```

### Changes to existing modules

#### `bharat/serving/gguf_writer.py`

- Expand `_SUPPORTED_GGML_TYPES` to include `GGML_TYPE_Q8_0`.
- `GGUFTensorInventoryEntry` — unchanged (shape only, no type).
- `build_gguf_tensor_descriptors()` — needs a `ggml_type` parameter (default `GGML_TYPE_F32`). Currently hardcodes `GGML_TYPE_F32`. Change to accept an optional `ggml_type` that applies to all tensors (uniform quantization). When `ggml_type != GGML_TYPE_F32`, compute block-aligned byte sizes instead of `_f32_byte_size`.

#### `bharat/serving/gguf_tensor_writer.py`

- No changes needed (remains F32-only). The quant writer is a separate module.

#### `bharat/serving/export_writer.py`

- Add `LocalGGUFQ8_0ExportWriter` — analogous to `LocalGGUFF32ExportWriter` but calls `gguf_quant_writer.write_gguf_quant_tensors(..., ggml_type=GGML_TYPE_Q8_0)`.
- Register it in `ExportWriterRegistry` when the preflight specifies `quantization="q8_0"`.

#### `bharat/serving/gguf_preflight.py`

- Add `quantization: str | None` field to `GGUFPreflightResult`.
- Validate: must be `None`, `"none"`, `"f32"`, or `"q8_0"`; other values rejected.
- The metadata JSON gains an optional `"quantization"` field.

#### `bharat/serving/gguf_reader.py`

- Expand `_SUPPORTED_GGML_TYPES` in reader validation to include `GGML_TYPE_Q8_0`.
- Adjust per-element byte-size calculation: `_element_byte_size(ggml_type)` — 4 for F32, 34/32 = 1.0625 for Q8_0.
- Validate block-aligned offset/total-size for quantized tensors.

## 12. CLI Design

### `scripts/run_export_plan.py`

New argument:

```
--quantize {none,f32,q8_0}    Quantization scheme for GGUF tensors (default: none)
```

- `--quantize f32` — equivalent to current F32-only behaviour.
- `--quantize none` — same as `f32` (dry-run default).
- `--quantize q8_0` — enable Q8_0 quantization.
- Requires `--format gguf`.
- Must be consistent with `--gguf-metadata-path` metadata (if present, the `quantization` field in metadata must match).

Validation flow with quantization:

1. Parse CLI arguments.
2. If `--quantize` set to anything other than `none`/`f32`, verify `--format gguf` (error if safetensors).
3. Preflight: validate metadata JSON; extract `quantization` field.
4. Writer selection: `quantization` determines which `ExportWriter` is selected from the registry.
5. Tensor loading + quantization (if applicable).
6. Atomic write + manifest.

## 13. Manifest Design

The existing `ExportManifest` records `export_format` and `writer_name`. With quantization, the manifest should also record:

```
"quantization": "q8_0" | "f32" | null
```

Add optional `quantization` field to `ExportWriteResult`.

## 14. Compatibility Levels

| Reader scenario | Result |
|----------------|--------|
| Old reader, old file (F32) | ✅ Works (backward compatible) |
| Old reader, Q8_0 file | ❌ Fails at `ggml_type != GGML_TYPE_F32` check — clear error |
| New reader, old file (F32) | ✅ Works |
| New reader, Q8_0 file | ✅ Works |
| llama.cpp loading Q8_0 | ✅ Works (Q8_0 is a standard ggml type) |
| llama.cpp loading F32 | ✅ Works (no change) |

## 15. Failure Behaviour

| Failure mode | Handling |
|-------------|----------|
| Unsupported quantization value | Error before tensor loading |
| NaN/Inf in tensor | Error at tensor-normalisation layer |
| Quantisation overflow (block) | Clamp to int8 range [-128, 127] via `roundf` |
| Zero-length tensor | Error (existing validation) |
| Mismatched preflight/CLI quantization | Error before tensor loading |
| Disk full during write | Temp file cleanup via existing `try/finally` |
| Concurrent write to output | Existing `FileExistsError` via `os.link()` |
| Non-GGUF output extension | Error (existing validation) |

## 16. Security Boundaries

| Boundary | Enforcement |
|----------|------------|
| Checkpoint loading | `torch.load(weights_only=True, map_location="cpu")` |
| Remote URLs | Rejected at CLI and writer layers (existing) |
| Output path overwrite | Rejected by `os.link()` atomic publish (existing) |
| Temp file leakage | Cleaned up in `finally` (existing) |
| Manifest overwrite | Rejected (existing) |
| Arbitrary code in metadata | Metadata is parsed JSON, no `eval`/`exec` |
| Integer overflow in size calc | Python big integers; validated against uint64 bounds |

## 17. Phased PR Plan

### PR #50 — Architecture document (this PR)
- `docs/GGUF_QUANTIZATION_ARCHITECTURE.md`
- `docs/ROADMAP.md` updated

### PR #51 — Q8_0 quant writer + preflight + reader changes
- `bharat/serving/gguf_quant_writer.py` — `quantize_q8_0`, `build_gguf_quant_payload`, `write_gguf_quant_tensors`
- `bharat/serving/gguf_writer.py` — expand `_SUPPORTED_GGML_TYPES`, parameterise `build_gguf_tensor_descriptors()` with `ggml_type`
- `bharat/serving/gguf_preflight.py` — add `quantization` field
- `bharat/serving/gguf_reader.py` — support `GGML_TYPE_Q8_0`
- Full test suite for Q8_0 round-trip (quantise → write → read → dequantise → compare)

### PR #52 — Writer registry + CLI integration
- `bharat/serving/export_writer.py` — `LocalGGUFQ8_0ExportWriter`, registry update
- `scripts/run_export_plan.py` — `--quantize` argument
- `bharat/serving/export.py` / `ExportWriteResult` — `quantization` field
- End-to-end CLI tests

### PR #53 — Q4_0 format (deferred, future)
- Block structure: `block_q4_0` (18 bytes/32 elements)
- Required for meaningful size comparisons
- Not part of this milestone

## 18. Acceptance Criteria

| # | Criterion | Verification |
|---|-----------|-------------|
| 1 | Q8_0 writer produces deterministic output | SHA-256 of output identical across 3 runs |
| 2 | Q8_0 round-trip fidelity | max relative error < 1% for uniform random data; < 0.5% for realistic weight distributions |
| 3 | Reader correctly parses Q8_0 descriptors | tensor names, shapes, types, offsets match preflight |
| 4 | Reader rejects F32-only files with Q8_0 reader path | error message includes type mismatch |
| 5 | CLI `--quantize q8_0` writes valid GGUF | file loads in `read_gguf_subset`; tensor types are Q8_0 |
| 6 | CLI `--quantize none` is equivalent to current F32 path | output identical to existing writer for same input |
| 7 | Unsupported quantization values rejected before tensor loading | error on `--quantize q4_0` (not yet implemented) |
| 8 | All existing F32 export tests continue to pass | 168+ export tests |
| 9 | No overwrite, atomic publication, temp cleanup | existing tests verify |
| 10 | All `torch.load` calls use `weights_only=True` | grep confirms |
| 11 | Dry-run is default; `--execute` required for real write | CLI test confirms |
| 12 | GGUF v3 compliance | output conforms to spec (magic, version, alignment, zero padding) |
| 13 | Lint (ruff) + typecheck (mypy) clean | CI |
| 14 | `general.quantization_version` metadata present | reader validates |

## 19. Deferred Work

| Feature | Reason |
|---------|--------|
| Q4_0 quantization | Higher complexity (non-uniform quantisation, asymmetric). Separate PR. |
| K-quants (Q4_K, Q5_K, Q6_K, Q8_K) | Super-block structure requires architecture-aware importance weighting. Separate milestone. |
| IQ formats (IQ2_XXS, IQ1_S, etc.) | Non-standard, limited inference engine support. Not planned. |
| Mixed-precision per-tensor | Requires tensor-role metadata. Future. |
| GPU quantization | Out of scope (CPU-only design). |
| Streaming quantisation | Memory optimisation for >1B models. Future. |
| Quantisation-aware metric reporting | PSNR, SQNR per tensor. Future. |
| F16/BF16 tensor support | Different format; not block-quantization. Future. |

## 20. Glossary

| Term | Definition |
|------|-----------|
| Block | The atomic unit of quantisation. For Q8_0, one block contains 32 float32 values, quantised to 32 int8 values plus one float16 scale. |
| Block size (`QK`) | Number of float32 elements per block. Q8_0 = 32, Q4_0 = 32. |
| Scale (`d`) | Float16 value per block that maps quantised integers back to approximate float32 range. |
| GGML type | Integer enum identifying the tensor data format (F32=0, F16=1, Q4_0=2, Q8_0=8). |
| Round-trip | Quantise → dequantise → compare against original. Used to measure precision loss. |
| K-quant | A family of quantisation formats (Q2_K through Q6_K, Q8_K) that use a super-block structure with sub-blocks sharing scale factors. Named for the `QK_K` block size. |
| IQ | Importance-weighted quantisation formats from `ggml` that allocate more bits to important weights. Non-deterministic by nature. |
