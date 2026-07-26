# Milestone 5.4 — Q8_0 GGUF Binary Integration

## Objective

Integrate the standalone Q8_0 quantizer (`quantize_q8_0` / `dequantize_q8_0` from
`bharat/serving/gguf_quant_writer.py`) into the GGUF serialisation pipeline so that
the repository can produce and re-read GGUF v3 files with Q8_0-quantised tensor
payloads.

## Changes

### `bharat/serving/gguf_writer.py`

- Added `GGML_TYPE_Q8_0 = 8` constant.
- Expanded `_SUPPORTED_GGML_TYPES` to `frozenset({GGML_TYPE_F32, GGML_TYPE_Q8_0})`.
- Added `_tensor_byte_size(shape, ggml_type)` — returns the on-disk byte count
  given a shape and GGML type. F32 → `total * 4`; Q8_0 → `(total // 32) * 34`,
  raising `ValueError` when the element count is not divisible by 32. The
  original `_f32_byte_size` is preserved for internal compat.
- `build_gguf_tensor_descriptors()` gains an optional `ggml_type` parameter
  (default `GGML_TYPE_F32`) that is applied uniformly to every descriptor.
- `build_gguf_header_and_descriptors()` uses the descriptor’s own `ggml_type`
  (via `_tensor_byte_size`) for the `projected_tensor_data_bytes` calculation.

### `bharat/serving/gguf_reader.py`

- Imports `GGML_TYPE_Q8_0` and defines `_GGUF_SUPPORTED_GGML_TYPES`.
- The type-check `ggml_type != GGML_TYPE_F32` was replaced with membership in
  `_GGUF_SUPPORTED_GGML_TYPES`.
- For Q8_0 tensors the reader validates that the total element count is a
  multiple of 32.
- The payload bounds check computes tensor byte size per type:
  - F32: `element_count * 4`
  - Q8_0: `(element_count // 32) * 34`

### `bharat/serving/gguf_tensor_writer.py`

- Added `_validate_q8_0_shapes()` — raises if any tensor’s numel is not a
  multiple of 32.
- `build_gguf_q8_0_payload(preflight, tensors) → bytes` — produces a complete
  GGUF v3 payload with Q8_0 quantized tensor data. Reuses the existing
  `_normalize_f32_tensors`, `quantize_q8_0`, and type-parameterised descriptor
  builders.
- `write_gguf_q8_0_tensors(preflight, tensors, output_path) → GGUFTensorWriteResult`
  — writes the payload to a `.gguf` file with atomic no-overwrite publication.

### `bharat/serving/__init__.py`

Exports `GGML_TYPE_Q8_0`, `build_gguf_q8_0_payload`, `write_gguf_q8_0_tensors`.

### Tests (`tests/serving/test_gguf_q8_0_integration.py`)

61 tests covering:
- **Constants** — `GGML_TYPE_Q8_0 == 8`, block constants
- **Descriptor construction** — Q8_0 type accepted, F32 default unchanged,
  unsupported types rejected
- **Byte-size calculation** — 1/2/3 block sizes, 2D/3D tensors, non-multiple
  rejection
- **Descriptor offsets** — alignment, ordering, duplication
- **Payload building** — determinism, offset matching, header validity,
  count/type/shape validation
- **File writing** — creation, no-overwrite, suffix/parent checks
- **Reader integration** — Q8_0 file reading, metadata, names, offsets,
  bounds/magic/suffix validation
- **Round-trip quality** — linear ramp, zeros, negative, mixed sign, multiple
  tensors, 2D; tolerance derived from Q8_0 error bound `amax / 254`
- **Backward compatibility** — F32 payload/descriptors/reader unchanged
- **Edge cases** — 1 block, large, sparse rejection, non-contiguous, counts

## Key Design Decisions

1. **Uniform ggml_type per `build_gguf_tensor_descriptors` call** — simplifies
   the API; mixed-type files can be added later when needed.
2. **Byte-size helper is type-aware** — `_tensor_byte_size` centralises the
   mapping from GGML type to on-disk size, avoiding duplication across
   writer, reader, and descriptors.
3. **Existing F32 path untouched** — no function signature changes for
   `build_gguf_f32_payload`, `write_gguf_f32_tensors`, or the reader’s
   `read_gguf_subset` (accepts both types transparently).
4. **Quantizer reused** — `quantize_q8_0` and `dequantize_q8_0` are called
   directly; no duplicate quantisation logic.

## File Inventory

| File | Lines |
|------|-------|
| `bharat/serving/gguf_writer.py` | +32 / −6 |
| `bharat/serving/gguf_reader.py` | +19 / −7 |
| `bharat/serving/gguf_tensor_writer.py` | +90 |
| `bharat/serving/__init__.py` | +5 / −0 |
| `tests/serving/test_gguf_q8_0_integration.py` | +554 (new) |
