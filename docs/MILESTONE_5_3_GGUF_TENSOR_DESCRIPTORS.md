# Milestone 5.3 — GGUF Tensor Descriptor Encoding

**Status:** Implemented (descriptors only — no tensor payload bytes)

## Objective

Encode deterministic GGUF v3 tensor descriptors and calculate aligned data
offsets without writing actual tensor payload bytes.  Only unquantized F32
tensor descriptors are supported.

## GGUF specification source

- [ggml/docs/gguf.md](https://github.com/ggml-org/ggml/blob/master/docs/gguf.md)
- [gguf.h reference](https://github.com/ggml-org/ggml/blob/master/include/gguf.h)
- GGUF v3, little-endian, default alignment 32 bytes

## Verified descriptor binary layout

Each tensor descriptor is encoded sequentially as follows:

| Field | Width | Value |
|-------|-------|-------|
| name_length | uint64 | byte count of UTF-8 name |
| name_data | uint8[name_length] | UTF-8 tensor name |
| n_dimensions | uint32 | rank (number of shape dimensions) |
| dimensions | uint64[n_dimensions] | shape, **reversed** (column-major) |
| ggml_type | uint32 | GGML type enum identifier |
| offset | uint64 | offset relative to tensor-data section start |

Dimensions are stored in **reverse order** (GGML column-major convention).
A shape `[M, N]` is stored as `[N, M]`.

## GGML type identifiers

| Name | Value | Description |
|------|-------|-------------|
| `GGML_TYPE_F32` | 0 | 32-bit float (only supported type) |

## F32-only scope

Only `GGML_TYPE_F32` (value `0`) is accepted.  All other GGML types are
rejected deterministically.  This PR does not implement F16, BF16, or any
quantized types.

## Descriptor data model

```python
@dataclass(frozen=True)
class GGUFTensorInventoryEntry:
    name: str
    shape: tuple[int, ...]

@dataclass(frozen=True)
class GGUFTensorDescriptor:
    name: str
    shape: tuple[int, ...]
    ggml_type: int
    offset: int
```

## Validation rules

- Name must be non-empty.
- Shape must be non-empty.
- Each dimension must be a positive integer (bool rejected).
- Rank must fit in uint32.
- GGML type must be one of the supported values (only F32).
- Offset must be non-negative and fit within uint64.
- Duplicate tensor names are rejected.

## F32 byte-size calculation

```
tensor_byte_size = prod(shape) * 4
```

Calculated using Python integer arithmetic.  No explicit uint64 overflow
check is needed because Python handles arbitrary-precision integers and
the practical shape sizes used in models are well within range.

## Sorting rules

- Tensor descriptors are sorted deterministically by name before encoding.
- Input insertion order does not affect descriptor order, offsets, emitted
  bytes, or result JSON.

## Offset calculation

- The first tensor offset is always `0`.
- Each subsequent tensor offset is the aligned end of the previous tensor's
  payload.
- Payload sizes are calculated as the F32 byte size described above.
- Offsets are relative to the start of the tensor-data section (the aligned
  boundary after all descriptors).

## Alignment calculation

```
GGML_PAD(x, a) = ((x + a - 1) // a) * a
```

- Alignment is taken from the GGUF preflight result (typically 32 bytes).
- Non-power-of-two alignment values are rejected.
- After encoding all descriptors, padding (`0x00` bytes) is appended to
  reach the next alignment boundary.  This padded position is the start of
  the tensor-data section.

## Descriptor-only output limitation

The file written by `write_gguf_header_and_descriptors()` contains:

1. Magic bytes (`GGUF`)
2. Version (`3`, uint32 LE)
3. Tensor count (uint64 LE)
4. Metadata count (uint64 LE)
5. Metadata entries (key + type + value)
6. Tensor descriptors (name + rank + dimensions + type + offset)
7. Alignment padding to tensor-data section start

**No tensor payload bytes are written.**  The file is **not** a complete
runnable GGUF model.  The result object reports `descriptor_only=True`.

## API

### `build_gguf_tensor_descriptors(tensors, *, alignment)`

Build sorted tensor descriptors with computed offsets.

```python
descs = build_gguf_tensor_descriptors(
    [
        GGUFTensorInventoryEntry(name="weight", shape=(4096, 4096)),
        GGUFTensorInventoryEntry(name="bias", shape=(4096,)),
    ],
    alignment=32,
)
```

### `build_gguf_header_and_descriptors(preflight, descriptors)`

Build deterministic GGUF v3 bytes including header, metadata, descriptors,
and alignment padding up to the tensor-data section.

### `write_gguf_header_and_descriptors(preflight, descriptors, output_path)`

Write a local descriptor-only `.gguf` file using the same atomic no-overwrite
publication as the existing header writer.

## Result model

```python
@dataclass(frozen=True)
class GGUFDescriptorResult:
    output_path: Path
    bytes_written: int
    metadata_count: int
    tensor_count: int
    descriptor_count: int
    alignment: int
    tensor_data_start_offset: int
    projected_tensor_data_bytes: int
    descriptor_only: bool = True
```

## Zero-tensor compatibility

- `tensor_count == 0` with empty descriptors produces output identical to
  the existing metadata-only `build_gguf_header()`.
- Nonzero `tensor_count` with empty descriptors (or mismatched count) is
  rejected.

## No-overwrite behavior

Shared with the existing `write_gguf_header()`:
- `.gguf` suffix enforcement.
- Parent directory must exist.
- Existing output is rejected.
- Same-directory temporary creation.
- `fsync` before publication.
- Atomic hard-link publication via `os.link`.
- Temporary file cleanup on success and failure.

## Safety boundaries

This PR does **not**:

- Call `torch.load` or load model tensor payloads.
- Write tensor payload bytes.
- Write quantized tensors.
- Perform F16/BF16 conversion.
- Implement Q4, Q5, Q8, or other GGML quantization.
- Invoke llama.cpp or external tools.
- Use subprocesses or network access.
- Integrate `--execute` GGUF into the CLI.
- Claim the resulting file is a complete runnable GGUF model.

## Examples

### Single tensor

```python
from bharat.serving.gguf_writer import (
    GGML_TYPE_F32,
    GGUFTensorInventoryEntry,
    build_gguf_tensor_descriptors,
    build_gguf_header_and_descriptors,
    write_gguf_header_and_descriptors,
)

entry = GGUFTensorInventoryEntry(name="weight", shape=(64, 64))
descriptors = build_gguf_tensor_descriptors([entry], alignment=32)

preflight = GGUFPreflightResult(
    schema_version=1, architecture="bharat",
    alignment=32, tensor_count=1,
    output_file="model.gguf",
    metadata=(GGUFMetadataEntry("general.name", "string", "Bharat"),),
)

result = write_gguf_header_and_descriptors(preflight, descriptors, Path("model.gguf"))
print(result.to_dict())
# {
#   "descriptor_only": True,
#   "projected_tensor_data_bytes": 16384,  # 64*64*4 = 16384
#   "tensor_count": 1,
#   ...
# }
```

### Multiple tensors

```python
tensors = [
    GGUFTensorInventoryEntry(name="bias", shape=(4096,)),
    GGUFTensorInventoryEntry(name="weight", shape=(4096, 4096)),
]
descriptors = build_gguf_tensor_descriptors(tensors, alignment=32)
# descriptors[0].name == "bias", offset == 0
# descriptors[1].name == "weight", offset == align(4096*4, 32) == 16384
```

## Milestone status

Milestone 5.3 remains incomplete.  GGUF tensor payload serialization is
not implemented.  GGUF real CLI execution is not implemented.
