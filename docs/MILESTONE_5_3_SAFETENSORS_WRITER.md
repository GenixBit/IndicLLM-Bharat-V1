# Milestone 5.3 — Local Safetensors Writer

**Status:** Implemented

## Objective

Implement the first real, local-only safetensors model-weight writer for Bharat checkpoints.

## Supported checkpoint input format

The writer supports the **Bharat Model Format** (used for model distribution and serving):

- A **directory** containing `config.json` and `model.pt`, where `model.pt` is a raw `state_dict()` (a flat `OrderedDict` mapping parameter names to tensors).
- A **`.pt` or `.pth` file** directly containing a state_dict.
- A **training checkpoint** (`.pt` file with a `"model"` key whose value is a state_dict).

The checkpoint is loaded with `torch.load(..., map_location="cpu", weights_only=True)` for safe, restricted deserialization. Non-dict top-level structures are rejected.

## Writer architecture

Module: `bharat/serving/safetensors_writer.py`

Public API:

```python
@dataclass(frozen=True)
class SafetensorsWriteResult:
    output_path: Path
    tensor_count: int
    bytes_written: int
    metadata: dict[str, str]

def write_safetensors_checkpoint(
    checkpoint_path: Path,
    output_path: Path,
    model_name: str | None = None,
    metadata: Mapping[str, str] | None = None,
) -> SafetensorsWriteResult: ...
```

### Execution flow

1. Reject remote paths.
2. Resolve both paths.
3. Locate the `.pt` file (directory → `model.pt`, or direct file).
4. Reject output inside checkpoint directory.
5. Reject existing output or missing output parent.
6. Load state dict with `weights_only=True`.
7. Validate state dict (non-empty, valid names, tensor values, strided layout).
8. Build deterministic metadata.
9. Prepare tensors (detach, CPU, contiguous, sort by name).
10. Write atomically via temp-file-and-rename.
11. Return `SafetensorsWriteResult`.

## State-dict extraction rules

1. If the loaded object is a `dict` and has a `"model"` key whose value is a `dict`, use `obj["model"]` (training checkpoint format).
2. Otherwise, use the loaded `dict` directly (Bharat model format or raw state dict).
3. Non-dict objects are rejected with "unsupported checkpoint structure".

## Dtype behaviour

Supported dtypes are those accepted by `safetensors.torch.save_file()` (typically `float32`, `float16`, `bfloat16`, `float64`, `int8`, `int16`, `int32`, `int64`, `uint8`, `bool`, and float8 variants where available).

Unsupported dtypes (e.g. `complex32`) or unsupported tensor layouts (e.g. sparse tensors) raise `ValueError` with a clear message.

## Metadata

Stable metadata included in every safetensors file:

| Key | Value |
|-----|-------|
| `format` | `bharat-safetensors-v1` |
| `writer_version` | `1` |
| `model_name` | caller-provided (optional) |

Reserved keys (`format`, `writer_version`) cannot be overridden by caller metadata.
All caller metadata values must be strings.
Metadata is sorted deterministically.

## Atomic-write behaviour

1. Create a uniquely named temporary file in the output directory via `tempfile.mkstemp`.
2. Write the safetensors payload to the temporary file.
3. Verify the temporary file exists and is non-empty.
4. Double-check that the output path does not exist (handles concurrent-creation race).
5. Atomically rename the temporary file to the output path via `os.rename` (atomic on Unix; raises `FileExistsError` if target exists).
6. If any step fails, remove the temporary file.

## No-overwrite guarantee

- The output path is checked for existence before writing begins.
- After writing to the temporary file, the output path is checked again before rename.
- `os.rename` on Unix raises `FileExistsError` if the target already exists.
- This provides a strong no-overwrite guarantee for practical concurrent scenarios. The fundamental limitation is that `os.rename` is atomic on the same filesystem but may not be atomic across filesystems or on all platforms; however, since the temporary file is created in the same directory as the output, cross-filesystem renaming is not required.

## Failure-cleanup behaviour

- If any step fails after temporary file creation, the temporary file is removed.
- No partial write is ever visible at the output path.
- No temporary files are left behind after failure.
- The source checkpoint is never modified.

## Safety boundaries

This writer does not:
- implement GGUF writing
- add quantization
- download models or datasets
- call external APIs
- require GPU
- accept arbitrary pickle objects
- weaken existing readiness validation
- replace existing dry-run writers
- integrate with the export CLI (deferred to next PR)

## Milestone status

Milestone 5.3 remains incomplete. The repository still does not contain a GGUF writer. The safetensors writer is not yet integrated into the export registry or CLI.

## Example library usage

```python
from pathlib import Path
from bharat.serving.safetensors_writer import write_safetensors_checkpoint

result = write_safetensors_checkpoint(
    checkpoint_path=Path("./checkpoints/bharat"),
    output_path=Path("./exports/bharat.safetensors"),
    model_name="bharat-local",
)
print(result.to_json())
```
