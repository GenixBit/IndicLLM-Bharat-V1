# Milestone 5.3 — Export CLI (Foundation Extension)

**Status:** Dry-run and real export CLI implemented for safetensors and GGUF F32

## Objective

Add a local CLI that validates and runs an export plan using the
existing export planning and writer contract APIs.

## CLI

`scripts/run_export_plan.py`

### Usage

#### GGUF dry-run (default)

```bash
python scripts/run_export_plan.py \
  --checkpoint-path checkpoints/bharat \
  --output-path exports/bharat.gguf \
  --format gguf \
  --model-name bharat-local
```

#### GGUF real execution

```bash
python scripts/run_export_plan.py \
  --checkpoint-path checkpoints/bharat \
  --output-path exports/bharat.gguf \
  --format gguf \
  --model-name bharat-local \
  --gguf-metadata-path metadata.json \
  --execute
```

#### GGUF real execution with manifest

```bash
python scripts/run_export_plan.py \
  --checkpoint-path checkpoints/bharat \
  --output-path exports/bharat.gguf \
  --format gguf \
  --model-name bharat-local \
  --gguf-metadata-path metadata.json \
  --manifest-path manifest.json \
  --execute
```

#### GGUF real execution with inventory

```bash
python scripts/run_export_plan.py \
  --checkpoint-path checkpoints/bharat \
  --output-path exports/bharat.gguf \
  --format gguf \
  --model-name bharat-local \
  --gguf-metadata-path metadata.json \
  --include-inventory \
  --execute
```

#### Safetensors dry-run (default)

```bash
python scripts/run_export_plan.py \
  --checkpoint-path checkpoints/bharat \
  --output-path exports/bharat.safetensors \
  --format safetensors \
  --model-name bharat-local
```

#### Safetensors real execution

```bash
python scripts/run_export_plan.py \
  --checkpoint-path checkpoints/bharat \
  --output-path exports/bharat.safetensors \
  --format safetensors \
  --model-name bharat-local \
  --execute
```

### Output (stdout, JSON)

#### Dry-run output

```json
{
  "checkpoint_path": "checkpoints/bharat",
  "output_path": "exports/bharat.safetensors",
  "export_format": "safetensors",
  "model_name": "bharat-local",
  "dry_run": true,
  "writer_name": "safetensors-dry-run",
  "bytes_written": 0
}
```

#### Real GGUF execution output

```json
{
  "checkpoint_path": "checkpoints/bharat",
  "output_path": "exports/bharat.gguf",
  "export_format": "gguf",
  "model_name": "bharat-local",
  "dry_run": false,
  "writer_name": "gguf-f32-local",
  "bytes_written": 123456,
  "export_path_readiness": { ... },
  "gguf_preflight": { ... },
  "writer_readiness": { ... },
  "manifest_path": "manifest.json",
  "manifest_schema_version": "1.0"
}
```

### Validation

- Rejects remote checkpoint and output paths (`http://`, `https://`,
  `ftp://`, `s3://`, `gs://` and their Path-normalized `:/` forms)
- Rejects wrong file suffix for the chosen format
- Rejects empty model names
- Rejects missing required arguments
- Rejects `--execute --format gguf` without `--gguf-metadata-path`
- Rejects `--safetensors-metadata-path` without `--format safetensors`
- Rejects `--gguf-metadata-path` without `--format gguf`
- Rejects remote metadata paths

### Exit codes

| Exit code | Meaning                         |
|-----------|---------------------------------|
| 0         | Plan validated, export written  |
| 1         | Validation error or runtime error |

## Real GGUF execution requirements

- `--format gguf` and `--execute` must be provided together
- `--gguf-metadata-path` is required and must point to a local valid JSON file
- The metadata file must pass GGUF preflight validation against the checkpoint inventory
- Only `torch.float32` tensors are supported — F16, BF16, FP64, integer, bool, and quantized tensors are rejected
- Checkpoints must be local `.pt` or `.pth` files or directories containing `model.pt`
- Tensor count in the preflight metadata must match the actual checkpoint tensor count
- No network access occurs — all operations are offline

## Execution order

For real GGUF execution, the CLI follows this order:

1. CLI argument validation
2. Remote path rejection
3. Format-specific argument compatibility checks
4. `ExportRequest` construction
5. `ExportPlan` construction
6. Export path readiness validation (when manifest or metadata paths are provided)
7. Checkpoint inventory build (auto-enabled for execution)
8. GGUF metadata preflight validation
9. Tensor-count and inventory consistency validation
10. Writer readiness validation
11. Manifest readiness validation (when `--manifest-path` is provided)
12. Registry construction with validated GGUF preflight
13. Real GGUF F32 writer invocation
14. Output existence and size verification
15. Manifest creation (when requested)
16. Final JSON emission

No checkpoint tensor loading occurs before path, metadata, inventory, and readiness validation complete.

## Registry integration

The ExportWriterRegistry is constructed with the validated GGUFPreflightResult:

```python
registry = ExportWriterRegistry(gguf_preflight=gguf_preflight)
result = registry.write(plan)
```

The registry selects the real GGUF F32 writer only when:
- `dry_run=False`
- A validated `GGUFPreflightResult` is supplied

## Manifest behavior

- The manifest is created only after the GGUF model file is successfully written
- If manifest writing fails after successful model output, the CLI returns non-zero and emits an error explaining the transaction boundary — the valid model output remains
- No partial or false success JSON is emitted on any failure

## Failure guarantees

- On failure before model writing: no output file, no manifest, no success JSON
- On writer failure: no final output, no manifest, temporary files cleaned, pre-existing files preserved
- On manifest failure after successful model writing: valid model output may remain, no partial success JSON, error clearly explains the boundary
- No dry-run fallback occurs on failure

## No-overwrite guarantees

- Existing output files are never overwritten
- Concurrent-destination protection via atomic `os.link` publication
- Temporary files in the same directory are cleaned up on success and failure
- No `--force` or overwrite support

## Safety boundary

Model weights are loaded, converted, serialized, and written only when
`--execute` is explicitly provided. All operations are offline, CPU-only,
and deterministic.

No training, fine-tuning, downloads, APIs, scraping, uploads, GGUF tools,
subprocesses, or network access occurs.

Quantization remains unsupported in this milestone.
