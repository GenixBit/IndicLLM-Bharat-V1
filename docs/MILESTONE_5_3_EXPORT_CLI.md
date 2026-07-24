# Milestone 5.3 — Export CLI (Foundation Extension)

**Status:** Dry-run CLI implemented (no real weight serialization)

## Objective

Add a local CLI that validates and dry-runs an export plan using the
existing export planning and writer contract APIs.

## CLI

`scripts/run_export_plan.py`

### Usage

```bash
python scripts/run_export_plan.py \
  --checkpoint-path checkpoints/bharat \
  --output-path exports/bharat.safetensors \
  --format safetensors \
  --model-name bharat-local
```

### Output (stdout, JSON)

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

### Validation

- Rejects remote checkpoint and output paths (`http://`, `https://`,
  `ftp://`, `s3://`, `gs://` and their Path-normalized `:/` forms)
- Rejects wrong file suffix for the chosen format
- Rejects empty model names
- Rejects missing required arguments

### Exit codes

| Exit code | Meaning                         |
|-----------|---------------------------------|
| 0         | Plan validated, dry-run written |
| 1         | Validation error or runtime error |

## Safety boundary

No model weights are loaded, converted, serialized, or written.
All output is deterministic and offline.
