# Milestone 5.3 — Safetensors Preflight CLI Integration

**Status:** Complete

## Objective

Expose the local safetensors metadata preflight validator through the existing dry-run export CLI.

## Implemented

- Adds `--safetensors-metadata-path` to `scripts/run_export_plan.py`.
- Accepts the option only with `--format safetensors`.
- Rejects remote metadata paths and normalized remote path forms.
- Builds a deterministic local checkpoint inventory for preflight validation.
- Emits stable `safetensors_preflight` JSON containing schema version, tensor count, total tensor bytes, and sorted tensor metadata.
- Preserves the existing output shape when preflight is not requested.
- Adds deterministic subprocess tests with tiny local fixtures.

## Safety boundary

This integration reads local JSON metadata and checkpoint file inventory records only. It does not parse tensor payloads, load tensors, convert checkpoints, serialize model weights, call external tools, use network access, upload files, or publish workflow artifacts. Milestone 5.3 is now complete. Both local safetensors and GGUF F32 writers are implemented, tested, and integrated into the export CLI. See [MILESTONE_5_3_EXPORT_CLOSURE.md](MILESTONE_5_3_EXPORT_CLOSURE.md) for details.
