# Milestone 5.3 — Local GGUF Execute Writer

**Status:** Integrated — registry selection, CLI wiring, and tests complete

## Objective

Connect the existing deterministic GGUF F32 payload writer to the export writer registry so an explicitly requested non-dry-run plan can serialize a local PyTorch state dict.

## Supported behavior

- local `.pt` or `.pth` checkpoint files
- checkpoint directories containing `model.pt`
- `torch.load(..., weights_only=True, map_location="cpu")`
- plain state dictionaries or dictionaries containing a `model` state dictionary
- prevalidated GGUF metadata supplied to the writer registry
- dense, unquantized `torch.float32` tensors only
- deterministic tensor-name ordering inherited from the GGUF payload writer
- atomic no-overwrite publication and temporary-file cleanup
- CLI integration via `scripts/run_export_plan.py`:
  - `--execute --format gguf --gguf-metadata-path <path>` activates real GGUF writing
  - Missing `--gguf-metadata-path` is rejected before any checkpoint loading
  - Validate GGUF preflight is passed into `ExportWriterRegistry`
  - Full execution ordering: path readiness → inventory → preflight → writer readiness → manifest readiness → registry → write → manifest
  - Output verified for existence and correct byte count after writing
  - Manifest written only after successful GGUF output

## CLI requirements

- `--execute` is required for real writing; dry-run remains the default
- `--gguf-metadata-path` is required for real GGUF execution
- `--format gguf` must match the metadata format
- Remote metadata paths are rejected
- Incompatible format-option combinations are rejected

## CLI execution order

1. Argument validation
2. Remote path rejection
3. Format-specific argument compatibility
4. ExportRequest → ExportPlan
5. Export path readiness
6. Checkpoint inventory
7. GGUF metadata preflight
8. Tensor-count and inventory consistency
9. Writer readiness
10. Manifest readiness
11. Registry with validated preflight → write
12. Output verification
13. Manifest creation (optional)
14. Final JSON

## Safety boundary

This slice does not add training, downloads, external APIs, scraping, uploads, workflow artifacts, external GGUF tools, quantization, subprocesses, or network access.

The writer only loads a caller-selected local PyTorch checkpoint with `weights_only=True`. GGUF execution remains unavailable unless a validated `GGUFPreflightResult` is explicitly supplied to the registry.

## Acceptance evidence

Offline tests cover:

1. successful registry execution for a tiny local F32 state dict
2. rejection when execution is requested without GGUF preflight
3. rejection of non-F32 checkpoint tensors
4. checkpoint-directory resolution through `model.pt`
5. CLI dry-run unchanged for GGUF
6. CLI --execute --format gguf with metadata succeeds
7. Missing --gguf-metadata-path rejected before checkpoint loading
8. Invalid metadata JSON rejected
9. Remote metadata rejected
10. Tensor-count mismatch rejected
11. Non-F32 tensor rejected at writer level
12. Existing output not overwritten
13. Manifest written after successful GGUF output
14. Manifest not written on failure
15. No partial JSON on failure
16. No dry-run fallback on failure
17. Plain and `{"model": ...}` state dicts supported
18. Directory checkpoint resolves through `model.pt`
19. GGUF magic, version, tensor count, and metadata count verified
20. Output bytes_written matches actual file size

## Remaining Milestone 5.3 work

- independent compatibility validation with a trusted local GGUF reader
- architecture-complete metadata for runnable Bharat checkpoints
- broader GGML tensor types or quantization only in separately approved scope
- milestone closure review
