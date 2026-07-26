# Milestone 5.3 — Local GGUF Execute Writer

**Status:** Draft implementation for deterministic local F32 checkpoint export

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

## Safety boundary

This slice does not add training, downloads, external APIs, scraping, uploads, workflow artifacts, external GGUF tools, quantization, automatic remote loading, or network access.

The writer only loads a caller-selected local PyTorch checkpoint with `weights_only=True`. GGUF execution remains unavailable unless a validated `GGUFPreflightResult` is explicitly supplied to the registry.

## Acceptance evidence

Offline tests cover:

1. successful registry execution for a tiny local F32 state dict
2. rejection when execution is requested without GGUF preflight
3. rejection of non-F32 checkpoint tensors
4. checkpoint-directory resolution through `model.pt`

## Remaining Milestone 5.3 work

- CLI wiring that passes validated `--gguf-metadata-path` preflight into the registry
- independent compatibility validation with a trusted local GGUF reader
- architecture-complete metadata for runnable Bharat checkpoints
- broader GGML tensor types or quantization only in separately approved scope
