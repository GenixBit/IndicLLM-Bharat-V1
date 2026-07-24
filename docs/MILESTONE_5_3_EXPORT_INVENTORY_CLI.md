# Milestone 5.3 — Export Inventory CLI Integration

**Status:** Foundation extension implemented

## Objective

Expose deterministic local checkpoint inventory metadata through the existing dry-run export CLI.

## Implemented

- `run-export-plan --include-inventory` opt-in flag.
- Local checkpoint inventory generated with the existing `build_checkpoint_inventory` API.
- Stable JSON output containing checkpoint path, total bytes, ordered files, sizes, and SHA-256 digests.
- Existing CLI behavior remains unchanged when the flag is omitted.
- Missing, non-directory, and empty checkpoint inputs fail with a non-zero exit code when inventory is requested.
- Deterministic subprocess tests use tiny local fixtures only.

## Safety boundary

This extension hashes local checkpoint files but does not load, interpret, convert, serialize, upload, or publish model weights. It performs no network calls, downloads, external conversions, scraping, or artifact uploads. Milestone 5.3 remains incomplete until actual local safetensors and GGUF writers are implemented and tested.
