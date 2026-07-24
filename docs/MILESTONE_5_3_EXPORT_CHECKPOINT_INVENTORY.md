# Milestone 5.3 — Local Checkpoint Export Inventory

**Status:** Implemented foundation

## Objective

Add a deterministic, offline inventory of local checkpoint files before real format-specific export writers are introduced.

## Implemented

- Recursively inventories files under a local checkpoint directory.
- Sorts relative paths for deterministic output.
- Records file sizes and SHA-256 digests.
- Produces stable dictionary and JSON serialization.
- Rejects missing paths, non-directory paths, and empty checkpoint directories.
- Uses tiny local fixtures in tests.

## Safety boundary

This implementation does not load, interpret, convert, serialize, upload, or publish model weights. It performs no network calls, downloads, external conversions, scraping, or artifact uploads. Milestone 5.3 remains incomplete until actual local safetensors and GGUF writers are implemented and tested.
