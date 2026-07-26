# Milestone 5.3 — GGUF Metadata Preflight

**Status:** Implemented foundation

## Objective

Add deterministic, offline validation for a planned local GGUF export before a model-weight writer is introduced.

## Implemented

- Validates schema version, architecture, alignment, tensor count, and output filename.
- Requires alignment to be a positive power of two.
- Confirms the referenced `.gguf` output exists in the local checkpoint inventory fixture.
- Validates supported scalar metadata value types: `bool`, `float`, `int`, and `string`.
- Requires every metadata entry to include a scalar value matching its declared type.
- Rejects Python booleans where an integer is required and integers where a float is required.
- Rejects duplicate metadata keys.
- Sorts metadata keys for deterministic dictionary and JSON output.
- Preserves validated scalar values in the preflight result so a later GGUF writer can serialize them without reinterpreting the input.
- Uses tiny local fixtures in tests.

## Safety boundary

This validator reads local JSON metadata and checkpoint inventory records only. It does not parse GGUF binary headers, read tensor payloads, convert checkpoints, serialize model weights, call external tools, use network access, download data, upload files, or publish workflow artifacts. Milestone 5.3 remains incomplete until a real local GGUF model-weight writer is implemented and tested.
