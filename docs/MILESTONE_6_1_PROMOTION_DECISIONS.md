# Milestone 6.1 — Promotion Decision Records

`bharat.tokenizer.promotion_decision` records an operator's offline review decision for a candidate production-tokenizer evidence manifest.

The record binds the exact manifest and readiness-report bytes with SHA-256 digests, requires the readiness report to match a fresh local validation, and permits `approve` only when the candidate is ready for human promotion. `reject` remains available when blockers exist.

The utility does **not** mutate the evidence manifest, change its status, train a tokenizer, download data or benchmarks, call external APIs, scrape, upload files, load checkpoints, run subprocesses, or use the network. A decision record is review evidence only; a separate controlled promotion step is still required.

Output is canonical JSON, created exclusively without overwrite, flushed to disk, read back, and byte-verified. Failed writes are removed when owned by the current operation.
