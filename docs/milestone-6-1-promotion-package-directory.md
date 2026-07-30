# Milestone 6.1 — Promotion Package Directory Verification

`bharat.tokenizer.promotion_package_directory.verify_promotion_package_directory` verifies that a local promotion package directory contains exactly `manifest.json`, `readiness.json`, and `decision.json`.

The verifier rejects missing or unexpected entries and requires every package entry to be a regular file, preventing symlink, directory, socket, and FIFO substitution. It then delegates to the existing promotion package verifier for fresh readiness validation and exact digest binding.

The utility is deterministic, read-only, and offline. It does not mutate or promote evidence, train a tokenizer or model, download datasets or benchmarks, call external APIs, scrape, upload files or artifacts, load checkpoints, run subprocesses, or use the network. Successful verification is review evidence only; a separate controlled promotion step remains required.
