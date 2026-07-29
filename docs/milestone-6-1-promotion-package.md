# Milestone 6.1 — Promotion Package Verification

`bharat.tokenizer.promotion_package.verify_promotion_package` verifies that a local candidate evidence manifest, readiness report, and operator approval record form one consistent review package.

The verifier recomputes readiness locally, rejects manifest mutation during validation, requires promotion-ready candidate evidence, requires an explicit `approve` decision, and checks the exact manifest and readiness SHA-256 digests recorded by the operator decision.

The utility is read-only. It does not mutate or promote evidence, train a tokenizer or model, download datasets or benchmarks, call external APIs, scrape, upload files or artifacts, load checkpoints, run subprocesses, or use the network. Successful verification is review evidence only; a separate controlled promotion step remains required.
