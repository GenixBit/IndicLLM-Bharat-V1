# Milestone 6.1 production-input audit

Milestone 6.1 — 64K BPE Tokenizer Validation cannot claim production readiness until the required production inputs, authorizations, identities, and storage controls are available and independently verified.

This audit is intentionally documentation-only. It does not authorize tokenizer training, dataset or benchmark downloads, external APIs, scraping, uploads, artifact uploads, or network-dependent execution.

## Required inputs and evidence

| Area | Required evidence | Acceptance condition | Current state |
| --- | --- | --- | --- |
| Training corpus | Approved corpus inventory, license/provenance record, language distribution, deduplication policy, and written authorization to process it | Every source is approved and traceable; processing boundaries are explicit | Blocked — not supplied |
| Tokenizer artifact | Genuine candidate 64K tokenizer artifact with immutable digest, build provenance, configuration, and authorization to evaluate; alternatively explicit authorization to train | Artifact or training authorization is available and digest-bound | Blocked — not supplied |
| Evaluation set | Approved multilingual evaluation inventory with provenance, licenses, language/script coverage, and local access authorization | Evaluation inputs are approved and usable offline | Blocked — not supplied |
| Threshold policy | Versioned production acceptance thresholds for fertility, unknown-token behavior, byte fallback, normalization, round-trip integrity, language/script coverage, and regressions | Thresholds are approved before results are interpreted | Blocked — not supplied |
| Independent identities | Named assembler/operator and independent reviewer with non-identical identities and documented responsibilities | Evidence can be assembled and independently reviewed | Blocked — not supplied |
| Compute | Approved execution environment, resource limits, reproducibility settings, and deterministic command plan | Evaluation or training can run within approved boundaries | Blocked — not supplied |
| Secure storage | Approved local or controlled storage location with retention, access, digest, and backup policy | Inputs and outputs can be stored without unapproved uploads | Blocked — not supplied |

## Audit decisions

For each row, the milestone owner must record:

1. evidence location and immutable SHA-256 where applicable;
2. approving person or authority and approval date;
3. allowed operations and explicitly prohibited operations;
4. expiry or review date, if any;
5. residual risks and required mitigations.

Do not place confidential datasets, tokenizer artifacts, benchmark contents, credentials, or generated artifacts in this repository. References must identify approved controlled locations without copying protected material.

## Software-gap determination

After the required inputs are supplied, compare them against the existing offline tokenizer validation and acceptance-evidence tooling. Open a software PR only when the audit identifies a concrete missing capability with:

- a precise acceptance criterion;
- local deterministic fixtures that do not contain production data;
- no new network, download, upload, scraping, training, or external-API behavior unless separately approved;
- a defined failure mode and test plan;
- an explicit relationship to Milestone 6.1 production readiness.

A missing production input is not a software defect and must not be addressed by adding another evidence wrapper.

## Milestone acceptance gate

Milestone 6.1 remains incomplete until all required inputs are accepted, the approved evaluation or training work is executed, results satisfy the pre-approved threshold policy, evidence is independently reviewed, and the final artifact and report are digest-bound in approved secure storage.
