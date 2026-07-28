# Milestone 6.1 — Production Tokenizer Evidence Contract

## Status

Planned validation contract. This document does **not** claim that a production
64K tokenizer has been trained, evaluated, accepted, or committed.

## Purpose

Define the minimum offline evidence package required before Milestone 6.1 can
claim production tokenizer acceptance. The package is supplied by an operator
from approved local artifacts. Repository automation must validate it without
network access, downloads, uploads, scraping, or training.

## Required package

A production evidence directory must contain exactly these logical inputs:

1. `manifest.json` — canonical provenance manifest using
   `tokenizer-production-evidence-manifest-v1`.
2. A caller-provided tokenizer artifact.
3. A caller-provided approved evaluation corpus or immutable corpus manifest.
4. An evaluation report produced by the repository evaluation framework.
5. An acceptance decision produced by the repository acceptance gate.
6. The exact acceptance-threshold configuration used for the decision.

Large tokenizer artifacts and corpora remain outside Git history unless a later
approved milestone explicitly authorizes committing them.

## Required provenance

The manifest must record:

- tokenizer artifact path, SHA-256 digest, fingerprint, vocabulary capacity,
  normalization contract, and byte-alphabet status;
- evaluation input path or manifest path and SHA-256 digest;
- evaluation-report path and SHA-256 digest;
- acceptance-decision path and SHA-256 digest;
- threshold-configuration path and SHA-256 digest;
- repository commit SHA used to generate the report and decision;
- evidence scope `production-local-approved`;
- status `candidate` or `accepted`;
- deterministic generation commands as an ordered array;
- required language coverage and per-language record counts.

All paths must be relative to the evidence package root, remain within that
root after resolution, and must not traverse symlinks outside it.

## Acceptance invariants

An evidence package may use status `accepted` only when all of the following
are true:

1. Every referenced file exists locally and matches its declared SHA-256.
2. The tokenizer fingerprint recomputed from the artifact matches the manifest.
3. The evaluation report passes strict schema validation and contains only
   finite numeric values.
4. The acceptance decision passes strict schema validation, references the
   exact evaluation report and threshold configuration, and has `passed=true`.
5. Complete byte coverage is reported and independently verifiable by the
   tokenizer adapter.
6. Required NFC and canonical-equivalence round-trip thresholds pass.
7. Unknown-token, aggregate fertility, and per-language fertility thresholds
   pass.
8. Every required language has at least the configured minimum record count.
9. The report, decision, and manifest use canonical JSON with sorted keys,
   compact separators, UTF-8 encoding, and no non-finite values.
10. Repeating validation against unchanged inputs produces byte-identical
    canonical validation output.

A package with missing evidence, provisional thresholds, failed checks, or an
unverifiable tokenizer must remain `candidate` and must not close Milestone 6.1.

## Safety boundary

Validation must be deterministic, CPU-local, and read-only except for an
optional caller-selected local result file created with no-overwrite semantics.
It must not:

- train or modify a tokenizer or model;
- download datasets, benchmarks, models, or dependencies;
- call external APIs or access the network;
- scrape content;
- upload evidence or workflow artifacts;
- infer approval from file names, directory names, or operator claims.

## Repository integration sequence

1. Add a strict manifest schema and parser.
2. Add a local validator that cross-checks all digests, fingerprints, report
   references, threshold references, and acceptance status.
3. Add deterministic synthetic fixtures for accepted and rejected package
   shapes; these fixtures must not be described as production evidence.
4. Run the validator on caller-provided production evidence outside CI.
5. Commit only the small canonical validation record and closure documentation
   after independent review. Do not commit large artifacts by default.

## Milestone boundary

This contract prepares Milestone 6.1 PR 19f. Milestone 6.1 remains open until a
real 64K tokenizer evidence package satisfies this contract and receives an
independent review. Bharat-350M tokenizer integration, model smoke tests, and
training remain later slices.