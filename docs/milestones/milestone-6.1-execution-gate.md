# Milestone 6.1 execution gate

Milestone 6.1 — 64K BPE Tokenizer Validation remains an authorization-gated milestone. This document defines the next safe repository action without introducing production data, training, or network execution.

## Current gate

Do not start tokenizer training or production evaluation from this repository until every required production-input category has an accepted, independently reviewed submission record:

- training corpus
- tokenizer artifact or tokenizer-training authorization
- evaluation set
- threshold policy
- independent identities
- compute
- secure storage

Protected datasets, tokenizer artifacts, benchmark contents, credentials, generated artifacts, and secret storage locations must remain outside the repository in approved controlled storage.

## Permitted repository work before authorization

The repository may continue to add or maintain deterministic, local-only validation code, fixtures, documentation, and tests that do not require production inputs. Such changes must not imply that Milestone 6.1 is complete.

Permitted work must remain:

- offline and deterministic;
- free of dataset or benchmark downloads;
- free of external APIs and scraping;
- free of uploads and artifact publication;
- free of credentials and protected material;
- covered by local tests where executable behavior changes.

## Authorization transition

When an input category is approved, record the scope explicitly in its controlled submission record before using it. The record must identify the approval authority, permitted operations, prohibited operations, execution/storage boundary, provenance, and independent review where required.

Only after all required records are accepted should an authorized evaluation or training run be considered for execution. Its results must then be checked against the pre-approved threshold policy and independently reviewed before any promotion decision.

## Completion boundary

This gate is not evidence of tokenizer quality and does not authorize training. It only prevents the repository workflow from confusing missing controlled inputs with a software defect or from bypassing the Milestone 6.1 acceptance boundary.

The milestone remains incomplete until its required inputs are authorized, the approved evaluation/training work is executed within the approved boundary, thresholds are satisfied, evidence is independently reviewed, and final artifacts/reports are digest-bound in approved secure storage.
