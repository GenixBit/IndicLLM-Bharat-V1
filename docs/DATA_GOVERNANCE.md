# Data Governance

## Purpose

The Bharat AI data-source registry provides a single, versioned record of
every dataset considered for use in pretraining, fine-tuning (SFT/DPO) and
evaluation.  It enforces licence decisions, provenance requirements,
immutable revision pins and integrity checks — all **offline**, with no
network access required during validation.

**Important**: This registry is an **engineering control**, not legal advice.
Licence decisions recorded here reflect the project's current policy and
must not be treated as a substitute for independent legal review.

---

## Source lifecycle

```
PROPOSED → APPROVED
  │            │
  │            ├── (later review) → REJECTED
  │            └── (superseded by newer source) → DEPRECATED
  │
  └── (initial review) → REJECTED
```

| Status | Meaning |
|--------|---------|
| `proposed` | Candidate under consideration; not yet reviewed. |
| `approved` | Fully reviewed, licence-allowed, integrity-pinned; ready for pipeline consumption. |
| `rejected` | Reviewed and excluded; requires a reason in `notes`. |
| `deprecated` | Previously approved but superseded; excluded from `approved_for()` queries. |

---

## Licence decisions

| Decision | Meaning |
|----------|---------|
| `allow` | The licence is known and permitted for all required purposes. |
| `review` | The licence requires human or legal review before use. |
| `deny` | The licence is prohibited and must never be used. |

### Default-deny

**Unknown or missing licence identifiers default to `deny`** — they cannot
be approved until explicitly reviewed and added to the policy.

### All `allow` requirements

Before a licence can be marked `allow`:
1. A verifier must record `evidence_url`, `verified_at` and `verified_by`.
2. The licence must explicitly permit commercial use, model training and
   redistribution (or the policy must accept the specific restrictions).
3. Non-commercial or research-only restrictions must not be silently
   treated as commercially approved.

---

## Provenance requirements

Every source must record:
- **Who** the provider is.
- **What** the URI is (no embedded credentials).
- **Where** the data came from (upstream sources, if derived).
- **How** it was collected.
- **When** it was registered and last updated.

### Immutable revisions

- **Hugging Face** sources must pin an **immutable commit SHA** — not
  `main`, `master`, or `latest`.
- **HTTP** sources must provide a **SHA-256 checksum** of the content.
- **Cloud or local** sources must provide a SHA-256 checksum, a checksummed
  manifest, or an immutable version ID.

All SHA-256 values must be 64-character lowercase hexadecimal strings.

---

## Integrity checks

| Source kind | Required integrity pin |
|-------------|----------------------|
| `huggingface` | `revision` (commit SHA) + `sha256` |
| `http` | `revision` + `sha256` |
| `s3`, `gcs`, `azure_blob`, `local` | `revision` + (`sha256` or `manifest_sha256`) |
| `other` | `revision` + (`sha256` or `manifest_sha256`) |

---

## Secret handling

- **No raw API tokens, passwords or private keys** may be stored in
  registry records.
- Credentials are referenced by **environment-variable name only**
  (e.g. `credentials_env: "HF_TOKEN"`).
- The registry validator rejects any string that looks like an embedded
  key or secret.

---

## Registry digest

The registry digest is a **SHA-256** hash over the canonical JSON
representation of every normalised source record in deterministic order.
It is used to:

- Verifiy registry integrity across environments.
- Pin the exact set of approved sources for a training run.
- Detect unannounced changes to the registry.

The digest includes **all** records (approved, proposed, rejected and
deprecated).

---

## Source approval workflow

1. A dataset is proposed by adding a YAML file to `data_registry/sources/`
   with status `proposed`.
2. The project team reviews:
   - Licence compatibility.
   - Data quality and collection method.
   - Provenance and upstream sources.
3. If approved, the status is changed to `approved` and the licence is
   verified with evidence.
4. If rejected, the status is changed to `rejected` with a reason.
5. If a newer version supersedes an older one, the older is marked
   `deprecated`.

---

## Deprecation and supersession

When a source is superseded by a newer version:
1. The new source uses `supersedes` to reference the old source ID.
2. The old source should be marked `deprecated`.
3. `approved_for()` only returns non-deprecated sources.
4. Self-supersession and supersession cycles are rejected.

---

## Pipeline consumption

Future pipeline stages (`bharat/data/...`) will consume only sources
returned by `registry.approved_for()`.  This guarantees:

- Licence compliance.
- Immutable, integrity-checked data.
- Deterministic, reproducible training.

---

## Status

- [x] Registry infrastructure exists and is validated offline.
- [x] Default-deny licensing is enforced.
- [x] Approved sources require evidence and immutable provenance.
- [ ] No dataset has been automatically legally approved.
- [ ] No data has been downloaded or processed.
- [ ] Quality filtering and deduplication remain future work.
- [ ] The legacy `data/` pipelines remain unchanged.
