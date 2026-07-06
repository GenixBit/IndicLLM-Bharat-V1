# Bharat AI Vision

## What We're Building

Production-grade, open-source large language models for Indian languages, trained from scratch with transparent data, reproducible evaluation, and safety built in.

## Design Principles

1. **From scratch** — no fine-tuning of English models; we train our own tokenizer and model on multilingual Indian language data
2. **Validated, not claimed** — every capability claim backed by benchmark evidence
3. **Open data** — all data sources, filters, and deduplication pipelines are public; data manifests include provenance, licence, and quality metrics
4. **Reproducible** — same config + data + seed = same loss curve; checkpoint metadata includes git SHA, tokenizer hash, config, and data version
5. **Production-ready** — streaming API with auth, rate limiting, metrics; safety evaluations at every release
6. **Iterative** — milestone-based delivery with tests, linting, and review at every step

## Target Capabilities

| Capability | Current | Milestone 1 | Milestone 2 | Release |
|-----------|---------|-------------|-------------|---------|
| English text generation | ✅ GPT-2 10M | ✅ Fixed SFT/DPO | ⏳ | — |
| Indic text generation | ⚠️ Tokenize only | — | — | Planned |
| Instruction following | — | ✅ SFT loss masking | — | Planned |
| Preference alignment | — | ✅ DPO per-sample | — | Planned |
| Modern architecture | — | — | ✅ RoPE, SwiGLU, GQA | Planned |
| Production API | ⚠️ No auth | ✅ Configurable CORS | — | Planned |
| Evaluated | ⚠️ Perplexity only | ✅ Test suite | ✅ Benchmark suite | Planned |

## Non-Goals

- Claiming frontier AI capability without evidence
- Training on data with unknown or non-commercial licences
- Releasing models without safety evaluation
- Keeping non-reproducible training runs
- Cloud-only development — all milestones testable on a laptop

## Roadmap

See [ROADMAP.md](ROADMAP.md) for the detailed milestone plan.
