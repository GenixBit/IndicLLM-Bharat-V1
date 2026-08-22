# Production Deployment Guide: IndicLLM-Bharat Universal Hybrid AI

This guide details deployment options across 5 supported modes.

---

## 1. Supported Deployment Modes

- **Mode A: 100% Local Sovereign Mode** (Air-gapped, zero cloud egress).
- **Mode B: Local + Cloud Fallback** (Local-first, escapes to AWS Bedrock if GPU load $> 85\%$).
- **Mode C: Cloud-First Mode** (AWS Bedrock primary with local fallback).
- **Mode D: Enterprise Private Cloud** (Deploy on AWS EKS / ECS with Amazon OpenSearch & S3 Knowledge Base).
- **Mode E: Hybrid Distributed Cluster** (Distributed multi-GPU local cluster with dynamic bursting to AWS).

---

## 2. Launching the Universal Gateway

```bash
# Launch Gateway on Port 8000
python3 scripts/start_universal_gateway.py --tier 1b --port 8000 --host 0.0.0.0
```
