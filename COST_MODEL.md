# Cost Optimization & Cloud Spend Model

This document outlines the economic governance and cost minimization framework for IndicLLM-Bharat.

---

## 1. Cost Routing Hierarchy

1. **Exact & Semantic Cache**: \$0.00 cost (0 compute tokens).
2. **Local Sovereign Compute**: \$0.00 incremental cloud cost.
3. **Local Hybrid RAG**: \$0.00 incremental cloud cost.
4. **Live Web Intelligence**: Minimal search query API cost (~0.0001/query).
5. **AWS Bedrock Cloud**: Managed token spend (~0.0015/1k input, $0.0030/1k output) only invoked when local resources are overloaded or complex theorems require large frontier cloud models.

---

## 2. Estimated Cost Comparison (100,000 queries)

| Architecture | Monthly Cloud Cost |
|:---|:---:|
| **Naive Pure Cloud LLM** | \$450.00 |
| **IndicLLM-Bharat Universal Hybrid (Local + Cache + Router + Bedrock)** | **\$14.50 (96.8% Cost Savings)** |
