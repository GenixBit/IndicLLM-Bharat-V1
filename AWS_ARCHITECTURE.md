# AWS Cloud & Hybrid Scaling Architecture

This document specifies the AWS infrastructure, Bedrock integration, prompt caching, regional failover, and auto-scaling policies for IndicLLM-Bharat.

---

## 1. Hybrid Compute Architecture

```mermaid
graph TD
    Gateway[Universal AI Gateway] --> AutoScaler[Telemetry & Auto-Scaling Coordinator]
    AutoScaler -->|Low Concurrency / Private| Local[Local Metal/MPS/CUDA Foundation Model]
    AutoScaler -->|High GPU Load > 85%| Bedrock[AWS Bedrock Foundation Models]
    
    Bedrock --> S3[Amazon S3 Knowledge Lake: s3://knowledge/]
    Bedrock --> OpenSearch[Amazon OpenSearch Serverless / pgvector]
    Bedrock --> PromptCache[AWS Bedrock Managed Prompt Cache]
```

---

## 2. Regional Failover Policy

- **Primary Region**: `us-east-1` (N. Virginia)
- **Secondary Backup Region**: `us-west-2` (Oregon) / `ap-south-1` (Mumbai)
- **Local Fallback**: In the event of network disruption or cloud quota saturation, requests automatically failover to local sovereign weights with zero user-facing errors.
