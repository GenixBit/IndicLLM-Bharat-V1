# Offline tokenizer promotion receipt

Milestone 6.1 promotion review remains a human-controlled, local-only process. A promotion receipt records that the exact manifest, readiness report, and approval decision already verified in a promotion package were acknowledged by the same operator.

The receipt is a UTF-8 JSON object with exactly these fields:

```json
{
  "schema_version": "tokenizer-promotion-receipt-v1",
  "manifest_sha256": "<sha256>",
  "readiness_sha256": "<sha256>",
  "decision_sha256": "<sha256>",
  "operator": "<non-empty operator identifier>"
}
```

`verify_promotion_receipt` first re-verifies the complete local promotion package directory, then checks that every digest and the operator are bound to that verified package. The verifier rejects symlinked or non-regular receipt paths, malformed JSON, unexpected fields, unsupported schemas, digest mismatches, and operator mismatches.

Verification is deterministic, read-only, and offline. It does not train or load a tokenizer, download datasets or benchmarks, call external APIs, scrape content, upload files or artifacts, mutate package contents, or perform promotion.
