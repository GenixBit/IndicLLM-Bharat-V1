# Security, Privacy & Prompt Injection Defense Architecture

This document establishes the security standards, data isolation namespaces, untrusted input containment, and prompt injection defenses for IndicLLM-Bharat.

---

## 1. Security Axiom: DATA ≠ INSTRUCTIONS

Retrieved external web content, user-uploaded PDFs, and database records are treated as strictly untrusted evidence data:
- System prompt and developer boundary instructions are cryptographically delimited.
- External documents are placed in isolated evidentiary blocks (`--- Evidence ---`) and stripped of control sequences.
- AST Python execution is restricted to safe builtins with complete disabling of `subprocess`, `os.system`, or unvetted network calls.

---

## 2. Privacy Namespaces

| Namespace | Access Permission | Routing Constraint |
|:---|:---|:---|
| `PUBLIC` | Open access | May use web search, cloud models, or local compute |
| `COMPANY` | Verified enterprise token | Internal vector search only; encrypted in transit |
| `PRIVATE` | Strict single-user session | **100% Local Sovereign Processing Enforced**; zero cloud transmission |

## 3. Gateway runtime hardening (development baseline)

- The gateway requires `BHARAT_GATEWAY_API_TOKEN` (at least 32 characters) and HTTP `Authorization: Bearer ...` on API routes.
- The launcher binds to `127.0.0.1` by default; configure network access and TLS only behind approved controls.
- CORS is deny-by-default; explicitly configure `BHARAT_GATEWAY_CORS_ORIGINS` if browser access is required.
- Shared cross-request prompt caching is disabled until verified identity and tenant isolation are implemented.
- Explicitly private requests cannot force cloud or external web routing.
- Simulated web passages are never served as live evidence from this gateway; live web routing returns a 503 until a real authorized provider is implemented.
- Model tiers other than `tiny` and `small` require an existing checkpoint. Demo tiers do not imply trained-model quality.
- A single bearer token is not multi-tenant identity. External/public deployments still require a separately reviewed authorization, abuse prevention and security model.
