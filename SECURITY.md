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
