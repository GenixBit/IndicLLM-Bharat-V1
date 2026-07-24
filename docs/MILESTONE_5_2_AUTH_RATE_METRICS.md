# Milestone 5.2 — Authentication, Rate Limiting, Metrics

**Status:** Implemented

## Objective

Add a local production-serving control layer with authentication,
in-memory rate limiting, and local metrics around the existing
Milestone 5.1 streaming API foundation.

## Implemented in this PR

### Authentication (`bharat/serving/auth.py`)

- `AuthConfig` — frozen dataclass holding a tuple of valid API keys
- `AuthResult` — frozen dataclass with `ok`, `error`, and `client_id`
- `ApiKeyAuthenticator` — validates API keys against a configured set
  - Valid key → `AuthResult(ok=True, client_id=<prefix>)`
  - Missing key → `AuthResult(ok=False, error="Missing API key")`
  - Invalid key → `AuthResult(ok=False, error="Invalid API key")`

### Rate Limiting (`bharat/serving/rate_limit.py`)

- `RateLimitConfig` — `max_requests` and `window_seconds`
- `RateLimitResult` — `ok`, `remaining`, `reset_after`, `error`
- `InMemoryRateLimiter` — per-client sliding-window rate limiter
  - Injected `time_fn` for deterministic clock control in tests
  - Rejects requests after the configured limit
  - Allows requests again after the window resets
  - `check(client_id)` returns `RateLimitResult`

### Metrics (`bharat/serving/metrics.py`)

- `ServingMetrics` — mutable dataclass tracking:
  - `requests_started`, `requests_completed`
  - `auth_failures`, `rate_limit_rejections`
  - `streaming_events_emitted`, `streaming_errors`
  - `total_duration_ms`
- `MetricsSnapshot` — frozen dataclass with `to_dict()` and `to_json()`
- No Prometheus, OpenTelemetry, or external export

### Serving Controller (`bharat/serving/controller.py`)

- `ServingController` — framework-independent wrapper
  - Accepts optional `AuthConfig`, `RateLimitConfig`, `ServingMetrics`
  - `handle_request(request, api_key, client_id) → list[StreamEvent]`
  - Flow: authenticate → rate limit → stream → record metrics
  - Auth failure → error events returned, no streaming
  - Rate-limit rejection → error events returned, no streaming
  - Success → streaming events returned, metrics recorded

### CLI (`scripts/run_serving_control_smoke.py`)

- `--prompt`, `--api-key`, `--output`, `--format`, `--json`
- Rejects remote output paths (`http://`, `https://`, `ftp://`, `s3://`, `gs://`)

## Authentication Behavior

| Scenario | Result |
|----------|--------|
| Valid API key | AuthResult(ok=True) |
| Missing API key (None) | AuthResult(ok=False, error="Missing API key") |
| Invalid API key | AuthResult(ok=False, error="Invalid API key") |
| Empty string API key | AuthResult(ok=False, error="Invalid API key") |

## Rate-Limit Behavior

| Scenario | Result |
|----------|--------|
| Under limit | RateLimitResult(ok=True, remaining=N) |
| At or over limit | RateLimitResult(ok=False, error="Rate limit exceeded") |
| After window reset | Allowed again, remaining resets |

## Metrics

| Counter | Description |
|---------|-------------|
| `requests_started` | Total requests received |
| `requests_completed` | Total requests processed |
| `auth_failures` | Authentication failures |
| `rate_limit_rejections` | Rate-limit rejections |
| `streaming_events_emitted` | Stream events generated |
| `streaming_errors` | Error events in streams |
| `total_duration_ms` | Cumulative processing time |

## Controller Flow

```
handle_request(request, api_key)
  │
  ├─ Auth enabled? ── failure ──► error events + metrics
  │
  ├─ Rate limit enabled? ── failure ──► error events + metrics
  │
  └─ Stream ──► events + metrics
```

## API

```python
from bharat.serving import (
    ApiKeyAuthenticator, AuthConfig,
    InMemoryRateLimiter, RateLimitConfig,
    ServingMetrics,
    ServingController,
    StreamRequest,
)

controller = ServingController(
    auth_config=AuthConfig(valid_api_keys=("sk-test-001",)),
    rate_limit_config=RateLimitConfig(max_requests=60, window_seconds=60.0),
    metrics=ServingMetrics(),
)

events = controller.handle_request(
    request=StreamRequest(prompt="Hello"),
    api_key="sk-test-001",
)
print(controller.metrics.snapshot().to_json())
```

## What Is Not Included

- No HTTP server (will be added in a later milestone)
- No Prometheus or OpenTelemetry export
- No JWTs or OAuth — simple API keys only
- No database, Redis, or external state
- No real secrets or environment variables required
- No model training, downloads, or network calls

## Offline Guarantee

All components are deterministic and offline. No external services,
databases, or network calls are used.
