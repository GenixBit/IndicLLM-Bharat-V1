# Milestone 5.1 — Streaming API Foundation

**Status:** Implemented

## Objective

Add typed streaming request and event models with a deterministic local
streamer for offline tests. Establish schema-only function specification
validation without real function/tool execution.

## Implemented in this PR

- `FunctionSpec` — frozen dataclass with `name`, `description`, and
  `parameters` (JSON Schema dict). Validates that `name` is non-empty
  and `parameters` contains a `type` field.
- `FunctionCall` — frozen dataclass with `name` and `arguments` (JSON
  string). Represents a function/tool invocation in a streaming event.
- `StreamEvent` — frozen dataclass with `event_type` (one of
  `text_delta`, `function_call`, `error`, `done`), optional `delta`,
  `function_call`, `error`, `index`, and `finish_reason`.
- `StreamRequest` — frozen dataclass with `prompt`, `max_tokens`,
  `temperature`, `functions`, and `stream` flag.
- `LocalStreamer` — deterministic local streamer that generates
  synthetic events without any model loading or network calls.
  - Text mode: emits word-by-word `text_delta` events for
    "Hello! How can I help you today?" then a `done` event.
  - Function-call mode: emits a `function_call` event for the first
    registered function with placeholder arguments, then a `done` event.
- `stream_events_to_json()` / `stream_events_to_jsonl()` — serialize
  event sequences to JSON array or JSONL format.
- `scripts/stream_local.py` — CLI with `--prompt`, `--max-tokens`,
  `--temperature`, `--function` (repeatable), `--format` (json/jsonl),
  `--output`, and `--json` machine-readable output.
- 24 tests covering all models, validation, serialization, streamer
  behavior, and CLI integration.

## CLI Usage

```bash
# Basic text streaming (JSONL to stdout)
python scripts/stream_local.py --prompt "Hello"

# JSON output format
python scripts/stream_local.py --prompt "Hi" --format json

# Function call streaming
python scripts/stream_local.py \
  --prompt "Weather?" \
  --function specs/get_weather.json

# Output to file with machine-readable result
python scripts/stream_local.py \
  --prompt "Hello" \
  --output events.jsonl \
  --json
```

## Function Spec File Format

```json
{
  "name": "get_weather",
  "description": "Get the current weather for a location",
  "parameters": {
    "type": "object",
    "properties": {
      "location": {
        "type": "string",
        "description": "City name"
      }
    },
    "required": ["location"]
  }
}
```

## Event Types

| Event Type | Fields | Description |
|------------|--------|-------------|
| `text_delta` | `delta`, `index` | A chunk of generated text |
| `function_call` | `function_call`, `index` | A function/tool invocation |
| `error` | `error`, `index` | An error message |
| `done` | `index`, `finish_reason` | Stream complete (`stop`, `length`, `function_call`) |

## API

```python
from bharat.serving import (
    FunctionSpec,
    FunctionCall,
    StreamEvent,
    StreamRequest,
    LocalStreamer,
    stream_events_to_jsonl,
)

req = StreamRequest(prompt="Hello")
streamer = LocalStreamer(req)
events = streamer.generate()
print(stream_events_to_jsonl(events))
```

## Offline Guarantee

No model training, checkpoint downloads, external APIs, scraping,
uploads, or real function execution. All output is deterministic
and purely synthetic.
