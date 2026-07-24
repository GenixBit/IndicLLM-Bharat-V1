from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

_URL_RE = re.compile(r"^(https?|ftp|s3|gs)://", re.IGNORECASE)


def _is_remote_url(path: str) -> bool:
    return bool(_URL_RE.match(path))


def _validate_slug(value: str, label: str) -> list[str]:
    errors: list[str] = []
    if not value:
        errors.append(f"{label} must not be empty")
    return errors


def _validate_json_schema(schema: dict[str, Any], label: str) -> list[str]:
    errors: list[str] = []
    if not isinstance(schema, dict):
        errors.append(f"{label} must be a dict (JSON Schema object)")
        return errors
    if schema and "type" not in schema:
        errors.append(f"{label} JSON Schema is missing 'type'")
    return errors


_VALID_EVENT_TYPES = frozenset({"text_delta", "function_call", "error", "done"})


@dataclass(frozen=True)
class FunctionSpec:
    name: str
    description: str = ""
    parameters: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        issues: list[str] = []
        issues.extend(_validate_slug(self.name, "Function name"))
        if isinstance(self.parameters, dict):
            issues.extend(
                _validate_json_schema(self.parameters, f"Function {self.name!r} parameters")
            )
        if issues:
            raise ValueError("; ".join(issues))

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"name": self.name}
        if self.description:
            d["description"] = self.description
        if self.parameters:
            d["parameters"] = self.parameters
        return d

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> FunctionSpec:
        return cls(
            name=str(data.get("name", "")),
            description=str(data.get("description", "")),
            parameters=dict(data.get("parameters", {})),
        )


@dataclass(frozen=True)
class FunctionCall:
    name: str
    arguments: str

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("FunctionCall name must not be empty")

    def to_dict(self) -> dict[str, str]:
        return {"name": self.name, "arguments": self.arguments}

    @classmethod
    def from_dict(cls, data: Mapping[str, str]) -> FunctionCall:
        return cls(
            name=str(data.get("name", "")),
            arguments=str(data.get("arguments", "")),
        )


@dataclass(frozen=True)
class StreamEvent:
    event_type: str
    delta: str | None = None
    function_call: FunctionCall | None = None
    error: str | None = None
    index: int = 0
    finish_reason: str | None = None

    def __post_init__(self) -> None:
        if self.event_type not in _VALID_EVENT_TYPES:
            raise ValueError(
                f"Invalid event_type {self.event_type!r}; "
                f"must be one of {sorted(_VALID_EVENT_TYPES)}"
            )
        if self.event_type == "text_delta" and self.delta is None:
            raise ValueError("text_delta events must have a delta")
        if self.event_type == "function_call" and self.function_call is None:
            raise ValueError("function_call events must have a function_call")
        if self.event_type == "error" and self.error is None:
            raise ValueError("error events must have an error message")
        if self.index < 0:
            raise ValueError(f"index must be >= 0, got {self.index}")

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "event_type": self.event_type,
            "index": self.index,
        }
        if self.delta is not None:
            d["delta"] = self.delta
        if self.function_call is not None:
            d["function_call"] = self.function_call.to_dict()
        if self.error is not None:
            d["error"] = self.error
        if self.finish_reason is not None:
            d["finish_reason"] = self.finish_reason
        return d

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> StreamEvent:
        fc_raw = data.get("function_call")
        function_call: FunctionCall | None = None
        if fc_raw is not None:
            if isinstance(fc_raw, dict):
                function_call = FunctionCall.from_dict(fc_raw)
            else:
                raise ValueError("function_call must be a dict")
        return cls(
            event_type=str(data.get("event_type", "")),
            delta=data.get("delta"),
            function_call=function_call,
            error=data.get("error"),
            index=int(data.get("index", 0)),
            finish_reason=data.get("finish_reason"),
        )


@dataclass(frozen=True)
class StreamRequest:
    prompt: str
    max_tokens: int = 256
    temperature: float = 1.0
    functions: tuple[FunctionSpec, ...] = ()
    stream: bool = True

    def __post_init__(self) -> None:
        if not self.prompt:
            raise ValueError("prompt must not be empty")
        if self.max_tokens < 1:
            raise ValueError(f"max_tokens must be >= 1, got {self.max_tokens}")
        if self.temperature < 0.0:
            raise ValueError(f"temperature must be >= 0.0, got {self.temperature}")

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "prompt": self.prompt,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "stream": self.stream,
        }
        if self.functions:
            d["functions"] = [f.to_dict() for f in self.functions]
        return d

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> StreamRequest:
        functions_raw = data.get("functions", [])
        functions: list[FunctionSpec] = []
        if isinstance(functions_raw, list | tuple):
            for f in functions_raw:
                if isinstance(f, dict):
                    functions.append(FunctionSpec.from_dict(f))
        return cls(
            prompt=str(data.get("prompt", "")),
            max_tokens=int(data.get("max_tokens", 256)),
            temperature=float(data.get("temperature", 1.0)),
            functions=tuple(functions),
            stream=bool(data.get("stream", True)),
        )


class LocalStreamer:
    _SAMPLE_RESPONSE = "Hello! How can I help you today?"

    def __init__(self, request: StreamRequest) -> None:
        self._request = request

    def generate(self) -> list[StreamEvent]:
        if self._request.functions:
            return self._generate_function_call()
        return self._generate_text()

    def _generate_text(self) -> list[StreamEvent]:
        words = self._SAMPLE_RESPONSE.split()
        events: list[StreamEvent] = []
        for i, word in enumerate(words):
            delta = word + (" " if i < len(words) - 1 else "")
            events.append(
                StreamEvent(
                    event_type="text_delta",
                    delta=delta,
                    index=i,
                )
            )
        events.append(
            StreamEvent(
                event_type="done",
                index=len(words),
                finish_reason="stop",
            )
        )
        return events

    def _generate_function_call(self) -> list[StreamEvent]:
        spec = self._request.functions[0]
        args: dict[str, Any] = {}
        props = spec.parameters.get("properties", {}) if isinstance(spec.parameters, dict) else {}
        for key in props:
            args[key] = f"<{key}>"
        events: list[StreamEvent] = [
            StreamEvent(
                event_type="function_call",
                function_call=FunctionCall(
                    name=spec.name,
                    arguments=json.dumps(args, sort_keys=True),
                ),
                index=0,
            ),
            StreamEvent(
                event_type="done",
                index=1,
                finish_reason="function_call",
            ),
        ]
        return events


def stream_events_to_jsonl(events: Sequence[StreamEvent]) -> str:
    lines = [json.dumps(e.to_dict(), sort_keys=True) for e in events]
    return "\n".join(lines) + "\n"


def stream_events_to_json(events: Sequence[StreamEvent], indent: int = 2) -> str:
    return json.dumps(
        {
            "events": [e.to_dict() for e in events],
            "event_count": len(events),
        },
        indent=indent,
        sort_keys=True,
    )
