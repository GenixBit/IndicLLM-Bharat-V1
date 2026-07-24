from __future__ import annotations

import json
from pathlib import Path

import pytest

from bharat.serving.streaming import (
    FunctionCall,
    FunctionSpec,
    LocalStreamer,
    StreamEvent,
    StreamRequest,
    stream_events_to_json,
    stream_events_to_jsonl,
)


class TestFunctionSpec:
    def test_valid_minimal(self) -> None:
        spec = FunctionSpec(name="get_weather")
        assert spec.name == "get_weather"
        assert spec.description == ""
        assert spec.parameters == {}

    def test_valid_full(self) -> None:
        spec = FunctionSpec(
            name="get_weather",
            description="Get the weather for a location",
            parameters={
                "type": "object",
                "properties": {
                    "location": {"type": "string"},
                },
            },
        )
        assert spec.name == "get_weather"
        assert spec.description == "Get the weather for a location"

    def test_empty_name_raises(self) -> None:
        with pytest.raises(ValueError, match="Function name must not be empty"):
            FunctionSpec(name="")

    def test_missing_type_in_parameters(self) -> None:
        with pytest.raises(ValueError, match="missing 'type'"):
            FunctionSpec(name="test", parameters={"properties": {}})

    def test_to_dict_roundtrip(self) -> None:
        spec1 = FunctionSpec(
            name="get_weather",
            description="Get weather",
            parameters={"type": "object", "properties": {"loc": {"type": "string"}}},
        )
        d = spec1.to_dict()
        spec2 = FunctionSpec.from_dict(d)
        assert spec1 == spec2


class TestFunctionCall:
    def test_valid(self) -> None:
        fc = FunctionCall(name="get_weather", arguments='{"location": "Mumbai"}')
        assert fc.name == "get_weather"

    def test_empty_name_raises(self) -> None:
        with pytest.raises(ValueError, match="FunctionCall name must not be empty"):
            FunctionCall(name="", arguments="{}")

    def test_to_dict(self) -> None:
        fc = FunctionCall(name="get_weather", arguments='{"loc": "Mumbai"}')
        d = fc.to_dict()
        assert d["name"] == "get_weather"
        assert d["arguments"] == '{"loc": "Mumbai"}'


class TestStreamEvent:
    def test_text_delta(self) -> None:
        event = StreamEvent(event_type="text_delta", delta="Hello", index=0)
        assert event.event_type == "text_delta"
        assert event.delta == "Hello"

    def test_function_call_event(self) -> None:
        fc = FunctionCall(name="get_weather", arguments="{}")
        event = StreamEvent(
            event_type="function_call",
            function_call=fc,
            index=0,
            finish_reason="function_call",
        )
        assert event.event_type == "function_call"
        assert event.function_call == fc

    def test_error_event(self) -> None:
        event = StreamEvent(event_type="error", error="Something went wrong", index=0)
        assert event.error == "Something went wrong"

    def test_done_event(self) -> None:
        event = StreamEvent(event_type="done", index=1, finish_reason="stop")
        assert event.finish_reason == "stop"

    def test_invalid_event_type_raises(self) -> None:
        with pytest.raises(ValueError, match="Invalid event_type"):
            StreamEvent(event_type="invalid", delta="x", index=0)

    def test_text_delta_missing_delta_raises(self) -> None:
        with pytest.raises(ValueError, match="text_delta events must have a delta"):
            StreamEvent(event_type="text_delta", index=0)

    def test_function_call_missing_call_raises(self) -> None:
        with pytest.raises(ValueError, match="function_call events must have a function_call"):
            StreamEvent(event_type="function_call", index=0)

    def test_error_missing_message_raises(self) -> None:
        with pytest.raises(ValueError, match="error events must have an error message"):
            StreamEvent(event_type="error", index=0)

    def test_negative_index_raises(self) -> None:
        with pytest.raises(ValueError, match="index must be >= 0"):
            StreamEvent(event_type="done", index=-1, finish_reason="stop")

    def test_to_dict_text_delta(self) -> None:
        event = StreamEvent(event_type="text_delta", delta="Hello", index=0)
        d = event.to_dict()
        assert d["event_type"] == "text_delta"
        assert d["delta"] == "Hello"
        assert d["index"] == 0

    def test_to_dict_function_call(self) -> None:
        fc = FunctionCall(name="get_weather", arguments="{}")
        event = StreamEvent(event_type="function_call", function_call=fc, index=0)
        d = event.to_dict()
        assert d["function_call"]["name"] == "get_weather"

    def test_to_dict_done(self) -> None:
        event = StreamEvent(event_type="done", index=1, finish_reason="stop")
        d = event.to_dict()
        assert d["finish_reason"] == "stop"

    def test_from_dict_roundtrip_text(self) -> None:
        event1 = StreamEvent(event_type="text_delta", delta="Hi", index=0)
        d = event1.to_dict()
        event2 = StreamEvent.from_dict(d)
        assert event1 == event2

    def test_from_dict_roundtrip_function_call(self) -> None:
        fc = FunctionCall(name="get_weather", arguments='{"loc": "Mumbai"}')
        event1 = StreamEvent(event_type="function_call", function_call=fc, index=0)
        d = event1.to_dict()
        event2 = StreamEvent.from_dict(d)
        assert event1 == event2
        assert event2.function_call is not None
        assert event2.function_call.name == "get_weather"


class TestStreamRequest:
    def test_valid_minimal(self) -> None:
        req = StreamRequest(prompt="Hello")
        assert req.prompt == "Hello"
        assert req.max_tokens == 256
        assert req.temperature == 1.0
        assert req.functions == ()
        assert req.stream is True

    def test_empty_prompt_raises(self) -> None:
        with pytest.raises(ValueError, match="prompt must not be empty"):
            StreamRequest(prompt="")

    def test_zero_max_tokens_raises(self) -> None:
        with pytest.raises(ValueError, match="max_tokens must be >= 1"):
            StreamRequest(prompt="Hi", max_tokens=0)

    def test_negative_temperature_raises(self) -> None:
        with pytest.raises(ValueError, match="temperature must be >= 0.0"):
            StreamRequest(prompt="Hi", temperature=-1.0)

    def test_valid_with_functions(self) -> None:
        func = FunctionSpec(name="get_weather", parameters={"type": "object"})
        req = StreamRequest(prompt="Hi", functions=(func,))
        assert len(req.functions) == 1

    def test_to_dict_roundtrip(self) -> None:
        req1 = StreamRequest(
            prompt="Hello",
            max_tokens=128,
            temperature=0.7,
            functions=(FunctionSpec(name="test", parameters={"type": "object"}),),
        )
        d = req1.to_dict()
        req2 = StreamRequest.from_dict(d)
        assert req1.prompt == req2.prompt
        assert req1.max_tokens == req2.max_tokens
        assert len(req2.functions) == 1


class TestLocalStreamer:
    def test_text_generation_events(self) -> None:
        req = StreamRequest(prompt="Hello")
        streamer = LocalStreamer(req)
        events = streamer.generate()
        assert len(events) >= 2
        assert events[0].event_type == "text_delta"
        assert events[-1].event_type == "done"
        assert events[-1].finish_reason == "stop"

    def test_text_generation_ordering(self) -> None:
        req = StreamRequest(prompt="Hello")
        streamer = LocalStreamer(req)
        events = streamer.generate()
        for i, event in enumerate(events[:-1]):
            assert event.index == i

    def test_text_generation_delta_content(self) -> None:
        req = StreamRequest(prompt="Hello")
        streamer = LocalStreamer(req)
        events = streamer.generate()
        text_events = [e for e in events if e.event_type == "text_delta"]
        full_text = "".join(e.delta or "" for e in text_events)
        assert full_text == "Hello! How can I help you today?"

    def test_function_call_generation(self) -> None:
        func = FunctionSpec(
            name="get_weather",
            parameters={
                "type": "object",
                "properties": {"location": {"type": "string"}},
            },
        )
        req = StreamRequest(prompt="Weather?", functions=(func,))
        streamer = LocalStreamer(req)
        events = streamer.generate()
        assert len(events) == 2
        assert events[0].event_type == "function_call"
        assert events[0].function_call is not None
        assert events[0].function_call.name == "get_weather"
        assert events[1].event_type == "done"
        assert events[1].finish_reason == "function_call"

    def test_function_call_arguments_have_placeholders(self) -> None:
        func = FunctionSpec(
            name="get_weather",
            parameters={
                "type": "object",
                "properties": {"location": {"type": "string"}},
            },
        )
        req = StreamRequest(prompt="Weather?", functions=(func,))
        streamer = LocalStreamer(req)
        events = streamer.generate()
        args = json.loads(events[0].function_call.arguments)
        assert "location" in args
        assert args["location"] == "<location>"

    def test_deterministic_output(self) -> None:
        req = StreamRequest(prompt="Hi")
        events1 = LocalStreamer(req).generate()
        events2 = LocalStreamer(req).generate()
        assert events1 == events2

    def test_function_call_without_properties(self) -> None:
        func = FunctionSpec(name="get_time", parameters={"type": "object"})
        req = StreamRequest(prompt="Time?", functions=(func,))
        streamer = LocalStreamer(req)
        events = streamer.generate()
        assert events[0].event_type == "function_call"
        args = json.loads(events[0].function_call.arguments)
        assert args == {}


class TestSerialization:
    def test_stream_events_to_jsonl(self) -> None:
        events = [
            StreamEvent(event_type="text_delta", delta="Hello", index=0),
            StreamEvent(event_type="done", index=1, finish_reason="stop"),
        ]
        output = stream_events_to_jsonl(events)
        lines = output.strip().split("\n")
        assert len(lines) == 2
        for line in lines:
            data = json.loads(line)
            assert "event_type" in data

    def test_stream_events_to_json(self) -> None:
        events = [
            StreamEvent(event_type="text_delta", delta="Hi", index=0),
            StreamEvent(event_type="done", index=1, finish_reason="stop"),
        ]
        output = stream_events_to_json(events)
        data = json.loads(output)
        assert "events" in data
        assert "event_count" in data
        assert data["event_count"] == 2

    def test_deterministic_jsonl(self) -> None:
        events = [StreamEvent(event_type="done", index=0, finish_reason="stop")]
        assert stream_events_to_jsonl(events) == stream_events_to_jsonl(events)

    def test_empty_events(self) -> None:
        assert stream_events_to_jsonl([]) == "\n"
        data = json.loads(stream_events_to_json([]))
        assert data["events"] == []
        assert data["event_count"] == 0


class TestCLIIntegration:
    def test_text_stream_via_script(self) -> None:
        import sys

        from scripts.stream_local import main as stream_main

        original_argv = sys.argv
        try:
            sys.argv = ["stream_local.py", "--prompt", "Hello", "--format", "json"]
            stream_main()
        except SystemExit:
            pass
        finally:
            sys.argv = original_argv

    def test_json_output_format(self) -> None:
        import sys

        from scripts.stream_local import main as stream_main

        original_argv = sys.argv
        try:
            sys.argv = ["stream_local.py", "--prompt", "Hi", "--format", "json"]
            stream_main()
        except SystemExit:
            pass
        finally:
            sys.argv = original_argv

    def test_function_spec_file(self, tmp_path: Path) -> None:
        spec_path = tmp_path / "get_weather.json"
        spec_path.write_text(
            json.dumps(
                {
                    "name": "get_weather",
                    "description": "Get weather",
                    "parameters": {"type": "object", "properties": {"loc": {"type": "string"}}},
                }
            )
        )
        import sys

        from scripts.stream_local import main as stream_main

        original_argv = sys.argv
        try:
            sys.argv = [
                "stream_local.py",
                "--prompt",
                "Weather?",
                "--function",
                str(spec_path),
                "--format",
                "json",
            ]
            stream_main()
        except SystemExit:
            pass
        finally:
            sys.argv = original_argv

    def test_output_to_file(self, tmp_path: Path) -> None:
        output_file = tmp_path / "output.jsonl"
        import sys

        from scripts.stream_local import main as stream_main

        original_argv = sys.argv
        try:
            sys.argv = [
                "stream_local.py",
                "--prompt",
                "Hi",
                "--output",
                str(output_file),
            ]
            stream_main()
        except SystemExit:
            pass
        finally:
            sys.argv = original_argv
        assert output_file.exists()
        content = output_file.read_text()
        assert "text_delta" in content
