from __future__ import annotations

import io
from unittest.mock import MagicMock

from bharat.serving.openai_server import (
    BharatInferenceEngine,
    ChatMessage,
    create_openai_handler,
)
from scripts.start_api_server import parse_args


class TestOpenAIServer:
    def test_inference_engine_generation(self):
        engine = BharatInferenceEngine(tier="tiny", device="cpu")
        prompt = "User: Hello\n\nAssistant:"
        out = engine.generate(prompt, max_new_tokens=5)
        assert isinstance(out, str)

        chunks = list(engine.generate_stream(prompt, max_new_tokens=5))
        assert len(chunks) > 0

    def test_chat_prompt_formatting(self):
        engine = BharatInferenceEngine(tier="tiny", device="cpu")
        msgs = [
            ChatMessage(role="system", content="You are Bharat AI."),
            ChatMessage(role="user", content="What is ISRO?"),
        ]
        formatted = engine.format_chat_prompt(msgs)
        assert "System: You are Bharat AI." in formatted
        assert "User: What is ISRO?" in formatted
        assert formatted.endswith("Assistant: ")

    def test_handler_get_health(self):
        engine = BharatInferenceEngine(tier="tiny", device="cpu")
        handler_cls = create_openai_handler(engine, api_key="secret-123")

        # Mock request handler
        handler = handler_cls.__new__(handler_cls)
        handler.path = "/v1/health"
        handler.headers = {}
        handler.wfile = io.BytesIO()
        handler.send_response = MagicMock()
        handler.send_header = MagicMock()
        handler.end_headers = MagicMock()

        handler.do_GET()
        assert handler.send_response.called
        response_bytes = handler.wfile.getvalue()
        assert b'"status": "healthy"' in response_bytes

    def test_handler_get_models_authorized(self):
        engine = BharatInferenceEngine(tier="tiny", device="cpu")
        handler_cls = create_openai_handler(engine, api_key="secret-123")

        handler = handler_cls.__new__(handler_cls)
        handler.path = "/v1/models"
        handler.headers = {"Authorization": "Bearer secret-123"}
        handler.wfile = io.BytesIO()
        handler.send_response = MagicMock()
        handler.send_header = MagicMock()
        handler.end_headers = MagicMock()

        handler.do_GET()
        assert handler.send_response.called
        response_bytes = handler.wfile.getvalue()
        assert b"bharat-tiny" in response_bytes

    def test_handler_get_models_unauthorized(self):
        engine = BharatInferenceEngine(tier="tiny", device="cpu")
        handler_cls = create_openai_handler(engine, api_key="secret-123")

        handler = handler_cls.__new__(handler_cls)
        handler.path = "/v1/models"
        handler.headers = {"Authorization": "Bearer wrong-token"}
        handler.wfile = io.BytesIO()
        handler.send_response = MagicMock()
        handler.send_header = MagicMock()
        handler.end_headers = MagicMock()

        handler.do_GET()
        handler.send_response.assert_called_with(401)

    def test_cli_parse_args(self):
        args = parse_args(["--port", "9000", "--tier", "1b", "--api-key", "secret-token"])
        assert args.port == 9000
        assert args.tier == "1b"
        assert args.api_key == "secret-token"
