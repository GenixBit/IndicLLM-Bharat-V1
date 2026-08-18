from __future__ import annotations

import pytest
import torch
from fastapi.testclient import TestClient

from bharat.models.bharat_model import BharatForCausalLM
from bharat.models.config import BharatModelConfig
from inference.playground import (
    INDIC_LANGUAGE_PRESETS,
    create_playground_app,
    get_default_tokenizer,
    main,
    parse_args,
    synthesize_indic_response,
)


@pytest.fixture
def playground_client() -> TestClient:
    cfg = BharatModelConfig(
        vocab_size=256,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=128,
    )
    model = BharatForCausalLM(cfg)
    tokenizer = get_default_tokenizer()
    app = create_playground_app(
        model=model,
        config=cfg,
        tokenizer=tokenizer,
        device=torch.device("cpu"),
        model_name="Bharat-Test",
    )
    return TestClient(app)


class TestPlayground:
    def test_parse_args_defaults(self):
        args = parse_args([])
        assert args.model_size == "350m"
        assert args.host == "127.0.0.1"
        assert args.port == 7860
        assert args.device == "auto"

    def test_parse_args_explicit(self):
        args = parse_args(
            [
                "--model-size",
                "tiny",
                "--host",
                "0.0.0.0",
                "--port",
                "8080",
                "--device",
                "cpu",
            ]
        )
        assert args.model_size == "tiny"
        assert args.host == "0.0.0.0"
        assert args.port == 8080
        assert args.device == "cpu"

    def test_parse_args_10b_model(self):
        args = parse_args(["--model-size", "10b"])
        assert args.model_size == "10b"

    def test_index_page(self, playground_client: TestClient):
        res = playground_client.get("/")
        assert res.status_code == 200
        assert "text/html" in res.headers["content-type"]
        assert "IndicLLM-Bharat Playground" in res.text

    def test_api_health(self, playground_client: TestClient):
        res = playground_client.get("/api/health")
        assert res.status_code == 200
        data = res.json()
        assert data["status"] == "ok"
        assert data["model"] == "Bharat-Test"

    def test_api_info(self, playground_client: TestClient):
        res = playground_client.get("/api/info")
        assert res.status_code == 200
        data = res.json()
        assert data["model_name"] == "Bharat-Test"
        assert data["vocab_size"] == 256
        assert data["hidden_size"] == 64
        assert data["num_layers"] == 2

    def test_api_languages(self, playground_client: TestClient):
        res = playground_client.get("/api/languages")
        assert res.status_code == 200
        data = res.json()
        assert "hi" in data
        assert "ta" in data
        assert "te" in data
        assert "bn" in data
        assert len(data) == len(INDIC_LANGUAGE_PRESETS)

    def test_api_generate_streaming(self, playground_client: TestClient):
        payload = {
            "prompt": "नमस्ते",
            "system_prompt": "You are Bharat AI.",
            "temperature": 0.0,
            "max_tokens": 5,
        }
        res = playground_client.post("/api/generate", json=payload)
        assert res.status_code == 200
        assert "text/event-stream" in res.headers["content-type"]
        body = res.text
        assert "data: " in body
        assert '"done": true' in body.lower()

    def test_synthesize_indic_greetings(self):
        english_greeting = synthesize_indic_response("Hello")
        assert "IndicLLM-Bharat" in english_greeting
        assert "insightful query" not in english_greeting.lower()

        hi_greeting = synthesize_indic_response("Hi!")
        assert "IndicLLM-Bharat" in hi_greeting

        hindi_greeting = synthesize_indic_response("नमस्ते")
        assert "IndicLLM-Bharat" in hindi_greeting

    def test_synthesize_indic_persona_and_curriculum(self):
        persona = synthesize_indic_response("Who are you?")
        assert "IndicLLM-Bharat" in persona
        assert "GenixBit" in persona

        gqa_resp = synthesize_indic_response("Explain GQA and RoPE in modern LLMs")
        assert "Grouped-Query Attention" in gqa_resp
        assert "Rotary Position Embeddings" in gqa_resp

        math_resp = synthesize_indic_response("45 * 12")
        assert "540" in math_resp

    def test_main_missing_config_fails(self, capsys: pytest.CaptureFixture[str]):
        code = main(["--model-config", "/nonexistent/path.yaml"])
        assert code == 1
        captured = capsys.readouterr()
        assert "Error: Model config not found" in captured.err

    def test_main_missing_checkpoint_fails(self, capsys: pytest.CaptureFixture[str]):
        code = main(["--checkpoint", "/nonexistent/checkpoint.pt"])
        assert code == 1
        captured = capsys.readouterr()
        assert "Error: Checkpoint not found" in captured.err
