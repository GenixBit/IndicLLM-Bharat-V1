from __future__ import annotations

from pathlib import Path

import pytest
import torch
from fastapi.testclient import TestClient

from bharat.models.bharat_model import BharatForCausalLM
from bharat.models.config import BharatModelConfig
from inference.api import app, load_model


@pytest.fixture(scope="module")
def bharat_checkpoint_for_api(tmp_path_factory: pytest.TempPathFactory) -> Path:
    tmp_path = tmp_path_factory.mktemp("api_test")
    config = BharatModelConfig(
        vocab_size=50257,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=128,
    )
    model = BharatForCausalLM(config)
    ckpt_file = tmp_path / "bharat-mini.pt"
    torch.save(
        {
            "model": model.state_dict(),
            "model_config": config.to_dict(),
            "step": 100,
        },
        ckpt_file,
    )
    return ckpt_file


@pytest.fixture(scope="module")
def client(bharat_checkpoint_for_api: Path) -> TestClient:
    load_model(bharat_checkpoint_for_api, device_override="cpu")
    return TestClient(app)


class TestInferenceAPI:
    def test_health_endpoint(self, client: TestClient) -> None:
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "healthy"
        assert data["device"] == "cpu"

    def test_list_models_endpoint(self, client: TestClient) -> None:
        resp = client.get("/v1/models")
        assert resp.status_code == 200
        data = resp.json()
        assert data["object"] == "list"
        assert len(data["data"]) == 1
        assert data["data"][0]["id"] == "bharat-mini"

    def test_chat_completions_json(self, client: TestClient) -> None:
        payload = {
            "messages": [
                {"role": "user", "content": "नमस्ते"},
            ],
            "max_tokens": 5,
            "temperature": 0.8,
            "stream": False,
        }
        resp = client.post("/v1/chat/completions", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert data["object"] == "chat.completion"
        assert len(data["choices"]) == 1
        assert data["choices"][0]["message"]["role"] == "assistant"
        assert "usage" in data
        assert data["usage"]["total_tokens"] > 0

    def test_chat_completions_streaming(self, client: TestClient) -> None:
        payload = {
            "messages": [
                {"role": "user", "content": "भारत"},
            ],
            "max_tokens": 5,
            "temperature": 0.8,
            "stream": True,
        }
        resp = client.post("/v1/chat/completions", json=payload)
        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers["content-type"]
        body = resp.text
        assert "data: " in body
        assert "[DONE]" in body

    def test_completions_endpoint(self, client: TestClient) -> None:
        payload = {
            "prompt": "भारत एक",
            "max_tokens": 5,
            "temperature": 0.8,
        }
        resp = client.post("/v1/completions", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert data["object"] == "text_completion"
        assert len(data["choices"]) == 1
        assert "text" in data["choices"][0]
