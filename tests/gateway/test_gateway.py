from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from bharat.gateway.server import create_app


class TestUniversalGateway:
    def test_gateway_endpoints(self, monkeypatch):
        monkeypatch.setenv("BHARAT_GATEWAY_API_TOKEN", "sample-32-character-test-token-only-123")
        headers = {"Authorization": "Bearer sample-32-character-test-token-only-123"}
        app = create_app(tier="tiny")
        client = TestClient(app)

        # Health endpoint
        res_h = client.get("/v1/health", headers=headers)
        assert res_h.status_code == 200
        data_h = res_h.json()
        assert data_h["status"] == "healthy"
        assert data_h["compute_mode"] == "UNIVERSAL_HYBRID"

        # Models endpoint
        res_m = client.get("/v1/models", headers=headers)
        assert res_m.status_code == 200
        assert len(res_m.json()["models"]) >= 2

        # Search endpoint
        res_s = client.post("/v1/search", json={"query": "ISRO", "top_k": 2}, headers=headers)
        assert res_s.status_code == 200
        assert len(res_s.json()["results"]) > 0

        # Chat endpoint
        res_c = client.post(
            "/v1/chat",
            json={
                "messages": [{"role": "user", "content": "What is Chandrayaan-3?"}],
                "max_tokens": 16,
            },
            headers=headers,
        )
        assert res_c.status_code == 200
        data_c = res_c.json()
        assert "response" in data_c
        assert "destination" in data_c

    def test_token_is_required(self, monkeypatch):
        monkeypatch.delenv("BHARAT_GATEWAY_API_TOKEN", raising=False)
        with pytest.raises(RuntimeError, match="BHARAT_GATEWAY_API_TOKEN"):
            create_app(tier="tiny")

    def test_unauthorized_request_rejected(self, monkeypatch):
        monkeypatch.setenv("BHARAT_GATEWAY_API_TOKEN", "sample-32-character-test-token-only-123")
        client = TestClient(create_app(tier="tiny"))
        assert client.get("/v1/models").status_code == 401
        response = client.get("/v1/metrics", headers={"Authorization": "Bearer wrong"})
        assert response.status_code == 401

    def test_private_request_cannot_force_cloud(self, monkeypatch):
        token = "sample-32-character-test-token-only-123"
        monkeypatch.setenv("BHARAT_GATEWAY_API_TOKEN", token)
        client = TestClient(create_app(tier="tiny"))
        response = client.post(
            "/v1/chat",
            headers={"Authorization": f"Bearer {token}"},
            json={
                "messages": [{"role": "user", "content": "Private internal data"}],
                "is_private": True,
                "force_destination": "AWS_BEDROCK",
            },
        )
        assert response.status_code == 403

    def test_live_web_does_not_serve_sample_facts(self, monkeypatch):
        token = "sample-32-character-test-token-only-123"
        monkeypatch.setenv("BHARAT_GATEWAY_API_TOKEN", token)
        client = TestClient(create_app(tier="tiny"))
        response = client.post(
            "/v1/chat",
            headers={"Authorization": f"Bearer {token}"},
            json={
                "messages": [{"role": "user", "content": "latest news today"}],
                "force_destination": "LIVE_WEB",
            },
        )
        assert response.status_code == 503

    def test_real_checkpoint_required_for_full_tiers(self, monkeypatch):
        monkeypatch.setenv("BHARAT_GATEWAY_API_TOKEN", "sample-32-character-test-token-only-123")
        with pytest.raises(FileNotFoundError, match="real checkpoint"):
            create_app(tier="1b", checkpoint_path=None)
