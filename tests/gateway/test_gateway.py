from __future__ import annotations

from fastapi.testclient import TestClient

from bharat.gateway.server import create_app


class TestUniversalGateway:
    def test_gateway_endpoints(self):
        app = create_app(tier="tiny")
        client = TestClient(app)

        # Health endpoint
        res_h = client.get("/v1/health")
        assert res_h.status_code == 200
        data_h = res_h.json()
        assert data_h["status"] == "healthy"
        assert data_h["compute_mode"] == "UNIVERSAL_HYBRID"

        # Models endpoint
        res_m = client.get("/v1/models")
        assert res_m.status_code == 200
        assert len(res_m.json()["models"]) >= 2

        # Search endpoint
        res_s = client.post("/v1/search", json={"query": "ISRO", "top_k": 2})
        assert res_s.status_code == 200
        assert len(res_s.json()["results"]) > 0

        # Chat endpoint
        res_c = client.post(
            "/v1/chat",
            json={
                "messages": [{"role": "user", "content": "What is Chandrayaan-3?"}],
                "max_tokens": 16,
            },
        )
        assert res_c.status_code == 200
        data_c = res_c.json()
        assert "response" in data_c
        assert "destination" in data_c
