"""
tests/test_app.py
==================
Integration tests for the FastAPI fraud scoring service.

Uses httpx AsyncClient (ASGI transport) – no real server started.
The model registry is patched to avoid needing real model artefacts.
"""

from __future__ import annotations

import numpy as np
import pytest
import pytest_asyncio
from unittest.mock import MagicMock, patch, AsyncMock
from httpx import AsyncClient, ASGITransport


# ---------------------------------------------------------------------------
# Helpers / mocks
# ---------------------------------------------------------------------------

def _make_mock_model(proba: float = 0.85):
    """Return a mock sklearn-compatible model that always returns `proba`."""
    model = MagicMock()
    model.predict_proba.return_value = np.array([[1 - proba, proba]])
    return model


SAMPLE_FEATURES = {
    "V1": -1.36,  "V2": -0.073, "V3": 2.536,
    "V4": 1.378,  "V5": -0.338, "Amount_log1p": 5.42,
    "hour_sin": 0.87, "hour_cos": 0.49,
}


# ---------------------------------------------------------------------------
# App fixture with mocked registry
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def patch_registry():
    """
    Replace the model registry with a lightweight mock so no disk I/O
    or real models are needed during tests.
    """
    mock_model = _make_mock_model(proba=0.85)

    with patch("app.REGISTRY") as mock_registry:
        mock_registry.primary_model  = mock_model
        mock_registry.model_name     = "stacking_ensemble"
        mock_registry.threshold      = 0.5
        mock_registry.version        = "test-1.0.0"
        mock_registry.feature_names  = list(SAMPLE_FEATURES.keys())
        mock_registry.explainer      = None   # disable SHAP to speed up tests
        mock_registry.background_data = None
        yield mock_registry


@pytest_asyncio.fixture()
async def client():
    """Async ASGI test client – bypasses HTTP stack entirely."""
    # Import AFTER patching to pick up the mock
    from app import app

    # Override lifespan to skip model loading
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as c:
        yield c


# ---------------------------------------------------------------------------
# Health / info endpoints
# ---------------------------------------------------------------------------

class TestHealthEndpoint:
    @pytest.mark.asyncio
    async def test_health_returns_200(self, client):
        resp = await client.get("/health")
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_health_body_has_status_ok(self, client):
        resp = await client.get("/health")
        data = resp.json()
        assert data["status"] == "ok"

    @pytest.mark.asyncio
    async def test_health_includes_model_name(self, client):
        resp = await client.get("/health")
        assert "model" in resp.json()


class TestModelInfoEndpoint:
    @pytest.mark.asyncio
    async def test_model_info_200(self, client):
        resp = await client.get("/model/info")
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_model_info_has_threshold(self, client):
        resp = await client.get("/model/info")
        assert "threshold" in resp.json()


# ---------------------------------------------------------------------------
# Single score endpoint
# ---------------------------------------------------------------------------

class TestScoreEndpoint:
    @pytest.mark.asyncio
    async def test_score_returns_200(self, client):
        resp = await client.post("/score", json={"features": SAMPLE_FEATURES})
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_score_response_schema(self, client):
        resp = await client.post("/score", json={"features": SAMPLE_FEATURES})
        data = resp.json()
        assert "risk_score" in data
        assert "fraud_flag" in data
        assert "threshold" in data
        assert "shap_top3" in data
        assert "model_version" in data
        assert "latency_ms" in data

    @pytest.mark.asyncio
    async def test_risk_score_bounded(self, client):
        resp = await client.post("/score", json={"features": SAMPLE_FEATURES})
        score = resp.json()["risk_score"]
        assert 0.0 <= score <= 1.0

    @pytest.mark.asyncio
    async def test_fraud_flag_true_above_threshold(self, client, patch_registry):
        """Score = 0.85 > threshold = 0.5 → fraud_flag should be True."""
        patch_registry.threshold = 0.5
        resp = await client.post("/score", json={"features": SAMPLE_FEATURES})
        assert resp.json()["fraud_flag"] is True

    @pytest.mark.asyncio
    async def test_fraud_flag_false_below_threshold(self, client, patch_registry):
        """Score = 0.85 < threshold = 0.99 → fraud_flag should be False."""
        patch_registry.threshold = 0.99
        patch_registry.primary_model.predict_proba.return_value = np.array([[0.2, 0.2]])
        resp = await client.post("/score", json={"features": SAMPLE_FEATURES})
        assert resp.json()["fraud_flag"] is False

    @pytest.mark.asyncio
    async def test_empty_features_returns_422(self, client):
        resp = await client.post("/score", json={"features": {}})
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_missing_features_key_returns_422(self, client):
        resp = await client.post("/score", json={})
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_model_version_present(self, client):
        resp = await client.post("/score", json={"features": SAMPLE_FEATURES})
        assert resp.json()["model_version"] == "test-1.0.0"

    @pytest.mark.asyncio
    async def test_latency_ms_positive(self, client):
        resp = await client.post("/score", json={"features": SAMPLE_FEATURES})
        assert resp.json()["latency_ms"] >= 0.0


# ---------------------------------------------------------------------------
# Batch score endpoint
# ---------------------------------------------------------------------------

class TestBatchScoreEndpoint:
    def _make_batch(self, n: int):
        return [
            {"transaction_id": f"tx_{i:04d}", "features": SAMPLE_FEATURES}
            for i in range(n)
        ]

    @pytest.mark.asyncio
    async def test_batch_score_returns_200(self, client):
        resp = await client.post("/score/batch", json=self._make_batch(5))
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_batch_result_count_matches(self, client):
        batch_size = 10
        resp = await client.post("/score/batch", json=self._make_batch(batch_size))
        data = resp.json()
        assert data["batch_size"] == batch_size
        assert len(data["results"]) == batch_size

    @pytest.mark.asyncio
    async def test_batch_too_large_returns_422(self, client):
        resp = await client.post("/score/batch", json=self._make_batch(501))
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_batch_total_latency_present(self, client):
        resp = await client.post("/score/batch", json=self._make_batch(3))
        assert "total_latency_ms" in resp.json()

    @pytest.mark.asyncio
    async def test_batch_transaction_ids_preserved(self, client):
        batch = self._make_batch(3)
        resp = await client.post("/score/batch", json=batch)
        returned_ids = [r["transaction_id"] for r in resp.json()["results"]]
        expected_ids = [b["transaction_id"] for b in batch]
        assert returned_ids == expected_ids


# ---------------------------------------------------------------------------
# Metrics endpoint
# ---------------------------------------------------------------------------

class TestMetricsEndpoint:
    @pytest.mark.asyncio
    async def test_metrics_returns_200(self, client):
        resp = await client.get("/metrics")
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_metrics_content_type_text(self, client):
        resp = await client.get("/metrics")
        assert "text/plain" in resp.headers["content-type"]

    @pytest.mark.asyncio
    async def test_metrics_contains_counter(self, client):
        resp = await client.get("/metrics")
        assert "fraud_requests_total" in resp.text
