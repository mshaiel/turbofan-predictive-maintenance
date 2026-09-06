"""
Unit tests for Antigravity REST API endpoints (/health, /predict).
"""

from unittest.mock import MagicMock, patch
import pytest
from api.app import app


@pytest.fixture
def client():
    """Create Flask test client."""
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


def test_health_endpoint(client):
    """Verify GET /health returns 200 and valid status."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.get_json()
    assert data["status"] == "ok"
    assert "model_loaded" in data


def test_predict_empty_payload(client):
    """Verify POST /predict returns 400 when body is empty."""
    response = client.post("/predict", json={})
    assert response.status_code == 400
    data = response.get_json()
    assert "error" in data


def test_predict_empty_cycles(client):
    """Verify POST /predict returns 400 when cycles list is empty."""
    response = client.post("/predict", json={"engine_id": 1, "cycles": []})
    assert response.status_code == 400
    data = response.get_json()
    assert "error" in data


@patch("api.app.inference_engine")
def test_predict_successful_mock(mock_engine, client):
    """Verify POST /predict returns 200 with schema-compliant output."""
    mock_engine.is_ready = True
    mock_engine.predict.return_value = {
        "engine_id": 42,
        "predicted_rul": 37.2,
        "alert_level": "WARNING",
        "top_shap_contributors": [
            {"feature": "s12_mean_10", "shap_value": -18.4, "direction": "accelerates_failure"},
            {"feature": "s7_std_10", "shap_value": -11.2, "direction": "accelerates_failure"},
            {"feature": "cycle_norm", "shap_value": -8.9, "direction": "accelerates_failure"}
        ],
        "confidence_note": "Within degradation zone (RUL <= 50)"
    }

    payload = {
        "engine_id": 42,
        "cycles": [
            {"cycle": 1, "op_setting_1": 0.0, "op_setting_2": 0.0, "op_setting_3": 100.0, "s2": 642.0}
        ]
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.get_json()
    assert data["engine_id"] == 42
    assert data["predicted_rul"] == 37.2
    assert data["alert_level"] == "WARNING"
    assert len(data["top_shap_contributors"]) == 3
