from __future__ import annotations

from fastapi.testclient import TestClient

from api.main import app
from api.services import model_service


class DummyModel:
    def __init__(self, prob: float = 0.8) -> None:
        self._prob = prob

    def predict_proba(self, X):
        import numpy as np

        n = len(X)
        probs_1 = np.full(shape=(n,), fill_value=self._prob, dtype=float)
        probs_0 = 1.0 - probs_1
        return np.column_stack([probs_0, probs_1])


class DummyScaler:
    def transform(self, X):
        return X


def setup_dummy_service(prob: float = 0.8):
    model_service.model = DummyModel(prob=prob)
    model_service.scaler = DummyScaler()
    model_service.feature_cols = [
        "DSRI",
        "GMI",
        "AQI",
        "SGI",
        "DEPI",
        "SGAI",
        "LVGI",
        "TATA",
        "RSST_Accruals",
        "Delta_Receivables",
        "Delta_Inventory",
        "Delta_Cash_Sales",
    ]
    model_service.model_name = "DummyModel"


def valid_payload() -> dict:
    return {
        "DSRI": 1.12,
        "GMI": 0.98,
        "AQI": 1.05,
        "SGI": 1.20,
        "DEPI": 0.95,
        "SGAI": 1.08,
        "LVGI": 1.10,
        "TATA": 0.03,
        "RSST_Accruals": 0.02,
        "Delta_Receivables": 0.01,
        "Delta_Inventory": 0.02,
        "Delta_Cash_Sales": 0.15,
    }


client = TestClient(app)


def test_root_endpoint():
    response = client.get("/")

    assert response.status_code == 200
    data = response.json()
    assert "message" in data
    assert "docs" in data


def test_health_endpoint_with_dummy_model():
    setup_dummy_service(prob=0.8)

    response = client.get("/health")

    assert response.status_code == 200
    data = response.json()
    assert data["status"] in {"ok", "degraded"}
    assert data["model_loaded"] is True
    assert data["scaler_loaded"] is True
    assert data["feature_count"] == 12
    assert data["model_name"] == "DummyModel"


def test_predict_endpoint_returns_fraud_when_probability_above_threshold():
    setup_dummy_service(prob=0.8)

    response = client.post("/predict", json=valid_payload())

    assert response.status_code == 200
    data = response.json()
    assert data["label"] == "Fraud"
    assert data["fraud_prediction"] == 1
    assert 0.0 <= data["fraud_probability"] <= 1.0
    assert data["model_name"] == "DummyModel"


def test_predict_endpoint_returns_non_fraud_when_probability_below_threshold():
    setup_dummy_service(prob=0.2)

    response = client.post("/predict", json=valid_payload())

    assert response.status_code == 200
    data = response.json()
    assert data["label"] == "Non-Fraud"
    assert data["fraud_prediction"] == 0
    assert 0.0 <= data["fraud_probability"] <= 1.0


def test_predict_endpoint_returns_422_for_missing_required_field():
    setup_dummy_service(prob=0.8)

    bad_payload = valid_payload()
    bad_payload.pop("DSRI")

    response = client.post("/predict", json=bad_payload)

    assert response.status_code == 422


def test_predict_endpoint_returns_503_when_model_not_loaded():
    model_service.model = None
    model_service.scaler = None
    model_service.feature_cols = []
    model_service.model_name = "unknown"

    response = client.post("/predict", json=valid_payload())

    assert response.status_code == 503