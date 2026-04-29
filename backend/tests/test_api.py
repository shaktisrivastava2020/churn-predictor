"""Integration tests for the FastAPI churn prediction service."""
import os
import sys

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import app

client = TestClient(app)


VALID_FEATURES = {
    "tenure_days": 365,
    "total_orders": 8,
    "total_spend": 80000,
    "avg_order_value": 10000,
    "days_since_last_order": 15,
    "orders_per_month": 2.0,
    "unique_products": 6,
    "unique_payment_methods": 3,
    "weekday_order_ratio": 0.7,
    "tier_standard": 0,
    "tier_premium": 1,
    "tier_gold": 0,
}


def test_root():
    """Root endpoint returns service info."""
    r = client.get("/")
    assert r.status_code == 200
    assert "service" in r.json()


def test_health():
    """Health endpoint returns healthy status."""
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "healthy"


def test_model_info():
    """Model info exposes 12 features and metadata."""
    r = client.get("/model/info")
    assert r.status_code == 200
    body = r.json()
    assert body["input_dim"] == 12
    assert len(body["feature_columns"]) == 12


def test_predict_valid():
    """Valid input returns probability + label."""
    r = client.post("/predict", json=VALID_FEATURES)
    assert r.status_code == 200
    body = r.json()
    assert 0 <= body["churn_probability"] <= 1
    assert body["churned_label"] in [0, 1]


def test_predict_negative_tenure_rejected():
    """Negative tenure should fail validation."""
    bad = {**VALID_FEATURES, "tenure_days": -10}
    r = client.post("/predict", json=bad)
    assert r.status_code == 422


def test_predict_missing_field_rejected():
    """Missing required field should fail validation."""
    bad = {k: v for k, v in VALID_FEATURES.items() if k != "tenure_days"}
    r = client.post("/predict", json=bad)
    assert r.status_code == 422


def test_predict_weekday_ratio_out_of_range():
    """weekday_order_ratio > 1 should fail validation."""
    bad = {**VALID_FEATURES, "weekday_order_ratio": 1.5}
    r = client.post("/predict", json=bad)
    assert r.status_code == 422


def test_predict_batch():
    """Batch endpoint returns multiple predictions."""
    r = client.post(
        "/predict/batch",
        json={"threshold": 0.4, "customers": [VALID_FEATURES, VALID_FEATURES]},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["count"] == 2
    assert len(body["predictions"]) == 2
    assert all(p["threshold_used"] == 0.4 for p in body["predictions"])
