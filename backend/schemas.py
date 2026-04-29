"""
Pydantic schemas for API request/response validation.

Defines the JSON contract for the churn prediction endpoints.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class CustomerFeatures(BaseModel):
    """Input features for a single customer prediction."""

    tenure_days: float = Field(..., ge=0, description="Days since customer signup")
    total_orders: int = Field(..., ge=0, description="Lifetime order count")
    total_spend: float = Field(..., ge=0, description="Lifetime spend in INR")
    avg_order_value: float = Field(..., ge=0, description="Average order value in INR")
    days_since_last_order: float = Field(..., ge=0, description="Days since most recent order")
    orders_per_month: float = Field(..., ge=0, description="Order frequency")
    unique_products: int = Field(..., ge=0, description="Distinct products purchased")
    unique_payment_methods: int = Field(..., ge=0, description="Distinct payment methods used")
    weekday_order_ratio: float = Field(..., ge=0, le=1, description="Fraction of orders on weekdays")
    tier_standard: Literal[0, 1] = Field(..., description="One-hot: customer in Standard tier")
    tier_premium: Literal[0, 1] = Field(..., description="One-hot: customer in Premium tier")
    tier_gold: Literal[0, 1] = Field(..., description="One-hot: customer in Gold tier")

    model_config = {
        "json_schema_extra": {
            "example": {
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
        }
    }


class PredictionResponse(BaseModel):
    """Single-customer prediction response."""

    churn_probability: float = Field(..., ge=0, le=1, description="Probability customer will churn")
    churned_label: Literal[0, 1] = Field(..., description="Binary classification (0=active, 1=churned)")
    threshold_used: float = Field(..., description="Probability threshold used for the binary label")


class BatchPredictionRequest(BaseModel):
    """Batch input — list of customer feature sets."""

    customers: list[CustomerFeatures]
    threshold: float | None = Field(default=None, ge=0, le=1, description="Override default threshold (uses config default if omitted)")


class BatchPredictionResponse(BaseModel):
    """Batch output."""

    predictions: list[PredictionResponse]
    count: int


class HealthResponse(BaseModel):
    """Health check response."""

    status: Literal["healthy"] = "healthy"
    service: str = "churn-predictor"
    version: str


class ModelInfoResponse(BaseModel):
    """Model metadata for /metrics endpoint."""

    trained_at: str | None
    input_dim: int
    feature_columns: list[str]
    best_epoch: int | None
    n_train: int | None
    n_test: int | None
