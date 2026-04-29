"""
Inference wrapper for the churn prediction model.

Loads model + scaler at startup and provides a clean predict() interface.
This module is the boundary between trained-model artifacts and the API.
"""
from __future__ import annotations

import json
import os
from typing import Any

import joblib
import numpy as np
import torch

from features import FEATURE_COLUMNS
from model import ChurnNet

MODEL_PATH = "models/churn_model.pth"
SCALER_PATH = "models/scaler.joblib"
METADATA_PATH = "models/metadata.json"


class ChurnPredictor:
    """Loads model artifacts once and exposes prediction methods."""

    def __init__(
        self,
        model_path: str = MODEL_PATH,
        scaler_path: str = SCALER_PATH,
        metadata_path: str = METADATA_PATH,
    ):
        # Load metadata first (gives us input_dim + version info)
        with open(metadata_path) as f:
            self.metadata: dict[str, Any] = json.load(f)

        self.feature_columns: list[str] = self.metadata["feature_columns"]

        # Load scaler
        self.scaler = joblib.load(scaler_path)

        # Load model with architecture from metadata (supports v1 and v2)
        hp = self.metadata.get("hyperparameters", {})
        self.model = ChurnNet(
            input_dim=self.metadata["input_dim"],
            hidden1=hp.get("hidden1", 32),
            hidden2=hp.get("hidden2", 16),
            dropout=hp.get("dropout", 0.3),
        )
        self.model.load_state_dict(torch.load(model_path, weights_only=True))
        self.model.eval()

    def predict_one(self, features: dict[str, float], threshold: float | None = None) -> dict[str, Any]:
        """
        Predict churn for a single customer.

        Args:
            features: dict of feature_name -> value (must match FEATURE_COLUMNS)
            threshold: probability threshold for binary classification

        Returns:
            dict with churn_probability, churned_label, threshold_used
        """
        # Use config default threshold if not specified
        if threshold is None:
            from config import settings
            threshold = settings.CHURN_THRESHOLD

        # Validate all features present
        missing = set(self.feature_columns) - set(features.keys())
        if missing:
            raise ValueError(f"Missing features: {sorted(missing)}")

        # Build input vector in the exact training order
        x = np.array([[features[col] for col in self.feature_columns]], dtype=np.float32)
        x_scaled = self.scaler.transform(x)

        with torch.no_grad():
            logits = self.model(torch.from_numpy(x_scaled).float()).squeeze(-1)
            prob = torch.sigmoid(logits).item()

        return {
            "churn_probability": round(prob, 4),
            "churned_label": int(prob >= threshold),
            "threshold_used": threshold,
        }

    def predict_batch(self, batch: list[dict[str, float]], threshold: float | None = None) -> list[dict[str, Any]]:
        """Predict for a list of customers."""
        return [self.predict_one(f, threshold) for f in batch]

    def info(self) -> dict[str, Any]:
        """Return model metadata for /metrics endpoint."""
        return {
            "trained_at": self.metadata.get("trained_at"),
            "input_dim": self.metadata.get("input_dim"),
            "feature_columns": self.feature_columns,
            "best_epoch": self.metadata.get("best_epoch"),
            "n_train": self.metadata.get("n_train"),
            "n_test": self.metadata.get("n_test"),
        }


# Singleton instance — loaded once on import
_predictor: ChurnPredictor | None = None


def get_predictor() -> ChurnPredictor:
    """Lazy-init singleton. Used as FastAPI dependency."""
    global _predictor
    if _predictor is None:
        _predictor = ChurnPredictor()
    return _predictor
