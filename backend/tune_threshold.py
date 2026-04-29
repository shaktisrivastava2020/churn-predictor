"""
Threshold tuning for the trained churn model.

Doesn't retrain — just sweeps decision thresholds and reports metrics.
The 0.5 default is rarely optimal for imbalanced classes.
"""
from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from features import FEATURE_COLUMNS
from model import ChurnNet

SEED = 42
DATA_PATH = "data/training.csv"
MODEL_PATH = "models/churn_model.pth"
SCALER_PATH = "models/scaler.joblib"


def main():
    df = pd.read_csv(DATA_PATH)
    X = df[FEATURE_COLUMNS].values.astype(np.float32)
    y = df["churned"].values.astype(np.float32)

    # Reproduce same test split as training (same seed)
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)

    scaler = joblib.load(SCALER_PATH)
    X_test_scaled = scaler.transform(X_test)

    model = ChurnNet(input_dim=len(FEATURE_COLUMNS))
    model.load_state_dict(torch.load(MODEL_PATH, weights_only=True))
    model.eval()

    with torch.no_grad():
        logits = model(torch.from_numpy(X_test_scaled).float()).squeeze(-1)
        probs = torch.sigmoid(logits).numpy()

    auc = roc_auc_score(y_test, probs)

    print(f"{'Threshold':>10} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10}")
    print("-" * 55)

    best_f1 = -1.0
    best_threshold = 0.5
    rows = []
    for t in np.arange(0.20, 0.55, 0.05):
        preds = (probs >= t).astype(int)
        acc = accuracy_score(y_test, preds)
        prec = precision_score(y_test, preds, zero_division=0)
        rec = recall_score(y_test, preds, zero_division=0)
        f1 = f1_score(y_test, preds, zero_division=0)
        print(f"{t:>10.2f} {acc:>10.3f} {prec:>10.3f} {rec:>10.3f} {f1:>10.3f}")
        rows.append({"threshold": float(t), "accuracy": float(acc), "precision": float(prec), "recall": float(rec), "f1": float(f1)})
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = float(t)

    print()
    print(f"ROC-AUC (threshold-independent): {auc:.3f}")
    print(f"Best F1 = {best_f1:.3f} at threshold = {best_threshold:.2f}")

    with open("models/threshold_sweep.json", "w") as f:
        json.dump({"sweep": rows, "best_threshold": best_threshold, "best_f1": best_f1, "roc_auc": float(auc)}, f, indent=2)
    print("\n✅ Saved: models/threshold_sweep.json")


if __name__ == "__main__":
    main()
