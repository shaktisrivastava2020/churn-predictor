"""
Evaluation script for the trained churn model.

Loads model + scaler + test data, computes metrics, prints report.
Saves metrics to models/eval_report.json for portfolio/audit purposes.

Usage: python evaluate.py
"""
from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
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
REPORT_PATH = "models/eval_report.json"
THRESHOLD = 0.5


def load_test_set():
    """Reproduce the exact same test split used in training (same seed)."""
    df = pd.read_csv(DATA_PATH)
    X = df[FEATURE_COLUMNS].values.astype(np.float32)
    y = df["churned"].values.astype(np.float32)

    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEED
    )
    return X_test, y_test


def main():
    X_test, y_test = load_test_set()
    scaler = joblib.load(SCALER_PATH)
    X_test_scaled = scaler.transform(X_test)

    model = ChurnNet(input_dim=len(FEATURE_COLUMNS))
    model.load_state_dict(torch.load(MODEL_PATH, weights_only=True))
    model.eval()

    with torch.no_grad():
        logits = model(torch.from_numpy(X_test_scaled).float()).squeeze(-1)
        probs = torch.sigmoid(logits).numpy()

    preds = (probs >= THRESHOLD).astype(int)

    metrics = {
        "n_test": int(len(y_test)),
        "threshold": THRESHOLD,
        "accuracy": float(accuracy_score(y_test, preds)),
        "precision": float(precision_score(y_test, preds, zero_division=0)),
        "recall": float(recall_score(y_test, preds, zero_division=0)),
        "f1": float(f1_score(y_test, preds, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, probs)),
    }

    cm = confusion_matrix(y_test, preds, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    metrics["confusion_matrix"] = {
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
    }

    print("=" * 60)
    print("CHURN MODEL — TEST SET EVALUATION")
    print("=" * 60)
    print(f"Test samples: {metrics['n_test']}")
    print(f"Threshold:    {metrics['threshold']}")
    print()
    print(f"Accuracy:     {metrics['accuracy']:.3f}")
    print(f"Precision:    {metrics['precision']:.3f}")
    print(f"Recall:       {metrics['recall']:.3f}")
    print(f"F1:           {metrics['f1']:.3f}")
    print(f"ROC-AUC:      {metrics['roc_auc']:.3f}")
    print()
    print("Confusion Matrix:")
    print(f"                 Predicted Active  Predicted Churned")
    print(f"  Actual Active        {tn:4d}             {fp:4d}")
    print(f"  Actual Churned       {fn:4d}             {tp:4d}")
    print()

    with open(REPORT_PATH, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"✅ Report saved: {REPORT_PATH}")


if __name__ == "__main__":
    main()
