"""
Training script for the churn prediction model.

Reads data/training.csv, trains ChurnNet with stratified splits,
saves model weights + scaler + metadata to models/.

Usage: python train.py
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from features import FEATURE_COLUMNS
from model import ChurnNet

# Reproducibility
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

# Hyperparameters
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
MAX_EPOCHS = 100
EARLY_STOPPING_PATIENCE = 15
WEIGHT_DECAY = 1e-4

DATA_PATH = "data/training.csv"
MODEL_DIR = "models"
MODEL_PATH = f"{MODEL_DIR}/churn_model.pth"
SCALER_PATH = f"{MODEL_DIR}/scaler.joblib"
METADATA_PATH = f"{MODEL_DIR}/metadata.json"


def load_data():
    df = pd.read_csv(DATA_PATH)
    X = df[FEATURE_COLUMNS].values.astype(np.float32)
    y = df["churned"].values.astype(np.float32)
    return X, y


def make_loaders(X, y):
    # 60% train, 20% val, 20% test (stratified)
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEED
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.25, stratify=y_temp, random_state=SEED
    )

    scaler = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    def to_loader(X, y, shuffle):
        ds = TensorDataset(torch.from_numpy(X).float(), torch.from_numpy(y).float())
        return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=shuffle)

    return (
        to_loader(X_train, y_train, True),
        to_loader(X_val, y_val, False),
        to_loader(X_test, y_test, False),
        scaler,
        (len(y_train), len(y_val), len(y_test)),
        y_train,
    )


def train_one_epoch(model, loader, optimizer, loss_fn):
    model.train()
    total_loss = 0.0
    for xb, yb in loader:
        optimizer.zero_grad()
        logits = model(xb).squeeze(-1)
        loss = loss_fn(logits, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(xb)
    return total_loss / len(loader.dataset)


def evaluate_loss(model, loader, loss_fn):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for xb, yb in loader:
            logits = model(xb).squeeze(-1)
            loss = loss_fn(logits, yb)
            total_loss += loss.item() * len(xb)
    return total_loss / len(loader.dataset)


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)
    X, y = load_data()
    train_loader, val_loader, test_loader, scaler, sizes, y_train = make_loaders(X, y)
    print(f"Splits — train: {sizes[0]}, val: {sizes[1]}, test: {sizes[2]}")
    print(f"Train churn rate: {y_train.mean():.1%}")

    # Class weight for imbalance: pos_weight = neg / pos
    pos_weight = torch.tensor([(1 - y_train.mean()) / y_train.mean()], dtype=torch.float32)
    print(f"Pos weight (for loss): {pos_weight.item():.2f}")

    model = ChurnNet(input_dim=len(FEATURE_COLUMNS))
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    best_val_loss = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0

    for epoch in range(1, MAX_EPOCHS + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn)
        val_loss = evaluate_loss(model, val_loader, loss_fn)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save(model.state_dict(), MODEL_PATH)
        else:
            epochs_without_improvement += 1

        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d} — train: {train_loss:.4f}, val: {val_loss:.4f}")

        if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
            print(f"Early stopping at epoch {epoch} (best epoch: {best_epoch})")
            break

    # Save scaler + metadata
    joblib.dump(scaler, SCALER_PATH)
    metadata = {
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "feature_columns": FEATURE_COLUMNS,
        "input_dim": len(FEATURE_COLUMNS),
        "best_epoch": best_epoch,
        "best_val_loss": float(best_val_loss),
        "pos_weight": float(pos_weight.item()),
        "n_train": sizes[0],
        "n_val": sizes[1],
        "n_test": sizes[2],
        "train_churn_rate": float(y_train.mean()),
        "seed": SEED,
        "hyperparameters": {
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "max_epochs": MAX_EPOCHS,
            "early_stopping_patience": EARLY_STOPPING_PATIENCE,
            "weight_decay": WEIGHT_DECAY,
        },
    }
    with open(METADATA_PATH, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\n✅ Model saved: {MODEL_PATH}")
    print(f"✅ Scaler saved: {SCALER_PATH}")
    print(f"✅ Metadata saved: {METADATA_PATH}")


if __name__ == "__main__":
    main()
