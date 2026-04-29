"""
PyTorch model architecture for churn prediction.

A small feed-forward neural network designed for tabular data:
  - 12 input features
  - 2 hidden layers (32 -> 16 units) with ReLU + BatchNorm + Dropout
  - 1 output (sigmoid for binary classification probability)

Architecture choices:
  - Small network (small dataset: 176 customers)
  - Dropout 0.3 prevents overfitting
  - BatchNorm stabilizes training
  - Sigmoid output gives churn probability in [0, 1]
"""
from __future__ import annotations

import torch
import torch.nn as nn


class ChurnNet(nn.Module):
    """Feed-forward network for binary churn classification."""

    def __init__(self, input_dim: int = 12, hidden1: int = 32, hidden2: int = 16, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden1),
            nn.BatchNorm1d(hidden1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden1, hidden2),
            nn.BatchNorm1d(hidden2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns logits (apply sigmoid externally for probability)."""
        return self.net(x)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Returns churn probability in [0, 1]."""
        self.eval()
        with torch.no_grad():
            return torch.sigmoid(self.forward(x))
