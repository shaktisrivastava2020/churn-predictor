"""
Build the training dataset: features + labels per customer.
Saves to data/training.csv (gitignored).

Usage: python build_dataset.py
"""
from __future__ import annotations

import pandas as pd
from sqlalchemy import text

from database import get_engine
from features import build_features, FEATURE_COLUMNS
from labeling import compute_signals


# Customers excluded from training: insufficient history
MIN_TENURE_DAYS = 60
MIN_ORDERS = 2


def build_training_dataset(reference_date: pd.Timestamp | None = None) -> pd.DataFrame:
    if reference_date is None:
        reference_date = pd.Timestamp.now().normalize()

    engine = get_engine()
    with engine.connect() as conn:
        customers = pd.read_sql(
            text("SELECT customer_id, join_date, customer_tier FROM customers"), conn
        )
        orders = pd.read_sql(
            text(
                "SELECT customer_id, order_date, order_amount, product_id, "
                "payment_method, order_status FROM orders"
            ),
            conn,
        )

    feats = build_features(customers, orders, reference_date)
    labels = compute_signals(orders, reference_date)

    # Merge features with labels (inner join drops 0-order customers)
    dataset = feats.merge(
        labels[["customer_id", "churned", "signals_fired"]],
        on="customer_id",
        how="inner",
    )

    # Filter: only customers with enough history to have meaningful labels
    before = len(dataset)
    dataset = dataset[
        (dataset["tenure_days"] >= MIN_TENURE_DAYS)
        & (dataset["total_orders"] >= MIN_ORDERS)
    ].reset_index(drop=True)
    after = len(dataset)
    print(f"Filtered: {before} -> {after} customers (excluded {before - after} with insufficient history)")

    return dataset


if __name__ == "__main__":
    reference_date = pd.Timestamp("2026-04-29")
    df = build_training_dataset(reference_date)

    print()
    print(f"Final dataset: {df.shape[0]} customers x {df.shape[1]} columns")
    print(f"Churn rate: {df['churned'].mean():.1%}")
    print(f"Class balance: {df['churned'].value_counts().to_dict()}")

    output_path = "data/training.csv"
    df.to_csv(output_path, index=False)
    print(f"\n✅ Saved to {output_path}")
