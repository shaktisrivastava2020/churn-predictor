"""
Feature engineering for churn prediction.

Builds a per-customer feature matrix from raw customer + order data.
Features are designed to be DIFFERENT from labeling signals to avoid
trivial leakage (model memorizing the labeling rule).
"""
from __future__ import annotations

import pandas as pd

# Customer tier categories (one-hot encoded)
TIER_CATEGORIES = ["Standard", "Premium", "Gold"]


def build_features(
    customers_df: pd.DataFrame,
    orders_df: pd.DataFrame,
    reference_date: pd.Timestamp,
) -> pd.DataFrame:
    """
    Build per-customer feature matrix.

    Args:
        customers_df: [customer_id, join_date, customer_tier]
        orders_df:    [customer_id, order_date, order_amount, product_id, payment_method, order_status]
        reference_date: the 'today' for feature computation.

    Returns:
        DataFrame with one row per customer and engineered features.
    """
    cust = customers_df.copy()
    cust["join_date"] = pd.to_datetime(cust["join_date"])

    orders = orders_df.copy()
    orders["order_date"] = pd.to_datetime(orders["order_date"])

    # Initialize feature frame keyed by customer_id
    feats = pd.DataFrame({"customer_id": cust["customer_id"].unique()}).set_index("customer_id")

    # Tenure (days since join)
    join_lookup = cust.set_index("customer_id")["join_date"]
    feats["tenure_days"] = (reference_date - join_lookup.reindex(feats.index)).dt.days

    # Lifetime aggregates
    feats["total_orders"] = orders.groupby("customer_id").size().reindex(feats.index, fill_value=0)
    feats["total_spend"] = (
        orders.groupby("customer_id")["order_amount"].sum().reindex(feats.index, fill_value=0).astype(float)
    )
    feats["avg_order_value"] = (feats["total_spend"] / feats["total_orders"].replace(0, 1)).fillna(0)

    # Activity: days since last order
    last_order = orders.groupby("customer_id")["order_date"].max()
    feats["days_since_last_order"] = (reference_date - last_order).dt.days
    # If never ordered, set to tenure (max possible inactivity)
    feats["days_since_last_order"] = feats["days_since_last_order"].fillna(feats["tenure_days"])

    # Order frequency (orders per 30 days of tenure)
    feats["orders_per_month"] = (
        feats["total_orders"] / (feats["tenure_days"].clip(lower=30) / 30.0)
    ).fillna(0)

    # Diversity features
    feats["unique_products"] = (
        orders.groupby("customer_id")["product_id"].nunique().reindex(feats.index, fill_value=0)
    )
    feats["unique_payment_methods"] = (
        orders.groupby("customer_id")["payment_method"].nunique().reindex(feats.index, fill_value=0)
    )

    # Behavior pattern: weekday order ratio (0=Monday, 6=Sunday; weekday = 0-4)
    orders["is_weekday"] = orders["order_date"].dt.dayofweek < 5
    weekday_ratio = orders.groupby("customer_id")["is_weekday"].mean()
    feats["weekday_order_ratio"] = weekday_ratio.reindex(feats.index, fill_value=0.5)  # 0.5 = neutral default

    # Customer tier (one-hot encoded)
    tier_lookup = cust.set_index("customer_id")["customer_tier"]
    tier_series = tier_lookup.reindex(feats.index)
    for tier in TIER_CATEGORIES:
        feats[f"tier_{tier.lower()}"] = (tier_series == tier).astype(int)

    return feats.reset_index()


FEATURE_COLUMNS = [
    "tenure_days",
    "total_orders",
    "total_spend",
    "avg_order_value",
    "days_since_last_order",
    "orders_per_month",
    "unique_products",
    "unique_payment_methods",
    "weekday_order_ratio",
    "tier_standard",
    "tier_premium",
    "tier_gold",
]
