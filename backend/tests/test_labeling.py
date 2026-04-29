"""Unit tests for labeling.py — the churn rule logic."""
import pandas as pd
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from labeling import compute_signals


REF_DATE = pd.Timestamp("2026-04-29")


def _orders(rows):
    """Helper to build orders DataFrame."""
    return pd.DataFrame(rows, columns=["customer_id", "order_date", "order_amount", "order_status"])


def test_active_customer_not_churned():
    """Customer ordering steadily across both windows → not churned."""
    orders = _orders([
        (1, "2026-04-15", 100, "Delivered"),  # last 60d
        (1, "2026-04-01", 100, "Delivered"),  # last 60d
        (1, "2026-03-15", 100, "Delivered"),  # last 60d
        (1, "2026-02-15", 100, "Delivered"),  # prior 60d
        (1, "2026-01-15", 100, "Delivered"),  # prior 60d
    ])
    result = compute_signals(orders, REF_DATE)
    assert result.iloc[0]["churned"] == 0, "Active customer should not be churned"


def test_inactive_customer_is_churned():
    """Customer with no orders in last 90 days → churned."""
    orders = _orders([
        (1, "2026-01-01", 100, "Delivered"),
        (1, "2025-12-01", 100, "Delivered"),
        (1, "2025-11-01", 100, "Delivered"),
    ])
    result = compute_signals(orders, REF_DATE)
    assert result.iloc[0]["churned"] == 1, "Inactive customer should be churned"
    assert result.iloc[0]["signal_recency"] == 1


def test_frequency_drop_signal():
    """Customer with 1 order recent, 3 orders prior → frequency signal fires."""
    orders = _orders([
        (1, "2026-04-15", 100, "Delivered"),  # 1 order last 60d
        (1, "2026-02-15", 100, "Delivered"),  # 3 orders prior 60d
        (1, "2026-02-01", 100, "Delivered"),
        (1, "2026-01-15", 100, "Delivered"),
    ])
    result = compute_signals(orders, REF_DATE)
    assert result.iloc[0]["signal_frequency"] == 1


def test_no_prior_period_no_frequency_signal():
    """New customer with only recent orders → frequency signal should NOT fire."""
    orders = _orders([
        (1, "2026-04-15", 100, "Delivered"),
        (1, "2026-04-10", 100, "Delivered"),
    ])
    result = compute_signals(orders, REF_DATE)
    assert result.iloc[0]["signal_frequency"] == 0, "No prior orders means signal should not fire"


def test_negative_ratio_signal():
    """Customer with mostly cancelled orders → negative signal fires."""
    orders = _orders([
        (1, "2026-04-15", 100, "Cancelled"),
        (1, "2026-04-10", 100, "Cancelled"),
        (1, "2026-04-05", 100, "Returned"),
        (1, "2026-04-01", 100, "Delivered"),
    ])
    result = compute_signals(orders, REF_DATE)
    assert result.iloc[0]["signal_negative"] == 1, "75% negative orders should fire signal"


def test_empty_orders_returns_empty():
    """Empty input → empty output, no crash."""
    orders = pd.DataFrame(columns=["customer_id", "order_date", "order_amount", "order_status"])
    result = compute_signals(orders, REF_DATE)
    assert result.empty
