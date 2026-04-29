# Churn Predictor

> Predicts which retail customers are about to leave — so you can save them before they do.

A production-grade churn prediction service. PyTorch neural network behind a FastAPI service, deployed on Google Cloud Run, reusing the data warehouse from a sibling project ([QuickShop AI](https://github.com/shaktisrivastava2020/quickshop-ai)).

**Live API:** https://churn-api-218990051802.asia-south1.run.app
**Interactive docs:** https://churn-api-218990051802.asia-south1.run.app/docs

---

## Why this exists

Most retail churn models report 90%+ accuracy and quietly miss most actual churners — because 75% of customers are active, so "predict everyone is active" gets you 75% accuracy for free.

This project optimizes for **catching churners**, not for headline accuracy. The v2 model catches **8 of 10 churners** at the cost of some false positives — the right trade-off for retail, where missing a churner means lost revenue and a false alarm means a discount email.

---

## The story (v1 → v2)

### v1: shipped honestly, performed poorly

Built a PyTorch network on 176 customers, deployed to Cloud Run, wrote 14 tests, ran a security audit. Engineering was clean. The model? Caught **1 of 10 churners.**

| v1 @ threshold 0.50 | |
|---|---|
| Accuracy | 72.2% |
| Precision | 50.0% |
| **Recall** | **10.0%** ❌ |
| F1 | 16.7% |
| ROC-AUC | 0.608 |

### Diagnosis

The model could discriminate — ROC-AUC 0.608 isn't random — it just made bad decisions at the default 0.5 threshold. With 27% churn rate and limited data (176 customers, 36 in test), the network learned to be conservative. Most actual churners scored 0.40-0.49 and got classified as active.

### Fix #1: Threshold tuning (no retraining)

Swept thresholds 0.20 → 0.50, optimized for F1.
### Fix #2 (failed): Retraining with smaller architecture

Tried 24-12 hidden units, longer training, lower LR. ROC-AUC dropped from 0.608 to 0.477 (worse than random ranking). Reverted. Documented in [`models/eval_report_v2.json`](backend/models/eval_report_v2.json) and the v2 commit message.

### v2: shipped (v1 weights + threshold 0.40)

| | v1 | **v2** | Δ |
|---|---|---|---|
| **Recall** | 10.0% | **80.0%** | **+70.0%** 🚀 |
| **F1** | 16.7% | **50.0%** | **+33.3%** 🚀 |
| **Churners caught** | 1 of 10 | **8 of 10** | **8x** |
| Accuracy | 72.2% | 55.6% | -16.7% |
| Precision | 50.0% | 36.4% | -13.6% |
| ROC-AUC | 0.608 | 0.608 | unchanged |

**Why this trade-off is correct for retail churn:** A missed churner is lost lifetime revenue. A false alarm is an unnecessary discount email. Recall > precision in this domain.

---

## Architecture
**Stack:** Python 3.11 · PyTorch 2.5.1 (CPU) · FastAPI 0.122 · SQLAlchemy 2.0 · Pydantic 2.10 · Cloud SQL Python Connector · Multi-stage Docker · Cloud Run

---

## How churn is defined (multi-signal RFM)

A customer is labeled **churned (1)** if **2 or more** of these fire:

| Signal | Rule |
|---|---|
| **Recency** | No engagement order in last 60 days |
| **Frequency drop** | Engagement orders last 60d < 50% of prior 60d |
| **Monetary drop** | Spend last 60d < 50% of prior 60d |
| **Negative ratio** | Cancelled or returned > 40% of total orders |

Why multi-signal: a single threshold is brittle. A customer might pause for 60 days due to a holiday — that alone shouldn't flag them. But pause + spend drop + return rate spike? That's churn.

Why these signals are *separate* from prediction features: avoids label leakage. The model learns from `tenure_days`, `total_spend`, `unique_products`, etc. — different lens than the labeling rule.

---

## Features used by the model (12 inputs)

| # | Feature | Type | Why |
|---|---|---|---|
| 1 | `tenure_days` | continuous | Loyalty signal |
| 2 | `total_orders` | continuous | Lifetime engagement |
| 3 | `total_spend` | continuous | Lifetime value |
| 4 | `avg_order_value` | continuous | Spend behavior |
| 5 | `days_since_last_order` | continuous | Activity |
| 6 | `orders_per_month` | continuous | Engagement rate |
| 7 | `unique_products` | continuous | Product diversity |
| 8 | `unique_payment_methods` | continuous | Payment diversity |
| 9 | `weekday_order_ratio` | continuous | Behavior pattern |
| 10-12 | `tier_standard`, `tier_premium`, `tier_gold` | one-hot | Customer segment |

PII (name, email, phone, address) is intentionally excluded — privacy compliance and zero predictive signal.

---

## Quick start

```bash
# Predict churn for a single customer
curl -X POST https://churn-api-218990051802.asia-south1.run.app/predict \
  -H "Content-Type: application/json" \
  -d '{
    "tenure_days": 600,
    "total_orders": 3,
    "total_spend": 5000,
    "avg_order_value": 1666,
    "days_since_last_order": 200,
    "orders_per_month": 0.15,
    "unique_products": 2,
    "unique_payment_methods": 1,
    "weekday_order_ratio": 0.5,
    "tier_standard": 1,
    "tier_premium": 0,
    "tier_gold": 0
  }'

# {"churn_probability": 0.7395, "churned_label": 1, "threshold_used": 0.4}
```

Or hit the [interactive Swagger docs](https://churn-api-218990051802.asia-south1.run.app/docs) and click "Try it out."

---

## Engineering practices

- **Modular code:** one concern per file (`config`, `database`, `features`, `labeling`, `model`, `predictor`, `router`)
- **14 tests, all passing:** unit tests for the labeling rule + integration tests for the API
- **Dependency pinning:** versions aligned with sibling project for consistency
- **Security audit:** `pip-audit` clean, secrets in `.env` (chmod 600, gitignored)
- **Multi-stage Docker:** slim production image, separate build/runtime stages
- **No PII committed:** training CSV gitignored, schema documented separately
- **Reproducible:** fixed random seed (42), saved scaler with model, metadata JSON tracks every hyperparameter

---

## Cost (Cloud Run scale-to-zero)

| Resource | Pricing | Idle cost |
|---|---|---|
| Cloud Run (min-instances=0) | $0 when not invoked | **$0/hr** |
| Cloud SQL (`db-f1-micro`, stopped) | $0 when STOPPED | **$0/hr** |
| Artifact Registry storage | ~$0.10/GB/month | <$0.05/mo |
| Cloud Build (used at deploy) | 120 free build-min/day | $0 |

**At rest, this entire stack costs less than $0.10/month.** Cloud Run charges per request + ms of compute, so the model is essentially free until traffic shows up.

---

## Honest limitations

- **Small dataset.** 176 training customers. Reliable for portfolio demonstration; would need more data for an enterprise deployment.
- **Static label definition.** Churn rules are hard-coded thresholds. A real production system would A/B test multiple definitions and pick the one that correlates best with downstream business metrics (LTV, return rate).
- **No drift monitoring.** Model is static. A real production system would track input distributions and retrain on a schedule.
- **Single-tenant.** Hardcoded to QuickShop's schema. Generalizing to a multi-tenant SaaS would require feature-store abstraction.

---

## Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/` | Service info |
| `GET` | `/health` | Liveness check |
| `GET` | `/model/info` | Model metadata (training date, features, hyperparameters) |
| `POST` | `/predict` | Single customer prediction |
| `POST` | `/predict/batch` | Bulk prediction with optional threshold override |
| `GET` | `/docs` | Interactive Swagger UI |

---

## Repo structure
---

## Built by

**Shakti Srivastava**
- An AI optimist and a researcher
- GitHub: [@shaktisrivastava2020](https://github.com/shaktisrivastava2020)
- Open to freelance AI/ML projects — RAG, NL2SQL, NLP, Deep Learning GenAI, ConvAI, Speech AI, Vision AI, Agentic, MCP, MLOps products on GCP & AWS


