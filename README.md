# Fraud Detection

An end-to-end machine learning pipeline for detecting fraudulent credit card transactions: data preprocessing, imbalance handling, multi-model training, explainable evaluation, and a real-time scoring API.

Built as the final deliverable for the **IBM Machine Learning and Applied AI** program.

---

## Table of Contents

1. [Overview](#overview)
2. [Key Features](#key-features)
3. [Dataset](#dataset)
4. [Project Structure](#project-structure)
5. [Pipeline Architecture](#pipeline-architecture)
6. [Models](#models)
7. [Tech Stack](#tech-stack)
8. [Getting Started](#getting-started)
9. [Usage](#usage)
10. [Evaluation & Explainability](#evaluation--explainability)
11. [REST API](#rest-api)
12. [Docker](#docker)
13. [Testing](#testing)
14. [Configuration Reference](#configuration-reference)
15. [Results](#results)
16. [Notes & Limitations](#notes--limitations)
17. [Acknowledgements](#acknowledgements)
18. [Author](#author)

---

## Overview

Fraudulent transactions are extremely rare compared to legitimate ones, which makes plain accuracy a misleading metric and naive models ineffective. This project addresses that with:

- Careful preprocessing that avoids data leakage (scalers are fit on the training split only).
- Multiple strategies for class imbalance (SMOTE-Tomek, ADASYN, cost weighting).
- A diverse model zoo: gradient-boosted trees, deep learning, unsupervised anomaly detectors, and a calibrated stacking ensemble.
- Imbalance-aware evaluation centered on **PR-AUC**, recall at fixed false-positive rates, and a configurable business cost matrix.
- SHAP-based explanations, both global and per transaction.
- A FastAPI service that returns a calibrated risk score and the top-3 driving features for every transaction.

## Key Features

- **Flexible data loading:** reads a CSV directly or straight from a `.zip` archive (no manual extraction).
- **Temporal feature engineering:** cyclical sin/cos encoding of hour-of-day and day-of-week.
- **Amount handling:** `log1p` transform followed by robust scaling.
- **Imbalance strategies:** `smote_tomek`, `adasyn`, `cost_weight`, or `none`.
- **Rolling behavioural features:** per-card velocity, spend, recency, and merchant diversity over 1 hour, 24 hour, and 7 day windows.
- **Seven model types** plus a calibrated stacking meta-learner.
- **Cost-utility analysis:** quantifies the trade-off between missed fraud and false alarms.
- **Production-oriented serving:** health checks, batch scoring, Prometheus-style metrics, and a multi-stage Docker image running as a non-root user.
- **Automated tests** for the pipeline, features, models, evaluation, and API.

## Dataset

- **Source:** [Credit Card Fraud dataset on Kaggle](https://www.kaggle.com/datasets/jacklizhi/creditcard)
- **Schema:** European Credit Card style, with columns `Time`, `Amount`, `V1`–`V28` (anonymized PCA components), and `Class` (`1` = fraud, `0` = legitimate).
- **Class balance:** highly imbalanced, with fraud making up well under 1% of transactions.
- **Placement:** the pipeline expects the dataset as a ZIP containing exactly one CSV:

  ```
  data/raw/creditcard.zip
  ```

> The pipeline also supports other schemas (for example IEEE-CIS) by overriding column names in `PipelineConfig`.
>
> Never commit large dataset files or trained models to Git.

## Project Structure

```
fraud_detection/
├── data/
│   ├── raw/
│   │   └── creditcard.zip          # Raw dataset (download from Kaggle)
│   ├── processed/                  # Created by data_pipeline.py (train/val/test CSVs)
│   └── .gitkeep
├── models/                         # Trained model artifacts (created by train.py)
│   └── .gitkeep
├── reports/                        # Evaluation outputs and coverage report
│   └── .gitkeep
├── tests/
│   ├── __init__.py
│   ├── test_app.py                 # API endpoint tests
│   ├── test_data_pipeline.py       # Loading, cleaning, encoding, splitting, leakage checks
│   ├── test_evaluate.py            # Metrics, cost-utility, config
│   ├── test_feature_engineering.py # Rolling-window feature tests
│   └── test_models.py              # Neural network architectures and serialization
├── app.py                          # FastAPI real-time scoring service
├── conftest.py                     # Shared pytest setup (adds project root to sys.path)
├── data_pipeline.py                # Ingestion, preprocessing, resampling, splitting
├── Dockerfile                      # Multi-stage container build
├── evaluate.py                     # Metrics, cost analysis, SHAP, calibration
├── feature_engineering.py          # Rolling behavioural features
├── pyproject.toml                  # Pytest and coverage configuration
├── requirements.txt                # Python dependencies
├── SETUP.txt                       # Quickstart guide
├── train.py                        # Multi-model training orchestrator
└── README.md
```

### Module Responsibilities

| File | Purpose |
| --- | --- |
| `data_pipeline.py` | `FraudDataPipeline` and `PipelineConfig`: load, clean, engineer temporal features, scale amounts, split (stratified train/val/test), resample the training set, save to `data/processed/`. |
| `feature_engineering.py` | `RollingFeatureEngineer`: transaction count, total/mean/std spend, velocity spike ratio, unique merchants, and time since last transaction per card. Supports offline batch mode and single-transaction online mode. |
| `train.py` | Trains and saves every model; computes `scale_pos_weight` from the training data; persists `train_config.json`. |
| `evaluate.py` | `ModelEvaluator`: PR-AUC, ROC-AUC, F1, recall at fixed FPR, cost-utility, SHAP explanations, PR curves, calibration plots. |
| `app.py` | FastAPI service with a model registry, SHAP explainer, and in-process metrics. |

## Pipeline Architecture

```
 data/raw/creditcard.zip
          │
          ▼
 ┌──────────────────────┐
 │  data_pipeline.py    │  clean → temporal encoding → log1p(Amount)
 │                      │  → stratified split → RobustScaler (fit on train)
 │                      │  → resample train (SMOTE-Tomek / ADASYN / none)
 └──────────┬───────────┘
            ▼
     data/processed/  (X_train, X_val, X_test, y_train, y_val, y_test)
            │
            ▼
 ┌──────────────────────┐
 │      train.py        │  XGBoost · LightGBM · CatBoost · MLP · LSTM/GRU
 │                      │  Isolation Forest · Autoencoder · Stacking
 └──────────┬───────────┘
            ▼
         models/
            │
     ┌──────┴────────┐
     ▼               ▼
┌─────────────┐  ┌──────────┐
│ evaluate.py │  │  app.py  │  FastAPI: /score, /score/batch, /health ...
└──────┬──────┘  └──────────┘
       ▼
    reports/  (metrics CSVs, PR curves, SHAP plots)
```

## Models

| Category | Model | Notes |
| --- | --- | --- |
| Tree ensembles | **XGBoost**, **LightGBM**, **CatBoost** | Class imbalance handled via `scale_pos_weight`; early stopping on validation PR-AUC. |
| Deep learning | **Residual MLP** | Batch norm, dropout, residual blocks, weighted BCE loss, AdamW, LR scheduling, gradient clipping. |
| Deep learning | **LSTM / GRU** | Sliding window of the last 10 transactions; selectable via `seq_model_type`. |
| Anomaly detection | **Isolation Forest** | Unsupervised; scores are normalized for evaluation. |
| Anomaly detection | **Autoencoder** | Trained on legitimate transactions only; reconstruction error is the anomaly score, with a 99th-percentile threshold. |
| Meta-learner | **Stacking ensemble** | Logistic-regression meta-learner over XGBoost, LightGBM, and CatBoost, with Platt (sigmoid) calibration. This is the default model served by the API. |

## Tech Stack

- **Language:** Python 3.10+ (the Docker image uses Python 3.11)
- **Data and ML:** NumPy, pandas, scikit-learn, imbalanced-learn, SciPy, joblib
- **Boosting:** XGBoost, LightGBM, CatBoost
- **Deep learning:** PyTorch
- **Explainability:** SHAP
- **Visualization:** Matplotlib
- **Serving:** FastAPI, Uvicorn, Pydantic
- **Testing:** pytest, pytest-asyncio, pytest-cov, httpx
- **Packaging:** Docker (multi-stage build)

## Getting Started

### Prerequisites

- Python 3.10 or higher
- (Optional) NVIDIA GPU with CUDA. The deep learning models use the GPU automatically if available.
- (Optional) Docker, for containerized deployment

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/anay-sinha/IBM-Machine-Learning-and-Applied-AI--Final-Deliverable.git
cd IBM-Machine-Learning-and-Applied-AI--Final-Deliverable/fraud_detection

# 2. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate          # macOS / Linux
# .venv\Scripts\activate           # Windows

# 3. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

> For GPU acceleration, install the CUDA build of PyTorch first by following the [official instructions](https://pytorch.org/get-started/locally/).

### Add the Dataset

1. Download the data from [Kaggle](https://www.kaggle.com/datasets/jacklizhi/creditcard).
2. Make sure it is a ZIP containing a single CSV, and place it at `data/raw/creditcard.zip`.

## Usage

Run all commands from inside the `fraud_detection/` directory, in this order.

### Step 1: Preprocess the data

```bash
python data_pipeline.py --data data/raw/creditcard.zip --strategy smote_tomek
```

This loads and cleans the data, encodes time cyclically, log-transforms and scales amounts, resamples the training split, and writes the splits to `data/processed/`.

| `--strategy` | Behavior |
| --- | --- |
| `smote_tomek` (default) | SMOTE oversampling combined with Tomek-link cleaning |
| `adasyn` | Adaptive density-based oversampling |
| `cost_weight` | No resampling; models use `scale_pos_weight` |
| `none` | No imbalance handling (baseline) |

You can also pass a JSON file of `PipelineConfig` fields with `--config path/to/config.json`.

### Step 2: Train the models

```bash
python train.py --processed-dir data/processed
```

Optional: `--epochs N` sets the number of epochs for the MLP and sequence model (default 30).

Artifacts written to `models/`:

| Artifact | Description |
| --- | --- |
| `xgboost.joblib`, `lightgbm.joblib`, `catboost.joblib` | Tree ensembles |
| `isolation_forest.joblib` | Isolation Forest |
| `stacking_ensemble.joblib` | Calibrated stacking ensemble |
| `mlp.pt`, `lstm.pt` (or `gru.pt`) | PyTorch weights |
| `autoencoder.pt`, `ae_threshold.npy` | Autoencoder weights and anomaly threshold |
| `train_config.json` | Training configuration used |

### Step 3: Evaluate the models

```bash
python evaluate.py --processed-dir data/processed
```

By default this evaluates `xgboost`, `lightgbm`, `catboost`, and `stacking_ensemble`. To choose models:

```bash
python evaluate.py --models xgboost lightgbm stacking_ensemble
```

Outputs are saved to `reports/` (see [Evaluation & Explainability](#evaluation--explainability)).

### Step 4: Start the API

```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```

Interactive docs are served at <http://localhost:8000/docs>.

### Optional: Rolling features

`feature_engineering.py` can add per-card behavioural features to any transaction CSV that includes a card identifier:

```bash
python feature_engineering.py --input data/your_transactions.csv \
    --output data/features.csv --card-col card_id --amount-col Amount_log1p
```

> The default European Credit Card dataset has no card identifier, so this step applies to datasets that do (for example, a custom or IEEE-CIS style schema).

## Evaluation & Explainability

Because fraud is rare, the evaluation emphasizes metrics that stay meaningful under heavy class imbalance.

| Metric | Why it matters |
| --- | --- |
| **PR-AUC** (Average Precision) | Primary metric; focuses on the minority (fraud) class |
| **ROC-AUC** | Secondary, threshold-independent ranking quality |
| **Recall @ 1% / 5% FPR** | Fraud caught at fixed false-alarm budgets |
| **F1 / Precision / Recall** | At the threshold that maximizes F1 |
| **Cost-utility** | Weighs missed fraud (`cost_fn`, default 10) against false alarms (`cost_fp`, default 1) and reports savings versus detecting nothing |

**Explainability**

- `TreeExplainer` for tree-based models (fast); `KernelExplainer` as a model-agnostic fallback.
- Global SHAP beeswarm plot of feature importance.
- Local explanations: the top-k features driving a single prediction, with direction (`increases_fraud_risk` / `decreases_fraud_risk`).

**Files written to `reports/`**

| File | Content |
| --- | --- |
| `model_comparison.csv` | Side-by-side summary table |
| `full_metrics.csv` | All metrics, thresholds, and cost figures per model |
| `pr_curve_<model>.png` | Precision-recall curve per model |
| `shap_summary_<model>.png` | Global SHAP feature importance |
| `coverage/` | HTML test coverage report (after running tests) |

A calibration plotting helper, `plot_calibration_curve(models, X, y)`, is available in `evaluate.py`.

## REST API

The service loads `models/stacking_ensemble.joblib` at startup.

| Method | Endpoint | Description |
| --- | --- | --- |
| `POST` | `/score` | Score one transaction in real time |
| `POST` | `/score/batch` | Score up to 500 transactions per request |
| `GET` | `/health` | Liveness check (status, model name, version) |
| `GET` | `/model/info` | Loaded model metadata and SHAP explainer type |
| `GET` | `/metrics` | Prometheus-compatible text metrics |

### Example: score a transaction

```bash
curl -X POST http://localhost:8000/score \
     -H "Content-Type: application/json" \
     -d '{
           "features": {
             "V1": -1.36,
             "V2": -0.07,
             "V3": 2.54,
             "Amount_log1p": 5.42,
             "hour_sin": 0.87,
             "hour_cos": 0.49
           }
         }'
```

Illustrative response:

```json
{
  "risk_score": 0.923,
  "fraud_flag": true,
  "threshold": 0.5,
  "shap_top3": [
    {"feature": "V14", "value": -5.12, "shap_value": 0.41, "direction": "increases_fraud_risk"},
    {"feature": "V4",  "value":  2.30, "shap_value": 0.29, "direction": "increases_fraud_risk"},
    {"feature": "V12", "value": -3.10, "shap_value": 0.18, "direction": "increases_fraud_risk"}
  ],
  "model_version": "1.0.0",
  "latency_ms": 4.2
}
```

### Example: batch scoring

```bash
curl -X POST http://localhost:8000/score/batch \
     -H "Content-Type: application/json" \
     -d '[
           {"transaction_id": "tx-001", "features": {"V1": -1.36, "Amount_log1p": 5.42}},
           {"transaction_id": "tx-002", "features": {"V1": 0.21,  "Amount_log1p": 2.10}}
         ]'
```

### Response fields

| Field | Description |
| --- | --- |
| `risk_score` | Calibrated fraud probability in [0, 1] |
| `fraud_flag` | `true` when `risk_score >= threshold` |
| `threshold` | Decision boundary used (default 0.5) |
| `shap_top3` | Top three features driving the prediction |
| `model_version` | Version of the loaded model artifact |
| `latency_ms` | Server-side inference time |

### Optional serving artifacts

The service reads these from `models/` if present:

| File | Purpose | Default if absent |
| --- | --- | --- |
| `threshold.txt` | Decision threshold | `0.5` |
| `version.txt` | Model version string | `1.0.0` |
| `feature_names.txt` | Training feature order, one per line | Features used as sent |

When `feature_names.txt` exists, incoming payloads are aligned to the training column order: missing features are zero-filled and extra features are dropped.

## Docker

The `Dockerfile` is a two-stage build (a builder stage installs dependencies into a virtual environment; a lean runtime stage serves the app as a non-root user) with a built-in `/health` check.

```bash
# Build
docker build -t fraud-detection:latest .

# Run (CPU)
docker run -p 8000:8000 \
    -v $(pwd)/models:/app/models:ro \
    -v $(pwd)/data:/app/data:ro \
    fraud-detection:latest

# Run (GPU, requires nvidia-container-toolkit)
docker run --gpus all -p 8000:8000 \
    -v $(pwd)/models:/app/models:ro \
    -v $(pwd)/data:/app/data:ro \
    fraud-detection:latest
```

Trained models and data are mounted at runtime rather than baked into the image. The API is then available at <http://localhost:8000>.

> The Dockerfile copies `requirements.txt` and a `fraud_detection/` directory from the build context. Make sure your build context matches this layout, or adjust the `COPY` lines.

## Testing

```bash
pytest tests/
```

Pytest and coverage settings live in `pyproject.toml`. A run produces a terminal coverage summary and an HTML report in `reports/coverage/`, and fails if coverage drops below 70%.

| Test module | Covers |
| --- | --- |
| `test_data_pipeline.py` | CSV loading, missing-file errors, de-duplication, null handling, cyclical encoding, amount scaling, `scale_pos_weight`, split integrity, and scaler/data leakage |
| `test_feature_engineering.py` | Rolling column generation, window semantics, velocity spike, recency, single-transaction transform |
| `test_models.py` | MLP and sequence model shapes and output bounds, gradient flow, sequence dataset construction, autoencoder, model serialization |
| `test_evaluate.py` | Probability extraction, PR-AUC/ROC-AUC bounds, recall at FPR, optimal threshold, cost-utility |
| `test_app.py` | `/health`, `/model/info`, `/score`, `/score/batch`, `/metrics` |

## Configuration Reference

**`PipelineConfig`** (`data_pipeline.py`)

| Parameter | Default | Description |
| --- | --- | --- |
| `imbalance_strategy` | `smote_tomek` | `smote_tomek`, `adasyn`, `cost_weight`, or `none` |
| `test_size` / `val_size` | `0.15` / `0.15` | Hold-out fractions (70/15/15 split) |
| `target_col` | `Class` | Label column |
| `amount_col` / `time_col` | `Amount` / `Time` | Source columns for amount and time features |
| `timestamp_col` | `None` | Real datetime column, if the dataset has one |
| `log_transform_amount` | `True` | Apply `log1p` to the amount |
| `random_state` | `42` | Reproducibility seed |

**`TrainConfig`** (`train.py`)

| Parameter | Default | Description |
| --- | --- | --- |
| `n_cv_folds` | `5` | Cross-validation folds for stacking |
| `xgb_n_estimators` / `lgb_n_estimators` / `catboost_iterations` | `500` | Boosting rounds |
| `tree_learning_rate` / `tree_max_depth` | `0.05` / `6` | Shared tree hyperparameters |
| `mlp_hidden_dims` | `[256, 128, 64]` | MLP layer sizes |
| `seq_model_type` | `LSTM` | `LSTM` or `GRU` |
| `seq_window` | `10` | Past transactions per sequence |
| `iso_contamination` | `0.002` | Expected fraud rate for Isolation Forest |

**`EvalConfig`** (`evaluate.py`)

| Parameter | Default | Description |
| --- | --- | --- |
| `cost_fn` / `cost_fp` | `10.0` / `1.0` | Relative cost of a missed fraud / a false alarm |
| `fpr_targets` | `[0.01, 0.05]` | FPR points for recall reporting |
| `shap_max_display` | `20` | Features shown in SHAP plots |

## Results

_Add your final model comparison here after running `evaluate.py` (see `reports/model_comparison.csv`)._

| Model | PR-AUC | ROC-AUC | F1 | Recall @ 1% FPR | Recall @ 5% FPR |
| --- | --- | --- | --- | --- | --- |
| XGBoost | | | | | |
| LightGBM | | | | | |
| CatBoost | | | | | |
| Stacking ensemble | | | | | |

## Notes & Limitations

- `data/` and `models/` are excluded from version control because of file size; never commit secrets or `.env` files.
- The stacking ensemble combines the tree models only; the deep learning and anomaly models are trained and evaluated separately.
- SHAP explanations are computed per request. `TreeExplainer` is fast, while `KernelExplainer` (used as a fallback for non-tree models) is noticeably slower.
- Resampling is applied to the training split only, so validation and test sets keep the true class distribution.
- The sample CORS configuration in `app.py` allows all origins; restrict it before any real deployment.

## Acknowledgements

- Dataset: [Kaggle: Credit Card Fraud](https://www.kaggle.com/datasets/jacklizhi/creditcard)
- Program: IBM Machine Learning and Applied AI

## Author

**Anay Sinha**: [GitHub](https://github.com/anay-sinha)
