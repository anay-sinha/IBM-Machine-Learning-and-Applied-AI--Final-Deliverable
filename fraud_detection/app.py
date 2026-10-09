"""
app.py
======
Low-latency FastAPI inference service for real-time fraud scoring.

Endpoints
---------
POST /score          → fraud risk score, flag, top-3 SHAP features
POST /score/batch    → batch scoring (up to 500 transactions)
GET  /health         → liveness probe
GET  /model/info     → loaded model metadata
GET  /metrics        → Prometheus-compatible text metrics (counter / histogram)

Response design
---------------
- risk_score  : float [0, 1] – calibrated fraud probability
- fraud_flag  : bool  – True when score ≥ threshold
- threshold   : float – decision boundary used
- shap_top3   : list  – top-3 SHAP driving features with direction and value
- model_version: str  – model artifact version tag
- latency_ms  : float – server-side inference time
"""

from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
import shap
import torch
import uvicorn
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, Field, field_validator

from evaluate import ModelEvaluator

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

MODELS_DIR = Path("models")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Prometheus-style in-process metrics (lightweight, no dependency)
# ---------------------------------------------------------------------------

class _Metrics:
    def __init__(self):
        self.requests_total: int = 0
        self.fraud_flags_total: int = 0
        self.latency_sum_ms: float = 0.0
        self.latency_count: int = 0
        self.errors_total: int = 0

    def record(self, latency_ms: float, is_fraud: bool) -> None:
        self.requests_total += 1
        self.latency_sum_ms += latency_ms
        self.latency_count += 1
        if is_fraud:
            self.fraud_flags_total += 1

    def to_prometheus(self) -> str:
        avg_lat = self.latency_sum_ms / max(self.latency_count, 1)
        return (
            f"# HELP fraud_requests_total Total scoring requests\n"
            f"# TYPE fraud_requests_total counter\n"
            f"fraud_requests_total {self.requests_total}\n\n"
            f"# HELP fraud_flags_total Transactions flagged as fraud\n"
            f"# TYPE fraud_flags_total counter\n"
            f"fraud_flags_total {self.fraud_flags_total}\n\n"
            f"# HELP fraud_inference_latency_ms_avg Average inference latency\n"
            f"# TYPE fraud_inference_latency_ms_avg gauge\n"
            f"fraud_inference_latency_ms_avg {avg_lat:.2f}\n\n"
            f"# HELP fraud_errors_total Total inference errors\n"
            f"# TYPE fraud_errors_total counter\n"
            f"fraud_errors_total {self.errors_total}\n"
        )


APP_METRICS = _Metrics()


# ---------------------------------------------------------------------------
# Model Registry (singleton loaded at startup)
# ---------------------------------------------------------------------------

class ModelRegistry:
    """Holds all loaded model artefacts and SHAP explainers."""

    def __init__(self):
        self.primary_model: Any = None
        self.model_name: str = ""
        self.threshold: float = 0.5
        self.feature_names: List[str] = []
        self.explainer: Any = None
        self.background_data: Optional[pd.DataFrame] = None
        self.version: str = "unknown"

    def load(
        self,
        model_name: str = "stacking_ensemble",
        threshold: float = 0.5,
        background_csv: Optional[str] = None,
    ) -> None:
        """Load the primary scoring model and prepare SHAP explainer."""
        model_path = MODELS_DIR / f"{model_name}.joblib"
        if not model_path.exists():
            raise FileNotFoundError(f"Model artefact not found: {model_path}")

        self.primary_model = joblib.load(model_path)
        self.model_name = model_name
        self.threshold = threshold
        self.version = self._read_version()

        # Load feature names
        feat_path = MODELS_DIR / "feature_names.txt"
        if feat_path.exists():
            self.feature_names = feat_path.read_text().strip().splitlines()

        # Load background data for KernelExplainer if provided
        if background_csv and Path(background_csv).exists():
            bg = pd.read_csv(background_csv)
            self.background_data = bg.sample(min(300, len(bg)), random_state=42)
            self._build_explainer()
        else:
            # Try TreeExplainer first (no background needed)
            self._build_explainer()

        logger.info(
            "Model registry loaded: %s (threshold=%.3f, version=%s)",
            model_name, threshold, self.version,
        )

    def _build_explainer(self) -> None:
        tree_types = (
            "XGBClassifier", "LGBMClassifier", "CatBoostClassifier",
        )
        model_type = type(self.primary_model).__name__
        try:
            if model_type in tree_types:
                self.explainer = shap.TreeExplainer(self.primary_model)
                logger.info("SHAP TreeExplainer ready.")
            elif self.background_data is not None:
                self.explainer = shap.KernelExplainer(
                    lambda x: self.primary_model.predict_proba(
                        pd.DataFrame(x, columns=self.feature_names or None)
                    )[:, 1],
                    shap.sample(self.background_data, 200),
                )
                logger.info("SHAP KernelExplainer ready.")
            else:
                logger.warning("No SHAP explainer built (no background data for non-tree model).")
        except Exception as e:
            logger.warning("SHAP explainer init failed: %s", e)

    def _read_version(self) -> str:
        v_path = MODELS_DIR / "version.txt"
        if v_path.exists():
            return v_path.read_text().strip()
        return "1.0.0"


REGISTRY = ModelRegistry()


# ---------------------------------------------------------------------------
# Pydantic Schemas
# ---------------------------------------------------------------------------

class TransactionFeatures(BaseModel):
    """
    Raw transaction feature payload.
    Mirrors the engineered feature set produced by the pipeline.
    Extend with your actual feature names.
    """
    features: Dict[str, float] = Field(
        ...,
        description="Key-value map of feature name → value. "
                    "Must include all features the model was trained on.",
        examples=[{"V1": -1.36, "V2": -0.073, "Amount_log1p": 5.42, "hour_sin": 0.87}],
    )

    @field_validator("features")
    @classmethod
    def must_not_be_empty(cls, v):
        if not v:
            raise ValueError("features dict must not be empty")
        return v


class SHAPFeature(BaseModel):
    feature: str
    value: float
    shap_value: float
    direction: str  # "increases_fraud_risk" | "decreases_fraud_risk"


class ScoreResponse(BaseModel):
    transaction_id: Optional[str] = None
    risk_score: float = Field(..., ge=0.0, le=1.0)
    fraud_flag: bool
    threshold: float
    shap_top3: List[SHAPFeature]
    model_version: str
    latency_ms: float


class BatchTransactionItem(BaseModel):
    transaction_id: Optional[str] = None
    features: Dict[str, float]


class BatchScoreResponse(BaseModel):
    results: List[ScoreResponse]
    batch_size: int
    total_latency_ms: float


# ---------------------------------------------------------------------------
# Inference Logic
# ---------------------------------------------------------------------------

def _build_feature_row(
    features: Dict[str, float],
    expected_cols: List[str],
) -> pd.DataFrame:
    """
    Align incoming features to the training column order.
    Missing columns are zero-filled; extra columns are dropped.
    """
    if expected_cols:
        row = {col: features.get(col, 0.0) for col in expected_cols}
    else:
        row = features
    return pd.DataFrame([row])


def _score_single(
    features: Dict[str, float],
    transaction_id: Optional[str] = None,
) -> ScoreResponse:
    """Core single-transaction inference."""
    t0 = time.perf_counter()

    row_df = _build_feature_row(features, REGISTRY.feature_names)

    # Primary model scoring
    if hasattr(REGISTRY.primary_model, "predict_proba"):
        proba = float(REGISTRY.primary_model.predict_proba(row_df)[0, 1])
    elif hasattr(REGISTRY.primary_model, "decision_function"):
        raw = float(REGISTRY.primary_model.decision_function(row_df)[0])
        proba = float(1 / (1 + np.exp(-raw)))  # sigmoid normalisation
    else:
        raise RuntimeError("Model does not support predict_proba or decision_function")

    fraud_flag = proba >= REGISTRY.threshold

    # SHAP local explanation
    shap_top3: List[SHAPFeature] = []
    if REGISTRY.explainer is not None:
        try:
            evaluator = ModelEvaluator()
            explanations = evaluator.local_explanation(
                REGISTRY.primary_model,
                row_df,
                background=REGISTRY.background_data if REGISTRY.background_data is not None else row_df,
                top_k=3,
            )
            shap_top3 = [SHAPFeature(**e) for e in explanations]
        except Exception as e:
            logger.warning("SHAP explanation failed: %s", e)

    latency_ms = (time.perf_counter() - t0) * 1000
    APP_METRICS.record(latency_ms, fraud_flag)

    return ScoreResponse(
        transaction_id=transaction_id,
        risk_score=round(proba, 6),
        fraud_flag=fraud_flag,
        threshold=REGISTRY.threshold,
        shap_top3=shap_top3,
        model_version=REGISTRY.version,
        latency_ms=round(latency_ms, 3),
    )


# ---------------------------------------------------------------------------
# FastAPI App
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models at startup, release resources at shutdown."""
    model_name = "stacking_ensemble"
    threshold  = float(Path("models/threshold.txt").read_text().strip()
                       if Path("models/threshold.txt").exists() else "0.5")
    background_csv = "data/processed/X_val.csv"

    logger.info("Loading model: %s", model_name)
    REGISTRY.load(
        model_name=model_name,
        threshold=threshold,
        background_csv=background_csv,
    )
    logger.info("Service ready.")
    yield
    logger.info("Service shutting down.")


app = FastAPI(
    title="Fraud Detection Scoring Service",
    version="1.0.0",
    description="Real-time transaction fraud scoring with SHAP explanations.",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Request Middleware – request-ID logging
# ---------------------------------------------------------------------------

@app.middleware("http")
async def log_requests(request: Request, call_next):
    request_id = request.headers.get("X-Request-ID", "—")
    logger.info("→ %s %s [request_id=%s]", request.method, request.url.path, request_id)
    response = await call_next(request)
    logger.info("← %d [request_id=%s]", response.status_code, request_id)
    return response


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health", status_code=status.HTTP_200_OK, tags=["ops"])
async def health():
    """Liveness probe."""
    return {
        "status": "ok",
        "model": REGISTRY.model_name,
        "version": REGISTRY.version,
    }


@app.get("/model/info", tags=["ops"])
async def model_info():
    """Return loaded model metadata."""
    return {
        "model_name": REGISTRY.model_name,
        "model_version": REGISTRY.version,
        "threshold": REGISTRY.threshold,
        "n_features": len(REGISTRY.feature_names),
        "feature_names": REGISTRY.feature_names[:20],  # preview first 20
        "shap_explainer": type(REGISTRY.explainer).__name__ if REGISTRY.explainer else None,
    }


@app.post("/score", response_model=ScoreResponse, tags=["inference"])
async def score_transaction(payload: TransactionFeatures):
    """
    Score a single transaction in real time.

    Returns a calibrated fraud probability, binary flag, and
    top-3 SHAP features driving the prediction.
    """
    try:
        return _score_single(payload.features)
    except Exception as e:
        APP_METRICS.errors_total += 1
        logger.error("Scoring error: %s", e, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference error: {str(e)}",
        )


@app.post("/score/batch", response_model=BatchScoreResponse, tags=["inference"])
async def score_batch(transactions: List[BatchTransactionItem]):
    """
    Score up to 500 transactions in a single request.
    SHAP explanations are computed per-transaction.
    """
    if len(transactions) > 500:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Batch size must not exceed 500.",
        )
    t0 = time.perf_counter()
    results = []
    for tx in transactions:
        try:
            result = _score_single(tx.features, transaction_id=tx.transaction_id)
            results.append(result)
        except Exception as e:
            APP_METRICS.errors_total += 1
            logger.warning("Error scoring tx %s: %s", tx.transaction_id, e)

    total_ms = (time.perf_counter() - t0) * 1000
    return BatchScoreResponse(
        results=results,
        batch_size=len(results),
        total_latency_ms=round(total_ms, 3),
    )


@app.get("/metrics", response_class=PlainTextResponse, tags=["ops"])
async def prometheus_metrics():
    """Prometheus-compatible text metrics."""
    return APP_METRICS.to_prometheus()


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        workers=1,  # single worker for GPU/model sharing
        log_level="info",
        access_log=True,
    )
