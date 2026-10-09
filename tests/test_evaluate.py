"""
tests/test_evaluate.py
=======================
Unit tests for the evaluation and explainability suite.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

from evaluate import ModelEvaluator, EvalConfig


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

N = 300
FRAUD_RATE = 0.05


@pytest.fixture()
def y_true_proba():
    rng = np.random.default_rng(99)
    y = rng.choice([0, 1], N, p=[1 - FRAUD_RATE, FRAUD_RATE])
    # Slightly better-than-random model: add signal to fraud samples
    proba = rng.uniform(0, 0.3, N)
    proba[y == 1] += rng.uniform(0.3, 0.7, y.sum())
    proba = np.clip(proba, 0, 1)
    return pd.Series(y, name="Class"), proba


@pytest.fixture()
def mock_sklearn_model(y_true_proba):
    _, proba = y_true_proba
    model = MagicMock()
    model.predict_proba.return_value = np.column_stack([1 - proba, proba])
    return model


@pytest.fixture()
def evaluator(tmp_path):
    cfg = EvalConfig(cost_fn=10.0, cost_fp=1.0)
    ev = ModelEvaluator(cfg)
    # Redirect REPORTS_DIR to tmp
    import evaluate as ev_module
    ev_module.REPORTS_DIR = tmp_path
    return ev


# ---------------------------------------------------------------------------
# _get_proba
# ---------------------------------------------------------------------------

class TestGetProba:
    def test_predict_proba_model(self, evaluator, mock_sklearn_model, y_true_proba):
        y_true, _ = y_true_proba
        X = pd.DataFrame(np.random.randn(N, 5))
        proba = evaluator._get_proba(mock_sklearn_model, X, "mock")
        assert proba is not None
        assert len(proba) == N
        assert ((proba >= 0) & (proba <= 1)).all()

    def test_decision_function_model(self, evaluator):
        model = MagicMock(spec=[])  # no predict_proba
        model.decision_function = MagicMock(return_value=np.random.randn(N))
        X = pd.DataFrame(np.random.randn(N, 5))
        proba = evaluator._get_proba(model, X, "df_model")
        assert proba is not None
        assert ((proba >= 0) & (proba <= 1)).all()

    def test_unsupported_model_returns_none(self, evaluator):
        model = MagicMock(spec=[])  # neither predict_proba nor decision_function
        X = pd.DataFrame(np.random.randn(5, 3))
        proba = evaluator._get_proba(model, X, "unsupported")
        assert proba is None


# ---------------------------------------------------------------------------
# _compute_metrics
# ---------------------------------------------------------------------------

class TestComputeMetrics:
    def test_pr_auc_between_zero_and_one(self, evaluator, y_true_proba):
        y_true, proba = y_true_proba
        metrics = evaluator._compute_metrics(proba, y_true, "test_model")
        assert 0.0 <= metrics["pr_auc"] <= 1.0

    def test_roc_auc_between_zero_and_one(self, evaluator, y_true_proba):
        y_true, proba = y_true_proba
        metrics = evaluator._compute_metrics(proba, y_true, "test_model")
        assert 0.0 <= metrics["roc_auc"] <= 1.0

    def test_recall_at_fpr_present(self, evaluator, y_true_proba):
        y_true, proba = y_true_proba
        metrics = evaluator._compute_metrics(proba, y_true, "test_model")
        assert "recall_at_1pct_fpr" in metrics
        assert "recall_at_5pct_fpr" in metrics

    def test_optimal_threshold_in_zero_one(self, evaluator, y_true_proba):
        y_true, proba = y_true_proba
        metrics = evaluator._compute_metrics(proba, y_true, "test_model")
        assert 0.0 <= metrics["optimal_threshold"] <= 1.0

    def test_better_model_higher_prauc(self, evaluator):
        """A perfect model should score higher than random."""
        y = pd.Series([0] * 270 + [1] * 30)
        perfect_proba = y.values.astype(float)
        random_proba = np.random.default_rng(0).uniform(0, 1, len(y))

        m_perfect = evaluator._compute_metrics(perfect_proba, y, "perfect")
        m_random  = evaluator._compute_metrics(random_proba,  y, "random")

        assert m_perfect["pr_auc"] > m_random["pr_auc"]


# ---------------------------------------------------------------------------
# Cost-utility matrix
# ---------------------------------------------------------------------------

class TestCostUtility:
    def test_zero_cost_for_perfect_predictor(self, evaluator):
        y = np.array([0, 0, 0, 1, 1])
        y_pred = np.array([0, 0, 0, 1, 1])
        result = evaluator.compute_cost(y, y_pred)
        assert result["total_cost"] == 0.0
        assert result["FN"] == 0
        assert result["FP"] == 0

    def test_fn_cost_dominates_fn_model(self, evaluator):
        """Missing fraud (FN) should be more costly than false alarm (FP)."""
        y = np.array([1] * 10 + [0] * 90)
        # Model that misses all fraud
        y_pred_fn = np.zeros(100, dtype=int)
        # Model that flags everything
        y_pred_fp = np.ones(100, dtype=int)

        cost_fn_heavy = evaluator.compute_cost(y, y_pred_fn)
        cost_fp_heavy = evaluator.compute_cost(y, y_pred_fp)

        assert cost_fn_heavy["total_cost"] > cost_fp_heavy["total_cost"]

    def test_cost_keys_present(self, evaluator):
        y = np.array([0, 1, 0, 1])
        y_pred = np.array([0, 1, 1, 0])
        result = evaluator.compute_cost(y, y_pred)
        for key in ("TP", "FP", "FN", "TN", "total_cost", "baseline_cost",
                    "cost_savings", "savings_pct"):
            assert key in result

    def test_savings_pct_bounded(self, evaluator):
        y = np.array([0, 0, 0, 1, 1])
        y_pred = np.array([0, 0, 0, 1, 1])
        result = evaluator.compute_cost(y, y_pred)
        assert 0.0 <= result["savings_pct"] <= 100.0


# ---------------------------------------------------------------------------
# EvalConfig defaults
# ---------------------------------------------------------------------------

class TestEvalConfig:
    def test_default_fpr_targets(self):
        cfg = EvalConfig()
        assert cfg.fpr_targets == [0.01, 0.05]

    def test_custom_costs(self):
        cfg = EvalConfig(cost_fn=50.0, cost_fp=2.0)
        assert cfg.cost_fn == 50.0
        assert cfg.cost_fp == 2.0
