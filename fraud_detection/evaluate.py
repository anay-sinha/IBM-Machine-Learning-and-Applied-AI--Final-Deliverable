"""
evaluate.py
===========
Comprehensive model evaluation and explainability suite.

Metrics
-------
- Precision-Recall AUC (Average Precision)  ← primary metric
- Recall @ fixed FPR (1%, 5%)
- ROC-AUC (secondary)
- F1, Precision, Recall at optimal / business thresholds
- Cost-Utility matrix (configurable fraud/false-alarm costs)
- Calibration curves

Explainability
--------------
- SHAP TreeExplainer   → XGBoost / LightGBM / CatBoost / stacking
- SHAP KernelExplainer → MLP / generic black-box fallback
- Global feature importance bar chart (saved as PNG)
- Local per-prediction explanation (top-k driving features + direction)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for servers
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import torch
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
    f1_score,
    precision_score,
    recall_score,
    classification_report,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

REPORTS_DIR = Path("reports")
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class EvalConfig:
    # Business cost parameters
    cost_fn: float = 10.0      # cost of a false negative (missed fraud) – relative
    cost_fp: float = 1.0       # cost of a false positive (blocked legit tx) – relative

    # Fixed FPR points at which to report recall
    fpr_targets: List[float] = None  # defaults to [0.01, 0.05]

    # SHAP
    shap_max_display: int = 20
    shap_background_samples: int = 500  # for KernelExplainer

    # Decision threshold sweep
    threshold_n_steps: int = 200

    def __post_init__(self):
        if self.fpr_targets is None:
            self.fpr_targets = [0.01, 0.05]


# ---------------------------------------------------------------------------
# Core Evaluator
# ---------------------------------------------------------------------------

class ModelEvaluator:
    """
    Evaluate one or more models on a held-out test set.

    Usage
    -----
    >>> ev = ModelEvaluator(cfg)
    >>> results = ev.evaluate_all(models_dict, X_test, y_test)
    >>> ev.print_report(results)
    """

    def __init__(self, config: Optional[EvalConfig] = None) -> None:
        self.cfg = config or EvalConfig()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate_all(
        self,
        models: Dict[str, Any],
        X_test: pd.DataFrame,
        y_test: pd.Series,
        feature_names: Optional[List[str]] = None,
    ) -> Dict[str, Dict]:
        results = {}
        for name, model in models.items():
            if name in ("autoencoder", "ae_threshold", "seq_model"):
                continue  # handled separately
            logger.info("Evaluating: %s", name)
            probs = self._get_proba(model, X_test, name)
            if probs is None:
                continue
            results[name] = self._compute_metrics(probs, y_test, name)

        # Autoencoder evaluation
        if "autoencoder" in models and "ae_threshold" in models:
            ae_scores = self._autoencoder_scores(models["autoencoder"], X_test)
            results["autoencoder"] = self._compute_metrics(ae_scores, y_test, "autoencoder")

        # Isolation Forest (anomaly score: decision_function → invert for fraud probability)
        if "isolation_forest" in models:
            iso = models["isolation_forest"]
            iso_scores = -iso.decision_function(X_test)  # higher = more anomalous
            iso_scores = (iso_scores - iso_scores.min()) / (iso_scores.max() - iso_scores.min() + 1e-9)
            results["isolation_forest"] = self._compute_metrics(
                iso_scores, y_test, "isolation_forest"
            )

        self._save_comparison_table(results)
        return results

    def explain_model(
        self,
        model: Any,
        X: pd.DataFrame,
        model_name: str,
        feature_names: Optional[List[str]] = None,
        n_samples: int = 1000,
    ) -> shap.Explanation:
        """
        Compute SHAP values for `model` on `X` (subsampled for speed).
        Auto-selects TreeExplainer vs KernelExplainer.
        """
        feat_names = feature_names or list(X.columns)
        X_sample = X.sample(min(n_samples, len(X)), random_state=42)

        explainer = self._build_explainer(model, X_sample)
        if explainer is None:
            logger.warning("Could not build SHAP explainer for %s", model_name)
            return None

        shap_vals = explainer(X_sample)
        self._plot_shap_summary(shap_vals, feat_names, model_name)
        return shap_vals

    def local_explanation(
        self,
        model: Any,
        row: pd.DataFrame,
        background: pd.DataFrame,
        top_k: int = 3,
    ) -> List[Dict]:
        """
        Return top-k SHAP driving features for a single transaction.

        Returns
        -------
        list of dicts: [{"feature": str, "value": float, "shap_value": float, "direction": str}]
        """
        explainer = self._build_explainer(model, background)
        if explainer is None:
            return []

        shap_vals = explainer(row)
        # For classifiers the output shape may be (1, n_features, 2) – take class 1
        if hasattr(shap_vals, "values"):
            vals = shap_vals.values
            if vals.ndim == 3:
                vals = vals[:, :, 1]
            vals = vals[0]  # single row
        else:
            vals = np.array(shap_vals)[0]

        feature_cols = list(row.columns)
        top_idx = np.argsort(np.abs(vals))[::-1][:top_k]

        explanations = []
        for idx in top_idx:
            sv = float(vals[idx])
            explanations.append({
                "feature": feature_cols[idx],
                "value": float(row.iloc[0, idx]),
                "shap_value": sv,
                "direction": "increases_fraud_risk" if sv > 0 else "decreases_fraud_risk",
            })
        return explanations

    def compute_cost(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> Dict[str, float]:
        """
        Cost-utility matrix evaluation.

        True Negatives  → 0 cost (correct block of legit tx is free)
        False Positives → cost_fp per transaction
        False Negatives → cost_fn per transaction (missed fraud)
        True Positives  → 0 cost (fraud caught)
        """
        TP = np.sum((y_pred == 1) & (y_true == 1))
        FP = np.sum((y_pred == 1) & (y_true == 0))
        FN = np.sum((y_pred == 0) & (y_true == 1))
        TN = np.sum((y_pred == 0) & (y_true == 0))

        total_cost = self.cfg.cost_fp * FP + self.cfg.cost_fn * FN
        baseline_cost = self.cfg.cost_fn * np.sum(y_true)  # cost if we detect nothing

        return {
            "TP": int(TP), "FP": int(FP), "FN": int(FN), "TN": int(TN),
            "total_cost": float(total_cost),
            "baseline_cost": float(baseline_cost),
            "cost_savings": float(baseline_cost - total_cost),
            "savings_pct": float((baseline_cost - total_cost) / max(baseline_cost, 1) * 100),
        }

    def print_report(self, results: Dict[str, Dict]) -> None:
        """Pretty-print evaluation table to stdout and save to reports/."""
        rows = []
        for name, m in results.items():
            rows.append({
                "Model": name,
                "PR-AUC": f"{m.get('pr_auc', 0):.4f}",
                "ROC-AUC": f"{m.get('roc_auc', 0):.4f}",
                "F1@opt": f"{m.get('f1_at_opt', 0):.4f}",
                "Recall@1%FPR": f"{m.get('recall_at_1pct_fpr', 0):.4f}",
                "Recall@5%FPR": f"{m.get('recall_at_5pct_fpr', 0):.4f}",
            })
        df = pd.DataFrame(rows).set_index("Model")
        print("\n" + "=" * 70)
        print("FRAUD DETECTION MODEL COMPARISON")
        print("=" * 70)
        print(df.to_string())
        print("=" * 70 + "\n")
        df.to_csv(REPORTS_DIR / "model_comparison.csv")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_proba(self, model: Any, X: pd.DataFrame, name: str) -> Optional[np.ndarray]:
        """Extract class-1 probability from any sklearn-compatible model."""
        try:
            if hasattr(model, "predict_proba"):
                probs = model.predict_proba(X)
                if probs.ndim == 2:
                    return probs[:, 1]
                return probs
            elif hasattr(model, "decision_function"):
                scores = model.decision_function(X)
                # Normalise to [0, 1]
                return (scores - scores.min()) / (scores.max() - scores.min() + 1e-9)
        except Exception as e:
            logger.warning("Could not get probabilities for %s: %s", name, e)
        return None

    def _compute_metrics(
        self,
        probs: np.ndarray,
        y_true: pd.Series,
        name: str,
    ) -> Dict:
        y = y_true.values

        pr_auc   = average_precision_score(y, probs)
        roc_auc  = roc_auc_score(y, probs)
        fpr_arr, tpr_arr, _ = roc_curve(y, probs)

        recall_at_fpr = {}
        for target_fpr in self.cfg.fpr_targets:
            # Find the recall value at the point where FPR first exceeds target
            idx = np.searchsorted(fpr_arr, target_fpr)
            idx = min(idx, len(tpr_arr) - 1)
            key = f"recall_at_{int(target_fpr*100)}pct_fpr"
            recall_at_fpr[key] = float(tpr_arr[idx])

        # Optimal threshold (maximise F1)
        precision_arr, recall_arr, thresholds = precision_recall_curve(y, probs)
        f1_scores = 2 * precision_arr * recall_arr / (precision_arr + recall_arr + 1e-9)
        best_idx = np.argmax(f1_scores[:-1])  # last element has no threshold
        best_thresh = float(thresholds[best_idx])
        y_pred_opt = (probs >= best_thresh).astype(int)

        cost_report = self.compute_cost(y, y_pred_opt)

        metrics = {
            "pr_auc": float(pr_auc),
            "roc_auc": float(roc_auc),
            "f1_at_opt": float(f1_scores[best_idx]),
            "precision_at_opt": float(precision_score(y, y_pred_opt, zero_division=0)),
            "recall_at_opt": float(recall_score(y, y_pred_opt, zero_division=0)),
            "optimal_threshold": best_thresh,
            **recall_at_fpr,
            **cost_report,
        }

        # Save PR curve
        self._plot_pr_curve(precision_arr, recall_arr, pr_auc, name)

        logger.info(
            "%s | PR-AUC=%.4f | ROC-AUC=%.4f | F1=%.4f | Threshold=%.4f",
            name, pr_auc, roc_auc, f1_scores[best_idx], best_thresh,
        )
        return metrics

    def _autoencoder_scores(self, ae_model: Any, X: pd.DataFrame) -> np.ndarray:
        """Compute per-row reconstruction error as anomaly score."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ae_model.eval()
        X_t = torch.tensor(X.values, dtype=torch.float32, device=device)
        with torch.no_grad():
            recon = ae_model(X_t).cpu().numpy()
        errors = np.mean((X.values - recon) ** 2, axis=1)
        # Normalise to [0, 1]
        return (errors - errors.min()) / (errors.max() - errors.min() + 1e-9)

    def _build_explainer(self, model: Any, X_background: pd.DataFrame) -> Any:
        """Choose and build the appropriate SHAP explainer."""
        # Tree models: fast TreeExplainer
        tree_types = (
            "XGBClassifier", "LGBMClassifier", "CatBoostClassifier",
            "RandomForestClassifier", "GradientBoostingClassifier",
        )
        model_type_name = type(model).__name__

        if model_type_name in tree_types:
            return shap.TreeExplainer(model)

        # Calibrated stacking wrapper – try to extract underlying model
        if hasattr(model, "calibrated_classifiers_"):
            try:
                inner = model.calibrated_classifiers_[0].estimator
                if type(inner).__name__ in tree_types:
                    return shap.TreeExplainer(inner)
            except Exception:
                pass

        # Fallback: KernelExplainer (model-agnostic but slower)
        logger.info(
            "Using KernelExplainer for %s (slower; background=%d samples)",
            model_type_name, len(X_background),
        )
        background = shap.sample(X_background, self.cfg.shap_background_samples)
        if hasattr(model, "predict_proba"):
            return shap.KernelExplainer(
                lambda x: model.predict_proba(pd.DataFrame(x, columns=X_background.columns))[:, 1],
                background,
            )
        return None

    def _plot_pr_curve(
        self,
        precision: np.ndarray,
        recall: np.ndarray,
        auc: float,
        name: str,
    ) -> None:
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.plot(recall, precision, lw=1.5, color="#3b82d4")
        ax.fill_between(recall, precision, alpha=0.15, color="#3b82d4")
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title(f"Precision-Recall Curve – {name}\nPR-AUC = {auc:.4f}")
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1.05])
        ax.grid(True, linestyle="--", alpha=0.5)
        fig.tight_layout()
        fig.savefig(REPORTS_DIR / f"pr_curve_{name}.png", dpi=150)
        plt.close(fig)

    def _plot_shap_summary(
        self,
        shap_vals: shap.Explanation,
        feature_names: List[str],
        name: str,
    ) -> None:
        fig, ax = plt.subplots(figsize=(10, 6))
        shap.plots.beeswarm(shap_vals, max_display=self.cfg.shap_max_display, show=False)
        plt.title(f"SHAP Feature Importance – {name}")
        plt.tight_layout()
        plt.savefig(REPORTS_DIR / f"shap_summary_{name}.png", dpi=150, bbox_inches="tight")
        plt.close()

    def _save_comparison_table(self, results: Dict[str, Dict]) -> None:
        rows = []
        for name, metrics in results.items():
            row = {"model": name}
            row.update(metrics)
            rows.append(row)
        df = pd.DataFrame(rows)
        df.to_csv(REPORTS_DIR / "full_metrics.csv", index=False)
        logger.info("Full metrics saved to %s/full_metrics.csv", REPORTS_DIR)


# ---------------------------------------------------------------------------
# Calibration Diagnostic
# ---------------------------------------------------------------------------

def plot_calibration_curve(
    models: Dict[str, Any],
    X: pd.DataFrame,
    y: pd.Series,
    n_bins: int = 15,
) -> None:
    """Plot calibration curves for all models into one figure."""
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration")

    colors = ["#3b82d4", "#7c5cd8", "#e87c3e", "#2ea043", "#d94040"]
    for (name, model), color in zip(models.items(), colors):
        if not hasattr(model, "predict_proba"):
            continue
        probs = model.predict_proba(X)[:, 1]
        frac_pos, mean_pred = calibration_curve(y, probs, n_bins=n_bins, strategy="uniform")
        ax.plot(mean_pred, frac_pos, "s-", label=name, color=color, lw=1.5)

    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of positives")
    ax.set_title("Calibration Curves")
    ax.legend(loc="upper left")
    ax.grid(True, linestyle="--", alpha=0.5)
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / "calibration_curves.png", dpi=150)
    plt.close(fig)
    logger.info("Calibration curves saved.")


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    from train import load_splits, load_model

    parser = argparse.ArgumentParser(description="Evaluate trained fraud detection models")
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument(
        "--models", nargs="+",
        default=["xgboost", "lightgbm", "catboost", "stacking_ensemble"],
    )
    args = parser.parse_args()

    _, _, X_test, _, _, y_test = load_splits(args.processed_dir)
    models_dict = {name: load_model(name) for name in args.models}

    evaluator = ModelEvaluator()
    results = evaluator.evaluate_all(models_dict, X_test, y_test)
    evaluator.print_report(results)

    # SHAP for the stacking ensemble
    if "stacking_ensemble" in models_dict:
        shap_vals = evaluator.explain_model(
            models_dict["stacking_ensemble"],
            X_test.sample(min(500, len(X_test)), random_state=42),
            model_name="stacking_ensemble",
        )
