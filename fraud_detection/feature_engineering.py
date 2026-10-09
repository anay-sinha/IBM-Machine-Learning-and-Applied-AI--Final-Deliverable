"""
feature_engineering.py
=======================
Rolling-window velocity and aggregation features for fraud detection.

Computes per-card / per-user behavioural signals across multiple lookback
windows (1 h, 24 h, 7 d) including:
  - Transaction count
  - Total and mean spend
  - Velocity spike ratio  (current amount vs. rolling mean)
  - Time-since-last transaction (recency)
  - Unique merchant count
  - Std-dev of amounts  (spending variability)

Designed to work both offline (DataFrame-based) and online (single-row
incremental update against a Redis-backed feature store stub).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class FeatureConfig:
    """Configuration for rolling-window feature generation."""

    # Lookback windows in seconds
    windows_seconds: List[int] = field(
        default_factory=lambda: [3_600, 86_400, 604_800]  # 1h, 24h, 7d
    )
    window_labels: List[str] = field(
        default_factory=lambda: ["1h", "24h", "7d"]
    )

    # Column references
    card_col: str = "card_id"        # grouping key (user / card identifier)
    amount_col: str = "Amount_log1p" # pre-scaled amount column from pipeline
    merchant_col: Optional[str] = "merchant_id"
    timestamp_col: str = "timestamp" # real datetime; or "Time" seconds fallback
    time_col: str = "Time"           # ECC seconds-elapsed column

    # Whether to use real timestamp or derive from ECC Time column
    use_real_timestamp: bool = False


# ---------------------------------------------------------------------------
# Feature Engineer
# ---------------------------------------------------------------------------

class RollingFeatureEngineer:
    """
    Generates rolling behavioural aggregation features.

    Workflow
    --------
    1. Call `fit_transform(df)` for offline batch feature generation.
    2. Call `transform_single(row, history_df)` for online single-transaction scoring.

    Parameters
    ----------
    config : FeatureConfig
    """

    def __init__(self, config: Optional[FeatureConfig] = None) -> None:
        self.cfg = config or FeatureConfig()
        self._generated_cols: List[str] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add rolling features to a sorted transaction DataFrame.

        The DataFrame MUST be sorted by (card_id, timestamp) before calling.
        """
        df = df.copy()
        df = self._ensure_timestamp(df)
        df = df.sort_values([self.cfg.card_col, "_ts"]).reset_index(drop=True)

        logger.info("Generating rolling window features on %d rows...", len(df))

        new_feature_frames: List[pd.DataFrame] = []

        for window_s, label in zip(self.cfg.windows_seconds, self.cfg.window_labels):
            features = self._rolling_features_for_window(df, window_s, label)
            new_feature_frames.append(features)

        # Recency: time since last transaction per card
        recency = self._recency_feature(df)
        new_feature_frames.append(recency)

        all_new = pd.concat(new_feature_frames, axis=1)
        df = pd.concat([df, all_new], axis=1)

        # Drop internal timestamp column if it was synthesised
        if "_ts_synthetic" in df.columns:
            df.drop(columns=["_ts_synthetic", "_ts"], inplace=True, errors="ignore")
        else:
            df.drop(columns=["_ts"], inplace=True, errors="ignore")

        self._generated_cols = list(all_new.columns) + ["time_since_last_tx"]
        logger.info("Generated %d new features.", len(self._generated_cols))
        return df

    def transform_single(
        self,
        transaction: Dict,
        history: pd.DataFrame,
    ) -> Dict:
        """
        Compute rolling features for a single incoming transaction
        against a historical DataFrame (most recent N rows for this card).

        Suitable for real-time scoring in the FastAPI service.

        Parameters
        ----------
        transaction : dict with keys matching column names
        history : pd.DataFrame of prior transactions for this card

        Returns
        -------
        dict with all rolling feature values
        """
        ts = pd.Timestamp(transaction.get("timestamp", pd.Timestamp.now()))
        features: Dict[str, float] = {}

        for window_s, label in zip(self.cfg.windows_seconds, self.cfg.window_labels):
            cutoff = ts - pd.Timedelta(seconds=window_s)
            window_df = history[history["_ts"] >= cutoff] if "_ts" in history.columns else history

            amt = transaction.get(self.cfg.amount_col, 0.0)
            rolling_amounts = window_df[self.cfg.amount_col].values if len(window_df) else np.array([])

            features[f"tx_count_{label}"] = len(window_df)
            features[f"tx_total_amt_{label}"] = float(np.sum(rolling_amounts))
            features[f"tx_mean_amt_{label}"] = float(np.mean(rolling_amounts)) if len(rolling_amounts) else 0.0
            features[f"tx_std_amt_{label}"] = float(np.std(rolling_amounts)) if len(rolling_amounts) > 1 else 0.0

            mean_amt = features[f"tx_mean_amt_{label}"]
            features[f"velocity_spike_{label}"] = (
                float(amt / (mean_amt + 1e-9)) if mean_amt > 0 else float(amt)
            )

            if self.cfg.merchant_col and self.cfg.merchant_col in window_df.columns:
                features[f"unique_merchants_{label}"] = int(
                    window_df[self.cfg.merchant_col].nunique()
                )

        # Recency
        if len(history) and "_ts" in history.columns:
            last_ts = history["_ts"].max()
            features["time_since_last_tx"] = (ts - last_ts).total_seconds()
        else:
            features["time_since_last_tx"] = -1.0

        return features

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ensure_timestamp(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ensure a unified `_ts` datetime column exists."""
        if self.cfg.use_real_timestamp and self.cfg.timestamp_col in df.columns:
            df["_ts"] = pd.to_datetime(df[self.cfg.timestamp_col], errors="coerce")
        elif self.cfg.time_col in df.columns:
            # ECC: seconds since first transaction → synthesise absolute timestamp
            base = pd.Timestamp("2023-01-01")
            df["_ts"] = base + pd.to_timedelta(df[self.cfg.time_col], unit="s")
            df["_ts_synthetic"] = True
        else:
            raise ValueError(
                "No usable timestamp column found. "
                f"Looked for '{self.cfg.timestamp_col}' and '{self.cfg.time_col}'."
            )
        return df

    def _rolling_features_for_window(
        self, df: pd.DataFrame, window_s: int, label: str
    ) -> pd.DataFrame:
        """
        For each row, look back `window_s` seconds within the same card group
        and compute aggregation statistics.

        Uses pandas merge_asof-style logic via a vectorised group apply.
        """
        results = []

        for _, grp in df.groupby(self.cfg.card_col, sort=False):
            grp = grp.sort_values("_ts").reset_index(drop=True)
            grp_results = []

            for i, row in grp.iterrows():
                cutoff = row["_ts"] - pd.Timedelta(seconds=window_s)
                # Exclude current transaction from its own window
                hist = grp[(grp["_ts"] >= cutoff) & (grp["_ts"] < row["_ts"])]
                amounts = hist[self.cfg.amount_col].values
                current_amt = row[self.cfg.amount_col]

                row_features = {
                    "orig_idx": i,
                    f"tx_count_{label}": len(hist),
                    f"tx_total_amt_{label}": float(np.sum(amounts)),
                    f"tx_mean_amt_{label}": float(np.mean(amounts)) if len(amounts) else 0.0,
                    f"tx_std_amt_{label}": float(np.std(amounts)) if len(amounts) > 1 else 0.0,
                    f"velocity_spike_{label}": (
                        float(current_amt / (np.mean(amounts) + 1e-9))
                        if len(amounts) > 0 else float(current_amt)
                    ),
                }

                if self.cfg.merchant_col and self.cfg.merchant_col in hist.columns:
                    row_features[f"unique_merchants_{label}"] = int(
                        hist[self.cfg.merchant_col].nunique()
                    )

                grp_results.append(row_features)

            results.extend(grp_results)

        result_df = pd.DataFrame(results).set_index("orig_idx").sort_index()
        # Drop orig_idx column – it's the index now
        feat_cols = [c for c in result_df.columns if c != "orig_idx"]
        return result_df[feat_cols]

    def _recency_feature(self, df: pd.DataFrame) -> pd.DataFrame:
        """Seconds since the previous transaction on the same card."""
        df["time_since_last_tx"] = (
            df.groupby(self.cfg.card_col)["_ts"]
            .diff()
            .dt.total_seconds()
            .fillna(-1)
        )
        recency = df[["time_since_last_tx"]].copy()
        df.drop(columns=["time_since_last_tx"], inplace=True)
        return recency

    @property
    def generated_feature_names(self) -> List[str]:
        return self._generated_cols


# ---------------------------------------------------------------------------
# Convenience wrapper for pipeline integration
# ---------------------------------------------------------------------------

def add_rolling_features(
    df: pd.DataFrame,
    config: Optional[FeatureConfig] = None,
) -> pd.DataFrame:
    """
    Drop-in function to add rolling features to a transaction DataFrame.

    Parameters
    ----------
    df : pd.DataFrame – must contain card_id, amount, and a time column
    config : FeatureConfig (uses defaults if None)

    Returns
    -------
    pd.DataFrame with additional rolling feature columns
    """
    engineer = RollingFeatureEngineer(config)
    return engineer.fit_transform(df)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate rolling features for a transaction CSV")
    parser.add_argument("--input", required=True, help="Path to transactions CSV")
    parser.add_argument("--output", default="data/features.csv")
    parser.add_argument("--card-col", default="card_id")
    parser.add_argument("--amount-col", default="Amount_log1p")
    args = parser.parse_args()

    cfg = FeatureConfig(card_col=args.card_col, amount_col=args.amount_col)
    df_in = pd.read_csv(args.input)
    df_out = add_rolling_features(df_in, cfg)
    df_out.to_csv(args.output, index=False)
    logger.info("Features saved to %s", args.output)
