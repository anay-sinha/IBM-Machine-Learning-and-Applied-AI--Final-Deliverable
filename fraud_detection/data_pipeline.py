"""
data_pipeline.py
================
Enterprise-grade data ingestion, preprocessing, and re-sampling pipeline
for financial fraud detection.

Key responsibilities:
  - Load raw transaction CSVs (European Credit Card or IEEE-CIS schema)
  - Robust scaling for transaction amounts and log-transform skewed features
  - Cyclical (sin/cos) encoding for temporal features (hour, day-of-week)
  - Class-imbalance handling via SMOTE-Tomek, ADASYN, or cost-weight pass-through
  - Stratified temporal-aware train/val/test splits
"""

from __future__ import annotations

import io
import logging
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional, Tuple

import numpy as np
import pandas as pd
from imblearn.combine import SMOTETomek
from imblearn.over_sampling import ADASYN
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class PipelineConfig:
    """All tuneable knobs for the data pipeline, centralised in one place."""

    raw_data_path: str = "data/transactions.csv"
    processed_dir: str = "data/processed"

    # Column name mappings – override for different dataset schemas
    target_col: str = "Class"            # 1 = fraud, 0 = legitimate
    amount_col: str = "Amount"
    time_col: str = "Time"               # seconds elapsed (ECC) or timestamp
    timestamp_col: Optional[str] = None  # if a real datetime column exists

    # Temporal encoding
    encode_hour: bool = True
    encode_dow: bool = True              # day-of-week

    # Imbalance strategy: "smote_tomek" | "adasyn" | "cost_weight" | "none"
    imbalance_strategy: Literal["smote_tomek", "adasyn", "cost_weight", "none"] = "smote_tomek"

    # Splits
    test_size: float = 0.15
    val_size: float = 0.15
    random_state: int = 42

    # Feature columns to drop before modelling (IDs, raw timestamps, etc.)
    drop_cols: list[str] = field(default_factory=lambda: [])

    # Amount scaling
    log_transform_amount: bool = True


# ---------------------------------------------------------------------------
# Core Pipeline Class
# ---------------------------------------------------------------------------

class FraudDataPipeline:
    """
    End-to-end preprocessing pipeline.

    Usage
    -----
    >>> cfg = PipelineConfig(raw_data_path="data/creditcard.csv")
    >>> pipeline = FraudDataPipeline(cfg)
    >>> X_train, X_val, X_test, y_train, y_val, y_test, scale_pos_weight = pipeline.run()
    """

    def __init__(self, config: PipelineConfig) -> None:
        self.cfg = config
        self.scaler = RobustScaler()
        self.feature_names_: list[str] = []
        self.scale_pos_weight_: float = 1.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
    ) -> Tuple[
        pd.DataFrame, pd.DataFrame, pd.DataFrame,
        pd.Series, pd.Series, pd.Series,
        float,
    ]:
        """Execute the full pipeline and return split arrays + scale_pos_weight."""
        logger.info("Loading raw data from %s", self.cfg.raw_data_path)
        df = self._load(self.cfg.raw_data_path)

        logger.info("Shape after load: %s | Fraud rate: %.4f%%",
                    df.shape, df[self.cfg.target_col].mean() * 100)

        df = self._clean(df)
        df = self._engineer_temporal(df)
        df = self._scale_amount(df)
        df = self._drop_columns(df)

        X = df.drop(columns=[self.cfg.target_col])
        y = df[self.cfg.target_col]

        self.feature_names_ = list(X.columns)
        self.scale_pos_weight_ = self._compute_scale_pos_weight(y)

        X_tr, X_val, X_te, y_tr, y_val, y_te = self._split(X, y)

        # Fit scaler on train only, transform all splits
        X_tr = pd.DataFrame(
            self.scaler.fit_transform(X_tr), columns=X_tr.columns, index=X_tr.index
        )
        X_val = pd.DataFrame(
            self.scaler.transform(X_val), columns=X_val.columns, index=X_val.index
        )
        X_te = pd.DataFrame(
            self.scaler.transform(X_te), columns=X_te.columns, index=X_te.index
        )

        X_tr, y_tr = self._handle_imbalance(X_tr, y_tr)

        logger.info(
            "Final splits – train: %s | val: %s | test: %s",
            X_tr.shape, X_val.shape, X_te.shape,
        )
        logger.info("Train class distribution after re-sampling: %s", y_tr.value_counts().to_dict())

        self._save(X_tr, X_val, X_te, y_tr, y_val, y_te)

        return X_tr, X_val, X_te, y_tr, y_val, y_te, self.scale_pos_weight_

    # ------------------------------------------------------------------
    # Internal steps
    # ------------------------------------------------------------------

    def _load(self, path: str) -> pd.DataFrame:
        """
        Load a CSV or ZIP file into a DataFrame.

        ZIP handling
        ------------
        - If the path ends with `.zip`, the archive is opened in-memory.
        - The first `.csv` entry found inside the archive is read.
        - No file is extracted to disk; memory only.
        - If the ZIP contains multiple CSVs, the first one (alphabetically) is used;
          set `raw_data_path` to point at the specific member if needed.
        """
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Dataset not found at {path}")

        if p.suffix.lower() == ".zip":
            with zipfile.ZipFile(p, "r") as zf:
                csv_members = sorted(
                    name for name in zf.namelist()
                    if name.lower().endswith(".csv") and not name.startswith("__MACOSX")
                )
                if not csv_members:
                    raise ValueError(f"No CSV file found inside {path}")
                if len(csv_members) > 1:
                    logger.warning(
                        "ZIP contains multiple CSVs %s — loading '%s'. "
                        "Set raw_data_path to the specific member to override.",
                        csv_members, csv_members[0],
                    )
                member = csv_members[0]
                logger.info("Reading '%s' from ZIP archive %s", member, p.name)
                with zf.open(member) as f:
                    df = pd.read_csv(io.TextIOWrapper(f, encoding="utf-8"), low_memory=False)
        else:
            df = pd.read_csv(p, low_memory=False)

        logger.info("Columns: %s", list(df.columns))
        return df

    def _clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """Drop duplicates, handle nulls conservatively."""
        before = len(df)
        df = df.drop_duplicates()
        df = df.dropna(subset=[self.cfg.target_col])

        # For numeric feature columns, impute with median
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        num_cols = [c for c in num_cols if c != self.cfg.target_col]
        df[num_cols] = df[num_cols].fillna(df[num_cols].median())

        logger.info("Cleaned: removed %d rows (dupes/nulls)", before - len(df))
        return df

    def _engineer_temporal(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Derive hour-of-day and day-of-week features, then apply cyclical
        sin/cos encoding so the model sees temporal periodicity.

        Supports two modes:
          1. Real datetime column (self.cfg.timestamp_col is set)
          2. ECC-style `Time` column = seconds since first transaction
        """
        if self.cfg.timestamp_col and self.cfg.timestamp_col in df.columns:
            ts = pd.to_datetime(df[self.cfg.timestamp_col], errors="coerce")
            df["_hour"] = ts.dt.hour
            df["_dow"] = ts.dt.dayofweek
        elif self.cfg.time_col in df.columns:
            # ECC dataset: Time is elapsed seconds – wrap to hour-of-day
            df["_hour"] = (df[self.cfg.time_col] // 3600) % 24
            df["_dow"] = (df[self.cfg.time_col] // 86400) % 7

        if self.cfg.encode_hour and "_hour" in df.columns:
            df["hour_sin"] = np.sin(2 * np.pi * df["_hour"] / 24)
            df["hour_cos"] = np.cos(2 * np.pi * df["_hour"] / 24)
            df.drop(columns=["_hour"], inplace=True)

        if self.cfg.encode_dow and "_dow" in df.columns:
            df["dow_sin"] = np.sin(2 * np.pi * df["_dow"] / 7)
            df["dow_cos"] = np.cos(2 * np.pi * df["_dow"] / 7)
            df.drop(columns=["_dow"], inplace=True)

        # Drop raw time col – not useful as raw input
        if self.cfg.time_col in df.columns:
            df.drop(columns=[self.cfg.time_col], inplace=True)

        return df

    def _scale_amount(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Log1p transform + keep original column for model flexibility.
        Note: RobustScaler applied AFTER split to avoid leakage.
        """
        if self.cfg.amount_col in df.columns:
            if self.cfg.log_transform_amount:
                df[f"{self.cfg.amount_col}_log1p"] = np.log1p(df[self.cfg.amount_col])
            df.drop(columns=[self.cfg.amount_col], inplace=True)
        return df

    def _drop_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        cols_to_drop = [c for c in self.cfg.drop_cols if c in df.columns]
        if cols_to_drop:
            df.drop(columns=cols_to_drop, inplace=True)
        return df

    def _split(
        self, X: pd.DataFrame, y: pd.Series
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
        """
        Stratified split: train | val | test.
        Uses a two-step split to honour both val and test fractions.
        """
        X_tr, X_tmp, y_tr, y_tmp = train_test_split(
            X, y,
            test_size=self.cfg.test_size + self.cfg.val_size,
            stratify=y,
            random_state=self.cfg.random_state,
        )
        relative_val = self.cfg.val_size / (self.cfg.test_size + self.cfg.val_size)
        X_val, X_te, y_val, y_te = train_test_split(
            X_tmp, y_tmp,
            test_size=1.0 - relative_val,
            stratify=y_tmp,
            random_state=self.cfg.random_state,
        )
        return X_tr, X_val, X_te, y_tr, y_val, y_te

    def _handle_imbalance(
        self, X: pd.DataFrame, y: pd.Series
    ) -> Tuple[pd.DataFrame, pd.Series]:
        strategy = self.cfg.imbalance_strategy

        if strategy == "smote_tomek":
            logger.info("Applying SMOTE-Tomek re-sampling...")
            resampler = SMOTETomek(random_state=self.cfg.random_state)
            X_res, y_res = resampler.fit_resample(X, y)

        elif strategy == "adasyn":
            logger.info("Applying ADASYN re-sampling...")
            resampler = ADASYN(random_state=self.cfg.random_state)
            X_res, y_res = resampler.fit_resample(X, y)

        elif strategy in ("cost_weight", "none"):
            logger.info(
                "Imbalance strategy '%s': no re-sampling applied (use scale_pos_weight=%.2f).",
                strategy, self.scale_pos_weight_,
            )
            return X, y

        else:
            raise ValueError(f"Unknown imbalance strategy: {strategy}")

        X_res = pd.DataFrame(X_res, columns=X.columns)
        y_res = pd.Series(y_res, name=y.name)
        return X_res, y_res

    @staticmethod
    def _compute_scale_pos_weight(y: pd.Series) -> float:
        """Ratio of negative to positive samples – used by XGBoost / LightGBM."""
        n_neg = (y == 0).sum()
        n_pos = (y == 1).sum()
        if n_pos == 0:
            return 1.0
        return float(n_neg / n_pos)

    def _save(
        self,
        X_tr, X_val, X_te,
        y_tr, y_val, y_te,
    ) -> None:
        out = Path(self.cfg.processed_dir)
        out.mkdir(parents=True, exist_ok=True)

        for name, arr in [
            ("X_train", X_tr), ("X_val", X_val), ("X_test", X_te),
            ("y_train", y_tr), ("y_val", y_val), ("y_test", y_te),
        ]:
            arr.to_csv(out / f"{name}.csv", index=False)

        logger.info("Processed splits saved to %s", out)


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Run the fraud data pipeline")
    parser.add_argument("--config", type=str, default=None, help="Path to JSON config")
    parser.add_argument("--data", type=str, default="data/creditcard.csv")
    parser.add_argument(
        "--strategy",
        choices=["smote_tomek", "adasyn", "cost_weight", "none"],
        default="smote_tomek",
    )
    args = parser.parse_args()

    if args.config:
        with open(args.config) as f:
            cfg_dict = json.load(f)
        cfg = PipelineConfig(**cfg_dict)
    else:
        cfg = PipelineConfig(
            raw_data_path=args.data,
            imbalance_strategy=args.strategy,
        )

    pipeline = FraudDataPipeline(cfg)
    pipeline.run()
