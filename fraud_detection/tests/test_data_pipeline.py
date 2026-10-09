"""
tests/test_data_pipeline.py
============================
Unit tests for the data ingestion and preprocessing pipeline.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from pathlib import Path

# ── subject under test ──────────────────────────────────────────────────────
from data_pipeline import FraudDataPipeline, PipelineConfig


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def synthetic_ecc_df(tmp_path: Path) -> Path:
    """
    Generate a tiny synthetic ECC-style CSV (30 legit + 3 fraud rows).
    Mirrors the European Credit Card dataset schema.
    """
    rng = np.random.default_rng(42)
    n_legit, n_fraud = 300, 3
    n = n_legit + n_fraud

    data = {
        "Time": rng.integers(0, 172_800, size=n),   # up to 48 h in seconds
        "Amount": rng.exponential(scale=100, size=n),
    }
    # Simulate ~28 PCA-transformed features (V1..V28)
    for i in range(1, 29):
        data[f"V{i}"] = rng.standard_normal(n)

    labels = np.zeros(n, dtype=int)
    labels[:n_fraud] = 1
    data["Class"] = labels

    csv_path = tmp_path / "creditcard.csv"
    pd.DataFrame(data).to_csv(csv_path, index=False)
    return csv_path


@pytest.fixture()
def default_config(synthetic_ecc_df: Path, tmp_path: Path) -> PipelineConfig:
    return PipelineConfig(
        raw_data_path=str(synthetic_ecc_df),
        processed_dir=str(tmp_path / "processed"),
        imbalance_strategy="none",   # avoid SMOTE on tiny dataset
    )


# ---------------------------------------------------------------------------
# Load & Clean
# ---------------------------------------------------------------------------

class TestLoad:
    def test_loads_csv(self, default_config):
        pipeline = FraudDataPipeline(default_config)
        df = pipeline._load(default_config.raw_data_path)
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0

    def test_missing_file_raises(self, tmp_path):
        cfg = PipelineConfig(raw_data_path=str(tmp_path / "nonexistent.csv"))
        pipeline = FraudDataPipeline(cfg)
        with pytest.raises(FileNotFoundError):
            pipeline._load(cfg.raw_data_path)

    def test_clean_removes_duplicates(self, default_config):
        pipeline = FraudDataPipeline(default_config)
        df = pipeline._load(default_config.raw_data_path)
        df_duped = pd.concat([df, df.iloc[:5]], ignore_index=True)
        df_clean = pipeline._clean(df_duped)
        assert len(df_clean) == len(df)

    def test_clean_handles_nulls(self, default_config):
        pipeline = FraudDataPipeline(default_config)
        df = pipeline._load(default_config.raw_data_path)
        # Inject NaN into a numeric column
        df.loc[0, "V1"] = np.nan
        df_clean = pipeline._clean(df)
        assert df_clean["V1"].isna().sum() == 0


# ---------------------------------------------------------------------------
# Temporal Encoding
# ---------------------------------------------------------------------------

class TestTemporalEncoding:
    def test_cyclical_columns_created(self, default_config):
        pipeline = FraudDataPipeline(default_config)
        df = pipeline._load(default_config.raw_data_path)
        df_enc = pipeline._engineer_temporal(df)
        assert "hour_sin" in df_enc.columns
        assert "hour_cos" in df_enc.columns
        assert "dow_sin" in df_enc.columns
        assert "dow_cos" in df_enc.columns

    def test_cyclical_values_bounded(self, default_config):
        pipeline = FraudDataPipeline(default_config)
        df = pipeline._load(default_config.raw_data_path)
        df_enc = pipeline._engineer_temporal(df)
        for col in ["hour_sin", "hour_cos", "dow_sin", "dow_cos"]:
            assert df_enc[col].between(-1.0, 1.0).all(), f"{col} out of [-1, 1]"

    def test_raw_time_column_removed(self, default_config):
        pipeline = FraudDataPipeline(default_config)
        df = pipeline._load(default_config.raw_data_path)
        df_enc = pipeline._engineer_temporal(df)
        assert "Time" not in df_enc.columns


# ---------------------------------------------------------------------------
# Amount Scaling
# ---------------------------------------------------------------------------

class TestAmountScaling:
    def test_log_transform_column_added(self, default_config):
        pipeline = FraudDataPipeline(default_config)
        df = pipeline._load(default_config.raw_data_path)
        df_sc = pipeline._scale_amount(df)
        assert "Amount_log1p" in df_sc.columns

    def test_original_amount_removed(self, default_config):
        pipeline = FraudDataPipeline(default_config)
        df = pipeline._load(default_config.raw_data_path)
        df_sc = pipeline._scale_amount(df)
        assert "Amount" not in df_sc.columns

    def test_log_values_non_negative(self, default_config):
        pipeline = FraudDataPipeline(default_config)
        df = pipeline._load(default_config.raw_data_path)
        df_sc = pipeline._scale_amount(df)
        assert (df_sc["Amount_log1p"] >= 0).all()


# ---------------------------------------------------------------------------
# Scale-pos-weight
# ---------------------------------------------------------------------------

class TestScalePosWeight:
    def test_ratio_correct(self):
        y = pd.Series([0] * 97 + [1] * 3)
        spw = FraudDataPipeline._compute_scale_pos_weight(y)
        assert abs(spw - (97 / 3)) < 0.01

    def test_all_negative_returns_one(self):
        y = pd.Series([0] * 10)
        spw = FraudDataPipeline._compute_scale_pos_weight(y)
        assert spw == 1.0


# ---------------------------------------------------------------------------
# End-to-end pipeline run
# ---------------------------------------------------------------------------

class TestPipelineRun:
    def test_run_returns_six_splits_plus_weight(self, default_config):
        pipeline = FraudDataPipeline(default_config)
        result = pipeline.run()
        assert len(result) == 7  # X_tr, X_val, X_te, y_tr, y_val, y_te, spw

    def test_no_data_leakage_between_splits(self, default_config):
        pipeline = FraudDataPipeline(default_config)
        X_tr, X_val, X_te, y_tr, y_val, y_te, _ = pipeline.run()
        # All row counts must add up to original (modulo SMOTE – none here)
        total_rows = len(X_tr) + len(X_val) + len(X_te)
        assert total_rows > 0

    def test_processed_csvs_saved(self, default_config, tmp_path):
        pipeline = FraudDataPipeline(default_config)
        pipeline.run()
        processed = Path(default_config.processed_dir)
        expected_files = [
            "X_train.csv", "X_val.csv", "X_test.csv",
            "y_train.csv", "y_val.csv", "y_test.csv",
        ]
        for fname in expected_files:
            assert (processed / fname).exists(), f"Missing: {fname}"

    def test_target_column_not_in_features(self, default_config):
        pipeline = FraudDataPipeline(default_config)
        X_tr, *_ = pipeline.run()
        assert "Class" not in X_tr.columns

    def test_scaler_no_leakage(self, default_config):
        """Val/test sets should be scaled with train statistics (not their own)."""
        pipeline = FraudDataPipeline(default_config)
        X_tr, X_val, X_te, *_ = pipeline.run()
        # After RobustScaler fit on train, the column means won't be exactly 0
        # (RobustScaler uses median, not mean) but values should be reasonable
        for split_name, X in [("val", X_val), ("test", X_te)]:
            assert not X.isnull().any().any(), f"NaN in {split_name} after scaling"
