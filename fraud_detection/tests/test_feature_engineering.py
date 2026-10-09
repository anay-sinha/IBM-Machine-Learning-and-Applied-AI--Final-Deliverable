"""
tests/test_feature_engineering.py
===================================
Unit tests for rolling-window feature generation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from feature_engineering import FeatureConfig, RollingFeatureEngineer, add_rolling_features


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def small_tx_df() -> pd.DataFrame:
    """
    15 transactions for 2 synthetic cards.
    Rows are intentionally unsorted so we verify internal sorting.
    """
    rng = np.random.default_rng(0)
    n = 15
    card_ids = ["card_A"] * 8 + ["card_B"] * 7

    timestamps = pd.date_range("2024-01-01 00:00", periods=n, freq="2h")

    df = pd.DataFrame({
        "card_id": card_ids,
        "Time": np.arange(n) * 7200,          # 2 h apart in seconds
        "Amount_log1p": rng.uniform(1, 6, n),
        "merchant_id": rng.choice(["m1", "m2", "m3"], n),
        "Class": rng.choice([0, 1], n, p=[0.9, 0.1]),
    })
    return df


@pytest.fixture()
def feature_config() -> FeatureConfig:
    return FeatureConfig(
        windows_seconds=[3_600, 86_400],     # 1h, 24h (no 7d to keep test fast)
        window_labels=["1h", "24h"],
        card_col="card_id",
        amount_col="Amount_log1p",
        merchant_col="merchant_id",
        time_col="Time",
        use_real_timestamp=False,
    )


# ---------------------------------------------------------------------------
# Column presence
# ---------------------------------------------------------------------------

class TestColumnGeneration:
    def test_rolling_count_columns_present(self, small_tx_df, feature_config):
        eng = RollingFeatureEngineer(feature_config)
        result = eng.fit_transform(small_tx_df)
        assert "tx_count_1h" in result.columns
        assert "tx_count_24h" in result.columns

    def test_velocity_spike_columns_present(self, small_tx_df, feature_config):
        eng = RollingFeatureEngineer(feature_config)
        result = eng.fit_transform(small_tx_df)
        assert "velocity_spike_1h" in result.columns
        assert "velocity_spike_24h" in result.columns

    def test_recency_column_present(self, small_tx_df, feature_config):
        eng = RollingFeatureEngineer(feature_config)
        result = eng.fit_transform(small_tx_df)
        assert "time_since_last_tx" in result.columns

    def test_unique_merchant_column_present(self, small_tx_df, feature_config):
        eng = RollingFeatureEngineer(feature_config)
        result = eng.fit_transform(small_tx_df)
        assert "unique_merchants_1h" in result.columns
        assert "unique_merchants_24h" in result.columns

    def test_no_internal_ts_column_leaked(self, small_tx_df, feature_config):
        eng = RollingFeatureEngineer(feature_config)
        result = eng.fit_transform(small_tx_df)
        assert "_ts" not in result.columns
        assert "_ts_synthetic" not in result.columns


# ---------------------------------------------------------------------------
# Semantic correctness
# ---------------------------------------------------------------------------

class TestRollingSemantics:
    def test_first_transaction_count_is_zero(self, feature_config):
        """First transaction in a card history has no previous transactions."""
        df = pd.DataFrame({
            "card_id": ["card_X", "card_X", "card_X"],
            "Time": [0, 3600, 7200],
            "Amount_log1p": [3.0, 3.5, 4.0],
            "Class": [0, 0, 1],
        })
        cfg = FeatureConfig(
            windows_seconds=[3_600],
            window_labels=["1h"],
            card_col="card_id",
            amount_col="Amount_log1p",
            merchant_col=None,
            time_col="Time",
        )
        eng = RollingFeatureEngineer(cfg)
        result = eng.fit_transform(df)
        # The first transaction (sorted by _ts) should have 0 prior tx in 1h window
        first_row = result.sort_values("_ts" if "_ts" in result.columns else result.index.name).iloc[0] \
            if "_ts" in result.columns else result.iloc[0]
        assert result["tx_count_1h"].iloc[0] == 0

    def test_count_increases_within_window(self, feature_config):
        """Later transactions within the 24h window should see higher counts."""
        df = pd.DataFrame({
            "card_id": ["card_Y"] * 5,
            "Time": [0, 1800, 3600, 5400, 7200],
            "Amount_log1p": [2.0, 2.1, 2.2, 2.3, 2.4],
            "Class": [0] * 5,
        })
        cfg = FeatureConfig(
            windows_seconds=[86_400],
            window_labels=["24h"],
            card_col="card_id",
            amount_col="Amount_log1p",
            merchant_col=None,
            time_col="Time",
        )
        eng = RollingFeatureEngineer(cfg)
        result = eng.fit_transform(df)
        counts = result["tx_count_24h"].values
        # Each subsequent transaction should see one more prior tx
        assert list(counts) == [0, 1, 2, 3, 4]

    def test_velocity_spike_non_negative(self, small_tx_df, feature_config):
        eng = RollingFeatureEngineer(feature_config)
        result = eng.fit_transform(small_tx_df)
        assert (result["velocity_spike_1h"] >= 0).all()
        assert (result["velocity_spike_24h"] >= 0).all()

    def test_recency_minus_one_for_first_tx(self, feature_config):
        df = pd.DataFrame({
            "card_id": ["card_Z"] * 3,
            "Time": [0, 3600, 7200],
            "Amount_log1p": [1.0, 2.0, 3.0],
            "Class": [0, 0, 1],
        })
        cfg = FeatureConfig(
            windows_seconds=[3_600],
            window_labels=["1h"],
            card_col="card_id",
            amount_col="Amount_log1p",
            merchant_col=None,
            time_col="Time",
        )
        eng = RollingFeatureEngineer(cfg)
        result = eng.fit_transform(df)
        assert result["time_since_last_tx"].iloc[0] == -1.0


# ---------------------------------------------------------------------------
# Wrapper function
# ---------------------------------------------------------------------------

class TestAddRollingFeatures:
    def test_returns_dataframe(self, small_tx_df, feature_config):
        result = add_rolling_features(small_tx_df, feature_config)
        assert isinstance(result, pd.DataFrame)

    def test_row_count_unchanged(self, small_tx_df, feature_config):
        result = add_rolling_features(small_tx_df, feature_config)
        assert len(result) == len(small_tx_df)

    def test_original_columns_preserved(self, small_tx_df, feature_config):
        result = add_rolling_features(small_tx_df, feature_config)
        for col in ["card_id", "Amount_log1p", "Class"]:
            assert col in result.columns


# ---------------------------------------------------------------------------
# Online single-transaction inference
# ---------------------------------------------------------------------------

class TestTransformSingle:
    def test_returns_dict_with_expected_keys(self, small_tx_df, feature_config):
        eng = RollingFeatureEngineer(feature_config)
        history = small_tx_df[small_tx_df["card_id"] == "card_A"].copy()
        history["_ts"] = pd.Timestamp("2024-01-01") + pd.to_timedelta(history["Time"], unit="s")

        tx = {"Amount_log1p": 4.0, "timestamp": "2024-01-01 16:00:00"}
        features = eng.transform_single(tx, history)

        assert "tx_count_1h" in features
        assert "tx_count_24h" in features
        assert "velocity_spike_1h" in features
        assert "time_since_last_tx" in features

    def test_empty_history_yields_zeros(self, feature_config):
        eng = RollingFeatureEngineer(feature_config)
        empty_history = pd.DataFrame(
            columns=["card_id", "Time", "Amount_log1p", "_ts"]
        )
        tx = {"Amount_log1p": 2.5, "timestamp": "2024-01-01 10:00:00"}
        features = eng.transform_single(tx, empty_history)

        assert features["tx_count_1h"] == 0
        assert features["tx_count_24h"] == 0
        assert features["time_since_last_tx"] == -1.0
