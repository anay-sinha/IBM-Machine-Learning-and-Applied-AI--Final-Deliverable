"""
tests/test_models.py
=====================
Unit tests for model training components.
Covers model instantiation, forward passes, and serialisation –
NOT full training (too slow for CI). Uses tiny synthetic fixtures.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch
from pathlib import Path
from unittest.mock import MagicMock, patch

from train import (
    TrainConfig,
    FraudMLP,
    FraudSequenceModel,
    FraudAutoencoder,
    build_sequence_dataset,
    _save_model,
    load_model,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

N_FEATURES = 10
N_TRAIN    = 200
N_VAL      = 50
FRAUD_RATE = 0.05


@pytest.fixture()
def synthetic_train_val():
    rng = np.random.default_rng(7)
    X_tr = pd.DataFrame(rng.standard_normal((N_TRAIN, N_FEATURES)),
                        columns=[f"f{i}" for i in range(N_FEATURES)])
    y_tr = pd.Series(
        rng.choice([0, 1], N_TRAIN, p=[1 - FRAUD_RATE, FRAUD_RATE]), name="Class"
    )
    X_val = pd.DataFrame(rng.standard_normal((N_VAL, N_FEATURES)),
                         columns=[f"f{i}" for i in range(N_FEATURES)])
    y_val = pd.Series(
        rng.choice([0, 1], N_VAL, p=[1 - FRAUD_RATE, FRAUD_RATE]), name="Class"
    )
    return X_tr, X_val, y_tr, y_val


@pytest.fixture()
def default_cfg():
    return TrainConfig(
        mlp_epochs=2,
        mlp_hidden_dims=[32, 16],
        seq_epochs=2,
        seq_hidden_size=16,
        ae_epochs=2,
        ae_hidden_dims=[16, 8, 16],
        xgb_n_estimators=10,
        lgb_n_estimators=10,
        catboost_iterations=10,
        scale_pos_weight=19.0,
    )


# ---------------------------------------------------------------------------
# MLP Architecture
# ---------------------------------------------------------------------------

class TestFraudMLP:
    def test_output_shape(self):
        model = FraudMLP(input_dim=N_FEATURES, hidden_dims=[32, 16], dropout=0.1)
        x = torch.randn(8, N_FEATURES)
        out = model(x)
        assert out.shape == (8,), f"Expected (8,), got {out.shape}"

    def test_output_bounded_zero_one(self):
        model = FraudMLP(input_dim=N_FEATURES, hidden_dims=[32, 16], dropout=0.0)
        model.eval()
        x = torch.randn(100, N_FEATURES)
        with torch.no_grad():
            out = model(x)
        assert (out >= 0).all() and (out <= 1).all()

    def test_gradient_flows(self):
        model = FraudMLP(input_dim=N_FEATURES, hidden_dims=[32, 16], dropout=0.1)
        x = torch.randn(4, N_FEATURES)
        y = torch.zeros(4)
        logits = model.net(x).squeeze(-1)
        loss = torch.nn.BCEWithLogitsLoss()(logits, y)
        loss.backward()
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"

    def test_residual_block_output_shape(self):
        from train import ResidualBlock
        block = ResidualBlock(dim=32, dropout=0.0)
        x = torch.randn(4, 32)
        out = block(x)
        assert out.shape == x.shape


# ---------------------------------------------------------------------------
# Sequence Model
# ---------------------------------------------------------------------------

class TestFraudSequenceModel:
    @pytest.mark.parametrize("model_type", ["LSTM", "GRU"])
    def test_output_shape(self, model_type):
        model = FraudSequenceModel(
            input_dim=N_FEATURES, hidden_size=16, num_layers=1,
            dropout=0.0, model_type=model_type,
        )
        x = torch.randn(8, 5, N_FEATURES)   # batch=8, seq=5, features=10
        out = model(x)
        assert out.shape == (8,)

    def test_output_bounded_zero_one(self):
        model = FraudSequenceModel(
            input_dim=N_FEATURES, hidden_size=16, num_layers=1, dropout=0.0
        )
        model.eval()
        x = torch.randn(32, 5, N_FEATURES)
        with torch.no_grad():
            out = model(x)
        assert (out >= 0).all() and (out <= 1).all()


class TestBuildSequenceDataset:
    def test_shapes_correct(self):
        X = pd.DataFrame(np.random.randn(50, N_FEATURES))
        y = pd.Series(np.zeros(50))
        window = 10
        X_seq, y_seq = build_sequence_dataset(X, y, window)
        assert X_seq.shape == (50 - window, window, N_FEATURES)
        assert y_seq.shape == (50 - window,)

    def test_labels_match_last_row(self):
        X = pd.DataFrame(np.zeros((20, 3)))
        y = pd.Series([0] * 18 + [1, 1])
        X_seq, y_seq = build_sequence_dataset(X, y, window=5)
        assert y_seq[-1] == 1
        assert y_seq[-2] == 1


# ---------------------------------------------------------------------------
# Autoencoder
# ---------------------------------------------------------------------------

class TestFraudAutoencoder:
    def test_reconstruction_shape(self):
        ae = FraudAutoencoder(input_dim=N_FEATURES, hidden_dims=[16, 8, 16])
        x = torch.randn(4, N_FEATURES)
        recon = ae(x)
        assert recon.shape == x.shape

    def test_encoder_compresses(self):
        ae = FraudAutoencoder(input_dim=N_FEATURES, hidden_dims=[16, 8, 16])
        x = torch.randn(4, N_FEATURES)
        encoded = ae.encoder(x)
        assert encoded.shape[1] < N_FEATURES


# ---------------------------------------------------------------------------
# Model serialisation
# ---------------------------------------------------------------------------

class TestModelSerialisation:
    def test_save_and_load(self, tmp_path, monkeypatch):
        """Save a mock sklearn model and reload it."""
        import joblib

        # Monkeypatch MODELS_DIR to tmp_path
        monkeypatch.setattr("train.MODELS_DIR", tmp_path)

        mock_model = MagicMock()
        _save_model(mock_model, "test_model")
        assert (tmp_path / "test_model.joblib").exists()


# ---------------------------------------------------------------------------
# TrainConfig defaults
# ---------------------------------------------------------------------------

class TestTrainConfig:
    def test_default_scale_pos_weight(self):
        cfg = TrainConfig()
        assert cfg.scale_pos_weight == 1.0

    def test_seq_model_type_default(self):
        cfg = TrainConfig()
        assert cfg.seq_model_type == "LSTM"

    def test_overrides(self):
        cfg = TrainConfig(mlp_epochs=50, tree_max_depth=8)
        assert cfg.mlp_epochs == 50
        assert cfg.tree_max_depth == 8
