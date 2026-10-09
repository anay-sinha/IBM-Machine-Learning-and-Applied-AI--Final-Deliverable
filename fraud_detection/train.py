"""
train.py
========
Multi-model training orchestrator for fraud detection.

Model zoo
---------
1. Tree Ensembles     – XGBoost, LightGBM, CatBoost (scale_pos_weight / focal loss)
2. Deep Learning      – MLP with residual skip connections + Dropout
                      – LSTM / GRU for chronological transaction sequences
3. Anomaly Detection  – Isolation Forest (unsupervised)
                      – Autoencoder reconstruction-error scoring
4. Meta-Learner       – Stacking classifier with Platt / Isotonic calibration

All models are serialised to `models/` with joblib / torch.save.
Out-of-fold predictions are stored for the stacker's training.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import IsolationForest, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, TensorDataset

# Tree ensemble libraries
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

MODELS_DIR = Path("models")
MODELS_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info("PyTorch device: %s", DEVICE)


# ---------------------------------------------------------------------------
# Training Configuration
# ---------------------------------------------------------------------------

@dataclass
class TrainConfig:
    scale_pos_weight: float = 1.0          # populated from data pipeline
    n_cv_folds: int = 5
    random_state: int = 42

    # Tree ensemble hyper-params
    xgb_n_estimators: int = 500
    lgb_n_estimators: int = 500
    catboost_iterations: int = 500
    tree_learning_rate: float = 0.05
    tree_max_depth: int = 6
    tree_subsample: float = 0.8

    # MLP
    mlp_hidden_dims: List[int] = field(default_factory=lambda: [256, 128, 64])
    mlp_dropout: float = 0.3
    mlp_epochs: int = 30
    mlp_batch_size: int = 2048
    mlp_lr: float = 1e-3

    # LSTM / GRU
    seq_model_type: str = "LSTM"          # "LSTM" or "GRU"
    seq_hidden_size: int = 128
    seq_num_layers: int = 2
    seq_dropout: float = 0.3
    seq_window: int = 10                  # number of past transactions as context
    seq_epochs: int = 20
    seq_batch_size: int = 512
    seq_lr: float = 1e-3

    # Isolation Forest
    iso_n_estimators: int = 200
    iso_contamination: float = 0.002      # approximate fraud rate

    # Autoencoder
    ae_hidden_dims: List[int] = field(default_factory=lambda: [64, 32, 16, 32, 64])
    ae_epochs: int = 30
    ae_batch_size: int = 2048
    ae_lr: float = 1e-3
    ae_reconstruction_threshold_quantile: float = 0.99


# ---------------------------------------------------------------------------
# 1. Tree Ensemble Models
# ---------------------------------------------------------------------------

def train_xgboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    cfg: TrainConfig,
) -> xgb.XGBClassifier:
    """XGBoost with early stopping and focal-loss via scale_pos_weight."""
    model = xgb.XGBClassifier(
        n_estimators=cfg.xgb_n_estimators,
        learning_rate=cfg.tree_learning_rate,
        max_depth=cfg.tree_max_depth,
        subsample=cfg.tree_subsample,
        colsample_bytree=0.8,
        scale_pos_weight=cfg.scale_pos_weight,
        eval_metric="aucpr",
        early_stopping_rounds=30,
        use_label_encoder=False,
        random_state=cfg.random_state,
        n_jobs=-1,
        tree_method="hist",
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=50,
    )
    _save_model(model, "xgboost")
    logger.info(
        "XGBoost val PR-AUC: %.4f",
        average_precision_score(y_val, model.predict_proba(X_val)[:, 1]),
    )
    return model


def train_lightgbm(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    cfg: TrainConfig,
) -> lgb.LGBMClassifier:
    """LightGBM with focal loss via is_unbalance / scale_pos_weight."""
    model = lgb.LGBMClassifier(
        n_estimators=cfg.lgb_n_estimators,
        learning_rate=cfg.tree_learning_rate,
        max_depth=cfg.tree_max_depth,
        subsample=cfg.tree_subsample,
        colsample_bytree=0.8,
        scale_pos_weight=cfg.scale_pos_weight,
        objective="binary",
        metric="average_precision",
        random_state=cfg.random_state,
        n_jobs=-1,
    )
    callbacks = [lgb.early_stopping(30, verbose=False), lgb.log_evaluation(50)]
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=callbacks,
    )
    _save_model(model, "lightgbm")
    logger.info(
        "LightGBM val PR-AUC: %.4f",
        average_precision_score(y_val, model.predict_proba(X_val)[:, 1]),
    )
    return model


def train_catboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    cfg: TrainConfig,
) -> CatBoostClassifier:
    """CatBoost with built-in class imbalance support."""
    model = CatBoostClassifier(
        iterations=cfg.catboost_iterations,
        learning_rate=cfg.tree_learning_rate,
        depth=cfg.tree_max_depth,
        loss_function="Logloss",
        eval_metric="AveragePrecision",
        scale_pos_weight=cfg.scale_pos_weight,
        early_stopping_rounds=30,
        random_seed=cfg.random_state,
        verbose=50,
    )
    model.fit(
        X_train, y_train,
        eval_set=(X_val, y_val),
        use_best_model=True,
    )
    _save_model(model, "catboost")
    logger.info(
        "CatBoost val PR-AUC: %.4f",
        average_precision_score(y_val, model.predict_proba(X_val)[:, 1]),
    )
    return model


# ---------------------------------------------------------------------------
# 2a. MLP with Residual Skip Connections
# ---------------------------------------------------------------------------

class ResidualBlock(nn.Module):
    """Dense residual block: Linear → BN → ReLU → Dropout → Linear + skip."""

    def __init__(self, dim: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
        )
        self.act = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.net(x))


class FraudMLP(nn.Module):
    """
    Multi-Layer Perceptron with residual connections.

    Architecture
    ------------
    Input → [Linear → ResidualBlock] × n_layers → Linear(1) → Sigmoid
    """

    def __init__(self, input_dim: int, hidden_dims: List[int], dropout: float) -> None:
        super().__init__()
        layers: List[nn.Module] = []

        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            if prev_dim == h_dim:
                layers.append(ResidualBlock(h_dim, dropout))
            prev_dim = h_dim

        layers.append(nn.Linear(prev_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.net(x)).squeeze(-1)


def train_mlp(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    cfg: TrainConfig,
) -> FraudMLP:
    """Train the residual MLP with weighted BCE loss."""
    input_dim = X_train.shape[1]
    model = FraudMLP(input_dim, cfg.mlp_hidden_dims, cfg.mlp_dropout).to(DEVICE)

    pos_weight = torch.tensor([cfg.scale_pos_weight], dtype=torch.float32, device=DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.mlp_lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", patience=5, factor=0.5
    )

    X_tr_t = torch.tensor(X_train.values, dtype=torch.float32, device=DEVICE)
    y_tr_t = torch.tensor(y_train.values, dtype=torch.float32, device=DEVICE)
    X_val_t = torch.tensor(X_val.values, dtype=torch.float32, device=DEVICE)
    y_val_t = torch.tensor(y_val.values, dtype=torch.float32, device=DEVICE)

    train_ds = TensorDataset(X_tr_t, y_tr_t)
    train_loader = DataLoader(train_ds, batch_size=cfg.mlp_batch_size, shuffle=True)

    best_val_ap = 0.0
    best_state = None

    for epoch in range(1, cfg.mlp_epochs + 1):
        model.train()
        epoch_loss = 0.0
        for xb, yb in train_loader:
            optimizer.zero_grad()
            logits = model.net(xb).squeeze(-1)
            loss = criterion(logits, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()

        model.eval()
        with torch.no_grad():
            val_probs = torch.sigmoid(model.net(X_val_t).squeeze(-1)).cpu().numpy()
        val_ap = average_precision_score(y_val.values, val_probs)
        scheduler.step(val_ap)

        if val_ap > best_val_ap:
            best_val_ap = val_ap
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if epoch % 5 == 0:
            logger.info("MLP epoch %d/%d | loss %.4f | val PR-AUC %.4f",
                        epoch, cfg.mlp_epochs, epoch_loss / len(train_loader), val_ap)

    model.load_state_dict(best_state)
    torch.save(model.state_dict(), MODELS_DIR / "mlp.pt")
    logger.info("MLP best val PR-AUC: %.4f", best_val_ap)
    return model


# ---------------------------------------------------------------------------
# 2b. Temporal Sequence Model (LSTM / GRU)
# ---------------------------------------------------------------------------

class FraudSequenceModel(nn.Module):
    """
    LSTM or GRU encoder over a fixed-length window of past transactions.

    Input  : (batch, seq_len, input_dim)
    Output : scalar fraud probability per sequence
    """

    def __init__(
        self,
        input_dim: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        model_type: str = "LSTM",
    ) -> None:
        super().__init__()
        rnn_cls = nn.LSTM if model_type == "LSTM" else nn.GRU
        self.rnn = rnn_cls(
            input_dim, hidden_size, num_layers=num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output, _ = self.rnn(x)
        # Use last time-step hidden state
        last_hidden = output[:, -1, :]
        return torch.sigmoid(self.head(last_hidden)).squeeze(-1)


def build_sequence_dataset(
    X: pd.DataFrame, y: pd.Series, window: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert tabular data into overlapping fixed-length sequences.

    Each sample is a (window, features) tensor; label is the last row's label.
    Assumes X is already sorted chronologically.
    """
    X_arr = X.values
    y_arr = y.values
    n = len(X_arr)

    sequences, labels = [], []
    for i in range(window, n):
        sequences.append(X_arr[i - window:i])
        labels.append(y_arr[i])

    return np.array(sequences, dtype=np.float32), np.array(labels, dtype=np.float32)


def train_sequence_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    cfg: TrainConfig,
) -> FraudSequenceModel:
    """Train an LSTM or GRU fraud detector on sequential transaction windows."""
    X_seq_tr, y_seq_tr = build_sequence_dataset(X_train, y_train, cfg.seq_window)
    X_seq_val, y_seq_val = build_sequence_dataset(X_val, y_val, cfg.seq_window)

    input_dim = X_seq_tr.shape[2]
    model = FraudSequenceModel(
        input_dim, cfg.seq_hidden_size, cfg.seq_num_layers,
        cfg.seq_dropout, cfg.seq_model_type,
    ).to(DEVICE)

    pos_weight = torch.tensor([cfg.scale_pos_weight], dtype=torch.float32, device=DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.seq_lr)

    tr_ds = TensorDataset(
        torch.from_numpy(X_seq_tr).to(DEVICE),
        torch.from_numpy(y_seq_tr).to(DEVICE),
    )
    tr_loader = DataLoader(tr_ds, batch_size=cfg.seq_batch_size, shuffle=False)  # keep order

    best_ap, best_state = 0.0, None
    X_val_t = torch.from_numpy(X_seq_val).to(DEVICE)
    y_val_np = y_seq_val

    for epoch in range(1, cfg.seq_epochs + 1):
        model.train()
        total_loss = 0.0
        for xb, yb in tr_loader:
            optimizer.zero_grad()
            # BCEWithLogitsLoss expects raw logits
            logits = model.rnn(xb)[0][:, -1, :]
            logits = model.head(logits).squeeze(-1)
            loss = criterion(logits, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()

        model.eval()
        with torch.no_grad():
            val_probs = model(X_val_t).cpu().numpy()
        val_ap = average_precision_score(y_val_np, val_probs)

        if val_ap > best_ap:
            best_ap = val_ap
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if epoch % 5 == 0:
            logger.info(
                "%s epoch %d/%d | loss %.4f | val PR-AUC %.4f",
                cfg.seq_model_type, epoch, cfg.seq_epochs,
                total_loss / len(tr_loader), val_ap,
            )

    model.load_state_dict(best_state)
    torch.save(model.state_dict(), MODELS_DIR / f"{cfg.seq_model_type.lower()}.pt")
    logger.info("%s best val PR-AUC: %.4f", cfg.seq_model_type, best_ap)
    return model


# ---------------------------------------------------------------------------
# 3a. Isolation Forest
# ---------------------------------------------------------------------------

def train_isolation_forest(
    X_train: pd.DataFrame,
    cfg: TrainConfig,
) -> IsolationForest:
    """Unsupervised anomaly detector. Trained on FULL data (both classes)."""
    model = IsolationForest(
        n_estimators=cfg.iso_n_estimators,
        contamination=cfg.iso_contamination,
        random_state=cfg.random_state,
        n_jobs=-1,
    )
    model.fit(X_train)
    _save_model(model, "isolation_forest")
    logger.info("Isolation Forest trained.")
    return model


# ---------------------------------------------------------------------------
# 3b. Autoencoder Reconstruction-Error Scoring
# ---------------------------------------------------------------------------

class FraudAutoencoder(nn.Module):
    """
    Symmetric autoencoder trained on LEGITIMATE transactions only.
    High reconstruction error → anomaly (potential fraud).
    """

    def __init__(self, input_dim: int, hidden_dims: List[int]) -> None:
        super().__init__()
        enc_layers: List[nn.Module] = []
        prev = input_dim
        for h in hidden_dims[: len(hidden_dims) // 2 + 1]:
            enc_layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h

        dec_layers: List[nn.Module] = []
        for h in hidden_dims[len(hidden_dims) // 2 + 1:]:
            dec_layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        dec_layers.append(nn.Linear(prev, input_dim))

        self.encoder = nn.Sequential(*enc_layers)
        self.decoder = nn.Sequential(*dec_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))


def train_autoencoder(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    cfg: TrainConfig,
) -> Tuple[FraudAutoencoder, float]:
    """
    Train autoencoder on legitimate (class=0) transactions only.
    Returns model and reconstruction-error threshold for anomaly detection.
    """
    # Train ONLY on legitimate samples
    X_legit = X_train[y_train == 0].values.astype(np.float32)
    input_dim = X_legit.shape[1]

    model = FraudAutoencoder(input_dim, cfg.ae_hidden_dims).to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.ae_lr)

    ds = TensorDataset(torch.from_numpy(X_legit))
    loader = DataLoader(ds, batch_size=cfg.ae_batch_size, shuffle=True)

    for epoch in range(1, cfg.ae_epochs + 1):
        model.train()
        total_loss = 0.0
        for (xb,) in loader:
            xb = xb.to(DEVICE)
            optimizer.zero_grad()
            recon = model(xb)
            loss = criterion(recon, xb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if epoch % 10 == 0:
            logger.info("Autoencoder epoch %d/%d | MSE %.6f",
                        epoch, cfg.ae_epochs, total_loss / len(loader))

    # Compute per-sample reconstruction error on validation set
    model.eval()
    X_val_t = torch.from_numpy(X_val.values.astype(np.float32)).to(DEVICE)
    with torch.no_grad():
        recon_val = model(X_val_t).cpu().numpy()
    recon_errors = np.mean((X_val.values - recon_val) ** 2, axis=1)

    # Threshold at 99th percentile of legitimate reconstruction errors
    legit_errors = recon_errors[y_val == 0]
    threshold = float(np.quantile(legit_errors, cfg.ae_reconstruction_threshold_quantile))

    torch.save(model.state_dict(), MODELS_DIR / "autoencoder.pt")
    np.save(MODELS_DIR / "ae_threshold.npy", np.array([threshold]))

    val_ap = average_precision_score(y_val, recon_errors)
    logger.info("Autoencoder val PR-AUC: %.4f | threshold: %.6f", val_ap, threshold)
    return model, threshold


# ---------------------------------------------------------------------------
# 4. Stacking Ensemble with Calibration
# ---------------------------------------------------------------------------

def train_stacking_ensemble(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    base_models: Dict[str, Any],
    cfg: TrainConfig,
    calibration_method: str = "sigmoid",  # "sigmoid" (Platt) or "isotonic"
) -> StackingClassifier:
    """
    Build a calibrated stacking meta-learner on top of base model probabilities.

    The stacker uses the base-model sklearn-compatible estimators directly;
    deep learning models are wrapped in a thin sklearn-compatible shim.
    """
    estimators = [
        (name, model)
        for name, model in base_models.items()
        if hasattr(model, "predict_proba")  # sklearn-compatible only
    ]

    logger.info("Stacking with base estimators: %s", [e[0] for e in estimators])

    meta_learner = LogisticRegression(
        C=0.1,
        class_weight="balanced",
        max_iter=1000,
        random_state=cfg.random_state,
    )

    stacker = StackingClassifier(
        estimators=estimators,
        final_estimator=meta_learner,
        stack_method="predict_proba",
        cv=cfg.n_cv_folds,
        n_jobs=-1,
        passthrough=False,
    )
    stacker.fit(X_train, y_train)

    # Calibrate the stacker's output probabilities
    calibrated = CalibratedClassifierCV(
        stacker, method=calibration_method, cv="prefit"
    )
    calibrated.fit(X_val, y_val)

    val_ap = average_precision_score(
        y_val, calibrated.predict_proba(X_val)[:, 1]
    )
    _save_model(calibrated, "stacking_ensemble")
    logger.info(
        "Stacking ensemble val PR-AUC: %.4f (calibration: %s)",
        val_ap, calibration_method,
    )
    return calibrated


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _save_model(model: Any, name: str) -> None:
    path = MODELS_DIR / f"{name}.joblib"
    joblib.dump(model, path)
    logger.info("Model saved: %s", path)


def load_model(name: str) -> Any:
    path = MODELS_DIR / f"{name}.joblib"
    if not path.exists():
        raise FileNotFoundError(f"Model not found: {path}")
    return joblib.load(path)


def load_splits(processed_dir: str = "data/processed") -> Tuple:
    """Load preprocessed train/val/test splits from CSV files."""
    d = Path(processed_dir)
    X_train = pd.read_csv(d / "X_train.csv")
    X_val   = pd.read_csv(d / "X_val.csv")
    X_test  = pd.read_csv(d / "X_test.csv")
    y_train = pd.read_csv(d / "y_train.csv").squeeze()
    y_val   = pd.read_csv(d / "y_val.csv").squeeze()
    y_test  = pd.read_csv(d / "y_test.csv").squeeze()
    return X_train, X_val, X_test, y_train, y_val, y_test


# ---------------------------------------------------------------------------
# Main Training Orchestrator
# ---------------------------------------------------------------------------

def run_training(
    cfg: Optional[TrainConfig] = None,
    processed_dir: str = "data/processed",
) -> Dict[str, Any]:
    """
    Full training run. Loads processed splits, trains all models, returns
    a dict mapping model names to fitted objects.
    """
    if cfg is None:
        cfg = TrainConfig()

    X_train, X_val, X_test, y_train, y_val, y_test = load_splits(processed_dir)
    cfg.scale_pos_weight = float((y_train == 0).sum() / max((y_train == 1).sum(), 1))

    logger.info("Training started. scale_pos_weight=%.2f", cfg.scale_pos_weight)

    trained_models: Dict[str, Any] = {}

    # -- Tree ensembles --
    trained_models["xgboost"]   = train_xgboost(X_train, y_train, X_val, y_val, cfg)
    trained_models["lightgbm"]  = train_lightgbm(X_train, y_train, X_val, y_val, cfg)
    trained_models["catboost"]  = train_catboost(X_train, y_train, X_val, y_val, cfg)

    # -- MLP --
    trained_models["mlp"] = train_mlp(X_train, y_train, X_val, y_val, cfg)

    # -- Sequence model --
    trained_models["seq_model"] = train_sequence_model(X_train, y_train, X_val, y_val, cfg)

    # -- Anomaly detectors --
    trained_models["isolation_forest"] = train_isolation_forest(X_train, cfg)
    ae_model, ae_threshold = train_autoencoder(X_train, y_train, X_val, y_val, cfg)
    trained_models["autoencoder"] = ae_model
    trained_models["ae_threshold"] = ae_threshold

    # -- Stacking ensemble (tree models only, sklearn-compatible) --
    sklearn_models = {
        k: v for k, v in trained_models.items()
        if k in ("xgboost", "lightgbm", "catboost")
    }
    trained_models["stacking"] = train_stacking_ensemble(
        X_train, y_train, X_val, y_val, sklearn_models, cfg
    )

    # Persist config
    cfg_path = MODELS_DIR / "train_config.json"
    with open(cfg_path, "w") as f:
        json.dump(asdict(cfg), f, indent=2)

    logger.info("All models trained and saved to %s", MODELS_DIR)
    return trained_models


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train all fraud detection models")
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--strategy", default="smote_tomek",
                        choices=["smote_tomek", "adasyn", "cost_weight", "none"])
    args = parser.parse_args()

    train_cfg = TrainConfig(mlp_epochs=args.epochs, seq_epochs=args.epochs)
    run_training(cfg=train_cfg, processed_dir=args.processed_dir)
