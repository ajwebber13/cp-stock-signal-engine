"""
CP Analytics | Stock Signal Engine
model_trainer.py — XGBoost, LightGBM, Random Forest, LSTM ensemble
"""

import numpy as np
import pandas as pd
import joblib
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import lightgbm as lgb
import warnings
warnings.filterwarnings("ignore")

from data_pipeline import FEATURE_COLS


# ── Config ─────────────────────────────────────────────────────────────────

MODEL_DIR   = "models"
N_SPLITS    = 5          # walk-forward CV folds
LSTM_WINDOW = 10         # sequence length for LSTM
os.makedirs(MODEL_DIR, exist_ok=True)


# ── Walk-Forward Validation ─────────────────────────────────────────────────

def walk_forward_cv(model, X, y, label="Model"):
    """Time-series aware CV — never tests on data before training data."""
    tscv   = TimeSeriesSplit(n_splits=N_SPLITS)
    scores = []
    for fold, (train_idx, test_idx) in enumerate(tscv.split(X), 1):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]
        model.fit(X_tr, y_tr)
        prob   = model.predict_proba(X_te)[:, 1]
        auc    = roc_auc_score(y_te, prob)
        scores.append(auc)
        print(f"  {label} Fold {fold}: AUC = {auc:.3f}")
    return np.mean(scores)


# ── Model 1: XGBoost ────────────────────────────────────────────────────────

def train_xgboost(X_train, y_train):
    print("\n[1/4] Training XGBoost...")
    model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=42,
        verbosity=0,
    )
    auc = walk_forward_cv(model, X_train, y_train, "XGBoost")
    model.fit(X_train, y_train)                         # final fit on all training data
    joblib.dump(model, f"{MODEL_DIR}/xgboost.pkl")
    print(f"  XGBoost avg AUC: {auc:.3f}")
    return model, auc


# ── Model 2: LightGBM ───────────────────────────────────────────────────────

def train_lightgbm(X_train, y_train):
    print("\n[2/4] Training LightGBM...")
    model = lgb.LGBMClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbosity=-1,
    )
    auc = walk_forward_cv(model, X_train, y_train, "LightGBM")
    model.fit(X_train, y_train)
    joblib.dump(model, f"{MODEL_DIR}/lightgbm.pkl")
    print(f"  LightGBM avg AUC: {auc:.3f}")
    return model, auc


# ── Model 3: Random Forest ──────────────────────────────────────────────────

def train_random_forest(X_train, y_train):
    print("\n[3/4] Training Random Forest...")
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        min_samples_split=20,       # prevents overfitting on small data
        max_features="sqrt",
        random_state=42,
        n_jobs=-1,
    )
    auc = walk_forward_cv(model, X_train, y_train, "RandomForest")
    model.fit(X_train, y_train)
    joblib.dump(model, f"{MODEL_DIR}/random_forest.pkl")
    print(f"  Random Forest avg AUC: {auc:.3f}")
    return model, auc


# ── Model 4: LSTM ───────────────────────────────────────────────────────────

def build_lstm_sequences(X, y, window=LSTM_WINDOW):
    """Convert flat features into sliding window sequences for LSTM."""
    Xs, ys = [], []
    for i in range(window, len(X)):
        Xs.append(X[i - window:i])
        ys.append(y[i])
    return np.array(Xs), np.array(ys)


def train_lstm(X_train, y_train, scaler):
    print("\n[4/4] Training LSTM...")
    try:
        import tensorflow as tf
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import LSTM, Dense, Dropout
        from tensorflow.keras.callbacks import EarlyStopping

        X_seq, y_seq = build_lstm_sequences(X_train, y_train)
        split    = int(len(X_seq) * 0.8)
        X_tr, X_val = X_seq[:split], X_seq[split:]
        y_tr, y_val = y_seq[:split], y_seq[split:]

        model = Sequential([
            LSTM(64, input_shape=(LSTM_WINDOW, X_train.shape[1]), return_sequences=True),
            Dropout(0.2),
            LSTM(32),
            Dropout(0.2),
            Dense(16, activation="relu"),
            Dense(1, activation="sigmoid"),
        ])
        model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["auc"])

        early_stop = EarlyStopping(patience=5, restore_best_weights=True, verbose=0)
        model.fit(
            X_tr, y_tr,
            validation_data=(X_val, y_val),
            epochs=50,
            batch_size=32,
            callbacks=[early_stop],
            verbose=0,
        )

        val_prob = model.predict(X_val, verbose=0).flatten()
        auc      = roc_auc_score(y_val, val_prob)
        model.save(f"{MODEL_DIR}/lstm.keras")
        print(f"  LSTM validation AUC: {auc:.3f}")
        return model, auc

    except ImportError:
        print("  TensorFlow not installed — skipping LSTM. Run: pip install tensorflow")
        return None, None


# ── Ensemble Scorer ─────────────────────────────────────────────────────────

class EnsembleScorer:
    """
    Weighted average of all trained models.
    Weights derived from each model's validation AUC.
    LSTM gets slightly lower weight because it sees less data (windowed).
    """

    def __init__(self, models: dict, aucs: dict, scaler):
        self.models  = models
        self.weights = self._compute_weights(aucs)
        self.scaler  = scaler
        print(f"\nEnsemble weights: {self.weights}")

    def _compute_weights(self, aucs: dict) -> dict:
        total  = sum(v for v in aucs.values() if v is not None)
        return {k: (v / total if v is not None else 0) for k, v in aucs.items()}

    def predict_proba(self, X_raw: np.ndarray) -> np.ndarray:
        X = self.scaler.transform(X_raw)
        probs = []

        for name, model in self.models.items():
            w = self.weights.get(name, 0)
            if w == 0 or model is None:
                continue

            if name == "lstm":
                try:
                    X_seq = np.array([X[-LSTM_WINDOW:]])           # last N rows as sequence
                    p     = model.predict(X_seq, verbose=0).flatten()
                    probs.append(p * w)
                except Exception:
                    pass
            else:
                p = model.predict_proba(X)[:, 1]
                probs.append(p * w)

        return np.sum(probs, axis=0)

    def score_ticker(self, X_raw: np.ndarray) -> float:
        """Return a single signal probability for the most recent row."""
        prob = self.predict_proba(X_raw)
        return float(prob[-1])


# ── Main Training Loop ──────────────────────────────────────────────────────

def train_all(csv_path: str = "stock_features.csv"):
    print("Loading features...")
    df = pd.read_csv(csv_path, index_col=0)
    df.dropna(subset=FEATURE_COLS + ["target"], inplace=True)

    X_raw = df[FEATURE_COLS].values
    y     = df["target"].values

    # Scale features (all models benefit; LSTM requires it)
    scaler = StandardScaler()
    X      = scaler.fit_transform(X_raw)
    joblib.dump(scaler, f"{MODEL_DIR}/scaler.pkl")

    # Train all four
    xgb_model, xgb_auc = train_xgboost(X, y)
    lgb_model, lgb_auc = train_lightgbm(X, y)
    rf_model,  rf_auc  = train_random_forest(X, y)
    lstm_model, lstm_auc = train_lstm(X, y, scaler)

    models = {
        "xgboost":      xgb_model,
        "lightgbm":     lgb_model,
        "random_forest": rf_model,
        "lstm":          lstm_model,
    }
    aucs = {
        "xgboost":      xgb_auc,
        "lightgbm":     lgb_auc,
        "random_forest": rf_auc,
        "lstm":          lstm_auc,
    }

    ensemble = EnsembleScorer(models, aucs, scaler)
    joblib.dump(ensemble, f"{MODEL_DIR}/ensemble.pkl")

    print("\n✅ All models trained and saved to /models")
    return ensemble


if __name__ == "__main__":
    train_all()
