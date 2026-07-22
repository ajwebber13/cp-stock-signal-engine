"""
CP Analytics | Stock Signal Engine
model_trainer.py — XGBoost only. One model per target (day, swing).

This is meant to run occasionally (weekly), NOT on every daily scan.
signal_engine.py loads the saved models instead of retraining.
"""

import numpy as np
import pandas as pd
import joblib
import os
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import warnings
warnings.filterwarnings("ignore")

from data_pipeline import FEATURE_COLS, TARGET_COLS


MODEL_DIR = "models"
N_SPLITS  = 5    # walk-forward CV folds
os.makedirs(MODEL_DIR, exist_ok=True)


def walk_forward_cv(model, X, y, label="Model"):
    """Time-series aware CV — never tests on data before training data."""
    tscv   = TimeSeriesSplit(n_splits=N_SPLITS)
    scores = []
    for fold, (train_idx, test_idx) in enumerate(tscv.split(X), 1):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]
        model.fit(X_tr, y_tr)
        prob = model.predict_proba(X_te)[:, 1]
        auc  = roc_auc_score(y_te, prob)
        scores.append(auc)
        print(f"  {label} Fold {fold}: AUC = {auc:.3f}")
    return np.mean(scores)


def train_xgboost(X_train, y_train, label="XGBoost"):
    model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=42,
        verbosity=0,
    )
    auc = walk_forward_cv(model, X_train, y_train, label)
    model.fit(X_train, y_train)   # final fit on all training data
    print(f"  {label} avg AUC: {auc:.3f}")
    return model, auc


def train_target(csv_path: str, target_name: str):
    """
    Train one XGBoost model for one target ('day' or 'swing').
    Saves models/xgboost_<target_name>.pkl and models/scaler_<target_name>.pkl
    """
    target_col = TARGET_COLS[target_name]
    print(f"\n{'='*50}")
    print(f"Training target: {target_name}  (column: {target_col})")
    print(f"{'='*50}")

    df = pd.read_csv(csv_path, index_col=0)
    df = df.dropna(subset=FEATURE_COLS + [target_col])

    X_raw = df[FEATURE_COLS].values
    y     = df[target_col].values

    print(f"  Rows: {len(df)}   Positive rate: {y.mean():.1%}")

    scaler = StandardScaler()
    X      = scaler.fit_transform(X_raw)

    model, auc = train_xgboost(X, y, label=f"XGBoost-{target_name}")

    joblib.dump(model, f"{MODEL_DIR}/xgboost_{target_name}.pkl")
    joblib.dump(scaler, f"{MODEL_DIR}/scaler_{target_name}.pkl")

    print(f"  Saved: {MODEL_DIR}/xgboost_{target_name}.pkl")
    return model, scaler, auc


def train_all(csv_path: str = "stock_features.csv"):
    """Train both the day model and the swing model."""
    results = {}
    for target_name in TARGET_COLS:
        model, scaler, auc = train_target(csv_path, target_name)
        results[target_name] = {"model": model, "scaler": scaler, "auc": auc}

    print(f"\n✅ Training complete.")
    for name, r in results.items():
        print(f"  {name}: AUC {r['auc']:.3f}")

    return results


if __name__ == "__main__":
    train_all()
