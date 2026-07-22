"""
CP Analytics | Stock Signal Engine
train_models.py — Run this WEEKLY (not daily).

Rebuilds the feature dataset and retrains the day + swing XGBoost models.
signal_engine.py just loads whatever this last saved — it never trains itself.
"""

from data_pipeline import build_dataset, WATCHLIST
from model_trainer import train_all


def main():
    print("Building dataset...")
    df = build_dataset(WATCHLIST)
    df.to_csv("stock_features.csv")
    print(f"Dataset ready: {len(df)} rows")
    print(f"  Day target positive rate:   {df['target_day'].mean():.1%}")
    print(f"  Swing target positive rate: {df['target_swing'].mean():.1%}")

    train_all("stock_features.csv")


if __name__ == "__main__":
    main()
