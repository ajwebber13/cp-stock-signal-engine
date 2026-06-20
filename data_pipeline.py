"""
CP Analytics | Stock Signal Engine
data_pipeline.py — Price data ingestion and feature engineering
"""

import yfinance as yf
import pandas as pd
import ta as ta_lib
import numpy as np
from datetime import datetime, timedelta


WATCHLIST = [
    "AAPL", "MSFT", "NVDA", "TSLA", "AMZN",
    "META", "GOOGL", "AMD", "SOFI", "PLTR",
]

LOOKBACK_DAYS = 365
FORWARD_DAYS  = 5       # how many days ahead we're predicting
TARGET_MOVE   = 0.03    # 3% gain = positive label


def fetch_ticker(symbol: str) -> pd.DataFrame:
    """Download OHLCV data for a single ticker."""
    end   = datetime.today()
    start = end - timedelta(days=LOOKBACK_DAYS)
    df = yf.download(symbol, start=start, end=end, progress=False, auto_adjust=True)
    if df.empty:
        return pd.DataFrame()
    df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    df["Symbol"] = symbol
    return df


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add technical indicators as features."""
    # Momentum
    df["rsi_14"]      = ta.rsi(df["Close"], length=14)
    macd              = ta.macd(df["Close"])
    df["macd"]        = macd["MACD_12_26_9"]
    df["macd_signal"] = macd["MACDs_12_26_9"]
    df["macd_hist"]   = macd["MACDh_12_26_9"]

    # Moving averages
    df["sma_20"]      = ta.sma(df["Close"], length=20)
    df["sma_50"]      = ta.sma(df["Close"], length=50)
    df["sma_200"]     = ta.sma(df["Close"], length=200)
    df["ema_12"]      = ta.ema(df["Close"], length=12)
    df["ema_26"]      = ta.ema(df["Close"], length=26)

    # Price relative to MAs (normalized distance)
    df["price_vs_sma20"]  = (df["Close"] - df["sma_20"])  / df["sma_20"]
    df["price_vs_sma50"]  = (df["Close"] - df["sma_50"])  / df["sma_50"]
    df["price_vs_sma200"] = (df["Close"] - df["sma_200"]) / df["sma_200"]

    # Volatility
    bb                = ta.bbands(df["Close"], length=20)
    df["bb_upper"]    = bb["BBU_20_2.0"]
    df["bb_lower"]    = bb["BBL_20_2.0"]
    df["bb_width"]    = (df["bb_upper"] - df["bb_lower"]) / df["Close"]
    df["bb_position"] = (df["Close"] - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"])
    df["atr_14"]      = ta.atr(df["High"], df["Low"], df["Close"], length=14)

    # Volume
    df["vol_sma20"]   = df["Volume"].rolling(20).mean()
    df["vol_ratio"]   = df["Volume"] / df["vol_sma20"]

    # Candlestick body/wick features
    df["body_size"]   = abs(df["Close"] - df["Open"]) / df["Open"]
    df["upper_wick"]  = (df["High"] - df[["Close","Open"]].max(axis=1)) / df["Open"]
    df["lower_wick"]  = (df[["Close","Open"]].min(axis=1) - df["Low"]) / df["Open"]
    df["is_green"]    = (df["Close"] > df["Open"]).astype(int)

    # Momentum over multiple windows
    for n in [3, 5, 10, 20]:
        df[f"return_{n}d"] = df["Close"].pct_change(n)

    # Day of week (0=Mon, 4=Fri) — markets have weekday patterns
    df["day_of_week"] = pd.to_datetime(df.index).dayofweek

    return df


def add_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Binary label: did price rise >= TARGET_MOVE within FORWARD_DAYS?
    Only used for training — not available for live signals.
    """
    future_max = df["Close"].shift(-1).rolling(FORWARD_DAYS).max().shift(-(FORWARD_DAYS - 1))
    df["target"] = ((future_max - df["Close"]) / df["Close"] >= TARGET_MOVE).astype(int)
    return df


def build_dataset(symbols: list = WATCHLIST) -> pd.DataFrame:
    """Fetch, engineer, and label data for all tickers."""
    frames = []
    for sym in symbols:
        print(f"  Fetching {sym}...")
        df = fetch_ticker(sym)
        if df.empty:
            continue
        df = add_indicators(df)
        df = add_labels(df)
        frames.append(df)

    combined = pd.concat(frames)
    combined.dropna(inplace=True)
    return combined


FEATURE_COLS = [
    "rsi_14", "macd", "macd_signal", "macd_hist",
    "price_vs_sma20", "price_vs_sma50", "price_vs_sma200",
    "bb_width", "bb_position", "atr_14",
    "vol_ratio",
    "body_size", "upper_wick", "lower_wick", "is_green",
    "return_3d", "return_5d", "return_10d", "return_20d",
    "day_of_week",
]


if __name__ == "__main__":
    print("Building dataset...")
    df = build_dataset()
    df.to_csv("stock_features.csv")
    print(f"Done. {len(df)} rows, {df['target'].mean():.1%} positive labels.")
