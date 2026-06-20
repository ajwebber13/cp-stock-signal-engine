"""
CP Analytics | Stock Signal Engine
data_pipeline.py — Price data ingestion and feature engineering
Uses the 'ta' library instead of pandas_ta for better compatibility.
"""

import yfinance as yf
import pandas as pd
import numpy as np
import ta
from datetime import datetime, timedelta


WATCHLIST = [
    "AAPL", "MSFT", "NVDA", "TSLA", "AMZN",
    "META", "GOOGL", "AMD", "SOFI", "PLTR",
]

LOOKBACK_DAYS = 365
FORWARD_DAYS  = 5
TARGET_MOVE   = 0.03


def fetch_ticker(symbol: str) -> pd.DataFrame:
    end   = datetime.today()
    start = end - timedelta(days=LOOKBACK_DAYS)
    df = yf.download(symbol, start=start, end=end, progress=False, auto_adjust=True)
    if df.empty:
        return pd.DataFrame()
    df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    df["Symbol"] = symbol
    return df


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    close = df["Close"]
    high  = df["High"]
    low   = df["Low"]
    vol   = df["Volume"]

    # Momentum
    df["rsi_14"]      = ta.momentum.RSIIndicator(close, window=14).rsi()
    macd              = ta.trend.MACD(close)
    df["macd"]        = macd.macd()
    df["macd_signal"] = macd.macd_signal()
    df["macd_hist"]   = macd.macd_diff()

    # Moving averages
    df["sma_20"]  = ta.trend.SMAIndicator(close, window=20).sma_indicator()
    df["sma_50"]  = ta.trend.SMAIndicator(close, window=50).sma_indicator()
    df["sma_200"] = ta.trend.SMAIndicator(close, window=200).sma_indicator()
    df["ema_12"]  = ta.trend.EMAIndicator(close, window=12).ema_indicator()
    df["ema_26"]  = ta.trend.EMAIndicator(close, window=26).ema_indicator()

    # Price relative to MAs
    df["price_vs_sma20"]  = (close - df["sma_20"])  / df["sma_20"]
    df["price_vs_sma50"]  = (close - df["sma_50"])  / df["sma_50"]
    df["price_vs_sma200"] = (close - df["sma_200"]) / df["sma_200"]

    # Bollinger Bands
    bb = ta.volatility.BollingerBands(close, window=20)
    df["bb_upper"]    = bb.bollinger_hband()
    df["bb_lower"]    = bb.bollinger_lband()
    df["bb_width"]    = (df["bb_upper"] - df["bb_lower"]) / close
    df["bb_position"] = (close - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"])

    # ATR
    df["atr_14"] = ta.volatility.AverageTrueRange(high, low, close, window=14).average_true_range()

    # Volume
    df["vol_sma20"] = vol.rolling(20).mean()
    df["vol_ratio"] = vol / df["vol_sma20"]

    # Candlestick features
    df["body_size"]  = abs(close - df["Open"]) / df["Open"]
    df["upper_wick"] = (high - df[["Close","Open"]].max(axis=1)) / df["Open"]
    df["lower_wick"] = (df[["Close","Open"]].min(axis=1) - low) / df["Open"]
    df["is_green"]   = (close > df["Open"]).astype(int)

    # Momentum returns
    for n in [3, 5, 10, 20]:
        df[f"return_{n}d"] = close.pct_change(n)

    # Day of week
    df["day_of_week"] = pd.to_datetime(df.index).dayofweek

    return df


def add_labels(df: pd.DataFrame) -> pd.DataFrame:
    future_max = df["Close"].shift(-1).rolling(FORWARD_DAYS).max().shift(-(FORWARD_DAYS - 1))
    df["target"] = ((future_max - df["Close"]) / df["Close"] >= TARGET_MOVE).astype(int)
    return df


def build_dataset(symbols: list = WATCHLIST) -> pd.DataFrame:
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
