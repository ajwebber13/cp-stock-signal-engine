"""
CP Analytics | Stock Signal Engine
signal_engine.py — Daily scanner, signal scoring, Telegram alerts
Designed for GitHub Actions — auto-trains on each run, no persisted model needed.
"""

import os
import numpy as np
import pandas as pd
import requests
from datetime import datetime
from data_pipeline import fetch_ticker, add_indicators, FEATURE_COLS, WATCHLIST, build_dataset
from model_trainer import train_all, EnsembleScorer


# ── Config ──────────────────────────────────────────────────────────────────

TELEGRAM_TOKEN  = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT   = os.getenv("TELEGRAM_CHAT_ID", "")

SIGNAL_THRESHOLD = 0.60     # fire alert if ensemble prob >= 60%
MAX_ALERTS       = 5        # cap daily alerts to top N picks


# ── Build & Train (runs fresh each time) ────────────────────────────────────

def build_and_train() -> EnsembleScorer:
    """Fetch data, engineer features, train ensemble. ~3-5 min on Actions."""
    print("Building dataset...")
    df = build_dataset(WATCHLIST)
    df.to_csv("stock_features.csv")
    print(f"Dataset ready: {len(df)} rows, {df['target'].mean():.1%} positive labels")
    return train_all("stock_features.csv")


# ── Signal Scoring ───────────────────────────────────────────────────────────

def score_ticker(symbol: str, ensemble: EnsembleScorer) -> dict | None:
    """Fetch latest data, engineer features, return signal score for one ticker."""
    df = fetch_ticker(symbol)
    if df.empty or len(df) < 30:
        return None

    df = add_indicators(df)
    df.dropna(subset=FEATURE_COLS, inplace=True)
    if df.empty:
        return None

    X_raw = df[FEATURE_COLS].values
    prob  = ensemble.score_ticker(X_raw)

    latest = df.iloc[-1]

    return {
        "symbol":       symbol,
        "prob":         round(prob, 4),
        "price":        round(float(latest["Close"]), 2),
        "rsi":          round(float(latest["rsi_14"]), 1),
        "macd_hist":    round(float(latest["macd_hist"]), 4),
        "vol_ratio":    round(float(latest["vol_ratio"]), 2),
        "price_vs_50d": round(float(latest["price_vs_sma50"]) * 100, 2),
        "return_5d":    round(float(latest["return_5d"]) * 100, 2),
        "above_200d":   bool(latest["Close"] > latest["sma_200"]),
        "signal":       prob >= SIGNAL_THRESHOLD,
        "date":         str(df.index[-1].date()),
    }


# ── Signal Interpretation ────────────────────────────────────────────────────

def signal_strength(prob: float) -> str:
    if prob >= 0.75: return "🔥 Strong"
    if prob >= 0.65: return "✅ Moderate"
    return "👀 Watch"


def rsi_label(rsi: float) -> str:
    if rsi >= 70: return f"Overbought ({rsi})"
    if rsi <= 30: return f"Oversold ({rsi})"
    return f"Neutral ({rsi})"


# ── Telegram Formatting ──────────────────────────────────────────────────────

def format_alert(signal: dict) -> str:
    strength = signal_strength(signal["prob"])
    trend    = "📈" if signal["above_200d"] else "📉"
    vol_flag = "⚡ Volume spike" if signal["vol_ratio"] > 1.5 else ""

    return f"""
📊 *CP Analytics | Stock Signal*
━━━━━━━━━━━━━━━━━━━━
*${signal['symbol']}*  {trend}  {strength}

*Signal Score:*  `{signal['prob'] * 100:.1f}%`
*Price:*         `${signal['price']}`
*RSI:*           `{rsi_label(signal['rsi'])}`
*vs 50-day MA:*  `{signal['price_vs_50d']:+.1f}%`
*5-day return:*  `{signal['return_5d']:+.1f}%`
*Volume ratio:*  `{signal['vol_ratio']}x avg` {vol_flag}
*MACD hist:*     `{signal['macd_hist']}`

📅 {signal['date']}
━━━━━━━━━━━━━━━━━━━━
⚠️ _Educational signal only. Not financial advice._
""".strip()


def format_summary(signals: list) -> str:
    header = f"📊 *CP Analytics | Daily Watchlist*\n📅 {datetime.today().strftime('%b %d, %Y')}\n━━━━━━━━━━━━━━━━━━━━\n"
    rows = []
    for s in signals[:10]:
        bar = "█" * int(s["prob"] * 10) + "░" * (10 - int(s["prob"] * 10))
        rows.append(f"*${s['symbol']}*  `{bar}`  `{s['prob']*100:.0f}%`  ${s['price']}")
    footer = "\n━━━━━━━━━━━━━━━━━━━━\n⚠️ _Educational only. Not financial advice._"
    return header + "\n".join(rows) + footer


# ── Telegram Dispatch ────────────────────────────────────────────────────────

def send_telegram(message: str):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT:
        print("[Telegram] No credentials — printing to console instead.")
        print(message)
        return

    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    resp = requests.post(url, json={
        "chat_id":    TELEGRAM_CHAT,
        "text":       message,
        "parse_mode": "Markdown",
    }, timeout=10)

    if resp.status_code == 200:
        print("[Telegram] Sent successfully.")
    else:
        print(f"[Telegram] Error {resp.status_code}: {resp.text}")


# ── Main Scanner ─────────────────────────────────────────────────────────────

def run_daily_scan():
    print(f"\n{'='*50}")
    print(f"CP Analytics Stock Scanner — {datetime.today().strftime('%Y-%m-%d %H:%M')}")
    print(f"{'='*50}")

    # Train fresh on each run (required for GitHub Actions stateless env)
    ensemble = build_and_train()
    results  = []

    print("\nScoring watchlist...")
    for sym in WATCHLIST:
        print(f"  {sym}...", end=" ")
        signal = score_ticker(sym, ensemble)
        if signal:
            results.append(signal)
            print(f"{signal['prob']*100:.1f}% {'← SIGNAL' if signal['signal'] else ''}")
        else:
            print("skipped")

    results.sort(key=lambda x: x["prob"], reverse=True)

    fired = 0
    for signal in results:
        if signal["signal"] and fired < MAX_ALERTS:
            send_telegram(format_alert(signal))
            fired += 1

    if fired == 0:
        print("  No signals above threshold — sending summary.")
        send_telegram(format_summary(results))

    print(f"\n✅ Scan complete. {fired} signal(s) sent.")
    fired_signals = [s for s in results if s["signal"]]
    return fired_signals


if __name__ == "__main__":
    from picks_tracker import run_tracker
    fired_signals = run_daily_scan()
    run_tracker(fired_signals)
