"""
CP Analytics | Stock Signal Engine
signal_engine.py — Daily scanner, signal scoring, Discord alerts.

Does NOT train. Loads models saved by train_models.py (run weekly).
If no saved models exist, this will tell you to run train_models.py first.
"""

import os
import joblib
import numpy as np
import pandas as pd
import requests
from datetime import datetime
from data_pipeline import fetch_ticker, add_indicators, FEATURE_COLS, WATCHLIST, TARGET_COLS


# ── Config ──────────────────────────────────────────────────────────────────

DISCORD_WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL", "")

MODEL_DIR = "models"
SIGNAL_THRESHOLD = 0.60     # fire alert if model prob >= 60%
MAX_ALERTS       = 5        # cap alerts per target type, per run


# ── Load Saved Models ────────────────────────────────────────────────────────

def load_model(target_name: str):
    """Load the saved XGBoost model + scaler for one target ('day' or 'swing')."""
    model_path  = f"{MODEL_DIR}/xgboost_{target_name}.pkl"
    scaler_path = f"{MODEL_DIR}/scaler_{target_name}.pkl"

    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        raise FileNotFoundError(
            f"No saved model for '{target_name}'. Run train_models.py first."
        )

    model  = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler


def load_all_models() -> dict:
    models = {}
    for target_name in TARGET_COLS:
        models[target_name] = load_model(target_name)
    return models


# ── Signal Scoring ───────────────────────────────────────────────────────────

def score_ticker(symbol: str, models: dict) -> dict | None:
    """Fetch latest data, engineer features, score against both models."""
    df = fetch_ticker(symbol)
    if df.empty or len(df) < 30:
        return None

    df = add_indicators(df)
    df.dropna(subset=FEATURE_COLS, inplace=True)
    if df.empty:
        return None

    X_raw   = df[FEATURE_COLS].values[-1:]   # most recent row only
    latest  = df.iloc[-1]

    result = {
        "symbol":       symbol,
        "price":        round(float(latest["Close"]), 2),
        "rsi":          round(float(latest["rsi_14"]), 1),
        "macd_hist":    round(float(latest["macd_hist"]), 4),
        "vol_ratio":    round(float(latest["vol_ratio"]), 2),
        "price_vs_50d": round(float(latest["price_vs_sma50"]) * 100, 2),
        "return_5d":    round(float(latest["return_5d"]) * 100, 2),
        "above_200d":   bool(latest["Close"] > latest["sma_200"]),
        "date":         str(df.index[-1].date()),
    }

    for target_name, (model, scaler) in models.items():
        X = scaler.transform(X_raw)
        prob = float(model.predict_proba(X)[0, 1])
        result[f"{target_name}_prob"]   = round(prob, 4)
        result[f"{target_name}_signal"] = prob >= SIGNAL_THRESHOLD

    return result


# ── Signal Interpretation ────────────────────────────────────────────────────

def signal_strength(prob: float) -> str:
    if prob >= 0.75: return "🔥 Strong"
    if prob >= 0.65: return "✅ Moderate"
    return "👀 Watch"


def rsi_label(rsi: float) -> str:
    if rsi >= 70: return f"Overbought ({rsi})"
    if rsi <= 30: return f"Oversold ({rsi})"
    return f"Neutral ({rsi})"


# ── Discord Formatting ───────────────────────────────────────────────────────
# Discord markdown: **bold**, `code`, _italic_ (different from Telegram's *bold*)

def format_alert(signal: dict, target_name: str) -> str:
    prob     = signal[f"{target_name}_prob"]
    strength = signal_strength(prob)
    trend    = "📈" if signal["above_200d"] else "📉"
    vol_flag = "⚡ Volume spike" if signal["vol_ratio"] > 1.5 else ""
    label    = "Day Trade" if target_name == "day" else "Swing Trade"

    return f"""
📊 **CP Analytics | {label} Signal**
━━━━━━━━━━━━━━━━━━━━
**${signal['symbol']}**  {trend}  {strength}

**Signal Score:**  `{prob * 100:.1f}%`
**Price:**         `${signal['price']}`
**RSI:**           `{rsi_label(signal['rsi'])}`
**vs 50-day MA:**  `{signal['price_vs_50d']:+.1f}%`
**5-day return:**  `{signal['return_5d']:+.1f}%`
**Volume ratio:**  `{signal['vol_ratio']}x avg` {vol_flag}
**MACD hist:**     `{signal['macd_hist']}`

📅 {signal['date']}
━━━━━━━━━━━━━━━━━━━━
⚠️ _Educational signal only. Not financial advice._
""".strip()


def format_summary(signals: list, target_name: str) -> str:
    label = "Day Trade" if target_name == "day" else "Swing Trade"
    prob_key = f"{target_name}_prob"
    header = (
        f"📊 **CP Analytics | {label} Watchlist**\n"
        f"📅 {datetime.today().strftime('%b %d, %Y')}\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
    )
    ranked = sorted(signals, key=lambda x: x[prob_key], reverse=True)
    rows = []
    for s in ranked[:10]:
        prob = s[prob_key]
        bar = "█" * int(prob * 10) + "░" * (10 - int(prob * 10))
        rows.append(f"**${s['symbol']}**  `{bar}`  `{prob*100:.0f}%`  ${s['price']}")
    footer = "\n━━━━━━━━━━━━━━━━━━━━\n⚠️ _Educational only. Not financial advice._"
    return header + "\n".join(rows) + footer


# ── Discord Dispatch ──────────────────────────────────────────────────────────

def send_discord(message: str):
    if not DISCORD_WEBHOOK_URL:
        print("[Discord] No webhook URL set — printing to console instead.")
        print(message)
        return

    resp = requests.post(DISCORD_WEBHOOK_URL, json={"content": message}, timeout=10)

    if resp.status_code in (200, 204):
        print("[Discord] Sent successfully.")
    else:
        print(f"[Discord] Error {resp.status_code}: {resp.text}")


# ── Main Scanner ─────────────────────────────────────────────────────────────

def run_daily_scan():
    print(f"\n{'='*50}")
    print(f"CP Analytics Stock Scanner — {datetime.today().strftime('%Y-%m-%d %H:%M')}")
    print(f"{'='*50}")

    models = load_all_models()
    results = []

    print("\nScoring watchlist...")
    for sym in WATCHLIST:
        print(f"  {sym}...", end=" ")
        signal = score_ticker(sym, models)
        if signal:
            results.append(signal)
            print(f"day {signal['day_prob']*100:.0f}% / swing {signal['swing_prob']*100:.0f}%")
        else:
            print("skipped")

    all_fired = []

    for target_name in TARGET_COLS:
        signal_key = f"{target_name}_signal"
        fired_list = [s for s in results if s[signal_key]]
        fired_list.sort(key=lambda x: x[f"{target_name}_prob"], reverse=True)

        fired = 0
        for signal in fired_list:
            if fired < MAX_ALERTS:
                send_discord(format_alert(signal, target_name))
                fired += 1

        if fired == 0:
            print(f"  No {target_name} signals above threshold — sending summary.")
            send_discord(format_summary(results, target_name))
        else:
            for s in fired_list[:MAX_ALERTS]:
                s["signal_type"] = target_name
                all_fired.append(s)

    print(f"\n✅ Scan complete. {len(all_fired)} signal(s) sent.")
    return all_fired


if __name__ == "__main__":
    from picks_tracker import run_tracker
    fired_signals = run_daily_scan()
    run_tracker(fired_signals)
