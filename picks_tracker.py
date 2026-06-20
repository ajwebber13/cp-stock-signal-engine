"""
CP Analytics | Stock Signal Engine
picks_tracker.py — Log signals and track real-world outcomes
Stores picks in a CSV, checks results after 5 days, reports accuracy.
"""

import os
import csv
import json
import requests
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta


PICKS_FILE      = "picks_log.csv"
FORWARD_DAYS    = 5
TARGET_MOVE     = 0.03       # 3% gain = win
TELEGRAM_TOKEN  = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT   = os.getenv("TELEGRAM_CHAT_ID", "")

FIELDNAMES = [
    "date", "symbol", "prob", "price_at_signal",
    "target_date", "price_at_target", "pct_change",
    "hit_target", "resolved"
]


# ── File Setup ───────────────────────────────────────────────────────────────

def init_log():
    if not os.path.exists(PICKS_FILE):
        with open(PICKS_FILE, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
            writer.writeheader()


def load_log() -> pd.DataFrame:
    init_log()
    df = pd.read_csv(PICKS_FILE)
    if df.empty:
        return pd.DataFrame(columns=FIELDNAMES)
    return df


def save_log(df: pd.DataFrame):
    df.to_csv(PICKS_FILE, index=False)


# ── Log a New Signal ─────────────────────────────────────────────────────────

def log_signal(signal: dict):
    """Call this for every ticker that fires a signal."""
    init_log()
    signal_date  = datetime.strptime(signal["date"], "%Y-%m-%d")
    target_date  = signal_date + timedelta(days=FORWARD_DAYS)

    row = {
        "date":             signal["date"],
        "symbol":           signal["symbol"],
        "prob":             signal["prob"],
        "price_at_signal":  signal["price"],
        "target_date":      target_date.strftime("%Y-%m-%d"),
        "price_at_target":  "",
        "pct_change":       "",
        "hit_target":       "",
        "resolved":         False,
    }

    with open(PICKS_FILE, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writerow(row)

    print(f"  [Tracker] Logged {signal['symbol']} @ ${signal['price']} (prob: {signal['prob']*100:.1f}%)")


# ── Resolve Pending Picks ────────────────────────────────────────────────────

def resolve_picks():
    """Check outcomes for any picks whose target date has passed."""
    df = load_log()
    if df.empty:
        print("[Tracker] No picks to resolve.")
        return df

    today       = datetime.today().date()
    pending     = df[df["resolved"] == False]
    newly_resolved = []

    for idx, row in pending.iterrows():
        target_date = datetime.strptime(row["target_date"], "%Y-%m-%d").date()
        if today < target_date:
            continue

        # Fetch price on or after target date
        start = target_date
        end   = target_date + timedelta(days=5)
        hist  = yf.download(row["symbol"], start=start, end=end,
                            progress=False, auto_adjust=True)

        if hist.empty:
            continue

        price_at_target = float(hist["Close"].iloc[0])
        pct_change      = (price_at_target - float(row["price_at_signal"])) / float(row["price_at_signal"])
        hit_target      = pct_change >= TARGET_MOVE

        df.at[idx, "price_at_target"] = round(price_at_target, 2)
        df.at[idx, "pct_change"]      = round(pct_change * 100, 2)
        df.at[idx, "hit_target"]      = hit_target
        df.at[idx, "resolved"]        = True
        newly_resolved.append(row["symbol"])

    save_log(df)

    if newly_resolved:
        print(f"[Tracker] Resolved: {', '.join(newly_resolved)}")

    return df


# ── Performance Report ───────────────────────────────────────────────────────

def performance_report(df: pd.DataFrame) -> str:
    resolved = df[df["resolved"] == True]

    if resolved.empty:
        return "📊 *CP Analytics | Picks Tracker*\n\nNo resolved picks yet — check back after 5 trading days."

    total    = len(resolved)
    wins     = resolved["hit_target"].sum()
    losses   = total - wins
    win_rate = wins / total * 100
    avg_move = resolved["pct_change"].mean()
    best     = resolved.loc[resolved["pct_change"].idxmax()]
    worst    = resolved.loc[resolved["pct_change"].idxmin()]

    # Last 10 picks
    recent = resolved.tail(10)
    rows   = []
    for _, r in recent.iterrows():
        result = "✅" if r["hit_target"] else "❌"
        rows.append(f"{result} *${r['symbol']}*  `{r['pct_change']:+.1f}%`  {r['date']}")

    report = f"""
📊 *CP Analytics | Picks Performance*
━━━━━━━━━━━━━━━━━━━━
*Total picks:*   `{total}`
*Win rate:*      `{win_rate:.1f}%`  ({int(wins)}W / {int(losses)}L)
*Avg move:*      `{avg_move:+.1f}%`
*Best pick:*     `${best['symbol']} {best['pct_change']:+.1f}%`
*Worst pick:*    `${worst['symbol']} {worst['pct_change']:+.1f}%`

*Recent picks:*
{chr(10).join(rows)}
━━━━━━━━━━━━━━━━━━━━
⚠️ _Educational only. Not financial advice._
""".strip()

    return report


# ── Telegram ─────────────────────────────────────────────────────────────────

def send_telegram(message: str):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT:
        print(message)
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    requests.post(url, json={
        "chat_id":    TELEGRAM_CHAT,
        "text":       message,
        "parse_mode": "Markdown",
    }, timeout=10)


# ── Main ─────────────────────────────────────────────────────────────────────

def run_tracker(signals: list = None):
    """
    Call after run_daily_scan().
    signals = list of dicts from score_ticker that fired above threshold.
    """
    print("\n[Tracker] Running picks tracker...")

    # Log any new signals
    if signals:
        for s in signals:
            log_signal(s)

    # Resolve pending picks
    df = resolve_picks()

    # Send performance report if we have resolved picks
    resolved = df[df["resolved"] == True] if not df.empty else pd.DataFrame()
    if not resolved.empty:
        report = performance_report(df)
        send_telegram(report)
        print("[Tracker] Performance report sent.")
    else:
        print("[Tracker] No resolved picks yet.")


if __name__ == "__main__":
    run_tracker()
