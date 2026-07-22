"""
CP Analytics | Stock Signal Engine
picks_tracker.py — Log signals and track real-world outcomes.
Tracks day and swing picks SEPARATELY so you know which one actually wins.
"""

import os
import csv
import requests
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta


PICKS_FILE = "picks_log.csv"

# Must match data_pipeline.py
FORWARD_DAYS = {"day": 1, "swing": 5}
TARGET_MOVE  = {"day": 0.015, "swing": 0.03}

DISCORD_WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL", "")

FIELDNAMES = [
    "date", "symbol", "signal_type", "prob", "price_at_signal",
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
    """
    Call this for every ticker that fires a signal.
    signal must include 'signal_type' ('day' or 'swing') and the matching
    '<type>_prob' key, set by signal_engine.py.
    """
    init_log()
    signal_type = signal["signal_type"]
    forward_days = FORWARD_DAYS[signal_type]

    signal_date = datetime.strptime(signal["date"], "%Y-%m-%d")
    target_date = signal_date + timedelta(days=forward_days)

    row = {
        "date":             signal["date"],
        "symbol":           signal["symbol"],
        "signal_type":      signal_type,
        "prob":             signal[f"{signal_type}_prob"],
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

    print(f"  [Tracker] Logged {signal['symbol']} ({signal_type}) @ ${signal['price']} "
          f"(prob: {row['prob']*100:.1f}%)")


# ── Resolve Pending Picks ────────────────────────────────────────────────────

def resolve_picks():
    """Check outcomes for any picks whose target date has passed."""
    df = load_log()
    if df.empty:
        print("[Tracker] No picks to resolve.")
        return df

    today   = datetime.today().date()
    pending = df[df["resolved"] == False]
    newly_resolved = []

    for idx, row in pending.iterrows():
        target_date = datetime.strptime(row["target_date"], "%Y-%m-%d").date()
        if today < target_date:
            continue

        start = target_date
        end   = target_date + timedelta(days=5)
        hist  = yf.download(row["symbol"], start=start, end=end,
                             progress=False, auto_adjust=True)

        if hist.empty:
            continue

        target_move      = TARGET_MOVE[row["signal_type"]]
        price_at_target  = float(hist["Close"].iloc[0])
        pct_change       = (price_at_target - float(row["price_at_signal"])) / float(row["price_at_signal"])
        hit_target       = pct_change >= target_move

        df.at[idx, "price_at_target"] = round(price_at_target, 2)
        df.at[idx, "pct_change"]      = round(pct_change * 100, 2)
        df.at[idx, "hit_target"]      = hit_target
        df.at[idx, "resolved"]        = True
        newly_resolved.append(f"{row['symbol']} ({row['signal_type']})")

    save_log(df)

    if newly_resolved:
        print(f"[Tracker] Resolved: {', '.join(newly_resolved)}")

    return df


# ── Performance Report ───────────────────────────────────────────────────────

def performance_report(df: pd.DataFrame) -> str:
    resolved = df[df["resolved"] == True]

    if resolved.empty:
        return "📊 **CP Analytics | Picks Tracker**\n\nNo resolved picks yet."

    sections = ["📊 **CP Analytics | Picks Performance**", "━━━━━━━━━━━━━━━━━━━━"]

    for signal_type in ["day", "swing"]:
        subset = resolved[resolved["signal_type"] == signal_type]
        label = "Day Trade" if signal_type == "day" else "Swing Trade"

        if subset.empty:
            sections.append(f"**{label}:** no resolved picks yet.")
            continue

        total    = len(subset)
        wins     = subset["hit_target"].sum()
        losses   = total - wins
        win_rate = wins / total * 100
        avg_move = subset["pct_change"].mean()

        sections.append(
            f"**{label}**\n"
            f"Total: `{total}`   Win rate: `{win_rate:.1f}%`  ({int(wins)}W / {int(losses)}L)\n"
            f"Avg move: `{avg_move:+.1f}%`"
        )

    recent = resolved.tail(10)
    rows = []
    for _, r in recent.iterrows():
        result = "✅" if r["hit_target"] else "❌"
        tag = "D" if r["signal_type"] == "day" else "S"
        rows.append(f"{result} **${r['symbol']}** [{tag}]  `{r['pct_change']:+.1f}%`  {r['date']}")

    sections.append("**Recent picks:**\n" + "\n".join(rows))
    sections.append("━━━━━━━━━━━━━━━━━━━━\n⚠️ _Educational only. Not financial advice._")

    return "\n\n".join(sections)


# ── Discord ──────────────────────────────────────────────────────────────────

def send_discord(message: str):
    if not DISCORD_WEBHOOK_URL:
        print(message)
        return
    requests.post(DISCORD_WEBHOOK_URL, json={"content": message}, timeout=10)


# ── Main ─────────────────────────────────────────────────────────────────────

def run_tracker(signals: list = None):
    """
    Call after run_daily_scan().
    signals = list of dicts, each with 'signal_type' set ('day' or 'swing').
    """
    print("\n[Tracker] Running picks tracker...")

    if signals:
        for s in signals:
            log_signal(s)

    df = resolve_picks()

    resolved = df[df["resolved"] == True] if not df.empty else pd.DataFrame()
    if not resolved.empty:
        report = performance_report(df)
        send_discord(report)
        print("[Tracker] Performance report sent.")
    else:
        print("[Tracker] No resolved picks yet.")


if __name__ == "__main__":
    run_tracker()
