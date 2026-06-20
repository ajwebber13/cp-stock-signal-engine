# CP Analytics | Stock Signal Engine

Ensemble stock signal model built for the Culture & Pulse Picks brand.
Mirrors the sports predictor architecture — probability score → Telegram alert.

---

## Models

| Model | Strength | Role |
|---|---|---|
| XGBoost | Fast, interpretable, great on tabular data | Primary signal |
| LightGBM | Faster on large data, strong on feature interactions | Supporting signal |
| Random Forest | Low overfitting, stable baseline | Sanity check |
| LSTM | Learns time-series patterns across 10-day windows | Sequence layer |

Final output = **weighted ensemble** where each model's weight = its validation AUC.

---

## Pipeline

```
data_pipeline.py     → fetch prices, build 20 features
model_trainer.py     → train all 4 models, save to /models
signal_engine.py     → daily scan, score tickers, fire Telegram alerts
```

---

## Setup

```bash
pip install -r requirements.txt

# For LSTM support:
pip install tensorflow
```

---

## Environment Variables

```bash
TELEGRAM_BOT_TOKEN=your_bot_token
TELEGRAM_CHAT_ID=your_chat_id
```

---

## Run Order

```bash
# Step 1 — build training data (run once, refresh monthly)
python data_pipeline.py

# Step 2 — train all models
python model_trainer.py

# Step 3 — run daily scan (schedule with cron or Render cron job)
python signal_engine.py
```

---

## Render Deployment

Add to your existing Render project alongside the sports predictor.

`render.yaml` cron job:
```yaml
- type: cron
  name: cp-stock-scanner
  schedule: "0 14 * * 1-5"    # 9am ET, weekdays only
  buildCommand: pip install -r requirements.txt
  startCommand: python signal_engine.py
```

---

## Key Design Decisions

- **Walk-forward CV only** — no random splits. Prevents data leakage on time-series data.
- **20 features max** — keeps the model lean and reduces overfitting risk.
- **Ensemble weighted by AUC** — better models get more vote weight automatically.
- **Signal threshold: 60%** — conservative. Adjust in `signal_engine.py` as you track performance.
- **Daily summary fallback** — if nothing clears threshold, sends ranked watchlist instead of silence.

---

## Disclaimer

All signals are for educational purposes only. Not financial advice.
