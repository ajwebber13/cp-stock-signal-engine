# CP Analytics | Stock Signal Engine

XGBoost stock signal models built for the Culture & Pulse Picks brand.
Two separate models: one for day trades, one for swing trades.

---

## Models

| Model | Target | Window |
|---|---|---|
| `xgboost_day` | 1.5%+ move | next 1 day |
| `xgboost_swing` | 3%+ move | next 5 days |

Both are plain XGBoost — no ensemble. Add more models later only once
you have tracked results proving one alone isn't enough.

---

## Pipeline

```
data_pipeline.py   → fetch prices, build 20 features, build BOTH targets
model_trainer.py   → train xgboost_day and xgboost_swing, save to /models
train_models.py    → entry point — run WEEKLY to rebuild data + retrain
signal_engine.py   → entry point — run DAILY, loads saved models, scores, alerts
picks_tracker.py   → logs every signal, checks outcomes, reports win rate per type
```

**Key change from the original version:** `signal_engine.py` no longer trains
on every run. It loads whatever `train_models.py` last saved. This keeps the
model stable day to day instead of quietly changing its mind each morning.

---

## Setup

```bash
pip install -r requirements.txt
```

---

## Environment Variables

```bash
DISCORD_WEBHOOK_URL=your_discord_webhook_url
```

(Set this as a repo secret named `DISCORD_WEBHOOK_URL` in GitHub Actions.)

---

## Run Order

```bash
# Step 1 — build training data + train both models (run weekly)
python train_models.py

# Step 2 — run daily scan (loads saved models, no training)
python signal_engine.py
```

---

## GitHub Actions

Two separate workflows:

- **`.github/workflows/retrain_models.yml`** — Saturdays, rebuilds data and
  retrains both models, commits `/models` back to the repo.
- **`.github/workflows/daily_scan.yml`** — weekdays 9am ET, loads committed
  models, scores the watchlist, sends Discord alerts, commits `picks_log.csv`.

Both `models/*.pkl` and `picks_log.csv` are committed to the repo (not
gitignored) — GitHub Actions runs are stateless, so without committing these
files, your model and your track record would reset every run.

---

## Key Design Decisions

- **Walk-forward CV only** — no random splits. Prevents data leakage on time-series data.
- **XGBoost only, for now** — LightGBM, Random Forest, and LSTM were cut. Add them
  back later only if you have tracked results showing a single model isn't enough.
  Running near-duplicate models (like XGBoost + LightGBM) doesn't add day/swing/options
  coverage — that comes from training on different *targets*, which is what the
  day/swing split above does.
- **Train weekly, score daily** — training is separated from scoring so the model
  doesn't change its mind every morning.
- **Signal threshold: 60%** — conservative. Adjust in `signal_engine.py` as you track performance.
- **Day and swing tracked separately** in `picks_tracker.py` — so you can see which
  one is actually working, instead of one blended number hiding the truth.
- **Options is not a model.** Once the swing signal proves itself, `options/` becomes
  a rules layer on top of it (strike/expiration selection using the signal + IV) —
  not a fifth ML model.

---

## Disclaimer

All signals are for educational purposes only. Not financial advice.
