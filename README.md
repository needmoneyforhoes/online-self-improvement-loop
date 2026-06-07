# online-self-improvement-loop

Online SGD self-improvement loop that incrementally re-fits Polymarket strategy gates between runs.

Part of a Polymarket quant-trading suite for **5-minute crypto up/down markets** (BTC/XRP). This repo is the *learning brain*: it consumes the shadow-fire history that the live engine logs and continuously re-fits a per-strategy classifier, raising or lowering each strategy's fire threshold (and pausing dead ones) as fresh market outcomes arrive.

## Why it exists
Strategy edges decay. Rather than re-tuning gates by hand, this loop turns every settled shadow market into a training example, drives per-`(strategy, side)` win-rate toward a 70% target via adaptive thresholds, and only graduates a strategy to LIVE once its Wilson lower-CI clears the bar with permutation-test significance — keeping the live allowlist honest without manual intervention.

## What's inside

| File | Role |
| --- | --- |
| `self_improve_engine.py` | **Online incremental learner.** One warm-started `SGDClassifier` (log-loss, L2) per `(strategy, side)`; `partial_fit` after each settle. Adaptive per-strategy threshold (Wilson-CI driven), auto-pause below `MIN_WR_FOR_LIVE`. Exposes the runtime API `should_fire()` / `update_with_outcome()` the bot calls. State in `self_improve_state.pkl`, audit log in `self_improve_log.jsonl`. |
| `self_improve_loop.py` | **Periodic re-trainer.** Every N new markets, walk-forward trains a 5-model ensemble (LR L1/L2, RF, GB, calibrated SVM) per strategy, auto-inverts anti-correlated strategies (AUC < 0.45), scans thresholds, gates on Wilson_lo + 500-shuffle permutation p-value. Writes `deploy_map.json` + `per_strategy_models.pkl` (atomic) and a `self_improve_history.jsonl` learning curve. |

Both share the same fire-eligibility filter (skips gated/vetoed/dedup'd fires) and the same Wilson-lower-CI math, so the online engine and the batch loop agree on what counts as a clean fire.

### Vendored modules
- `flip_features` — imported via `sys.path.insert(0, '/home/polybot/polymarket-bot')` to build the feature vector from tick history. **Vendored from the engine — edit upstream, not here.**

## Requirements
- Python 3.9+
- `numpy`, `scikit-learn`
- Read access to the live engine checkout at `/home/polybot/polymarket-bot` (for `flip_features`) and its `data/` directory (see **Data**).
- No wallet, key, or RPC access — this repo never touches funds; it learns offline from logged history only.

## Usage

```bash
# One-time warm start: train initial per-strategy models from 600+ historical markets
python3 self_improve_engine.py --warm-start
python3 self_improve_engine.py --report      # per-strategy WR / Wilson_lo / threshold / status

# Periodic re-train: retrains only if >= RETRAIN_EVERY_N (default 10) new markets accrued
python3 self_improve_loop.py --once
python3 self_improve_loop.py --force         # force a full retrain now
python3 self_improve_loop.py --report        # Wilson_lo trajectory across recent retrains

# Continuous (cron or while-loop)
while true; do python3 self_improve_loop.py --once; sleep 300; done
```

Set `RETRAIN_EVERY_N=<n>` to change retrain cadence.

The bot integrates the online engine directly:

```python
from self_improve_engine import should_fire, update_with_outcome
res = should_fire(strategy, side, features)        # {'fire': bool, 'p_win': float, ...}
update_with_outcome(strategy, side, features, won, hypo_pnl)  # after settle
```

## Data
`self_improve_loop.py` runs from `/home/polybot/polymarket-bot/data` and both scripts read `market_history.jsonl` (tick traces) and `market_recap_history.jsonl` (logged fires) from there. These come from the **private `polymarket-data` repo** — point a `DATA_DIR`/working path at that checkout. Model/state artifacts (`*.pkl`, `deploy_map.json`, `*.jsonl`) are runtime outputs and are git-ignored.

> Private research software. No warranty; trades/handles real funds at your own risk.
