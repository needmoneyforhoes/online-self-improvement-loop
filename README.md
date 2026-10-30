# online-self-improvement-loop

Online learning loop that re-fits Polymarket strategy fire-gates from settled shadow markets. Part of a quant-trading suite for 5-minute crypto up/down markets (BTC/XRP).

It reads the live engine's shadow-fire history, re-fits a per-`(strategy, side)` classifier as outcomes arrive, and raises, lowers, or pauses each strategy's fire threshold. A strategy graduates to LIVE only when its Wilson lower-CI clears the bar with permutation-test significance.

`flip_features` is vendored from the engine via `sys.path.insert(0, ...)` to build feature vectors from tick history. Edit it upstream, not here.

## Contents

| File | What it does |
| --- | --- |
| `self_improve_engine.py` | Online learner. One warm-started `SGDClassifier` (log-loss, L2) per `(strategy, side)`; `partial_fit` on each settle. Adaptive Wilson-CI threshold, auto-pauses below `MIN_WR_FOR_LIVE` (0.55). Exposes `should_fire()` / `update_with_outcome()` for the bot. State in `self_improve_state.pkl`, audit log in `self_improve_log.jsonl`. |
| `self_improve_loop.py` | Batch re-trainer. Every `RETRAIN_EVERY_N` markets, walk-forward trains a 5-model ensemble (LR L1/L2, RF, GB, calibrated SVM) per strategy. Inverts anti-correlated strategies (AUC < 0.45), scans thresholds, gates on Wilson_lo plus 500-shuffle permutation p-value. Writes `deploy_map.json` and `per_strategy_models.pkl` atomically, appends `self_improve_history.jsonl`. |

Both share the same fire-eligibility filter (skips gated, vetoed, and dedup'd fires) and the same Wilson lower-CI math.

## Requirements

Python 3.9+, `numpy`, `scikit-learn`. Read access to the live engine checkout (for `flip_features`) and its `data/` directory. Read-only; no credentials required.

## Usage

```bash
# Warm start: train initial per-strategy models from 600+ historical markets
python3 self_improve_engine.py --warm-start
python3 self_improve_engine.py --report      # per-strategy WR / Wilson_lo / threshold / status

# Batch re-train: retrains only if >= RETRAIN_EVERY_N (default 10) new markets accrued
python3 self_improve_loop.py --once
python3 self_improve_loop.py --force         # force a full retrain
python3 self_improve_loop.py --report        # Wilson_lo trajectory across recent retrains

# Continuous
while true; do python3 self_improve_loop.py --once; sleep 300; done
```

Set `RETRAIN_EVERY_N=<n>` to change retrain cadence.

Bot-side integration:

```python
from self_improve_engine import should_fire, update_with_outcome
res = should_fire(strategy, side, features)                  # {'fire': bool, 'p_win': float, ...}
update_with_outcome(strategy, side, features, won, hypo_pnl)  # after settle
```

## Data

`self_improve_loop.py` runs from `$DATA_DIR`. Both scripts read `market_history.jsonl` (tick traces) and `market_recap_history.jsonl` (logged fires) from there. These come from the private `polymarket-data` repo. Model and state artifacts (`*.pkl`, `deploy_map.json`, `*.jsonl`) are runtime outputs and are git-ignored.

## Feedback

Suggestions, improvements, or something not working? Please
[open an issue](https://github.com/needmoneyforhoes/online-self-improvement-loop/issues/new). Pull requests are welcome too.
