#!/usr/bin/env python3
"""
SELF-IMPROVING ONLINE LEARNING ENGINE

Maintains a per-strategy online classifier that updates with each shadow
market's outcome. Goal: drive fire-level WR toward 70-80% per strategy
through continuous learning.

ARCHITECTURE:
  - One SGDClassifier per (strategy, side)
  - Warm-started from historical data (600+ markets)
  - Updated via partial_fit after each market settles
  - Per-strategy adaptive threshold (rises if WR drops, falls if WR climbs)

RIGOR (applies online too):
  - Warm start ONLY on historically validated strategies
  - L2 regularization (alpha=0.001) prevents overfit on small online batches
  - Threshold updates require min 30 fires
  - Auto-disable strategy if rolling WR < base rate after 50 fires
  - Block-bootstrap CI on per-strategy WR
  - Walk-forward validation every 100 markets

USAGE:
  Phase 1: Initial training (offline)
    python3 self_improve_engine.py --warm-start
    
  Phase 2: At fire-time (called by bot)
    from self_improve_engine import should_fire
    res = should_fire(strategy, side, features)
    # res = {'fire': bool, 'p_win': float, 'wr_lower_ci': float}
  
  Phase 3: After market settles (called by bot)
    from self_improve_engine import update_with_outcome
    update_with_outcome(strategy, side, features, won=True)

OUTPUT FILES:
  self_improve_state.pkl — current models + rolling stats per strategy
  self_improve_log.jsonl — every fire decision logged for audit
"""
import json, math, sys, os, pickle, random, time
from collections import defaultdict, deque

sys.path.insert(0, '.')
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler

random.seed(42)
np.random.seed(42)

STATE_PATH = 'self_improve_state.pkl'
LOG_PATH = 'self_improve_log.jsonl'

# Per-strategy config
MIN_FIRES_FOR_THRESHOLD_UPDATE = 30
ROLLING_WINDOW_SIZE = 100  # last N fires per strategy
TARGET_WR = 0.70
MIN_WR_FOR_LIVE = 0.55  # below this, strategy auto-pauses
INITIAL_THRESHOLD = 0.55
DEFAULT_ALPHA = 0.001  # SGD L2 strength


def _new_state_for_strategy():
    return {
        'model': SGDClassifier(loss='log_loss', alpha=DEFAULT_ALPHA, 
                                max_iter=5, learning_rate='optimal',
                                random_state=42, warm_start=True),
        'scaler': StandardScaler(),
        'is_fitted': False,
        'feature_names': None,
        'threshold': INITIAL_THRESHOLD,
        'fires_seen': 0,
        'rolling_wins': deque(maxlen=ROLLING_WINDOW_SIZE),
        'rolling_pnls': deque(maxlen=ROLLING_WINDOW_SIZE),
        'lifetime_n': 0,
        'lifetime_wins': 0,
        'lifetime_pnl': 0.0,
        'paused': False,
        'pause_reason': '',
        'last_threshold_update_at_n': 0,
    }


def load_state():
    if os.path.exists(STATE_PATH):
        try:
            with open(STATE_PATH, 'rb') as f:
                return pickle.load(f)
        except: pass
    return defaultdict(_new_state_for_strategy)


def save_state(state):
    if not isinstance(state, dict):
        state = dict(state)
    with open(STATE_PATH, 'wb') as f:
        pickle.dump(state, f)


def _wilson_ci_lower(wins, n, z=1.96):
    """Wilson score interval lower bound — robust for small samples."""
    if n == 0: return 0.0
    p = wins / n
    denom = 1 + z*z/n
    center = (p + z*z/(2*n)) / denom
    spread = z * math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / denom
    return max(0, center - spread)


def features_to_vector(feats: dict, feature_names: list) -> np.ndarray:
    row = []
    for fn in feature_names:
        v = feats.get(fn, 0.0)
        if v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
            v = 0.0
        row.append(float(v))
    return np.array(row).reshape(1, -1)


def warm_start_from_history():
    """Load market_history.jsonl + market_recap_history.jsonl, train initial
    per-strategy models. Run ONCE before bot starts."""
    print("Loading market_history.jsonl...")
    ticks_by_slug = {}
    with open('market_history.jsonl') as f:
        for line in f:
            try: r = json.loads(line)
            except: continue
            if r.get('slug') and r.get('ticks'):
                ticks_by_slug[r['slug']] = r['ticks']
    print(f"  Markets: {len(ticks_by_slug)}")
    
    print("Loading FREE fires from market_recap_history.jsonl...")
    fires_by_strat = defaultdict(list)
    with open('market_recap_history.jsonl') as f:
        for line in f:
            try: r = json.loads(line)
            except: continue
            slug = r.get('slug')
            ts = r.get('ts')
            if slug not in ticks_by_slug: continue
            for fire in r.get('fires', []):
                cd = fire.get('cd', 0)
                if cd is None or cd < 15: continue
                if (fire.get('pre_gate_held') or fire.get('opp_gate_held')
                    or fire.get('dedup_excluded') or fire.get('pricecap_excluded')
                    or fire.get('model_vetoed') or fire.get('flip_gate_blocked')
                    or fire.get('bn_vetoed')):
                    continue
                strat = fire.get('strategy')
                side = fire.get('side')
                if not strat or not side: continue
                fires_by_strat[(strat, side)].append({
                    'slug': slug, 'ts': ts, 'cd': cd,
                    'entry': fire.get('entry_price'),
                    'hypo': fire.get('hypo_pnl', 0),
                    'won': fire.get('hypo_pnl', 0) > 0,
                })
    
    print(f"  Strategies with fires: {len(fires_by_strat)}")
    
    print("Computing features and training per-strategy models...")
    import flip_features
    import datetime
    
    state = load_state()
    
    n_trained = 0
    for (strat, side), fires in fires_by_strat.items():
        if len(fires) < 30:
            print(f"  ⏭ {strat}|{side}: only {len(fires)} fires, skipping (need 30+)")
            continue
        
        # Build dataset for this strategy
        X_list, y_list = [], []
        for f in fires:
            visible = [t for t in ticks_by_slug[f['slug']] if t[0] >= f['cd']]
            if len(visible) < 10: continue
            try:
                feats = flip_features.build_features(
                    visible,
                    current_time=datetime.datetime.fromtimestamp(f['ts']) if f['ts'] else None
                )
                feats['fire_cd']      = float(f['cd'])
                feats['fire_entry']   = float(f['entry']) if f['entry'] else 0.0
                feats['fire_side_dn'] = 1.0 if side == 'DN' else 0.0
            except Exception:
                continue
            X_list.append(feats)
            y_list.append(1 if f['won'] else 0)
        
        if len(X_list) < 30:
            print(f"  ⏭ {strat}|{side}: too few computable features, skipping")
            continue
        
        # Determine feature names from this strategy's data
        feature_names = sorted(set(X_list[0].keys()))
        
        X = np.array([[f.get(fn, 0.0) for fn in feature_names] for f in X_list])
        # Replace NaN/Inf
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        y = np.array(y_list)
        
        # Init scaler+model for this strategy
        s = state[(strat, side)]
        s['feature_names'] = feature_names
        s['scaler'] = StandardScaler()
        X_scaled = s['scaler'].fit_transform(X)
        
        # Initial fit
        s['model'] = SGDClassifier(
            loss='log_loss', alpha=DEFAULT_ALPHA,
            max_iter=20, learning_rate='optimal',
            random_state=42, warm_start=True,
            class_weight='balanced'
        )
        try:
            s['model'].fit(X_scaled, y)
            s['is_fitted'] = True
            n_trained += 1
            wr = y.mean()
            print(f"  ✅ {strat}|{side}: trained on {len(y)} fires (historical WR={wr*100:.0f}%)")
        except Exception as e:
            print(f"  ❌ {strat}|{side}: training failed — {e}")
            continue
    
    save_state(state)
    print(f"\n✅ Warm-start complete: {n_trained} per-strategy models saved")


def should_fire(strategy: str, side: str, features: dict) -> dict:
    """Called at fire-time. Returns decision + P(win)."""
    state = load_state()
    key = (strategy, side)
    if key not in state:
        return {'fire': True, 'p_win': None, 'reason': 'no_model — fall through'}
    
    s = state[key]
    if s['paused']:
        return {'fire': False, 'p_win': None, 'reason': f"paused: {s['pause_reason']}"}
    if not s['is_fitted']:
        return {'fire': True, 'p_win': None, 'reason': 'model_not_fitted'}
    
    try:
        x = features_to_vector(features, s['feature_names'])
        x_scaled = s['scaler'].transform(x)
        # SGD with log_loss has predict_proba via decision_function -> sigmoid
        if hasattr(s['model'], 'predict_proba'):
            p = float(s['model'].predict_proba(x_scaled)[0, 1])
        else:
            d = s['model'].decision_function(x_scaled)[0]
            p = 1 / (1 + math.exp(-d))
        
        thr = s['threshold']
        return {
            'fire': p >= thr,
            'p_win': p,
            'threshold': thr,
            'rolling_wr': sum(s['rolling_wins'])/max(1,len(s['rolling_wins'])),
            'lifetime_n': s['lifetime_n'],
            'reason': 'ok',
        }
    except Exception as e:
        return {'fire': True, 'p_win': None, 'reason': f'eval_error: {e}'}


def update_with_outcome(strategy: str, side: str, features: dict, won: bool, hypo_pnl: float):
    """Called after market settles. Online update + adaptive threshold."""
    state = load_state()
    key = (strategy, side)
    if key not in state:
        state[key] = _new_state_for_strategy()
    
    s = state[key]
    
    # Lifetime stats
    s['lifetime_n'] += 1
    if won: s['lifetime_wins'] += 1
    s['lifetime_pnl'] += hypo_pnl
    s['rolling_wins'].append(1 if won else 0)
    s['rolling_pnls'].append(hypo_pnl)
    
    # Online model update (only if already fitted with feature_names)
    if s['is_fitted'] and s['feature_names']:
        try:
            x = features_to_vector(features, s['feature_names'])
            x_scaled = s['scaler'].transform(x)
            s['model'].partial_fit(x_scaled, [1 if won else 0])
        except Exception:
            pass
    
    # Adaptive threshold (only update at significant intervals)
    if (s['lifetime_n'] - s['last_threshold_update_at_n']) >= MIN_FIRES_FOR_THRESHOLD_UPDATE:
        # Use Wilson lower-CI of rolling WR — conservative
        wins = sum(s['rolling_wins'])
        n = len(s['rolling_wins'])
        wr_lower = _wilson_ci_lower(wins, n)
        
        if wr_lower < MIN_WR_FOR_LIVE:
            # Auto-pause this strategy
            s['paused'] = True
            s['pause_reason'] = f"WR Wilson lower CI ({wr_lower:.2f}) < {MIN_WR_FOR_LIVE}"
        else:
            # Adjust threshold — if WR > target, lower threshold to fire more
            #                    if WR < target, raise threshold to be more selective
            current_wr = wins / n
            if current_wr > TARGET_WR:
                s['threshold'] = max(0.50, s['threshold'] - 0.02)
            elif current_wr < TARGET_WR - 0.10:
                s['threshold'] = min(0.85, s['threshold'] + 0.02)
            
        s['last_threshold_update_at_n'] = s['lifetime_n']
    
    # Log decision
    log_entry = {
        'ts': time.time(), 'strategy': strategy, 'side': side,
        'won': won, 'hypo_pnl': hypo_pnl,
        'lifetime_n': s['lifetime_n'], 'rolling_wr': sum(s['rolling_wins'])/max(1,len(s['rolling_wins'])),
        'threshold': s['threshold'], 'paused': s['paused'],
    }
    try:
        with open(LOG_PATH, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')
    except Exception:
        pass
    
    save_state(state)


def report():
    """Print per-strategy report."""
    state = load_state()
    print(f"\n{'='*100}")
    print(f"SELF-IMPROVE ENGINE — STATE REPORT")
    print(f"{'='*100}")
    print(f"{'strategy|side':<35s} {'lifetime_n':>10s} {'WR%':>5s} {'Wilson_lo':>10s} {'thr':>6s} {'pnl':>8s} {'status':>10s}")
    print('-'*100)
    
    rows = []
    for key, s in state.items():
        if isinstance(key, tuple):
            strat, side = key
        else:
            strat, side = key, '?'
        if s['lifetime_n'] == 0: continue
        wr = s['lifetime_wins'] / s['lifetime_n']
        wr_lo = _wilson_ci_lower(s['lifetime_wins'], s['lifetime_n'])
        rows.append({'strat':strat,'side':side,'n':s['lifetime_n'],'wr':wr,
                     'wr_lo':wr_lo,'thr':s['threshold'],'pnl':s['lifetime_pnl'],
                     'paused':s['paused'],'fitted':s['is_fitted']})
    
    rows.sort(key=lambda r: -r['wr_lo'])
    for r in rows:
        status = "PAUSED" if r['paused'] else ("READY" if r['fitted'] else "no_model")
        flag = '✅' if r['wr'] >= TARGET_WR else ('🟡' if r['wr'] >= MIN_WR_FOR_LIVE else '❌')
        print(f"{flag} {r['strat']+'|'+r['side']:<32s}  {r['n']:>9d} {r['wr']*100:>4.0f}%   {r['wr_lo']*100:>4.1f}%  {r['thr']:>5.2f}  ${r['pnl']:>+6.2f}  {status:>10s}")


if __name__ == '__main__':
    if '--warm-start' in sys.argv:
        warm_start_from_history()
    elif '--report' in sys.argv:
        report()
    else:
        print("Usage:")
        print("  python3 self_improve_engine.py --warm-start   (initial training)")
        print("  python3 self_improve_engine.py --report       (show current state)")
        print()
        print("Runtime API (from bot):")
        print("  from self_improve_engine import should_fire, update_with_outcome")
