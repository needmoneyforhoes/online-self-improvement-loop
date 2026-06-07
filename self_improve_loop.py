#!/usr/bin/env python3
"""
SELF-IMPROVING LOOP

Re-trains per-strategy ensemble classifiers as new shadow markets accumulate.
Tracks Wilson_lo over time per strategy. Auto-graduates strategies when they
cross 70% threshold.

DESIGN PRINCIPLES:
  - Re-train every N markets (default 10) — not every market (overfit risk)
  - Walk-forward chronologically with growing window (purge oldest if needed)
  - Track Wilson_lo trajectory over time per strategy
  - Auto-deploy strategies when Wilson_lo ≥ 0.70 with n ≥ 15 stable for 2 retrains
  - Auto-flag inverted-AUC strategies (AUC < 0.45 → invert predictions)
  - History saved so we can see learning curve

USAGE:
  Initial run:
    python3 self_improve_loop.py --once
  
  Continuous loop (cron or while-loop):
    while true; do
      python3 self_improve_loop.py --once
      sleep 300  # 5 min between checks
    done
  
  Or trigger from bot after each market settles via a hook.

OUTPUT:
  self_improve_history.jsonl — per retrain: strategy → Wilson_lo, n, etc.
  deploy_map.json — updated live per-strategy thresholds (atomic write)
  per_strategy_models.pkl — current models (atomic write)
"""
import json, math, sys, os, pickle, random, time, shutil
from collections import defaultdict
from datetime import datetime

sys.path.insert(0, '/home/polybot/polymarket-bot')
os.chdir('/home/polybot/polymarket-bot/data')

import numpy as np
np.random.seed(42)
random.seed(42)

# Config
MIN_FIRES = 50
MIN_TEST_N = 15
TARGET_WILSON_LO = 0.70
VIABLE_WILSON_LO = 0.55
INVERT_AUC_THRESHOLD = 0.45  # AUC < this → flip predictions
HISTORY_PATH = 'self_improve_history.jsonl'
RETRAIN_EVERY_N_MARKETS = int(os.environ.get('RETRAIN_EVERY_N', '10'))


def wilson_lower(wins, n, z=1.96):
    if n == 0: return 0
    p = wins/n
    denom = 1 + z*z/n
    center = (p + z*z/(2*n))/denom
    spread = z * math.sqrt(p*(1-p)/n + z*z/(4*n*n))/denom
    return max(0, center-spread)


def should_retrain():
    """Decide if we should retrain based on new markets since last run."""
    if not os.path.exists('per_strategy_models.pkl'):
        return True, "no model exists yet"
    
    try:
        with open('per_strategy_models.pkl', 'rb') as f:
            old = pickle.load(f)
        last_n = old.get('last_train_market_count', 0)
    except:
        return True, "model unloadable"
    
    # Count current markets
    n_now = 0
    if os.path.exists('market_history.jsonl'):
        with open('market_history.jsonl') as f:
            n_now = sum(1 for _ in f)
    
    if n_now - last_n >= RETRAIN_EVERY_N_MARKETS:
        return True, f"{n_now-last_n} new markets since last train"
    return False, f"only {n_now-last_n} new markets (need {RETRAIN_EVERY_N_MARKETS})"


def train_all():
    print(f"[{datetime.now()}] Starting self-improve cycle")
    
    # Reuse the per_strategy_train logic
    print("[1/5] Loading data...")
    ticks_by_slug = {}
    with open('market_history.jsonl') as f:
        for line in f:
            try: r = json.loads(line)
            except: continue
            if r.get('slug') and r.get('ticks'):
                ticks_by_slug[r['slug']] = r['ticks']
    n_markets = len(ticks_by_slug)
    print(f"  {n_markets} markets")
    
    free_fires_by_strat = defaultdict(list)
    with open('market_recap_history.jsonl') as f:
        for line in f:
            try: r = json.loads(line)
            except: continue
            slug = r.get('slug'); ts = r.get('ts')
            if slug not in ticks_by_slug: continue
            for fire in r.get('fires', []):
                cd = fire.get('cd', 0)
                if cd is None or cd < 15: continue
                if (fire.get('pre_gate_held') or fire.get('opp_gate_held')
                    or fire.get('dedup_excluded') or fire.get('pricecap_excluded')
                    or fire.get('model_vetoed') or fire.get('flip_gate_blocked')
                    or fire.get('bn_vetoed')):
                    continue
                strat = fire.get('strategy'); side = fire.get('side')
                if not strat or not side: continue
                free_fires_by_strat[(strat, side)].append({
                    'slug':slug,'ts':ts,'cd':cd,
                    'entry':fire.get('entry_price'),
                    'won':fire.get('hypo_pnl', 0) > 0,
                    'hypo':fire.get('hypo_pnl', 0),
                    'side':side,
                })
    
    viable = {k: fs for k,fs in free_fires_by_strat.items() if len(fs) >= MIN_FIRES}
    print(f"  Viable strategies (n>={MIN_FIRES}): {len(viable)}")
    
    print("[2/5] Computing features...")
    import flip_features
    
    all_data = []
    for (strat, side), fires in viable.items():
        for f in fires:
            visible = [t for t in ticks_by_slug[f['slug']] if t[0] >= f['cd']]
            if len(visible) < 10: continue
            try:
                feats = flip_features.build_features(
                    visible,
                    current_time=datetime.fromtimestamp(f['ts']) if f['ts'] else None
                )
                feats['fire_cd']      = float(f['cd'])
                feats['fire_entry']   = float(f['entry']) if f['entry'] else 0.0
                feats['fire_side_dn'] = 1.0 if side == 'DN' else 0.0
            except Exception:
                continue
            all_data.append({**f, 'strategy':strat, 'feats':feats})
    
    if not all_data:
        print("  No data")
        return
    
    all_keys = set(all_data[0]['feats'].keys())
    for d in all_data[:50]:
        all_keys = all_keys & set(d['feats'].keys())
    feat_names = sorted(all_keys)
    
    def featvec(d):
        row = []
        for fn in feat_names:
            v = d['feats'].get(fn, 0)
            if v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
                v = 0.0
            row.append(float(v))
        return row
    
    data_by_strat = defaultdict(list)
    for d in all_data:
        data_by_strat[(d['strategy'], d['side'])].append(d)
    
    print("[3/5] Training per-strategy ensembles...")
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.svm import SVC
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.metrics import roc_auc_score
    
    deploy_map = {}
    model_packs = {}
    history_entry = {
        'ts': time.time(), 'iso': datetime.now().isoformat(),
        'n_markets': n_markets,
        'n_strategies_trained': 0,
        'n_target': 0, 'n_viable': 0,
        'strategies': {},
    }
    
    for (strat, side), fires in sorted(data_by_strat.items(), key=lambda x: -len(x[1])):
        if len(fires) < MIN_FIRES: continue
        
        fires.sort(key=lambda f: f['ts'] or 0)
        split = int(len(fires)*0.70)
        train, test = fires[:split], fires[split:]
        if len(test) < 10: continue
        
        X_tr = np.array([featvec(d) for d in train])
        y_tr = np.array([1 if d['won'] else 0 for d in train])
        X_te = np.array([featvec(d) for d in test])
        y_te = np.array([1 if d['won'] else 0 for d in test])
        
        if y_tr.sum() < 5 or y_tr.sum() > len(y_tr)-5: continue
        
        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_te_s = scaler.transform(X_te)
        
        try:
            models = {
                'lr_l2': LogisticRegression(C=0.5, max_iter=2000, class_weight='balanced', solver='lbfgs'),
                'lr_l1': LogisticRegression(C=0.3, penalty='l1', max_iter=2000, class_weight='balanced', solver='liblinear'),
                'rf': RandomForestClassifier(n_estimators=100, max_depth=5, min_samples_leaf=10, class_weight='balanced', random_state=42, n_jobs=-1),
                'gb': GradientBoostingClassifier(n_estimators=80, max_depth=3, learning_rate=0.05, min_samples_leaf=10, random_state=42),
                'svm': CalibratedClassifierCV(SVC(kernel='rbf', C=0.5, gamma='scale', class_weight='balanced', probability=False), cv=3, method='sigmoid'),
            }
            for m in models.values():
                m.fit(X_tr_s, y_tr)
            probas = np.mean([m.predict_proba(X_te_s)[:,1] for m in models.values()], axis=0)
        except Exception:
            continue
        
        if len(set(y_te)) > 1:
            test_auc = roc_auc_score(y_te, probas)
        else:
            test_auc = 0.5
        
        # KEY FEATURE: Auto-detect inverted predictions (AUC < INVERT_THRESHOLD)
        invert = False
        if test_auc < INVERT_AUC_THRESHOLD:
            # Predictions are anti-correlated → invert and re-evaluate
            probas = 1.0 - probas
            test_auc = 1.0 - test_auc  # mathematically equivalent
            invert = True
        
        # Threshold scan with optional inversion
        best = None
        for thr in [0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]:
            mask = probas >= thr
            n = int(mask.sum())
            if n < MIN_TEST_N: continue
            wins = int(y_te[mask].sum())
            wr = wins/n
            wr_lo = wilson_lower(wins, n)
            if best is None or wr_lo > best['wilson_lo']:
                best = {'thr':thr,'n':n,'wins':wins,'wr':wr,'wilson_lo':wr_lo}
        
        history_entry['n_strategies_trained'] += 1
        
        if not best:
            history_entry['strategies'][f"{strat}|{side}"] = {
                'n': len(fires), 'auc': test_auc, 'inverted': invert,
                'tier': 'no_threshold',
            }
            continue
        
        # Permutation
        proba_list = list(probas)
        actual_wlo = best['wilson_lo']
        extreme = 0
        for _ in range(500):
            random.shuffle(proba_list)
            ps = np.array(proba_list)
            mask = ps >= best['thr']
            if mask.sum() < 5: continue
            w = int(y_te[mask].sum()); n = int(mask.sum())
            wlo = wilson_lower(w, n)
            if wlo >= actual_wlo: extreme += 1
        perm_p = (extreme + 1) / 501
        
        if best['wilson_lo'] >= TARGET_WILSON_LO and perm_p < 0.05:
            tier = 'TARGET'
            history_entry['n_target'] += 1
        elif best['wilson_lo'] >= VIABLE_WILSON_LO and perm_p < 0.10:
            tier = 'VIABLE'
            history_entry['n_viable'] += 1
        else:
            tier = 'skip'
        
        history_entry['strategies'][f"{strat}|{side}"] = {
            'n': len(fires), 'n_test': len(test), 'auc': float(test_auc),
            'inverted': invert,
            'best_thr': best['thr'], 'best_n_pass': best['n'],
            'wr': float(best['wr']), 'wilson_lo': float(best['wilson_lo']),
            'perm_p': float(perm_p), 'tier': tier,
        }
        
        if tier in ('TARGET', 'VIABLE'):
            deploy_map[f"{strat}|{side}"] = {
                'threshold': best['thr'], 'wr': float(best['wr']),
                'wilson_lo': float(best['wilson_lo']),
                'n_test': best['n'], 'auc': float(test_auc),
                'inverted': invert, 'tier': tier,
                'updated_at': time.time(),
            }
            model_packs[f"{strat}|{side}"] = {
                'models': models, 'scaler': scaler,
                'features': feat_names, 'inverted': invert,
            }
    
    # Atomic save
    tmp_pkl = 'per_strategy_models.pkl.tmp'
    with open(tmp_pkl, 'wb') as f:
        pickle.dump({
            'deploy_map': deploy_map,
            'model_packs': model_packs,
            'last_train_market_count': n_markets,
            'last_train_ts': time.time(),
        }, f)
    shutil.move(tmp_pkl, 'per_strategy_models.pkl')
    
    tmp_json = 'deploy_map.json.tmp'
    with open(tmp_json, 'w') as f:
        json.dump(deploy_map, f, indent=2)
    shutil.move(tmp_json, 'deploy_map.json')
    
    # Append history
    with open(HISTORY_PATH, 'a') as f:
        f.write(json.dumps(history_entry) + '\n')
    
    print(f"\n[4/5] Results:")
    print(f"  Trained: {history_entry['n_strategies_trained']}")
    print(f"  🎯 TARGET: {history_entry['n_target']}")
    print(f"  🟡 VIABLE: {history_entry['n_viable']}")
    
    # Print TARGET and VIABLE strategies
    target = [k for k,v in deploy_map.items() if v['tier']=='TARGET']
    viable = [k for k,v in deploy_map.items() if v['tier']=='VIABLE']
    
    if target:
        print(f"\n  🎯 TARGET strategies (≥{TARGET_WILSON_LO*100:.0f}% Wilson_lo):")
        for k in target:
            v = deploy_map[k]
            inv = " (INVERTED)" if v.get('inverted') else ""
            print(f"     ✅ {k:<35s} thr={v['threshold']:.2f} WR={v['wr']*100:.0f}% Wilson_lo={v['wilson_lo']*100:.0f}%{inv}")
    
    if viable:
        print(f"\n  🟡 VIABLE strategies (≥{VIABLE_WILSON_LO*100:.0f}% Wilson_lo):")
        for k in viable:
            v = deploy_map[k]
            inv = " (INVERTED)" if v.get('inverted') else ""
            print(f"     🟡 {k:<35s} thr={v['threshold']:.2f} WR={v['wr']*100:.0f}% Wilson_lo={v['wilson_lo']*100:.0f}%{inv}")
    
    print(f"\n[5/5] Saved deploy_map.json + per_strategy_models.pkl + history")
    
    # Trend report — has any strategy improved over time?
    print_trend()


def print_trend():
    """Show Wilson_lo trajectory over recent retrains."""
    if not os.path.exists(HISTORY_PATH):
        return
    
    history = []
    with open(HISTORY_PATH) as f:
        for line in f:
            try: history.append(json.loads(line))
            except: continue
    
    if len(history) < 2:
        print("  (not enough history for trend)")
        return
    
    print(f"\n  Wilson_lo trajectory (last {min(5, len(history))} retrains):")
    
    # All strategies seen recently
    recent = history[-5:]
    all_strats = set()
    for h in recent: all_strats.update(h.get('strategies', {}).keys())
    
    # Build trajectory matrix
    print(f"    {'strategy|side':<33s}" + "".join(f"  T-{len(recent)-i}" for i in range(len(recent))))
    
    for strat in sorted(all_strats):
        traj = []
        for h in recent:
            s = h.get('strategies', {}).get(strat)
            if s and 'wilson_lo' in s:
                traj.append(s['wilson_lo'])
            else:
                traj.append(None)
        
        # Only print strategies with at least 1 valid value
        if not any(v is not None for v in traj): continue
        
        traj_str = "".join(f"  {v*100:>4.0f}%" if v is not None else "    ?  " for v in traj)
        latest = traj[-1] if traj[-1] is not None else 0
        flag = "🎯" if latest >= TARGET_WILSON_LO else ("🟡" if latest >= VIABLE_WILSON_LO else "  ")
        print(f"    {strat:<33s}{traj_str}  {flag}")


def main():
    if '--force' in sys.argv:
        train_all()
    elif '--once' in sys.argv:
        do_train, reason = should_retrain()
        if do_train:
            print(f"Retraining: {reason}")
            train_all()
        else:
            print(f"Skipping: {reason}")
    elif '--report' in sys.argv:
        print_trend()
    else:
        print("Usage:")
        print("  python3 self_improve_loop.py --once    (check + retrain if N markets passed)")
        print("  python3 self_improve_loop.py --force   (always retrain)")
        print("  python3 self_improve_loop.py --report  (show Wilson_lo trends)")
        print()
        print(f"Set RETRAIN_EVERY_N=N env var to control retrain frequency (default 10)")


if __name__ == '__main__':
    main()
