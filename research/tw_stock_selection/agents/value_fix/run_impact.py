#!/usr/bin/env python3
"""用 value_fixed.parquet 重算主策略選股，與原面板比較（不改其他研究的檔案，只 import）。
A. revenue_longhistory/lhlib.py：M2、長期月營收，n10、n30（2005 起同一組換股日，比較 2024 起）
B. exec_rebuild/exlib.py：M2、rev_5y，n10（另附 n30）
輸出 results_impact.json、picks_diff.csv
用法：python3 -I -B run_impact.py [修正版 value 路徑] [輸出檔名後綴]"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
AG = HERE.parent
sys.path.insert(0, str(AG / 'revenue_longhistory')); sys.path.insert(0, str(AG / 'exec_rebuild'))
import lhlib as L   # noqa: E402
import exlib as X   # noqa: E402

VFP = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / 'value_fixed.parquet'
SUF = sys.argv[2] if len(sys.argv) > 2 else ''
VF = pd.read_parquet(VFP).reindex(index=L.IDX, columns=L.COLS)
assert VF.index.equals(X.IDX) and VF.columns.equals(X.COLS)
res = {}
rows = []
W_L = {'2024-01_2026-10': [('2024-01-01', '2026-10-31')], '2024-01_2025-12': [('2024-01-01', '2025-12-31')],
       'src3_2024-02-15_2025-09-23': [('2024-02-15', '2025-09-23')], 'overlap_2021-03_2026-10': [('2021-03-01', '2026-10-31')]}
KEYS = ('CAGR', 'maxDD', 't_vs_0050_m', 't_vs_EW_m', 'excess_vs_0050_pct_m', 'EW_CAGR', 'yearly_pct')

def short(o): return {k: o[k] for k in KEYS if o and k in o}

def diff_picks(tag, n, ev0, ev1, IDX, cols, pool0=None, pool1=None):
    out = []
    for ex in sorted(set(ev0) | set(ev1)):
        sig = IDX[ex - 1]
        if sig < pd.Timestamp('2023-06-01'): continue
        a = set(cols[ev0[ex][1]]) if ex in ev0 else set(); b = set(cols[ev1[ex][1]]) if ex in ev1 else set()
        r = {'strategy': tag, 'n': n, 'signal_date': str(sig.date()), 'n_same': len(a & b), 'removed(orig_only)': ' '.join(sorted(a - b)), 'added(fixed_only)': ' '.join(sorted(b - a))}
        if pool0 is not None:
            p0 = set(cols[pool0[ex][1]]) if ex in pool0 else set(); p1 = set(cols[pool1[ex][1]]) if ex in pool1 else set()
            r['pool_diff_names'] = len(p0 ^ p1) // 2 if len(p0) == len(p1) else len(p0 ^ p1); r['pool_size_orig'] = len(p0); r['pool_size_fixed'] = len(p1)
        out.append(r)
    return out

# ---------------- A. lhlib（長期月營收） ----------------
YL = L.yoy_long()
S0 = L.Setup(YL, start='2005-01-01')
v60_orig = L.v60_fix
L.v60_fix = VF.loc[L.GOOD].rolling(60).mean().reindex(L.IDX)
S1 = L.Setup(YL, start='2005-01-01')
L.v60_fix = v60_orig
assert S0.dates == S1.dates
t0 = L.pos(S0.dates[0]); t1 = L.pos(S0.dates[-1])
resA = {}
for n in (30, 10):
    ev0, pl0 = S0.selection(n); ev1, pl1 = S1.selection(n)
    rows += diff_picks('lhlib_M2_long', n, ev0, ev1, L.IDX, L.COLS, pl0, pl1)
    ew0, _ = L.sim_portfolio(pl0, 0, 0, t0, t1); ew1, _ = L.sim_portfolio(pl1, 0, 0, t0, t1)
    for slip in (0.001, 0.003, 'stress'):
        b, s = L.cost_rates(slip)
        n0, _ = L.sim_portfolio(ev0, b, s, t0, t1); n1, _ = L.sim_portfolio(ev1, b, s, t0, t1)
        resA[f'n{n}_slip{slip}'] = {w: {'orig': short(L.daily_stats(n0, sg, S0.DATES, L.E50, ew0)), 'fixed': short(L.daily_stats(n1, sg, S1.DATES, L.E50, ew1))} for w, sg in W_L.items()}
        if slip == 0.001:
            resA[f'n{n}_slip{slip}']['full_2006-04_2020-12_unchanged_check'] = {
                'orig': short(L.daily_stats(n0, [(S0.dates[0], '2020-12-31')], S0.DATES, L.E50, ew0))['CAGR'],
                'fixed': short(L.daily_stats(n1, [(S1.dates[0], '2020-12-31')], S1.DATES, L.E50, ew1))['CAGR']}
            resA[f'n{n}_slip{slip}']['nav_end_ratio_fixed_over_orig'] = round(float(n1.iloc[-1] / n0.iloc[-1]), 4)
        print('A', n, slip, {w: (v['orig'].get('CAGR'), v['fixed'].get('CAGR')) for w, v in resA[f'n{n}_slip{slip}'].items() if isinstance(v, dict) and 'orig' in v and isinstance(v['orig'], dict)}, flush=True)
res['lhlib_M2_long'] = resA

# ---------------- B. exlib（rev_5y） ----------------
ex0 = {n: X.selection('M2', n) for n in (10, 30)}
X.v60_all = VF.rolling(60).mean()
X.t300_d = X.t300_at(X.DATES); X.t300_dm1 = X.t300_at(X.prev_day)
ex1 = {n: X.selection('M2', n) for n in (10, 30)}
u0 = X.pos(X.dates[0]); u1 = X.pos(X.dates[-1])
W_X = {'full_2021-03_2026-10': [('2021-03-01', '2026-10-31')], 'valid_2024-01_2026-10': [('2024-01-01', '2026-10-31')],
       'valid_ex2026_2024-01_2025-12': [('2024-01-01', '2025-12-31')], 'src3_2024-02-15_2025-09-23': [('2024-02-15', '2025-09-23')]}
resB = {}
for n in (10, 30):
    (ev0, pl0), (ev1, pl1) = ex0[n], ex1[n]
    rows += diff_picks('exlib_M2_rev5y', n, ev0, ev1, X.IDX, X.COLS, pl0, pl1)
    ew0, _ = X.sim_portfolio(pl0, 0, 0, u0, u1); ew1, _ = X.sim_portfolio(pl1, 0, 0, u0, u1)
    for slip in (0.001, 0.003, 'stress'):
        b, s = X.cost_rates(slip)
        n0, _ = X.sim_portfolio(ev0, b, s, u0, u1); n1, _ = X.sim_portfolio(ev1, b, s, u0, u1)
        resB[f'n{n}_slip{slip}'] = {w: {'orig': short(X.daily_stats(n0, sg, X.E50_NAV, ew0)), 'fixed': short(X.daily_stats(n1, sg, X.E50_NAV, ew1))} for w, sg in W_X.items()}
        resB[f'n{n}_slip{slip}']['nav_end_ratio_fixed_over_orig'] = round(float(n1.iloc[-1] / n0.iloc[-1]), 4)
        print('B', n, slip, {w: (v['orig'].get('CAGR'), v['fixed'].get('CAGR'), v['orig'].get('t_vs_0050_m'), v['fixed'].get('t_vs_0050_m')) for w, v in resB[f'n{n}_slip{slip}'].items() if isinstance(v, dict)}, flush=True)
res['exlib_M2_rev5y'] = resB

D = pd.DataFrame(rows)
D.to_csv(HERE / f'picks_diff{SUF}.csv', index=False)
chg = D[(D['removed(orig_only)'] != '') | (D['pool_diff_names'] > 0)]
res['rebalances_with_pick_change'] = {f'{k[0]}_n{k[1]}': g.loc[g['removed(orig_only)'] != '', ['signal_date', 'removed(orig_only)', 'added(fixed_only)']].to_dict('records') for k, g in D.groupby(['strategy', 'n'])}
res['pool_diff_by_signal_date'] = {f'{k[0]}_n{k[1]}': {r['signal_date']: r['pool_diff_names'] for _, r in g.iterrows() if r['pool_diff_names'] > 0} for k, g in D.groupby(['strategy', 'n'])}
(HERE / f'results_impact{SUF}.json').write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str), encoding='utf-8')
print(json.dumps({k: res[k] for k in ('rebalances_with_pick_change', 'pool_diff_by_signal_date')}, ensure_ascii=False, indent=0))
