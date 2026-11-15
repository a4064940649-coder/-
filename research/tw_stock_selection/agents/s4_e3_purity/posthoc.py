#!/usr/bin/env python3
"""S4 事後分析（看過 results.json 之後才加；不影響分類）。python3 -I -B posthoc.py → results_posthoc.json
A. E3' 空位 0050 的 20 條路徑，以及 F1、F5、N15 與它逐條相減（同空位設定）。
B. E3' 交易依進場名次分組，對照「同一段持有期改抱 0050」的報酬（0050 還原開盤進、還原開盤出；不扣成本）。"""
import sys, json, time, resource, gc
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
import s4lib
from s4lib import s2lib, stlib, COSTS

H = Path(__file__).resolve().parent
t0 = time.time()
P = s4lib.load(); U0 = stlib.universe(P, 'twse'); RT = s2lib.rev_tables(P)
INNOV, _ = s4lib.innov_panel(P); U = U0 & ~INNOV
for a in ('H', 'L', 'VOL', 'CHG', 'HF', 'VOLF', 'adjH', 'HI240', 'MA60', 'nbar', 'F_HI', 'IDXC', 'AO', 'listed'):
    if hasattr(P, a):
        delattr(P, a)
gc.collect()
D19 = P.IDX[P.IDX < pd.Timestamp('2019-01-01')][-1]; DE = P.IDX[-1]
res = json.load(open(H / 'results.json'))
CAPS = (6e5, 8e5, 1e6, 1.5e6, 2e6)
KW = {"E3'": dict(depth=None, mom_lo=0.10, exit='top20')}
out = {'A_paths_E3p_e50': {}}
vals = {}
for s in (0, 1, 2, 3):
    S = s4lib.make_signal(P, U, RT, shift=s, **KW["E3'"])
    for cap in CAPS:
        nav, _, _ = s4lib.simulate(P, S, U, idle='e50', cost=COSTS['strict'], capital=cap)
        vals[f'{int(cap/1e4)}萬|+{s}'] = stlib.cagr(nav, D19, DE)
    del S; gc.collect()
v = np.array(list(vals.values()))
out['A_paths_E3p_e50'] = dict(values=vals, min=float(v.min()), median=float(np.median(v)), max=float(v.max()))
for cid in ('F1', 'F5', 'N15', 'N5', 'Nd15'):
    pv = res['cells'][cid]['paths20']['values']
    d = np.array([pv[k] - vals[k] for k in vals])
    out['A_paths_E3p_e50'][f'{cid}_minus_E3p_e50'] = dict(min=float(d.min()), median=float(np.median(d)), max=float(d.max()), n_pos=int((d > 0).sum()))

# B. E3' 交易 vs 同期 0050
tr = pd.read_csv(H / 'trades.csv.gz', parse_dates=['ent_date', 'exit_date'])
x = tr[(tr['run'] == "E3'|cash|strict") & (tr['why'] != 'open_end')].copy()
di = {d: i for i, d in enumerate(P.IDX)}
e = np.array([di[d] for d in x['ent_date']]); q = np.array([di[d] for d in x['exit_date']])
x['r0050'] = P.E50O[q] / P.E50O[e] - 1
x['excess'] = x['ret'] - x['r0050']
grp = {'1-10': x['pos_ma'] <= 10, '11-20': (x['pos_ma'] > 10) & (x['pos_ma'] <= 20), '21+': x['pos_ma'] > 20, '11+': x['pos_ma'] > 10}
out['B_trades_vs_0050'] = {}
for k, f in grp.items():
    y = x[f]
    out['B_trades_vs_0050'][k] = dict(n=int(len(y)), ret_mean=float(y['ret'].mean()), r0050_mean=float(y['r0050'].mean()), excess_mean=float(y['excess'].mean()),
                                      excess_median=float(y['excess'].median()), excess_t=float(y['excess'].mean() / (y['excess'].std(ddof=1) / np.sqrt(len(y)))),
                                      hold_mean=float(y['hold_days'].mean()))
    for nm, a, b in (('half1', '2015-01-01', '2021-02-19'), ('half2', '2021-02-20', '2027-01-01')):
        z = y[(y['ent_date'] >= a) & (y['ent_date'] <= b)]
        out['B_trades_vs_0050'][k][nm] = dict(n=int(len(z)), excess_mean=float(z['excess'].mean()), ret_mean=float(z['ret'].mean()))
out['meta'] = dict(runtime_s=round(time.time() - t0, 1), maxrss_mb=round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024))
(H / 'results_posthoc.json').write_text(json.dumps(out, ensure_ascii=False, indent=1))
print(json.dumps(out, ensure_ascii=False, indent=1))
