#!/usr/bin/env python3
"""S5 事後分析（看過 results.json 之後才加；不影響分類，不新增格子、不重跑任何回測）。python3 -I -B posthoc.py → results_posthoc.json
P1 逐年對 0050 的差、贏的年數。P2 滾動 252 個交易日報酬對 0050 的差（最差、最好、落後的比例）。P3 最大回撤的高點與低點日期。
P4 主格交易依進場名次 1～5／6～10 對照同期 0050。P5 XQ 近似：MinSurge 固定門檻 2.2／2.5／3.0 選出的名單與真正「L_MA 前 10 名中 RET20 > 10%」的重疊。"""
import sys, json, time, resource, gc
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
import s5lib
from s5lib import s4lib, s2lib, stlib

H = Path(__file__).resolve().parent
t0 = time.time()
res = json.load(open(H / 'results.json'))
nv = pd.read_csv(H / 'navs.csv.gz', index_col=0, parse_dates=True)
tr = pd.read_csv(H / 'trades.csv.gz', parse_dates=['sig_date', 'ent_date', 'exit_date'])
CELLS = list(res['cells'].keys())
b = nv['0050']
out = {'P1_yearly_vs_0050': {}, 'P2_rolling252_vs_0050': {}, 'P3_maxdd_dates': {}}
for c in CELLS:
    y = res['cells'][c]['runs']['strict']['yearly']
    d = {k: v['strat'] - v['e0050'] for k, v in y.items() if k != '2015'}
    out['P1_yearly_vs_0050'][c] = dict(diff=d, n_years=len(d), n_beat=int(sum(v > 0 for v in d.values())))
    n = nv[f'{c}|strict']
    rr = (n / n.shift(252) - 1) - (b / b.shift(252) - 1)
    rr = rr.loc['2016-03-01':].dropna()
    out['P2_rolling252_vs_0050'][c] = dict(n=int(len(rr)), worst=float(rr.min()), worst_end=str(rr.idxmin().date()), best=float(rr.max()), best_end=str(rr.idxmax().date()),
                                         frac_behind=float((rr < 0).mean()), median=float(rr.median()))
for c in ['0050'] + CELLS:
    n = nv['0050'] if c == '0050' else nv[f'{c}|strict']
    dd = n / n.cummax() - 1
    tr_ = dd.idxmin(); pk = n.loc[:tr_].idxmax()
    rec = n.loc[tr_:]; rec = rec[rec >= n.loc[pk]]
    out['P3_maxdd_dates'][c] = dict(maxdd=float(dd.min()), peak=str(pk.date()), trough=str(tr_.date()), recovered=str(rec.index[0].date()) if len(rec) else None)

# P4 主格依名次
P = s5lib.load()
x = tr[(tr['run'] == 'W70_T10|strict') & (tr['why'] != 'open_end')].copy()
di = {d: i for i, d in enumerate(P.IDX)}
e = np.array([di[d] for d in x['ent_date']]); q = np.array([di[d] for d in x['exit_date']])
x['excess'] = x['ret'] - (P.E50O[q] / P.E50O[e] - 1)
out['P4_main_by_rank'] = {}
for lab, f in (('1-5', x['pos_ma'] <= 5), ('6-10', x['pos_ma'] > 5)):
    z = x[f]
    out['P4_main_by_rank'][lab] = dict(n=int(len(z)), ret_mean=float(z['ret'].mean()), ret_median=float(z['ret'].median()), excess_mean=float(z['excess'].mean()),
                                       excess_t=float(z['excess'].mean() / (z['excess'].std(ddof=1) / np.sqrt(len(z)))), loss_gt15=float((z['ret'] < -0.15).mean()))

# P5 XQ 固定門檻近似的名單重疊（只算名單，不算報酬）
U0 = stlib.universe(P, 'twse'); RT = s2lib.rev_tables(P)
INNOV, _ = s4lib.innov_panel(P); U = U0 & ~INNOV
del U0; gc.collect()
S, av = RT['S'], RT['av']
rows = []
for m in S.index:
    k = av[m]
    if k >= P.T or k < 61:
        continue
    s = S.loc[m].to_numpy()
    if int((U[k] & np.isfinite(s)).sum()) < 50:
        continue
    ok = U[k] & np.isfinite(s) & P.F_MA60[k]
    o = np.flatnonzero(ok)[np.argsort(-s[ok], kind='stable')]
    r20 = P.RET20[k]
    true = set(o[:10][r20[o[:10]] > 0.10].tolist())
    row = dict(date=str(P.IDX[k].date()), n_true=len(true))
    for th in (2.2, 2.5, 3.0):
        cand = o[(r20[o] > 0.10) & (s[o] >= th)][:10]          # XQ：SURGE ≥ 門檻且已在漲，最多 10 檔（依 SURGE 排）
        cs = set(cand.tolist())
        row[f'n_{th}'] = len(cs); row[f'overlap_{th}'] = len(cs & true); row[f'extra_{th}'] = len(cs - true); row[f'miss_{th}'] = len(true - cs)
    rows.append(row)
df = pd.DataFrame(rows)
out['P5_xq_minsurge_overlap'] = {str(th): dict(months=int(len(df)), mean_n_true=float(df['n_true'].mean()), mean_n_xq=float(df[f'n_{th}'].mean()),
                                               share_true_captured=float(df[f'overlap_{th}'].sum() / df['n_true'].sum()),
                                               share_xq_outside_top10=float(df[f'extra_{th}'].sum() / max(df[f'n_{th}'].sum(), 1)),
                                               months_exact=int(((df[f'extra_{th}'] == 0) & (df[f'miss_{th}'] == 0)).sum()))
                                 for th in (2.2, 2.5, 3.0)}
out['meta'] = dict(runtime_s=round(time.time() - t0, 1), maxrss_mb=round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024))
(H / 'results_posthoc.json').write_text(json.dumps(out, ensure_ascii=False, indent=1))
print(json.dumps(out, ensure_ascii=False, indent=1))
