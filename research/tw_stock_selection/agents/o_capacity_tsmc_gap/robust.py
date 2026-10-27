#!/usr/bin/env python3
"""事後穩健性（PREREG 之後、不改判定，只描述）：
(1) 後見之明檢查：30% 改放「換股日 60 日均成交值第 1 名」（事前可知，不指定台積電）。
(2) n30 ＋ 台積電 30%。
(3) 逐年贏 0050 的年數：w0、w30、0050 30% 混合。
(4) 台積電 30% 版去掉台積電最好的 3 年（把那 3 年 2330 的報酬換成 0050）。
輸出 results_robust.json。"""
import sys, json, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
import ocore as O
K = O.K
R = json.loads((O.HERE / 'results.json').read_text(encoding='utf-8'))
assert 'OOS' in R, '先跑 oos.py'
T0 = time.time()
out = {'note': '事後描述，不改變事前判定'}
S = O.V2R(1, 300)
pl = S.plan(10); t0, t1, sd = S.bounds(); P = O.periods(sd)
ew, _ = K.simulate([(p[0], p[2]) for p in pl], 0, 0, t0, t1)
KEYS = ('SEL', 'VAL', 'HOLD', 'S1', 'S2', 'S3', 'ALL')
def st(nav, e=ew, sd_=sd):
    return {k: {kk: vv for kk, vv in O.segstat(nav, e, sd_, *P[k]).items() if kk != 'yearly'} for k in KEYS}

# (1) 流動性第 1 名
plan1 = []; firsts = []
for p in pl:
    cs = S.cross_section(p[3]); j1 = int(np.argmin(cs['rank'])); firsts.append(O.COLS[j1])
    pk = np.asarray(p[1], int)
    plan1.append((p[0], np.concatenate([pk, [j1]]), np.concatenate([np.full(len(pk), 0.7 / len(pk)), [0.3]])))
nav1, _ = O.simulate_w(plan1, *O.NORMAL, t0, t1)
out['rank1_30pct'] = st(nav1)
out['rank1_codes_share'] = pd.Series(firsts).value_counts(normalize=True).round(3).head(8).to_dict()
print(time.time() - T0, 'rank1', {k: (v['CAGR'], v['t_0050'], v['maxDD']) for k, v in out['rank1_30pct'].items()}, out['rank1_codes_share'], flush=True)

# (2) n30 ＋ 30%
pl30 = S.plan(30)
ew30, _ = K.simulate([(p[0], p[2]) for p in pl30], 0, 0, t0, t1)
for w in (0.0, 0.3):
    nv, _ = O.simulate_w(O.tsmc_plan(pl30, w), *O.NORMAL, t0, t1)
    out[f'n30_w{int(w*100)}'] = st(nv, ew30)
print(time.time() - T0, 'n30', {k: (out['n30_w0'][k]['t_0050'], out['n30_w30'][k]['t_0050'], out['n30_w0'][k]['CAGR'], out['n30_w30'][k]['CAGR']) for k in KEYS}, flush=True)

# (3) 逐年
O_ = json.loads((O.HERE / 'results_oos.json').read_text(encoding='utf-8'))
nav0, _ = O.simulate_w(O.tsmc_plan(pl, 0.0), *O.NORMAL, t0, t1)
e50y = O.segstat(nav0, ew, sd, *P['ALL'])['yearly']
y0 = O_['tsmc']['w0']['yearly']; y30 = O_['tsmc']['w30']['yearly']; ym = O_['mix0050']['0050_30']['yearly']
E = K.E50.reindex(nav0.index); Ey = {int(y): round(float(g.iloc[-1] / E[E.index < g.index[0]].iloc[-1] - 1) * 100, 1) for y, g in E[E.index >= nav0.index[1]].groupby(E[E.index >= nav0.index[1]].index.year)}
out['yearly_vs_0050'] = {str(y): {'0050': Ey[int(y)], 'w0': y0[str(y)], 'w30': y30[str(y)], 'mix0050_30': ym[str(y)]} for y in y0}
out['years_beating_0050'] = {k: int(sum(1 for y in y0 if out['yearly_vs_0050'][y][k] > out['yearly_vs_0050'][y]['0050'])) for k in ('w0', 'w30', 'mix0050_30')}
out['n_years'] = len(y0)
print('years', out['years_beating_0050'], flush=True)

# (4) 去掉台積電最好的 3 年（2330 該年日報酬換成 0050 日報酬）
ts = pd.Series(K.GROW[:, O.J2330], index=O.IDX)
e50r = K.E50.pct_change().fillna(0) + 1
yr = {y: float(g.prod()) for y, g in ts[ts.index >= O.IDX[t0]].groupby(ts[ts.index >= O.IDX[t0]].index.year)}
ey = {y: float(g.prod()) for y, g in e50r[e50r.index >= O.IDX[t0]].groupby(e50r[e50r.index >= O.IDX[t0]].index.year)}
best3 = sorted(yr, key=lambda y: yr[y] / ey[y], reverse=True)[:3]
G2 = K.GROW.copy()
m = np.isin(O.IDX.year, best3)
G2[m, O.J2330] = e50r.to_numpy()[m]
nav4, _ = O.simulate_w(O.tsmc_plan(pl, 0.3), *O.NORMAL, t0, t1, G=G2)
out['w30_drop_tsmc_best3_years'] = {'years': [int(y) for y in best3], 'stats': st(nav4)}
print('best3', best3, {k: (v['CAGR'], v['t_0050'], v['maxDD']) for k, v in out['w30_drop_tsmc_best3_years']['stats'].items()}, flush=True)
(O.HERE / 'results_robust.json').write_text(O.dumps(out), encoding='utf-8')
print('done', time.time() - T0)
