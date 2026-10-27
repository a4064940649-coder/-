#!/usr/bin/env python3
"""診斷：依 lhlib.Setup/selection 原始碼語意在本檔重寫（不 import），產生其持股，再用 ia_core 引擎（不凍結＝其作法）重跑，與 oos_nav.parquet 逐日比對。"""
import sys; from pathlib import Path; sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd, ia_core as K
n = int(sys.argv[1]) if len(sys.argv) > 1 else 30
IDX = K.IDX; COLS = K.COLS
d = pd.read_csv(K.LONG, dtype={'code': str, 'market': str, 'name': str})
d['p'] = pd.PeriodIndex(pd.to_datetime(d['month']), freq='M')
d = d.sort_values(['p', 'market']).drop_duplicates(['p', 'code'], keep='first'); d = d[d['code'].isin(COLS)]
L = d.pivot_table(index='p', columns='code', values='revenue_k_twd', aggfunc='last').sort_index()
L = L.reindex(pd.period_range(L.index.min(), L.index.max(), freq='M')); prev = L.shift(12)
Y = ((L / prev - 1) * 100).where(prev > 0)
YOY3 = Y.rolling(3, min_periods=3).mean()
GOOD = K.ADJ.notna().sum(axis=1) >= 300
ag = K.ADJ.loc[GOOD]; vg = K.V.loc[GOOD]
s52_all = (ag / ag.rolling(252).max()).reindex(IDX); v60_all = vg.rolling(60).mean().reindex(IDX)
S = K.Strategy(K.yoy_from_long(), '2005-01-01')
dates = S.sig_days + [S.end_day]; DATES = pd.DatetimeIndex(dates)
mm = [pd.Period(x, 'M') - 1 for x in dates]
yoy3 = pd.DataFrame(YOY3.reindex(mm).to_numpy(), index=DATES, columns=YOY3.columns).reindex(columns=COLS)
v60 = v60_all.reindex(DATES); px = K.C.reindex(DATES); base = (v60 >= 1e7) & (px >= 10)
t300 = base & (v60.where(base).rank(axis=1, ascending=False) <= 300)
s52 = s52_all.reindex(DATES)
inD = pd.DataFrame(False, index=DATES, columns=COLS)
for _, r in K.DSP.iterrows():
    if r['code'] in inD.columns and pd.notna(r['start']) and pd.notna(r['end']):
        inD.loc[(inD.index >= r['start']) & (inD.index <= r['end']), r['code']] = True
el = t300 & s52.notna() & ~inD & yoy3.notna()
sc = 0.5 * s52.where(el).rank(axis=1, pct=True) + 0.5 * yoy3.where(el).rank(axis=1, pct=True)
AC = K.ADJ.ffill().to_numpy(); AOr = K.On * (K.ADJ / K.C).to_numpy()
ACp = np.vstack([np.full((1, K.N), np.nan), AC[:-1]])
LIM = np.where(np.asarray(IDX < pd.Timestamp("2015-06-01")), 0.07, 0.10).reshape(-1, 1)
GU = (AOr / ACp - 1) >= LIM - 0.006
held = set(); plan = []; theirs = {}
for x in dates[:-1]:
    e = IDX.get_loc(x) + 1
    ok = el.loc[x].to_numpy() & np.isfinite(K.On[e]) & np.isfinite(K.ADJ.to_numpy()[e]) & np.isfinite(sc.loc[x].to_numpy())
    if ok.sum() < 50: continue
    order = sc.loc[x].to_numpy().copy(); order[~ok] = np.nan
    cand = np.argsort(-np.nan_to_num(order, nan=-9))[:int(ok.sum())]
    pick = []
    for j in cand:
        if GU[e, j] and j not in held: continue
        pick.append(j)
        if len(pick) == n: break
    plan.append((e, np.array(pick))); held = set(pick); theirs[x] = pick
t0 = IDX.get_loc(dates[0]); t1 = IDX.get_loc(dates[-1])
b, s = K.rates(0.001)
nav, _ = K.simulate(plan, b, s, t0, t1, freeze=False)
ref = pd.read_parquet(K.HERE.parent / 'revenue_longhistory/oos_nav.parquet')[f'M2_n{n}_slip0.001']
q = (nav.pct_change() - ref.reindex(nav.index).pct_change()).abs()
print('n', n, 'max daily diff vs oos_nav', float(q.max()), 'end', nav.iloc[-1], ref.iloc[-1])
o = K.stats(nav, '2006-04-01', '2020-12-31', dates, K.E50)
print('replica CAGR P1', round(o['CAGR'] * 100, 2), 't0050', round(o['t_0050'], 2))
# 我的選股 vs 其選股
mine = {p[3]: set(p[1]) for p in S.plan(n)}
nd = sum(1 for x in theirs if set(theirs[x]) != mine.get(x)); print('periods with different picks (mine vs theirs)', nd, 'of', len(theirs))
nav2, _ = K.simulate(plan, b, s, t0, t1, freeze=True)
o2 = K.stats(nav2, '2006-04-01', '2020-12-31', dates, K.E50); print('their picks + freeze', round(o2['CAGR'] * 100, 2), round(o2['t_0050'], 2))
