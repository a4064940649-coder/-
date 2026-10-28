#!/usr/bin/env python3
"""j 掃描替代定義：j＝當月第一個日期 ≥ 11+j 的面板交易日。"""
import sys, json, pickle
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
import rlib as L, rstrat as S
from r1_main import XS, DATES
G = L.GOOD.to_numpy()
GI = L.IDX[G]
s = pd.Series(GI, index=GI); s = s[s >= '2006-04-01']
months = [(y, m) for (y, m), _ in s.groupby([s.index.year, s.index.month])]
grp = {k: g for k, g in s.groupby([s.index.year, s.index.month])}
off = []
for d in DATES:
    g = grp[(d.year, d.month)]; off.append(int((g < d).sum()))
print('official j distribution', pd.Series(off).value_counts().sort_index().to_dict())
cache = L.OUT / 'xs_jcal.pkl'
XJ = pickle.load(open(cache, 'rb')) if cache.exists() else {}
rows = []
for j in range(13):
    dj = []
    for d in DATES[1:]:
        g = grp[(d.year, d.month)]
        gg = g[g.dt.day >= 11 + j]; x = gg.iloc[0] if len(gg) else g.iloc[-1]
        if L.POS[x] + 1 >= len(L.IDX): x = L.IDX[-2]
        while not G[L.POS[x] + 1]: x = GI[GI > x][0]
        dj.append(x)
    dj = pd.DatetimeIndex(dj); lab = dict(zip(dj, DATES[1:]))
    W = {}; held = set()
    for d in dj[:-1]:
        if d not in XJ: XJ[d] = XS[d] if d in XS else S.XS(d)
        w, _, _ = S.f1_weights(XJ[d], held); W[d] = w; held = set(np.where(w > 0)[0])
    nv, _ = S.simulate(W, 'normal')
    wr = S.win_returns(nv, dj); b = L.E50.reindex(dj); bw = (b.shift(-1) / b - 1).iloc[:-1]
    ex = (wr - bw).dropna(); ex.index = [lab[d] for d in ex.index]
    r = {'j': j}
    for p in ('SEL', 'VAL', 'POST2019'):
        x = S.sel(ex, p); r[p + '_ex'] = round(x.mean() * 100, 3); r[p + '_t'] = round(S.tstat(x), 2)
    rows.append(r); print(r, flush=True)
pickle.dump(XJ, open(cache, 'wb'))
df = pd.DataFrame(rows).set_index('j')
summ = {'VAL_t_median': df.VAL_t.median(), 'VAL_t_p10': df.VAL_t.quantile(0.1), 'VAL_ex_median': df.VAL_ex.median(),
        'VAL_ex_p10': df.VAL_ex.quantile(0.1), 'POST_t_median': df.POST2019_t.median(), 'SEL_t_median': df.SEL_t.median(), 'SEL_t_max': df.SEL_t.max()}
json.dump({'jscan_alt': df.to_dict(orient='index'), 'summary': summ}, open(L.OUT / 'r3c_jscan_cal.json', 'w'), indent=1, default=float)
print(summ)
