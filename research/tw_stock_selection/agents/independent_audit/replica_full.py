#!/usr/bin/env python3
"""診斷：依 exlib.py 原始碼語意（pandas 寫法）在本檔重寫 M2 選股，與 ia_core 逐期比對持股。不 import exlib。"""
import sys; from pathlib import Path; sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd, ia_core as K
n = int(sys.argv[1]) if len(sys.argv) > 1 else 30
X = K.X; COLS = K.COLS; IDX = K.IDX
rev = pd.read_csv(X / 'rev_5y.csv.gz', dtype={'code': str}); rev = rev[rev['code'].isin(COLS)]; rev['p'] = pd.PeriodIndex(rev['ym'], freq='M')
YOY = rev.pivot_table(index='p', columns='code', values='yoy', aggfunc='last').sort_index()
YOY = YOY.reindex(pd.period_range(YOY.index.min(), YOY.index.max(), freq='M')); YOY3 = YOY.rolling(3, min_periods=3).mean()
S = K.Strategy(K.yoy_from_rev5y(), '2021-01-01')
dates = S.sig_days + [S.end_day]; DATES = pd.DatetimeIndex(dates)
mm = [pd.Period(d, 'M') - 1 for d in dates]
yoy3 = pd.DataFrame(YOY3.reindex(mm).to_numpy(), index=DATES, columns=YOY3.columns).reindex(columns=COLS)
v60 = K.V.rolling(60).mean().reindex(DATES); px = K.C.reindex(DATES); base = (v60 >= 1e7) & (px >= 10)
t300 = base & (v60.where(base).rank(axis=1, ascending=False) <= 300)
s52 = (K.ADJ / K.ADJ.rolling(252).max()).reindex(DATES)
inD = pd.DataFrame(False, index=DATES, columns=COLS)
for _, r in K.DSP.iterrows():
    if r['code'] in inD.columns and pd.notna(r['start']) and pd.notna(r['end']):
        inD.loc[(inD.index >= r['start']) & (inD.index <= r['end']), r['code']] = True
el = t300 & yoy3.notna() & s52.notna() & ~inD
sc = 0.5 * s52.where(el).rank(axis=1, pct=True) + 0.5 * yoy3.where(el).rank(axis=1, pct=True)
pl = {p[3]: p for p in S.plan(n)}
ndiff = 0
for d in dates[:-1]:
    cs = S.cross_section(d)
    e = cs['e']
    okb = el.loc[d].to_numpy() & np.isfinite(K.On[e]) & np.isfinite(K.ADJ.to_numpy()[e])
    elme = cs['el']
    if (okb != elme).any() or np.nanmax(np.abs(sc.loc[d].to_numpy()[okb] - cs['sc'][okb])) > 1e-12:
        ndiff += 1
        onlyT = [COLS[j] for j in np.where(okb & ~elme)[0]]; onlyM = [COLS[j] for j in np.where(~okb & elme)[0]]
        print(d.date(), 'pool diff theirs-only', onlyT[:10], 'mine-only', onlyM[:10])
        for c in (onlyT + onlyM)[:4]:
            j = K.CI[c]
            print('   ', c, 'yoy3 theirs', yoy3.loc[d, c], 'mine', cs['y3'][j], 'top', t300.loc[d, c], cs['top'][j], 'inD', inD.loc[d, c])
print('periods with pool/score diff', ndiff)

# ---- 依其語意產生持股，與我的持股比較 ----
AC = K.ADJ.ffill().to_numpy(); AOr = K.On * (K.ADJ / K.C).to_numpy()
ACp = np.vstack([np.full((1, K.N), np.nan), AC[:-1]])
GU = (AOr / ACp - 1) >= 0.10 - 0.006
held = set(); theirs = {}
for d in dates[:-1]:
    e = IDX.get_loc(d) + 1
    ok = el.loc[d].to_numpy() & np.isfinite(K.On[e]) & np.isfinite(K.ADJ.to_numpy()[e])
    order = sc.loc[d].to_numpy().copy(); order[~ok] = np.nan
    cand = np.argsort(-np.nan_to_num(order, nan=-9))[:int(ok.sum())]
    pick = []
    for j in cand:
        if GU[e, j] and j not in held: continue
        pick.append(j)
        if len(pick) == n: break
    theirs[d] = pick; held = set(pick)
nd = 0
for d in dates[:-1]:
    a = set(pl[d][1]); b = set(theirs[d])
    if a != b:
        nd += 1
        cs = S.cross_section(d)
        print(d.date(), 'mine-only', [(K.COLS[j], round(cs['sc'][j], 4)) for j in a - b], 'theirs-only', [(K.COLS[j], round(cs['sc'][j], 4), round(float(sc.loc[d].iloc[j]), 4)) for j in b - a])
print('pick diff periods', nd)
