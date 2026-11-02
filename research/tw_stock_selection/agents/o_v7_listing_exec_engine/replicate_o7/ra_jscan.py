#!/usr/bin/env python3
"""複製員 o7 軌 A：j＝0～12（st_core.day_for）；AB＝st_core.unavail_for(m,d,seed) seeds 0～39；V7、V7_B10；一般／壓力。
引擎＝v4lib.V4（一字不改），只換 sig_days／end_day，y3 月份用錨定月 m。"""
import sys, json, time
from pathlib import Path
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
import numpy as np, pandas as pd
import r_env
L = r_env.load(); K, V = L.K, L.V
OUT = HERE / 'out'; T0 = time.time()
def log(*a): print(f'[{time.time() - T0:6.0f}s]', *a, flush=True)
XR = L.setup_px(); YFX, Y5F = L.load_rfix(); GAP = L.gap_dict(); DMT = L.disp_mask_fn('trade')
sys.path.insert(0, str(L.AG / 'o_stagger_tranche')); import st_core as ST; sys.path.pop(0)
assert ST.K is K
S0 = L.V4(YFX, Y5F, DMT); _, _, SD = S0.bounds()
ANCH = [pd.Period(d, 'M') for d in SD[:-1]]
class VJ(L.V4):
    def __init__(self, days, end_day, mof, **kw):
        super().__init__(YFX, Y5F, DMT, **kw)
        self.sig_days = list(days); self.end_day = end_day; self.mof = mof
    def y3(self, d, back=0):
        m = self.mof.get(d, pd.Period(d, 'M'))
        Y = self.Y5 if d >= V.SWITCH else self.YL
        y = K.yoy3_for_month(Y, m - back)
        if self.exfin: y = np.where(V.FINMASK, np.nan, y)
        return y
SPECS = {'V7': dict(), 'V7_B10': dict(gap=GAP, pfix=1.0)}
navs = {}; tj = {}; jd = {}
mean_nav = lambda L_: (1 + pd.concat(L_, axis=1).mean(axis=1).fillna(0)).cumprod()
for j in range(13):
    days = [(m, ST.day_for(m, j)) for m in ANCH]; days = [(m, d) for m, d in days if d is not None]
    de = ST.day_for(pd.Period('2026-10', 'M'), j)
    if de is None or de > K.IDX[-1]: de = K.IDX[-1]
    days = [(m, d) for m, d in days if d < de]
    mof = {d: m for m, d in days}; dl = [d for m, d in days]
    jd[f'j{j}'] = [str(d.date()) for d in dl] + [str(de.date())]
    for k, kw in SPECS.items():
        rr = {'normal': [], 'stress': []}
        for seed in range(40):
            U = (lambda s: (lambda d: ST.unavail_for(mof[d], d, s)))(seed)
            S = VJ(dl, de, mof, unavail=U, **kw); t0, t1, sd = S.bounds(); pl = S.plan(10, 'official')
            for c in rr:
                nav, _ = L.sim_plan(pl, t0, t1, c, XR); rr[c].append(nav.pct_change())
                tj[f'{k}|{c}|j{j}|s{seed}'] = L.seg(nav, nav, sd, '2006-04-01', '2026-10-12')['t_0050']
        for c in rr: navs[f'AB_{k}|j{j}|{c}'] = mean_nav(rr[c])
        log(j, k, len(dl), str(de.date()))
    pd.DataFrame(navs).to_parquet(OUT / 'navs_jscan_a.parquet')
json.dump(tj, open(OUT / 'jscan_seed_t.json', 'w')); json.dump(jd, open(OUT / 'jdays.json', 'w'))
log('done')
