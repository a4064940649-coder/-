#!/usr/bin/env python3
"""複製員 o6：j＝0～12 換股日分布（AB_V6、AB_V6_B10；一般／壓力成本）。
換股日＝st_core.day_for(m, j)（m＝V6 正式訊號日的月份）；AB＝st_core.unavail_for(m, d, seed)，seed 0～39。
策略物件＝v4lib.V4（一字不改）；只把 sig_days／end_day 換成 j 日，y3 的月份改用錨定月 m（j 日可能跨月，例如 2019-02-15 期限）。
用法：O6R_PANEL=compat python3 -I -B r8_jscan.py"""
import os, sys, json, time
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import numpy as np, pandas as pd
import r4_env
PANEL = os.environ.get('O6R_PANEL', 'compat')
L = r4_env.load(PANEL); K, V = L.K, L.V
OUT = HERE / f'out_{PANEL}'
T0 = time.time()
def log(*a): print(f'[{time.time() - T0:6.0f}s]', *a, flush=True)
XR = L.setup_px()
YFX, Y5F = L.load_rfix(); GAP = L.gap_dict(); DMT = L.disp_mask_fn('trade')
sys.path.insert(0, str(L.AG / 'o_stagger_tranche'))
import st_core as ST
sys.path.pop(0)
assert ST.K is K
PJ = json.load(open(OUT / 'plans.json'))
SD = pd.DatetimeIndex(PJ['sig_days'][:-1])
ANCH = [pd.Period(d, 'M') for d in SD]

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

SPECS = {'V6': dict(), 'V6_B10': dict(gap=GAP, pfix=1.0)}
navs = {}; tj = {}
mean_nav = lambda L_: (1 + pd.concat(L_, axis=1).mean(axis=1).fillna(0)).cumprod()
for j in range(13):
    days = [(m, ST.day_for(m, j)) for m in ANCH]
    days = [(m, d) for m, d in days if d is not None]
    de = ST.day_for(pd.Period('2026-10', 'M'), j)
    if de is None or de > K.IDX[-1]: de = K.IDX[-1]
    days = [(m, d) for m, d in days if d < de]
    mof = {d: m for m, d in days}
    dl = [d for m, d in days]
    for k, kw in SPECS.items():
        rr = {'normal': [], 'stress': []}; ee = []
        for seed in range(40):
            U = (lambda s: (lambda d: ST.unavail_for(mof[d], d, s)))(seed)
            S = VJ(dl, de, mof, unavail=U, **kw)
            t0, t1, sd = S.bounds(); pl = S.plan(10, 'official')
            for c in rr:
                nav, _ = L.sim_plan(pl, t0, t1, c, XR); rr[c].append(nav.pct_change())
                tj[f'{k}|{c}|j{j}|s{seed}'] = L.seg(nav, nav, sd, '2006-04-01', '2026-10-12')['t_0050']
            if seed == 0:
                ew, _ = K.simulate([(p[0], p[2]) for p in pl], 0, 0, t0, t1); navs[f'{k}|j{j}|EW'] = ew
        for c in rr: navs[f'AB_{k}|j{j}|{c}'] = mean_nav(rr[c])
        log(j, k, len(dl), str(de.date()))
    json.dump({'j': j}, open(OUT / 'jscan_progress.json', 'w'))
    pd.DataFrame(navs).to_parquet(OUT / 'navs_jscan.parquet')
    json.dump({f'j{jj}': [str(d.date()) for m, d in [(m, ST.day_for(m, jj)) for m in ANCH] if d is not None] for jj in [j]},
              open(OUT / f'jdays_{j}.json', 'w'))
json.dump(tj, open(OUT / 'jscan_seed_t.json', 'w'))
log('done')
