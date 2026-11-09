#!/usr/bin/env python3
"""重建 V7X 名單（AB_V7_B10，13 j × seeds）：逐行沿用 o_v7_listing_exec_engine/s4_jscan7.py 的 JV7／days_for（只讀 import v7env→v4lib→ia_core(V7 面板)、st_core），
只多一個選項：處置遮罩 dmode ∈ {'orig'（＝DMT＝V4 官方名單 trade 口徑）, 'union', 'replace', 'union_noovl', 'union14'}（dispmask_official.make_mask）。
不 import v7common（避免 check_prereg 副作用）；dmode='orig' 必須逐筆重現 data/plans_j7.csv（smoke 斷言）。
python3 -I -B v7plans.py <dmode> <out.csv> [seeds=20]"""
import sys, time
from pathlib import Path
from multiprocessing import Pool
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent; AG = HERE.parent
sys.path.insert(0, str(AG / 'o_v7_listing_exec_engine'))
import v7env
L = v7env.load('v7'); K, V = L.K, L.V
sys.path.pop(0)
XR = L.setup_px()
YFX, Y5F = L.load_rfix(); DMT = L.disp_mask_fn('trade'); GAP = L.gap_dict()
sys.path.insert(0, str(AG / 'o_stagger_tranche')); import st_core as STC; sys.path.pop(0)
sys.path.insert(0, str(HERE)); import dispmask_official as DO; sys.path.pop(0)
OFF = L.V4(YFX, Y5F, DMT)
ANCH = [pd.Period(d, 'M') for d in OFF.sig_days]

class JV7(L.V4):
    def set_days(self, days, end_day):
        self.anchor = {d: m for m, d in days}; self.sig_days = [d for m, d in days]; self.end_day = end_day; self._cs = {}
    def y3(self, d, back=0):
        m = self.anchor[d] - back
        Y = self.Y5 if d >= V.SWITCH else self.YL
        y = K.yoy3_for_month(Y, m)
        if self.exfin: y = np.where(V.FINMASK, np.nan, y)
        return y

def days_for(j):
    out, skip = [], []
    for m in ANCH:
        d = STC.day_for(m, j)
        if d is None: skip.append((str(m), 'no_day')); continue
        t = K.IDX.get_loc(d)
        if not K.GOOD[t + 1]: skip.append((str(m), 'next_not_good')); continue
        out.append((m, d))
    de = STC.day_for(pd.Period('2026-10', 'M'), j)
    if de is None or de > K.IDX[-1]: de = K.IDX[-1]
    out = [(m, d) for m, d in out if d < de]
    return out, de, skip

DMODE = sys.argv[1] if len(sys.argv) > 1 else 'orig'
DM = DMT if DMODE == 'orig' else DO.make_mask(DMT, K.IDX, K.CI, K.N, DMODE)

def job(a):
    j, seed = a
    days, de, skip = days_for(j)
    S = JV7(YFX, Y5F, DM, unavail=(lambda d, _s=seed: STC.unavail_for(S.anchor[d], d, _s)), gap=GAP, pfix=1.0)
    S.set_days(days, de)
    pl = S.plan(10, 'official')
    return [('AB_V7_B10', j, seed, str(p[3].date()), str(K.IDX[p[0]].date()), int(p[0]), ','.join(K.COLS[jj] for jj in p[1])) for p in pl]

if __name__ == '__main__':
    out = sys.argv[2]; ns = int(sys.argv[3]) if len(sys.argv) > 3 else 20
    js = [int(x) for x in sys.argv[4].split(',')] if len(sys.argv) > 4 else list(range(13))
    T0 = time.time()
    jobs = [(j, s) for j in js for s in range(ns)]
    with Pool(4) as P: res = P.map(job, jobs, chunksize=1)
    rows = [r for x in res for r in x]
    pd.DataFrame(rows, columns=['strat', 'j', 'seed', 'signal', 'exec', 'exec_idx', 'picks']).to_csv(out, index=False)
    print(DMODE, len(rows), round(time.time() - T0), 's', flush=True)
