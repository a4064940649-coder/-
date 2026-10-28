#!/usr/bin/env python3
"""PREREG 前的流程檢查用：只產生少量名單（不算報酬）到 scratchpad，供 run_exec7.py 的 DRY 模式使用。"""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
import v7env, v7common as C
OUT = Path(sys.argv[1]); OUT.mkdir(parents=True, exist_ok=True)
L = v7env.load('v7'); K, V = L.K, L.V
XR = L.setup_px(); YFX, Y5F = L.load_rfix(); DMT = L.disp_mask_fn('trade'); GAP = L.gap_dict()
sys.path.insert(0, str(C.AG / 'o_stagger_tranche')); import st_core as STC; sys.path.pop(0)
rows = []
for tag, kw in (('AB_V7', {}), ('AB_V7_B10', dict(gap=GAP, pfix=1.0))):
    for seed in (0, 1):
        S = L.V4(YFX, Y5F, DMT, unavail=V.make_unavail('mc', seed), **kw); pl = S.plan(10, 'official')
        rows += [(tag, seed, str(p[3].date()), str(K.IDX[p[0]].date()), int(p[0]), ','.join(K.COLS[j] for j in p[1])) for p in pl if p[3] <= pd.Timestamp('2009-12-31')]
pd.DataFrame(rows, columns=['strat', 'seed', 'signal', 'exec', 'exec_idx', 'picks']).to_csv(OUT / 'plans_ab7.csv', index=False)
OFF = L.V4(YFX, Y5F, DMT); ANCH = [pd.Period(d, 'M') for d in OFF.sig_days]
class JV7(L.V4):
    def set_days(self, days, end_day):
        self.anchor = {d: m for m, d in days}; self.sig_days = [d for m, d in days]; self.end_day = end_day; self._cs = {}
    def y3(self, d, back=0):
        m = self.anchor[d] - back; Y = self.Y5 if d >= V.SWITCH else self.YL
        y = K.yoy3_for_month(Y, m)
        return np.where(V.FINMASK, np.nan, y) if self.exfin else y
j = 3; days = []
for m in ANCH:
    d = STC.day_for(m, j)
    if d is not None and K.GOOD[K.IDX.get_loc(d) + 1] and d <= pd.Timestamp('2009-12-31'): days.append((m, d))
rows = []
for tag, kw in (('AB_V7', {}), ('AB_V7_B10', dict(gap=GAP, pfix=1.0))):
    S = JV7(YFX, Y5F, DMT, unavail=(lambda d, _s=0: STC.unavail_for(S.anchor[d], d, _s)), **kw); S.set_days(days, pd.Timestamp('2010-01-11'))
    pl = S.plan(10, 'official')
    rows += [(tag, j, 0, str(p[3].date()), str(K.IDX[p[0]].date()), int(p[0]), ','.join(K.COLS[jj] for jj in p[1])) for p in pl]
pd.DataFrame(rows, columns=['strat', 'j', 'seed', 'signal', 'exec', 'exec_idx', 'picks']).to_csv(OUT / 'plans_j7.csv', index=False)
json.dump({'AB_V7|J3': '2010-01-11', 'AB_V7_B10|J3': '2010-01-11'}, open(OUT / 'jscan_end_days.json', 'w'))
print('dry plans written')
