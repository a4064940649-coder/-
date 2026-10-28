#!/usr/bin/env python3
"""換股日分布（j＝0～12）的 AB 名單：V4 凍結版資料與規則（o_v4_revenue_patch_official/v4lib.py），j 定義與 AB 抽樣沿用
o_stagger_tranche/st_core.py（day_for、unavail_for；與 o_v5_pit_revenue/jscan_v5.py 相同，只是資料用 V4、不做首次值替換）。
兩種名單：AB_V4（gap=None）與 AB_V4_B10（gap=GAP, pfix=1.0）；seed 0～19（事前登記：j-scan 用 20 seeds 以控制計算量）。
同時用零衝擊一般成本（O3.simulate）算每個 j 的 20 seed 平均淨值，存 data/jscan_zero_navs.parquet（只當對照，不是判定）。
輸出 data/plans_j.csv（strat, j, seed, signal, exec, exec_idx, picks）、data/jscan_meta.json。
用法：python3 -I -B gen_jplans.py
"""
import sys, json, time
from pathlib import Path
HERE = Path(__file__).resolve().parent
AG = HERE.parent
sys.path.insert(0, str(AG / 'o_v4_revenue_patch_official'))
import numpy as np, pandas as pd
from multiprocessing import Pool
import v4lib as L
sys.path.pop(0)
K, V = L.K, L.V
XR = L.setup_px()
YFX, Y5F = L.load_rfix(); DMT = L.disp_mask_fn('trade'); GAP = L.gap_dict()
sys.path.insert(0, str(AG / 'o_stagger_tranche'))
import st_core as STC          # noqa: E402（只讀 import；在 P 補價之後 import）
sys.path.pop(0)
OFF = L.V4(YFX, Y5F, DMT)
ANCH = [pd.Period(d, 'M') for d in OFF.sig_days]
SEEDS = list(range(20)); JS = list(range(13))

class JV4(L.V4):
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

def job(a):
    strat, j, seed = a
    days, de, skip = days_for(j)
    kw = dict(gap=GAP, pfix=1.0) if strat == 'B10' else {}
    S = JV4(YFX, Y5F, DMT, unavail=(lambda d, _s=seed: STC.unavail_for(S.anchor[d], d, _s)), **kw)
    S.set_days(days, de)
    pl = S.plan(10, 'official')
    t0, t1 = K.IDX.get_loc(S.sig_days[0]), K.IDX.get_loc(de)
    nav, _ = L.sim_plan(pl, t0, t1, 'normal', XR)
    rows = [(strat, j, seed, str(p[3].date()), str(K.IDX[p[0]].date()), int(p[0]), ','.join(K.COLS[jj] for jj in p[1])) for p in pl]
    return strat, j, seed, nav.pct_change(), rows, str(de.date()), len(skip)

if __name__ == '__main__':
    T0 = time.time()
    with Pool(4) as P: out = P.map(job, [(s, j, sd) for s in ('AB', 'B10') for j in JS for sd in SEEDS], chunksize=2)
    rows = [r for o in out for r in o[4]]
    pd.DataFrame(rows, columns=['strat', 'j', 'seed', 'signal', 'exec', 'exec_idx', 'picks']).to_csv(HERE / 'data' / 'plans_j.csv', index=False)
    navs = {}; meta = {}
    E50 = K.E50
    for s in ('AB', 'B10'):
        for j in JS:
            R = pd.concat([o[3] for o in out if o[0] == s and o[1] == j], axis=1).mean(axis=1).fillna(0)
            nv = (1 + R).cumprod(); navs[f'{s}|j{j}'] = nv
            sd = sorted({pd.Timestamp(r[3]) for r in rows if r[0] == s and r[1] == j}) + [pd.Timestamp(next(o[5] for o in out if o[0] == s and o[1] == j))]
            x = L.excess(nv, E50.reindex(nv.index), sd, '2006-04-01', '2026-10-12')
            meta[f'{s}|j{j}'] = dict(t=float(L.tstat(x)), n=int(len(x)), end=next(o[5] for o in out if o[0] == s and o[1] == j),
                                    skipped=next(o[6] for o in out if o[0] == s and o[1] == j))
    pd.DataFrame(navs).to_parquet(HERE / 'data' / 'jscan_zero_navs.parquet')
    json.dump(dict(meta=meta, seeds=SEEDS, js=JS, runtime_s=time.time() - T0), open(HERE / 'data' / 'jscan_meta.json', 'w'), ensure_ascii=False, indent=1)
    print({k: round(v['t'], 3) for k, v in meta.items()})
