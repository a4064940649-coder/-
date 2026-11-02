#!/usr/bin/env python3
"""軌 B 延伸（自寫，口徑依凍結規格文字自行解讀）：零衝擊引擎加兩個執行細節，官方日 AB／B10 40 seeds。
 disp：換股日賣出（全賣或減碼）若成交日落在官方處置期間（start≤e≤end），賣出部分以當日收盤減一跳成交（其餘照開盤）。
 mdelay：X 規則 type＝cash 的持股，自下市日起以 0050 含息計值；撥款日＝下市日＋20 個面板交易日；撥款後第一個換股日才再投入。
比較對象：研究員 navs_ideal7.parquet 的 OFF 欄（base／disp／mdelay／both）。"""
import sys, json
from pathlib import Path
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
import numpy as np, pandas as pd
import rb_core as B
from multiprocessing import Pool
E50n = B.E50.to_numpy()
EM = pd.read_csv(B.AG / 'o_infra_v3_rebuild' / 'merger_consideration.csv', dtype={'code': str})
EM['delist_date'] = pd.to_datetime(EM['delist_date'], errors='coerce')
CASHJ = {}
for r in EM.itertuples():
    j = B.CI.get(r.code)
    if j is not None and r.type == 'cash' and j in B.XR and pd.notna(r.delist_date):
        idd = int(np.searchsorted(B.IDX, r.delist_date)); CASHJ[j] = (idd, idd + 20)

def dispon(t):
    x = np.datetime64(B.IDX[t]); m = np.zeros(B.N, bool); m[B.DJ[(B.DS <= x) & (B.DE >= x)]] = True; return m

def simulate(pl, b, s, t0, t1, disp=False, mdelay=False):
    reb = {p['e']: p['picks'] for p in pl}
    val = np.zeros(B.N); cash = 1.0; nav = np.empty(t1 - t0 + 1); pend = {}
    for t in range(t0, t1 + 1):
        if t in reb:
            pk = reb[t]; tr = B.TR[t] & np.isfinite(B.On[t])
            h = val * np.where(tr, B.G[t] * B.On[t] / np.where(tr, B.Cn[t], 1.0), 1.0)
            frz = (h > 0) & ~tr
            for j in np.where(frz)[0]:
                if j in pend:
                    if t >= pend[j][2]: cash += h[j]; h[j] = 0.0; frz[j] = False; del pend[j]
                    continue
                if j in B.XR and B.XR[j][1] <= t < B.XR[j][2] and not (mdelay and j in CASHJ):
                    cash += h[j] * B.XR[j][0]; h[j] = 0.0; frz[j] = False
            htr = np.where(frz, 0.0, h)
            rho = np.ones(B.N)
            if disp:
                dm = dispon(t) & tr & (htr > 0)
                if dm.any():
                    c = B.Cn[t]; rho = np.where(dm, (c - B.tick(c)) / np.where(tr, B.On[t], 1.0), 1.0)
            inpk = np.zeros(B.N, bool); inpk[pk] = True
            def gap(T):   # 現金需求 − 現金；T 越大越大
                tg = np.where(inpk, T, 0.0); d = tg - htr
                buy = np.clip(d, 0, None); sell = np.clip(-d, 0, None)
                return (buy * (1 + b)).sum() - (sell * rho * (1 - s)).sum() - cash
            lo, hi = 0.0, (htr.sum() * max(rho.max(), 1.0) + cash) / len(pk)
            for _ in range(80):
                mid = 0.5 * (lo + hi)
                if gap(mid) > 0: hi = mid
                else: lo = mid
            T = lo; cash = -gap(T)
            nv = np.where(frz, h, 0.0); nv[pk] += T
            val = np.where(frz, nv, nv * np.where(tr, B.Cn[t] / np.where(tr, B.On[t], 1.0), 1.0))
        elif t > t0:
            val = val * B.G[t]
        if mdelay:
            for j, (idd, ipay) in CASHJ.items():
                if val[j] > 0 and j not in pend and t >= idd and B.XR[j][2] > t and not B.TR[t, j]:
                    pend[j] = (val[j] * B.XR[j][0], idd, ipay)
            for j, (v0, idd, ipay) in pend.items():
                val[j] = v0 * E50n[t] / E50n[idd]
        nav[t - t0] = val.sum() + cash
    return pd.Series(nav, index=B.IDX[t0:t1 + 1])

SD = B.official_days(); T0 = B.TPOS[SD[0]]; T1 = B.TPOS[B.END]
def job(a):
    k, sd = a
    pl, _ = B.plan(SD, B10=(k == 'B10'), unavail=B.unavail_mc(sd)); bb, ss = B.rates('normal')
    return k, sd, {m: simulate(pl, bb, ss, T0, T1, disp=m in ('disp', 'both'), mdelay=m in ('mdelay', 'both')).pct_change()
                   for m in ('base', 'disp', 'mdelay', 'both')}
if __name__ == '__main__':
    with Pool(4) as P: res = P.map(job, [(k, s) for k in ('AB', 'B10') for s in range(40)])
    I = pd.read_parquet(HERE.parent / 'navs_ideal7.parquet'); out = {}
    for k in ('AB', 'B10'):
        nv = {m: (1 + pd.concat([r[2][m] for r in res if r[0] == k], axis=1).mean(axis=1).fillna(0)).cumprod() for m in ('base', 'disp', 'mdelay', 'both')}
        cg = {m: B.cagr_mdd(n, '2006-04-01', '2026-10-12')[0] for m, n in nv.items()}
        rc = {m: B.cagr_mdd(I[f'{k}|OFF|{m}'].dropna(), '2006-04-01', '2026-10-12')[0] for m in nv}
        dif = {m: float((I[f'{k}|OFF|{m}'].dropna() / I[f'{k}|OFF|{m}'].dropna().iloc[0] - nv[m].reindex(I[f'{k}|OFF|{m}'].dropna().index)).abs().max()) for m in nv}
        out[k] = dict(mine_drag_pp={m: round(cg[m] - cg['base'], 4) for m in nv}, researcher_drag_pp={m: round(rc[m] - rc['base'], 4) for m in nv},
                      base_cagr=round(cg['base'], 4), researcher_base=round(rc['base'], 4), nav_maxdiff=dif)
    json.dump(out, open(HERE / 'out' / 'ideal.json', 'w'), indent=1); print(json.dumps(out, indent=1))
