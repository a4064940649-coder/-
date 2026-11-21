#!/usr/bin/env python3
"""S86 驗收：先重現正式 48.76%／381 筆；引擎副本全預設與原引擎逐位相同；只記錄股利（稅 0、延後 0）時淨值逐位相同；
0050 基準函式全 0 時＝P.E50；0050 除息事件日的還原報酬與事件表一致。輸出 accept.json。"""
import json, time, resource, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
import s86lib as L
from s86lib import s10lib, stlib, COSTS
t0 = time.time()
P, U, SG = L.setup((0,))
Sg = SG[0]
kw = dict(sat_w=0.7, sizing='W10', lock=False, cost=COSTS['strict'], capital=1e6)
nav, lg, st = s10lib.simulate_s10(P, Sg, U, **kw)
D0, DE = P.IDX[P.t0], P.IDX[-1]; D19 = P.IDX[P.IDX < pd.Timestamp('2019-01-01')][-1]
c19, cf = stlib.cagr(nav, D19, DE), stlib.cagr(nav, D0, DE); ncl = int((lg['why'] != 'open_end').sum())
site = pd.read_csv('/home/user/-/xq_scripts/site/navs_official.csv.gz', index_col=0, parse_dates=True)
d_site = float((nav.loc[D0:].round(2) - site['E0|W10_70']).abs().max())
nav2, lg2, st2 = L.simulate_s86(P, Sg, U, **kw)
DY, DYM, NQ, dy50, e50ev, info = L.div_arrays(P)
nav3, lg3, st3 = L.simulate_s86(P, Sg, U, DY=DY, dy50=dy50, EXPY=L.expy(P, DY), **kw)
b0, _, _ = L.bench50(P, dy50)
bref = pd.Series(P.E50, index=P.IDX).loc[D0:]; bref = bref / bref.iloc[0] * 1e6
# 0050 除息日：還原序列當天報酬 應 ≈ 原始收盤 ÷ 參考價
e = pd.read_csv(L.CACHE / 'e50_ohlc.csv', parse_dates=['date']).set_index('date')
chk = []
for d, v, pc in e50ev:
    t = P.IDX.get_loc(pd.Timestamp(d)); chk.append(dict(date=d, value=v, pre_close=pc, tr_ret=float(P.E50[t] / P.E50[t - 1] - 1),
                                                         raw_close_over_ref=float(e['close'].reindex(P.IDX).iloc[t] / (pc - v) - 1) if 'close' in e else None))
out = dict(official_since2019=c19, official_full=cf, n_closed=ncl, nav_vs_site_max_abs=d_site,
           copy_default_identical=bool(np.array_equal(nav.to_numpy(), nav2.to_numpy(), equal_nan=True) and len(lg) == len(lg2)),
           copy_divrecord_identical=bool(np.array_equal(nav.to_numpy(), nav3.to_numpy(), equal_nan=True)),
           bench_zero_vs_E50_max_rel=float(np.nanmax(np.abs(b0.loc[D0:] / bref - 1))),
           e50_since2019=stlib.cagr(b0, D19, DE), e50_full=stlib.cagr(b0, D0, DE), div_info=info, e50_event_check=chk,
           secs=round(time.time() - t0), peak_mb=round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024))
out['pass'] = bool(abs(c19 - L.ACC['since2019']) < 1e-12 and ncl == 381 and d_site == 0 and out['copy_default_identical'] and out['copy_divrecord_identical'] and out['bench_zero_vs_E50_max_rel'] < 1e-12)
json.dump(out, open('accept.json', 'w'), ensure_ascii=False, indent=1, default=str)
print(json.dumps({k: v for k, v in out.items() if k != 'e50_event_check'}, ensure_ascii=False, default=str))
for c in chk: print(c)
