#!/usr/bin/env python3
"""S104 第 0 步：載入正式面板 → 精簡陣列快取（scratchpad/s104/cache/data.npz）；跑現行正式策略 E0｜W10_70（100 萬、嚴成本）
訊號位移 +0～+19 日共 20 條，存淨值與逐筆交易（K3 用）；驗收位移 0 與 navs_official.csv.gz 逐日差 0。
只數筆數／涵蓋率（data_check.json），不算任何新策略的報酬。用法：python3 -I -B prep.py"""
import sys, json, time, resource, gc
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import s104lib as L
t0 = time.time(); rss = lambda: round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
P, U, RT, M = L.load_full()
s10lib, cslib, COSTS = M['s10lib'], M['cslib'], M['COSTS']
print('load', round(time.time() - t0), 's', rss(), 'MB', flush=True)
for a in ('H', 'L', 'VOL', 'CHG', 'HF', 'VOLF', 'adjH', 'HI240', 'MA60', 'nbar', 'F_HI', 'IDXC', 'NEARHI', 'V60', 'liq'):
    if hasattr(P, a):
        delattr(P, a)
gc.collect()
MC = L.mcap_panel(P)
AMT = np.where(np.isfinite(P.AMT), P.AMT, 0.0)
AMT20 = pd.DataFrame(AMT).rolling(20, min_periods=20).mean().to_numpy()
# K1：每月 SURGE（遮掉 15 日期限公司看不到的值）× 位移
S, av = RT['S'], RT['av']
months = []; k0s = []
for m in S.index:
    k0 = av[m]
    if k0 >= P.T or k0 < 61:
        continue
    s = S.loc[m].to_numpy()
    if int((U[k0] & np.isfinite(s)).sum()) < 50:
        continue
    months.append(str(m)); k0s.append(int(k0))
NS = len(L.SHIFTS)
SG = np.full((NS, len(months), P.N), np.nan, np.float32); KK = np.full((NS, len(months)), -1, np.int32)
for mi, (m, k0) in enumerate(zip(months, k0s)):
    s = S.loc[pd.Period(m, 'M')].to_numpy()
    for si in L.SHIFTS:
        k = k0 + si
        if k >= P.T:
            continue
        SG[si, mi] = cslib.ins15_mask_row(P, pd.Period(m, 'M'), k, s)
        KK[si, mi] = k
np.savez(L.CACHE, O=P.O, CF=P.CF, R=P.R, REF=P.REF, LIMUP=P.LIMUP, LIMDN=P.LIMDN, G=P.G, AMT20=AMT20, MC=MC, U=U, FMA60=P.F_MA60,
         MK=P.MK, E50=P.E50, IDX=P.IDX.values.astype('datetime64[ns]'), COLS=P.COLS.astype(str), SG=SG, KK=KK, MONTHS=np.array(months), RET20=P.RET20)
print('cache', round(time.time() - t0), 's', rss(), 'MB', flush=True)
del AMT, AMT20, SG
gc.collect()
# 現行正式策略 20 條位移路徑
ref = pd.read_csv(L.OFFNAV, index_col=0, parse_dates=True)['E0|W10_70']
navs = {}; trades = []; chk = {}
for si in L.SHIFTS:
    Sg = s10lib.make_signal_s10(P, U, RT, shift=si)
    nav, lg, st = s10lib.simulate_s10(P, Sg, U, sat_w=0.7, sizing='W10', lock=False, cost=COSTS['strict'], capital=1e6, block=None)
    navs[si] = nav.reindex(P.IDX)
    x = lg[['j', 'sig_t', 'ent_t', 'exit_t', 'cost', 'proceeds', 'why']].copy(); x['shift'] = si
    trades.append(x)
    if si == 0:
        mine = nav.round(2)
        chk = dict(same_index=bool(mine.index.equals(ref.index)), max_abs_diff=float((mine - ref).abs().max()), n_trades=int(len(lg)))
    del Sg, nav, lg, st; gc.collect()
    print('shift', si, round(time.time() - t0), 's', rss(), 'MB', flush=True)
pd.DataFrame(navs).to_parquet(L.TMP / 'off_navs.parquet')
pd.concat(trades).to_parquet(L.TMP / 'off_trades.parquet')
# 資料檢查（只數）
idx = P.IDX
U_n = U.sum(1)
mc_cov = (np.isfinite(MC) & U).sum(1) / np.maximum(U_n, 1)
out = dict(official_shift0_vs_navs_official=chk, panel=dict(T=int(P.T), N=int(P.N), start=str(idx[0].date()), end=str(idx[-1].date())),
           n_signal_months=len(months), first_month=months[0], last_month=months[-1],
           universe_avg_n_2016on=float(U_n[idx >= '2016-01-01'].mean()), mcap_coverage_in_U_2016on_mean=float(mc_cov[idx >= '2016-01-01'].mean()),
           mcap_coverage_min=float(mc_cov[idx >= '2016-01-01'].min()),
           meta=dict(runtime_s=round(time.time() - t0), maxrss_mb=rss()))
(HERE / 'data_check.json').write_text(json.dumps(out, ensure_ascii=False, indent=1))
print(json.dumps(out, ensure_ascii=False))
assert chk['same_index'] and chk['max_abs_diff'] == 0, '現行策略重現失敗'
