#!/usr/bin/env python3
"""獨立複製（只依 ../frozen_spec.txt 第一～三、五節 P1 重寫；不 import core.py、不讀 x_cache.npz、不讀研究員任何結果檔）。
資料直接讀 V7 面板 parquet＋P 下市價、官方名單、0050、加權指數。輸出 replicate/rep_results.json。
用法：python3 -I -B replicate/rep.py"""
import json, math
from pathlib import Path
import numpy as np, pandas as pd

AG = Path(__file__).resolve().parents[2]
PN = AG / 'o_v7_listing_exec_engine' / 'data'
cl = pd.read_parquet(PN / 'close_raw.parquet'); idx = cl.index; cols = cl.columns
rt = pd.read_parquet(PN / 'ret.parquet').reindex(index=idx, columns=cols)
vl = pd.read_parquet(PN / 'value.parquet').reindex(index=idx, columns=cols)
dp = pd.read_parquet(AG / 'o_infra_v3_rebuild' / 'delist_prices.parquet')
dp = dp[dp['code'].isin(cols)]
for k, src in (('close', cl), ('ret', rt), ('value', vl)):
    w = dp.pivot_table(index='date', columns='code', values=k, aggfunc='last').reindex(index=idx, columns=cols)
    src.update(w.where(src.isna()))    # 只補缺值格
has = cl.notna()
# 前一日 60 日均值：只用有值檔數 ≥ 300 的日子滾動，再對齊
good = has.sum(axis=1) >= 300
v60 = vl[good].rolling(60, min_periods=60).mean().reindex(idx).ffill(limit=3)
v60p = v60.shift(1); pcl = cl.ffill().shift(1)
elig = (v60p >= 1e7) & (pcl >= 10)
ew = rt.where(elig & has).mean(axis=1).fillna(0.0)
lew = np.log1p(ew.clip(lower=-0.99)).cumsum().to_numpy()
la = np.log1p(rt.fillna(0.0).clip(lower=-0.99)).cumsum().to_numpy()
HAS = has.to_numpy(); RN = rt.to_numpy(); V60P = v60p.to_numpy(); PCL = pcl.to_numpy()
last = np.array([np.flatnonzero(HAS[:, j]).max() if HAS[:, j].any() else -1 for j in range(len(cols))])
tx = pd.read_csv('/home/user/jonatw/finlab-txd/data/raw/taiex_twii.csv', parse_dates=['date']).set_index('date')['close'].sort_index()
bear = tx < tx.rolling(200, min_periods=200).mean()
def is_bear(d):
    s = bear[bear.index < d]
    return bool(s.iloc[-1]) if len(s) else False
CI = {c: j for j, c in enumerate(cols)}
lst = pd.read_csv(AG / 'o_disposal_rejudge' / 'disposal_hist_rejudge.csv', dtype={'code': str}, parse_dates=['announce', 'start', 'end'])
lst['o'] = range(len(lst)); lst['name'] = lst['name'].fillna('')
lst = lst[(lst.code.str.len() == 4) & ~lst.code.str.startswith('0') & ~lst.name.str.contains('DR')].copy()
lst['s'] = idx.searchsorted(lst.start.values); lst['e'] = idx.searchsorted(lst.end.values, side='right') - 1
lst = lst.sort_values(['code', 'start', 'announce', 'o'])
# second：先前記錄 e ≥ s−30 且 < s
sec = {}; prev = {}
for r in lst.itertuples():
    p = prev.get(r.code)
    sec[r.o] = p is not None and r.s - 30 <= p < r.s
    prev[r.code] = r.e if p is None else max(p, r.e)
# R1 鏈
take = {}; lastend = {}
for r in lst.itertuples():
    le = lastend.get(r.code)
    if le is not None and r.start <= le + pd.Timedelta(days=3): take[r.o] = False
    else: take[r.o] = True; lastend[r.code] = r.end
T = len(idx); rows = []; holdto = {}
for r in lst.sort_values(['s', 'announce', 'o']).itertuples():
    if not take[r.o] or r.code not in CI or r.s >= T or r.e <= r.s: continue
    j = CI[r.code]; s = r.s
    if holdto.get(r.code, -1) >= s: continue
    if is_bear(idx[s]): continue
    lim = 0.07 if idx[s] < pd.Timestamp('2015-06-01') else 0.10
    if not HAS[s, j] or not np.isfinite(RN[s, j]) or RN[s, j] >= lim - 0.01: continue
    if not (V60P[s, j] >= 1e7 and PCL[s, j] >= 10): continue
    k = min(r.e, last[j]); stop = min(T - 1, last[j])
    while k < stop:
        lk = 0.07 if idx[k] < pd.Timestamp('2015-06-01') else 0.10
        if HAS[k, j] and np.isfinite(RN[k, j]) and RN[k, j] > -(lk - 0.01): break
        k += 1
    holdto[r.code] = k
    raw = math.expm1(la[k, j] - la[s, j]); b = math.expm1(lew[k] - lew[s]); c = 0.012 if sec[r.o] else 0.008
    rows.append(dict(date=idx[s], code=r.code, raw=raw, gross=raw - b, net=raw - b - c, regime=r.regime))
D = pd.DataFrame(rows)
def S(x, col):
    m = x.groupby(x.date.dt.to_period('M'))[col].mean()
    return {'n': int(len(x)), 'mean_pct': round(float(x[col].mean() * 100), 3), 't_month': round(float(m.mean() / m.std(ddof=1) * math.sqrt(len(m))), 3)}
out = {}
for nm, a, b in (('OOS', '2006-04-01', '2021-09-30'), ('SEL', '2021-10-01', '2023-12-31'), ('DESC', '2024-01-01', '2026-10-22'), ('FULL', '2006-04-01', '2026-10-22')):
    x = D[(D.date >= a) & (D.date <= b)]
    out[nm] = {'net': S(x, 'net'), 'gross': S(x, 'gross')}
(Path(__file__).resolve().parent / 'rep_results.json').write_text(json.dumps(out, ensure_ascii=False, indent=1))
D.to_csv(Path(__file__).resolve().parent / 'rep_trades.csv.gz', index=False)
print(json.dumps(out, ensure_ascii=False))
