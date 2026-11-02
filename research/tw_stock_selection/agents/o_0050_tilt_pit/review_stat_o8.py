#!/usr/bin/env python3
"""反方審查（資料與統計，o8）：只讀研究員檔案，不改任何檔。輸出印到 stdout。
1. 由 data/oos_series.parquet 重算 j 中位數主欄（對 B、對 0050、對隨機）。
2. 合併 2014-01～2026-10 的對隨機 t（挑選期＋樣本外，檢查「機制不支持」是否過嚴或過寬）。
3. 逐年貢獻、刪一年 t、j 之間相關（13 個 j 是否只是一個樣本）。
4. 訊號覆蓋：每月層內可傾斜檔數（z≠0）、整月無傾斜的月份。
5. 股數前視敏感度：市值改用落後 63 個交易日的 PIT 股數（模擬季報公告延遲），重跑 T1/T2 對 B、對隨機。
"""
import sys, math, json
from pathlib import Path
HERE = Path('/home/user/-/research/tw_stock_selection/agents/o_0050_tilt_pit')
sys.path.insert(0, str(HERE))
import numpy as np, pandas as pd
SER = pd.read_parquet(HERE / 'data' / 'oos_series.parquet')
JS = list(range(13))
def tt(x):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    return float(x.mean() / (x.std(ddof=1) / math.sqrt(len(x)))) if len(x) > 2 else float('nan')
def nw_t(x, L=3):
    x = np.asarray(x, float); x = x[np.isfinite(x)]; n = len(x); u = x - x.mean()
    v = (u @ u) / n
    for l in range(1, L + 1): v += 2 * (1 - l / (L + 1)) * (u[l:] @ u[:-l]) / n
    return float(x.mean() / math.sqrt(v / n))
PER = {'SEL': ('2014-01-01', '2018-12-31'), 'POST2019': ('2019-01-01', '2026-10-31'), 'ALL': ('2014-01-01', '2026-10-31')}
out = {}
for v in ('T1', 'T2'):
    for pn, (a, z) in PER.items():
        rows = []
        for j in JS:
            df = SER[f'{v}|{j}'].dropna(how='all'); df = df[(df.index >= a) & (df.index <= z)]
            xB = df['T'] - df['B']; x50 = df['T'] - df['E50']; xr = df['T_g'] - df['PLC_g']
            rows.append({'j': j, 'n': len(df), 'vsB_pp': xB.mean() * 100, 'vsB_t': tt(xB), 'vsB_nw': nw_t(xB),
                         'vs50_pp': x50.mean() * 100, 'vs50_t': tt(x50), 'rand_pp': xr.mean() * 100, 'rand_t': tt(xr), 'rand_nw': nw_t(xr)})
        R = pd.DataFrame(rows)
        out[(v, pn)] = R
        print(v, pn, 'n', int(R.n.median()), {c: round(float(R[c].median()), 3) for c in R.columns if c not in ('j', 'n')},
              'rand_t range', round(R.rand_t.min(), 2), round(R.rand_t.max(), 2), flush=True)
# 3. 逐年
for v in ('T1', 'T2'):
    for col, nm in [(('T', 'B'), 'vsB'), (('T', 'E50'), 'vs50'), (('T_g', 'PLC_g'), 'rand')]:
        Y = []
        for j in JS:
            df = SER[f'{v}|{j}'].dropna(how='all'); df = df[df.index >= '2019-01-01']
            x = df[col[0]] - df[col[1]]; Y.append((x.groupby(x.index.year).sum() * 100))
        Y = pd.concat(Y, axis=1).median(axis=1).round(2)
        # 刪一年 t（j 中位數）
        loo = {}
        for yy in range(2019, 2027):
            ts = []
            for j in JS:
                df = SER[f'{v}|{j}'].dropna(how='all'); df = df[(df.index >= '2019-01-01') & (df.index.year != yy)]
                ts.append(tt(df[col[0]] - df[col[1]]))
            loo[yy] = round(float(np.median(ts)), 2)
        print(v, nm, 'POST2019 年度加總 pp（j 中位數）', Y.to_dict(), '刪一年 t', loo, flush=True)
# j 之間相關
for v in ('T1', 'T2'):
    M = pd.concat({j: (lambda d: (d['T_g'] - d['PLC_g']).groupby(d.index.to_period('M')).first())(SER[f'{v}|{j}'].dropna(how='all')) for j in JS}, axis=1)
    M = M[M.index >= pd.Period('2019-01', 'M')]
    c = M.corr().to_numpy(); print(v, 'j 間對隨機序列相關：相鄰 j 平均', round(float(np.mean([c[i, i + 1] for i in range(12)])), 3), 'j0 vs j12', round(float(c[0, 12]), 3), flush=True)

# 4/5 需要 tcore
import tcore as T, tlib as L
def coverage(kind, a, z):
    ds = T.sig_days('off', a, z); pre = T.precompute(ds, kind)
    nz = []; zero_months = []
    for d, U, wb, tz in pre:
        k = sum(int((zz != 0).sum()) for nm, (idx, zz) in tz.items())
        tot = sum(len(idx) for nm, (idx, zz) in tz.items())
        nz.append(k / max(tot, 1));
        if k == 0: zero_months.append(str(d.date()))
    return float(np.mean(nz)), float(np.min(nz)), zero_months
for kind in ('T1', 'T2'):
    m, mn, zm = coverage(kind, '2019-01-01', '2026-10-31')
    print(kind, '官方日 POST2019 可傾斜股中 z≠0 比例 平均', round(m, 3), '最低', round(mn, 3), '整月無傾斜', zm, flush=True)

def jmed_stats(kind, lam=1.5, cap=False, a='2019-01-01', z='2026-10-31', P=200):
    rs = []
    for j in JS:
        ds = T.sig_days(j, a, z); pre = T.precompute(ds, kind)
        PB = {d: (u, w) for d, u, w, tz in pre}; PT = {}
        for d, U, wb, tz in pre:
            wt = wb.copy()
            for nm, (idx, zz) in tz.items(): wt[idx] = T.tilt(wb[idx], zz, lam, cap)
            PT[d] = (U, wt)
        rT, _ = T.run(ds, PT); rB, _ = T.run(ds, PB); b = T.bench_win(ds)
        Rp, Rb = T.gross_matrix(ds, kind, lam, cap, P=P, seed=4242 + j)
        Rr, _ = T.gross_matrix(ds, kind, lam, cap, real=True)
        y = Rr[0] - Rp.mean(0)
        rs.append((tt(rT - rB), (rT - rB).mean() * 100, tt(rT - b), (rT - b).mean() * 100, tt(y), y.mean() * 100, tt(rB - b), (rB - b).mean() * 100))
    A = np.array(rs); md = np.median(A, axis=0)
    return dict(vsB_t=md[0], vsB_pp=md[1], vs50_t=md[2], vs50_pp=md[3], rand_t=md[4], rand_pp=md[5], B50_t=md[6], B50_pp=md[7])
SH0 = T.SH.copy()
for kind in ('T1', 'T2'):
    T._PRE.clear(); T.SH = SH0
    base = jmed_stats(kind)
    lag = 63
    SHL = np.full_like(SH0, np.nan); SHL[lag:] = SH0[:-lag]
    # 新上市股前 63 日沒有落後值 → 用當時值（只影響名次邊緣）
    SHL = np.where(np.isfinite(SHL), SHL, SH0)
    T.SH = SHL; T._PRE.clear()
    lagr = jmed_stats(kind)
    T.SH = SH0; T._PRE.clear()
    print(kind, 'PIT 原始', {k: round(float(v), 3) for k, v in base.items()}, flush=True)
    print(kind, '股數落後 63 日', {k: round(float(v), 3) for k, v in lagr.items()}, flush=True)
