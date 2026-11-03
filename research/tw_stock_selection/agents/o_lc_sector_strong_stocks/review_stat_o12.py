#!/usr/bin/env python3
"""反方審查（資料與統計角度）：只讀研究員檔案，不改任何檔。python3 -I -B review_stat_o12.py
1. 由 out/window_returns_key.parquet 重算可執行主欄（SEL、CTRL_SEL_PURE、CTRL_SEL）對 0050 的 j 中位數 t，核對 FINDINGS。
2. Newey-West t（lag 3）對簡單 t；13 個 j 之間的相關（有效樣本數）。
3. CTRL PURE 的年份集中度：拿掉 2021、拿掉 2021+2025、拿掉最好 3 年。
4. IND−CTRL 可執行配對差。
5. 存活者偏誤代理：每年 1 月訊號日、60 日均成交值前 100 中，沒有股數（SH 缺值）而被宇宙排除的檔數與其中日後下市的比例。
"""
import sys, math, json
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import numpy as np, pandas as pd
import lc_lib as X
X.set_mode('T1')

D = pd.read_parquet(HERE / 'out' / 'window_returns_key.parquet')
JS = range(13)
DAYS = {j: X.days_for(j, 'FULL') for j in JS}
WD = {j: pd.DatetimeIndex(DAYS[j][:-1]) for j in JS}
BEN = {j: pd.Series(X.bench_win(DAYS[j]), index=WD[j]) for j in JS}
PER = {'VAL': ('2017-01-01', '2026-12-31'), 'Y19': ('2019-01-01', '2026-12-31'), 'PIT': ('2014-01-01', '2026-12-31'), 'FULL': ('2006-04-01', '2026-12-31')}

def ser(col, j):
    s = D[col].dropna(); return s.reindex(WD[j])
def sl(s, p):
    a, b = PER[p]; return s[(s.index >= a) & (s.index <= b)]
def nw_t(x, L=3):
    x = np.asarray(x, float); x = x[np.isfinite(x)]; n = len(x); m = x.mean(); e = x - m
    g0 = (e @ e) / n; v = g0
    for l in range(1, L + 1):
        v += 2 * (1 - l / (L + 1)) * (e[l:] @ e[:-l]) / n
    return m / math.sqrt(v / n)

out = {}
for nm in ('SEL', 'SEL_PURE', 'CTRL_SEL', 'CTRL_SEL_PURE', 'IND_ONLY_PURE', 'B'):
    row = {}
    for p in PER:
        ts, nws, mus, cg, cb = [], [], [], [], []
        for j in JS:
            r = sl(ser(f'exec|{nm}|J{j}|100萬|GROW|normal', j), p).dropna(); b = BEN[j].reindex(r.index)
            x = (r - b).to_numpy(); ts.append(X.tstat(x)); nws.append(nw_t(x)); mus.append(x.mean() * 100)
            cg.append(X.cagr(r.to_numpy())); cb.append(X.cagr(b.to_numpy()))
        row[p] = dict(t=round(float(np.median(ts)), 2), t_NW3=round(float(np.median(nws)), 2), mean_pp=round(float(np.median(mus)), 3),
                      t_max_over_j=round(float(np.max(ts)), 2), CAGR=round(float(np.median(cg)) * 100, 1), CAGR0050=round(float(np.median(cb)) * 100, 1))
    out[nm] = row

# IND − CTRL（可執行，配對）
diff = {}
for a, b in (('SEL', 'CTRL_SEL'), ('SEL_PURE', 'CTRL_SEL_PURE')):
    for p in ('VAL', 'Y19', 'PIT'):
        ts, mus = [], []
        for j in JS:
            ra = sl(ser(f'exec|{a}|J{j}|100萬|GROW|normal', j), p); rb = sl(ser(f'exec|{b}|J{j}|100萬|GROW|normal', j), p)
            x = (ra - rb).dropna().to_numpy(); ts.append(X.tstat(x)); mus.append(x.mean() * 100)
        diff[f'{a}-{b}|{p}'] = dict(mean_pp=round(float(np.median(mus)), 3), t=round(float(np.median(ts)), 2))
out['IND_minus_CTRL_exec'] = diff

# 13 個 j 的相關（CTRL PURE 超額，PIT 段）
M = pd.concat([sl(ser(f'exec|CTRL_SEL_PURE|J{j}|100萬|GROW|normal', j) - BEN[j], 'PIT').rename(j).reset_index(drop=True) for j in JS], axis=1)
C = M.corr().to_numpy(); off = C[~np.eye(13, dtype=bool)]
out['j_corr_CTRL_PURE_PIT'] = dict(mean=round(float(off.mean()), 3), min=round(float(off.min()), 3))

# CTRL PURE 年份集中度（PIT 與 Y19）
conc = {}
for nm in ('CTRL_SEL_PURE', 'CTRL_SEL', 'SEL'):
    for p in ('PIT', 'Y19', 'VAL'):
        res = {}
        for lab, drop in (('base', []), ('ex2021', [2021]), ('ex2021_2025', [2021, 2025]), ('ex2025_2026', [2025, 2026])):
            ts = []
            for j in JS:
                r = sl(ser(f'exec|{nm}|J{j}|100萬|GROW|normal', j), p); x = (r - BEN[j].reindex(r.index)).dropna()
                x = x[~x.index.year.isin(drop)]; ts.append(X.tstat(x.to_numpy()))
            res[lab] = round(float(np.median(ts)), 2)
        # 拿掉最好 3 個日曆年（算術月超額年加總）
        ts = []
        for j in JS:
            r = sl(ser(f'exec|{nm}|J{j}|100萬|GROW|normal', j), p); x = (r - BEN[j].reindex(r.index)).dropna()
            yr = x.groupby(x.index.year).sum(); best = yr.sort_values(ascending=False).index[:3]
            ts.append(X.tstat(x[~x.index.year.isin(best)].to_numpy()))
        res['ex_best3'] = round(float(np.median(ts)), 2)
        # 各年超額加總（j 中位數，pp）
        yy = {}
        for j in JS:
            r = sl(ser(f'exec|{nm}|J{j}|100萬|GROW|normal', j), p); x = (r - BEN[j].reindex(r.index)).dropna()
            for y, v in x.groupby(x.index.year).sum().items(): yy.setdefault(int(y), []).append(v * 100)
        res['year_sum_pp'] = {y: round(float(np.median(v)), 1) for y, v in sorted(yy.items())}
        conc[f'{nm}|{p}'] = res
out['concentration'] = conc

# 存活者偏誤代理
Vv = X.L.V.reindex(X.IDX).to_numpy() if hasattr(X.L.V, 'reindex') else X.L.V
surv = []
last_idx = len(X.IDX) - 1
for y in range(2014, 2027):
    d = [x for x in DAYS[0][:-1] if x.year == y][0]
    t = X.IDX.get_loc(d)
    adv = np.nanmean(Vv[max(t - 59, 0):t + 1], axis=0)
    alive = X.L.TRADED[max(t - 5, 0):t + 1].any(axis=0) & np.isfinite(X.L.CFF[t]) & ~X.DRMASK
    adv = np.where(alive & np.isfinite(adv), adv, -np.inf)
    # 排除 ETF／受益憑證（0 開頭）
    etf = np.array([c.startswith('0') for c in X.COLS]); adv[etf] = -np.inf
    top = np.argsort(-adv)[:100]
    noshr = [i for i in top if not (np.isfinite(X.SH[t, i]) and X.SH[t, i] > 0)]
    later_dead = [i for i in noshr if X.L.LAST_TRADE[i] < last_idx - 5]
    # 有股數的 ADV 前 100 中日後下市的比例
    withshr = [i for i in top if i not in noshr]
    wd = [i for i in withshr if X.L.LAST_TRADE[i] < last_idx - 5]
    surv.append(dict(date=str(d.date()), n_noshares_in_ADVtop100=len(noshr), noshares_later_delisted=len(later_dead),
                     codes=[X.COLS[i] for i in noshr][:10], withshares_later_delisted=len(wd)))
out['survivorship_proxy'] = surv

# 宇宙內是否有任何日後下市股（每月平均）
dl = []
for d in DAYS[0][:-1]:
    if d < pd.Timestamp('2014-01-01'): continue
    cs = X.cs_cached(d); dl.append(int((X.L.LAST_TRADE[cs['U']] < last_idx - 5).sum()))
out['universe_later_delisted_per_month_PIT'] = dict(mean=round(float(np.mean(dl)), 2), max=int(np.max(dl)))

print(json.dumps(out, ensure_ascii=False, indent=1, default=str))
