#!/usr/bin/env python3
"""C50/C70、AB_V4 配對、端點、兩平點、其餘 F1 版本、檢定力、台積電。"""
import sys, json, math
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
from scipy import stats as sst
import rlib as L, rstrat as S
from r1_main import XS, run_f1, DATES

res = {}
navs = pd.read_parquet(L.OUT / 'navs_main.parquet')
F = navs['F|normal'].dropna()
wr = S.win_returns(F, DATES); b = S.bench_win(DATES); ex = (wr - b).dropna()

def mt(x): x = pd.Series(x).dropna(); return round(x.mean() * 100, 3), round(S.tstat(x), 2)

# ---- 核心＋衛星 ----
def core_sat(F, frac50):
    e = L.E50.reindex(F.index); dset = set(DATES)
    a = frac50; s0, s1 = a, 1 - a
    out = [1.0]
    for k in range(1, len(F)):
        s0 *= e.iloc[k] / e.iloc[k - 1]; s1 *= F.iloc[k] / F.iloc[k - 1]
        tot = s0 + s1
        if F.index[k] in dset:
            tr = abs(s1 - (1 - a) * tot)
            tot -= tr * (L.FEE + 0.001)
            s0, s1 = a * tot, (1 - a) * tot
        out.append(tot)
    return pd.Series(out, index=F.index)
for nm, a in (('C50', 0.5), ('C70', 0.7)):
    nv = core_sat(F, a)
    st = S.stats(nv, DATES)[0]
    res[nm] = {p: {k: round(v, 3) for k, v in st[p].items()} for p in ('SEL', 'VAL', 'POST2019')}

# ---- AB_V4 ----
v4 = pd.read_parquet(L.RAW / 'navs' / 'navs_v4.parquet')
ab = v4['AB_V4|normal'].reindex(L.IDX).dropna()
abw = S.win_returns(ab, DATES); abex = (abw - b).dropna()
res['AB_V4'] = {}
for p in ('SEL', 'VAL', 'POST2019', 'PRE'):
    x = S.sel(abex, p); te = x.std() * math.sqrt(12)
    d = S.sel((wr - abw).dropna(), p)
    res['AB_V4'][p] = dict(ex=mt(x), IR=round(x.mean() * 12 / te, 2), TE=round(te * 100, 1), F_minus_AB=mt(d))
# 其他對照組
v6 = pd.read_parquet(L.RAW / 'navs' / 'navs_v6.parquet')
for nm, src in (('B10', v4['AB_V4_B10|normal']), ('AB_V6', v6['AB_V6|normal']), ('EW', v4['AB_V4|EW'])):
    w2 = S.win_returns(src.reindex(L.IDX).dropna(), DATES)
    res['ctrl_' + nm] = {p: mt(S.sel((w2 - b).dropna(), p)) for p in ('VAL', 'POST2019')}
# β 對 (0050 − 同池等權)
ew = S.win_returns(v4['AB_V4|EW'].reindex(L.IDX).dropna(), DATES)
dd = (wr - abw); fac = (b - ew)
for p in ('PRE', 'POST2019'):
    y = S.sel(dd, p).dropna(); xx = S.sel(fac, p).reindex(y.index)
    res.setdefault('beta_diff_on_0050_minus_EW', {})[p] = round(float(np.polyfit(xx, y, 1)[0]), 2)

# ---- 端點 ----
v = S.sel(ex, 'VAL'); pst = S.sel(ex, 'POST2019')
res['endpoints'] = {f'VAL_drop_best{k}': mt(v.sort_values().iloc[:-k]) for k in (1, 3, 5)}
res['endpoints']['POST_drop_best5'] = mt(pst.sort_values().iloc[:-5])
res['endpoints']['VAL_drop_2023'] = mt(v[v.index >= '2024-01-01'])
nxt = pd.Series(DATES[1:], index=DATES[:-1])
s3a, s3b = pd.Timestamp('2024-02-15'), pd.Timestamp('2025-09-23')
keep = [d for d in v.index if not (d <= s3b and nxt[d] >= s3a)]
res['endpoints']['VAL_ex_src3'] = mt(v[keep]); res['endpoints']['n_ex_src3'] = len(keep)

# ---- 檢定力 ----
sd = v.std(); n = len(v)
res['power'] = dict(MDE80_pp=round((sst.norm.ppf(0.975) + sst.norm.ppf(0.8)) * sd / math.sqrt(n) * 100, 3),
                    p1_t=round(float(1 - sst.t.cdf(S.tstat(v), n - 1)), 4))

# ---- 兩平點（單邊滑價） ----
W, _ = run_f1()
REPW = {d: XS[d].R for d in DATES[:-1]}
def slip_costs(s):
    return lambda kind: (L.FEE + s, L.FEE + 0.003 + s)
orig = L.costs
be = {}
grid = [0.001, 0.005, 0.01, 0.015]
for s in grid:
    L.costs = slip_costs(s)
    nv, _ = S.simulate(W, 'x'); w_ = S.win_returns(nv, DATES)
    be[s] = {p: S.sel((w_ - b).dropna(), p).mean() * 100 for p in ('VAL', 'POST2019')}
L.costs = orig
res['slip_grid'] = {str(k): {p: round(x, 3) for p, x in vv.items()} for k, vv in be.items()}
for p in ('VAL', 'POST2019'):
    xs = np.array(grid); ys = np.array([be[s][p] for s in grid])
    res.setdefault('breakeven_slip_pct', {})[p] = round(float(np.interp(0, ys[::-1], xs[::-1]) * 100), 2)

# ---- 其餘 F1 版本 ----
res['F1_versions'] = {}
for K, N, wm in ((1, 10, 'eq'), (5, 10, 'eq'), (10, 10, 'eq'), (10, 20, 'sqrt'), (5, 20, 'eq'), (10, 20, 'eq'), (1, 20, 'eq')):
    Wv, _ = run_f1(K, N, wm)
    nv, _ = S.simulate(Wv, 'normal'); w_ = S.win_returns(nv, DATES)
    res['F1_versions'][f'F1_K{K}_N{N}_{wm}'] = {p: mt(S.sel((w_ - b).dropna(), p)) for p in ('SEL', 'VAL')}

# ---- 台積電 ----
tj = L.CI['2330']
act = [W[d][tj] - XS[d].R[tj] for d in DATES[:-1]]
res['tsmc_active_max_abs'] = float(np.max(np.abs(act)))
w50 = pd.read_csv(L.RAW / 'screener' / 'etf0050_w.csv', dtype={'code': str})
diffs = {}
for asof in sorted(set(w50['asof'])):
    a = w50[(w50['asof'] == asof) & (w50.code == '2330')].weight.iloc[0]
    dlast = [d for d in DATES[:-1] if d <= pd.Timestamp(asof)][-1]
    diffs[asof] = round(W[dlast][tj] * 100 - a, 2)
res['tsmc_w_minus_actual_pp'] = diffs

json.dump(res, open(L.OUT / 'r2_extra.json', 'w'), ensure_ascii=False, indent=1, default=float)
print(json.dumps(res, ensure_ascii=False, indent=1, default=float))
