#!/usr/bin/env python3
"""反方審查（資料與統計）：o_exec_capacity_tiers。只讀研究員的檔案，不寫回任何研究員檔案；結果印到 stdout。
檢查：(1) 主格可執行 AB 對 0050 的月超額：Newey-West t、分年貢獻、拿掉單一年／2006-07 後的 t；
     (2) 逐 seed t 分布（單一真實帳戶只會是一條路徑）；(3) 30 萬 vs 300 萬 FIX 配對差的年份集中度、制度期；
     (4) 區塊 bootstrap 的 t 信賴區間；(5) 多重檢定門檻。
用法：python3 -I -B review_stat_o5.py
"""
import sys, json, math
from pathlib import Path
HERE = Path(__file__).resolve().parent
import numpy as np, pandas as pd

B = pd.read_parquet(HERE / 'data' / 'bench.parquet')
P4 = pd.read_csv(HERE / 'data' / 'plan_v4.csv')
SD = [pd.Timestamp(x) for x in P4.signal] + [pd.Timestamp('2026-10-12')]
NA = pd.read_parquet(HERE / 'navs_abavg.parquet')
NM = pd.read_parquet(HERE / 'navs_main.parquet')
IDX = NA.index
E50 = B['E50'].reindex(IDX); AB = B['AB_V4|normal'].reindex(IDX); AB = AB / AB.iloc[0]

def win_ret(nav):
    d = pd.DatetimeIndex([x for x in SD if x in nav.index]); nd = nav.reindex(d)
    return (nd.shift(-1) / nd - 1).iloc[:-1]
R50 = win_ret(E50)
def exc(nav): return (win_ret(nav) - R50).dropna()
def t(x):
    x = np.asarray(x, float); return float(x.mean() / (x.std(ddof=1) / math.sqrt(len(x))))
def nw_t(x, L):
    x = np.asarray(x, float); n = len(x); e = x - x.mean()
    g0 = (e @ e) / n; s = g0
    for l in range(1, L + 1):
        s += 2 * (1 - l / (L + 1)) * (e[l:] @ e[:-l]) / n
    return float(x.mean() / math.sqrt(s / n))
def acf(x, k): x = np.asarray(x) - np.mean(x); return float((x[k:] @ x[:-k]) / (x @ x))
def block_boot_t(x, b=6, n=4000, rs=0):
    rng = np.random.default_rng(rs); x = np.asarray(x); N = len(x); out = []
    for _ in range(n):
        idx = np.concatenate([np.arange(s, s + b) % N for s in rng.integers(0, N, N // b + 1)])[:N]
        out.append(x[idx].mean())
    out = np.array(out); se = out.std(ddof=1)
    return float(x.mean() / se), [float(np.percentile(out, 5) * 1200), float(np.percentile(out, 95) * 1200)], float((out <= 0).mean())

OUT = {}
cols = {'ZI_AB': AB}
for tier in ('30萬', '100萬', '300萬', '1000萬'):
    for v in ('GROW', 'FIX'):
        cols[f'{tier}|{v}'] = NA[f'main|{tier}|{v}|A']
for k, nav in cols.items():
    x = exc(nav); yr = x.groupby(x.index.year)
    contrib = (yr.sum() / x.sum() * 100).round(1)
    loo = {int(y): round(t(x[x.index.year != y]), 2) for y in yr.groups}
    worst = min(loo, key=loo.get)
    o = dict(n=len(x), t=round(t(x), 3), NW3=round(nw_t(x, 3), 3), NW6=round(nw_t(x, 6), 3), NW12=round(nw_t(x, 12), 3),
             acf1=round(acf(x, 1), 3), acf2=round(acf(x, 2), 3), acf3=round(acf(x, 3), 3),
             t_ex2006_07=round(t(x[x.index >= pd.Timestamp('2008-01-01')]), 3),
             t_2006_07_only=round(t(x[x.index < pd.Timestamp('2008-01-01')]), 3),
             share_excess_2006_07_pct=round(float(x[x.index < pd.Timestamp('2008-01-01')].sum() / x.sum() * 100), 1),
             t_2014_2026=round(t(x[x.index >= pd.Timestamp('2014-01-01')]), 3),
             t_2019_2026=round(t(x[x.index >= pd.Timestamp('2019-01-01')]), 3),
             loo_min=(worst, loo[worst]), n_years_whose_removal_drops_t_below_2=int(sum(v < 2.0 for v in loo.values())),
             top3_year_contrib_pct=contrib.sort_values(ascending=False).head(3).to_dict(),
             boot6=block_boot_t(x, 6), boot12=block_boot_t(x, 12))
    OUT[k] = o

# 逐 seed：單一真實帳戶
SEED = {}
for tier in ('30萬', '100萬', '300萬', '1000萬'):
    for v in ('GROW', 'FIX'):
        ts = []; nws = []
        for s in range(40):
            nv = NM[f'{tier}|{v}|A|s{s:02d}']; x = exc(nv); ts.append(t(x)); nws.append(nw_t(x, 6))
        ts, nws = np.array(ts), np.array(nws)
        SEED[f'{tier}|{v}'] = dict(t_med=round(float(np.median(ts)), 3), frac_t_ge_2=float((ts >= 2).mean()),
                                   NW6_med=round(float(np.median(nws)), 3), frac_NW6_ge_2=float((nws >= 2).mean()))

# 30 萬 vs 300 萬 FIX 配對差（月）
def paired(a, b):
    d = (win_ret(a) - win_ret(b)).dropna(); return d
PD = {}
for v in ('FIX', 'GROW'):
    d = paired(NA[f'main|30萬|{v}|A'], NA[f'main|300萬|{v}|A'])
    yr = d.groupby(d.index.year)
    loo = {int(y): round(t(d[d.index.year != y]), 2) for y in yr.groups}
    PD[v] = dict(t=round(t(d), 3), NW6=round(nw_t(d, 6), 3), mean_pp_yr=round(float(d.mean() * 1200), 3),
                 yearly_sum_pp={int(y): round(float(g.sum() * 100), 2) for y, g in yr},
                 loo_min=min(loo.items(), key=lambda z: z[1]),
                 t_pre_2020_10=round(t(d[d.index < pd.Timestamp('2020-10-26')]), 3),
                 t_post_2020_10=round(t(d[d.index >= pd.Timestamp('2020-10-26')]), 3),
                 mean_pp_yr_post=round(float(d[d.index >= pd.Timestamp('2020-10-26')].mean() * 1200), 3),
                 t_ex2009=round(t(d[d.index.year != 2009]), 3))

# 多重檢定門檻（兩尾 5%，Bonferroni）與 Šidák
from statistics import NormalDist
thr = {N: round(NormalDist().inv_cdf(1 - 0.025 / N), 3) for N in (1, 5, 10, 20, 50, 100)}
it_self = json.loads((HERE / 'iterations.json').read_text(encoding='utf-8'))['n_versions']
v4it = json.loads((HERE.parent / 'o_v4_revenue_patch_official' / 'iterations.json').read_text(encoding='utf-8'))['n_versions']
n_agent_dirs = sum(1 for p in HERE.parent.iterdir() if p.is_dir())
print(json.dumps(dict(series=OUT, per_seed=SEED, paired_30v300=PD, bonferroni_t=thr,
                      n_versions_this=it_self, n_versions_o_v4=v4it, n_agent_dirs=n_agent_dirs), ensure_ascii=False, indent=1, default=str))
