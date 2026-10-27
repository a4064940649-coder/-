#!/usr/bin/env python3
"""反方審查（資料與統計）：只讀研究員的淨值檔與 run json，不修改任何研究員檔案。
python3 -I -B review_stat_o3.py"""
import sys, json, math
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'main_v2'))
import numpy as np, pandas as pd
import v2_core as V
K = V.K

R1 = json.loads((HERE / 'run_P1.json').read_text(encoding='utf-8'))['runs']
N1 = pd.read_parquet(HERE / 'navs_P1.parquet')
N0 = pd.read_parquet(HERE / 'navs_P0.parquet')
_S = V.V2('gapfix', *V.load_revenue()[:2], exfin=True)
_t0, _t1, sd = _S.bounds()   # 與 run.py 相同的訊號日（含 2026-10-12 終點）

def exs(nav, bench=None):
    nav = nav.dropna()
    D = pd.DatetimeIndex([d for d in sd if d in nav.index])
    nd = nav.reindex(D); pr = (nd.shift(-1) / nd - 1).iloc[:-1]
    b = K.E50 if bench is None else bench
    bd = b.reindex(D); bp = (bd.shift(-1) / bd - 1).iloc[:-1]
    x = (pr - bp).dropna(); return x[(x.index >= '2006-04-01') & (x.index <= '2026-10-12')]

def t(x): x = np.asarray(x, float); return x.mean() / (x.std(ddof=1) / math.sqrt(len(x)))
def nw(x, L):
    x = np.asarray(x, float); n = len(x); e = x - x.mean()
    g0 = (e @ e) / n; s = g0
    for l in range(1, L + 1):
        s += 2 * (1 - l / (L + 1)) * (e[l:] @ e[:-l]) / n
    return x.mean() / math.sqrt(s / n)

cols = {'v2': (N0, 'P0|R0|X0|Dv2|EXFIN|n10|normal'), 'V3': (N1, 'P1|R1|X1|Dhi|EXFIN|n10|normal'),
        'V3_lo': (N1, 'P1|R1|X1|Dlo|EXFIN|n10|normal'), 'B05': (N1, 'P1|R1|Dhi|EXFIN|B05|n10|normal'),
        'B10': (N1, 'P1|R1|Dhi|EXFIN|B10|n10|normal'), 'V3_stress': (N1, 'P1|R1|X1|Dhi|EXFIN|n10|stress')}
out = {}
X = {}
for k, (N, c) in cols.items():
    x = exs(N[c]); X[k] = x
    ac = [float(pd.Series(x.to_numpy()).autocorr(l)) for l in (1, 2, 3)]
    yr = x.groupby(x.index.year)
    # 逐年剔除：剔掉某一年後的 t
    drop1 = {int(y): t(x[x.index.year != y]) for y in yr.groups}
    ys = sorted(drop1, key=drop1.get)
    # 剔除貢獻最大的 k 個年份
    ysum = yr.sum().sort_values(ascending=False)
    dropk = {kk: t(x[~x.index.year.isin(ysum.index[:kk])]) for kk in (1, 2, 3)}
    # 剔除最大的 k 個月
    xs = x.sort_values(ascending=False)
    dropm = {kk: t(x.drop(xs.index[:kk])) for kk in (1, 3, 5, 10)}
    # 區塊 bootstrap（12 個月區塊）單尾 p
    rng = np.random.default_rng(0); a = x.to_numpy() - x.mean(); n = len(a); B = 20000; cnt = 0
    for _ in range(B):
        st = rng.integers(0, n, size=n // 12 + 1)
        s = np.concatenate([np.take(a, np.arange(s0, s0 + 12), mode='wrap') for s0 in st])[:n]
        if t(s) >= t(x): cnt += 1
    last = {'last_window_excess_pp': float(x.iloc[-1] * 100), 't_drop_last_window': t(x.iloc[:-1]), 't_drop_2021': t(x[x.index.year != 2021]),
            't_thru_2025': t(x[x.index < '2026-01-01']), 't_drop_2021_2026': t(x[~x.index.year.isin([2021, 2026])])}
    out[k] = {**last, 'n': len(x), 'mean_pp': float(x.mean() * 100), 't': t(x), 'NW3': nw(x, 3), 'NW6': nw(x, 6), 'NW12': nw(x, 12),
              'ac1_3': ac, 'top_years_by_excess_sum': {int(a_): float(b_) for a_, b_ in ysum.head(4).items()},
              't_drop_top_k_years': dropk, 't_drop_single_year_min': (ys[0], drop1[ys[0]]),
              't_drop_top_k_months': dropm, 'block_boot_p_one_sided': cnt / B,
              'frac_months_positive': float((x > 0).mean())}
# V3 − v2、B05 − V3 的配對月差
for a_, b_ in (('V3', 'v2'), ('B05', 'V3'), ('B10', 'V3'), ('V3_lo', 'V3')):
    d = (X[a_] - X[b_]).dropna()
    out[f'pair_{a_}-{b_}'] = {'mean_pp': float(d.mean() * 100), 't': t(d) if d.std() > 0 else None, 'n_diff_months': int((d.abs() > 1e-12).sum())}
# 多重檢定：Bonferroni / Sidak 單尾門檻
from statistics import NormalDist
nd_ = NormalDist()
out['hurdle_t_one_sided_5pct'] = {m: nd_.inv_cdf(1 - 0.05 / m) for m in (1, 5, 10, 20, 50, 100)}
# 若選版只是在同一資料上挑出最好的 m 個近似獨立規格之一，觀察到的 t 對應的 Bonferroni 調整 p
pV3 = 1 - nd_.cdf(out['V3']['t'])
out['V3_bonferroni_p'] = {m: min(1.0, pV3 * m) for m in (1, 5, 10, 20, 50)}
# 營收截斷：B05 的 n10 沒有任何缺營收股入選，卻掉 0.75pp → 純排名擾動
out['B05_gap_picks_n10'] = R1['P1|R1|Dhi|EXFIN|B05|n10|normal']['n_gap_picks']
out['B10_gap_picks_n10'] = R1['P1|R1|Dhi|EXFIN|B10|n10|normal']['n_gap_picks']
out['cal_t_V3'] = R1['P1|R1|X1|Dhi|EXFIN|n10|normal']['seg']['ALL']['t_0050_cal']
print(json.dumps(out, ensure_ascii=False, indent=1, default=float))
