#!/usr/bin/env python3
"""反方審查（資料與統計角度，o8）：只讀研究員輸出，不改任何檔案。
1. 重算 R4 正式表 8 格（確認 j 中位數年化、超額、t）
2. 年份集中度：全期超額去掉 2006、去掉最佳兩年；2019 年起去掉 2024
3. 13 個 j 的視窗超額相關 → 有效獨立數；j 中位數超額為正的 j 數
4. 「輸」判定的邊界：300萬 GROW、1000萬 FIX 在 R4／R4c／R2c 與引擎路徑雜訊（R0 對原表逐 j 差）比較
5. 區塊 bootstrap（12 窗一塊）全期超額 CI
"""
import sys, json
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import x_stats as S

OUT = HERE / 'out'
R4 = pd.read_parquet(OUT / 'navs_oos_R4.parquet')
R4c = pd.read_parquet(OUT / 'navs_oos_R4c.parquet')
R0 = pd.read_parquet(OUT / 'navs_oos_R0.parquet')
ORIG = pd.read_parquet(HERE.parent / 'o_v7_listing_exec_engine' / 'navs_exec7.parquet')
E50 = S.E50
CELLS = [(t, v) for t in ('30萬', '100萬', '300萬', '1000萬') for v in ('FIX', 'GROW')]
rep = {}


def col(df, j, t, v, m='K5T', c='normal', lab='B10'):
    return df[f'{lab}|J{j}|{m}|{c}|{t}|{v}']


def ann_chain(nav, e, years_drop=(), start=None):
    """以日曆年報酬串接（可去掉某些年），回傳策略與 0050 年化差（pp）。"""
    nav = nav.loc[:e].dropna()
    if start is not None: nav = nav[nav.index >= nav.index[nav.index < start][-1]] if (nav.index < start).any() else nav
    b = E50.reindex(nav.index)
    gs, gb, days = 1.0, 1.0, 0
    for y in sorted(set(nav.index.year)):
        if y in years_drop: continue
        s = nav[nav.index.year == y]
        prev = nav[nav.index < s.index[0]]
        a = prev.index[-1] if len(prev) else s.index[0]
        gs *= nav[s.index[-1]] / nav[a]; gb *= b[s.index[-1]] / b[a]; days += (s.index[-1] - a).days
    yrs = days / 365.25
    return (gs ** (1 / yrs) - gb ** (1 / yrs)) * 100


# 1 + 2 + 3
for t, v in CELLS:
    per = []; W = {}
    for j in range(13):
        n = col(R4, j, t, v); e = S.end_day('AB_V7_B10', j)
        s = S.stats(n, 'AB_V7_B10', j)
        W[j] = S.windows(n.loc[:e].dropna(), 'AB_V7_B10', j)
        ye = S.yearly_exc(n, 'AB_V7_B10', j)
        best2 = sorted([y for y in ye if y <= 2026], key=lambda y: -ye[y])[:2]
        per.append(dict(exc=s['exc'], t=s['t_0050'], exc_Y19=s['exc_Y19on'],
                        exc_no2006=ann_chain(n, e, years_drop=(2006,)),
                        exc_no_best2=ann_chain(n, e, years_drop=tuple(best2)), best2=best2,
                        exc_Y19_no2024=ann_chain(n, e, years_drop=(2024,), start=pd.Timestamp('2019-01-01')),
                        exc_Y19_no2024_2026=ann_chain(n, e, years_drop=(2024, 2026), start=pd.Timestamp('2019-01-01'))))
    med = lambda k: float(np.median([p[k] for p in per]))
    # 視窗超額跨 j 相關（以月對齊）
    M = pd.DataFrame({j: w.groupby(w.index.to_period('M')).sum() for j, w in W.items()}).dropna()
    C = M.corr().to_numpy(); rho = (C.sum() - 13) / (13 * 12)
    rep[f'{t}|{v}'] = dict(exc_med=med('exc'), t_med=med('t'), exc_Y19_med=med('exc_Y19'),
                          n_j_exc_pos=int(sum(p['exc'] > 0 for p in per)),
                          exc_no2006_med=med('exc_no2006'), exc_no_best2_med=med('exc_no_best2'),
                          exc_Y19_no2024_med=med('exc_Y19_no2024'), exc_Y19_no2024_2026_med=med('exc_Y19_no2024_2026'),
                          n_j_Y19_no2024_pos=int(sum(p['exc_Y19_no2024'] > 0 for p in per)),
                          mean_corr_across_j=float(rho), n_eff_j=float(13 / (1 + 12 * rho)))

# 4 邊界：R4c 與 引擎雜訊
for t, v in [('300萬', 'GROW'), ('1000萬', 'FIX'), ('1000萬', 'GROW')]:
    exc4c = [S.stats(col(R4c, j, t, v), 'AB_V7_B10', j)['exc'] for j in range(13)]
    rep[f'{t}|{v}']['R4c_exc_med'] = float(np.median(exc4c))
# 引擎路徑雜訊：R0 對原表逐 j 年化差（一般成本 K5T）
noise = []
for t, v in CELLS:
    for j in range(13):
        a = S.stats(col(R0, j, t, v), 'AB_V7_B10', j)['CAGR']
        b = S.stats(col(ORIG, j, t, v), 'AB_V7_B10', j)['CAGR']
        noise.append(a - b)
noise = np.array(noise)
rep['ENGINE_NOISE_R0_vs_orig_perj'] = dict(abs_p50=float(np.median(abs(noise))), abs_p90=float(np.percentile(abs(noise), 90)),
                                          abs_max=float(abs(noise).max()), sd=float(noise.std(ddof=1)))

# 5 區塊 bootstrap：J 中位數 j（以 j=6 代表）全期視窗超額平均的 90% CI
rng = np.random.default_rng(0)
for t, v in [('300萬', 'GROW'), ('1000萬', 'FIX')]:
    n = col(R4, 6, t, v); e = S.end_day('AB_V7_B10', 6)
    x = S.windows(n.loc[:e].dropna(), 'AB_V7_B10', 6).dropna().to_numpy()
    L = 12; nb = int(np.ceil(len(x) / L)); bs = []
    for _ in range(4000):
        st = rng.integers(0, len(x) - L + 1, nb)
        bs.append(np.concatenate([x[s:s + L] for s in st])[:len(x)].mean())
    rep[f'{t}|{v}']['J6_window_mean_pp'] = float(x.mean() * 100)
    rep[f'{t}|{v}']['J6_block_boot_90CI_pp'] = [float(np.percentile(bs, 5) * 100), float(np.percentile(bs, 95) * 100)]
    rep[f'{t}|{v}']['n_windows'] = int(len(x))

print(json.dumps(rep, ensure_ascii=False, indent=1))
