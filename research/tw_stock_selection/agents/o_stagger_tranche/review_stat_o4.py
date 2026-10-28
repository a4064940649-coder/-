#!/usr/bin/env python3
"""反方審查（資料與統計）：只讀研究員檔案，不寫任何檔案。python3 -I -B review_stat_o4.py"""
import sys, json, math
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import numpy as np, pandas as pd
import st_core as C

NV = pd.read_parquet(HERE / 'navs_oos.parquet')
off_days, off_end = C.official_list()
WO = C.windows_from(off_days, off_end)
E50 = C.E50
JS = list(range(-8, 13))

def ex_own(j):
    days, end = C.version_days(j)
    W = C.windows_from(days, end)
    nav = NV[f'J{j:+d}'].dropna()
    return C.wret(nav, W) - C.wret(E50.reindex(nav.index), W)

def ex_common(col):
    nav = NV[col].dropna()
    W = [w for w in WO if w[2] <= nav.index[-1]]
    return C.wret(nav, W) - C.wret(E50.reindex(nav.index), W)

def nw_t(x, L=None):
    x = np.asarray(x, float); x = x[np.isfinite(x)]; n = len(x)
    if L is None: L = int(math.floor(4 * (n / 100) ** (2 / 9)))
    e = x - x.mean(); g0 = (e @ e) / n; s = g0
    for l in range(1, L + 1):
        w = 1 - l / (L + 1); s += 2 * w * (e[l:] @ e[:-l]) / n
    return float(x.mean() / math.sqrt(s / n)), L

out = {}
# 1) 重算 21 版自己的窗 t（核對）與 j>=0 子集
tt = {}
EXO = {}
for j in JS:
    e = C.sel(ex_own(j), 'ALL').dropna(); EXO[j] = e; tt[j] = C.tstat(e)
out['t_own_ALL'] = {j: round(v, 3) for j, v in tt.items()}
arr = np.array([tt[j] for j in JS]); post = np.array([tt[j] for j in JS if j >= 0]); w1 = np.array([tt[j] for j in range(0, 5)])
out['dist_all21'] = dict(p10=float(np.percentile(arr, 10)), med=float(np.median(arr)))
out['dist_j_ge0'] = dict(n=len(post), p10=float(np.percentile(post, 10)), med=float(np.median(post)), max=float(post.max()))
out['dist_J0_J4'] = dict(med=float(np.median(w1)), min=float(w1.min()), max=float(w1.max()))

# 2) 共同窗：各版月超額相關（有效版本數）
M = pd.DataFrame({j: C.sel(ex_common(f'J{j:+d}'), 'ALL') for j in JS}).dropna()
cm = M.corr().to_numpy(); lam = np.linalg.eigvalsh(cm)
out['common_corr'] = dict(mean_offdiag=float((cm.sum() - 21) / (21 * 20)), eff_n_eig=float(lam.sum() ** 2 / (lam ** 2).sum()))

# 3) t 隨 j 的趨勢（j>=0）：每月配對 J0..J4 均值 − J8..J12 均值（共同窗）
for k in ('SEL', 'POST', 'ALL'):
    Mk = pd.DataFrame({j: C.sel(ex_common(f'J{j:+d}'), k) for j in JS}).dropna()
    d = Mk[[0, 1, 2, 3, 4]].mean(axis=1) - Mk[[8, 9, 10, 11, 12]].mean(axis=1)
    pre = Mk[[0, 1, 2, 3, 4]].mean(axis=1) - Mk[list(range(-8, 0))].mean(axis=1)
    out[f'early_minus_late_{k}'] = dict(pp=float(d.mean() * 100), t=C.tstat(d), n=len(d))
    out[f'J0J4_minus_pre_{k}'] = dict(pp=float(pre.mean() * 100), t=C.tstat(pre))

# 4) 官方日 vs J0 / J+1 / 官方 AB（共同窗配對）
def pair(a, b, k='ALL'):
    na = NV[a].dropna(); nb = NV[b].dropna()
    end = min(na.index[-1], nb.index[-1]); W = [w for w in WO if w[2] <= end]
    d = C.sel(C.wret(na, W) - C.wret(nb, W), k).dropna()
    return dict(pp=round(float(d.mean() * 100), 4), t=round(C.tstat(d), 3), n=len(d))
for a, b in (('OFF', 'OFF_AB0'), ('OFF_AB0', 'J+0'), ('OFF_AB0', 'J+1'), ('OFF', 'J+1'), ('OFF_AB0', 'J1_common'),
             ('V08_normal', 'OFF'), ('V08_normal', 'OFF_AB0'), ('V08_normal', 'J1_common')):
    out[f'pair_{a}_vs_{b}'] = {k: pair(a, b, k) for k in ('SEL', 'POST', 'ALL')}

# 5) 官方日與 V08 的 t：普通 vs Newey-West；拿掉最好年份
for col in ('OFF', 'OFF_AB0', 'V08_normal', 'J1_common'):
    e = C.sel(ex_common(col), 'ALL').dropna()
    yr = e.groupby(e.index.year).sum()
    out[f'tNW_{col}'] = dict(t=C.tstat(e), nw=nw_t(e), top3_years=[(int(y), round(float(v * 100), 1)) for y, v in yr.sort_values(ascending=False).head(3).items()],
                            share_top3yr=float(yr.sort_values(ascending=False).head(3).sum() / yr.sum()),
                            t_drop_top3yr=C.tstat(e[~e.index.year.isin(yr.sort_values(ascending=False).head(3).index)]))
# 6) 官方日 j 分布中 j=-4、-1 那兩個月對 OFF 的貢獻
dd = C.sel(C.wret(NV['OFF'].dropna(), WO) - C.wret(NV['OFF_AB0'].dropna(), WO), 'ALL')
out['OFF_minus_AB_top_months'] = [(str(i), round(float(v * 100), 2)) for i, v in dd.sort_values(ascending=False).head(5).items()]

# 7) V08 標準差下降 6.9% 的抽樣不確定性（挑選期，月份 bootstrap）
ev = C.sel(C.wret(NV['V08_normal'].dropna(), WO) - C.wret(E50.reindex(NV.index), WO), 'SEL').dropna()
eb = C.sel(C.wret(NV['J1_common'].dropna(), WO) - C.wret(E50.reindex(NV.index), WO), 'SEL').dropna()
ix = ev.index.intersection(eb.index); a_ = ev[ix].to_numpy(); b_ = eb[ix].to_numpy()
rng = np.random.default_rng(1); bs = []
for _ in range(4000):
    s = rng.integers(0, len(a_), len(a_)); bs.append(a_[s].std(ddof=1) / b_[s].std(ddof=1) - 1)
out['V08_sdchg_SEL'] = dict(point=float(a_.std(ddof=1) / b_.std(ddof=1) - 1), ci90=[float(np.percentile(bs, 5)), float(np.percentile(bs, 95))],
                           p_le_m10=float(np.mean(np.array(bs) <= -0.10)))
print(json.dumps(out, ensure_ascii=False, indent=1, default=float))

# 8) 前視：V3 正式（未做 AB）減官方 AB，按年份
d8 = (C.wret(NV['OFF'].dropna(), WO) - C.wret(NV['OFF_AB0'].dropna(), WO)).dropna()
print('OFF-AB by year pp', {int(y): round(float(v * 100), 2) for y, v in d8.groupby(d8.index.year).sum().items() if y >= 2019})
h = C.sel(d8, 'HOLD'); print('HOLD OFF-AB', round(float(h.mean() * 100), 3), round(C.tstat(h), 2), 'n', len(h))
# 9) V08 c1 調回等額交易（全在官方日開盤、未計入 capacity）
import iterate as IT
spec = IT.SPECS['V08']
nav, navs, lg, pls, t0 = IT.run_spec(spec, C.NORMAL, max_m=pd.Period('2018-12', 'M'), log=True)
rc = np.array([c[2] for c in lg['reeq_cost']])
print('V08 reeq one-way turnover of account per month: mean', round(float(rc.mean()), 4), 'p90', round(float(np.percentile(rc, 90)), 4), 'n', len(rc))
