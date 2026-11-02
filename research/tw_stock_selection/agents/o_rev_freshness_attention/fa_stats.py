#!/usr/bin/env python3
"""檢定量（寫死；PREREG 雜湊涵蓋本檔）。
  每月 m、每組 g：b_g(m)＝α_g(m, j) 對 j 的 OLS 斜率（至少 10 個 j 有值）；b_g＝月平均；t＝月份分群（月斜率的平均 ÷ 標準誤）。
  （j 設計在每個月相同時，這等於合併 OLS 斜率與月份分群標準誤；缺 j 的月份用該月自己的 j 做 OLS。）
  I(m)＝b_高(m) − b_低(m)；I＝月平均，t 同上。DDD＝I_後 − I_前（Welch t，兩期月份獨立）。
  b_低,後 的 95% 區間＝平均 ± 1.96×標準誤。
  單位：α 是每窗（約一個月）的報酬；斜率單位＝每多一個交易日，月 α 變動幾 pp（報告時 ×100）。"""
import math
import numpy as np, pandas as pd
import fa_common as C

def month_slopes(A, scheme, leg, g, col='alpha', per=None):
    x = A[(A.scheme == scheme) & (A.leg == leg) & (A.g == g)]
    if per is not None:
        mm = pd.PeriodIndex(x.m, freq='M'); x = x[(mm >= per[0]) & (mm <= per[1])]
    out = {}
    for m, gg in x.groupby('m'):
        out[m] = C.slope_month(dict(zip(gg.j.astype(int), gg[col])))
    return pd.Series(out, dtype=float).dropna()

def summ(s):
    s = pd.Series(s).dropna()
    n = len(s)
    if n < 3: return dict(mean_pp=float('nan'), t=float('nan'), n=n, ci95_pp=[float('nan')] * 2)
    se = s.std(ddof=1) / math.sqrt(n)
    return dict(mean_pp=float(s.mean() * 100), t=C.tstat(s.to_numpy()), n=int(n), se_pp=float(se * 100),
                ci95_pp=[float((s.mean() - 1.96 * se) * 100), float((s.mean() + 1.96 * se) * 100)])

def interaction(A, scheme='A', leg='main', hi=2, lo=0, col='alpha', per=None):
    bh = month_slopes(A, scheme, leg, hi, col, per); bl = month_slopes(A, scheme, leg, lo, col, per)
    ix = bh.index.intersection(bl.index)
    I = (bh.reindex(ix) - bl.reindex(ix))
    return dict(b_hi=summ(bh), b_lo=summ(bl), I=summ(I), ratio_abs_blo_bhi=float(abs(bl.mean()) / abs(bh.mean())) if bh.mean() != 0 else float('nan'),
                t_diff=C.tstat(I.to_numpy())), I, bh, bl

def ddd(I_post, I_pre):
    return dict(DDD_pp=float((I_post.mean() - I_pre.mean()) * 100), t=C.welch(I_post.to_numpy(), I_pre.to_numpy()), n_post=len(I_post), n_pre=len(I_pre))

def alpha_by_j(A, scheme, leg, g, col='alpha', per=None):
    x = A[(A.scheme == scheme) & (A.leg == leg) & (A.g == g)]
    if per is not None:
        mm = pd.PeriodIndex(x.m, freq='M'); x = x[(mm >= per[0]) & (mm <= per[1])]
    return {int(j): dict(mean_pp=float(gg[col].mean() * 100), t=C.tstat(gg[col].to_numpy()), n=int(len(gg))) for j, gg in x.groupby('j')}

def loyo(I):
    """逐年剔除（錨定月的年份）：回傳每年剔除後的 I 平均（pp）與 t。"""
    yrs = sorted(set(str(m)[:4] for m in I.index))
    return {y: dict(mean_pp=float(I[[str(m)[:4] != y for m in I.index]].mean() * 100), t=C.tstat(I[[str(m)[:4] != y for m in I.index]].to_numpy())) for y in yrs}

def year_share(I):
    """各年對 Σ I(m) 的占比（同號總和為分母：以 I 總和的符號為準，該年加總 ÷ 總和）。"""
    tot = I.sum()
    by = I.groupby([str(m)[:4] for m in I.index]).sum()
    return {y: float(v / tot) for y, v in by.items()}, float((by / tot).max())

def stock_contrib(CF, per, js_all=C.JS):
    """CF：逐檔權重表（m, j, g, col, w, r）。每月斜率權重 (j − j̄)/Σ(j − j̄)²（該月有值的 j），
    c_i＝Σ_m Σ_j sw_j × w_i × r_i，高組減低組；除以月數＝該股對 I 的貢獻（與 I 的月平均相加一致，未扣成本）。"""
    x = CF.copy(); mm = pd.PeriodIndex(x.m, freq='M'); x = x[(mm >= per[0]) & (mm <= per[1])]
    # 每月有值的 j（高低組都要有）
    jm = x.groupby(['m', 'g']).j.unique()
    sw = {}
    months = sorted(set(x.m))
    used = []
    for m in months:
        try: jh, jl = set(jm[(m, 2)]), set(jm[(m, 0)])
        except KeyError: continue
        if len(jh) < C.MIN_J or len(jl) < C.MIN_J: continue
        for g, js in ((2, jh), (0, jl)):
            js = sorted(js); w = C.slope_weights(js)
            for jj, ww in zip(js, w): sw[(m, g, int(jj))] = ww
        used.append(m)
    x = x[x.m.isin(used)]
    key = list(zip(x.m, x.g.astype(int), x.j.astype(int)))
    x = x.assign(sw=[sw[k] for k in key])
    x['c'] = x.sw * x.w * x.r * np.where(x.g == 2, 1.0, -1.0)
    c = x.groupby('col').c.sum() / len(used)
    return c, len(used)

def stock_bootstrap(c, B=4000, seed=20261102):
    rng = np.random.default_rng(seed)
    v = c.to_numpy(); n = len(v)
    bs = np.array([v[rng.integers(0, n, n)].sum() for _ in range(B)])
    tot = v.sum()
    share = c / tot
    return dict(I_pp=float(tot * 100), boot_ci95_pp=[float(np.percentile(bs, 2.5) * 100), float(np.percentile(bs, 97.5) * 100)],
                p_boot_ge0=float((bs >= 0).mean()), max_share=float(share.max()), top5=[(str(k), float(v)) for k, v in share.sort_values(ascending=False).head(5).items()])
