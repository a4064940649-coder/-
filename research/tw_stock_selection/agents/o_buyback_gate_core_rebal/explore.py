# -*- coding: utf-8 -*-
"""探索（只用挑選期 SEL）：主策略對 0050 相對報酬的自相關、變異數比（區塊 bootstrap CI）、兩腿相關逐年、理論分散報酬 vs 實際。"""
import sys, json
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import numpy as np, pandas as pd
import rlib as L
a, b = L.PER['SEL']
A = L.bench('MAIN|normal', a, b); B = L.bench('E50', a, b)
rA = L.win(A); rB = L.win(B)
d = (rA - rB).to_numpy(); n = len(d)
ld = (np.log1p(rA) - np.log1p(rB)).to_numpy()
LAGS = [1, 2, 3, 6, 12, 24, 36, 60]
def acf(x, k):
    x = x - x.mean(); return float((x[k:] @ x[:-k]) / (x @ x)) if k < len(x) else np.nan
def vr(x, k):
    s = pd.Series(x).rolling(k).sum().dropna().to_numpy()
    return float(s.var(ddof=1) / (k * x.var(ddof=1)))
def hcorr(x, k):
    s = pd.Series(x).rolling(k).sum()
    return float(s.corr(s.shift(-k)))
rng = np.random.default_rng(20261027)
def mbb(x, blk=12, B=2000):
    n = len(x); nb = int(np.ceil(n / blk)); out = []
    for _ in range(B):
        st = rng.integers(0, n - blk + 1, nb)
        out.append(np.concatenate([x[s:s + blk] for s in st])[:n])
    return out
BS = mbb(ld)
res = dict(n_windows=n, mean_rel_m_pp=float(d.mean() * 100), sd_rel_m_pp=float(d.std(ddof=1) * 100), TE_ann=float(d.std(ddof=1) * np.sqrt(12) * 100))
res['acf'] = {}; res['vr'] = {}; res['hcorr'] = {}
for k in LAGS:
    est = acf(ld, k)
    bs = [acf(x, k) for x in BS]
    res['acf'][k] = dict(est=est, ci90=[float(np.nanpercentile(bs, 5)), float(np.nanpercentile(bs, 95))], se_iid=float(1 / np.sqrt(n)))
    if k >= 2 and k <= 60:
        e = vr(ld, k); bsv = [vr(x, k) for x in BS]
        res['vr'][k] = dict(est=e, ci90=[float(np.percentile(bsv, 5)), float(np.percentile(bsv, 95))])
    if k <= 36:
        res['hcorr'][k] = dict(est=hcorr(ld, k), n_pairs=int(n - 2 * k + 1))
# 不放回原序的 iid 置換（虛無：無序列相關）下 VR 分布
perm = [rng.permutation(ld) for _ in range(2000)]
res['vr_null_iid'] = {k: [float(np.percentile([vr(x, k) for x in perm], 5)), float(np.percentile([vr(x, k) for x in perm], 95))] for k in (3, 6, 12, 24, 36, 60)}
# 12 個月相對動能預測下 12 個月（重疊，NW 看 sign）
s12 = pd.Series(ld).rolling(12).sum()
res['pred_12_on_12'] = dict(corr=float(s12.corr(s12.shift(-12))), n=int(s12.notna().sum() - 12))
# 兩腿相關逐年（日報酬與訊號窗報酬）
dA = A.pct_change().dropna(); dB = B.pct_change().reindex(dA.index)
res['corr_daily_by_year'] = {int(y): float(dA[dA.index.year == y].corr(dB[dB.index.year == y])) for y in sorted(set(dA.index.year))}
res['corr_monthly_SEL'] = float(pd.Series(rA.values).corr(pd.Series(rB.values)))
res['rel_by_year_pp'] = {int(k): float(v) for k, v in L.yearly_diff(A, B).items()}
# 理論 vs 實際分散報酬
VARd = float(d.var(ddof=1) * 12)
gA = L.cagr(A); gB = L.cagr(B)
res['legs'] = dict(CAGR_A=gA * 100, CAGR_B=gB * 100, var_rel_ann=VARd, vol_A=float(rA.std() * np.sqrt(12) * 100), vol_B=float(rB.std() * np.sqrt(12) * 100))
res['div_return'] = {}
for w in (0.2, 0.3, 0.4):
    M = L.sim(w, 'M', a, b)['nav']; NN = L.sim(w, 'NONE', a, b)['nav']
    Mg = L.sim(w, 'M', a, b, cost='normal')
    lw = (1 - w) * np.log1p(gA) + w * np.log1p(gB)
    res['div_return'][w] = dict(theory_pp=0.5 * w * (1 - w) * VARd * 100,
                                M_minus_weighted_geo_pp=(np.log1p(L.cagr(M)) - lw) * 100,
                                M_minus_NONE_pp=(L.cagr(M) - L.cagr(NN)) * 100,
                                NONE_end_wB=float(NN.iloc[-1] and 0) if False else None,
                                CAGR_M=L.cagr(M) * 100, CAGR_NONE=L.cagr(NN) * 100, MDD_M=L.mdd(M) * 100, MDD_NONE=L.mdd(NN) * 100)
    wbN = L.sim(w, 'NONE', a, b)
    res['div_return'][w]['NONE_wB_end'] = float(np.nan)
# 漂移版期末 0050 權重
for w in (0.2, 0.3, 0.4):
    x = (1 - w) * A / A.iloc[0]; y = w * B / B.iloc[0]
    res['div_return'][w]['NONE_wB_end'] = float((y / (x + y)).iloc[-1])
    res['div_return'][w]['NONE_wB_min'] = float((y / (x + y)).min())
# 子期間：S1、S2 的相對報酬 lag-1、12 自相關
for p in ('S1', 'S2'):
    aa, bb = L.PER[p]
    x = (np.log1p(L.win(L.bench('MAIN|normal', aa, bb))) - np.log1p(L.win(L.bench('E50', aa, bb)))).to_numpy()
    res[f'acf_{p}'] = {k: acf(x, k) for k in (1, 3, 6, 12, 24)}
(HERE / 'explore.json').write_text(json.dumps(res, ensure_ascii=False, indent=1, default=float), encoding='utf-8')
print(json.dumps(res, ensure_ascii=False, indent=1, default=float))
