#!/usr/bin/env python3
"""步驟 11：模擬精度的內部檢查（不是獨立複製）。D1（iid 常態）下 R2、R4 的「邊際」越界機率可用多元常態 CDF 精確算：
z_n 在檢視點的相關＝√(n_i/n_j)，均值＝α√n/σ。與 s5 的校準目標（4%、55%）及 OC 模擬值比較。
用法：python3 -I -B s11_check.py"""
import sys, math
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
from scipy import stats
import fslib as F, seq2 as Q


def p_any_above(looks, bounds, alpha, sigma):
    n = np.array(looks, float)
    C = np.sqrt(np.minimum.outer(n, n) / np.maximum.outer(n, n))
    mu = alpha * np.sqrt(n) / sigma
    return float(1 - stats.multivariate_normal(mean=mu, cov=C).cdf(np.asarray(bounds)))


def main():
    res = F.load_res(); oc = res['OC_FINAL']; par = oc['params']; sig = oc['sigma_design']
    L6 = [l for l in Q.looks6(60) if l <= 36]; L12 = [12, 24, 36]
    out = {}
    out['R2_up_a0'] = p_any_above(L6, [par['c_u'] * math.sqrt(36 / n) for n in L6], 0.0, sig)
    out['R2_down_a_m1'] = p_any_above(L6, [par['c_d']] * len(L6), 1.0, sig)          # 對稱：−z 越過 c_d
    out['R2_down_a0'] = p_any_above(L6, [par['c_d']] * len(L6), 0.0, sig)
    out['R4_up_a0'] = p_any_above(L12, [par['c_u4']] * 3, 0.0, sig)
    out['R4_down_a_m1'] = p_any_above(L12, [par['c_d4']] * 3, 1.0, sig)
    out['R4_down_a0'] = p_any_above(L12, [par['c_d4']] * 3, 0.0, sig)
    oc_tab = pd.read_csv(F.DATA / 'oc_final.csv')
    g = oc_tab[(oc_tab.dgp == 'D1') & (oc_tab.h == 36)]
    sim = {}
    for r in ('R2', 'R4'):
        sim[f'{r}_up_a0'] = float(g[(g.rule == r) & (g.alpha == 0)].up.iloc[0])
        sim[f'{r}_down_a_m1'] = float(g[(g.rule == r) & (g.alpha == -1)].down.iloc[0])
        sim[f'{r}_down_a0'] = float(g[(g.rule == r) & (g.alpha == 0)].down.iloc[0])
    F.save_key('CHECK_MVN', dict(exact_marginal=out, simulated_absorbing=sim, note='模擬值含吸收（先觸發者），所以會略低於邊際精確值'), need=['OC_FINAL'])
    for k in out: print(k, round(out[k], 4), round(sim[k], 4))


if __name__ == '__main__':
    main()
