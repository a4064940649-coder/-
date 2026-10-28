#!/usr/bin/env python3
"""內部交叉核對（不是獨立複製）：
(1) G2 第二實作：B-spline 基底（scipy BSpline，df 5、三次）＋ L-BFGS-B（數值梯度），θ 格點改為 [−6,6] 間距 0.01；
    比較 MAIN 族群下 AB、B10、EXEC、EXEC_J 的事後平均與 P(α>0)。
(2) 模擬第二實作：不透過 sc_lib.simulate，直接以 log 財富比加總計算 x＝100%（無再平衡成本）的 10 年勝率與 x* 曲線（x＝0～1、無成本），
    種子不同（12345），比較 P_win 與 x*。
用法：python3 -I -B crosscheck.py"""
import sys, json, math
from pathlib import Path
import numpy as np, pandas as pd
from scipy import optimize, stats
from scipy.interpolate import BSpline
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import sc_lib as S

def g2_bspline(xhat, s2, th, df=5, c0=None):
    c0 = S.G2_C0 if c0 is None else c0
    k = 3
    nint = df - k + 1
    inner = np.linspace(th[0], th[-1], nint + 1)
    t = np.r_[[th[0]] * k, inner, [th[-1]] * k]
    Bm = BSpline.design_matrix(th, t, k).toarray()[:, 1:]       # 去掉一欄避免與常數共線
    Bm = Bm - Bm.mean(axis=0); Bm = Bm / np.sqrt((Bm ** 2).sum(axis=0))
    P = np.exp(-0.5 * (xhat[:, None] - th[None, :]) ** 2 / s2[:, None]) / np.sqrt(s2)[:, None]
    def f(b):
        a = Bm @ b; a = a - a.max(); g = np.exp(a); g /= g.sum()
        return -(np.log(P @ g + 1e-300).sum() - c0 * math.sqrt((b * b).sum() + 1e-12))
    r = optimize.minimize(f, np.zeros(Bm.shape[1]), method='L-BFGS-B', options={'maxiter': 20000, 'maxfun': 200000})
    a = Bm @ r.x; a = a - a.max(); g = np.exp(a); g /= g.sum()
    return g

def main():
    S.check_prereg()
    X, M = S.load_family()
    F = S.load_focus()
    fam = json.loads((S.DATA / 'families.json').read_text(encoding='utf-8'))
    meta = json.loads((S.DATA / 'focus_meta.json').read_text(encoding='utf-8'))
    sdc = meta['cost_uncertainty']['sd_cost_pp']
    res = S.load_res()
    post = res['RESULTS_POSTERIOR']['families']['MAIN']['post']['G2']
    fx = {c: (F[c] - F['r0050']).values * 100 for c in ['AB', 'B10', 'EXEC', 'EXEC_J']}
    dec = S.decompose(X[fam['MAIN']], fx, mode='decomp')
    th = np.round(np.arange(-6, 6.00001, 0.01), 10)
    g = g2_bspline(dec['xhat'], dec['s2'], th)
    out = {'G2_bspline': {}}
    for c in fx:
        s2f = dec['focus_s2'][c] + (sdc ** 2 if c.startswith('EXEC') else 0)
        lp = np.log(g + 1e-300) - 0.5 * (dec['focus_xhat'][c] - th) ** 2 / s2f
        pb = np.exp(lp - lp.max()); pb /= pb.sum()
        # 與 N(0, vP) 卷積：均值不變；P(α>0)＝Σ pb Φ(θ/√vP)
        m = float((th * pb).sum()); pp = float((pb * stats.norm.cdf(th / math.sqrt(dec['vP']))).sum())
        out['G2_bspline'][c] = {'mean': m, 'p_pos': pp, 'main_impl_mean': post[c]['mean'], 'main_impl_p_pos': post[c]['p_pos'],
                                'd_mean': m - post[c]['mean'], 'd_p_pos': pp - post[c]['p_pos']}
    # 模擬第二實作
    D = np.load(S.DATA / 'post_dens_V4.npz')
    alloc = res.get('RESULTS_ALLOC', {})
    sim = {}
    rng = np.random.default_rng(12345)
    r50 = F['r0050'].values; n = len(r50); B = 20000; H = 120
    # 平穩 bootstrap（自寫，不用 sc_lib.sb_index）
    start = rng.integers(0, n, (B, H)); newb = rng.random((B, H)) < 1 / 6; newb[:, 0] = True
    pos = np.zeros((B, H), np.int64)
    for t in range(H):
        pos[:, t] = np.where(newb[:, t], start[:, t], (pos[:, t - 1] + 1) % n if t > 0 else start[:, t])
    for c in fx:
        e = F[c].values - r50; res_ = e - e.mean()
        th0 = D['theta']; d = D[f'MAIN__G2__{c}']
        a = th0[rng.choice(len(th0), size=B, p=d)] / 100
        r5 = r50[pos]; rm = r5 + a[:, None] + res_[pos]
        lr = np.log1p(rm).sum(axis=1) - np.log1p(r5).sum(axis=1)
        pw = float((lr > 0).mean())
        el = []
        for x in np.arange(0, 1.0001, 0.1):
            el.append(float((np.log1p(r5 + x * (a[:, None] + res_[pos])).sum(axis=1) - np.log1p(r5).sum(axis=1)).mean()))
        xs = int(np.argmax(el) * 10)
        main_pw = alloc.get('main', {}).get(c, {}).get('10y', {}).get('100', {}).get('P_win')
        main_xs = alloc.get('kelly_grid', {}).get(c, {}).get('G2|L6|10y', {}).get('x_star')
        sim[c] = {'P_win_x100_10y': pw, 'x_star_nocost': xs, 'main_P_win': main_pw, 'main_x_star': main_xs}
    out['sim_second_impl'] = sim
    S.save_key('CROSSCHECK', out)
    print(json.dumps(out, ensure_ascii=False, indent=1))

if __name__ == '__main__':
    main()
