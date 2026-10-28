#!/usr/bin/env python3
"""反方審查（資料與統計）：o_main_spa_breaks。只讀研究員的檔案，結果只印在畫面，不寫任何檔案。
python3 -I -B review_stat_o5.py
1) 獨立實作 SPA_c（焦點檢定量；平穩 bootstrap、bootstrap 變異數、不同種子）重算判定欄。
2) 焦點換成存活者偏誤保守版 AB_V4_B10（同一族群）：對 0050 與對「自己的」同池等權。
3) 保留期把 2023-12-11 那一窗（跨 2024-01-02～02-14 面板缺口、包含整段日曆落後）算進去。
4) 集中度因子回歸是否近乎恆等式：CF 與 (EW − 0050) 的相關；x50 對 (EW−0050) 的回歸 α 與 xEW 均值。
5) 年份集中：拿掉 2006～2007、拿掉 2021 的 t；前後半段 t。
"""
import math
from pathlib import Path
import numpy as np, pandas as pd

H = Path(__file__).resolve().parent
V4 = H.parent / 'o_v4_revenue_patch_official'
MAIN = 'o_v4_revenue_patch_official/navs_v4.parquet::AB_V4|normal'
B10 = 'o_v4_revenue_patch_official/navs_v4.parquet::AB_V4_B10|normal'

def t_(x):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    return x.mean() / x.std(ddof=1) * math.sqrt(len(x))

def sb_idx(n, L, B, rng, n_out=None):
    p = 1 / L; no = n if n_out is None else n_out; idx = np.empty((B, no), np.int64); idx[:, 0] = rng.integers(0, n, B)
    for t in range(1, idx.shape[1]):
        new = rng.random(B) < p
        idx[:, t] = np.where(new, rng.integers(0, n, B), (idx[:, t - 1] + 1) % n)
    return idx

def spa_focus(X, foci, L=6, B=4000, seed=1):
    """X n×N；回傳每個焦點欄的 SPA_c 焦點 p（bootstrap 變異數標準化）與族群最大值 p。"""
    n, N = X.shape
    rng = np.random.default_rng(seed)
    idx = sb_idx(n, L, B, rng)
    dbar = X.mean(0)
    Mb = np.empty((B, N), np.float32)
    for s in range(0, B, 250):
        Mb[s:s + 250] = X[idx[s:s + 250]].mean(1)
    om = np.sqrt(n * Mb.astype(np.float64).var(0))
    tst = math.sqrt(n) * dbar / om
    thr = -om / math.sqrt(n) * math.sqrt(2 * math.log(math.log(n)))
    g = np.where(dbar >= thr, dbar, 0.0)
    Tb = np.maximum(((Mb - g) * (math.sqrt(n) / om)).max(1), 0)
    out = {'p_familymax': float((Tb >= max(tst.max(), 0)).mean()), 'max_t': float(tst.max())}
    for f in foci:
        out[f] = {'t': float(tst[f]), 'rank': int((tst > tst[f]).sum() + 1), 'p_SPA_c_focus': float((Tb >= tst[f]).mean())}
    return out

def win_ret(nav, grid):
    v = nav.reindex(nav.index.union(grid)).ffill().reindex(grid).to_numpy()
    return pd.Series(v[1:] / v[:-1] - 1, index=grid[:-1])

def main():
    D = pd.read_parquet(H / 'data/family_excess.parquet'); MM = pd.read_csv(H / 'data/family_members.csv')
    F = pd.read_parquet(H / 'data/family_ret.parquet'); Bn = pd.read_parquet(H / 'data/bench.parquet')
    T = pd.read_parquet(H / 'data/tsmc_inputs.parquet')
    mem = MM[MM.F_long].member_id.tolist()
    m = (D.index >= '2006-04-01') & (D.index <= '2026-10-12')
    X = D.loc[m, mem].fillna(0.0).to_numpy()
    fm, fb = mem.index(MAIN), mem.index(B10)
    import sys
    if "--skip1" in sys.argv: pass
    print('== 1) 獨立 SPA_c（bootstrap 變異數，種子不同）F_long W_long ==')
    for L in (() if '--skip1' in sys.argv else (3, 6, 12)):
        o = spa_focus(X, [fm, fb], L=L, seed=100 + L)
        print(f' L{L}: maxt {o["max_t"]:.2f} p_max {o["p_familymax"]:.4f} | AB_V4 {o[fm]} | AB_V4_B10 {o[fb]}')
    # 2) B10 對自己的同池等權
    nv = pd.read_parquet(V4 / 'navs_v4.parquet')
    grid = pd.DatetimeIndex(list(pd.to_datetime(pd.read_csv(V4 / 'plan_V4.csv')['signal'])) + [pd.Timestamp('2026-10-12')])
    r_b10 = win_ret(nv['AB_V4_B10|normal'], grid); ew_b10 = win_ret(nv['AB_V4_B10|EW'], grid)
    r_m = F[MAIN]; ew_m = Bn['rEW_ABV4']; r50 = Bn['r0050']
    print('\n== 2) 存活者偏誤保守版 ==')
    print(' t vs 0050: AB_V4 %.3f  AB_V4_B10 %.3f' % (t_(r_m - r50), t_(r_b10 - r50)))
    print(' t vs own EW: AB_V4 %.3f  AB_V4_B10(vs B10 EW) %.3f  AB_V4_B10(vs AB_V4 EW) %.3f' % (t_(r_m - ew_m), t_(r_b10 - ew_b10), t_(r_b10 - ew_m)))
    print(' mean xEW pp: AB_V4 %.3f  B10 %.3f' % ((r_m - ew_m).mean() * 100, (r_b10 - ew_b10).mean() * 100))
    De = pd.read_parquet(H / 'data/family_excess_ew.parquet')
    Xe = De.loc[m, mem].fillna(0.0).to_numpy().copy()
    xb = (r_b10 - ew_b10)[m].to_numpy(); Xe[:, fb] = xb
    o = spa_focus(Xe, [fm, fb], L=6, seed=7) if '--skip1' not in sys.argv else None
    if '--skip1' not in sys.argv: print(' xEW F_long L6: AB_V4', o[fm], '| B10 (own EW)', o[fb], 'p_max', round(o['p_familymax'], 4))
    for a, b in (('2006-04-01', '2019-04-30'), ('2019-05-01', '2026-10-12'), ('2024-01-01', '2026-10-12')):
        mm = (r_b10.index >= a) & (r_b10.index <= b)
        print(f'  {a}~{b}: B10 vs0050 t {t_((r_b10 - r50)[mm]):.2f}  B10 vsEW t {t_((r_b10 - ew_b10)[mm]):.2f}  AB vsEW t {t_((r_m - ew_m)[mm]):.2f}')
    # 3) 保留期含 2023-12-11 窗
    print('\n== 3) 保留期含跨缺口窗 ==')
    lx = np.log1p(r_m) - np.log1p(r50)
    pre = (lx.index >= '2006-04-01') & (lx.index < '2023-12-01')
    for lab, hm in (('prereg 32 窗', lx.index >= '2024-01-01'), ('含 2023-12-11 共 33 窗', lx.index >= '2023-12-01')):
        act = lx[hm].sum(); src = lx[pre].to_numpy(); nh = int(hm.sum())
        for L in (3, 6, 12):
            idx = sb_idx(len(src), L, 10000, np.random.default_rng(55 + L), n_out=nh)
            sim = src[idx].sum(1)
            print(f' {lab} L{L}: actual cumlog {act:+.4f}  p_lower {(sim <= act).mean():.3f}  q10 {np.quantile(sim, .1):+.3f} q50 {np.quantile(sim, .5):+.3f}')
    print(' 2023-12-11 窗: main %.4f 0050 %.4f EW %.4f' % (r_m.loc['2023-12-11'], r50.loc['2023-12-11'], ew_m.loc['2023-12-11']))
    # 4) CF 恆等式
    print('\n== 4) 集中度因子 ==')
    cf = T['CF']; ew50 = ew_m - r50
    ok = m & cf.notna()
    print(' corr(CF, EW-0050) = %.3f' % np.corrcoef(cf[ok], ew50[ok])[0, 1])
    print(' corr(CF, rEW_tsmc - rEW_ABV4) = %.3f' % np.corrcoef(T['rEW'][ok], ew_m[ok])[0, 1])
    for lab, mm in (('ALL', ok), ('HOLD', ok & (cf.index >= '2024-01-01'))):
        y = (r_m - r50)[mm].to_numpy(); xx = ew50[mm].to_numpy()
        Xr = np.column_stack([np.ones(len(y)), xx]); b = np.linalg.lstsq(Xr, y, rcond=None)[0]
        print(f' {lab}: x50 ~ (EW-0050): alpha {b[0] * 100:.2f} pp beta {b[1]:.2f} | mean xEW {(r_m - ew_m)[mm].mean() * 100:.2f} pp')
    # 5) 年份集中
    print('\n== 5) 年份集中（對 0050 窗超額）==')
    x = (r_m - r50)[m]
    print(' all t %.2f n %d' % (t_(x), len(x)))
    print(' drop 2006-2007 t %.2f' % t_(x[x.index.year > 2007]))
    print(' drop 2021 t %.2f' % t_(x[x.index.year != 2021]))
    print(' drop 2006,2007,2015,2021 t %.2f' % t_(x[~x.index.year.isin([2006, 2007, 2015, 2021])]))
    print(' 2008-04 起 t %.2f; 2009-01 起 t %.2f' % (t_(x[x.index >= '2008-04-01']), t_(x[x.index >= '2009-01-01'])))
    srt = np.sort(x.to_numpy())[::-1]
    print(' best 10 窗合計 %.1f pp / 全部合計 %.1f pp' % (srt[:10].sum() * 100, srt.sum() * 100))
    ys = x.groupby(x.index.year).sum().sort_values(ascending=False) * 100
    print(' 前 4 年合計 %.1f / 全部 %.1f pp' % (ys.iloc[:4].sum(), ys.sum()))

if __name__ == '__main__':
    main()
