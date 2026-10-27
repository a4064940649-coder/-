#!/usr/bin/env python3
"""反方審查（資料與統計）：python3 -I -B review_stat_o1.py  — 只讀研究員檔案，不寫回任何 json。
全部只用挑選期 2006-07～2013-12（同研究員），不碰 2014 年以後。
"""
import sys, math, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
import olib as O
V, K = O.V, O.K

R = O.revenue()
S = V.V2('gapfix', R['YL'], R['Y5'], exfin=True)
L = O.Leg(S)
d0 = O.leg_start(L)
days = [d for d in S.sig_days if d0 <= d <= O.SEL_END]
t1 = K.IDX.get_loc(K.IDX[K.IDX <= O.SEL_END][-1]); sd = days + [K.IDX[t1]]
NORMAL = K.rates(0.001)
plm = [p for p in S.plan(10, 'official') if p[3] in set(days)]
t0 = plm[0][0]
ew, _ = K.simulate([(p[0], p[2]) for p in plm], 0, 0, t0, t1)
wr_ew = O.window_returns(ew, sd)
yrs = np.array([d.year for d in wr_ew.index])

def ex_series(pl, b, s, **kw):
    nav, lg = K.simulate([(p[0], p[1]) for p in pl], b, s, t0, t1, **kw)
    return (O.window_returns(nav, sd) - wr_ew).to_numpy(), lg

def nw_t(x, L_=3):
    x = np.asarray(x, float); x = x[np.isfinite(x)]; n = len(x); e = x - x.mean()
    g = e @ e / n
    for l in range(1, L_ + 1): g += 2 * (1 - l / (L_ + 1)) * (e[l:] @ e[:-l]) / n
    return float(x.mean() / math.sqrt(g / n))

out = {}
# 1. 全 24 版，淨與毛
XN, XG = [], []
for v in O.ALL_VERSIONS:
    rev, band, n = v.split('|'); pl = L.plan(rev, band, int(n[1:]), days)
    xn, _ = ex_series(pl, *NORMAL); xg, _ = ex_series(pl, 0, 0)
    XN.append(xn); XG.append(xg)
XN = np.column_stack(XN); XG = np.column_stack(XG)
j = O.ALL_VERSIONS.index('AVG|B1|n30')
xn, xg = XN[:, j], XG[:, j]
out['repro'] = {'t_net': O.tstat(xn), 't_gross': O.tstat(xg), 'mean_net_pct': xn.mean() * 100, 'mean_gross_pct': xg.mean() * 100, 'T': len(xn)}
out['NW3'] = {'t_net': nw_t(xn), 't_gross': nw_t(xg)}

# 2. 逐年與剔除一年
out['by_year_gross_pct'] = {int(y): round(float(xg[yrs == y].mean() * 100), 3) for y in sorted(set(yrs))}
out['by_year_net_pct'] = {int(y): round(float(xn[yrs == y].mean() * 100), 3) for y in sorted(set(yrs))}
out['drop_one_year_t'] = {int(y): {'net': round(O.tstat(xn[yrs != y]), 2), 'gross': round(O.tstat(xg[yrs != y]), 2)} for y in sorted(set(yrs))}
e = yrs <= 2008
out['split'] = {'2006_2008': {'gross_t': O.tstat(xg[e]), 'gross_pct': xg[e].mean() * 100, 'net_t': O.tstat(xn[e]), 'n': int(e.sum())},
                '2009_2013': {'gross_t': O.tstat(xg[~e]), 'gross_pct': xg[~e].mean() * 100, 'net_t': O.tstat(xn[~e]), 'n': int((~e).sum())}}
# 剔除最好的 k 個月
o = np.argsort(-xg)
out['drop_best_months_gross_t'] = {k: round(O.tstat(np.delete(xg, o[:k])), 2) for k in (1, 3, 5)}
# 毛 t 在 2009～2013 對 24 版的 max-t
out['maxt_gross_2009_2013'] = O.maxt_bootstrap(XG[~e], B=3000, mean_block=6, seed=1)

# 3. 學生化 bootstrap max-t（每次重抽都重算 sd）＋不同區塊長度
def maxt_stud(X, B=3000, mb=6, seed=0):
    T, Kk = X.shape; t_obs = O.tvec(X); Xc = X - X.mean(axis=0)
    rng = np.random.default_rng(seed); idx = O.stationary_idx(T, B, mb, rng)
    mx = np.array([O.tvec(Xc[idx[b]]).max() for b in range(B)])
    tm = t_obs.max()
    return {'t_max': round(float(tm), 3), 'p': round(float((mx >= tm).mean()), 4), 'q95': round(float(np.quantile(mx, .95)), 3)}
out['maxt_studentized'] = {f'{lab}_mb{mb}': maxt_stud(X, mb=mb) for lab, X in (('net', XN), ('gross', XG)) for mb in (3, 6, 12)}

# 4. 鄰近檔數穩定性（不是新版本，不參與判定）
nb = {}
for n in (10, 15, 20, 25, 30, 40, 50):
    pl = L.plan('AVG', 'B1', n, days); a, lg = ex_series(pl, *NORMAL); g, _ = ex_series(pl, 0, 0)
    nb[n] = {'t_net': round(O.tstat(a), 2), 't_gross': round(O.tstat(g), 2), 'gross_pct': round(g.mean() * 100, 3), 'turn': round(lg['turnover'], 3)}
out['AVG_B1_n_neighbors'] = nb
nbb = {}
for band in O.BANDS:
    pl = L.plan('AVG', band, 30, days); g, _ = ex_series(pl, 0, 0)
    nbb[band] = round(O.tstat(g), 2)
out['AVG_n30_by_band_gross_t'] = nbb

# 5. 下市最壞情境（凍結持股下市市值歸零）
pl = L.plan('AVG', 'B1', 30, days)
a, _ = ex_series(pl, *NORMAL, delist_zero=True)
ewz, _ = K.simulate([(p[0], p[2]) for p in plm], 0, 0, t0, t1, delist_zero=True)
az = (O.window_returns(K.simulate([(p[0], p[1]) for p in pl], *NORMAL, t0, t1, delist_zero=True)[0], sd) - O.window_returns(ewz, sd)).to_numpy()
out['delist_zero'] = {'t_net_leg_dz_vs_EW_normal': O.tstat(a), 't_net_both_dz': O.tstat(az)}

# 6. 營收覆蓋／存活：股池基礎條件（流動性前 300）裡有多少沒有營收；其中之後下市的比例
lt = np.where(K.TRADED.any(axis=0), len(K.IDX) - 1 - np.argmax(K.TRADED[::-1], axis=0), -1)
dead_by = K.IDX.get_loc(K.IDX[K.IDX <= pd.Timestamp('2026-09-30')][-1])
cov = []
for d in days[::6]:
    cs = S.cross_section(d); t = cs['t']; v60 = S.V60[t]; px = K.Cn[t]
    base = np.isfinite(v60) & (v60 >= 1e7) & np.isfinite(px) & (px >= 10) & ~V.FINMASK
    order = np.argsort(-np.where(base, v60, -np.inf), kind='stable'); top = np.zeros(K.N, bool); top[order[:300]] = True; top &= base
    has = np.isfinite(cs['y3'])
    dead = lt < dead_by
    cov.append({'d': str(d.date()), 'top300': int(top.sum()), 'no_rev': int((top & ~has).sum()),
                'dead_in_top': int((top & dead).sum()), 'dead_with_rev': int((top & dead & has).sum())})
out['revenue_coverage'] = cov
# 選中版本持股中後來下市的比例 vs 股池
hd = [np.mean(lt[p[1]] < dead_by) for p in pl]; pd_ = [np.mean(lt[p[2]] < dead_by) for p in pl]
out['later_delisted_share'] = {'leg': float(np.mean(hd)), 'pool': float(np.mean(pd_))}

# 7. 每月成本拖累
out['cost_drag_pct_m'] = float((xg - xn).mean() * 100)
print(json.dumps(O.r4(out), ensure_ascii=False, indent=1))
