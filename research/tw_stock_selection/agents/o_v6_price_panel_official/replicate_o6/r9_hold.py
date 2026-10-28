#!/usr/bin/env python3
"""複製員 o6：保留期 p_lower、MDE、Chow、台積電缺口、跨缺口窗。
用法：O6R_PANEL=compat python3 -I -B r9_hold.py"""
import os, sys, json, math
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import numpy as np, pandas as pd
import r4_env
PANEL = os.environ.get('O6R_PANEL', 'compat')
L = r4_env.load(PANEL); K = L.K
OUT = HERE / f'out_{PANEL}'
NV = pd.read_parquet(OUT / 'navs.parquet')
PJ = json.load(open(OUT / 'plans.json'))
SD6 = pd.DatetimeIndex(PJ['sig_days'])
SD4 = pd.DatetimeIndex(list(pd.to_datetime(pd.read_csv(L.AG / 'o_v4_revenue_patch_official' / 'plan_V4.csv')['signal'])) + [pd.Timestamp('2026-10-12')])
V4N = pd.read_parquet(L.AG / 'o_v4_revenue_patch_official' / 'navs_v4.parquet')
E50 = K.E50
CUT = pd.Timestamp('2024-01-02')
def r(x, k=4): return None if x is None or not np.isfinite(x) else round(float(x), k)

def wins(nav, bench, grid):
    g = pd.DatetimeIndex([d for d in grid if d in nav.index])
    v = nav.reindex(g).to_numpy(); b = bench.reindex(g).to_numpy()
    w = pd.DataFrame({'start': g[:-1], 'end': g[1:], 'r': v[1:] / v[:-1] - 1, 'b': b[1:] / b[:-1] - 1})
    w['x'] = w['r'] - w['b']; w['lx'] = np.log((1 + w['r']) / (1 + w['b']))
    return w

def sb_sum(ctrl, n, L_, B, seed):
    """平穩 bootstrap：從對照樣本（環狀）抽長度 n 的序列，回傳 B 個總和。"""
    rng = np.random.default_rng(seed); m = len(ctrl); p = 1.0 / L_
    idx = np.empty((B, n), np.int64)
    idx[:, 0] = rng.integers(0, m, B)
    for t in range(1, n):
        new = rng.random(B) < p
        idx[:, t] = np.where(new, rng.integers(0, m, B), (idx[:, t - 1] + 1) % m)
    return ctrl[idx].sum(axis=1)

def holdtest(w, attr='cal', col='lx'):
    hold = w[w['end'] >= CUT] if attr == 'cal' else w[w['start'] >= pd.Timestamp('2024-01-01')]
    ctrl = w[(w['start'] >= pd.Timestamp('2006-04-01')) & (w['end'] < CUT)] if attr == 'cal' else \
        w[(w['start'] >= pd.Timestamp('2006-04-01')) & (w['start'] < pd.Timestamp('2024-01-01'))]
    act = hold[col].sum()
    o = {'n_hold': len(hold), 'n_ctrl': len(ctrl), 'actual': r(act)}
    for L_ in (3, 6, 12):
        sims = sb_sum(ctrl[col].to_numpy(), len(hold), L_, 10000, 5150 + L_)
        o[f'p_lower_L{L_}'] = r((sims <= act).mean())
    return o, hold, ctrl

res = {}
w6 = wins(NV['AB_V6|normal'], E50, SD6)
res['AB_V6|cal_V6grid'], hold, ctrl = holdtest(w6, 'cal')
res['AB_V6|start_V6grid'], _, _ = holdtest(w6, 'start')
res['AB_V6|cal_V4grid'], _, _ = holdtest(wins(NV['AB_V6|normal'], E50, SD4), 'cal')
res['AB_V6_B10|cal_V6grid'], _, _ = holdtest(wins(NV['AB_V6_B10|normal'], E50, SD6), 'cal')
wew = wins(NV['AB_V6|normal'], NV['AB_V6|EW'], SD6)
res['AB_V6|xEW_cal'], _, _ = holdtest(wew, 'cal')
res['AB_V4|cal_V4grid(核對)'], _, _ = holdtest(wins(V4N['AB_V4|normal'], E50, SD4), 'cal')

# ---------- MDE（80% 檢定力、單尾 5%）----------
z = 1.6448536269514722 + 0.8416212335729143
def hac_var_mean(x, lag=None):
    x = np.asarray(x, float); n = len(x); e = x - x.mean()
    lag = int(math.floor(4 * (n / 100) ** (2 / 9))) if lag is None else lag
    s = e @ e / n
    for l in range(1, lag + 1): s += 2 * (1 - l / (lag + 1)) * (e[l:] @ e[:-l]) / n
    return s / n
xh, xc = hold['x'].to_numpy(), ctrl['x'].to_numpy()
res['MDE_pp'] = {'two_sample_HAC': r(z * math.sqrt(hac_var_mean(xh) + hac_var_mean(xc)) * 100),
                 'hold_only_HAC': r(z * math.sqrt(hac_var_mean(xh)) * 100),
                 'hold_only_iid': r(z * xh.std(ddof=1) / math.sqrt(len(xh)) * 100)}
for L_ in (3, 6, 12):
    sims = sb_sum(xc - xc.mean(), len(xh), L_, 10000, 5150 + L_) / len(xh)
    res['MDE_pp'][f'boot_L{L_}'] = r(z * sims.std() * 100)
    res['MDE_pp'][f'boot_q_L{L_}'] = r((np.quantile(sims, 0.95) - np.quantile(sims, 0.20)) * 100)

# ---------- Chow ----------
sys.path.insert(0, str(L.AG / 'o_main_spa_breaks'))
import breaks_lib as BL
sys.path.pop(0)
def chow_at(w, a, col='x', attr='start'):
    y = w[w['start'] >= pd.Timestamp('2006-04-01')].reset_index(drop=True)
    k = int((y['start'] < pd.Timestamp(a)).sum()) if attr == 'start' else int((y['end'] < pd.Timestamp(a)).sum())
    o = BL.chow(y[col].to_numpy(), k, hac=True)
    o = {kk: (r(v) if isinstance(v, (float, np.floating)) else v) for kk, v in (o.items() if isinstance(o, dict) else {'out': o}.items())}
    o['k'] = k; o['mean_before_pp'] = r(y[col].iloc[:k].mean() * 100); o['mean_after_pp'] = r(y[col].iloc[k:].mean() * 100)
    return o
res['chow'] = {'x50|2019-05': chow_at(w6, '2019-05-01'), 'x50|2024-01|start': chow_at(w6, '2024-01-01'),
               'x50|2024-01|cal': chow_at(w6, '2024-01-02', attr='cal'),
               'xEW|2024-01|start': chow_at(wew, '2024-01-01'), 'xEW|2024-01|cal': chow_at(wew, '2024-01-02', attr='cal')}

# ---------- 台積電缺口 ----------
XT = K.X
ex = pd.read_csv(XT / 'ex_tsmc.csv', parse_dates=['date']).set_index('date')
sh = pd.read_csv(XT / 'shares_hist.csv', dtype={'code': str})
sh = sh.sort_values('ym').groupby('code')['shares'].last()
Cl = K.C
cols = [c for c in Cl.columns if c in sh.index and not c.startswith('0')]
def w_est(d):
    cap = (Cl.loc[d, cols] * sh[cols]).dropna().sort_values(ascending=False)
    top = cap.iloc[:50]
    return float(top.get('2330', np.nan) / top.sum())
ovl = [d for d in ex.index if d in Cl.index]
cal = np.mean([ex.loc[d, 'w2330'] / 100 - w_est(d) for d in ovl])
def w_at(d):
    if d >= pd.Timestamp('2025-09-26') and d in ex.index: return ex.loc[d, 'w2330'] / 100
    return w_est(d) + cal
A2330 = K.MYADJ['2330']
h = w6[w6['end'] >= CUT].copy()
h['w'] = [w_at(d) for d in h['start']]
h['r2330'] = [A2330.loc[e] / A2330.loc[s] - 1 for s, e in zip(h['start'], h['end'])]
h['rex'] = (h['b'] - h['w'] * h['r2330']) / (1 - h['w'])
lW50 = np.log1p(h['b']).sum(); lWm = np.log1p(h['r']).sum(); lWex = np.log1p(h['rex']).sum()
res['tsmc'] = {'calib_mean_diff': r(cal), 'n': len(h), 'lag_log(0050-main)': r(lW50 - lWm), 'tsmc_gap_log(0050-ex)': r(lW50 - lWex),
               'share': r((lW50 - lWex) / (lW50 - lWm)), 'main_minus_ex_log': r(lWm - lWex), 'w_first': r(h['w'].iloc[0]), 'w_last': r(h['w'].iloc[-1])}
# ---------- 跨缺口窗 2023-12-11 → 2024-02-15 ----------
a, b = pd.Timestamp('2023-12-11'), pd.Timestamp('2024-02-15')
res['gap_window'] = {'V6(AB_V6)': r((NV['AB_V6|normal'].loc[b] / NV['AB_V6|normal'].loc[a] - 1) * 100),
                     'V6(V6)': r((NV['V6|normal'].loc[b] / NV['V6|normal'].loc[a] - 1) * 100),
                     'V4(AB_V4)': r((V4N['AB_V4|normal'].loc[b] / V4N['AB_V4|normal'].loc[a] - 1) * 100),
                     'V4(V4)': r((V4N['V4|normal'].loc[b] / V4N['V4|normal'].loc[a] - 1) * 100),
                     '0050': r((E50.loc[b] / E50.loc[a] - 1) * 100), '2330': r((A2330.loc[b] / A2330.loc[a] - 1) * 100)}
json.dump(res, open(OUT / 'hold.json', 'w'), ensure_ascii=False, indent=1)
print(json.dumps(res, ensure_ascii=False, indent=1))
