#!/usr/bin/env python3
"""S24 第二步：讀 monthly.csv，照 SPEC.json 跑 CAPM、雙 β、滾動 β、多因子（M1～M5）、2019 起報酬拆解、敏感度、去掉最賺 5 個月，輸出 results.json。
python3 -I -B analyze.py"""
import json, hashlib
from pathlib import Path
import numpy as np, pandas as pd

H = Path(__file__).resolve().parent
assert hashlib.sha256((H / 'SPEC.json').read_bytes()).hexdigest() == (H / 'SPEC.sha256').read_text().split()[0]
D = pd.read_csv(H / 'monthly.csv', index_col='month')
D.index = pd.PeriodIndex(D.index, freq='M')
META = json.loads((H / 'build_meta.json').read_text())
SER = ('FULL', 'SAT50', 'SATCASH')
PER = {'full': '2015-07', 'since2019': '2019-01', 'since2024': '2024-01'}
LAG = 4
MODELS = {'M1_FF3': ['MKT', 'SMB', 'HML'], 'M2_C4': ['MKT', 'SMB', 'HML', 'MOM'], 'M3_MAIN': ['MKT', 'SMB', 'HML', 'MOM', 'RMOM'],
          'M4_EP': ['MKT', 'SMB', 'EPF', 'MOM', 'RMOM'], 'M5_VWMKT': ['MKTVW', 'SMB', 'HML', 'MOM', 'RMOM']}
rfm = lambda a: (1 + a) ** (1 / 12) - 1


def ols(y, X, lag=LAG, const=True):
    """OLS＋Newey-West（Bartlett）HAC；不做小樣本修正。回傳 dict(b, se_nw, t_nw, t_ols, r2, n, resid)。"""
    y = np.asarray(y, float); X = np.asarray(X, float)
    if X.ndim == 1:
        X = X[:, None]
    if const:
        X = np.column_stack([np.ones(len(y)), X])
    n, k = X.shape
    XtXi = np.linalg.inv(X.T @ X)
    b = XtXi @ X.T @ y
    e = y - X @ b
    u = X * e[:, None]
    S = u.T @ u
    for l in range(1, lag + 1):
        w = 1 - l / (lag + 1)
        G = u[l:].T @ u[:-l]
        S += w * (G + G.T)
    V = XtXi @ S @ XtXi
    se = np.sqrt(np.diag(V))
    s2 = e @ e / (n - k)
    se_ols = np.sqrt(np.diag(XtXi) * s2)
    r2 = 1 - (e @ e) / np.sum((y - y.mean()) ** 2)
    return dict(b=b, se=se, t=b / se, t_ols=b / se_ols, r2=float(r2), n=int(n), resid=e, V=V)


def frame(rf_annual=0.01):
    d = D.copy()
    d['RF'] = rfm(rf_annual)
    d['MKT'] = d['R_0050'] - d['RF']
    d['MKTVW'] = d['MKT_VW'] - d['RF']
    for s in SER:
        d['X_' + s] = d['R_' + s] - d['RF']
    return d


def sub(d, per):
    return d[d.index >= pd.Period(PER[per], 'M')]


def capm(d, s, lag=LAG):
    r = ols(d['X_' + s], d['MKT'], lag)
    return dict(n=r['n'], alpha_ann=float(r['b'][0] * 12), alpha_t=float(r['t'][0]), alpha_t_ols=float(r['t_ols'][0]),
                beta=float(r['b'][1]), beta_t=float(r['t'][1]), r2=r['r2'])


def updown(d, s):
    x = d['MKT'].to_numpy(); y = d['X_' + s].to_numpy()
    X = np.column_stack([np.maximum(x, 0), np.minimum(x, 0)])
    r = ols(y, X)
    c = np.array([0, 1, -1.0]); diff = c @ r['b']; se = np.sqrt(c @ r['V'] @ c)
    up = x > 0
    ru = ols(y[up], x[up]); rd = ols(y[~up], x[~up])
    return dict(beta_up=float(r['b'][1]), t_up=float(r['t'][1]), beta_down=float(r['b'][2]), t_down=float(r['t'][2]),
                diff=float(diff), diff_t=float(diff / se), alpha_ann=float(r['b'][0] * 12), alpha_t=float(r['t'][0]), r2=r['r2'],
                n_up=int(up.sum()), n_down=int((~up).sum()),
                separate=dict(up=dict(beta=float(ru['b'][1]), t=float(ru['t'][1]), alpha_m=float(ru['b'][0]), r2=ru['r2']),
                              down=dict(beta=float(rd['b'][1]), t=float(rd['t'][1]), alpha_m=float(rd['b'][0]), r2=rd['r2'])))


def mf(d, s, facs, lag=LAG):
    dd = d.dropna(subset=facs)
    r = ols(dd['X_' + s], dd[facs], lag)
    o = dict(n=r['n'], start=str(dd.index[0]), end=str(dd.index[-1]), alpha_ann=float(r['b'][0] * 12), alpha_t=float(r['t'][0]),
             alpha_t_ols=float(r['t_ols'][0]), r2=r['r2'])
    o['loadings'] = {f: dict(b=float(r['b'][i + 1]), t=float(r['t'][i + 1])) for i, f in enumerate(facs)}
    return o, r, dd


def ann_geo(r):
    r = np.asarray(r)
    return float(np.prod(1 + r) ** (12 / len(r)) - 1)


res = dict(spec_sha256=(H / 'SPEC.sha256').read_text().split()[0], build_meta={k: META[k] for k in ('acceptance_nav', 'universe_counts', 'corner_min', 'factor_first_valid')})
d = frame(0.01)
res['spec_deviations'] = [
    'MOM（12-1）需要形成日前 12 個月的含息指數，面板從 2015-01-05 起，所以 MOM 只能從 2016-02 起有值（SPEC 寫 2015-07）。含 MOM 的模型（M2～M5）全期樣本實際是 2016-02～2026-10（129 個月）；M1_FF3 與 CAPM 用 2015-07 起。',
]
# 基本統計
bs = {}
for per in PER:
    x = sub(d, per)
    bs[per] = {k: dict(arith_ann=float(x[c].mean() * 12), geo_ann=ann_geo(x[c]), vol_ann=float(x[c].std(ddof=1) * np.sqrt(12)), n=int(len(x)))
               for k, c in (('FULL', 'R_FULL'), ('SAT50', 'R_SAT50'), ('SATCASH', 'R_SATCASH'), ('0050', 'R_0050'))}
res['basic'] = bs
res['acceptance_0050_since2019_geo'] = bs['since2019']['0050']['geo_ann']
# CAPM
res['capm'] = {s: {per: capm(sub(d, per), s) for per in PER} for s in SER}
res['updown'] = {s: {per: updown(sub(d, per), s) for per in PER} for s in SER}
# 滾動 36 月 β
roll = {}
for s in ('FULL', 'SAT50'):
    vals = {}
    for i in range(36, len(d) + 1):
        w = d.iloc[i - 36:i]
        vals[str(w.index[-1])] = float(ols(w['X_' + s], w['MKT'], 0)['b'][1])
    v = pd.Series(vals)
    roll[s] = dict(min=float(v.min()), min_at=v.idxmin(), max=float(v.max()), max_at=v.idxmax(), median=float(v.median()),
                   at={k: float(v[k]) for k in ('2018-06', '2019-12', '2020-12', '2021-12', '2022-12', '2023-12', '2024-12', '2025-06', '2025-12', '2026-10') if k in v},
                   series=v.round(4).to_dict())
res['rolling36_beta'] = roll
# 因子統計
facs_all = ['MKT', 'MKTVW', 'SMB', 'HML', 'EPF', 'MOM', 'RMOM', 'SMB_HML', 'SMB_MOM', 'SMB_RMOM']
fs = {}
for per in PER:
    x = sub(d, per)
    o = {}
    for f in facs_all:
        y = x[f].dropna()
        r = ols(y.to_numpy(), np.zeros((len(y), 0)), LAG)
        o[f] = dict(mean_ann=float(y.mean() * 12), t_nw=float(r['t'][0]), n=int(len(y)), start=str(y.index[0]))
    fs[per] = o
res['factor_stats'] = fs
cx = d[['MKT', 'SMB', 'HML', 'EPF', 'MOM', 'RMOM']].dropna()
res['factor_corr_2016_02'] = cx.corr().round(3).to_dict()
# 多因子
res['multifactor'] = {m: {s: {per: mf(sub(d, per), s, fl)[0] for per in PER} for s in SER} for m, fl in MODELS.items()}


# ---------------------------------------------------------------- 拆解（2019 起，算術）
def decomp(dd_est, dd_use, s, facs):
    """用 dd_est 估係數，在 dd_use 上拆平均：mean(R)＝mean(Rf)＋β mean(MKT)＋Σ b mean(F)＋α＋殘差平均。年化 ×12。"""
    dd_est = dd_est.dropna(subset=facs); dd_use = dd_use.dropna(subset=facs)
    r = ols(dd_est['X_' + s], dd_est[facs])
    rows = dict(total=float(dd_use['R_' + s].mean() * 12), rf=float(dd_use['RF'].mean() * 12))
    for i, f in enumerate(facs):
        rows[f] = float(r['b'][i + 1] * dd_use[f].mean() * 12)
    rows['alpha'] = float(r['b'][0] * 12)
    rows['residual'] = rows['total'] - rows['rf'] - sum(rows[f] for f in facs) - rows['alpha']
    rows['alpha_t'] = float(r['t'][0])
    rows['loadings'] = {f: float(r['b'][i + 1]) for i, f in enumerate(facs)}
    rows['factor_mean_ann'] = {f: float(dd_use[f].mean() * 12) for f in facs}
    rows['geo_ann'] = ann_geo(dd_use['R_' + s]); rows['arith_ann'] = rows['total']
    rows['vol_ann'] = float(dd_use['R_' + s].std(ddof=1) * np.sqrt(12))
    rows['half_var'] = float(0.5 * dd_use['R_' + s].var(ddof=1) * 12)
    rows['n'] = int(len(dd_use))
    return rows


x19 = sub(d, 'since2019'); xfull = d
dec = {}
for s in SER:
    dec[s] = dict(M3_same=decomp(x19, x19, s, MODELS['M3_MAIN']), CAPM_same=decomp(x19, x19, s, ['MKT']),
                  M3_fullbeta=decomp(xfull, x19, s, MODELS['M3_MAIN']), M4_same=decomp(x19, x19, s, MODELS['M4_EP']),
                  M3_same_since2024=decomp(sub(d, 'since2024'), sub(d, 'since2024'), s, MODELS['M3_MAIN']),
                  M3_same_full=decomp(xfull, xfull, s, MODELS['M3_MAIN']))
dec['0050'] = dict(arith_ann=float(x19['R_0050'].mean() * 12), geo_ann=ann_geo(x19['R_0050']))
res['decomposition'] = dec


# ---------------------------------------------------------------- 穩健性
def drop_top(dd, s, col):
    v = dd[col] if col != 'excess_m' else dd['R_' + s] - dd['R_0050']
    top = v.sort_values(ascending=False).index[:5]
    return dd.drop(index=top), [str(t) for t in top], [float(v[t]) for t in top]


rob = {}
for s in ('FULL', 'SAT50', 'SATCASH'):
    o = {}
    for per in ('full', 'since2019'):
        x = sub(d, per)
        for tag, col in (('top5_ret', 'R_' + s), ('top5_excess', 'excess_m')):
            xd, months, vals = drop_top(x, s, col)
            o[f'{per}|{tag}'] = dict(months=months, values=vals, capm=capm(xd, s), M3=mf(xd, s, MODELS['M3_MAIN'])[0],
                                     dec_M3=decomp(xd, xd, s, MODELS['M3_MAIN']) if per == 'since2019' else None)
    rob[s] = o
res['robust_drop_top5'] = rob
sens = {}
for rfa in (0.0, 0.015):
    dr = frame(rfa)
    sens[f'rf_{rfa}'] = {s: {per: dict(capm=capm(sub(dr, per), s), M3=mf(sub(dr, per), s, MODELS['M3_MAIN'])[0]) for per in ('full', 'since2019')} for s in ('FULL', 'SAT50')}
for lag in (0, 12):
    sens[f'nwlag_{lag}'] = {s: {per: dict(capm=capm(sub(d, per), s, lag), M3=mf(sub(d, per), s, MODELS['M3_MAIN'], lag)[0]) for per in ('full', 'since2019')} for s in ('FULL', 'SAT50')}
res['sensitivity'] = sens
# S12 對照：R_FULL − R_0050 對 (股池等權 − 0050)
sc = {}
for per in PER:
    x = sub(d, per)
    for ew in ('EW_POOL', 'EW_ALL'):
        r = ols(x['R_FULL'] - x['R_0050'], x[ew] - x['R_0050'])
        sc[f'{per}|{ew}'] = dict(beta=float(r['b'][1]), t=float(r['t'][1]), alpha_ann=float(r['b'][0] * 12), alpha_t=float(r['t'][0]), r2=r['r2'],
                                 sc_mean_ann=float((x[ew] - x['R_0050']).mean() * 12))
res['s12_smallcap_check'] = sc
# 最賺月份清單
res['top_months'] = {s: sub(d, 'full')['R_' + s].sort_values(ascending=False).head(8).round(4).astype(float).to_dict() for s in ('FULL', 'SAT50')}
res['top_months'] = {s: {str(k): v for k, v in m.items()} for s, m in res['top_months'].items()}
(H / 'results.json').write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str))
print('ok')
