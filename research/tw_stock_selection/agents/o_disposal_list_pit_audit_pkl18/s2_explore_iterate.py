#!/usr/bin/env python3
"""S2b：探索（只用擬合樣本：2006-01～2018-12、2020-01～2023-12，排除 2019）與模型迭代（≤ 4 版）。
不產生任何 2019 月份的預測值（程式斷言設計矩陣不含 2019 列）。
python3 -I -B s2_explore_iterate.py → explore.json（S2 段）、iterations.json"""
import json, time, sys
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE)); import nblib as NB
F = pd.read_csv(HERE / 'data/monthly_features.csv'); F['month'] = pd.PeriodIndex(F.month, freq='M'); F = F.set_index('month')
fit_m = [m for m in F.index if (2006 <= m.year <= 2018) or (2020 <= m.year <= 2023)]
assert not any(m.year == 2019 for m in fit_m)
D = F.loc[fit_m].copy()
t = D.index.to_timestamp()
D['R14'] = (t >= '2014-04-01').astype(float); D['R15'] = (t >= '2015-06-01').astype(float); D['R17'] = (t >= '2017-05-01').astype(float)
D['COVID'] = ((t >= '2020-03-01') & (t <= '2020-12-31')).astype(float)
y = D['Y_start'].to_numpy(float)
# ---------------- 探索 ----------------
ex = {'fit_months': len(fit_m), 'y_mean': float(y.mean()), 'y_var': float(y.var()), 'var_over_mean': float(y.var() / y.mean()),
      'zero_months': int((y == 0).sum()), 'zero_month_list': [str(m) for m in D.index[y == 0]],
      'corr_spearman_Y_vs': {c: float(D['Y_start'].corr(D[c], method='spearman')) for c in ['FD_reg', 'FD_32', 'PDup_reg', 'VD', 'K_reg', 'K_32', 'K_reg_na', 'mkt_absret6_mean', 'n_stocks']},
      'regime_means': {nm: dict(Y=float(D.loc[m, 'Y_start'].mean()), K_reg=float(D.loc[m, 'K_reg'].mean()), FD_reg=float(D.loc[m, 'FD_reg'].mean()), ratio_Y_over_K=float(D.loc[m, 'Y_start'].sum() / max(D.loc[m, 'K_reg'].sum(), 1)))
                       for nm, m in {'2006-01~2014-03': t < '2014-04-01', '2014-04~2015-05': (t >= '2014-04-01') & (t < '2015-06-01'),
                                     '2015-06~2017-04': (t >= '2015-06-01') & (t < '2017-05-01'), '2017-05~2018-12': (t >= '2017-05-01') & (t < '2019-01-01'),
                                     '2020-01~2023-12': t >= '2020-01-01'}.items()},
      'yearly_fit_sample': {int(yy): dict(Y=int(g.Y_start.sum()), K_reg=int(g.K_reg.sum()), FD_reg=int(g.FD_reg.sum()), zero_months=int((g.Y_start == 0).sum()))
                            for yy, g in D.groupby(D.index.year)},
      'zero_month_given_K': {'months_Y0': int((y == 0).sum()), 'mean_K_reg_when_Y0': float(D.loc[y == 0, 'K_reg'].mean()) if (y == 0).any() else None,
                             'mean_K_reg_when_Y>0': float(D.loc[y > 0, 'K_reg'].mean())},
      'note': '只含擬合樣本；2019 不在任何統計內。'}
# ---------------- 迭代 ----------------
def X_of(D, spec):
    cols = [np.ones(len(D))]
    names = ['const']
    for v in spec['x']:
        if v.startswith('log1p_'): cols.append(np.log1p(D[v[6:]].to_numpy(float))); names.append(v)
        elif v == 'log_n_stocks': cols.append(np.log(D['n_stocks'].to_numpy(float))); names.append(v)
        else: cols.append(D[v].to_numpy(float)); names.append(v)
    return np.column_stack(cols), names
VERS = [
    dict(v=1, form='Poisson', nb=False, x=['log1p_FD_reg', 'R14', 'R15', 'R17', 'COVID'], why='基準：近似注意股票日數＋制度斷點'),
    dict(v=2, form='NB2', nb=True, x=['log1p_FD_reg', 'R14', 'R15', 'R17', 'COVID'], why='v1 的過度離散（var/mean>1）→ NB2'),
    dict(v=3, form='NB2', nb=True, x=['log1p_K_reg', 'R14', 'R15', 'R17', 'COVID'], why='改用「連續觸發」事件數（近似連續三次／五次、十日六次），比較接近處置的累計觸發機制'),
    dict(v=4, form='NB2', nb=True, x=['log1p_K_reg', 'log1p_FD_reg', 'R14', 'R15', 'R17', 'COVID'], why='事件數＋股票日數並列，檢查兩者是否互補'),
]
its = []
for spec in VERS:
    X, names = X_of(D, spec)
    Fm = NB.fit(X, y, nb=spec['nb'])
    m = NB.mu(Fm, X)
    se = np.sqrt(np.clip(np.diag(Fm['cov']), 0, None))
    a = float(np.exp(Fm['par'][-1])) if spec['nb'] else 0.0
    var = m + a * m ** 2
    pr = (y - m) / np.sqrt(var)
    Dy = pd.DataFrame({'y': y, 'm': m, 'var': var}, index=D.index).groupby(D.index.year).sum()
    yres = ((Dy.y - Dy.m) / np.sqrt(Dy['var'])).round(3)
    ac1 = float(pd.Series(pr).autocorr(1))
    its.append(dict(version=spec['v'], form=spec['form'], x=spec['x'], why=spec['why'], coef={n: [float(b), float(s)] for n, b, s in zip(names, Fm['par'], se)},
                    alpha=a, ll=float(Fm['ll']), aic=float(Fm['aic']), bic=float(Fm['bic']), converged=Fm['success'],
                    pearson_chi2_over_df=float((pr ** 2).sum() / (len(y) - len(Fm['par']))), resid_ac1=ac1,
                    yearly_std_resid_fit_sample={int(k): float(v) for k, v in yres.items()}, yearly_SSE=float((yres ** 2).sum()),
                    years_abs_gt2=[int(k) for k, v in yres.items() if abs(v) > 2]))
nbv = [i for i in its if i['form'] == 'NB2']
best = min(nbv, key=lambda i: i['aic'])
sel = dict(rule='NB2 版本中擬合樣本 AIC 最低者；若和次佳差 < 2，取變數較少者', chosen=best['version'])
alt = sorted(nbv, key=lambda i: i['aic'])
if len(alt) > 1 and alt[1]['aic'] - alt[0]['aic'] < 2 and len(alt[1]['x']) < len(alt[0]['x']): sel['chosen'] = alt[1]['version']
out = dict(utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), fit_sample='2006-01～2018-12、2020-01～2023-12（204 月；排除 2019）', n_versions=len(its),
           versions=its, selection=sel, note='沒有任何版本產生 2019 預測值；2019 預測只在 PREREG 寫入後由 s4_predict.py 產生')
(HERE / 'iterations.json').write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding='utf-8')
E = json.loads((HERE / 'explore.json').read_text(encoding='utf-8')) if (HERE / 'explore.json').exists() else {}
E['S2_model_explore_fit_sample_only'] = ex
(HERE / 'explore.json').write_text(json.dumps(E, ensure_ascii=False, indent=1), encoding='utf-8')
print(json.dumps(ex['corr_spearman_Y_vs'], indent=0)); print(json.dumps(ex['regime_means'], ensure_ascii=False, indent=0))
for i in its: print(i['version'], i['form'], round(i['aic'], 1), round(i['alpha'], 3), round(i['pearson_chi2_over_df'], 2), round(i['resid_ac1'], 2), round(i['yearly_SSE'], 1), i['years_abs_gt2'], {k: [round(x, 3) for x in v] for k, v in i['coef'].items()})
print(sel)
