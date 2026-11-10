#!/usr/bin/env python3
"""S2c（凍結後）：斷言 PREREG 存在且雜湊相符，重估凍結模型，產生 2019 預測區間與 2004～2026 逐年殘差。只跑一次。
python3 -I -B s4_predict.py → results.json RESULTS_MODEL"""
import json, hashlib, sys, time
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE)); import nblib as NB
RJ = HERE / 'results.json'; R = json.loads(RJ.read_text(encoding='utf-8'))
PR = R['PREREG']; assert hashlib.sha256(json.dumps(PR, ensure_ascii=False, sort_keys=True).encode()).hexdigest() == R['PREREG_sha256']
assert 'RESULTS_MODEL' not in R, '只跑一次'
for f, h in PR['file_sha256'].items(): assert hashlib.sha256((HERE / f).read_bytes()).hexdigest() == h, f
F = pd.read_csv(HERE / 'data/monthly_features.csv'); F['month'] = pd.PeriodIndex(F.month, freq='M'); F = F.set_index('month')
t = F.index.to_timestamp()
F['R14'] = (t >= '2014-04-01').astype(float); F['R15'] = (t >= '2015-06-01').astype(float); F['R17'] = (t >= '2017-05-01').astype(float)
F['COVID'] = ((t >= '2020-03-01') & (t <= '2020-12-31')).astype(float)
def X_of(D, xs):
    cols = [np.ones(len(D))]
    for v in xs: cols.append(np.log1p(D[v[6:]].to_numpy(float)) if v.startswith('log1p_') else D[v].to_numpy(float))
    return np.column_stack(cols)
fit_m = [m for m in F.index if (2006 <= m.year <= 2018) or (2020 <= m.year <= 2023)]
IT = json.loads((HERE / 'iterations.json').read_text(encoding='utf-8'))
MF = PR['model_frozen']
y = F.loc[fit_m, 'Y_start'].to_numpy(float)
Fm = NB.fit(X_of(F.loc[fit_m], MF['x']), y, nb=True)
names = ['const'] + MF['x']
assert max(abs(Fm['par'][i] - MF['coef'][n][0]) for i, n in enumerate(names)) < 1e-6, '重估係數與 PREREG 不符'
# φ（擬合樣本逐年）
Xf = X_of(F.loc[fit_m], MF['x']); sim_f = NB.simulate(Fm, Xf, n=10000, seed=1)
yr_f = pd.Index([m.year for m in fit_m])
phi_z = []
for yy in sorted(set(yr_f)):
    s = sim_f[:, yr_f == yy].sum(axis=1); o = y[yr_f == yy].sum(); phi_z.append((o - s.mean()) / s.std())
phi = float(np.mean(np.square(phi_z)))
# 全期逐年（2004～2026；2024-01、02 面板缺口月份排除並註記）
valid = ~((t >= '2024-01-01') & (t < '2024-03-01'))
Fa = F[valid & (t >= '2004-07-01')]       # 60 日均量需暖身：2004 前半年特徵不完整
Xa = X_of(Fa, MF['x'])
sim = NB.simulate(Fm, Xa, n=10000, seed=2); sim0 = NB.simulate(Fm, Xa, n=10000, seed=3, param_unc=False)
yrs = pd.Index([m.year for m in Fa.index])
YR = {}
for yy in sorted(set(yrs)):
    k = yrs == yy; s = sim[:, k].sum(axis=1); s0 = sim0[:, k].sum(axis=1); o = int(Fa.loc[k, 'Y_start'].sum())
    E = float(s.mean()); sd = float(s.std())
    lo, hi = np.percentile(s, [2.5, 97.5]); lo0, hi0 = np.percentile(s0, [2.5, 97.5])
    z = (o - E) / sd
    YR[int(yy)] = dict(O=o, E=round(E, 2), PI95=[float(lo), float(hi)], PI95_noparam=[float(lo0), float(hi0)],
                       PI95_calibrated=[round(max(E - 1.96 * sd * np.sqrt(phi), 0), 2), round(E + 1.96 * sd * np.sqrt(phi), 2)],
                       z=round(z, 3), z_calibrated=round(z / np.sqrt(phi), 3), months=int(k.sum()), in_fit=bool((2006 <= yy <= 2018) or (2020 <= yy <= 2023)),
                       below_PI=bool(o < lo), above_PI=bool(o > hi), q_obs=float((s <= o).mean()))
# 2019 月份
m19 = [m for m in Fa.index if m.year == 2019]; k19 = np.array([m.year == 2019 for m in Fa.index])
mo = {str(m): dict(O=int(Fa.loc[m, 'Y_start']), E=round(float(sim[:, i].mean()), 2), P0=float((sim[:, i] == 0).mean()), K_reg=int(Fa.loc[m, 'K_reg']), FD_reg=int(Fa.loc[m, 'FD_reg']))
      for i, m in zip(np.flatnonzero(k19), m19)}
# 假設區間：v2～v4 的 2019 點預測（各自在擬合樣本重估）
assm = {}
for v in IT['versions']:
    if v['form'] != 'NB2': continue
    Fv = NB.fit(X_of(F.loc[fit_m], v['x']), y, nb=True)
    assm[v['version']] = round(float(NB.mu(Fv, X_of(F.loc[m19], v['x'])).sum()), 2)
# 整月空白檢定：2019 的 0 列月份在模型下的機率
p_zero = {m: d['P0'] for m, d in mo.items() if d['O'] == 0}
res = dict(utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), phi_annual=round(phi, 3), yearly=YR, y2019_months=mo,
           y2019=dict(O=YR[2019]['O'], E=YR[2019]['E'], PI95=YR[2019]['PI95'], PI95_calibrated=YR[2019]['PI95_calibrated'], z=YR[2019]['z'], q_obs=YR[2019]['q_obs'],
                      assumption_interval_points=assm, zero_month_P0=p_zero, verdict_primary='H_real（在主 PI95 內）' if not (YR[2019]['below_PI'] or YR[2019]['above_PI']) else ('低於主 PI95 下界' if YR[2019]['below_PI'] else '高於主 PI95 上界')),
           years_below_minus2sigma=[y_ for y_, d in YR.items() if d['z'] < -2], years_below_PI=[y_ for y_, d in YR.items() if d['below_PI']],
           note='2024 年排除 1～2 月（面板缺口）；2004 年只含 7～12 月（60 日均量暖身）。2026 含 2026-08-10 新制（模型無此斷點，樣本外）。')
R = json.loads(RJ.read_text(encoding='utf-8')); R['RESULTS_MODEL'] = res
RJ.write_text(json.dumps(R, ensure_ascii=False, indent=1), encoding='utf-8')
print('phi', phi)
for yy, d in YR.items(): print(yy, d['O'], d['E'], [round(x) for x in d['PI95']], d['PI95_calibrated'], d['z'], d['z_calibrated'], 'fit' if d['in_fit'] else 'OOS', 'BELOW' if d['below_PI'] else ('ABOVE' if d['above_PI'] else ''))
print(json.dumps(res['y2019'], ensure_ascii=False)); print(json.dumps(mo, ensure_ascii=False))
