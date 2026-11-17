#!/usr/bin/env python3
"""S60 第二步：讀 factors_monthly.csv，跑 SPEC 的 NW 迴歸、超額拆解、穩健性，輸出 results.json。python3 -I -B analyze.py"""
import json, hashlib
from pathlib import Path
import numpy as np, pandas as pd

H = Path(__file__).resolve().parent
for f in ('SPEC', 'PREREG'):
    assert hashlib.sha256((H / f'{f}.json').read_bytes()).hexdigest() == (H / f'{f}.sha256').read_text().split()[0], f'{f} 不符'
D = pd.read_csv(H / 'factors_monthly.csv', index_col=0)
LAG = 4
MODELS = {'M1_CAPM': ['MKT'], 'M2_FF3': ['MKT', 'SMB', 'HML'], 'M3_C4': ['MKT', 'SMB', 'HML', 'MOM'],
          'M4_C4REV': ['MKT', 'SMB', 'HML', 'MOM', 'RMOM'], 'M5_FF5': ['MKT', 'SMB', 'HML', 'RMW', 'CMA'],
          'M6_FF5MOMREV': ['MKT', 'SMB', 'HML', 'RMW', 'CMA', 'MOM', 'RMOM']}
PRE = {'P1_CAPM_pre': ['MKT'], 'P2_SMB_pre': ['MKT', 'SMB_MR'], 'P3_MOM_pre': ['MKT', 'SMB_MR', 'MOM'], 'P4_REV_pre': ['MKT', 'SMB_MR', 'MOM', 'RMOM']}
WIN = {'W_FULL': ('2015-07', '2026-10'), 'W_19': ('2019-01', '2026-10'), 'W_24': ('2024-01', '2026-10')}


def nw_ols(y, X, lag=LAG):
    n, k = X.shape
    XtX_inv = np.linalg.inv(X.T @ X)
    b = XtX_inv @ X.T @ y
    e = y - X @ b
    u = X * e[:, None]
    S = u.T @ u
    for l in range(1, lag + 1):
        w = 1 - l / (lag + 1)
        G = u[l:].T @ u[:-l]
        S += w * (G + G.T)
    V = XtX_inv @ S @ XtX_inv
    se = np.sqrt(np.diag(V))
    r2 = 1 - e @ e / np.sum((y - y.mean()) ** 2)
    return b, se, r2, e


def reg(df, ycol, fac, top_drop=0):
    d = df[[ycol, 'RF', 'R_0050'] + fac].dropna()
    if top_drop:
        d = d.drop(index=d[ycol].nlargest(top_drop).index)
    y = (d[ycol] - d['RF']).to_numpy()
    X = np.column_stack([np.ones(len(d))] + [d[f].to_numpy() for f in fac])
    b, se, r2, e = nw_ols(y, X)
    out = dict(n=int(len(d)), start=d.index[0], end=d.index[-1], alpha_ann=round(12 * 100 * b[0], 2), alpha_t=round(b[0] / se[0], 2),
               r2=round(r2, 3), beta={f: round(b[i + 1], 3) for i, f in enumerate(fac)}, beta_t={f: round(b[i + 1] / se[i + 1], 2) for i, f in enumerate(fac)})
    # 對 0050 超額拆解（算術年化百分點）：mean(R−R0050) = α + (β_MKT−1)·mean(MKT) + Σ b_k mean(F_k)
    ex = 1200 * float((d[ycol] - d['R_0050']).mean())
    dec = {'α': 1200 * b[0]}
    for i, f in enumerate(fac):
        bb = b[i + 1] - (1 if f in ('MKT', 'MKT_VW') else 0)
        dec[f + ('（β−1）' if f == 'MKT' else '')] = 1200 * bb * float(d[f].mean())
    if 'MKT' not in fac:          # MKT_VW 版：0050 不在右邊 → 另列「−0050 超額」項
        dec['−(0050−rf)'] = -1200 * float((d['R_0050'] - d['RF']).mean())
    out['excess_vs_0050_ann'] = round(ex, 2)
    out['excess_decomp'] = {k: round(v, 2) for k, v in dec.items()}
    out['decomp_check'] = round(sum(dec.values()) - ex, 6)
    out['total_ann_arith'] = round(1200 * float(d[ycol].mean()), 2)
    return out


R = {'meta': dict(nw_lag=LAG, rf='1%/yr', note='α 為 12×月 α（算術，百分點）；excess_decomp 各項加總＝對 0050 算術年化超額')}
# 因子溢酬
prem = {}
for w, (a, b) in {**WIN, 'W_PRE': ('2008-01', '2013-01'), 'W_ALL': ('2008-01', '2026-10')}.items():
    d = D.loc[a:b]
    prem[w] = {}
    for f in ('MKT', 'MKT_VW', 'SMB', 'SMB_MR', 'HML', 'MOM', 'RMOM', 'RMW', 'CMA', 'EPF'):
        x = d[f].dropna().to_numpy()
        if len(x) < 12:
            continue
        bb, se, _, _ = nw_ols(x, np.ones((len(x), 1)))
        prem[w][f] = dict(ann=round(1200 * bb[0], 2), t=round(bb[0] / se[0], 2), n=int(len(x)))
R['factor_premia'] = prem
R['factor_corr_W_FULL'] = D.loc['2015-07':'2026-10', ['MKT', 'SMB', 'HML', 'MOM', 'RMOM', 'RMW', 'CMA']].corr().round(2).to_dict()

main = {}
for s in ('R_E0', 'R_XT', 'R_SATA'):
    main[s] = {}
    for w, (a, b) in WIN.items():
        d = D.loc[a:b]
        main[s][w] = {m: reg(d, s, f) for m, f in MODELS.items() if not (w == 'W_24' and m not in ('M1_CAPM', 'M4_C4REV', 'M5_FF5'))}
R['main'] = main
R['explained_share'] = {s: {w: round(1 - main[s][w]['M4_C4REV']['alpha_ann'] / main[s][w]['M1_CAPM']['alpha_ann'], 3) for w in WIN} for s in main}

rob = {}
for w in ('W_FULL', 'W_19'):
    a, b = WIN[w]; d = D.loc[a:b]
    rob[w] = {f'drop_top5_{m}': reg(d, 'R_E0', MODELS[m], top_drop=5) for m in ('M1_CAPM', 'M4_C4REV', 'M5_FF5')}
    rob[w]['top5_months'] = list(d['R_E0'].nlargest(5).index)
    rob[w]['M4_MKT_VW'] = reg(d, 'R_E0', ['MKT_VW', 'SMB', 'HML', 'MOM', 'RMOM'])
    rob[w]['M4_HML_EPF'] = reg(d, 'R_E0', ['MKT', 'SMB', 'EPF', 'MOM', 'RMOM'])
    rob[w]['M4_SMB_FF5def'] = reg(d, 'R_E0', ['MKT', 'SMB_FF5', 'HML', 'MOM', 'RMOM'])
    rob[w]['M4_noHML'] = reg(d, 'R_E0', ['MKT', 'SMB', 'MOM', 'RMOM'])
R['robustness'] = rob

pre = {}
PW = {'W_PRE': [('2008-01', '2013-01')], 'W_PRE2': [('2008-01', '2015-06')], 'W_COMB': [('2008-01', '2013-01'), ('2015-07', '2026-10')],
      'W_FULL_premodels': [('2015-07', '2026-10')]}
for w, spans in PW.items():
    d = pd.concat([D.loc[a:b] for a, b in spans])
    pre[w] = {m: reg(d, 'R_E0', f) for m, f in PRE.items()}
R['pre2015_exploratory'] = pre
R['pre2015_note'] = 'S55 凍結規則淨值（待 R55 重現、存活者偏誤、2014 前除權息日報酬記 0）；2015 前無財報故無 HML／RMW／CMA；探索性，不寫入履歷'
(H / 'results.json').write_text(json.dumps(R, ensure_ascii=False, indent=1, default=str))

# 簡表
for s in ('R_E0', 'R_XT', 'R_SATA'):
    for w in WIN:
        for m, r in main[s][w].items():
            print(s, w, m, r['n'], r['alpha_ann'], r['alpha_t'], r['beta'], r['r2'], 'ex', r['excess_vs_0050_ann'], r['excess_decomp'], r['decomp_check'])
print(json.dumps(R['explained_share']))
print(json.dumps(prem, ensure_ascii=False))
for w in rob:
    for k, r in rob[w].items():
        print('ROB', w, k, r if isinstance(r, list) else (r['alpha_ann'], r['alpha_t'], r['beta'], r['r2']))
for w in pre:
    for m, r in pre[w].items():
        print('PRE', w, m, r['n'], r['alpha_ann'], r['alpha_t'], r['beta'], r['beta_t'], r['r2'], r['excess_vs_0050_ann'], r['excess_decomp'])
