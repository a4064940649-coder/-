#!/usr/bin/env python3
"""步驟 6：台積電回歸、容量、入選股名次中位數。"""
import sys, json, pickle
import numpy as np, pandas as pd
sys.path.insert(0, '/home/user/-/research/tw_stock_selection/agents/o_capacity_tsmc_gap/replicate_o1')
import rcore as R
from r2_tsmc import crosses
from rstat import PER, S, B, T

NAV = pd.read_parquet(R.HERE / 'navs_tsmc.parquet')
PK = pickle.load(open(R.HERE / 'picks_main.pkl', 'rb'))
out = {}
ps = R.win_returns(NAV['n10|w0|normal'], S); pb = R.win_returns(B, S); pt = R.win_returns(T, S); pe = R.win_returns(NAV['EW'], S)
y = ps - pb; x = pt - pb; z = pe - pb


def ols(yv, X):
    X = np.column_stack([np.ones(len(yv))] + X); b, *_ = np.linalg.lstsq(X, yv, rcond=None)
    res = yv - X @ b; r2 = 1 - (res ** 2).sum() / ((yv - yv.mean()) ** 2).sum()
    return b, r2


def m(idx, a, b): return (idx >= pd.Timestamp(a)) & (idx <= pd.Timestamp(b))
reg = {}
for nm, a, b in [('2006-2013', '2006-01-01', '2013-12-31'), ('2014-2020', '2014-01-01', '2020-12-31'), ('2021-2026', '2021-01-01', '2026-12-31'),
                 ('VAL', PER['VAL'][2], PER['VAL'][3]), ('SEL', PER['SEL'][2], PER['SEL'][3]), ('HOLD', PER['HOLD'][2], PER['HOLD'][3]), ('ALL', PER['ALL'][2], PER['ALL'][3])]:
    k = m(y.index, a, b)
    bb, r2 = ols(y[k].to_numpy(), [x[k].to_numpy()])
    b2, r22 = ols(y[k].to_numpy(), [x[k].to_numpy(), z[k].to_numpy()])
    reg[nm] = {'beta': round(bb[1], 3), 'R2': round(r2, 3), 'two_factor_beta_tsmc': round(b2[1], 3), 'beta_ew': round(b2[2], 3), 'R2_2f': round(r22, 3), 'n': int(k.sum())}
out['reg'] = reg
bV = reg['VAL']['beta']
kH = m(y.index, PER['HOLD'][2], PER['HOLD'][3]); kS = m(y.index, PER['SEL'][2], PER['SEL'][3]); kV = m(y.index, PER['VAL'][2], PER['VAL'][3])
bVf, _ = ols(y[kV].to_numpy(), [x[kV].to_numpy()])
resid = (y - bVf[1] * x)[kH]
out['HOLD_raw_excess_pp'] = round(y[kH].mean() * 100, 3)
out['HOLD_resid_pp'] = round(resid.mean() * 100, 3); out['HOLD_resid_t'] = round(R.tstat(resid), 3)
out['SEL_raw_excess_pp'] = round(y[kS].mean() * 100, 3)
out['HOLD_EW_minus_0050_pp'] = round(z[kH].mean() * 100, 3)

# 容量：部位 ÷ 20 日均成交值（訊號日前 20 日，含訊號日）
ADV20 = R.V.rolling(20, min_periods=1).mean()
rows = []
for d, pk in PK[(10, 0.0)].items():
    cr = crosses[d]; a = ADV20.loc[d].to_numpy(float)
    for j in pk:
        rows.append({'d': d, 'code': R.COLS[j], 'vrank': cr.vrank[j], 'adv20': a[j]})
E = pd.DataFrame(rows)
cap = {}
for aum in (5e7, 1e7):
    E['ratio'] = aum / 10 / E['adv20']
    for p in ('SEL', 'VAL', 'HOLD', 'ALL'):
        k = m(E['d'], PER[p][2], PER[p][3]); s = E.loc[k, 'ratio']
        cap[f'{int(aum/1e4)}萬_{p}'] = {'median_pct': round(s.median() * 100, 3), 'gt5pct': round((s > 0.05).mean() * 100, 2)}
    k = m(E['d'], PER['SEL'][2], PER['SEL'][3])
    for lo, hi in [(1, 50), (51, 100), (101, 200), (201, 300), (1, 100), (101, 300)]:
        g = k & (E['vrank'] >= lo) & (E['vrank'] <= hi)
        cap[f'{int(aum/1e4)}萬_SEL_{lo}-{hi}_gt5pct'] = round((E.loc[g, 'ratio'] > 0.05).mean() * 100, 1)
out['capacity'] = cap
# 2,800 億基金建 1% 部位、每日占成交值 10%：2026 年訊號日，依名次組
days = {}
for lo, hi in [(1, 50), (51, 100), (101, 200), (201, 300)]:
    vals_pool = []; vals_pick = []
    for d, cr in crosses.items():
        if d.year != 2026: continue
        a = ADV20.loc[d].to_numpy(float)
        g = np.where(cr.el & (cr.vrank >= lo) & (cr.vrank <= hi))[0]
        vals_pool += list(2.8e11 * 0.01 / (0.1 * a[g]))
        top = np.where(np.isfinite(cr.vrank) & (cr.vrank >= lo) & (cr.vrank <= hi))[0]
        vals_pick += list(2.8e11 * 0.01 / (0.1 * a[top]))
    days[f'{lo}-{hi}'] = {'pool_median': round(float(np.median(vals_pool)), 1), 'pool_mean': round(float(np.mean(vals_pool)), 1), 'top300_median': round(float(np.median(vals_pick)), 1)}
out['fund_days'] = days
E['year'] = E['d'].dt.year
out['pick_rank_median_by_year'] = {int(k): float(v) for k, v in E.groupby('year')['vrank'].median().items()}
json.dump(out, open(R.HERE / 'res_reg_cap.json', 'w'), ensure_ascii=False, indent=1)
print(json.dumps(out, ensure_ascii=False, indent=1))
