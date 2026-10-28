#!/usr/bin/env python3
"""步驟 1：2019 年後誰在贏（描述，不調參）。2020-08～2026-10 ETF 月含息報酬（etf_tr_m 月底還原價）＋安聯台灣科技（訊號窗）＋00981A 淨值，
按對 0050 的月超額排序；對贏家做歸因迴歸：因子＝[0050、同池等權−0050、主策略−同池等權、2330−0050、電子業（台積電以外市值加權）−0050]。
主策略與同池等權用 V6（AB_V6|normal、AB_V6|EW；無 2024-01 缺口）。只定性影響設計，不用來選參數。寫 explore.json 的 STEP1。"""
import sys, json, math
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
import cmlib as L

nv6 = pd.read_parquet(L.RAW / 'navs' / 'navs_v6.parquet')
MAIN, EW = nv6['AB_V6|normal'], nv6['AB_V6|EW']
# 月底日期：面板月底（2024-01 面板缺 → V6 有官方日 K；用 V6 的月底）
me = MAIN.dropna().groupby(MAIN.dropna().index.to_period('M')).apply(lambda s: s.index[-1])
me = me[(me.index >= pd.Period('2020-08', 'M')) & (me.index <= pd.Period('2026-09', 'M'))]
def mret(series):
    s = series.dropna(); s = s.reindex(s.index.union(pd.DatetimeIndex(me.values))).ffill().reindex(pd.DatetimeIndex(me.values))
    r = s.pct_change(); r.index = me.index; return r
F = pd.DataFrame({'R0050': mret(L.E50), 'EW': mret(EW), 'MAIN': mret(MAIN)})
# 2330 與電子業（台積電以外、前一日市值加權）日報酬：用 exfix 的 G；2024-01 面板缺 → 該月與 2 月合併由 ffill 處理（月報酬只在有面板的月底算）
G = L.grow_exfix(); MC = L.shares_est() * L.CFF
el = np.where(L.ELECMASK)[0]; el = el[el != L.J2330]
r2330 = pd.Series(G[:, L.J2330] - 1, index=L.IDX)
mcp = np.vstack([np.full(len(el), np.nan), MC[:-1, el]])
wE = np.where(np.isfinite(mcp), mcp, 0.0); rE = np.where(np.isfinite(G[:, el]), G[:, el] - 1, 0.0)
relec = pd.Series((wE * rE).sum(axis=1) / np.where(wE.sum(axis=1) > 0, wE.sum(axis=1), np.nan), index=L.IDX)
def daily_to_m(r):
    lvl = (1 + r.fillna(0)).cumprod(); return mret(lvl)
F['R2330'] = daily_to_m(r2330); F['ELEC'] = daily_to_m(relec)
F = F.dropna()
X = pd.DataFrame({'0050': F.R0050, 'EW-0050': F.EW - F.R0050, 'MAIN-EW': F.MAIN - F.EW, '2330-0050': F.R2330 - F.R0050, 'ELEC-0050': F.ELEC - F.R0050})

def ols(y, X):
    d = pd.concat([y.rename('y'), X], axis=1).dropna()
    if len(d) < 18: return None
    A = np.column_stack([np.ones(len(d)), d[X.columns].to_numpy()]); yy = d.y.to_numpy()
    b, *_ = np.linalg.lstsq(A, yy, rcond=None); res = yy - A @ b
    s2 = res @ res / (len(d) - A.shape[1]); se = np.sqrt(np.diag(s2 * np.linalg.inv(A.T @ A)))
    r2 = 1 - res @ res / ((yy - yy.mean()) @ (yy - yy.mean()))
    ex = d.y - d['0050']
    expl = {c: float(b[i + 1] * d[c].mean() * 100) for i, c in enumerate(X.columns)}
    return {'n': int(len(d)), 'alpha_pp_m': float(b[0] * 100), 't_alpha': float(b[0] / se[0]),
            'betas': {c: round(float(b[i + 1]), 3) for i, c in enumerate(X.columns)},
            't_betas': {c: round(float(b[i + 1] / se[i + 1]), 2) for i, c in enumerate(X.columns)}, 'R2': float(r2),
            'mean_excess_vs_0050_pp_m': float(ex.mean() * 100), 'factor_contrib_pp_m': expl}

e = pd.read_csv(L.RAW / 'screener' / 'etf_tr_m.csv.gz', dtype={'code': str})
lv = e.pivot_table(index='ym', columns='code', values='tr', aggfunc='last'); lv.index = pd.PeriodIndex(lv.index, freq='M')
R = lv.pct_change()
R = R.where(R.abs() < 0.4)          # 分割或資料錯誤的跳點（|月報酬| ≥ 40%）設缺
R = R.reindex(me.index)
rows = []
for c in R.columns:
    r = R[c].dropna()
    if len(r) < 24: continue
    x = (r - F.R0050.reindex(r.index)).dropna()
    rows.append({'code': c, 'n': int(len(x)), 'mean_ex_pp_m': float(x.mean() * 100), 't': L.tstat(x), 'first': str(r.index[0])})
rk = pd.DataFrame(rows).sort_values('mean_ex_pp_m', ascending=False)
named = ['0050', '0052', '006208', '00733', '00881', '00891', '00892', '00919', '00929', '00940', '00981A', '0056', '00878', '00713']
attr = {}
for c in list(rk.code.head(10)) + named:
    if c in R.columns and c not in attr:
        o = ols(R[c], X)
        if o: attr[c] = o
# 00981A（淨值，2025-10 起，太短只描述）
n981 = pd.read_csv(L.RAW / 'winners' / 'nav_00981A.csv'); n981.columns = ['date', 'nav']; n981['date'] = pd.to_datetime(n981.date)
s981 = n981.set_index('date').nav; r981 = mret(s981).dropna()
d981 = {'n_months': int(len(r981)), 'mean_ex_pp_m': float((r981 - F.R0050.reindex(r981.index)).mean() * 100) if len(r981) else None}
# 安聯台灣科技（訊號窗月報酬，2006-04 起）：用同窗的 0050、V6 主策略與等權
am = pd.read_csv(L.RAW / 'winners' / 'allianz_monthly.csv'); am = am.rename(columns={am.columns[0]: 'd'}); am['d'] = pd.to_datetime(am.d)
sd = list(am.d) + [pd.Timestamp('2026-10-12')]
def wr(nav):
    s = nav.dropna(); s = s.reindex(s.index.union(pd.DatetimeIndex(sd))).ffill().reindex(pd.DatetimeIndex(sd))
    return (s.shift(-1) / s - 1).iloc[:-1]
lvl2330 = (1 + r2330.fillna(0)).cumprod(); lvlE = (1 + relec.fillna(0)).cumprod()
FW = pd.DataFrame({'0050': wr(L.E50), 'EW': wr(EW), 'MAIN': wr(MAIN), '2330': wr(lvl2330), 'ELEC': wr(lvlE)})
XW = pd.DataFrame({'0050': FW['0050'], 'EW-0050': FW.EW - FW['0050'], 'MAIN-EW': FW.MAIN - FW.EW, '2330-0050': FW['2330'] - FW['0050'], 'ELEC-0050': FW.ELEC - FW['0050']})
fund = am.set_index('d').FUND
alz = {}
for lab, a, b in [('2006-2018', '2006-01-01', '2018-12-31'), ('2019-2026', '2019-01-01', '2026-12-31'), ('2020-08-2026', '2020-08-01', '2026-12-31')]:
    m = (fund.index >= a) & (fund.index <= b)
    alz[lab] = ols(fund[m], XW[m])
out = json.loads((L.HERE / 'explore.json').read_text(encoding='utf-8'))
out['STEP1'] = {'note': '描述用；2020-08～2026-09 月報酬（含驗證期），只定性影響題目設計，不用來選參數。因子：0050、同池等權−0050、主策略(V6 AB)−同池等權、2330−0050、電子業(台積電以外市值加權)−0050。',
                'factor_means_pp_m': {c: float(X[c].mean() * 100) for c in X.columns}, 'factor_corr': X.corr().round(2).to_dict(),
                'etf_rank_top15': rk.head(15).to_dict('records'), 'etf_rank_bottom5': rk.tail(5).to_dict('records'), 'n_etf_ranked': int(len(rk)),
                'attribution': attr, 'allianz_tw_tech': alz, '00981A_nav': d981}
(L.HERE / 'explore.json').write_text(L.dumps(out), encoding='utf-8')
print(rk.head(15).to_string())
for c, o in attr.items(): print(c, o['n'], 'ex %.2f alpha %.2f t %.2f R2 %.2f' % (o['mean_excess_vs_0050_pp_m'], o['alpha_pp_m'], o['t_alpha'], o['R2']), o['betas'])
for k, o in alz.items(): print('ALZ', k, o and ('ex %.2f alpha %.2f t %.2f R2 %.2f' % (o['mean_excess_vs_0050_pp_m'], o['alpha_pp_m'], o['t_alpha'], o['R2'])), o and o['betas'])
print(out['STEP1']['factor_means_pp_m'])
