#!/usr/bin/env python3
"""S96 第 0 步（登記前）：①重現正式標準路徑（48.86%／34.46%／381 筆／對網站 navs_official 逐日差 0）；②估值資料抽查（2016、2019、2022 各 3 日 × 20 檔，
用日 K 收盤＋資產負債表每股淨值＋財報近四季淨利自行估算）與涵蓋率（data_check.json）；③只看訓練期（訊號日 2016-02～2021-12）現行已平倉交易的估值分組描述。
任何加了估值條件的模擬都不在這裡跑。用法：python3 -I -B step0.py"""
import sys, json, time, resource
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import s96lib as L
from s96lib import stlib
t0 = time.time(); rss = lambda: round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
P, U, RT = L.load()
lastb = lambda d: P.IDX[P.IDX < pd.Timestamp(d)][-1]
D0, DE, D19 = P.IDX[P.t0], P.IDX[-1], lastb('2019-01-01')
S0 = L.signal(P, U, RT, 0)
nav, lg, st = L.sim(P, S0, U)
site = pd.read_csv('/home/user/-/xq_scripts/site/navs_official.csv.gz', index_col=0, parse_dates=True)['E0|W10_70']
A0 = dict(since2019=stlib.cagr(nav, D19, DE), full=stlib.cagr(nav, D0, DE), maxdd=stlib.maxdd(nav, D0, DE), n_closed=int((lg['why'] != 'open_end').sum()), n_all=int(len(lg)),
          site_same_index=bool(site.index.equals(nav.index)), max_abs_diff_vs_site_nav=float((nav.round(2) - site).abs().max()), panel=[int(P.T), int(P.N), str(DE.date())])
A0['pass'] = bool(abs(A0['since2019'] - L.TARGET['since2019']) < 1e-12 and abs(A0['full'] - L.TARGET['full']) < 1e-12 and A0['n_closed'] == 381 and A0['n_all'] == 389
                  and A0['site_same_index'] and A0['max_abs_diff_vs_site_nav'] == 0)
print('A0', A0, f'{time.time()-t0:.0f}s', rss(), 'MB', flush=True)
assert A0['pass']

# ---- 估值矩陣與抽查
VL, chk = L.build_val(P)
np.savez_compressed(L.VAL_NPZ, **VL)
VW = L.Val(P, U, VL)
dc = dict(build=chk)
sys.path.insert(0, str(L.RAW))
from tsd_extra_load import load as tload
bv = tload('bs_hist'); bv = bv[bv['item'] == 'bvps_ref'][['period', 'code', 'value']].drop_duplicates(['period', 'code']).set_index(['period', 'code'])['value']
fx = tload('fs_xbrl', columns=['code', 'period', 'report', 'net_income_ytd'])
fx = fx.sort_values('report').drop_duplicates(['code', 'period'], keep='first').set_index(['period', 'code'])['net_income_ytd']   # cr（合併）優先
rng = np.random.default_rng(96); spot = []; rows = []
for y in (2016, 2019, 2022):
    for md, q in (('06-15', (y, 1)), ('09-15', (y, 2)), ('12-15', (y, 3))):
        d = str(P.IDX[P.IDX >= pd.Timestamp(f'{y}-{md}')][0].date())
        pr = tload('per_twse', d, d); pr = pr[pr['code'].str.fullmatch(r'[1-9]\d{3}') & pr['per'].notna() & pr['pbr'].notna()]
        dk = tload('daily', d, d, columns=['date', 'code', 'market', 'close', 'shares']); dk = dk[dk['market'] == 'twse'].set_index('code')
        pick = sorted(rng.choice(pr['code'].to_numpy(), size=20, replace=False))
        Y, Q = q; per_now, per_py, per_pq4 = f'{Y}Q{Q}', f'{Y-1}Q{Q}', f'{Y-1}Q4'
        for c in pick:
            r = pr[pr['code'] == c].iloc[0]; cl = float(dk.loc[c, 'close']); sh = float(dk.loc[c, 'shares'])
            b = bv.get((per_now, c), np.nan)
            ni = fx.get((per_now, c), np.nan) + fx.get((per_pq4, c), np.nan) - fx.get((per_py, c), np.nan)
            eps = ni / sh if sh > 0 else np.nan
            rows.append(dict(date=d, code=c, fs_quarter=str(r['fs_quarter']), assumed_quarter=per_now, close=cl, per=round(float(r['per']), 2), pbr=round(float(r['pbr']), 2), yld=round(float(r['yield_pct']), 2),
                             pbr_est=cl / b if b and b > 0 else np.nan, per_est=cl / eps if eps and eps > 0 else np.nan, eps_ttm_est=eps))
sp = pd.DataFrame(rows)
sp['pbr_ratio'] = sp['pbr'] / sp['pbr_est']; sp['per_ratio'] = sp['per'] / sp['per_est']
sp.round(4).to_csv(HERE / 'spot_check.csv', index=False)
def sm(x):
    x = x.dropna()
    return dict(n=int(len(x)), median_ratio=float(x.median()), within_5pct=float(((x - 1).abs() <= 0.05).mean()), within_25pct=float(((x - 1).abs() <= 0.25).mean()),
                within_x2=float(((x >= 0.5) & (x <= 2)).mean()), min=float(x.min()), max=float(x.max()))
dc['spot_check'] = dict(note='官方值 ÷ 自估值。自估淨值比＝日 K 收盤 ÷ 資產負債表每股參考淨值（假設用當季已公告的最新一季）；自估本益比＝日 K 收盤 ÷（近四季合併淨利 ÷ 當日發行股數）。'
                             '自估本益比用的是合併總淨利（含少數股權）與當日股數，和官方用的歸屬母公司每股盈餘有口徑差，只看量級。',
                        by_year={str(y): dict(pbr=sm(sp.loc[sp['date'].str[:4] == str(y), 'pbr_ratio']), per=sm(sp.loc[sp['date'].str[:4] == str(y), 'per_ratio'])) for y in (2016, 2019, 2022)},
                        all=dict(pbr=sm(sp['pbr_ratio']), per=sm(sp['per_ratio'])), dates=sorted(sp['date'].unique()), fs_quarter_values=sp.groupby('date')['fs_quarter'].agg(lambda v: v.value_counts().index[0]).to_dict())
del bv, fx
# ---- 涵蓋率
x = L.trade_features(P, VW, lg)
dc['coverage_trades'] = dict(n_all=int(len(x)), has_row=int(x['row'].sum()), has_pbr=int(x['pbr'].notna().sum()), has_yld=int(x['yld'].notna().sum()), per_positive=int(x['per_ok'].sum()),
                             per_null_or_loss=int((~x['per_ok']).sum()), yld_zero=int((x['yld'] == 0).sum()), has_per3y=int(x['per3y'].notna().sum()),
                             signal_days=len(S0['info']), signal_days_with_val_file=int(sum(VW.k2[i['k']] >= 0 for i in S0['info'])),
                             shift3_days_with_val_file=int(sum(VW.k2[min(i['k'] + 3, P.T - 1)] >= 0 for i in S0['info'])))
yrs = pd.Series(P.IDX.year)
dc['coverage_universe'] = {int(y): dict(frac_univ_with_row=float(np.mean([VW.ROW[VW.k2[k]][U[k]].mean() for k in np.flatnonzero((yrs == y).to_numpy()) if VW.k2[k] >= 0 and U[k].any()])),
                                        frac_univ_per_positive=float(np.mean([(VW.PER[VW.k2[k]][U[k]] > 0).mean() for k in np.flatnonzero((yrs == y).to_numpy()) if VW.k2[k] >= 0 and U[k].any()])))
                           for y in sorted(yrs.unique())}
(HERE / 'data_check.json').write_text(json.dumps(dc, ensure_ascii=False, indent=1, default=str))
print(json.dumps(dc['spot_check']['by_year'], ensure_ascii=False)); print(json.dumps(dc['spot_check']['all'])); print(dc['spot_check']['fs_quarter_values']); print(dc['coverage_trades']); print({k: v for k, v in chk.items()})

# ---- 訓練期描述（只有訓練期）
TA, TB = lastb('2016-02-01'), lastb('2022-01-01')
x = x[(x['why'] != 'open_end') & (x['sig_date'] > TA) & (x['sig_date'] <= TB)].copy()
n5 = int(round(len(x) * 0.05))
x['bigwin'] = x['ret'].rank(ascending=False, method='first') <= n5; x['biglose'] = x['ret'].rank(ascending=True, method='first') <= n5
x['top10pnl'] = x['pnl'].rank(ascending=False, method='first') <= 10


def grp(lab):
    o = {}
    for g, y in x.groupby(lab, observed=True):
        o[str(g)] = dict(n=int(len(y)), win=float((y['ret'] > 0).mean()), mean=float(y['ret'].mean()), median=float(y['ret'].median()), pnl_sum=float(y['pnl'].sum()),
                         bigwin=int(y['bigwin'].sum()), biglose=int(y['biglose'].sum()), top10pnl=int(y['top10pnl'].sum()))
    return o


D = {}; CUT = {}
x['g'] = np.where(~x['per_ok'], '0虧損或無', np.where(x['per'] < 15, '1<15', np.where(x['per'] <= 30, '2 15~30', '3>30'))); D['per_bins'] = grp('g')
x['g'] = np.where(~x['per_ok'], '0虧損或無', np.where(x['per'] <= 30, '1≤30', np.where(x['per'] <= 60, '2 30~60', '3>60'))); D['per_bins_high'] = grp('g')
for c, nm in (('pbr', 'pbr_tertile'), ('yld', 'yld_tertile')):
    q1, q2 = x[c].quantile([1 / 3, 2 / 3]); CUT[c] = [float(q1), float(q2)]
    x['g'] = np.where(x[c].isna(), '無', np.where(x[c] <= q1, f'1低(≤{q1:.2f})', np.where(x[c] <= q2, f'2中(≤{q2:.2f})', f'3高(>{q2:.2f})'))); D[nm] = grp('g')
x['g'] = np.where(x['yld'] == 0, '殖利率0', '有配息'); D['yld_zero'] = grp('g')
for c in ('p_pbr', 'p_yld', 'p_per'):
    x['g'] = np.where(x[c].isna(), '無', pd.cut(x[c], [-0.01, 1 / 3, 2 / 3, 0.8, 1.01], labels=['1股池最低1/3', '2中1/3', '3 67~80%', '4最高20%']).astype(str)); D[c] = grp('g')
x['g'] = np.where(x['per3y'].isna(), np.where(x['per_ok'], '歷史不足', '0虧損或無'), pd.cut(x['per3y'], [-0.01, 1 / 3, 2 / 3, 0.9, 1.01], labels=['1自己3年低1/3', '2中1/3', '3 67~90%', '4最高10%']).astype(str)); D['per3y'] = grp('g')
corr = {c: float(x[[c, 'ret']].corr(method='spearman').iloc[0, 1]) for c in ('per', 'pbr', 'yld', 'p_pbr', 'p_yld', 'per3y')}
lst = lambda df: [dict(code=str(r.code), ent_month=str(r.ent_date.date())[:7], ret=round(float(r.ret), 4), per=(None if not r.per_ok else r.per), pbr=r.pbr, yld=r.yld, p_pbr=round(r.p_pbr, 3),
                       per3y=(None if not np.isfinite(r.per3y) else round(r.per3y, 3))) for r in df.itertuples()]
res = dict(A0=A0, train_desc=dict(window=[str(TA.date()), str(TB.date())], n=int(len(x)), n5=n5, cuts=CUT, groups=D, spearman_ret=corr,
                                  bigwin=lst(x[x['bigwin']].sort_values('ret', ascending=False)), biglose=lst(x[x['biglose']].sort_values('ret'))),
           meta=dict(runtime_s=round(time.time() - t0, 1), maxrss_mb=rss()))
(HERE / 'step0.json').write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str))
for k, v in D.items():
    print(k, {g: (o['n'], round(o['win'], 2), round(o['mean'], 3), round(o['median'], 3), o['bigwin'], o['biglose'], o['top10pnl']) for g, o in v.items()})
print(corr); print(res['train_desc']['bigwin']); print(res['train_desc']['biglose']); print(res['meta'])
