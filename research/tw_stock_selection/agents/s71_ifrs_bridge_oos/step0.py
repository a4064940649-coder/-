#!/usr/bin/env python3
"""S71 第 0 步（純描述，只讀月營收檔；不載入價格、不算任何策略報酬）：量化 2013 IFRS 口徑斷點。
股票＝營收檔 market＝sii（上市）的代號。python3 -I -B step0.py → step0.json"""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
import s71lib as S
from s71lib import stlib

raw = pd.read_csv(stlib.REV_CSV, dtype={'code': str}, parse_dates=['month'])
sii = sorted(raw.loc[raw['market'] == 'sii', 'code'].unique())
names = raw.drop_duplicates('code', keep='last').set_index('code')['name']
Rv = S.rev_table(None)[sii]
P_ = lambda s: pd.Period(s, 'M')
out = dict(source=dict(file=stlib.REV_CSV, sha256=S.cslib.sha_file(stlib.REV_CSV), first=str(Rv.index[0]), last=str(Rv.index[-1]), n_sii_codes=len(sii),
                       fetch='niklavsd pipeline/fetch_tw.py：mopsov nas/t21/{market}/t21sc03_{民國年}_{月}[_0].html 的「當月營收」欄（去年同月欄未保存）'))

# ---- (1) 年增率截面分布，逐月與逐年
Y = (Rv / Rv.shift(12) - 1)
LY = np.log(Rv / Rv.shift(12))
rows = []
for m in pd.period_range('2010-01', '2015-12', freq='M'):
    y = Y.loc[m].dropna(); ly = LY.loc[m].dropna()
    q = y.quantile([.1, .25, .5, .75, .9])
    rows.append(dict(month=str(m), n=int(len(y)), median_pct=float(q[.5] * 100), p25_pct=float(q[.25] * 100), p75_pct=float(q[.75] * 100), p10_pct=float(q[.1] * 100),
                     p90_pct=float(q[.9] * 100), iqr_pp=float((q[.75] - q[.25]) * 100), sd_log=float(ly.std()),
                     mad_sd_log=float(1.4826 * (ly - ly.median()).abs().median()),
                     share_gt50=float((y > .5).mean()), share_gt100=float((y > 1).mean()), share_lt_m30=float((y < -.3).mean())))
M = pd.DataFrame(rows); M['year'] = M['month'].str[:4]
out['yoy_monthly'] = rows
out['yoy_by_year'] = {y: {k: float(g[k].mean()) for k in ('n', 'median_pct', 'p25_pct', 'p75_pct', 'p10_pct', 'p90_pct', 'iqr_pp', 'sd_log', 'mad_sd_log', 'share_gt50', 'share_gt100', 'share_lt_m30')}
                      for y, g in M.groupby('year')}
out['yoy_by_year_note'] = '各欄＝該年 12 個月截面統計的平均'
# 整年持續偏高（口徑位移的特徵）：一年 12 個月中至少 10 個月年增率 > +30%
pers = {}
for y in range(2010, 2016):
    yy = Y.loc[P_(f'{y}-01'):P_(f'{y}-12')]
    ok = yy.notna().sum() == 12
    pers[str(y)] = dict(n=int(ok.sum()), n_ge10_months_gt30=int(((yy > .3).sum() >= 10)[ok].sum()), share=float(((yy > .3).sum() >= 10)[ok].mean()))
out['persistent_high_yoy'] = pers

# ---- (2) 12 月→1 月水準跳動：2013 對其他年
jr = {}
for y in range(2006, 2016):
    j = np.log(Rv.loc[P_(f'{y}-01')] / Rv.loc[P_(f'{y - 1}-12')]).dropna()
    q = j.quantile([.1, .25, .5, .75, .9])
    jr[str(y)] = dict(n=int(len(j)), median=float(q[.5]), p25=float(q[.25]), p75=float(q[.75]), p90=float(q[.9]), share_ratio_gt1p5=float((j > np.log(1.5)).mean()),
                      share_ratio_gt2=float((j > np.log(2)).mean()), share_ratio_lt0p67=float((j < -np.log(1.5)).mean()))
out['dec_to_jan_log_jump'] = jr

# ---- (3) 異常跳動 a（B2 的估計量）：2013 對安慰劑年 2011、2012
ab = {}
A = {}
for y, pri in ((2011, range(2006, 2011)), (2012, range(2007, 2012)), (2013, S.PRIOR_YEARS)):
    a = S.abnormal_jump(Rv, y, list(pri)).dropna(); A[y] = a
    q = a.quantile([.05, .1, .25, .5, .75, .9, .95])
    ab[str(y)] = dict(prior_years=[int(v) for v in pri], n=int(len(a)), **{f'p{int(k * 100):02d}': float(v) for k, v in q.items()},
                      share_gt_ln1p2=float((a > np.log(1.2)).mean()), share_gt_ln1p5=float((a > np.log(1.5)).mean()), share_gt_ln2=float((a > np.log(2)).mean()),
                      share_lt_m_ln1p2=float((a < -np.log(1.2)).mean()))
plc = pd.concat([A[2011], A[2012]])
ab['placebo_pooled_2011_2012'] = dict(n=int(len(plc)), p90=float(plc.quantile(.9)), p95=float(plc.quantile(.95)), p10=float(plc.quantile(.1)))
out['abnormal_jump'] = ab

# ---- (4) 判定用摘要（PREREG 的主規則選擇只看這裡）
by = out['yoy_by_year']
ref_med = (by['2012']['median_pct'] + by['2014']['median_pct']) / 2
ref_g50 = (by['2012']['share_gt50'] + by['2014']['share_gt50']) / 2
out['decision_inputs'] = dict(median_2013_pct=by['2013']['median_pct'], median_ref_2012_2014_pct=ref_med, median_gap_pp=by['2013']['median_pct'] - ref_med,
                              share_gt50_2013=by['2013']['share_gt50'], share_gt50_ref=ref_g50, share_gt50_ratio=by['2013']['share_gt50'] / ref_g50,
                              placebo_p90_abnormal_jump=ab['placebo_pooled_2011_2012']['p90'])

# ---- (5) 誰受影響：2013 的 a 最大者與大型公司舉例；用 deadband＝安慰劑 P90 的 k 檢查「校正後 2013 年增率」是否回到正常（2 月起的月份未參與估計）
db = ab['placebo_pooled_2011_2012']['p90']
a = S.abnormal_jump(Rv, 2013, S.PRIOR_YEARS)                # 這一段是最初的單月、P90 不校正帶版本（描述用；定案的 B2 見 step0b／step0c 與 s71lib.scope_ratio）
k = pd.Series(np.where(a > db, np.exp(a), 1.0), index=a.index).where(a.notna())
r12 = Rv.loc[P_('2012-01'):P_('2012-12')].sum(); r13 = Rv.loc[P_('2013-01'):P_('2013-12')].sum()
tab = pd.DataFrame({'name': names.reindex(sii), 'a': a, 'k': k, 'rev_2012_12': Rv.loc[P_('2012-12')], 'rev_2013_01': Rv.loc[P_('2013-01')], 'sum2012': r12, 'sum2013': r13,
                    'annual_ratio': r13 / r12})
tt = tab.dropna(subset=['a'])
out['scope_ratio'] = dict(deadband_used=db, n_with_a=int(len(tt)), n_no_estimate=int(tab['a'].isna().sum() - (Rv.loc[P_('2013-01')].isna() & tab['a'].isna()).sum()),
                          n_k_gt1=int((tt['k'] > 1).sum()), share_k_gt1=float((tt['k'] > 1).mean()),
                          k_quantiles_among_gt1={f'p{int(q * 100)}': float(tt.loc[tt['k'] > 1, 'k'].quantile(q)) for q in (.25, .5, .75, .9)},
                          share_k_gt2=float((tt['k'] > 2).mean()),
                          corr_log_k_vs_log_annual_ratio=float(np.corrcoef(np.log(tt.loc[tt['k'] > 1, 'k']), np.log(tt.loc[tt['k'] > 1, 'annual_ratio']).fillna(0))[0, 1]))
top = tt.sort_values('a', ascending=False).head(15)
out['top15_by_abnormal_jump'] = [dict(code=c, name=str(r['name']), k=float(r['k']), rev_2012_12_k=float(r['rev_2012_12']), rev_2013_01_k=float(r['rev_2013_01']),
                                      annual_2013_over_2012=float(r['annual_ratio'])) for c, r in top.iterrows()]
ex = ['1101', '1216', '1301', '1402', '2002', '2317', '2330', '2382', '2412', '2454', '2882', '2891', '2912', '3045', '9904']
out['examples_large'] = [dict(code=c, name=str(tab.loc[c, 'name']), a=float(tab.loc[c, 'a']), k=float(tab.loc[c, 'k']), rev_2012_12_k=float(tab.loc[c, 'rev_2012_12']),
                              rev_2013_01_k=float(tab.loc[c, 'rev_2013_01']), annual_2013_over_2012=float(tab.loc[c, 'annual_ratio'])) for c in ex if c in tab.index]
# 校正後年增率（k 缺值的公司不算）
A2 = Rv.to_numpy(float).copy(); pre = np.asarray(Rv.index < S.BREAK)
A2[pre] = A2[pre] * k.reindex(Rv.columns).to_numpy()[None, :]
Ra = pd.DataFrame(A2, index=Rv.index, columns=Rv.columns)
Ya = Ra / Ra.shift(12) - 1
rows = []
for m in pd.period_range('2013-01', '2013-12', freq='M'):
    y = Ya.loc[m].dropna(); y0 = Y.loc[m].reindex(y.index)
    rows.append(dict(month=str(m), n=int(len(y)), median_raw_pct=float(y0.median() * 100), median_adj_pct=float(y.median() * 100), share_gt50_raw=float((y0 > .5).mean()),
                     share_gt50_adj=float((y > .5).mean()), share_lt_m30_adj=float((y < -.3).mean()), iqr_adj_pp=float((y.quantile(.75) - y.quantile(.25)) * 100)))
out['yoy_2013_after_B2'] = rows
C = pd.DataFrame(rows)
out['yoy_2013_after_B2_mean_feb_dec'] = {c: float(C.loc[1:, c].mean()) for c in C.columns if c != 'month'}
(S.HERE / 'step0.json').write_text(json.dumps(out, ensure_ascii=False, indent=1))
print(json.dumps({k: out[k] for k in ('yoy_by_year', 'persistent_high_yoy', 'dec_to_jan_log_jump', 'abnormal_jump', 'decision_inputs', 'scope_ratio', 'top15_by_abnormal_jump',
                                      'examples_large', 'yoy_2013_after_B2_mean_feb_dec')}, ensure_ascii=False, indent=None, default=str).replace('}, ', '},\n'))
