#!/usr/bin/env python3
"""S94 步驟 1（續）：盤點＋比對＋產生 rev_A/B/C.csv（暫存區）。用法：python3 -I -B s94_inventory.py"""
import json, re
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent
MASTER = '/home/user/niklavsd/bottleneck-signals/data/processed/tw_revenue.csv'
SC = Path('/tmp/claude-0/-home-user/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/s94')
OUT = HERE / 'out'
R = {}
raw = pd.read_parquet(SC / 'raw_long.parquet')
n0 = len(raw)
dupfull = int(raw.duplicated(['stock_id', 'period', 'market', 'rev', 'cum']).sum())
raw = raw.drop_duplicates(['stock_id', 'period', 'market', 'rev', 'cum'])
R['raw'] = dict(rows=n0, exact_dup_rows_dropped=dupfull, dup_years=[2006, 2011], rows_after=len(raw), period=[raw.period.min(), raw.period.max()],
                dup_code_period_market_after=int(raw.duplicated(['stock_id', 'period', 'market']).sum()))
x = raw[raw.duplicated(['stock_id', 'period'], keep=False)].sort_values(['stock_id', 'period'])
R['raw']['cross_market_dup_pairs'] = int(x.groupby(['stock_id', 'period']).ngroups)
R['raw']['cross_market_dup_same_value'] = int((x.groupby(['stock_id', 'period']).rev.nunique() == 1).sum())
R['raw']['cross_market_dup_examples'] = x.head(6)[['stock_id', 'name', 'period', 'market', 'rev']].to_dict('records')
raw['pri'] = raw['market'].map({'twse': 0, 'tpex': 1, 'esb': 2})
raw1 = raw.sort_values(['period', 'stock_id', 'pri']).drop_duplicates(['period', 'stock_id'], keep='first')
raw1 = raw1[raw1.stock_id.str.fullmatch(r'[1-9]\d{3}')]

ms = pd.read_csv(MASTER, dtype={'code': str})
ms['period'] = ms['month'].str[:7]
R['master'] = dict(rows=len(ms), period=[ms.period.min(), ms.period.max()], markets=ms.market.value_counts().to_dict(), n_codes=int(ms.code.nunique()),
                   non4digit_rows=int((~ms.code.str.fullmatch(r'[1-9]\d{3}')).sum()))
LO, HI = ms.period.min(), ms.period.max()

# ---- 單位、累計、IFRS
t = raw1.sort_values(['stock_id', 'period']).copy()
t['y'] = t.period.str[:4]; t['mm'] = t.period.str[5:]
jan = t[t.mm == '01']
R['cum_check'] = dict(jan_cum_eq_rev=float((jan.cum == jan.rev).mean()), n_jan=len(jan))
t['prev_cum'] = t.groupby(['stock_id', 'y']).cum.shift(1); t['prev_mm'] = t.groupby(['stock_id', 'y']).mm.shift(1)
q = t[(t.mm != '01') & (t.prev_mm.astype(float) == t.mm.astype(float) - 1)]
dd = (q.cum - q.prev_cum - q.rev).abs()
R['cum_check'].update(n_pairs=len(q), cum_diff_eq_rev_exact=float((dd == 0).mean()), within_1k=float((dd <= 1).mean()), within_1pct=float((dd <= 0.01 * q.rev.abs().clip(lower=1)).mean()),
                      by_era={k: float((dd[m] <= 1).mean()) for k, m in (('2003-2012', q.y <= '2012'), ('2013-2014', (q.y >= '2013') & (q.y <= '2014')), ('2015+', q.y >= '2015'))})
# IFRS 斷點：2013-01 對 2012-12 的比（上市）
a = raw1[(raw1.period == '2012-12') & (raw1.market == 'twse')].set_index('stock_id').rev; b = raw1[(raw1.period == '2013-01') & (raw1.market == 'twse')].set_index('stock_id').rev
a2 = raw1[(raw1.period == '2011-12') & (raw1.market == 'twse')].set_index('stock_id').rev; b2 = raw1[(raw1.period == '2012-01') & (raw1.market == 'twse')].set_index('stock_id').rev
rt = (b / a).replace([np.inf, -np.inf], np.nan).dropna(); rt2 = (b2 / a2).replace([np.inf, -np.inf], np.nan).dropna()
# 2013-01 檔的「去年當月營收」（＝2012-01 以 IFRS 合併口徑重編）對 2012-01 檔的當月營收
ly = raw1[(raw1.period == '2013-01') & (raw1.market == 'twse')].set_index('stock_id').rev_ly
r3 = (ly / b2).replace([np.inf, -np.inf], np.nan).dropna()
R['ifrs'] = dict(ratio_2013_01_over_2012_12=dict(median=float(rt.median()), p75=float(rt.quantile(.75)), frac_gt_1_5=float((rt > 1.5).mean()), n=len(rt)),
                 ratio_2012_01_over_2011_12=dict(median=float(rt2.median()), p75=float(rt2.quantile(.75)), frac_gt_1_5=float((rt2 > 1.5).mean()), n=len(rt2)),
                 rev_ly_in_2013_01_file_over_rev_in_2012_01_file=dict(median=float(r3.median()), frac_equal=float((r3 == 1).mean()), frac_gt_1_05=float((r3 > 1.05).mean()), n=len(r3)),
                 example_1101={p: float(raw1[(raw1.stock_id == '1101') & (raw1.period == p)].rev.iloc[0]) for p in ('2012-11', '2012-12', '2013-01', '2013-02')},
                 master_1101={p: float(ms[(ms.code == '1101') & (ms.period == p)].revenue_k_twd.iloc[0]) for p in ('2012-11', '2012-12', '2013-01', '2013-02')})

# ---- 逐筆比對（兩者都有）
m = ms.merge(raw1[['stock_id', 'period', 'market', 'rev', 'name']], left_on=['code', 'period'], right_on=['stock_id', 'period'], how='inner', suffixes=('', '_raw'))
df = (m.revenue_k_twd - m.rev); rel = df.abs() / m.revenue_k_twd.abs().clip(lower=1)
m['same'] = df == 0; m['rel'] = rel; m['y'] = m.period.str[:4].astype(int)
R['compare'] = dict(n_both=len(m), exact=float(m.same.mean()), n_diff=int((~m.same).sum()), within_0_1pct=float((rel <= 0.001).mean()), n_diff_gt_1pct=int((rel > 0.01).sum()),
                    n_diff_gt_10pct=int((rel > 0.1).sum()),
                    by_year={int(y): dict(n=len(g), n_diff=int((~g.same).sum()), n_gt1pct=int((g.rel > 0.01).sum())) for y, g in m.groupby('y')})
bad = m[~m.same].copy(); bad['ratio'] = bad.rev / bad.revenue_k_twd
bad[['code', 'name', 'period', 'market', 'market_raw', 'revenue_k_twd', 'rev', 'ratio']].to_csv(OUT / 'compare_diff_rows.csv', index=False)
R['compare']['diff_ratio_quantiles'] = {str(k): float(v) for k, v in bad.ratio.replace([np.inf, -np.inf], np.nan).quantile([.05, .25, .5, .75, .95]).items()}
R['compare']['diff_master_zero_or_neg'] = int((bad.revenue_k_twd <= 0).sum()); R['compare']['diff_raw_zero_or_neg'] = int((bad.rev <= 0).sum())

# ---- 涵蓋數：各月上市（raw market＝twse；master market＝sii）
cr = raw1[raw1.market == 'twse'].groupby('period').stock_id.nunique(); cm = ms[ms.market == 'sii'].groupby('period').code.nunique()
cnt = pd.DataFrame({'raw_twse': cr, 'master_sii': cm, 'raw_all': raw1.groupby('period').stock_id.nunique(), 'master_all': ms.groupby('period').code.nunique()})
cnt.to_csv(OUT / 'count_by_month.csv')
cnt['y'] = cnt.index.str[:4]
R['count_by_year_dec'] = {y: {k: (None if pd.isna(v) else int(v)) for k, v in g.iloc[-1][['raw_twse', 'master_sii', 'raw_all', 'master_all']].items()} for y, g in cnt.groupby('y')}

# ---- raw 有、主檔沒有
rr = raw1[(raw1.period >= LO) & (raw1.period <= HI)].copy()
have = set(zip(ms.code, ms.period))
rr['in_master'] = [k in have for k in zip(rr.stock_id, rr.period)]
add = rr[~rr.in_master].copy()
mcodes = set(ms.code)
add['code_in_master'] = add.stock_id.isin(mcodes)
names = raw.groupby('stock_id').name.agg(lambda s: '|'.join(pd.unique(s)))
isky = names.str.contains(r'KY|(?:^|\|)F\s*[-*]|-DR', regex=True)        # KY、舊稱 F-xx／F*xx、存託憑證（-DR）
KY = set(isky[isky].index)
z = np.load(SC / 'cache' / 'panel_cs.npz', allow_pickle=False)
IDX = pd.DatetimeIndex(z['IDX']); COLS = z['COLS'].astype(str); MK = z['MK']
pres = MK > 0
last = {c: IDX[np.flatnonzero(pres[:, j])[-1]] for j, c in enumerate(COLS) if pres[:, j].any()}
first = {c: IDX[np.flatnonzero(pres[:, j])[0]] for j, c in enumerate(COLS) if pres[:, j].any()}
ever_tw = {c for j, c in enumerate(COLS) if (MK[:, j] == 1).any()}
END = IDX[-1]
def cls(c):
    if c in KY: return 'KY／外國企業'
    if c not in last: return '不在上市櫃日 K 面板（興櫃或 2015 前已下市櫃）'
    if last[c] < END - pd.Timedelta(days=30): return '已下市（2015 後）'
    return '其他（仍掛牌）'
g = add.groupby('stock_id').agg(n_months=('period', 'size'), first=('period', 'min'), last=('period', 'max'), mkts=('market', lambda s: '/'.join(sorted(set(s)))), name=('name', 'last'), code_in_master=('code_in_master', 'first')).reset_index()
g['class'] = g.stock_id.map(cls); g['ever_twse_in_panel'] = g.stock_id.isin(ever_tw)
g['panel_first'] = g.stock_id.map(lambda c: str(first[c].date()) if c in first else ''); g['panel_last'] = g.stock_id.map(lambda c: str(last[c].date()) if c in last else '')
g['raw_twse'] = g.mkts.str.contains('twse')
g.to_csv(OUT / 'added_codes.csv', index=False)
def summ(x):
    return {k: dict(n_codes=len(v), n_months=int(v.n_months.sum()), first=v['first'].min(), last=v['last'].max()) for k, v in x.groupby('class')}
R['added'] = dict(n_pairs=len(add), n_codes=len(g), all=summ(g), not_in_master_at_all=summ(g[~g.code_in_master]), in_master_but_missing_months=summ(g[g.code_in_master]),
                  ever_twse_in_panel=summ(g[g.ever_twse_in_panel]), raw_twse_market=summ(g[g.raw_twse]),
                  by_year={y: int(n) for y, n in add.groupby(add.period.str[:4]).size().items()},
                  by_year_twse={y: int(n) for y, n in add[add.market == 'twse'].groupby(add.period.str[:4]).size().items()})
add['class'] = add.stock_id.map(cls)
R['added']['by_year_class_twse'] = {y: {k: int(v) for k, v in gg['class'].value_counts().items()} for y, gg in add[add.market == 'twse'].groupby(add.period.str[:4])}
R['ky_codes'] = sorted(KY); R['n_ky'] = len(KY)
# 主檔有、raw 沒有
rh = set(zip(raw1.stock_id, raw1.period))
only_m = ms[[k not in rh for k in zip(ms.code, ms.period)]]
R['master_only'] = dict(n_pairs=len(only_m), n_codes=int(only_m.code.nunique()), by_year={y: int(n) for y, n in only_m.groupby(only_m.period.str[:4]).size().items()},
                        non4digit=int((~only_m.code.str.fullmatch(r'[1-9]\d{3}')).sum()))

# ---- 產生三份營收檔
A = ms[['month', 'market', 'code', 'name', 'revenue_k_twd']]
ad = pd.DataFrame(dict(month=add.period + '-01', market=add.market.map({'twse': 'sii', 'tpex': 'otc', 'esb': 'otc'}), code=add.stock_id, name=add.name, revenue_k_twd=add.rev.astype(float)))
B = pd.concat([A, ad], ignore_index=True)
C = pd.concat([A, ad[~ad.code.isin(KY)]], ignore_index=True)
assert not B.duplicated(['month', 'code']).any()
A.to_csv(SC / 'rev_A.csv', index=False); B.to_csv(SC / 'rev_B.csv', index=False); C.to_csv(SC / 'rev_C.csv', index=False)
R['files'] = dict(A=len(A), B=len(B), C=len(C), added_B=len(ad), added_C=int((~ad.code.isin(KY)).sum()))
json.dump(R, open(OUT / 'inventory.json', 'w'), ensure_ascii=False, indent=1, default=str)
print(json.dumps({k: v for k, v in R.items() if k not in ('ky_codes',)}, ensure_ascii=False, indent=1, default=str)[:9000])
