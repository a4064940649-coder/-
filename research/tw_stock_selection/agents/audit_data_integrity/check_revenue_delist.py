#!/usr/bin/env python3
"""B5 月營收涵蓋、B6 下市股。python3 -I -B check_revenue_delist.py → parts/revenue_delist.json、tables/*.csv。只讀。"""
import sys; from pathlib import Path as _P; sys.path.insert(0, str(_P(__file__).resolve().parent))
import io, json, time, resource
import numpy as np, pandas as pd
from common import *                                    # noqa: E402

t0 = time.time(); R = {}
z = np.load(TMP / 'off.npz', allow_pickle=False)
IDX = pd.DatetimeIndex(z['IDX']); COLS = z['COLS'].astype(str); MK = z['MK']; C = z['C']
SEAM = pd.Timestamp('2015-01-05')
pre = pd.read_csv(PRE, dtype={'code': str}, usecols=['date', 'code', 'close']); pre['date'] = pd.to_datetime(pre['date'])
pre = pre[pre['date'] < SEAM]

# 每月月底在列的上市／上櫃代號
listed_tw, listed_ot = {}, {}
for m, g in pre.groupby(pre['date'].dt.to_period('M')):
    listed_tw[m] = set(g.loc[g['date'] == g['date'].max(), 'code'])
per = IDX.to_period('M')
for m in per.unique():
    t = np.flatnonzero(per == m)[-1]
    listed_tw[m] = set(COLS[MK[t] == 1]); listed_ot[m] = set(COLS[MK[t] == 2])

# ---------------------------------------------------------------- tw_revenue.csv
r = pd.read_csv(REV_CSV, dtype={'code': str}, parse_dates=['month'])
r['m'] = r['month'].dt.to_period('M')
b5 = dict(file=str(REV_CSV), sha256=sha_file(REV_CSV), n_rows=len(r), first=str(r['m'].min()), last=str(r['m'].max()), n_codes=int(r['code'].nunique()),
          by_market={k: int(v) for k, v in r['market'].value_counts().items()},
          n_nan=int(r['revenue_k_twd'].isna().sum()), n_zero=int((r['revenue_k_twd'] == 0).sum()), n_negative=int((r['revenue_k_twd'] < 0).sum()),
          n_month_not_first_day=int((r['month'].dt.day != 1).sum()))
r4 = r[r['code'].str.fullmatch(r'[1-9]\d{3}')]
neg = r4[r4['revenue_k_twd'] <= 0]
b5['zero_or_negative_4digit'] = dict(n=len(neg), n_zero=int((neg['revenue_k_twd'] == 0).sum()), n_negative=int((neg['revenue_k_twd'] < 0).sum()), n_codes=int(neg['code'].nunique()),
    by_year={int(k): int(v) for k, v in neg.groupby(neg['month'].dt.year).size().items()},
    negative_rows=[dict(month=str(x.m), market=x.market, code=x.code, name=x.name, revenue_k_twd=x.revenue_k_twd) for x in neg[neg['revenue_k_twd'] < 0].head(30).itertuples()],
    engine_treatment='引擎把營收 ≤ 0 當缺值（Rv.where(Rv > 0)）')
dup_same = r[r.duplicated(['m', 'market', 'code'], keep=False)]
dup_x = r[r.duplicated(['m', 'code'], keep=False) & ~r.duplicated(['m', 'market', 'code'], keep=False)]
dxg = dup_x.groupby(['m', 'code'])['revenue_k_twd'].agg(['min', 'max'])
b5['duplicates'] = dict(same_month_market_code_rows=len(dup_same), cross_market_month_code_pairs=int(len(dxg)), cross_market_value_differs=int((np.abs(dxg['max'] - dxg['min']) > 0.5).sum()),
    cross_market_by_year={int(k): int(v) for k, v in dup_x.drop_duplicates(['m', 'code']).groupby(dup_x.drop_duplicates(['m', 'code'])['month'].dt.year).size().items()},
    cross_market_sample=[dict(month=str(k[0]), code=k[1], min=v['min'], max=v['max']) for k, v in dxg[np.abs(dxg['max'] - dxg['min']) > 0.5].head(10).iterrows()],
    engine_treatment='同月同代號留一筆，上市（sii）優先')
r['pri'] = (r['market'] != 'sii').astype(int)
ru = r.sort_values(['m', 'code', 'pri']).drop_duplicates(['m', 'code'], keep='first')
RV = ru.pivot(index='m', columns='code', values='revenue_k_twd')
RV = RV.reindex(pd.period_range(RV.index.min(), RV.index.max(), freq='M'))
POS = RV > 0
rows = []
for m in RV.index:
    tw = listed_tw.get(m); ot = listed_ot.get(m)
    have = set(POS.columns[POS.loc[m].to_numpy()])
    rows.append(dict(month=str(m), n_rev_sii=int(((r['m'] == m) & (r['market'] == 'sii')).sum()), n_rev_otc=int(((r['m'] == m) & (r['market'] == 'otc')).sum()),
        n_listed_twse=len(tw) if tw else None, n_twse_with_rev=len(tw & have) if tw else None, cov_twse_pct=round(len(tw & have) / len(tw) * 100, 2) if tw else None,
        n_listed_twse_ex91=len({c for c in tw if not c.startswith('91')}) if tw else None,
        cov_twse_ex91_pct=round(len({c for c in tw if not c.startswith('91')} & have) / max(len({c for c in tw if not c.startswith('91')}), 1) * 100, 2) if tw else None,
        n_listed_tpex=len(ot) if ot else None, cov_tpex_pct=round(len(ot & have) / len(ot) * 100, 2) if ot else None))
cov = pd.DataFrame(rows); cov.to_csv(HERE / 'tables/revenue_coverage_monthly.csv', index=False)
cv = cov.dropna(subset=['cov_twse_pct']).copy(); cv['y'] = cv['month'].str[:4].astype(int)
b5['coverage_twse_by_year'] = {int(y): dict(min=float(g['cov_twse_pct'].min()), median=float(g['cov_twse_pct'].median()), max=float(g['cov_twse_pct'].max()),
                                            median_ex91=float(g['cov_twse_ex91_pct'].median()), n_listed_median=float(g['n_listed_twse'].median())) for y, g in cv.groupby('y')}
co = cov.dropna(subset=['cov_tpex_pct']).copy(); co['y'] = co['month'].str[:4].astype(int)
b5['coverage_tpex_by_year'] = {int(y): dict(min=float(g['cov_tpex_pct'].min()), median=float(g['cov_tpex_pct'].median()), max=float(g['cov_tpex_pct'].max())) for y, g in co.groupby('y')}
def holes(df, col, k=1.5):
    s = df.set_index('month')[col].astype(float); med = s.rolling(7, center=True, min_periods=3).median()
    bad = s[s < med - k]
    return [dict(month=m, cov_pct=float(v), neighbors_median=float(med[m])) for m, v in bad.items()]
b5['hole_months_twse'] = holes(cv, 'cov_twse_pct'); b5['hole_months_tpex'] = holes(co, 'cov_tpex_pct')
b5['hole_rule'] = '當月涵蓋率 < 前後共 7 個月中位數 − 1.5 個百分點'
n_sii = cov.set_index('month')['n_rev_sii'].astype(float); n_otc = cov.set_index('month')['n_rev_otc'].astype(float)
b5['row_count_drops'] = dict(sii=[dict(month=m, n=int(v), prev=int(n_sii.shift(1)[m])) for m, v in n_sii.items() if n_sii.shift(1).get(m, np.nan) > 0 and v < n_sii.shift(1)[m] * 0.97],
                             otc=[dict(month=m, n=int(v), prev=int(n_otc.shift(1)[m])) for m, v in n_otc.items() if n_otc.shift(1).get(m, np.nan) > 0 and v < n_otc.shift(1)[m] * 0.97])
# 上市期間內部缺月（第一筆～最後一筆之間缺值或 ≤0）
PA = POS.to_numpy(); anyv = PA.any(0)
fi = PA.argmax(0); la = PA.shape[0] - 1 - PA[::-1].argmax(0)
first = {c: (RV.index[fi[k]] if anyv[k] else None) for k, c in enumerate(POS.columns)}
last = {c: (RV.index[la[k]] if anyv[k] else None) for k, c in enumerate(POS.columns)}
inner = {}
for k, c in enumerate(POS.columns):
    if not anyv[k]:
        continue
    seg = PA[fi[k]:la[k] + 1, k]
    for off in np.flatnonzero(~seg):
        m = RV.index[fi[k] + off]
        inner.setdefault(m.year, []).append((str(m), c))
tw_ever = set().union(*listed_tw.values())
b5['inner_missing_code_months'] = dict(total=sum(len(v) for v in inner.values()), by_year={int(y): len(v) for y, v in sorted(inner.items())},
    twse_listed_at_that_month_by_year={int(y): sum(1 for m, c in v if c in listed_tw.get(pd.Period(m, 'M'), ())) for y, v in sorted(inner.items())})
tw_inner = [(m, c) for v in inner.values() for m, c in v if c in listed_tw.get(pd.Period(m, 'M'), ())]
pd.DataFrame(tw_inner, columns=['month', 'code']).to_csv(HERE / 'tables/revenue_inner_missing_twse.csv', index=False)

# 最新月份：營收檔沒有的月份靠官方倉庫 revenue_hist
H = pd.read_csv(TMP / 'revhist.csv', dtype=str); H['rev'] = pd.to_numeric(H['rev'], errors='coerce')
H['m'] = pd.PeriodIndex(H['period'], freq='M')
Ht = H[(H['market'] == 'twse')]
last_m = RV.index.max()
newm = sorted(set(Ht['m']) - set(RV.index))
meta = json.loads((TMP / 'off_meta.json').read_text())
b5['revenue_hist'] = dict(meta=meta['revenue_hist'], months_twse=[str(Ht['m'].min()), str(Ht['m'].max())], n_rows_twse=len(Ht), months_not_in_tw_revenue=[str(x) for x in newm])
lat = {}
for m in newm:
    tw = listed_tw.get(m) or listed_tw[max(listed_tw)]
    have = set(Ht.loc[(Ht['m'] == m) & (Ht['rev'] > 0), 'stock_id'])
    lat[str(m)] = dict(n_listed_twse=len(tw), n_with_rev=len(tw & have), cov_pct=round(len(tw & have) / len(tw) * 100, 2), missing=sorted(tw - have)[:40], source=sorted(set(Ht.loc[Ht['m'] == m, 'from'])))
b5['latest_months_from_revenue_hist'] = lat
# 兩個來源交叉（上市，重疊月份）
A = ru[ru['code'].isin(tw_ever)].set_index(['m', 'code'])['revenue_k_twd']
Bx = Ht.drop_duplicates(['m', 'stock_id'], keep='last').set_index(['m', 'stock_id'])['rev']; Bx.index.names = ['m', 'code']
j = pd.concat([A.rename('a'), Bx.rename('b')], axis=1, join='inner').dropna()
j['rel'] = (j['a'] / j['b'] - 1).abs()
j['y'] = [k[0].year for k in j.index]
b5['cross_tw_revenue_vs_revenue_hist'] = dict(n_pairs=len(j), exact_pct=round(float((np.abs(j['a'] - j['b']) < 0.6).mean() * 100), 3), within_1pct_pct=round(float((j['rel'] < 0.01).mean() * 100), 3),
    diff_gt_10pct=int((j['rel'] > 0.10).sum()),
    by_year={int(y): dict(n=len(g), exact_pct=round(float((np.abs(g['a'] - g['b']) < 0.6).mean() * 100), 2), diff_gt_1pct=int((g['rel'] >= 0.01).sum())) for y, g in j.groupby('y')},
    only_in_tw_revenue=int(len(A.index.difference(Bx.index)[[k[0] >= Ht['m'].min() for k in A.index.difference(Bx.index)]])) if len(A) else None,
    worst=[dict(month=str(k[0]), code=k[1], tw_revenue=v['a'], revenue_hist=v['b']) for k, v in j.sort_values('rel', ascending=False).head(15).iterrows()])
j[j['rel'] >= 0.01].reset_index().assign(m=lambda q: q['m'].astype(str)).to_csv(HERE / 'tables/revenue_two_sources_diff.csv', index=False)
# 2026-10 上市彙總表：封存版（11-02 前）與每日更新版的差異＝首次值／更正值會不會混
try:
    def rd(repo, f):
        s = [x for x in ls_tree(repo, 'data/mops/revenue_hist') if x[1].endswith(f)][0]
        for p, b in read_blobs(repo, [s], local_objects(repo)):
            return None if b is None else pd.read_csv(io.BytesIO(b), dtype=str)
    va, vb = rd(RAW_REPO, '2026-10_twse.csv'), rd(TSD, '2026-10_twse.csv')
    if va is not None and vb is not None:
        ja = va.set_index('stock_id')['當月營收'].astype(float); jb = vb.set_index('stock_id')['當月營收'].astype(float)
        cm = ja.index.intersection(jb.index)
        b5['revenue_hist_2026_10_versions'] = dict(archive_rows=len(va), latest_rows=len(vb), common=len(cm), value_changed=int((np.abs(ja[cm] - jb[cm]) > 0.5).sum()), only_latest=len(jb.index.difference(ja.index)))
    else:
        b5['revenue_hist_2026_10_versions'] = '無法驗證（其中一版本機沒有 blob）'
except Exception as e:
    b5['revenue_hist_2026_10_versions'] = repr(e)
# 既有研究結論（只引用）
b5['first_print_vs_revised'] = dict(source='agents/s39_first_print_revenue/FINDINGS.md（獨立複製成功）',
    quote='回測用更正後月營收、實盤只看得到首次公布值；2024-07～2026-10 用首次公布值重算：標準路徑年化 61.95% → 61.17%（−0.78 個百分點），20 條路徑差中位數 −0.93、最多 −5.21；上市股月 26,074 中 187 筆（0.72%）被更正；2024-06 以前沒有首次公布資料，無法量化',
    mixing='make_list.py：歷史月份讀 tw_revenue.csv（已含更正），當月用官方倉庫 MOPS 彙總表（首次值）→ 同一份名單內歷史用更正值、當月用首次值（S39 已說明）')
R['B5_revenue'] = b5

# ---------------------------------------------------------------- B6 下市股
dl = pd.read_csv(DELIST, dtype={'code': str}); dl['date'] = pd.to_datetime(dl['date'])
# 面板內每個代號的上市起訖（pre ∪ 官方 MK==1）與上櫃起訖
pre_last = pre.groupby('code')['date'].max(); pre_first = pre.groupby('code')['date'].min()
pre_last_px = pre[pre['close'] > 0].groupby('code')['date'].max()
info = {}
has = np.isfinite(C) & (C > 0)
for jx, c in enumerate(COLS):
    tw = np.flatnonzero(MK[:, jx] == 1); ot = np.flatnonzero(MK[:, jx] == 2); px = np.flatnonzero(has[:, jx] & (MK[:, jx] == 1))
    if len(tw):
        info[c] = dict(tw_first=IDX[tw[0]], tw_last=IDX[tw[-1]], px_last=IDX[px[-1]] if len(px) else pd.NaT, ot_after=bool(len(ot) and ot[-1] > tw[-1]), ot_last=IDX[ot[-1]] if len(ot) else pd.NaT)
for c in pre_last.index:
    if c in info:
        info[c]['tw_first'] = min(info[c]['tw_first'], pre_first[c])
    else:
        info[c] = dict(tw_first=pre_first[c], tw_last=pre_last[c], px_last=pre_last_px.get(c, pd.NaT), ot_after=False, ot_last=pd.NaT)
P = pd.DataFrame.from_dict(info, orient='index'); P.index.name = 'code'
cal = pd.DatetimeIndex(sorted(set(pre['date']))).append(IDX)
cutoff = cal[-31]
P['gone'] = P['tw_last'] < cutoff
P['to_tpex'] = P['gone'] & P['ot_after']
gone = P[P['gone'] & ~P['to_tpex']].copy()
lst = dl[(dl['date'] >= '2007-01-01') & dl['code'].str.fullmatch(r'[1-9]\d{3}')].drop_duplicates('code', keep='last').set_index('code')
both = gone.index.intersection(lst.index)
win = gone[gone['tw_last'] <= dl['date'].max()]
b6 = dict(official_list=dict(file=str(DELIST.relative_to(RS)), sha256=sha_file(DELIST), source='o_infra_v3_rebuild/raw/songdoyouho/delisted_stocks.csv（證交所終止上市公司名單的第三方封存；檔內沒有記載抓取網址）',
                             n_rows=len(dl), first=str(dl['date'].min().date()), last=str(dl['date'].max().date()), n_4digit_since_2007=len(lst)),
          panel=dict(n_twse_codes_ever=len(P), n_gone_from_twse=int(P['gone'].sum()), n_moved_to_tpex=int(P['to_tpex'].sum()), n_gone_not_to_tpex=len(gone),
                     gone_by_year={int(k): int(v) for k, v in gone.groupby(gone['tw_last'].dt.year).size().items()},
                     cutoff=str(cutoff.date())),
          list_vs_panel=dict(n_in_both=len(both), n_list_not_in_panel_as_gone=len(lst.index.difference(gone.index)),
                             list_not_in_panel_at_all=sorted(c for c in lst.index if c not in P.index),
                             list_in_panel_but_not_gone=sorted(c for c in lst.index if c in P.index and c not in gone.index),
                             n_panel_gone_within_list_window_not_in_list=len(win.index.difference(lst.index)),
                             panel_gone_within_list_window_not_in_list=sorted(win.index.difference(lst.index)),
                             n_panel_gone_after_list_end=int((gone['tw_last'] > dl['date'].max()).sum()),
                             panel_gone_after_list_end=[(c, str(v.date())) for c, v in gone.loc[gone['tw_last'] > dl['date'].max(), 'tw_last'].sort_values().items()]))
# 價格涵蓋到下市前幾天
g = gone.loc[both].join(lst[['date', 'name']])
ci = {d: i for i, d in enumerate(cal)}
def tdays(a, b):       # a 到 b 之間的交易日數（b 之前最後一個交易日 − a）
    ia = ci[a]; ib = int(np.searchsorted(cal.values, np.datetime64(b), side='left'))
    return ib - 1 - ia
g['gap_td_lastrow_to_delist'] = [tdays(a, b) for a, b in zip(g['tw_last'], g['date'])]
g['gap_td_lastpx_to_delist'] = [tdays(a, b) if pd.notna(a) else np.nan for a, b in zip(g['px_last'], g['date'])]
q = lambda s: dict(n=int(s.notna().sum()), min=float(s.min()), p25=float(s.quantile(.25)), median=float(s.median()), p75=float(s.quantile(.75)), p90=float(s.quantile(.9)), max=float(s.max()))
b6['price_coverage_before_delist'] = dict(def_='官方下市日之前最後一個交易日，距面板最後一列（或最後一個有收盤價日）幾個交易日；0＝涵蓋到最後交易日',
    last_row=q(g['gap_td_lastrow_to_delist']), last_price=q(g['gap_td_lastpx_to_delist']), n_last_row_gap_gt_5=int((g['gap_td_lastrow_to_delist'] > 5).sum()),
    n_last_row_gap_lt_0=int((g['gap_td_lastrow_to_delist'] < 0).sum()),
    worst=[dict(code=c, name=x['name'], delist=str(x['date'].date()), panel_last=str(x['tw_last'].date()), gap_td=int(x['gap_td_lastrow_to_delist'])) for c, x in g.sort_values('gap_td_lastrow_to_delist', ascending=False).head(15).iterrows()])
# 營收涵蓋到最後交易月前幾個月
lastrev = pd.Series({c: last[c] for c in POS.columns if last[c] is not None})
gg = gone.copy(); gg['last_rev'] = lastrev.reindex(gg.index)
gg['last_m'] = gg['tw_last'].dt.to_period('M')
gg['gap_m'] = [(a - b).n if pd.notna(b) else np.nan for a, b in zip(gg['last_m'], gg['last_rev'])]
gg['period'] = np.where(gg['tw_last'] < '2013-01-01', '2007–2012', np.where(gg['tw_last'] < '2015-07-01', '2013–2015H1', np.where(gg['tw_last'] < '2019-01-01', '2015H2–2018', '2019–')))
gg.reset_index().assign(last_rev=lambda q_: q_['last_rev'].astype(str), last_m=lambda q_: q_['last_m'].astype(str)).to_csv(HERE / 'tables/delisted_coverage.csv', index=False)
b6['revenue_coverage_before_last_trade'] = dict(def_='最後交易月 − 營收檔最後一個有值月份（月數；正常下市前 1～2 個月屬合理，因為最後一兩個月的營收可能不再申報）',
    n_gone=len(gg), n_no_revenue_at_all=int(gg['last_rev'].isna().sum()),
    by_period={p: dict(n=len(x), n_no_rev=int(x['last_rev'].isna().sum()), gap_months=q(x['gap_m']) if x['gap_m'].notna().any() else None, n_gap_le_3=int((x['gap_m'] <= 3).sum()), n_gap_ge_12=int((x['gap_m'] >= 12).sum()))
               for p, x in gg.groupby('period')})
b6['survivorship_existing_research'] = [
    dict(source='agents/o_rev_survivorship/FINDINGS.md', quote='補回國內下市公司營收後，排除金融 n10 跨循環年化 28.15% → 27.66%（−0.49 個百分點）；國內缺口覆蓋率只有 39%；極端最壞界限 −4.4 個百分點（舊主策略 v2 口徑，不是現行 E0）'),
    dict(source='agents/o_delist_rev_backfill_allpit/FINDINGS.md（o18）', quote='實際月營收只補回 2013～2018 放回槽的 19.8%；題目最終「待觀察」，存活者偏誤未排除（舊 V7X 口徑）'),
    dict(source='agents/s55_pre2015_oos/data_inventory.json', quote='2008～2012 下市的上市股 34 檔中只有 1 檔營收涵蓋到最後交易前 3 個月內；2013～2015H1 的 11 檔為 0；最後交易月−最後營收月 中位數 13 個月'),
    dict(source='agents/s75_continuous_2008_2026/FINDINGS.md', quote='下市股營收提早斷、2013 起營收檔只有現存公司（存活者偏誤，偏樂觀）')]
R['B6_delisted'] = b6
R['runtime_s'] = round(time.time() - t0, 1); R['maxrss_mb'] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
jdump(R, HERE / 'parts/revenue_delist.json')
for sec in ('B5_revenue', 'B6_delisted'):
    for k, v in R[sec].items():
        print('##', k, json.dumps(v, ensure_ascii=False, default=str)[:1500])
print(R['runtime_s'], R['maxrss_mb'])
