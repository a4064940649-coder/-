#!/usr/bin/env python3
"""S16 第 0 步（寫 PREREG 之前；不算任何報酬）：上櫃資料檢查。
(1) 官方日 K 原始檔的市場別、上櫃起始日、每年上櫃檔數、下市上櫃股、同日同代碼重複、上櫃名稱裡的「創」；
(2) 轉上市（上櫃 → 上市）與反向轉換的檔數與空窗；
(3) tw_revenue.csv 的市場別、上櫃涵蓋年份、逐月缺洞、下市股缺營收的比例（分市場）、轉上市股營收是否連續；上櫃 2018-10 缺洞用 MOPS 官方彙總表補（寫 rev_patch_otc_201810.csv）並逐筆驗證；
(4) 四個股池在換股日的平均檔數、SURGE 有值比例；
(5) 0050 在 2025-06-11～17 分割停牌日的開盤價處理。
python3 -I -B s0_datacheck.py → data_check.json"""
import sys, io, json, time, resource, csv
from pathlib import Path
from collections import defaultdict, Counter
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
import s16lib
from s16lib import s4lib, s2lib, stlib, cslib

H = Path(__file__).resolve().parent
t0 = time.time()
rss = lambda: round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
out = {}

# ---------------------------------------------------------------- (1) 原始檔掃描（只讀 date、stock_id、name、market）
lab_year = defaultdict(lambda: defaultdict(set))   # 年 → 市場別 → 代號集合（全部代號）
lab4_year = defaultdict(lambda: defaultdict(set))  # 只四碼、非 0 開頭
dup_days = 0; dup_rows = 0
tpex_innov_names = {}; tpex_first = None; ndays = 0
for path, fh in cslib._git_blobs('data/universe/daily'):
    try:
        d = pd.read_csv(fh, dtype={'stock_id': str}, usecols=['date', 'stock_id', 'name', 'market'])
    except Exception:
        continue
    if not len(d):
        continue
    ndays += 1
    y = int(str(d['date'].iloc[0])[:4])
    for mk, g in d.groupby('market'):
        lab_year[y][mk].update(g['stock_id'].tolist())
    d4 = d[d['stock_id'].str.fullmatch(r'[1-9]\d{3}')]
    for mk, g in d4.groupby('market'):
        lab4_year[y][mk].update(g['stock_id'].tolist())
    dd = d4[d4['market'].isin(['twse', 'tpex'])]
    nd = int(dd['stock_id'].duplicated().sum())
    if nd:
        dup_days += 1; dup_rows += nd
    tp = dd[dd['market'] == 'tpex']
    if len(tp) and tpex_first is None:
        tpex_first = str(d['date'].iloc[0])
    nm = tp[tp['name'].astype(str).str.contains('創')]
    for r in nm.itertuples():
        tpex_innov_names.setdefault(r.stock_id, r.name)
out['raw_daily'] = dict(
    n_files=ndays, tpex_first_date=tpex_first,
    market_labels_by_year={y: {mk: len(v) for mk, v in sorted(m.items())} for y, m in sorted(lab_year.items())},
    market_labels_4digit_by_year={y: {mk: len(v) for mk, v in sorted(m.items())} for y, m in sorted(lab4_year.items())},
    dup_same_day_same_code=dict(days=dup_days, rows=dup_rows),
    tpex_names_with_創=tpex_innov_names)
del lab_year, lab4_year
print('原始檔掃描', f'{time.time()-t0:.0f}s', rss(), 'MB', flush=True)

# ---------------------------------------------------------------- (2) 面板：上櫃檔數、下市、轉上市
P = s16lib.load()
assert (P.T, P.N, str(P.IDX[-1].date())) == (2876, 2142, '2026-10-22'), 'P0 失敗：面板和 S6～S15 不同'
MK = P.MK
yr = pd.DatetimeIndex(P.IDX).year
per_year = {}
for y in sorted(set(yr)):
    rows = yr == y
    m = MK[rows]
    per_year[int(y)] = dict(twse_codes=int((m == 1).any(axis=0).sum()), tpex_codes=int((m == 2).any(axis=0).sum()),
                            twse_avg_per_day=float((m == 1).sum(axis=1).mean()), tpex_avg_per_day=float((m == 2).sum(axis=1).mean()))
out['panel_codes_by_year'] = per_year
listed = MK > 0
last_t = np.where(listed.any(axis=0), P.T - 1 - np.argmax(listed[::-1], axis=0), -1)
first_t = np.where(listed.any(axis=0), np.argmax(listed, axis=0), -1)
last_mk = MK[last_t, np.arange(P.N)]
gone = last_t < P.T - 6                                         # 最後 5 個交易日前就消失＝下市（或改代號）
dl = defaultdict(lambda: Counter())
for j in np.flatnonzero(gone):
    dl[int(yr[last_t[j]])]['twse' if last_mk[j] == 1 else 'tpex'] += 1
out['panel_delisted_by_year（最後一根日 K 的年份與市場別）'] = {y: dict(c) for y, c in sorted(dl.items())}
out['panel_delisted_total'] = dict(twse=int((gone & (last_mk == 1)).sum()), tpex=int((gone & (last_mk == 2)).sum()))
# 轉換：MK 從 2 變 1（轉上市）或 1 變 2
tr = []
for j in range(P.N):
    m = MK[:, j]
    nz = np.flatnonzero(m > 0)
    if len(nz) < 2:
        continue
    v = m[nz]
    ch = np.flatnonzero(v[1:] != v[:-1])
    for c in ch:
        a, b = nz[c], nz[c + 1]
        tr.append(dict(code=str(P.COLS[j]), frm=int(v[c]), to=int(v[c + 1]), last_old=str(P.IDX[a].date()), first_new=str(P.IDX[b].date()), gap_days=int(b - a - 1)))
trd = pd.DataFrame(tr)
out['market_switch'] = dict(
    n_tpex_to_twse=int(((trd['frm'] == 2) & (trd['to'] == 1)).sum()) if len(trd) else 0,
    n_twse_to_tpex=int(((trd['frm'] == 1) & (trd['to'] == 2)).sum()) if len(trd) else 0,
    gap_trading_days_dist=trd['gap_days'].describe().to_dict() if len(trd) else {},
    gap_gt_5=trd[trd['gap_days'] > 5].to_dict('records') if len(trd) else [],
    by_year={f'{y}_{"上櫃→上市" if f == 2 else "上市→上櫃"}': int(n) for (y, f), n in trd.assign(y=trd['first_new'].str[:4]).groupby(['y', 'frm']).size().items()} if len(trd) else {},
    examples=trd.head(8).to_dict('records'))
print('面板', f'{time.time()-t0:.0f}s', rss(), 'MB', flush=True)

# ---------------------------------------------------------------- (3) 月營收
r = pd.read_csv(stlib.REV_CSV, dtype={'code': str}, parse_dates=['month'], usecols=['month', 'market', 'code', 'revenue_k_twd'])
r['y'] = r['month'].dt.year
out['revenue_markets'] = r['market'].value_counts().to_dict()
out['revenue_codes_by_year_market'] = {int(y): g.groupby('market')['code'].nunique().to_dict() for y, g in r.groupby('y')}
out['revenue_first_month_by_market'] = r.groupby('market')['month'].min().dt.strftime('%Y-%m').to_dict()
out['revenue_last_month_by_market'] = r.groupby('market')['month'].max().dt.strftime('%Y-%m').to_dict()
both = r.groupby('code')['market'].nunique()
both_codes = set(both[both > 1].index)
out['revenue_codes_in_two_markets'] = len(both_codes)
# 同月兩個市場都有（轉換月）
dm = r.groupby(['code', 'month'])['market'].nunique()
out['revenue_same_month_two_markets_rows'] = int((dm > 1).sum())
rev_codes = r.groupby('code')['month'].agg(['min', 'max'])
# 股池（共同門檻，未分市場）曾入池的代號，分市場：有無營收、下市股
INNOV, innov_names = s4lib.innov_panel(P)
UN = s16lib.universes(P, INNOV)
t_from = P.t0 - 1
cov = {}
for name, mkv in (('twse', 1), ('tpex', 2)):
    ever = (UN['ALL'][t_from:] & (MK[t_from:] == mkv)).any(axis=0)
    js = np.flatnonzero(ever)
    codes = P.COLS[js]
    has_rev = np.array([c in rev_codes.index for c in codes])
    dlj = gone[js]
    # 下市股：營收最後月份 vs 最後一根日 K
    lag = []
    for j, c, hv in zip(js, codes, has_rev):
        if gone[j] and hv:
            lm = rev_codes.loc[c, 'max']
            lag.append((P.IDX[last_t[j]].to_period('M') - lm.to_period('M')).n)
    cov[name] = dict(ever_in_pool_since_2015_07=int(len(js)), no_revenue_at_all=int((~has_rev).sum()), delisted=int(dlj.sum()),
                     delisted_no_revenue=int((dlj & ~has_rev).sum()), delisted_with_revenue=int((dlj & has_rev).sum()),
                     delisted_rev_last_month_gap_months=pd.Series(lag).describe().to_dict() if lag else {},
                     delisted_rev_within_3m_of_last_bar=int(sum(1 for g in lag if g <= 3)),
                     no_revenue_examples=[str(c) for c in codes[~has_rev]][:30])
out['revenue_coverage_of_pool_codes'] = cov
# 轉上市股營收是否連續（月份不中斷）
cont = []
for c in sorted(both_codes):
    ms = pd.PeriodIndex(r.loc[r['code'] == c, 'month'], freq='M').unique().sort_values()
    if len(ms) > 1:
        gaps = int(((ms[1:] - ms[:-1]).map(lambda x: x.n) > 1).sum())
        cont.append(gaps)
out['revenue_two_market_codes_with_month_gap'] = int(sum(1 for g in cont if g > 0))
out['revenue_two_market_codes_checked'] = len(cont)
# 每月上櫃／上市營收家數，找出缺月
cnt = r.groupby(['month', 'market'])['code'].nunique().unstack()
c15 = cnt.loc['2012-01-01':]
med = c15.rolling(5, center=True, min_periods=3).median()
holes = (c15 < 0.8 * med).stack()
holes = holes[holes]
out['revenue_month_holes（家數 < 前後 5 個月中位數 80%）'] = {f'{d.strftime("%Y-%m")}_{mk}': dict(n=int(c15.loc[d, mk]), median_nearby=float(med.loc[d, mk])) for (d, mk) in holes.index}
# 補洞：MOPS 上櫃 2018-10 彙總表（官方原始頁）
mo = s16lib.parse_mops_t21(s16lib.MOPS_OTC_201810, s16lib.MOPS_OTC_201810_SHA)
o = r[r['market'] == 'otc']
val = {}
for mon, col in (('2018-10-01', 'rev'), ('2018-09-01', 'prev_month'), ('2017-10-01', 'last_year')):
    x = o[o['month'] == mon].set_index('code')['revenue_k_twd']
    jj = mo.set_index('code')[col].reindex(x.index).dropna()
    xx = x.reindex(jj.index)
    val[f'{mon[:7]}_{col}'] = dict(csv_rows=int(len(x)), matched=int(len(jj)), exact=int((np.abs(xx - jj) < 0.5).sum()))
have = set(o.loc[o['month'] == '2018-10-01', 'code'])
pt = mo[~mo['code'].isin(have)].copy()
pt['month'] = '2018-10-01'; pt['market'] = 'otc'; pt['revenue_k_twd'] = pt['rev']
pt[['month', 'market', 'code', 'name', 'revenue_k_twd']].sort_values('code').to_csv(s16lib.PATCH_CSV, index=False)
out['patch_otc_201810'] = dict(source=str(s16lib.MOPS_OTC_201810), source_url='https://mopsov.twse.com.tw/nas/t21/otc/t21sc03_107_10_0.html', sha256=s16lib.MOPS_OTC_201810_SHA,
                               mops_rows=int(len(mo)), csv_rows_2018_10=int(len(have)), patch_rows=int(len(pt)), validation=val,
                               note='2018-10 原有 126 筆與 MOPS 逐筆相同；MOPS 的「上月營收」與 csv 2018-09 的 739 筆逐筆相同；「去年當月」與 csv 2017-10 有 16 筆不同（MOPS 顯示的是之後更正過的去年值，不影響 2018-10 當月數字）。')
del r
print('營收', f'{time.time()-t0:.0f}s', rss(), 'MB', flush=True)

# ---------------------------------------------------------------- (4) 股池在換股日的平均檔數與 SURGE 有值比例
RT0 = s2lib.rev_tables(P)
RT = s16lib.rev_tables_s16(P, patch=True)
A5 = s16lib.rev_tables_s16(P, patch=False)
out['A5_unpatched_equals_s2lib'] = bool(A5['S'].equals(RT0['S']))
del A5
S0 = RT0['S']
S, av = RT['S'], RT['av']
diff_cells = (np.isfinite(S.to_numpy()) != np.isfinite(S0.to_numpy()))
out['patch_effect_surge_cells_changed_by_year'] = {int(y): int(v) for y, v in pd.Series(diff_cells.sum(axis=1), index=S.index.year).groupby(level=0).sum().items() if v}
rows = []
for m in S.index:
    k = av[m]
    if k >= P.T or k < 61 or P.IDX[k] < pd.Timestamp('2016-01-01'):
        continue
    s = np.isfinite(S.loc[m].to_numpy())
    s0 = np.isfinite(S0.loc[m].to_numpy())
    row = dict(month=str(m), date=P.IDX[k])
    for c in ('OTC', 'ALL'):
        row[f'surge_unpatched_{c}'] = int((UN[c][k] & s0).sum())
    for c, U in UN.items():
        row[f'n_{c}'] = int(U[k].sum()); row[f'surge_{c}'] = int((U[k] & s).sum())
        row[f'ma_{c}'] = int((U[k] & s & P.F_MA60[k]).sum())
    rows.append(row)
df = pd.DataFrame(rows)
df['y'] = df['date'].dt.year
agg = {}
for y, g in df.groupby('y'):
    agg[int(y)] = {c: dict(pool=round(float(g[f'n_{c}'].mean()), 1), surge_ok=round(float(g[f'surge_{c}'].mean()), 1),
                           surge_share=round(float((g[f'surge_{c}'] / g[f'n_{c}'].clip(lower=1)).mean()), 3), above_ma60=round(float(g[f'ma_{c}'].mean()), 1)) for c in UN}
out['pool_by_year_on_signal_days'] = agg
out['pool_avg_since2016'] = {c: dict(pool=round(float(df[f'n_{c}'].mean()), 1), surge_ok=round(float(df[f'surge_{c}'].mean()), 1)) for c in UN}
out['months_with_lt50_surge'] = {c: int((df[f'surge_{c}'] < 50).sum()) for c in UN}
out['months_with_lt50_surge_unpatched'] = {c: int((df[f'surge_unpatched_{c}'] < 50).sum()) for c in ('OTC', 'ALL')}
out['unpatched_surge_share_by_year'] = {int(y): {c: round(float((g[f'surge_unpatched_{c}'] / g[f'n_{c}'].clip(lower=1)).mean()), 3) for c in ('OTC', 'ALL')} for y, g in df.groupby('y')}
out['otc_surge_months_lt50_list'] = df.loc[df['surge_OTC'] < 50, 'month'].tolist()
# 補洞對現行（只上市）訊號的影響：只比訊號矩陣，不算報酬
c0 = s16lib.surge_signal(P, UN['CUR'], RT0); c1 = s16lib.surge_signal(P, UN['CUR'], RT)
out['patch_effect_on_CUR_signal'] = dict(buy_cells_diff=int((c0['sig'] != c1['sig']).sum()), exit_cells_diff=int((c0['X'] != c1['X']).sum()),
                                         months_diff=[x['sig_date'] for x, y in zip(c0['info'], c1['info']) if x['targets'] != y['targets'] or x['n_ma'] != y['n_ma']])
del c0, c1
out['innov_names_in_panel'] = innov_names
out['innov_in_tpex_rows'] = int((INNOV & (MK == 2)).sum())
# 成交值單位：成交值 ÷（成交股數 × 收盤）的中位數，分市場分年（應 ≈ 1，表示兩市場都是元與股）
with np.errstate(all='ignore'):
    ratio = P.AMT / (P.VOL * P.C)
out['amount_unit_check'] = {('twse' if mk == 1 else 'tpex'): {int(y): round(float(np.nanmedian(ratio[yr == y][((P.MK == mk) & np.isfinite(ratio) & (P.VOL > 0))[yr == y]])), 3) for y in sorted(set(yr))} for mk in (1, 2)}
del ratio
# ---------------------------------------------------------------- (5) 0050 分割停牌
win = (P.IDX >= '2025-06-09') & (P.IDX <= '2025-06-19')
out['e50_split_window'] = [dict(date=str(d.date()), E50_adj_close=float(a), E50O_used=float(b), E50O_raw=(None if not np.isfinite(c) else float(c)))
                           for d, a, b, c in zip(P.IDX[win], P.E50[win], P.E50O[win], P.E50O_raw[win])]
out['meta'] = dict(runtime_s=round(time.time() - t0, 1), maxrss_mb=rss())
(H / 'data_check.json').write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str))
print(json.dumps({k: out[k] for k in ('patch_effect_on_CUR_signal', 'patch_otc_201810', 'A5_unpatched_equals_s2lib', 'revenue_month_holes（家數 < 前後 5 個月中位數 80%）', 'months_with_lt50_surge_unpatched', 'otc_surge_months_lt50_list', 'patch_effect_surge_cells_changed_by_year', 'panel_delisted_total', 'revenue_markets', 'revenue_first_month_by_market', 'revenue_coverage_of_pool_codes', 'pool_avg_since2016', 'months_with_lt50_surge', 'meta')}, ensure_ascii=False, indent=1, default=str))
