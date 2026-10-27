# -*- coding: utf-8 -*-
"""Build events.csv for o_issuance_avoid (NO return is computed here).

Sources (both GitHub mirrors, treated as untrusted data):
  A. FSC 申報案件彙總表 (fsc_cases_all.csv from build_fsc.py): filing (收文) date, effective date,
     amount, issue price, type. Years covered: 2003-2008, 2009-01~08, 2010-01~07, 2011, 2012,
     2013-01~11, 2014, 2015, 2017-2024, 2025-01~07. 2016 file is empty in the mirror.
  B. TPEx CB daily quote archive (david26984741-cell/tw-cb-data), 2007-01-02~2026-10-27, 2,170 bonds
     incl. matured/delisted. First day a CB code appears = listing date.
Output columns:
  code, etype (CB / ECB / SEO / GDR / SDIV), recv (filing), eff (effective), list_date (CB listing),
  amount (NT$ face), price (SEO issue price), proceeds (NT$), cb_code, src, match
"""
import os, re, json
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
P = '/tmp/claude-0/-home-user--/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/panel'
C = pd.read_parquet(os.path.join(P, 'close_raw.parquet'))
COLS = set(C.columns)
IDX = C.index

fsc = pd.read_csv(os.path.join(HERE, 'fsc_cases_all.csv'), dtype={'code': str}, parse_dates=['recv', 'eff'])
fsc['item_n'] = fsc['item'].astype(str).str.replace(r'\s+', '', regex=True)
fsc = fsc[(fsc['ctype'] == '上市') & (fsc['result'].isin(['生效', '核減']))].copy()


def classify(it):
    if '交換' in it or '附認股權' in it or '私募' in it:
        return None
    if '轉換公司債' in it:
        return 'ECB' if '海外' in it else 'CB'
    if it == '海外公司債':
        return 'ECB'
    if '存託憑證' in it:
        return None if '老股' in it else 'GDR'
    if it.startswith('現金') and '特' not in it:
        return 'SEO'
    if ('盈餘' in it or '資本公積' in it) and '現金' not in it and '減資' not in it:
        return 'SDIV'
    return None


fsc['etype'] = fsc['item_n'].map(classify)
fsc = fsc[fsc['etype'].notna()].copy()
fsc['proceeds'] = np.where(fsc['etype'] == 'SEO', fsc['amount'] / 10.0 * fsc['price'], fsc['amount'])
fsc.loc[fsc['etype'].isin(['ECB', 'GDR']) & (fsc['ccy'].astype(str).str.contains('美元')), 'proceeds'] = np.nan

# ---------------- TPEx CB listings ----------------
cbq = pd.read_parquet(os.path.join(HERE, 'raw', 'david_twcb', 'data', 'combined', 'cb_quotes_all.parquet'),
                      columns=['資料日期', '代號', '名稱', '交易'])
cbq = cbq[cbq['交易'] == '等價']
cbq['d'] = cbq['資料日期']
days = np.sort(cbq['d'].unique())
dpos = {d: i for i, d in enumerate(days)}
cbq['k'] = cbq['d'].map(dpos)
seg = []
for code, g in cbq.groupby('代號'):
    k = np.sort(g['k'].to_numpy())
    br = np.where(np.diff(k) > 40)[0]   # code reuse: >40 trading-day hole starts a new bond
    starts = np.r_[k[0], k[br + 1]]; ends = np.r_[k[br], k[-1]]
    nm = g.sort_values('k')['名稱'].iloc[0]
    for s, e in zip(starts, ends):
        seg.append(dict(cb_code=code, cb_name=str(nm).strip(), list_date=pd.Timestamp(days[s]), last_date=pd.Timestamp(days[e]),
                        n_days=int(e - s + 1)))
tp = pd.DataFrame(seg)
tp['issuer'] = tp['cb_code'].str[:4]
tp = tp[tp['cb_code'].str.len() == 5]
tp = tp[tp['list_date'] > pd.Timestamp('2007-01-02')]          # bonds already trading on day 1 are not new listings
tp['in_panel'] = tp['issuer'].isin(COLS)


def listed_on(code, d):
    if code not in COLS:
        return False
    s = C[code]
    w = s[(s.index >= d - pd.Timedelta(days=10)) & (s.index <= d + pd.Timedelta(days=10))]
    return bool(w.notna().any())


tp['issuer_twse'] = [listed_on(c, d) for c, d in zip(tp['issuer'], tp['list_date'])]
tp.to_csv(os.path.join(HERE, 'tpex_cb_listings.csv'), index=False)

# ---------------- match FSC CB (TWSE issuer) <-> TPEx listing ----------------
fcb = fsc[fsc['etype'] == 'CB'].copy().sort_values('eff')
fcb['list_date'] = pd.NaT; fcb['cb_code'] = None
used = set()
tpt = tp[tp['issuer_twse']].sort_values('list_date')
for i, r in fcb.iterrows():
    if pd.isna(r['eff']):
        continue
    cand = tpt[(tpt['issuer'] == r['code']) & (tpt['list_date'] >= r['eff'] - pd.Timedelta(days=5)) &
               (tpt['list_date'] <= r['eff'] + pd.Timedelta(days=200))]
    cand = cand[~cand.index.isin(used)]
    if len(cand):
        j = cand.index[0]; used.add(j)
        fcb.at[i, 'list_date'] = cand.at[j, 'list_date']; fcb.at[i, 'cb_code'] = cand.at[j, 'cb_code']

# FSC year coverage windows (filing date ranges actually present in the mirror)
cov = {}
allf = pd.read_csv(os.path.join(HERE, 'fsc_cases_all.csv'), parse_dates=['recv'])
for y, g in allf.groupby('roc_year'):
    cov[int(y) + 1911] = (g['recv'].min(), g['recv'].max())


def fsc_covered(d):
    y = d.year
    if y not in cov:
        return False
    a, b = cov[y]
    return (d >= a - pd.Timedelta(days=20)) & (d <= b)

# coverage: (i) FSC->TPEx: share of FSC TWSE CBs (eff in 2007-2025-06) that list on TPEx
m1 = fcb[(fcb['eff'] >= '2007-01-15') & (fcb['eff'] <= '2025-06-30')]
cov_fsc_to_tpex = float(m1['list_date'].notna().mean())
# (ii) TPEx->FSC: share of TPEx TWSE-issuer listings whose implied filing window (list-120d..list-7d) lies inside FSC coverage
tpt2 = tpt.copy()
tpt2['matched'] = tpt2.index.isin(used)
tpt2['covered'] = [fsc_covered(d - pd.Timedelta(days=45)) and fsc_covered(d - pd.Timedelta(days=100)) for d in tpt2['list_date']]
cc = tpt2[tpt2['covered'] & (tpt2['list_date'] <= '2025-07-25')]
cov_tpex_to_fsc = float(cc['matched'].mean())
unm = cc[~cc['matched']]

# ---------------- assemble events ----------------
ev = []
for _, r in fsc.iterrows():
    if r['etype'] == 'CB':
        continue
    ev.append(dict(code=r['code'], etype=r['etype'], recv=r['recv'], eff=r['eff'], list_date=pd.NaT, amount=r['amount'],
                   price=r['price'], proceeds=r['proceeds'], cb_code=None, src='FSC', match='fsc_only', name=r['name']))
for _, r in fcb.iterrows():
    ev.append(dict(code=r['code'], etype='CB', recv=r['recv'], eff=r['eff'], list_date=r['list_date'], amount=r['amount'],
                   price=np.nan, proceeds=r['amount'], cb_code=r['cb_code'], src='FSC+TPEx' if pd.notna(r['list_date']) else 'FSC',
                   match='both' if pd.notna(r['list_date']) else 'fsc_only', name=r['name']))
for j, r in tpt.iterrows():
    if j in used:
        continue
    ev.append(dict(code=r['issuer'], etype='CB', recv=pd.NaT, eff=pd.NaT, list_date=r['list_date'], amount=np.nan, price=np.nan,
                   proceeds=np.nan, cb_code=r['cb_code'], src='TPEx', match='tpex_only', name=r['cb_name']))
E = pd.DataFrame(ev)
E = E[E['code'].isin(COLS)].copy()
E['anchor'] = E['recv'].fillna(E['list_date'])
E = E.sort_values(['anchor', 'code', 'etype']).reset_index(drop=True)
E.insert(0, 'event_id', [f'E{i:05d}' for i in range(len(E))])
E.to_csv(os.path.join(HERE, 'events.csv'), index=False, date_format='%Y-%m-%d')

# year table
E['year'] = E['anchor'].dt.year
tab = E.pivot_table(index='year', columns='etype', values='event_id', aggfunc='count').fillna(0).astype(int)
summary = dict(
    n_events=int(len(E)), by_type=E['etype'].value_counts().to_dict(),
    fsc_year_coverage={str(k): [str(v[0].date()), str(v[1].date())] for k, v in cov.items()},
    tpex_listings_total=int(len(tp)), tpex_listings_twse_issuer=int(len(tpt)),
    coverage_fsc_cb_to_tpex_listing=round(cov_fsc_to_tpex, 4), n_fsc_cb_2007_2025=int(len(m1)),
    coverage_tpex_listing_to_fsc_cb=round(cov_tpex_to_fsc, 4), n_tpex_in_fsc_window=int(len(cc)),
    tpex_unmatched_examples=unm[['cb_code', 'cb_name', 'list_date']].astype(str).head(30).values.tolist(),
    fsc_cb_unmatched_examples=m1[m1['list_date'].isna()][['code', 'name', 'item', 'eff']].astype(str).head(30).values.tolist(),
    lag_eff_to_list_days=dict(zip(['p10', 'p50', 'p90'], [float(x) for x in np.nanpercentile((fcb['list_date'] - fcb['eff']).dt.days.dropna(), [10, 50, 90])])),
    lag_recv_to_eff_days_SEO=dict(zip(['p10', 'p50', 'p90'], [float(x) for x in np.nanpercentile((E[E.etype == 'SEO']['eff'] - E[E.etype == 'SEO']['recv']).dt.days.dropna(), [10, 50, 90])])),
    year_table=tab.to_dict(orient='index'))
json.dump(summary, open(os.path.join(HERE, 'events_build_summary.json'), 'w'), ensure_ascii=False, indent=1, default=str)
print(json.dumps({k: v for k, v in summary.items() if k != 'year_table'}, ensure_ascii=False, indent=1, default=str)[:4000])
print(tab.to_string())
