#!/usr/bin/env python3
"""複製員：自建事件表（FSC 申報＋櫃買 CB 日行情首見日）。輸出 my_events.csv、my_cb_listings.csv、my_events_summary.json。"""
import json, re
from pathlib import Path
import numpy as np, pandas as pd
H = Path(__file__).resolve().parent
RAW = H.parent / 'raw'
P = Path('/tmp/claude-0/-home-user--/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/panel')
PCOLS = set(pd.read_parquet(P / 'close_raw.parquet').columns)
D = pd.read_csv(H / 'fsc_parsed.csv', dtype={'code': str}, parse_dates=['recv', 'eff'])
D['item_n'] = D['item'].astype(str).str.replace(r'\s', '', regex=True)
note = (D['note1'].astype(str) + D['note2'].astype(str))
ok = D['result'].isin(['生效', '核減'])
it = D['item_n']
is_seo = it.str.startswith('現金') & ~it.str.contains('特') & ~note.str.fullmatch(r'.*nan特別股|特別股nan|.*^特別股$')
is_seo &= ~note.str.contains('特別股') | note.str.contains('普通股')
is_cb = it.str.contains('轉換公司債') & ~it.str.contains('海外') & ~it.str.contains('私募') & ~it.str.contains('交換')
is_ecb = it.str.contains('海外') & (it.str.contains('轉換') | (it == '海外公司債'))
is_gdr = it.str.contains('存託憑證') & ~it.str.contains('老股') & ~it.str.contains('台灣')
is_sd = (it.str.contains('盈餘') | it.str.contains('資本公積')) & ~it.str.contains('減資')
D['etype'] = np.select([is_seo, is_cb, is_ecb, is_gdr, is_sd], ['SEO', 'CB', 'ECB', 'GDR', 'SD'], '')
E = D[ok & (D['etype'] != '') & (D['ctype'] == '上市')].copy()
E = E.drop_duplicates(['code', 'etype', 'recv', 'amount'])
summ = {'listed_ctype': E['etype'].value_counts().to_dict(),
        'listed_ctype_inpanel': E[E['code'].isin(PCOLS)]['etype'].value_counts().to_dict()}

# ---------- 櫃買 CB 掛牌（首見日） ----------
q = pd.read_parquet(RAW / 'david_twcb' / 'data' / 'combined' / 'cb_quotes_all.parquet', columns=['資料日期', '代號', '名稱'])
q['代號'] = q['代號'].astype(str); q['名稱'] = q['名稱'].astype(str)
g = q.groupby('代號').agg(list_date=('資料日期', 'min'), last_date=('資料日期', 'max'), name=('名稱', 'first')).reset_index()
g = g[~(g['代號'].str.len() == 6) | ~g['name'].str.contains('E')]           # 交換債剔除
g['issuer'] = g['代號'].str[:4]
first_day = q['資料日期'].min()
g['left_censored'] = g['list_date'] <= first_day + pd.Timedelta(days=10)
g['issuer_twse'] = g['issuer'].isin(set(D.loc[D.ctype == '上市', 'code'])) | g['issuer'].isin(PCOLS)
g.to_csv(H / 'my_cb_listings.csv', index=False)

# ---------- FSC 涵蓋期 ----------
GAPS = [('2009-08-25', '2009-12-31'), ('2010-07-30', '2010-12-31'), ('2013-11-22', '2013-12-31'), ('2016-01-01', '2016-12-31'), ('2025-07-26', '2026-12-31')]
GAPS = [(pd.Timestamp(a), pd.Timestamp(b)) for a, b in GAPS]
def in_gap(d):
    return any(a <= d <= b for a, b in GAPS)

# FSC CB ↔ 櫃買掛牌配對：以掛牌為單位，同發行人、收文日落在 [掛牌日−LAG, 掛牌日]，優先配未配過的申報（最早者）
LAG = pd.Timedelta(days=270)
cb = E[E.etype == 'CB'].sort_values('recv').copy()
cb['list_date'] = pd.NaT; cb['cb_code'] = None
cball = D[ok & D['item_n'].str.contains('轉換公司債') & ~D['item_n'].str.contains('海外') & ~D['item_n'].str.contains('交換')]
L = g[~g['left_censored']].sort_values('list_date')
nmatch = {}
lmatch = {}
for _, x in L.iterrows():
    c = cb[(cb.code == x.issuer) & (cb.recv <= x.list_date) & (cb.recv >= x.list_date - LAG)]
    if len(c):
        c2 = c[cb.loc[c.index, 'cb_code'].isna()]
        i = (c2 if len(c2) else c).index[0]
        if pd.isna(cb.at[i, 'list_date']): cb.at[i, 'list_date'] = x.list_date; cb.at[i, 'cb_code'] = x['代號']
        lmatch[x['代號']] = 'listed'
    elif ((cball.code == x.issuer) & (cball.recv <= x.list_date) & (cball.recv >= x.list_date - LAG)).any():
        lmatch[x['代號']] = 'other_ctype'
used = set(lmatch)
E = E.drop(index=cb.index); E = pd.concat([E, cb])
E['src'] = 'FSC'
un = L[~L['代號'].isin(used) & L['issuer'].isin(PCOLS)].copy()
GAPMODE = __import__('os').environ.get('GAPMODE', 'in')
if GAPMODE == 'in':
    un['gap'] = un['list_date'].map(in_gap)
else:
    un['gap'] = un['list_date'].map(lambda d: in_gap(d) or in_gap(d - pd.Timedelta(days=int(GAPMODE))))
tp = un[un['gap']]
T = pd.DataFrame({'code': tp['issuer'].values, 'etype': 'CB', 'recv': pd.NaT, 'eff': pd.NaT, 'list_date': tp['list_date'].values,
                  'cb_code': tp['代號'].values, 'src': 'TPEX', 'name': tp['name'].values})
E = pd.concat([E, T], ignore_index=True)
E['anchor'] = E['recv'].fillna(E['list_date'])
E = E[['code', 'etype', 'recv', 'eff', 'list_date', 'cb_code', 'amount', 'price', 'item', 'name', 'src', 'anchor']].sort_values(['anchor', 'code']).reset_index(drop=True)
E.to_csv(H / 'my_events.csv', index=False)

# ---------- 交叉覆蓋率 ----------
lo, hi = first_day + pd.Timedelta(days=10), pd.Timestamp('2025-07-25')
cbf = cb[(cb.recv >= lo) & (cb.recv <= hi - LAG)]
cov1 = (float(cbf['list_date'].notna().mean()), len(cbf))
tl = L[L['issuer'].isin(set(cb.code)) & (L.list_date <= hi)]
tl = tl[~tl['list_date'].map(lambda d: in_gap(d) or in_gap(d - LAG))]
cov2 = (float(tl['代號'].isin(used).mean()), len(tl), float(tl['代號'].map(lmatch).eq('listed').mean()))
summ.update({'final_counts': E['etype'].value_counts().to_dict(), 'n_total': int(len(E)), 'cb_tpex_only': int(len(T)),
             'cov_fsc_cb_found_in_tpex': cov1, 'cov_tpex_found_in_fsc': cov2})
(H / 'my_events_summary.json').write_text(json.dumps(summ, ensure_ascii=False, indent=1, default=str))
print(json.dumps(summ, ensure_ascii=False, indent=1, default=str))
