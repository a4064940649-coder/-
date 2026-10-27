#!/usr/bin/env python3
"""獨立複製：從 TWSE 處置公告封存自己建名單（不讀研究員程式）。"""
import json, hashlib, sys
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
RAW = HERE.parent / 'raw' / 'tsungmin2001_tw-disposition' / 'raw'
P = Path('/tmp/claude-0/-home-user--/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/panel')
X = Path('/home/user/stevenhsiehsheng-ctrl/tw-stock-screener/data/extras')

def roc(s):
    y, m, d = s.strip().split('/')
    return pd.Timestamp(int(y) + 1911, int(m), int(d))

def load_raw():
    rows = []; shas = {}
    for y in range(2003, 2027):
        f = RAW / f'twse_{y}.json'
        b = f.read_bytes(); shas[f.name] = hashlib.sha256(b).hexdigest()
        d = json.loads(b)
        assert len(d['data']) == d['total'], (y, len(d['data']), d['total'])
        for r in d['data']:
            a, b2 = r[6].split('～')
            rows.append(dict(ann=roc(r[1]), code=r[2].strip(), name=r[3], cum=r[4], cond=r[5].strip(), meas=r[7].strip(),
                             content=r[8] or '', start=roc(a), end=roc(b2), fy=y))
    return pd.DataFrame(rows), shas

def nth_of(r):
    if r['meas'] in ('第二次處置', '收足全部款券'): return 2
    if '曾發布處置' in r['content']: return 2
    if '督導會報' in r['cond'] or '督導會報' in r['meas']: return 2
    return 1

def build():
    df, shas = load_raw()
    nraw = len(df)
    df = df.drop_duplicates(['ann', 'code', 'start', 'end', 'meas']).reset_index(drop=True)
    ndd = len(df)
    df['nth'] = df.apply(nth_of, axis=1)
    cols = pd.read_parquet(P / 'close_raw.parquet').columns
    IDX = pd.read_parquet(P / 'close_raw.parquet').index
    A = df[df['code'].isin(set(cols))].copy()
    A['srcl'] = 'twse'
    # disposal_5y 併入
    B = pd.read_csv(X / 'disposal_5y.csv.gz', dtype={'code': str}, parse_dates=['date', 'start', 'end'])
    Bp = B[B['code'].isin(set(cols))].copy()
    keyA = set(zip(A['code'], A['start'], A['end']))
    add = Bp[[ (c, s, e) not in keyA for c, s, e in zip(Bp['code'], Bp['start'], Bp['end'])]].copy()
    add = add.rename(columns={'date': 'ann'})
    add['srcl'] = 'd5y'; add['cond'] = ''; add['meas'] = ''; add['content'] = ''
    pos = {d: i for i, d in enumerate(IDX)}
    def tpos(d):
        return IDX.searchsorted(d)
    allr = pd.concat([A, add], ignore_index=True).sort_values(['code', 'start', 'ann']).reset_index(drop=True)
    # 推定 nth：前次處置 end 在 30 個交易日內 → 2
    for i in allr.index[allr['srcl'] == 'd5y']:
        c, s = allr.at[i, 'code'], allr.at[i, 'start']
        prev = allr[(allr['code'] == c) & (allr['start'] < s)]
        n = 1
        if len(prev):
            pe = prev['end'].max()
            if tpos(s) - tpos(pe) <= 30: n = 2
        allr.at[i, 'nth'] = n
    allr['nth'] = allr['nth'].astype(int)
    return df, A, add, allr, B, Bp, shas, nraw, ndd

if __name__ == '__main__':
    df, A, add, allr, B, Bp, shas, nraw, ndd = build()
    fz = json.loads((HERE.parent / 'data_freeze.json').read_text())
    print('raw', nraw, 'dedup', ndd, 'freeze', fz['source_A']['rows_raw'], fz['source_A']['rows_dedup'])
    print('sha match', all(shas[k] == v for k, v in fz['source_A']['file_sha256'].items()))
    print('in panel A', len(A), 'freeze n_in_panel', fz['n_in_panel'], 'added from d5y', len(add), 'all', len(allr))
    for col in ('ann', 'start'):
        m = (A[col] >= '2006-04-01') & (A[col] <= '2021-08-31')
        print('2006-04..2021-08 by', col, int(m.sum()))
    allr.to_csv(HERE / 'my_list.csv', index=False, columns=['code', 'ann', 'start', 'end', 'nth', 'srcl', 'cond', 'meas'])
    # 與 disposal_5y 比較（2021-09-24 ~ 2026-10-05）
    lo, hi = pd.Timestamp('2021-09-24'), pd.Timestamp('2026-10-05')
    a = A[(A['start'] >= lo) & (A['start'] <= hi)]
    b = Bp[(Bp['start'] >= lo) & (Bp['start'] <= hi)]
    ka = set(zip(a['code'], a['start'])); kb = set(zip(b['code'], b['start']))
    print('A', len(a), 'B', len(b), 'A in B', len(ka & kb) / len(ka), 'B in A', len(ka & kb) / len(kb))
    m = a.merge(b, on=['code', 'start'], suffixes=('_a', '_b'))
    print('matched', len(m), 'end same', (m['end_a'] == m['end_b']).mean(), 'ann same', (m['ann'] == m['date']).mean())
    # 逐年門檻
    yr = A.groupby(A['start'].dt.year).size()
    base = yr.loc[2022:2025].mean(); print('2022-25 mean', base)
    print('fail years', [int(y) for y, v in yr.items() if v < 0.5 * base and y <= 2021])
    print(yr.to_dict())
    print('nth dist', A['nth'].value_counts().to_dict())
