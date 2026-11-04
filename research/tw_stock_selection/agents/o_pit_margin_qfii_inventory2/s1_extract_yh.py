#!/usr/bin/env python3
"""從 Yuan-Hsi/SALL_Agency 的 TEJ 逐檔 CSV（stock_0406）抽出 2005 年起的融資融券欄位，存成精簡 parquet。
用法：python3 -I -B s1_extract_yh.py <checkout_dir> <out_parquet>"""
import sys, glob, os, pandas as pd
src, out = sys.argv[1], sys.argv[2]
cols = {'年月日': 'date', '融資餘額(張)': 'mbal', '融資買進(張)': 'mbuy', '融資賣出(張)': 'msell', '融券買進(張)': 'sbuy', '融券餘額(張)': 'sbal', '融券賣出(張)': 'ssell', '收盤價(元)': 'close'}
parts = []
for f in sorted(glob.glob(os.path.join(src, 'RL/File_Repository/stock_0406/*.csv'))):
    code = os.path.basename(f)[:-4]
    if not (len(code) == 4 and code.isdigit() and not code.startswith('00')): continue
    d = pd.read_csv(f, usecols=lambda c: c in cols, dtype=str, encoding='utf-8-sig').rename(columns=cols)
    d = d[d.date >= '2005/01/01']
    d['code'] = code
    parts.append(d)
x = pd.concat(parts, ignore_index=True)
x['date'] = pd.to_datetime(x.date, format='%Y/%m/%d')
for c in ['mbal', 'mbuy', 'msell', 'sbuy', 'sbal', 'ssell', 'close']: x[c] = pd.to_numeric(x[c], errors='coerce')
old = pd.read_parquet(out) if os.path.exists(out) else None
if old is not None: x = pd.concat([old, x]).drop_duplicates(['code', 'date'])
x.to_parquet(out, index=False)
print(len(parts), x.shape, x.code.nunique(), x.date.min(), x.date.max())
