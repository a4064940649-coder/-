#!/usr/bin/env python3
"""資料面稽核 S75：基礎表。只讀原始資料，不 import 研究中心任何程式。
官方日 K（raw/jb7k… git blob，GIT_NO_LAZY_FETCH=1）→ scratchpad/audit75/off.parquet（四碼普通股＋0050，長表）
hschi1106 stock_dayK → hs.parquet。python3 -I -B build_base.py"""
import io, os, subprocess, glob
from pathlib import Path
import pandas as pd, numpy as np
RS = Path('/home/user/-/research/tw_stock_selection')
RAW = RS / 'raw/jb7k6295r7-bot_tw-stock-data'
TMP = Path('/tmp/claude-0/-home-user/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/audit75')
env = dict(os.environ, GIT_NO_LAZY_FETCH='1')
ls = subprocess.run(['git', '-C', str(RAW), 'ls-tree', 'HEAD', 'data/universe/daily/'], capture_output=True, text=True, check=True, env=env).stdout
items = [(l.split()[2], l.split('\t')[1]) for l in ls.strip().split('\n') if l.endswith('.csv')]
p = subprocess.Popen(['git', '-C', str(RAW), 'cat-file', '--batch'], stdin=subprocess.PIPE, stdout=subprocess.PIPE, env=env)
fr = []; missing = []
for sha, path in items:
    p.stdin.write((sha + '\n').encode()); p.stdin.flush()
    hdr = p.stdout.readline().decode().split()
    if len(hdr) < 3 or hdr[1] != 'blob':
        missing.append(path); continue
    n = int(hdr[2]); data = p.stdout.read(n); p.stdout.read(1)
    d = pd.read_csv(io.BytesIO(data), dtype=str)
    d = d[(d['stock_id'].str.fullmatch(r'[1-9]\d{3}') & d['market'].isin(['twse', 'tpex'])) | (d['stock_id'] == '0050')]
    d['file_date'] = path.split('/')[-1][:10]
    fr.append(d[['file_date', 'date', 'stock_id', 'name', 'market', 'open', 'high', 'low', 'close', 'volume', 'amount', 'change']])
p.stdin.close(); p.wait()
d = pd.concat(fr, ignore_index=True)
for c in ['open', 'high', 'low', 'close', 'volume', 'amount', 'change']:
    d[c] = pd.to_numeric(d[c], errors='coerce')
print('files', len(items), 'missing blobs', len(missing), 'rows', len(d), 'date!=file_date', int((d['date'] != d['file_date']).sum()))
d['date'] = pd.to_datetime(d['date'])
d = d.drop(columns='file_date').rename(columns={'stock_id': 'code'}).drop_duplicates(['date', 'code'], keep='last')
d.to_parquet(TMP / 'off.parquet', index=False)
print(d['date'].min(), d['date'].max(), d['code'].nunique())
fr = []
for f in sorted(glob.glob('/home/user/hschi1106/stock_ga/data/stock_dayK/*.csv')):
    x = pd.read_csv(f, dtype={'stock_id': str}, usecols=['date', 'stock_id', 'open', 'max', 'min', 'close', 'spread', 'Trading_Volume', 'Trading_money'])
    fr.append(x)
h = pd.concat(fr, ignore_index=True).rename(columns={'stock_id': 'code'}); h['date'] = pd.to_datetime(h['date'])
h.to_parquet(TMP / 'hs.parquet', index=False)
print('hs', len(h), h['date'].min(), h['date'].max(), h['code'].nunique())
