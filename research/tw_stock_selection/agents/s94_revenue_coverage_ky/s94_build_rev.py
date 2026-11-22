#!/usr/bin/env python3
"""S94 步驟 1：把 raw/tsd_extra/revenue_raw（官方月營收彙總表原檔，first/＝本快照首見版）攤平成長表，
與主檔 tw_revenue.csv 比對，並產生三份營收檔（放暫存區，不進倉庫）：
  rev_A.csv＝主檔原樣；rev_B.csv＝主檔＋revenue_raw 補缺（只補主檔沒有的「公司×月份」，主檔有的不動；月份範圍限主檔的 2005-01～2026-09）；
  rev_C.csv＝同 B 但 KY／外國企業（名稱曾含 KY 或以 F- / F* 開頭）不補。
用法：python3 -I -B s94_build_rev.py"""
import sys, gzip, io, json, re, glob, os
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent
RAW = HERE.parents[1] / 'raw' / 'tsd_extra' / 'revenue_raw' / 'first'
MASTER = '/home/user/niklavsd/bottleneck-signals/data/processed/tw_revenue.csv'
SC = Path('/tmp/claude-0/-home-user/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/s94')
OUT = HERE / 'out'

def read_raw():
    fr = []
    for sub in ('early/revenue', 'mops/revenue_hist'):
        for f in sorted(glob.glob(str(RAW / sub / '*.csv.gz'))):
            d = pd.read_csv(f, dtype=str, encoding='utf-8-sig')
            d = d.rename(columns={'當月營收': 'rev', '當月累計營收': 'cum', '去年當月營收': 'rev_ly'})[['stock_id', 'name', 'period', 'market', 'rev', 'cum', 'rev_ly']]
            d['src'] = sub; d['file'] = os.path.basename(f)
            fr.append(d)
    d = pd.concat(fr, ignore_index=True)
    for c in ('rev', 'cum', 'rev_ly'):
        d[c] = pd.to_numeric(d[c].str.replace(',', ''), errors='coerce')
    return d

if __name__ == '__main__':
    d = read_raw()
    print('raw rows', len(d), d.period.min(), d.period.max(), d.market.value_counts().to_dict())
    print('file period mismatch', int((d.file.str[:7] != d.period).sum()), 'dup(code,period,market)', int(d.duplicated(['stock_id', 'period', 'market']).sum()),
          'dup(code,period)', int(d.duplicated(['stock_id', 'period']).sum()), 'rev NaN', int(d.rev.isna().sum()), 'rev<=0', int((d.rev <= 0).sum()))
    d.to_parquet(SC / 'raw_long.parquet')
