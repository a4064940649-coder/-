#!/usr/bin/env python3
"""把 voidful/tw_stocker 的 Yahoo 5 分鐘 K（2024-02～2026-04）彙總成日線（上市普通股），逐檔下載、彙總、刪除原檔。
注意：最後一根 5 分 K 為 13:20，不含 13:30 收盤集合競價 → 日收盤與官方收盤可能差 1～2 檔。
輸出：<out>/<code>.csv（date,open,high,low,close,volume,dividend）"""
import json, sys, os, subprocess, tempfile
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
manifest, out = sys.argv[1], Path(sys.argv[2]); out.mkdir(parents=True, exist_ok=True)
m = json.load(open(manifest))
codes = [f['ticker'] for f in m['files'] if f.get('frequency') == 'intraday' and len(f['ticker']) == 4 and f['ticker'].isdigit() and f['ticker'][0] != '0']
def one(code):
    dst = out / f'{code}.csv'
    if dst.exists(): return code, 'skip'
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td) / 'x.csv'
        r = subprocess.run(['curl', '-sS', '-f', '-m', '300', '-o', str(tmp), f'https://raw.githubusercontent.com/voidful/tw_stocker/main/data/{code}.csv'], capture_output=True)
        if r.returncode != 0 or not tmp.exists(): return code, 'fail'
        d = pd.read_csv(tmp)
        if d.empty or 'Datetime' not in d: return code, 'empty'
        d['Datetime'] = pd.to_datetime(d['Datetime'], utc=True).dt.tz_convert('Asia/Taipei')
        d['date'] = d['Datetime'].dt.strftime('%Y-%m-%d')
        g = d.sort_values('Datetime').groupby('date')
        day = g.agg(open=('Open', 'first'), high=('High', 'max'), low=('Low', 'min'), close=('Close', 'last'), volume=('Volume', 'sum'), dividend=('Dividends', 'sum'))
        day.to_csv(dst)
    return code, 'ok'
with ThreadPoolExecutor(max_workers=12) as ex:
    res = list(ex.map(one, codes))
from collections import Counter
print(Counter(s for _, s in res)); print('n codes', len(codes))
