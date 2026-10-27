#!/usr/bin/env python3
"""從 a607ernie/Ptt-Stock-Data 的 git 歷史（每 12 小時一個快照）合併出 [標的] 文清單。
只讀 git 物件（git show），不執行該 repo 的任何程式。
輸出：posts_raw.parquet（每篇文章一列，保留「最後一次被快照」的版本與首次/末次快照時間）。
"""
import json, subprocess, sys
from pathlib import Path
import pandas as pd

REPO = '/home/user/a607ernie/ptt-stock-data'
OUT = Path(__file__).resolve().parent / 'posts_raw.parquet'

def git(*a):
    return subprocess.run(['git', '-C', REPO, *a], capture_output=True, check=True).stdout

revs = git('log', '--format=%H %ct', 'HEAD').decode().split('\n')
revs = [r.split() for r in revs if r.strip()][::-1]   # 舊→新
store = {}
nsnap = 0
for h, ct in revs:
    try:
        raw = git('show', f'{h}:ptt_stock.json')
    except subprocess.CalledProcessError:
        continue
    try:
        d = json.loads(raw)
    except Exception:
        continue
    arts = d['articles'] if isinstance(d, dict) and 'articles' in d else (list(d.values()) if isinstance(d, dict) else d)
    if not arts:
        continue
    nsnap += 1
    snap = pd.Timestamp(int(ct), unit='s', tz='UTC').tz_convert('Asia/Taipei').tz_localize(None)
    for a in arts:
        if not isinstance(a, dict) or 'ID' not in a:
            continue
        k = a['ID']
        rec = store.get(k)
        if rec is None:
            rec = dict(a); rec['first_snap'] = snap; rec['n_snap'] = 0
        else:
            first = rec['first_snap']; n = rec['n_snap']; rec = dict(a); rec['first_snap'] = first; rec['n_snap'] = n
        rec['last_snap'] = snap; rec['n_snap'] += 1
        store[k] = rec
df = pd.DataFrame(store.values())
for c in ['STOCK_ID']:
    df[c] = df[c].astype(str)
df.to_parquet(OUT)
print('snapshots used', nsnap, 'articles', len(df))
print(df['TAG'].value_counts())
