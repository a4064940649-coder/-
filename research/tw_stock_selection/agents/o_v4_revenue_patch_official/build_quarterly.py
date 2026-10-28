#!/usr/bin/env python3
"""新來源 Q：taiwanan/TW_Stock_Exchange_Crawler 的 tse_earning_raw_data（季營收，千元，2013Q1 起；git 全歷史取每季最後一個非 n/a 版本）。
只當 CSV 讀（git cat-file，不執行來源內任何程式）。Q4 欄是全年合計 → Q4＝全年 − Q1 − Q2 − Q3；2018Q2 等「半年合計」也依同規則偵測（值 ≥ 1.6×前季）只標記不用。
輸出 raw_quarterly.parquet（code, q, rev_k, kind）。
"""
import sys, subprocess, json, re, io
from pathlib import Path
HERE = Path(__file__).resolve().parent
import numpy as np, pandas as pd
REPO = HERE / 'raw' / 'taiwanan'

def git(*a, inp=None):
    return subprocess.run(['git', '-C', str(REPO)] + list(a), input=inp, capture_output=True, check=True).stdout

commits = git('log', '--all', '--format=%H %ct', '--', 'tse_earning_raw_data').decode().split('\n')
commits = [c.split() for c in commits if c.strip()]
commits.sort(key=lambda x: int(x[1]))          # 舊 → 新
blobs = []
for h, ts in commits:
    out = git('ls-tree', '-r', h, 'tse_earning_raw_data').decode()
    for line in out.splitlines():
        meta, path = line.split('\t')
        m = re.search(r'/(\d{4})\.csv$', path)
        if m: blobs.append((int(ts), m.group(1), meta.split()[2]))
uniq = sorted({b[2] for b in blobs})
data = git('cat-file', '--batch', inp=('\n'.join(uniq) + '\n').encode())
content = {}
i = 0
while i < len(data):
    nl = data.index(b'\n', i); hdr = data[i:nl].decode().split()
    sz = int(hdr[2]); content[hdr[0]] = data[nl + 1:nl + 1 + sz]; i = nl + 1 + sz + 1
rec = {}
for ts, code, b in blobs:
    txt = content[b].decode('utf-8-sig', errors='replace').splitlines()
    for line in txt[2:]:
        p = line.split(',')
        if len(p) < 2 or not re.match(r'\d{4} Q[1-4]', p[0]): continue
        try: v = float(p[1])
        except ValueError: continue
        rec[(code, p[0])] = (v, ts)         # 新版覆蓋舊版
rows = [(c, q, v) for (c, q), (v, ts) in rec.items()]
Q = pd.DataFrame(rows, columns=['code', 'qs', 'raw']).sort_values(['code', 'qs'])
Q['year'] = Q.qs.str[:4].astype(int); Q['qn'] = Q.qs.str[-1].astype(int)
out = []
for (c, y), g in Q.groupby(['code', 'year']):
    v = dict(zip(g.qn, g.raw))
    for qn in (1, 2, 3):
        if qn in v: out.append((c, f'{y}Q{qn}', v[qn], 'reported'))
    if 4 in v and all(q in v for q in (1, 2, 3)):
        q4 = v[4] - v[1] - v[2] - v[3]
        out.append((c, f'{y}Q4', q4, 'annual_minus_q123'))
R = pd.DataFrame(out, columns=['code', 'q', 'rev_k', 'kind'])
# 半年合計偵測（taiwanan 最後一季 2018Q2 有「H1」型態）：該季 ≥ 1.6 × 同年前季 → 標記 suspect
R['suspect'] = False
for c, g in R.groupby('code'):
    g = g.sort_values('q'); vals = g.rev_k.to_numpy()
    for k in range(1, len(g)):
        if vals[k - 1] > 0 and vals[k] >= 1.6 * vals[k - 1] and g.q.iloc[k].endswith(('Q2', 'Q3')):
            R.loc[g.index[k], 'suspect'] = True
R = R[R.rev_k > 0]
R.to_parquet(HERE / 'raw_quarterly.parquet')
meta = {'repo_head': git('rev-parse', 'HEAD').decode().strip(), 'n_commits': len(commits), 'n_codes': int(R.code.nunique()), 'n_rows': int(len(R)),
        'q_range': [R.q.min(), R.q.max()], 'n_suspect': int(R.suspect.sum())}
(HERE / 'raw_quarterly_meta.json').write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding='utf-8')
print(meta)
