#!/usr/bin/env python3
"""來源 A：taiwanan/TW_Stock_Exchange_Crawler 的 monthly_raw_data（2018-06～2019-09 逐月爬 MOPS t21sc03，國內＋國外 上市＋上櫃）。
raw/ 下的 clone 視為不可信資料：只用 git 讀檔內容（git show），不執行其中任何程式。
對每個 monthly_raw_data/<code>.csv，取它在歷史中「最後一次存在」的版本（被刪掉的檔用刪除前一版）。
輸出 src_taiwanan.csv：code,name,ym,rev,rev_prev_month,rev_last_year,yoy_pct（單位千元，與 MOPS 同）。
"""
import subprocess, io, csv, re, hashlib, json
from pathlib import Path
import pandas as pd
HERE = Path(__file__).resolve().parent
REPO = HERE / 'raw' / 'taiwanan_TW_Stock_Exchange_Crawler'

def git(*a):
    return subprocess.run(['git', '-C', str(REPO), *a], capture_output=True, check=True).stdout

# 每個路徑最後存在的 commit
log = git('log', '--format=@%H', '--name-status', '--', 'monthly_raw_data').decode()
last = {}
cur = None
for line in log.splitlines():
    if line.startswith('@'):
        cur = line[1:]; continue
    if not line.strip(): continue
    st, path = line.split('\t')[0], line.split('\t')[-1]
    if not path.endswith('.csv') or path in last: continue
    # log 由新到舊；第一次看到：若是 D（刪除），內容在其父 commit
    last[path] = (cur + '^') if st.startswith('D') else cur
rows = []; meta = {}
for path, rev in sorted(last.items()):
    code = Path(path).stem
    if not re.fullmatch(r'\d{4}', code): continue
    raw = git('show', f'{rev}:{path}')
    txt = raw.decode('utf-8-sig', errors='replace')
    lines = txt.splitlines()
    name = lines[0].split(',')[0].strip() if lines else ''
    meta[code] = {'name': name, 'commit': rev, 'sha256': hashlib.sha256(raw).hexdigest()}
    for r in csv.reader(lines[2:]):
        if len(r) < 6 or not re.fullmatch(r'\d{4}/\d{2}', r[0].strip()): continue
        def f(x):
            x = x.strip().replace(',', '')
            try: return float(x)
            except ValueError: return float('nan')
        rows.append((code, name, r[0].strip().replace('/', '-'), f(r[1]), f(r[2]), f(r[3]), f(r[5])))
df = pd.DataFrame(rows, columns=['code', 'name', 'ym', 'rev', 'rev_prev_month', 'rev_last_year', 'yoy_pct'])
df = df.drop_duplicates(['code', 'ym'], keep='last').sort_values(['code', 'ym'])
df.to_csv(HERE / 'src_taiwanan.csv', index=False)
(HERE / 'src_taiwanan_meta.json').write_text(json.dumps({'n_files': len(meta), 'n_rows': len(df), 'ym_min': df.ym.min(), 'ym_max': df.ym.max(),
                                                       'head': git('rev-parse', 'HEAD').decode().strip(), 'files': meta}, ensure_ascii=False), encoding='utf-8')
print(len(meta), len(df), df.ym.min(), df.ym.max())
print(df.groupby('ym').code.nunique().describe())
