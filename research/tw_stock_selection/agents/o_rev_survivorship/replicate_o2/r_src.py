#!/usr/bin/env python3
"""複製員：從 taiwanan/TW_Stock_Exchange_Crawler 的 git 全歷史，取每檔 monthly_raw_data/<code>.csv「最後存在」的版本，
解析成長表 my_src.csv（code,name,p,rev,rev_ly,yoy_mops）。另外抽 tse_trading_raw_data（同樣取最後存在版本）給價格修補用。"""
import subprocess, io, json, sys, hashlib
from pathlib import Path
import pandas as pd, numpy as np
H = Path(__file__).resolve().parent
REPO = H.parent / 'raw' / 'taiwanan_TW_Stock_Exchange_Crawler'

def git(*a, binary=False):
    out = subprocess.run(['git', '-C', str(REPO), *a], capture_output=True, check=True).stdout
    return out if binary else out.decode('utf-8', 'replace')

def last_versions(sub):
    commits = git('log', '--reverse', '--format=%H', '--', sub).split()
    last = {}
    for c in commits:
        for line in git('ls-tree', c, sub + '/').splitlines():
            meta, path = line.split('\t')
            last[path] = (c, meta.split()[2])
    return commits, last

def cat_blob(b):
    return git('cat-file', '-p', b, binary=True).decode('utf-8-sig', 'replace')

head = git('rev-parse', 'HEAD').strip()
commits, last = last_versions('monthly_raw_data')
rows = []
for path, (c, b) in sorted(last.items()):
    if not path.endswith('.csv'): continue
    txt = cat_blob(b)
    lines = txt.splitlines()
    name, code = [x.strip() for x in lines[0].split(',')[:2]]
    code = code.strip()
    df = pd.read_csv(io.StringIO('\n'.join(lines[1:])), dtype=str)
    for _, r in df.iterrows():
        rows.append((code, name, r['日期'], r['當月營收'], r['去年同月營收'], r['YOY(%)'], c[:9]))
D = pd.DataFrame(rows, columns=['code', 'name', 'ym', 'rev', 'rev_ly', 'yoy_mops', 'commit'])
for k in ('rev', 'rev_ly', 'yoy_mops'):
    D[k] = pd.to_numeric(D[k].str.replace(',', ''), errors='coerce')
D['p'] = pd.PeriodIndex(D['ym'].str.replace('/', '-'), freq='M').astype(str)
D = D.drop_duplicates(['code', 'p'], keep='last')
D.to_csv(H / 'my_src.csv', index=False)
meta = {'head': head, 'n_commits': len(commits), 'n_files': D['code'].nunique(), 'n_rows': len(D), 'p_min': D['p'].min(), 'p_max': D['p'].max(),
        'sha256': hashlib.sha256((H / 'my_src.csv').read_bytes()).hexdigest()}
print(meta)

# 價格：tse_trading_raw_data
commits2, last2 = last_versions('tse_trading_raw_data')
want = set(sys.argv[1].split(',')) if len(sys.argv) > 1 else None
prow = []
for path, (c, b) in sorted(last2.items()):
    code = Path(path).stem
    if want is not None and code not in want: continue
    lines = cat_blob(b).splitlines()
    df = pd.read_csv(io.StringIO('\n'.join(lines[1:])), dtype=str)
    df.columns = [x.strip() for x in df.columns]
    df['code'] = code
    prow.append(df)
PX = pd.concat(prow, ignore_index=True)
def roc(s):
    y, m, d = s.split('/'); return pd.Timestamp(int(y) + 1911, int(m), int(d))
PX['date'] = PX['日期'].map(roc)
for k, kk in [('成交股數', 'vol'), ('開盤價', 'open'), ('最高價', 'high'), ('最低價', 'low'), ('收盤價', 'close'), ('漲跌價差', 'chg'), ('成交金額', 'value')]:
    PX[kk] = pd.to_numeric(PX[k].astype(str).str.replace(',', '').str.replace('X', ''), errors='coerce')
PX = PX[['code', 'date', 'vol', 'open', 'high', 'low', 'close', 'chg', 'value']].drop_duplicates(['code', 'date'], keep='last')
PX.to_parquet(H / 'my_tse_px.parquet')
meta['px'] = {'n_codes': PX['code'].nunique(), 'n_rows': len(PX), 'n_commits': len(commits2)}
(H / 'my_src_meta.json').write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding='utf-8')
print(meta['px'])
