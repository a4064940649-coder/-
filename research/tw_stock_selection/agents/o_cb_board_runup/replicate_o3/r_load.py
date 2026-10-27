#!/usr/bin/env python3
"""複製員：從 nccu_mops 原始 CSV 自己讀 MOPS 重大訊息標題。"""
import ast, glob, os
from pathlib import Path
import pandas as pd
HERE = Path(__file__).resolve().parent
D = HERE.parent / 'raw/nccu_mops/databases'
se = ast.literal_eval((D / 'SE.txt').read_text(encoding='utf-8'))
rows = []
for f in sorted(glob.glob(str(D / 'basic_info/*.csv'))):
    b = os.path.basename(f)[:-4].split('_')
    code = b[0]
    try:
        x = pd.read_csv(f, dtype=str)
    except Exception as e:
        print('ERR', f, e); continue
    x['file_code'] = code; x['file_roc'] = b[-2]
    rows.append(x)
d = pd.concat(rows, ignore_index=True)
d = d.rename(columns={'公司代號': 'code', '公司名稱': 'name', '發言日期': 'rdate', '發言時間': 'time', '主旨': 'title'})
def conv(s):
    try:
        y, m, dd = s.split('/'); return pd.Timestamp(int(y) + 1911, int(m), int(dd))
    except Exception: return pd.NaT
d['date'] = d['rdate'].map(conv)
d = d[['file_code', 'code', 'name', 'date', 'time', 'title', 'file_roc']]
print(len(d), d['file_code'].nunique(), d['date'].min(), d['date'].max(), 'SE list', len(se), 'codes in SE', len(set(d.file_code) & set(se)))
print((d.code != d.file_code).sum(), 'code mismatch')
d.to_parquet(HERE / 'my_titles.parquet')
