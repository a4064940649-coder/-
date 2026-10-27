#!/usr/bin/env python3
"""複製員：從 FSC 申報案件彙總表原始檔自行解析成統一表 fsc_parsed.csv。"""
import re, warnings, sys
from pathlib import Path
import numpy as np, pandas as pd
warnings.filterwarnings('ignore')
R = Path(__file__).resolve().parent.parent / 'raw' / 'fsc_cases'
OUT = Path(__file__).resolve().parent
SRC = {92: 'case-92.xls', 93: 'case-93.xls', 94: 'case-94.xls', 95: 'case-95.xls', 96: 'case-96.xls', 97: 'case-97.xls',
       98: 'recovered_98.csv', 99: 'recovered_99.csv', 100: 'case-100.xls', 101: 'case-101.xls', 102: 'recovered_102.csv',
       103: 'case-103.xlsx', 104: 'case-104.xls', 106: 'conv/case10612.xlsx', 107: 'conv/case10712.xlsx', 108: 'case10812.xls',
       109: 'case10912.xls', 110: 'case11012(xls).xls', 111: 'case11112.xlsx', 112: '112729申報案件彙總表 .xlsx',
       113: '11311473申報案件彙總表.xlsx', 114: '1147281申報案件彙總表.xlsx'}

def load(f):
    p = R / f
    if f.endswith('.csv'):
        d = pd.read_csv(p, dtype={'value': str})
        v = d.pivot_table(index='row', columns='col', values='value', aggfunc='first')
        v = v.reindex(range(int(d.row.max()) + 1)).reindex(columns=range(int(d.col.max()) + 1))
        v.columns = range(v.shape[1]); v.index = range(len(v))
        return v
    return pd.read_excel(p, header=None, dtype=str, engine=('xlrd' if f.endswith('.xls') else None))

def rocdate(x):
    if x is None or (isinstance(x, float) and np.isnan(x)): return pd.NaT
    s = str(x).strip()
    if s in ('', 'nan', 'None'): return pd.NaT
    s = s.split('.')[0] if re.fullmatch(r'\d+\.0', s) else s
    m = re.fullmatch(r'(\d{2,3})/(\d{1,2})/(\d{1,2})', s)
    if m: y, mo, d = map(int, m.groups())
    else:
        m = re.fullmatch(r'(\d{2,3})(\d{2})(\d{2})', s)
        if not m: return pd.NaT
        y, mo, d = map(int, m.groups())
    try: return pd.Timestamp(y + 1911, mo, d)
    except Exception: return pd.NaT

def num(x):
    s = str(x).replace(',', '').strip()
    try: return float(s)
    except Exception: return np.nan

rows = []
for yr, f in SRC.items():
    v = load(f)
    h = next(i for i in range(40) if any(str(x).strip() == '證券代號' for x in v.iloc[i].values))
    cols = [re.sub(r'\s', '', str(x)) for x in v.iloc[h].values]
    body = v.iloc[h + 1:].copy(); body.columns = range(body.shape[1])
    ci = {c: i for i, c in enumerate(cols)}
    def col(*names):
        for n in names:
            for c, i in ci.items():
                if c.startswith(n): return body[i]
        return pd.Series([np.nan] * len(body), index=body.index)
    new = '案件類別' in ci
    t = pd.DataFrame({'roc_year': yr, 'code': col('證券代號').astype(str).str.strip().str.replace(r'\.0$', '', regex=True), 'ctype': col('公司型態').astype(str).str.strip(),
                      'result': col('結案類型').astype(str).str.strip(), 'name': col('公司名稱').astype(str).str.strip(),
                      'item': (col('案件類別') if new else col('申報事項')).astype(str).str.strip(),
                      'amount': col('金').map(num), 'ccy': col('幣別') if new else np.nan, 'price': col('發行價格').map(num),
                      'recv': col('收文日期').map(rocdate), 'eff': (col('生效日期') if new else col('申報生效日期')).map(rocdate),
                      'note1': col('備註1').astype(str), 'note2': col('備註2').astype(str), 'src': f})
    if not new:
        t['amount'] = t['amount'] * 1000
        t['ctype'] = t['ctype'].str.replace(r'\.0$', '', regex=True).map({'1': '上市', '2': '上櫃', '3': '未上市櫃', '4': '補辦', '5': '興櫃', '6': '外國'}).fillna(t['ctype'])
        t['result'] = t['result'].str.replace(r'\.0$', '', regex=True).map({'1': '生效', '2': '退件', '3': '核減', '4': '自行撤回', '6': '撤銷'}).fillna(t['result'])
    t = t[t['code'].str.fullmatch(r'[0-9A-Z]{4,6}') & t['recv'].notna()]
    rows.append(t)
    print(yr, f, len(t), t['recv'].min().date(), t['recv'].max().date(), file=sys.stderr)
D = pd.concat(rows, ignore_index=True)
D.to_csv(OUT / 'fsc_parsed.csv', index=False)
L = D[D.ctype == '上市']
print(L.groupby('item').size().sort_values(ascending=False).head(70).to_string())
print(L.result.value_counts())
