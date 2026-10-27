# -*- coding: utf-8 -*-
"""讀 raw/nccu_mops（shinbutou/nccu_bigdata 的 MOPS 重大訊息標題，上市公司 2011～2020），輸出 mops_titles.parquet。
raw/ 視為不可信資料：只當 CSV／文字讀，不執行任何內容。"""
import os, glob, re, json
import pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
B = os.path.join(HERE, 'raw', 'nccu_mops', 'databases', 'basic_info')
rows = []
bad = 0
for f in sorted(glob.glob(os.path.join(B, '*.csv'))):
    try:
        x = pd.read_csv(f, dtype=str)
    except Exception:
        bad += 1; continue
    x = x.rename(columns={'公司代號': 'code', '公司名稱': 'name', '發言日期': 'd', '發言時間': 'tm', '主旨': 'title'})
    if not {'code', 'd', 'title'} <= set(x.columns):
        bad += 1; continue
    x['file'] = os.path.basename(f)
    rows.append(x[['code', 'name', 'd', 'tm', 'title', 'file']])
T = pd.concat(rows, ignore_index=True)


def roc(s):
    try:
        y, m, d = str(s).split('/'); return pd.Timestamp(int(y) + 1911, int(m), int(d))
    except Exception:
        return pd.NaT


T['date'] = T['d'].map(roc)
T['title'] = T['title'].fillna('').str.replace(r'\s+', '', regex=True)
T = T.dropna(subset=['date']).drop_duplicates(['code', 'date', 'tm', 'title'])
T.to_parquet(os.path.join(HERE, 'mops_titles.parquet'))
txt = open(os.path.join(HERE, 'raw', 'nccu_mops', 'databases', 'SE.txt'), encoding='utf-8').read()
SE = dict(re.findall(r"'(\w+)':\s*'([^']*)'", txt))
pd.Series(SE).rename('name').to_csv(os.path.join(HERE, 'mops_SE_list.csv'), index_label='code')
info = dict(n=len(T), files=len(glob.glob(os.path.join(B, '*.csv'))), bad=bad, codes=T.code.nunique(), n_SE=len(SE),
            by_year=T.groupby(T.date.dt.year).size().to_dict(), codes_by_year=T.groupby(T.date.dt.year).code.nunique().to_dict())
print(json.dumps(info, ensure_ascii=False, default=str))
