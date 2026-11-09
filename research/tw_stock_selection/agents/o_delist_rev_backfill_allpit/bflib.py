#!/usr/bin/env python3
"""o18 題 1 共用：tw_revenue 水準表、taiwanan 原始檔解析（版本 T1…）。不算任何報酬。"""
import re, csv, io
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent
LONG = Path('/home/user/niklavsd/bottleneck-signals/data/processed/tw_revenue.csv')
_LV = None
def tw_levels():
    """與 ia_core.yoy_from_long 相同的水準表（sii 優先去重）。"""
    global _LV
    if _LV is None:
        d = pd.read_csv(LONG, dtype={'code': str, 'market': str})
        d['p'] = pd.PeriodIndex(pd.to_datetime(d['month']), freq='M')
        d['pri'] = (d['market'] != 'sii').astype(int)
        d = d.sort_values(['p', 'code', 'pri']).drop_duplicates(['p', 'code'])
        lv = d.pivot(index='p', columns='code', values='revenue_k_twd')
        _LV = lv.reindex(pd.period_range(lv.index.min(), lv.index.max(), freq='M'))
    return _LV
def _f(x):
    x = x.strip().replace(',', '')
    try: return float(x)
    except ValueError: return np.nan
def taiwanan(ver='T1'):
    """T1：每檔最後存在版本；表頭第 2 行「日期,當月營收,上月營收,去年同月營收,MOM(%),YOY(%),…」；日期 YYYY/MM。
    回傳 code, ym, rev, prev, ly, mom, yoy, commit, commit_utc。"""
    d = pd.read_csv(HERE / 'raw/taiwanan_monthly_lastver.csv.gz', dtype=str, keep_default_na=False)
    rows = []
    for (code, commit, cu), g in d.groupby(['code', 'commit', 'commit_utc'], sort=False):
        L = g.sort_values('line_no', key=lambda s: s.astype(int)).line.tolist()
        if len(L) < 2 or not L[1].startswith('日期'): continue
        for r in csv.reader(L[2:]):
            if len(r) < 6 or not re.fullmatch(r'\d{4}/\d{2}', r[0].strip()): continue
            rows.append((code, r[0].strip().replace('/', '-'), _f(r[1]), _f(r[2]), _f(r[3]), _f(r[4]), _f(r[5]), commit, cu))
    T = pd.DataFrame(rows, columns=['code', 'ym', 'rev', 'prev', 'ly', 'mom', 'yoy', 'commit', 'commit_utc'])
    return T.drop_duplicates(['code', 'ym'], keep='last').reset_index(drop=True)
