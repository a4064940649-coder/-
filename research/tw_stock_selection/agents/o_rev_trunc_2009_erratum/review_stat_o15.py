#!/usr/bin/env python3
"""反方審查 o15（資料與統計）：只 import erlib，不改研究員檔案，不寫入任何檔案。python3 -I -B review_stat_o15.py"""
import sys, json
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import erlib as E
import numpy as np, pandas as pd
base = E.lists(E.VU('base')); days = sorted(base); D1 = [d for d in days if d < '2013-01-01']
_cause0 = E.cause
def run(causes, drop=(), mode='opt'):
    E.SURV = tuple(causes)
    E.cause = (lambda c, m: 'DROPPED' if c in drop else _cause0(c, m))
    try:
        R = E.lists(E.VU(mode, 'SURV'))
    finally:
        E.SURV = ('TRUNC', 'IFRS13', 'PRE12', 'HOLE', 'NEVER_DOM'); E.cause = _cause0
    ins = [(d, c) for d in days for c in R[d]['picks'] if c in set(R[d]['aff'])]
    u1 = sum(1 for d, c in ins if d < '2013'); u2 = len(ins)
    return dict(u1=u1, s1=round(u1/810, 4), u2=u2, s2=round(u2/930, 4)), ins
FULL = ('TRUNC', 'IFRS13', 'PRE12', 'HOLE', 'NEVER_DOM')
out = {}
r, ins_full = run(FULL); out['full'] = r
for k in ('TRUNC', 'PRE12', 'HOLE', 'IFRS13'):
    out['only_'+k] = run((k,))[0]
out['only_TRUNC_no5854'] = run(('TRUNC',), drop={'5854'})[0]
out['only_TRUNC_no5854_2473'] = run(('TRUNC',), drop={'5854', '2473'})[0]
out['TRUNC+PRE12_no5854'] = run(('TRUNC', 'PRE12'), drop={'5854'})[0]
# 金融：C0 FINMASK 是否涵蓋已下市金融
for c in ('5854', '2880', '2886', '2801', '2834', '6012', '6004', '2820', '2847'):
    if c in E.CI: out.setdefault('finmask', {})[c] = bool(E.V.FINMASK[E.CI[c]])
# 依 listing_master 刪除所有銀行/金控/證券代碼（28xx、58xx、60xx）後的全口徑
finlike = {c for c in E.COLS if c[:2] in ('28',) or c[:2] == '58' or c[:2] == '60'}
out['full_drop_finlike'] = run(FULL, drop=finlike)[0]
# 集中度：逐一去除前 k 大貢獻股後（重跑，含排擠互動）
cnt = pd.Series([c for d, c in ins_full if d < '2013']).value_counts()
out['top_contrib_2006_2012'] = cnt.head(8).to_dict()
for k in (3, 5, 8, 10):
    out[f'drop_top{k}'] = run(FULL, drop=set(cnt.index[:k]))[0]
# 依年份：2006～2012 份額若只用單年最大剔除
yr = pd.Series([d[:4] for d, c in ins_full if d < '2013']).value_counts().to_dict(); out['by_year_2006_2012'] = yr
# 仍上市（delist 空白）或下市於 2023 後的受影響股：不是 2014 前下市，與「事後頁只列當時存活者」說法的關係
still = []
for d, c in ins_full:
    dl = E.LM.get(c, {}).get('delist_date', '')
    if dl == '' or dl >= '2023': still.append((d, c, E.NAME.get(c, ''), dl, E.cause(c, pd.Period(d, 'M'))))
out['inserted_still_listed_or_late_delist'] = still
# HOLE 原因的受影響股月：缺的是哪些月份
hole_months = {}
for d in days:
    m = pd.Period(d, 'M')
    R = None
for d, c in ins_full:
    m = pd.Period(d, 'M')
    if E.cause(c, m) == 'HOLE':
        miss = [str(x) for x in [m-k for k in (1,2,3,13,14,15)] if not np.isfinite(E.LV[c].get(x, np.nan))]
        for x in miss: hole_months[x] = hole_months.get(x, 0) + 1
out['HOLE_missing_months_inserted'] = hole_months
# 二項近似：若真實命中率 = U_carry 口徑 16/810，5% 門檻的單側信賴上界
from math import comb
def ub(k, n, a=0.05):
    lo, hi = k/n, 1.0
    for _ in range(60):
        p = (lo+hi)/2; cdf = sum(comb(n, i)*p**i*(1-p)**(n-i) for i in range(k+1))
        lo, hi = (p, hi) if cdf > a else (lo, p)
    return round(p, 4)
out['carry_binom_ub95_2006_2012'] = ub(16, 810)
print(json.dumps(out, ensure_ascii=False, indent=1, default=str))
