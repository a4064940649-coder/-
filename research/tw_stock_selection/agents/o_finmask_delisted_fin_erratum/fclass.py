#!/usr/bin/env python3
"""本題分類器（F1 PIT／F2 靜態／F3 描述）：只讀既有原始檔，不 import 任何引擎。
所有可調部分（關鍵字、排除清單）由呼叫端傳入，迭代版本與凍結版本都用同一份程式。"""
import csv, re
from pathlib import Path
import pandas as pd

AG = Path(__file__).resolve().parent.parent
O13 = AG / 'o_pit_shares_industry_2006_2013/raw'
SL_PATH = Path('/home/user/stevenhsiehsheng-ctrl/tw-stock-screener/data/stock_list.csv')
REV_PATH = Path('/home/user/niklavsd/bottleneck-signals/data/processed/tw_revenue.csv')
LM_PATH = AG / 'o_v7_listing_exec_engine/listing_master.csv'
DELIST_PATH = AG / 'o_lc_sector_rotation_pit/raw/delist_official.csv'
TSE_FIN = 17
SNAP_FILES = {  # o13 dated 快照：生效日（UPDATE 列；無 UPDATE 列者用 commit 日；twse_list 用同 blob 的 UPDATE 2013-12-18）
    'industry_snap/goristock_14393dd2_grs_stock_no.csv': '2011-07-28',
    'industry_snap/goristock_d9d8662c_grs_stock_no.csv': '2011-08-17',
    'industry_snap/goristock_235da71f_grs_old_stock_no.csv': '2011-09-26',
    'industry_snap/goristock_c3c718ab_grs_stock_no.csv': '2012-03-28',
    'industry_snap/grs_893c8772_grs_stock_no.csv': '2012-07-31',
    'industry_snap/grs_568e6685_tools_twse_list.csv': '2013-12-18',
}
LATER_FILES = {  # 「最後已知分類」用的較晚來源（只在 dated 快照沒有該代號時使用；準 PIT）
    'center/StockList_20150522.csv': '2015-05-22', 'center/StockList_20160401.csv': '2016-04-01',
    'center/StockList_20170706.csv': '2017-07-06', 'center/isin_20230116_listed.csv': '2023-01-16',
}
FIN_LABELS = ('金融保險業', '金融保險', '金融業')

def _is4(c): return isinstance(c, str) and c.isdigit() and len(c) == 4

def load_snap(rel):
    fin, nm = {}, {}
    for r in list(csv.reader(open(O13 / rel, encoding='utf-8', errors='replace')))[1:]:
        r = [x.strip().strip('"').strip('﻿') for x in r]
        if len(r) >= 3 and _is4(r[0]):
            try:
                fin[r[0]] = int(r[2]) == TSE_FIN; nm[r[0]] = r[1]
            except ValueError:
                pass
    return fin, nm

def load_later(rel):
    fin, nm = {}, {}
    if 'isin' in rel:
        T = pd.read_csv(O13 / rel, dtype=str)
        for a, n, b in zip(T['code'], T['name'], T['industry']):
            if _is4(a) and isinstance(b, str): fin[a] = b.strip() in FIN_LABELS; nm[a] = n
    else:
        T = pd.read_csv(O13 / rel, dtype=str, on_bad_lines='skip')
        for a, n, b in zip(T['代號'], T['名稱'], T['產業別']):
            a = str(a).strip()
            if _is4(a) and isinstance(b, str): fin[a] = b.strip() in FIN_LABELS; nm[a] = n
    return fin, nm

SNAPS = []
for rel, eff in SNAP_FILES.items():
    f, n = load_snap(rel); SNAPS.append(dict(date=pd.Timestamp(eff), rel=rel, fin=f, name=n, kind='o13_dated'))
SNAPS.sort(key=lambda s: s['date'])
LATER = []
for rel, eff in LATER_FILES.items():
    f, n = load_later(rel); LATER.append(dict(date=pd.Timestamp(eff), rel=rel, fin=f, name=n, kind='later'))
_sl = pd.read_csv(SL_PATH, dtype=str).drop_duplicates('code')
SL2026 = dict(date=pd.Timestamp('2026-10-01'), rel='stock_list.csv(2026)', fin={c: (i in FIN_LABELS) for c, i in zip(_sl['code'], _sl['industry']) if _is4(c) and isinstance(i, str)},
              name={c: n for c, n in zip(_sl['code'], _sl['name']) if _is4(c)}, kind='later')
LATER.append(SL2026)
LATER.sort(key=lambda s: s['date'])
SL_CODES = set(_sl['code'])

# 名稱來源（F2 與 F1 證據）
NAMES = {}
def _addn(c, n):
    if _is4(c) and isinstance(n, str) and n.strip(): NAMES.setdefault(c, set()).add(n.strip())
for s in SNAPS + LATER:
    for c, n in s['name'].items(): _addn(c, n)
_rv = pd.read_csv(REV_PATH, dtype={'code': str}, usecols=['code', 'name']).drop_duplicates()
for c, n in zip(_rv['code'], _rv['name']): _addn(c, n)
DEL = {}
for r in csv.DictReader(open(DELIST_PATH, encoding='utf-8')):
    _addn(r['code'], r['name']); DEL.setdefault(r['code'], []).append(r['date'])
LM = {r['code']: r for r in csv.DictReader(open(LM_PATH, encoding='utf-8'))}

def is_delisted(c):
    """已下市：不在 2026 stock_list，或 listing_master 有下市日。"""
    return (c not in SL_CODES) or bool(LM.get(c, {}).get('delist_date', ''))

class Classifier:
    def __init__(self, kw, excl_f2=(), name_excl=()):
        """kw：關鍵字清單；excl_f2：F2 人工排除代號；name_excl：F1 證據與 F2 都不採用的名稱（關鍵字誤配）。"""
        self.kw = list(kw); self.KW = re.compile('|'.join(map(re.escape, self.kw)))
        self.excl_f2 = set(excl_f2); self.name_excl = set(name_excl)
        self._f1 = {}

    def kw_hit(self, c):
        return sorted(n for n in NAMES.get(c, ()) if n not in self.name_excl and self.KW.search(n))

    def evidence(self, c):
        return c[:2] in ('28', '58', '60') or bool(self.kw_hit(c))

    def f2(self, c):
        return (c[:2] in ('28', '58', '60') or bool(self.kw_hit(c))) and c not in self.excl_f2

    def f1(self, c, d):
        """回傳 (狀態, 來源)。狀態 ∈ FIN／NONFIN（PIT）、FIN_Q／NONFIN_Q（準 PIT）、UNKNOWN。"""
        d = pd.Timestamp(d)
        key = (c, d)
        if key in self._f1: return self._f1[key]
        pre = [s for s in SNAPS if s['date'] <= d and c in s['fin']]
        if pre:
            s = pre[-1]; out = ('FIN' if s['fin'][c] else 'NONFIN', s['rel'])
        else:
            post = sorted([s for s in SNAPS + LATER if s['date'] > d and c in s['fin']], key=lambda s: s['date'])
            if not post:
                out = ('UNKNOWN', 'no_source')
            else:
                s = post[0]; fin = s['fin'][c]; ev = self.evidence(c)
                if fin and ev: out = ('FIN_Q', s['rel'])
                elif (not fin) and (not ev): out = ('NONFIN_Q', s['rel'])
                else: out = ('UNKNOWN', s['rel'] + (':FIN_noevidence' if fin else ':NONFIN_withevidence'))
        self._f1[key] = out
        return out

    @staticmethod
    def f1_value(state, c0):
        if state in ('FIN', 'FIN_Q'): return True
        if state in ('NONFIN', 'NONFIN_Q'): return False
        return bool(c0)

    def describe(self):
        return dict(kw=self.kw, excl_f2=sorted(self.excl_f2), name_excl=sorted(self.name_excl))
