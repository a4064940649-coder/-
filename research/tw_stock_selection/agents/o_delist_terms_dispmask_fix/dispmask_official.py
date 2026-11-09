#!/usr/bin/env python3
"""官方全歷史處置遮罩（基礎設施勘誤；不改 ia_core.py）。
資料：o_rev_avail_ds_pit_count/raw/disposal_official_universe.csv（sha 2842f45d…；o_disposal_official_fullhist universe_flags 的副本；
起訖日全部來自官方處置公告 disposal_hist_rejudge；overlap_proxy＝與 disposal_bear_longhistory 代理事件重疊，不代表起訖由代理推得）。
cover(d, e)：官方處置期涵蓋訊號日 d 或成交日 e 的代碼集合。
模式（訊號日 d）：
  'union'   ＝原遮罩 ∪ cover（d ∈ [2006-01-01, 2021-09-23]）  ← 與 integrate_o17 C_DSP 定義相同（smoke 斷言重現 27.61%）
  'replace' ＝d ∈ [2014-01-01, 2021-09-23]：cover 取代原遮罩；其他日期＝原遮罩
  'union_noovl'＝同 union，但只用 overlap_proxy=False 的列（P5 對照；見 FINDINGS 說明）
  'union14' ＝同 union，視窗改 [2014-01-01, 2021-09-23]"""
import hashlib
from pathlib import Path
import numpy as np, pandas as pd
AG = Path(__file__).resolve().parent.parent
DU_F = AG / 'o_rev_avail_ds_pit_count' / 'raw' / 'disposal_official_universe.csv'
DU_SHA = '2842f45d825861bfe0168f7a9736364fc19adee2b4d9bc3b5a894d4e22db8b40'
assert hashlib.sha256(DU_F.read_bytes()).hexdigest() == DU_SHA
DU = pd.read_csv(DU_F, dtype={'code': str}, parse_dates=['start', 'end'])
HI = pd.Timestamp('2021-09-23')
WIN = {'union': (pd.Timestamp('2006-01-01'), HI), 'replace': (pd.Timestamp('2014-01-01'), HI),
       'union_noovl': (pd.Timestamp('2006-01-01'), HI), 'union14': (pd.Timestamp('2014-01-01'), HI)}


def cover_codes(d, e, rows='all'):
    X = DU if rows == 'all' else DU[DU.overlap_proxy == False]
    x = X[((X.start <= d) & (X.end >= d)) | ((X.start <= e) & (X.end >= e))]
    return set(x.code)


def make_mask(orig_fn, IDX, CI, N, mode):
    """回傳 callable(d)→bool[N]。orig_fn＝原遮罩（ia_core.disposal_mask 或 V7 DMT）。"""
    assert mode in WIN
    lo, hi = WIN[mode]; rows = 'noovl' if mode == 'union_noovl' else 'all'
    cache = {}
    def f(d):
        d = pd.Timestamp(d)
        if d in cache: return cache[d]
        m0 = orig_fn(d)
        if not (lo <= d <= hi):
            cache[d] = m0; return m0
        t = IDX.get_loc(d); e = IDX[t + 1] if t + 1 < len(IDX) else d
        m = np.zeros(N, bool) if mode == 'replace' else np.array(m0, bool).copy()
        for c in cover_codes(d, e, rows):
            if c in CI: m[CI[c]] = True
        cache[d] = m
        return m
    return f
