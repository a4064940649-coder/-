#!/usr/bin/env python3
"""稽核獨立複製用的掛鉤（只在暫存副本的 run_official 裡被呼叫；不改任何既有檔案）。
MODE：'base'（不動）、'A'（0050 含息序列換成自建）、'B'（減資／面額變更恢復買賣日用事件參考價）、'AB'、
      'Bp'（B＋3 個因颱風停市順延的事件套在實際恢復買賣日；只做穩健性檢查）、'ABp'。
A：載入面板後覆寫 P.E50（自建序列）與 P.E50O（＝P.E50 × 原始開盤 ÷ 原始收盤，同 stlib.load 的算法）。
B：衍生陣列（REF、R、漲跌停價、還原指數 G、季線等）都由漲跌價差一路算下去，載入後再覆寫會漏；所以改成包住 cslib.exright_fix_chg——
   原函式算完後，對「上市、當天有成交、當天在事件表」的格子設 漲跌價差＝收盤 − 事件參考價，之後引擎照原樣往下算。"""
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path('/home/user/-/research/tw_stock_selection/agents/r_fix_e50_reduce')
D = HERE / 'data'
INFO = {}
TYPHOON = {('2016-09-28', '2314'): '2016-09-29', ('2016-09-28', '3055'): '2016-09-29', ('2019-09-30', '2323'): '2019-10-01'}


def events(plus=False):
    ev = pd.concat([pd.read_csv(D / 'ev_reduce_tsd.csv', dtype=str), pd.read_csv(D / 'ev_parvalue_tsd.csv', dtype=str)], ignore_index=True)
    ev = ev.drop_duplicates(['date', 'stock_id'], keep='last')
    if plus:
        ev['date'] = [TYPHOON.get((d, c), d) for d, c in zip(ev['date'], ev['stock_id'])]
    ev['date'] = pd.to_datetime(ev['date']); ev['ref'] = pd.to_numeric(ev['ref_price'], errors='coerce')
    return ev[ev['ref'] > 0]


def pre(cslib, mode):
    if 'B' not in mode:
        return
    orig = cslib.exright_fix_chg
    ev = events(plus='p' in mode)

    def wrapped(IDX, COLS, C, CHG, MK, e):
        CHG2, info = orig(IDX, COLS, C, CHG, MK, e)
        di = {d: i for i, d in enumerate(pd.DatetimeIndex(IDX))}; ci = {c: j for j, c in enumerate(COLS)}
        n = 0; cells = []
        for d, c, ref in zip(ev['date'], ev['stock_id'], ev['ref']):
            if d in di and c in ci:
                t, j = di[d], ci[c]
                if MK[t, j] == 1 and np.isfinite(C[t, j]) and C[t, j] > 0:
                    CHG2[t, j] = C[t, j] - ref; n += 1; cells.append((str(d.date()), c))
        info = dict(info, n_reduce_parvalue=n)
        INFO['n_reduce_parvalue'] = n; INFO['cells'] = cells
        return CHG2, info
    cslib.exright_fix_chg = wrapped


def post(P, mode, cache):
    if 'A' not in mode:
        return
    e = pd.read_csv(D / 'e50_tr_mine.csv', parse_dates=['date']).set_index('date')['adj'].sort_index()
    old = np.array(P.E50, copy=True)
    new = e.reindex(P.IDX.union(e.index)).ffill().reindex(P.IDX).to_numpy()
    assert np.isfinite(new).all()
    o = pd.read_csv(Path(cache) / 'e50_ohlc.csv', parse_dates=['date']).set_index('date').reindex(P.IDX)
    ratio = (o['open'] / o['close']).to_numpy(); ratio = np.where(np.isfinite(ratio) & (ratio > 0), ratio, 1.0)
    P.E50 = new; P.E50O = new * ratio
    INFO['e50_missing_days'] = int(e.reindex(P.IDX).isna().sum())
    INFO['e50_end_ratio_new_over_old'] = float((new[-1] / new[P.t0]) / (old[-1] / old[P.t0]))
