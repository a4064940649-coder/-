#!/usr/bin/env python3
"""S99 共用程式庫：針對兩個弱點的優化——選題「同一期營收的月中第二次進場檢查」（MID2）。
引擎：正式管線（agents/official_nav/run_official.py 的 E0｜W10_70 寫法；EXRIGHT_FIX、DEADLINE_FIX、INS15_MASK、REDUCE_FIX、0050 官方含息 五個開關全開）。
      模擬＝s10lib.simulate_s10（原樣呼叫，不改；引擎本來就會在任何「前一列 sig 有值」的日子開盤買進）。
本題只在「非訊號列」多寫 sig／rank（月中補進場），訊號列的名單、出場 X、資金配置完全不動。不修改任何既有檔。
時點：訊號日列 k（當月營收已到齊）→ 第 k＋D 列收盤後，只用列 ≤ k＋D 的還原收盤（RET20、60 日均線）與「訊號日 k 當時已可見」的 SURGE 分數再檢查一次 → 第 k＋D＋1 列開盤成交。
暫存：scratchpad/s99/cache（面板快取；用完刪）。一律 python3 -I -B。"""
import sys, gc
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
AG = HERE.parent
TMP = Path('/tmp/claude-0/-home-user/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/s99/cache')
TMP.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(AG / 's19_xq_threshold'))
import s19lib                                              # noqa: E402
from s19lib import s10lib, s9lib, s8lib, s7lib, s6lib, s5lib, s4lib, s3lib, s2lib, stlib, cslib, COSTS   # noqa: E402
for mod, attr in ((s19lib, 'S19'), (s10lib, 'S10'), (s9lib, 'S9'), (s8lib, 'S8'), (s7lib, 'S7'), (s6lib, 'S6'), (s5lib, 'S5'),
                  (s4lib, 'S4'), (s3lib, 'S3'), (s2lib, 'S2'), (stlib, 'S1'), (cslib, 'TMP')):
    setattr(mod, attr, TMP)

CAPS = (6e5, 8e5, 1e6, 1.5e6, 2e6)
SHIFTS = (0, 1, 2, 3)
SEGMENTS = {'2022': ('2022-01-01', '2022-12-31'), '2024-07~2025-04': ('2024-07-01', '2025-04-30')}
SEGV = tuple(SEGMENTS)
VALID_FIRST_MONTH = pd.Period('2021-12', 'M')              # 驗證期：營收月份 ≥ 2021-12（訊號日 2022-01 起；同 S25／S88～S96）
TARGET = dict(since2019=0.4886246519182116, full=0.3446294016617564, maxdd=-0.3160938700293674, n_closed=381, n_all=389, paths_median=0.4641920784536262)

import s99cells as CE                                       # noqa: E402
CELLS, DAYS = CE.CELLS, CE.DAYS


def load():
    assert cslib.EXRIGHT_FIX is False and cslib.DEADLINE_FIX is False and cslib.INS15_MASK is False
    assert cslib.REDUCE_FIX is False and cslib.E50_SOURCE is None
    cslib.EXRIGHT_FIX = True; cslib.DEADLINE_FIX = True; cslib.INS15_MASK = True
    cslib.REDUCE_FIX = True; cslib.E50_SOURCE = cslib.E50_OFFICIAL_CSV
    P = s19lib.load()
    assert P.exright_fix and P.reduce_fix and P.e50_source == str(cslib.E50_OFFICIAL_CSV)
    s19lib.e50_fix_suspension(P)
    INNOV, _ = s4lib.innov_panel(P)
    U = stlib.universe(P, 'twse') & ~INNOV
    del INNOV
    S, _ = s19lib.rev_std(P)
    avA = s19lib.a10_rows(P, list(S.index))
    for a in ('H', 'L', 'VOL', 'CHG', 'HF', 'VOLF', 'adjH', 'HI240', 'MA60', 'nbar', 'F_HI', 'IDXC', 'listed', 'NEARHI', 'V60', 'liq', 'has'):
        if hasattr(P, a):
            delattr(P, a)
    gc.collect()
    return P, U, dict(S=S, av=avA)


def signal(P, U, RT, shift):
    return s10lib.make_signal_s10(P, U, RT, shift=shift)


def fresh_from(S, first_month=VALID_FIRST_MONTH):
    return CE.fresh_from(S, first_month)


def sim(P, S, U, cap=1e6, cost='strict', block=None):
    """W10_70、嚴成本、空位 0050（原引擎）。"""
    return s10lib.simulate_s10(P, S, U, sat_w=0.7, sizing='W10', lock=False, cost=COSTS[cost], capital=cap, block=block)


def seg_dd(nav, keys=SEGV):
    o = {}
    for k in keys:
        a, b = SEGMENTS[k]
        s = nav.loc[a:b].dropna()
        o[k] = float((s / s.cummax() - 1).min()) if len(s) > 2 else float('nan')
    return o


def apply_cell(P, U, RT, S, cell):
    return CE.apply_cell(P, U, RT, S, cell, cslib.ins15_mask_row)
