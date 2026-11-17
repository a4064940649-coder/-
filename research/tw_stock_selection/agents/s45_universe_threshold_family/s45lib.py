#!/usr/bin/env python3
"""S45 共用程式庫：股池門檻家族（60 日均成交值門檻 × 收盤價門檻），其他規則一律不動。
鏈：s20lib → s17lib → s10lib（全部不改）；正式設定 cslib.EXRIGHT_FIX、DEADLINE_FIX、INS15_MASK 全開（必須在 load() 之前設）。
現行股池（讀程式確認）：stlib.load() 裡
    amt = where(P.listed, nan_to_num(P.AMT), nan);  P.V60 = 60 日滾動平均（min_periods=60）
    P.liq = (P.V60 >= 1e7) & (P.C >= 10) & P.has & P.not91
    stlib.universe(P, 'twse') = P.liq & (P.MK == 1)；再 & ~創新板（s4lib.innov_panel）。
本檔只新增 universe(P, adv, px)：同一條公式，只把 1e7 與 10 換成參數；(1e7, 10) 必須和 stlib.universe(P,'twse') & ~INNOV 逐位相同（驗收 A1）。
P.C＝官方日 K 原始收盤（未還原）；P.AMT＝官方成交金額（元）。
中間檔放 scratchpad/s45/cache。規格見 SPEC.json、PREREG.json。一律 python3 -I -B。"""
import sys
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
AG = HERE.parent
TMP = Path('/tmp/claude-0/-home-user--/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/s45')
CACHE = TMP / 'cache'
sys.path.insert(0, str(AG / 's20_persistent_growth'))
import s20lib                                              # noqa: E402
from s20lib import s17lib, s10lib, s9lib, s8lib, s7lib, s6lib, s5lib, s4lib, s3lib, s2lib, stlib, cslib, COSTS   # noqa: E402,F401
for _m, _a in ((s20lib, 'S20'), (s17lib, 'S17'), (s10lib, 'S10'), (s9lib, 'S9'), (s8lib, 'S8'), (s7lib, 'S7'), (s6lib, 'S6'), (s5lib, 'S5'),
               (s4lib, 'S4'), (s3lib, 'S3'), (s2lib, 'S2'), (stlib, 'S1'), (cslib, 'TMP')):
    setattr(_m, _a, CACHE)
assert cslib.DEADLINE_FIX is False and cslib.INS15_MASK is False and cslib.EXRIGHT_FIX is False, '共用引擎預設應全關（本題自己開）'
cslib.EXRIGHT_FIX = True
cslib.DEADLINE_FIX = True
cslib.INS15_MASK = True

ADVS = (3e6, 5e6, 1e7, 2e7, 3e7)                           # 60 日均成交值門檻（元）
PXS = (5.0, 10.0, 20.0)                                    # 收盤價門檻（元）
CUR = 'A1000_P10'
CELL = dict(sat_w=0.7, sizing='W10', lock=False)           # W10_70
CAPS = (6e5, 8e5, 1e6, 1.5e6, 2e6)
SHIFTS = (0, 1, 2, 3)
ACC = dict(since2019=0.4875617247721724, full=0.3439657562340761, maxdd=-0.32119049020243395, n_closed=381)   # deadline_engine_fix ALL


def cid(adv, px):
    return f'A{int(adv / 1e4)}_P{int(px)}'


CELLS = [dict(id=cid(a, p), adv=a, px=p) for a in ADVS for p in PXS]


def load():
    CACHE.mkdir(parents=True, exist_ok=True)
    P = s20lib.load()
    assert P.exright_fix is True
    return P


def universe(P, INNOV, adv, px):
    """同 stlib.load 的 P.liq 公式（只換門檻）＋只上市（MK==1）＋排除創新板。"""
    return (P.V60 >= adv) & (P.C >= px) & P.has & P.not91[None, :] & (P.MK == 1) & ~INNOV


def sim(P, S, U, cap=1e6, cost='strict', cap_pct=0.05):
    S2, ekw = s10lib.variant_S(S, 'E0')
    return s10lib.simulate_s10(P, S2, U, cost=COSTS[cost], capital=cap, cap_pct=cap_pct, **CELL, **ekw)
