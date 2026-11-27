#!/usr/bin/env python3
"""S102 共用程式庫：U1「上月名單沿用遞補」、U2「破線賣出後站回再買回」。
引擎：正式管線（E0｜W10_70；EXRIGHT_FIX、DEADLINE_FIX、INS15_MASK、REDUCE_FIX、0050 官方含息 五個開關全開；載入方式逐行同 S100 s100lib.load）。
現行＝s10lib.simulate_s10 原樣呼叫＋s10lib.make_signal_s10 原樣；
U1＝make_signal_u(fill=…)（make_signal_s10 的 E0 部分逐行複製＋遞補；fill=None 時 sig／rank／X／sigrows 與原函式相同，step0 驗收 A2）＋原引擎；
U2＝原訊號（另附每個訊號列的 SURGE 分數與股池）＋s102sim.simulate_u(rebuy=…)（rebuy=None 時與原引擎逐筆相同，step0 驗收 A1）。
不修改任何既有檔。暫存：scratchpad/s102/cache（用完刪）。一律 python3 -I -B。"""
import sys, gc
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
AG = HERE.parent
TMP = Path('/tmp/claude-0/-home-user/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/s102/cache')
TMP.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(AG / 's19_xq_threshold'))
import s19lib                                              # noqa: E402
from s19lib import s10lib, s9lib, s8lib, s7lib, s6lib, s5lib, s4lib, s3lib, s2lib, stlib, cslib, COSTS   # noqa: E402
for mod, attr in ((s19lib, 'S19'), (s10lib, 'S10'), (s9lib, 'S9'), (s8lib, 'S8'), (s7lib, 'S7'), (s6lib, 'S6'), (s5lib, 'S5'),
                  (s4lib, 'S4'), (s3lib, 'S3'), (s2lib, 'S2'), (stlib, 'S1'), (cslib, 'TMP')):
    setattr(mod, attr, TMP)
import s102core as CORE                                    # noqa: E402
from s102core import CELLS, U1_CELLS, U2_CELLS, make_signal_u, ret5, sim_core, roll_sr_neg, top5_share, scaled_current   # noqa: E402,F401

CAPS = (6e5, 8e5, 1e6, 1.5e6, 2e6)
SHIFTS = (0, 1, 2, 3)
SEGMENTS = {'2022': ('2022-01-01', '2022-12-31'), '2024-07~2025-04': ('2024-07-01', '2025-04-30')}
SEGV = tuple(SEGMENTS)
VALID_FIRST_MONTH = pd.Period('2021-12', 'M')
TARGET = dict(since2019=0.4886246519182116, full=0.3446294016617564, maxdd=-0.3160938700293674, n_closed=381, n_all=389, paths_median=0.4641920784536262)


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
    P.RET5 = ret5(P)
    return P, U, dict(S=S, av=avA)


def signal(P, U, RT, shift, cell='CUR'):
    """CUR／U2 格 → make_signal_u(fill=None)（與 make_signal_s10 的 E0 相同，另附分數）；U1 格 → 遞補版。"""
    return make_signal_u(P, U, RT, shift=shift, fill=CELLS[cell].get('fill') if cell != 'CUR' else None)


def fresh_from(S, first_month=VALID_FIRST_MONTH):
    """只保留營收月份 ≥ first_month 的進場訊號；出場 X 不動（同 S100）。"""
    S2 = dict(S); sig = S['sig'].copy(); rank = S['rank'].copy()
    for x in S['info']:
        if pd.Period(x['month'], 'M') < first_month:
            sig[x['k'], :] = False; rank[x['k'], :] = np.nan
    S2['sig'] = sig; S2['rank'] = rank
    return S2


def sim(P, S, U, cap=1e6, cost='strict', block=None, cell='CUR'):
    kw = dict(sat_w=0.7, sizing='W10', lock=False, cost=COSTS[cost], capital=cap, block=block)
    return sim_core(P, S, U, cell, kw, s10lib.simulate_s10)


def seg_dd(nav, keys=SEGV):
    o = {}
    for k in keys:
        a, b = SEGMENTS[k]
        s = nav.loc[a:b].dropna()
        o[k] = float((s / s.cummax() - 1).min()) if len(s) > 2 else float('nan')
    return o
