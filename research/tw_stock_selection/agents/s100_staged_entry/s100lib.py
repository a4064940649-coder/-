#!/usr/bin/env python3
"""S100 共用程式庫：W2-A「撐過第二份營收才買滿」（分批進場）。
引擎：正式管線（agents/official_nav/run_official.py 的 E0｜W10_70；EXRIGHT_FIX、DEADLINE_FIX、INS15_MASK、REDUCE_FIX、0050 官方含息 五個開關全開）。
現行＝s10lib.simulate_s10 原樣呼叫；變體＝s100sim.simulate_staged（simulate_s10 的複製＋分批掛鉤；stage=None 時與原引擎逐筆相同，step0 驗收）。
訊號（名單、出場 X）完全不動。不修改任何既有檔。暫存：scratchpad/s100/cache（面板快取；用完刪）。一律 python3 -I -B。"""
import sys, gc
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
AG = HERE.parent
TMP = Path('/tmp/claude-0/-home-user/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/s100/cache')
TMP.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(AG / 's19_xq_threshold'))
import s19lib                                              # noqa: E402
from s19lib import s10lib, s9lib, s8lib, s7lib, s6lib, s5lib, s4lib, s3lib, s2lib, stlib, cslib, COSTS   # noqa: E402
for mod, attr in ((s19lib, 'S19'), (s10lib, 'S10'), (s9lib, 'S9'), (s8lib, 'S8'), (s7lib, 'S7'), (s6lib, 'S6'), (s5lib, 'S5'),
                  (s4lib, 'S4'), (s3lib, 'S3'), (s2lib, 'S2'), (stlib, 'S1'), (cslib, 'TMP')):
    setattr(mod, attr, TMP)
import s100sim                                             # noqa: E402

CAPS = (6e5, 8e5, 1e6, 1.5e6, 2e6)
SHIFTS = (0, 1, 2, 3)
SEGMENTS = {'2022': ('2022-01-01', '2022-12-31'), '2024-07~2025-04': ('2024-07-01', '2025-04-30')}
SEGV = tuple(SEGMENTS)
VALID_FIRST_MONTH = pd.Period('2021-12', 'M')              # 驗證期：營收月份 ≥ 2021-12（訊號日 2022-01 起；同 S25／S88～S99）
TARGET = dict(since2019=0.4886246519182116, full=0.3446294016617564, maxdd=-0.3160938700293674, n_closed=381, n_all=389, paths_median=0.4641920784536262)

from s100util import CELLS, scaled_current, topup_K, first_check_stats, conc_stats   # noqa: E402,F401


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
    """只保留營收月份 ≥ first_month 的進場訊號；出場 X 不動。"""
    S2 = dict(S); sig = S['sig'].copy(); rank = S['rank'].copy()
    for x in S['info']:
        if pd.Period(x['month'], 'M') < first_month:
            sig[x['k'], :] = False; rank[x['k'], :] = np.nan
    S2['sig'] = sig; S2['rank'] = rank
    return S2


def sim(P, S, U, cap=1e6, cost='strict', block=None, cell=None, rand=None, simulate=None):
    """W10_70、嚴成本、空位 0050。cell=None／'CUR' → 原引擎 s10lib.simulate_s10（simulate 可換成連續引擎用的同一函式）；
    cell＝CELLS 的鍵或 dict(init, final, check) → s100sim.simulate_staged。rand＝dict(K, rng) → 隨機對照。"""
    kw = dict(sat_w=0.7, sizing='W10', lock=False, cost=COSTS[cost], capital=cap, block=block)
    if cell is None or cell == 'CUR':
        return (simulate or s10lib.simulate_s10)(P, S, U, **kw)
    stg = dict(CELLS[cell]) if isinstance(cell, str) else dict(cell)
    if rand is not None:
        stg['rand'] = rand
    return s100sim.simulate_staged(P, S, U, stage=stg, **kw)


def seg_dd(nav, keys=SEGV):
    o = {}
    for k in keys:
        a, b = SEGMENTS[k]
        s = nav.loc[a:b].dropna()
        o[k] = float((s / s.cummax() - 1).min()) if len(s) > 2 else float('nan')
    return o
