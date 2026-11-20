#!/usr/bin/env python3
"""R75 共用：獨立重現 S75（不讀 S75 的 s75lib.py／run.py／build_panel.py）。
重用 agents/r55_replicate_s55/r55lib.py（快取改道、月度統計）與 agents/s71_ifrs_bridge_oos/s71lib.py（B2 規則 rev_rule）；共用引擎一律不改。
一律 python3 -I -B；不呼叫 git（快取先放好）。"""
import sys
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
AG = HERE.parent
RS = AG.parent
TMP = Path('/tmp/claude-0/-home-user/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/r75')
sys.path.insert(0, str(AG / 's71_ifrs_bridge_oos'))
import s71lib                                               # noqa: E402
from s71lib import R, L, s20lib, s17lib, s10lib, s4lib, stlib, cslib   # noqa: E402
R.TMP = TMP; s71lib.TMP = TMP


def setup(cache, start, first_month):
    R.point_cache(TMP / cache)
    stlib.START = pd.Timestamp(start)
    s17lib.FIRST_MONTH = pd.Period(first_month, 'M')
    assert cslib.EXRIGHT_FIX and cslib.DEADLINE_FIX and cslib.INS15_MASK


def load_all():
    P = s20lib.load()
    INNOV, _ = s4lib.innov_panel(P)
    U = stlib.universe(P, 'twse') & ~INNOV
    fix = s17lib.e50_fix_suspension(P)
    return P, U, fix


def run_rule(P, U, rule, first_month, cap=1e6, shift=0):
    F = s71lib.rev_rule(P, rule, first_month)
    sg = s10lib.make_signal_s10(P, U, dict(S=F['S']['STD'], av=F['av']), shift=shift)
    nav, lg, st = L.sim(P, sg, U, cap=cap)
    return F, sg, nav, lg, st


def dd_info(s):
    s = s.dropna(); dd = s / s.cummax() - 1; tr = dd.idxmin(); pk = s.loc[:tr].idxmax()
    rec = s.loc[tr:][s.loc[tr:] >= s.loc[pk]]
    return dict(maxdd=float(dd.min()), peak=str(pk.date()), trough=str(tr.date()), recover=str(rec.index[0].date()) if len(rec) else None)


def yearly(nav):
    yr = nav.groupby(nav.index.year).last(); prev = nav.iloc[0]; out = {}
    for y in yr.index:
        out[int(y)] = float(yr.loc[y] / prev - 1); prev = yr.loc[y]
    return out
