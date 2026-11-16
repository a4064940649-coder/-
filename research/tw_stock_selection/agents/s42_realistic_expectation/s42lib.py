#!/usr/bin/env python3
"""S42 共用：把 S39 FPM（首次公布營收）＋S40 ALL（期限例外＋看不到就缺值）＋執行情境（滑價、晚下單）疊加的最小包裝。
不改任何共用引擎或既有研究：只 import s39lib／s40lib 的函式（唯讀），再把所有 lib 的快取路徑改到 scratchpad/s42/cache。
一律 python3 -I -B。"""
import sys
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
AG = HERE.parent
TMP = Path('/tmp/claude-0/-home-user--/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/s42')
CACHE = TMP / 'cache'
sys.path.insert(0, str(AG / 's39_first_print_revenue'))
sys.path.insert(0, str(AG / 's40_revenue_deadline_pit'))
import s39lib                                              # noqa: E402  （函式：surge_variant、load_fp）
import s40lib                                              # noqa: E402  （函式：av_fix、mask_S、常數 EXT_ALL、LATE_PM、TZ）
from s20lib import s17lib, s10lib, s9lib, s8lib, s7lib, s6lib, s5lib, s4lib, s3lib, s2lib, stlib, cslib, COSTS   # noqa: E402,F401
import s20lib                                              # noqa: E402
for mod, attr in ((s20lib, 'S20'), (s17lib, 'S17'), (s10lib, 'S10'), (s9lib, 'S9'), (s8lib, 'S8'), (s7lib, 'S7'), (s6lib, 'S6'),
                  (s5lib, 'S5'), (s4lib, 'S4'), (s3lib, 'S3'), (s2lib, 'S2'), (stlib, 'S1'), (cslib, 'TMP')):
    setattr(mod, attr, CACHE)
s39lib.S39 = CACHE; s40lib.CACHE = CACHE; s40lib.TMP = TMP
cslib.EXRIGHT_FIX = True

VIS_CSV = AG / 's40_revenue_deadline_pit' / 'data' / 'visibility_cells.csv'
CAL_CSV = AG / 's40_revenue_deadline_pit' / 'data' / 'deadline_calendar.csv'
FAM_D = (10, 11, 12, 13, 15)                                # S9 A10～A15（營收資訊相同，只差訊號日）


def load():
    CACHE.mkdir(parents=True, exist_ok=True)
    P = s20lib.load()
    assert P.exright_fix, '未開 EXRIGHT_FIX'
    return P


def cost(xslip):
    return dict(cmult=1.0, xslip=float(xslip), oddx=0.0)


def av_family(P, months, D):
    """S9 A_D：d*＝營收月 m 的次月第 D 日（含）起第一個交易日；訊號列＝d*＋1。D＝10 時＝cslib.rev_avail_days。"""
    out = {}
    for m in months:
        d = (m + 1).to_timestamp() + pd.Timedelta(days=D - 1)
        out[m] = int(np.searchsorted(P.IDX.values, np.datetime64(d), side='left')) + 1
    return out


def av_combine(av_a, av_b):
    return {m: max(av_a[m], av_b[m]) for m in av_a}


class Late:
    """S40 run.py late_cells 的一般化：看不到的判斷用『該情境實際訊號列 av[m]＋位移』。"""

    def __init__(self, P):
        self.P = P
        self.C = pd.read_csv(VIS_CSV, dtype={'code': str})
        self.cal = pd.read_csv(CAL_CSV).set_index('rev_month')

    def cells(self, cats, av, shift):
        P, cal = self.P, self.cal
        d = self.C[self.C['cat'].isin(cats)]
        out = []
        for r in d.itertuples():
            m = pd.Period(r.rev_month, 'M')
            k = av[m] + shift
            if k >= P.T:
                continue
            sd = P.IDX[k]
            send = pd.Timestamp(sd.date()).tz_localize(s40lib.TZ) + pd.Timedelta(hours=23, minutes=59, seconds=59)
            if isinstance(r.T, str):
                if pd.Timestamp(r.T) <= send:
                    continue
            elif r.ins15 and isinstance(cal.loc[r.rev_month].get('deadline_ins15'), str):
                if pd.Timestamp(cal.loc[r.rev_month, 'deadline_ins15'][:10]) <= sd:
                    continue
            out.append((m, r.code))
        return out


def haircut(nav, bench, h, m0='2016-02', m1='2026-10'):
    """S27 HALF／ZERO 的歷史類比：d̄＝月底口徑 mean(r策略 − r0050)（m0～m1）；每個交易日報酬減 h×d̄×(月數÷交易日數)。回傳 (新淨值, d̄)。"""
    me = nav.resample('ME').last(); be = bench.reindex(nav.index).resample('ME').last()
    rs, rb = me.pct_change(), be.pct_change()
    sel = (rs.index >= pd.Period(m0, 'M').to_timestamp()) & (rs.index <= pd.Period(m1, 'M').to_timestamp(how='end'))
    dbar = float((rs[sel] - rb[sel]).mean())
    a = nav.index[nav.index > me.index[np.flatnonzero(sel)[0] - 1]]
    nd = int(((nav.index > me.index[np.flatnonzero(sel)[0] - 1]) & (nav.index <= me.index[np.flatnonzero(sel)[-1]])).sum())
    c = h * dbar * int(sel.sum()) / nd
    r = nav.pct_change().fillna(0.0) - c
    r.iloc[0] = 0.0
    del a
    return nav.iloc[0] * (1 + r).cumprod(), dbar
