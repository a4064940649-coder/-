#!/usr/bin/env python3
"""R55 共用：把共用引擎（s47lib → s20lib → s17lib → s10lib → … → stlib → cslib，全部不改）的快取目錄
在執行期改到 R55 的 scratchpad，並提供月度統計（依 S55 SPEC.json 的 monthly／stats 文字自寫）。
不讀 S55 的 s55lib.py／prep.py／run.py／posthoc.py／build_pre_panel.py。一律 python3 -I -B；不呼叫 git（快取先放好）。"""
import sys, math
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
AG = HERE.parent
TMP = Path('/tmp/claude-0/-home-user/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/r55')
sys.path.insert(0, str(AG / 's47_allocation_family'))
import s47lib as L                                          # noqa: E402  （import 時設 EXRIGHT_FIX／DEADLINE_FIX／INS15_MASK＝True）
from s47lib import s20lib, s17lib, s10lib, s9lib, s8lib, s7lib, s6lib, s5lib, s4lib, s3lib, s2lib, stlib, cslib   # noqa: E402


def point_cache(cache):
    """各層快取 → cache（需已放好 panel_cs.npz、e50_ohlc.csv、innov.npz，否則引擎會去呼叫 git）。"""
    cache = Path(cache)
    for f in ('panel_cs.npz', 'e50_ohlc.csv', 'innov.npz'):
        assert (cache / f).exists(), f'快取缺 {f}（不准呼叫 git）'
    for m, a in ((L, 'CACHE'), (s20lib, 'S20'), (s17lib, 'S17'), (s10lib, 'S10'), (s9lib, 'S9'), (s8lib, 'S8'), (s7lib, 'S7'), (s6lib, 'S6'),
                 (s5lib, 'S5'), (s4lib, 'S4'), (s3lib, 'S3'), (s2lib, 'S2'), (stlib, 'S1'), (cslib, 'TMP')):
        setattr(m, a, cache)


def run_current(start=None, first_month=None):
    """現行格：股池＝上市 & ~創新板；SURGE＝rev_family STD；F1＝make_signal_s10(shift=0)；E0；W10_70；100 萬；strict。"""
    if start is not None:
        stlib.START = pd.Timestamp(start)
    if first_month is not None:
        s17lib.FIRST_MONTH = pd.Period(first_month, 'M')
    assert cslib.EXRIGHT_FIX and cslib.DEADLINE_FIX and cslib.INS15_MASK
    P = s20lib.load()
    INNOV, _ = s4lib.innov_panel(P)
    U = stlib.universe(P, 'twse') & ~INNOV
    F = s17lib.rev_family(P, defs=['STD'])
    RT = dict(S=F['S']['STD'], av=F['av'])
    fix = s17lib.e50_fix_suspension(P)
    sg = s10lib.make_signal_s10(P, U, RT, shift=0)
    nav, lg, st = L.sim(P, sg, U, cap=1e6)
    return P, U, F, sg, nav, lg, st, fix


# ---------------------------------------------------------------- 月度統計（SPEC：月底＝每個日曆月最後交易日；第一個月以起日為基底）
def month_points(s):
    s = s.dropna()
    me = s.groupby(s.index.to_period('M')).tail(1)
    pts = pd.concat([s.iloc[:1], me])
    return pts


def monthly_returns(nav, bench):
    q = bench.reindex(nav.index)
    ps, pb = month_points(nav), month_points(q)
    rs = ps.pct_change().dropna(); rb = pb.pct_change().dropna()
    rs.index = rs.index.to_period('M'); rb.index = rb.index.to_period('M')
    return rs, rb


def nw_t(x, lag=6):
    x = np.asarray(x, float); T = len(x); m = x.mean(); e = x - m
    lr = (e * e).sum() / T
    for i in range(1, lag + 1):
        lr += 2 * (1 - i / (lag + 1)) * (e[i:] * e[:-i]).sum() / T
    return float(m / math.sqrt(lr / T))


def stats_block(rs, rb):
    ex = (rs - rb).to_numpy(); n = len(ex)
    m, sd = ex.mean(), ex.std(ddof=1)
    gs = float(np.prod(1 + rs.to_numpy()) ** (12 / n) - 1); gb = float(np.prod(1 + rb.to_numpy()) ** (12 / n) - 1)
    cov = np.cov(rs.to_numpy(), rb.to_numpy(), ddof=1)
    return dict(months=int(n), first=str(rs.index[0]), last=str(rs.index[-1]), cagr_strat=gs, cagr_0050=gb, geo_diff_pp=(gs - gb) * 100,
                ex_ann_pp=float(m * 12 * 100), te_ann_pp=float(sd * math.sqrt(12) * 100), IR=float(m / sd * math.sqrt(12)),
                t_iid=float(m / sd * math.sqrt(n)), t_nw6=nw_t(ex, 6), beta=float(cov[0, 1] / cov[1, 1]))


def maxdd(s):
    s = s.dropna()
    return float((s / s.cummax() - 1).min())


def yearly_excess(rs, rb):
    ys = (1 + rs).groupby(rs.index.year).prod() - 1; yb = (1 + rb).groupby(rb.index.year).prod() - 1
    return {int(y): dict(strat=float(ys[y]), e0050=float(yb[y]), ex_pp=float((ys[y] - yb[y]) * 100)) for y in ys.index}
