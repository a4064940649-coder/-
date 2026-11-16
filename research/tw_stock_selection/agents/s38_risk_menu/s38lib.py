#!/usr/bin/env python3
"""S38 程式庫：底倉＋空位＝w × 0050＋(1−w) × 定存（年 1%）的包裝，外加衛星比例參數。
改寫自 s30_defensive_base/s30lib.py（只複製 base_swap、capm_beta 與 posthoc.cash_mix 的做法；原檔不動）。
共用引擎 s20lib → s17lib → s10lib 直接 import、不改；cslib.EXRIGHT_FIX＝True。暫存放 scratchpad/s38/。規格見 SPEC.json。
一律 python3 -I -B。"""
import sys
from contextlib import contextmanager
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
AG = HERE.parent
S38 = Path('/tmp/claude-0/-home-user--/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/s38')
sys.path.insert(0, str(AG / 's20_persistent_growth'))
import s20lib                                              # noqa: E402
from s20lib import s17lib, s10lib, s9lib, s8lib, s7lib, s6lib, s5lib, s4lib, s3lib, s2lib, stlib, cslib, COSTS   # noqa: E402,F401
s20lib.S20 = S38; s17lib.S17 = S38; s10lib.S10 = S38; s9lib.S9 = S38; s8lib.S8 = S38; s7lib.S7 = S38; s6lib.S6 = S38; s5lib.S5 = S38; s4lib.S4 = S38
s3lib.S3 = S38; s2lib.S2 = S38; stlib.S1 = S38; cslib.TMP = S38
cslib.EXRIGHT_FIX = True

FEE = s10lib.FEE
TAX_ETF0 = s10lib.TAX_ETF                                  # 0.1%
RF = 0.01


def load():
    S38.mkdir(parents=True, exist_ok=True)
    P = s20lib.load()
    assert P.exright_fix, '必須開 EXRIGHT_FIX'
    return P


def surge(P):
    F = s17lib.rev_family(P, defs=['STD'])
    return dict(S=F['S']['STD'], av=F['av'])


def cash_mix(P, w):
    """w × 0050＋(1−w) × 定存（年 1%），每月最後交易日拉回；0050 進出計手續費＋半個跳動點＋賣出 0.1% 稅（同 S30 posthoc.cash_mix）。"""
    T = P.T; idx = pd.DatetimeIndex(P.IDX)
    me = set(pd.Series(np.arange(T), index=idx).groupby(idx.to_period('M')).max().tolist())
    Dc = np.full(T, np.nan); Do = np.full(T, np.nan)
    t0 = P.t0
    Dc[:t0 + 1] = P.E50[:t0 + 1]; Do[:t0 + 1] = P.E50O[:t0 + 1]
    a = w * P.E50[t0]; b = (1 - w) * P.E50[t0]
    for t in range(t0 + 1, T):
        rf = RF * (idx[t] - idx[t - 1]).days / 365.0
        tot0 = a + b
        Do[t] = Dc[t - 1] * (a * P.E50O[t] / P.E50[t - 1] + b) / tot0
        a *= P.E50[t] / P.E50[t - 1]; b *= 1 + rf
        Dc[t] = Dc[t - 1] * (a + b) / tot0
        if t in me:
            W = a + b; x = abs(a - w * W)
            po = P.E50C_raw[t] if np.isfinite(P.E50C_raw[t]) and P.E50C_raw[t] > 0 else 100.0
            c = x * (FEE + 0.5 * stlib.etf_tick(po) / po + (TAX_ETF0 if a > w * W else 0.0))
            W2 = W - c; a = w * W2; b = (1 - w) * W2; Dc[t] *= W2 / W
        s = Dc[t] / (a + b); a *= s; b *= s
    return dict(Dc=Dc, Do=Do)


def base_arrays(P, w):
    """w＝1 → None（原引擎，不換）。否則回傳 base_swap 用 dict。"""
    if w >= 1.0:
        return None
    D = cash_mix(P, w)
    return dict(E50=np.array(D['Dc'], float), E50O=np.array(D['Do'], float), E50O_raw=np.full(P.T, 100.0), E50C_raw=np.full(P.T, 100.0),
                sb=w * 0.0002, tax=w * TAX_ETF0)


@contextmanager
def base_swap(P, B):
    """同 s30lib.base_swap：B＝None 完全不動；B＝dict 暫時換 0050 陣列與成本，離開時還原（含例外）。sb＝None 只換陣列。"""
    if B is None:
        yield; return
    keys = ('E50', 'E50O', 'E50O_raw', 'E50C_raw')
    old = {k: getattr(P, k) for k in keys}
    old_tick, old_tax = stlib.etf_tick, s10lib.TAX_ETF
    try:
        for k in keys:
            setattr(P, k, B[k])
        if B.get('sb') is not None:
            sb = float(B['sb'])
            stlib.etf_tick = lambda p, _sb=sb: 2.0 * _sb * p
            s10lib.TAX_ETF = float(B['tax'])
        yield
    finally:
        for k in keys:
            setattr(P, k, old[k])
        stlib.etf_tick = old_tick; s10lib.TAX_ETF = old_tax


def sim(P, S1, U, sat_w=0.7, B=None, cap=1e6, cost='strict'):
    S2, ekw = s10lib.variant_S(S1, 'E0')
    with base_swap(P, B):
        return s10lib.simulate_s10(P, S2, U, cost=COSTS[cost], capital=cap, sat_w=sat_w, sizing='W10', lock=False, **ekw)


def capm_beta(nav, bench, a=None, b=None):
    x = nav.loc[a:b].dropna(); y = bench.reindex(x.index)
    rx = x.pct_change().dropna(); ry = y.pct_change().reindex(rx.index)
    ok = np.isfinite(rx.to_numpy()) & np.isfinite(ry.to_numpy()) & (ry.to_numpy() != 0)
    X = ry.to_numpy()[ok]; Y = rx.to_numpy()[ok]
    return float(np.cov(X, Y)[0, 1] / np.var(X, ddof=1))


def sharpe_rf(nav, a=None, b=None, rf=RF):
    r = nav.loc[a:b].dropna().pct_change().dropna() - rf / 252
    return float(r.mean() / r.std(ddof=1) * np.sqrt(252))


def roll12_worst(nav, bench, base='2016-01'):
    """月底口徑（同 S27）：基準月起，滾動 12 個月 策略報酬 − 0050 報酬 的最小值。"""
    s = nav.dropna(); q = bench.reindex(s.index)
    N = s.groupby(s.index.to_period('M')).last(); B = q.groupby(q.index.to_period('M')).last()
    N = N.loc[pd.Period(base, 'M'):].to_numpy(); B = B.loc[pd.Period(base, 'M'):].to_numpy()
    ex = N[12:] / N[:-12] - B[12:] / B[:-12]
    return float(ex.min())
