#!/usr/bin/env python3
"""S55 共用程式庫：把凍結的現行規則（F1 選股、E0 出場、A10 訊號日、W10_70 配置、嚴成本、EXRIGHT_FIX／DEADLINE_FIX／INS15_MASK 全開）
原封不動地跑在 2015-07 以前的面板上。引擎一律 import 既有程式（s47lib → s20lib → s17lib → s10lib → … → cslib），檔案不改；
本檔只在執行期改「資料位置與樣本窗」這幾個模組變數（不是策略參數）：
  - 各層快取目錄（cslib.TMP、stlib.S1、s2lib.S2 … s47lib.CACHE）→ 本題 scratchpad（面板 npz、0050 開高低收、創新板快取先放好，所以不會呼叫 git）；
  - 'oos'：stlib.START＝2008-01-02（回測起點）、s17lib.FIRST_MONTH＝2008-01（σ24 最早可算的訊號月）；
  - 'is' ：照現行（START 2015-07-01、FIRST_MONTH 2016-01），面板＝官方快取（重現 34.40%／381 筆用）。
一律 python3 -I -B。"""
import sys, shutil, hashlib
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
AG = HERE.parent
SCR = Path('/tmp/claude-0/-home-user/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/s55')
OFFCACHE = Path('/tmp/claude-0/-home-user--/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/eng/cache')
OFF_SHA = '434f64222b8fd1f10e1e25c9ac4553e7e6f6d11305896a3ceabb3b212650dfab'
sys.path.insert(0, str(AG / 's47_allocation_family'))
import s47lib as L                                         # noqa: E402  （import 時已把三個正式開關設成 True）
from s47lib import s20lib, s17lib, s10lib, s9lib, s8lib, s7lib, s6lib, s5lib, s4lib, s3lib, s2lib, stlib, cslib   # noqa: E402

OOS_START = pd.Timestamp('2008-01-02')
OOS_FIRST_MONTH = pd.Period('2008-01', 'M')
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()


def setup(which):
    cache = SCR / ('cache_is' if which == 'is' else 'cache_pre')
    cache.mkdir(parents=True, exist_ok=True)
    if which == 'is':
        assert sha(OFFCACHE / 'panel_cs.npz') == OFF_SHA, '官方面板快取不是預期版本'
        for f in ('panel_cs.npz', 'e50_ohlc.csv', 'innov.npz'):
            if not (cache / f).exists():
                shutil.copy2(OFFCACHE / f, cache / f)
    else:
        for f in ('panel_cs.npz', 'e50_ohlc.csv', 'innov.npz'):
            assert (cache / f).exists(), '先跑 build_pre_panel.py'
        stlib.START = OOS_START
        s17lib.FIRST_MONTH = OOS_FIRST_MONTH
    for m, a in ((s20lib, 'S20'), (s17lib, 'S17'), (s10lib, 'S10'), (s9lib, 'S9'), (s8lib, 'S8'), (s7lib, 'S7'), (s6lib, 'S6'), (s5lib, 'S5'),
                 (s4lib, 'S4'), (s3lib, 'S3'), (s2lib, 'S2'), (stlib, 'S1'), (cslib, 'TMP')):
        setattr(m, a, cache)
    L.CACHE = cache
    assert cslib.EXRIGHT_FIX and cslib.DEADLINE_FIX and cslib.INS15_MASK
    return cache


def build(which):
    """載入面板、股池、SURGE（STD）、訊號（位移 0）。回傳 P, U, RT, sg。"""
    setup(which)
    P = L.load()
    INNOV, _ = s4lib.innov_panel(P)
    U = stlib.universe(P, 'twse') & ~INNOV
    F = s17lib.rev_family(P, defs=['STD'])
    RT = dict(S=F['S']['STD'], av=F['av'])
    fix = s17lib.e50_fix_suspension(P)
    sg = s10lib.make_signal_s10(P, U, RT, shift=0)
    return P, U, RT, sg, fix


def monthly(nav, e50):
    """日淨值 → 日曆月最後一個交易日；第一個月以起點（nav 第一個有值日）為基底。回傳 DataFrame(strategy, b0050) 月報酬（PeriodIndex）。"""
    nav = nav.dropna()
    b = e50.reindex(nav.index)
    me = nav.groupby(nav.index.to_period('M')).tail(1)
    s = pd.concat([nav.iloc[:1], me]); bb = pd.concat([b.iloc[:1], b.reindex(me.index)])
    r = pd.DataFrame(dict(strategy=s.pct_change().iloc[1:].to_numpy(), b0050=bb.pct_change().iloc[1:].to_numpy()), index=me.index.to_period('M'))
    return r
