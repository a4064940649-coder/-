#!/usr/bin/env python3
"""S65 共用程式庫：新起跑散戶的進場方式（描述性；選股、出場、配置規則一律不動，只改「起跑日、本金分批投入、第一個換股日買不買、何時起跑」）。
鏈：s20lib → s17lib → s10lib（全部 import、不改）；正式開關 cslib.EXRIGHT_FIX／DEADLINE_FIX／INS15_MASK 全開；載入與訊號逐行照
agents/s64_retail_behavior_tax/s64lib.py（＝s56 run.py；現行格逐位重現 deadline_engine_fix ALL＝48.76%／381 筆）。
引擎副本 simulate_s65：inspect 取 s10lib.simulate_s10 原始碼，只做字串替換後 exec（原檔不動）。新參數（全部預設＝與原引擎逐位相同）：
    start_t ：起跑列號（None＝P.t0）；該列收盤時全部是現金，次一交易日起照規則。
    odd_from：'hist'（原行為：2020-10-26 起才可買零股）、'all'（全期可買零股＝現行制度回推）。
    inj     ：{列號 t: 金額}：t 日開盤（賣出之後、底倉拉回與買進之前）追加本金；底倉簿記 B 同步加 (1−sat_w)×金額。
    nobuy_t ：該列號當天不買個股（閒錢照引擎掃進 0050）。
    end_t   ：模擬到這一列為止（省時間；之前各列與跑到底逐位相同）。
中間檔放 scratchpad/s65/cache（結束刪除）。規格見 SPEC.json、PREREG.json。一律 python3 -I -B。"""
import sys, gc, inspect, hashlib
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
AG = HERE.parent
TMP = Path('/tmp/claude-0/-home-user/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/s65')
CACHE = TMP / 'cache'
sys.path.insert(0, str(AG / 's20_persistent_growth'))
import s20lib                                              # noqa: E402
from s20lib import s17lib, s10lib, s9lib, s8lib, s7lib, s6lib, s5lib, s4lib, s3lib, s2lib, stlib, cslib, COSTS   # noqa: E402,F401
for _m, _a in ((s20lib, 'S20'), (s17lib, 'S17'), (s10lib, 'S10'), (s9lib, 'S9'), (s8lib, 'S8'), (s7lib, 'S7'), (s6lib, 'S6'), (s5lib, 'S5'),
               (s4lib, 'S4'), (s3lib, 'S3'), (s2lib, 'S2'), (stlib, 'S1'), (cslib, 'TMP')):
    setattr(_m, _a, CACHE)
assert cslib.DEADLINE_FIX is False and cslib.INS15_MASK is False and cslib.EXRIGHT_FIX is False, '共用引擎預設應全關（本題自己開）'

ACC = dict(since2019=0.4875617247721724, full=0.3439657562340761, maxdd=-0.32119049020243395, n_closed=381)   # deadline_engine_fix ALL
CAP = 1e6
SHIFTS = (0, 1, 2, 3)

_SRC0 = inspect.getsource(s10lib.simulate_s10)
_REPL = [
    ("def simulate_s10(", "def simulate_s65("),
    ("exec_mode='open', trail=None, trail_daily=True, requal=None):",
     "exec_mode='open', trail=None, trail_daily=True, requal=None, start_t=None, odd_from='hist', inj=None, nobuy_t=None, end_t=None):"),
    ("    T, t0 = P.T, P.t0\n",
     "    T, t0 = P.T, (P.t0 if start_t is None else int(start_t))\n"),
    ("    odd_t = int(np.searchsorted(P.IDX.values, np.datetime64(cslib.ODD_DATE), side='left'))\n",
     "    odd_t = int(np.searchsorted(P.IDX.values, np.datetime64(cslib.ODD_DATE), side='left'))\n"
     "    assert odd_from in ('hist', 'all')\n"
     "    if odd_from == 'all':\n"
     "        odd_t = 0\n"),
    ("    for t in range(t0 + 1, T):\n",
     "    for t in range(t0 + 1, T if end_t is None else min(T, int(end_t) + 1)):\n"),
    ("""        if sat_w < 1.0 and t in exec_days:
            rebalance(t)
""",
     """        if inj and t in inj:
            cash += inj[t]; B += (1.0 - sat_w) * inj[t]
        if sat_w < 1.0 and t in exec_days:
            rebalance(t)
"""),
    ("""        cand = np.flatnonzero(sig[t - 1] & univ[t - 1])
""",
     """        cand = np.flatnonzero(sig[t - 1] & univ[t - 1]) if t != nobuy_t else np.zeros(0, int)
"""),
]
_SRC = _SRC0
for _a, _b in _REPL:
    assert _SRC.count(_a) == 1, f'替換點不唯一或不存在：{_a[:40]}'
    _SRC = _SRC.replace(_a, _b)
_G = dict(s10lib.__dict__)
exec(compile(_SRC, '<s65 simulate_s65>', 'exec'), _G)
simulate_s65 = _G['simulate_s65']
SRC_SHA = dict(simulate_s10=hashlib.sha256(_SRC0.encode()).hexdigest(), simulate_s65=hashlib.sha256(_SRC.encode()).hexdigest())


def load():
    """逐行照 s64lib.load（＝s56 run.py）的載入順序。回傳 (P, U, RT)。"""
    CACHE.mkdir(parents=True, exist_ok=True)
    cslib.EXRIGHT_FIX = True
    P = s20lib.load()
    assert P.exright_fix
    assert (P.T, P.N, str(P.IDX[-1].date())) == (2876, 2142, '2026-10-22'), '面板不符'
    INNOV, _ = s4lib.innov_panel(P)
    U = stlib.universe(P, 'twse') & ~INNOV
    del INNOV
    F = s17lib.rev_family(P, defs=['STD'])
    s17lib.e50_fix_suspension(P)
    for a in ('H', 'L', 'VOL', 'CHG', 'HF', 'VOLF', 'adjH', 'HI240', 'MA60', 'nbar', 'F_HI', 'IDXC', 'AO', 'listed', 'NEARHI', 'V60', 'liq'):
        if hasattr(P, a):
            delattr(P, a)
    S0 = F['S']['STD']; months = list(S0.index)
    del F; gc.collect()
    cslib.DEADLINE_FIX = True; cslib.INS15_MASK = True
    AV = cslib.rev_avail_days(P, months, True)
    return P, U, dict(S=S0, av=AV)


def base_signal(P, U, RT, shift):
    sg = s10lib.make_signal_s10(P, U, RT, shift=shift, depth=10, mom_lo=0.10, nsel=10)
    S2, ekw = s10lib.variant_S(sg, 'E0')
    assert ekw == {}
    return S2


KW = dict(sat_w=0.7, sizing='W10', lock=False, cost=COSTS['strict'], cap_pct=0.05)


def sim(P, S, U, capital=CAP, **kw):
    return simulate_s65(P, S, U, capital=capital, **KW, **kw)
