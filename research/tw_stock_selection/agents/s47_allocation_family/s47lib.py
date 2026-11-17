#!/usr/bin/env python3
"""S47 共用程式庫：資金配置家族（衛星比例 × 每檔份數，另加空位處理），選股、出場、股池一律不動。
鏈：s20lib → s17lib → s10lib（全部不改）；正式設定 cslib.EXRIGHT_FIX、DEADLINE_FIX、INS15_MASK 全開（必須在 load() 之前設）。
現行（讀程式確認）：s10lib.simulate_s10(sat_w=0.7, sizing='W10', lock=False, band=0.10, slots=10, sweep_min=2 萬, min_frac=0.5, cap_pct=0.05)
    底倉簿記 B＝(1−sat_w)×淨值（±10 個百分點帶狀拉回）；每檔目標＝(開盤估淨值 − B) ÷ 10；最多 10 檔；
    e50_net：現金 ≥ 2 萬就整筆掃進 0050（底倉與空位共用一個 0050 部位 v50）＝「空位買 0050」。
本檔新增 simulate_s47：用 inspect 取 s10lib.simulate_s10 原始碼，只做三處字串替換後 exec（原檔不動）：
    (1) 新參數 ndiv：每檔目標的除數（None＝照 sizing；8／12＝衛星 ÷ 8／÷ 12）；檔數上限 slots＝10 不變；
    (2) 新參數 idle：'e50'（原行為）或 'cash'（空位留現金）：0050 部位只維持在底倉簿記 B（差 ≥ 2 萬才補買／賣回），其餘閒置錢留現金（利率 0）；
    (3) 函式名改 simulate_s47。
    ndiv=None、idle='e50' 時與 simulate_s10 逐位相同；ndiv=10 時與 sizing='W10' 逐位相同（驗收 A3）。
中間檔放 scratchpad/s47/cache。規格見 SPEC.json、PREREG.json。一律 python3 -I -B。"""
import sys, inspect, hashlib
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
AG = HERE.parent
TMP = Path('/tmp/claude-0/-home-user/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/s47')
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

SATS = (0.5, 0.6, 0.7, 0.8, 1.0)                           # 衛星比例
DIVS = (8, 10, 12)                                         # 每檔＝衛星 ÷ d
CUR = 'S70_D10'
CASH = 'S70_D10_CASH'
CAPS = (6e5, 8e5, 1e6, 1.5e6, 2e6)
SHIFTS = (0, 1, 2, 3)
ACC = dict(since2019=0.4875617247721724, full=0.3439657562340761, maxdd=-0.32119049020243395, n_closed=381)   # deadline_engine_fix ALL


def cid(s, d, idle='e50'):
    return f'S{int(round(s * 100))}_D{d}' + ('_CASH' if idle == 'cash' else '')


CELLS = [dict(id=cid(s, d), sat_w=s, div=d, idle='e50') for s in SATS for d in DIVS]          # 15 格（主家族）
CASH_CELL = dict(id=CASH, sat_w=0.7, div=10, idle='cash')                                   # 空位處理對照（只比 70%×÷10）

# ---- 引擎副本（原始碼字串替換，原檔不動）
_SRC0 = inspect.getsource(s10lib.simulate_s10)
_REPL = [
    ("def simulate_s10(", "def simulate_s47("),
    ("exec_mode='open', trail=None, trail_daily=True, requal=None):",
     "exec_mode='open', trail=None, trail_daily=True, requal=None, ndiv=None, idle='e50'):"),
    ("div = {'W10': 10.0, 'W7': 7.0, 'WN': float(max(ntgt.get(k, 0), 5))}[sizing]",
     "div = float(ndiv) if ndiv is not None else {'W10': 10.0, 'W7': 7.0, 'WN': float(max(ntgt.get(k, 0), 5))}[sizing]"),
    ("""        elif cash >= sweep_min:
            f = max(minfee, cash * fee)""",
     """        elif idle == 'cash':
            gap = B - v50
            if gap >= sweep_min and cash >= sweep_min:
                x = min(cash, gap)
                f = max(minfee, x * fee); val = (x - f) / (1 + sl50)
                st['e50_cost'] += x - val; st['e50_orders'] += 1
                v50 += val; cash -= x
            elif -gap >= sweep_min:
                y = -gap
                g = y * (1 - sl50); f = max(minfee, g * fee); net = g - f - g * TAX_ETF * cm
                v50 -= y; cash += net; st['e50_cost'] += y - net; st['e50_orders'] += 1
        elif cash >= sweep_min:
            f = max(minfee, cash * fee)"""),
]
_SRC = _SRC0
for _a, _b in _REPL:
    assert _SRC.count(_a) == 1, f'替換點不唯一或不存在：{_a[:40]}'
    _SRC = _SRC.replace(_a, _b)
_G = dict(s10lib.__dict__)
exec(compile(_SRC, '<s47 simulate_s47>', 'exec'), _G)
simulate_s47 = _G['simulate_s47']
SRC_SHA = dict(simulate_s10=hashlib.sha256(_SRC0.encode()).hexdigest(), simulate_s47=hashlib.sha256(_SRC.encode()).hexdigest())


def load():
    CACHE.mkdir(parents=True, exist_ok=True)
    P = s20lib.load()
    assert P.exright_fix is True
    return P


def sim(P, S, U, sat_w=0.7, div=10, idle='e50', cap=1e6, cost='strict', force_copy=False):
    """÷10 且空位 0050 → 原引擎 simulate_s10(sizing='W10')；其餘 → simulate_s47。"""
    S2, ekw = s10lib.variant_S(S, 'E0')
    kw = dict(cost=COSTS[cost], capital=cap, cap_pct=0.05, sat_w=sat_w, sizing='W10', lock=False, **ekw)
    if div == 10 and idle == 'e50' and not force_copy:
        return s10lib.simulate_s10(P, S2, U, **kw)
    return simulate_s47(P, S2, U, ndiv=div, idle=idle, **kw)


def longest_lag(nav, bench):
    """相對 0050 最長落後期：rel＝淨值 ÷ 0050（同起點），rel 低於自己先前最高點的最長連續交易日數（期末未收復也算）。回傳 (天數, 起日, 止日)。"""
    rel = (nav / bench.reindex(nav.index)).dropna()
    uw = (rel < rel.cummax() * (1 - 1e-12)).to_numpy()
    best = (0, None, None); i = 0; n = len(uw)
    while i < n:
        if uw[i]:
            j = i
            while j + 1 < n and uw[j + 1]:
                j += 1
            if j - i + 1 > best[0]:
                best = (j - i + 1, str(rel.index[i - 1 if i > 0 else i].date()), str(rel.index[j].date()))
            i = j + 1
        else:
            i += 1
    return best
