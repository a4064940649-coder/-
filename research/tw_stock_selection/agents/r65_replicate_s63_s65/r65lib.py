#!/usr/bin/env python3
"""R65 獨立重現用程式庫（沒有讀 s63lib.py／s65lib.py／兩題的 run.py）。
鏈：s47lib → s20lib → s17lib → s10lib（全部 import、不改）；三個正式開關由 s47lib 匯入時全開。快取改放 scratchpad/r65/cache。
引擎副本 simulate_r65：inspect 取 s10lib.simulate_s10 原始碼做字串替換後 exec（原檔不動）。新參數全預設時應與原引擎逐位相同：
  start_t：新起跑列（該列收盤手上 capital 現金，次一列開盤開始動作）
  odd_all：True＝全期可買零股（現行制度回推）
  inj：{列號: 金額}，該列開盤前加入現金（分批投入）；底倉簿記 B 同步加 (1−sat_w)×金額（inj_base=True）或不加（False，敏感度）
一律 python3 -I -B。"""
import sys, inspect
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
AG = HERE.parent
TMP = Path('/tmp/claude-0/-home-user/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/r65')
CACHE = TMP / 'cache'
sys.path.insert(0, str(AG / 's47_allocation_family'))
import s47lib                                              # noqa: E402
from s47lib import s20lib, s17lib, s10lib, s9lib, s8lib, s7lib, s6lib, s5lib, s4lib, s3lib, s2lib, stlib, cslib, COSTS, simulate_s47   # noqa: E402,F401
for _m, _a in ((s20lib, 'S20'), (s17lib, 'S17'), (s10lib, 'S10'), (s9lib, 'S9'), (s8lib, 'S8'), (s7lib, 'S7'), (s6lib, 'S6'), (s5lib, 'S5'),
               (s4lib, 'S4'), (s3lib, 'S3'), (s2lib, 'S2'), (stlib, 'S1'), (cslib, 'TMP')):
    setattr(_m, _a, CACHE)
assert cslib.EXRIGHT_FIX and cslib.DEADLINE_FIX and cslib.INS15_MASK

_SRC0 = inspect.getsource(s10lib.simulate_s10)
_REPL = [
    ("def simulate_s10(", "def simulate_r65("),
    ("exec_mode='open', trail=None, trail_daily=True, requal=None):",
     "exec_mode='open', trail=None, trail_daily=True, requal=None, start_t=None, odd_all=False, inj=None, inj_base=True):"),
    ("    T, t0 = P.T, P.t0\n", "    T = P.T; t0 = P.t0 if start_t is None else int(start_t); inj = inj or {}\n"),
    ("    cash = float(capital); v50 = 0.0; pos = {}\n",
     "    if odd_all:\n        odd_t = 0\n    cash = float(capital); v50 = 0.0; pos = {}\n"),
    ("    for t in range(t0 + 1, T):\n        if CLOSE:",
     "    for t in range(t0 + 1, T):\n        if t in inj:\n            cash += inj[t]\n            if inj_base:\n                B += (1.0 - sat_w) * inj[t]\n        if CLOSE:"),
]
_SRC = _SRC0
for _a, _b in _REPL:
    assert _SRC.count(_a) == 1, f'替換點不唯一或不存在：{_a[:40]}'
    _SRC = _SRC.replace(_a, _b)
_G = dict(s10lib.__dict__)
exec(compile(_SRC, '<r65 simulate_r65>', 'exec'), _G)
simulate_r65 = _G['simulate_r65']


def load():
    CACHE.mkdir(parents=True, exist_ok=True)
    P = s20lib.load()
    assert P.exright_fix is True
    INNOV, _ = s4lib.innov_panel(P)
    U = stlib.universe(P, 'twse') & ~INNOV
    F = s17lib.rev_family(P, defs=['STD'])
    RT = dict(S=F['S']['STD'], av=F['av'])
    s17lib.e50_fix_suspension(P)
    return P, U, RT


KW = dict(sizing='W10', lock=False, cap_pct=0.05)


def empty_S(P, S):
    """清空訊號（只買 0050 用）。"""
    S2 = dict(S)
    S2['sig'] = np.zeros_like(S['sig']); S2['X'] = np.zeros_like(S['X']); S2['info'] = []; S2['sigrows'] = np.zeros(P.T, bool); S2['RQ'] = {}
    return S2


def quarterly_S(P, S, keep_months):
    """D 季換股：只保留 keep_months（營收月份字串集合）的訊號列；其餘列 sig／rank／X 清空、不算換股日（不買、不賣、不拉回）。"""
    S2 = dict(S)
    sig = S['sig'].copy(); rank = S['rank'].copy(); X = S['X'].copy(); sr = S['sigrows'].copy()
    info = []
    for x in S['info']:
        if x['month'] in keep_months:
            info.append(x)
        else:
            k = x['k']; sig[k, :] = False; rank[k, :] = np.nan; X[k, :] = False; sr[k] = False
    S2.update(sig=sig, rank=rank, X=X, sigrows=sr, info=info, RQ={x['k']: S['RQ'][x['k']] for x in info})
    return S2
