#!/usr/bin/env python3
"""R64 獨立重現用程式庫（沒有讀 s62lib.py／s64lib.py／兩題的 run.py）。
鏈：s20lib → s17lib → s10lib（全部 import、不改）；三個正式開關全開（順序照 agents/s56_industry_cap_family/run.py 的載入段）。
引擎副本 simulate_r64：inspect 取 s10lib.simulate_s10 原始碼做字串替換後 exec（寫法同 s47lib），新參數全預設時應與原引擎逐位相同：
  S64 用：stop=(X, M)（帳戶淨值回落停手）、gapx（開盤漲超過 x 不買）
  S62 用：start_t（新起跑列）、fee_rate（手續費率）、min_odd（零股與 0050 的低消）、odd_from（'hist'／'all'／'never'）
一律 python3 -I -B。"""
import sys, inspect, gc
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
AG = HERE.parent
TMP = Path('/tmp/claude-0/-home-user/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/r64')
CACHE = TMP / 'cache'
sys.path.insert(0, str(AG / 's20_persistent_growth'))
import s20lib                                              # noqa: E402
from s20lib import s17lib, s10lib, s9lib, s8lib, s7lib, s6lib, s5lib, s4lib, s3lib, s2lib, stlib, cslib, COSTS   # noqa: E402,F401
for _m, _a in ((s20lib, 'S20'), (s17lib, 'S17'), (s10lib, 'S10'), (s9lib, 'S9'), (s8lib, 'S8'), (s7lib, 'S7'), (s6lib, 'S6'), (s5lib, 'S5'),
               (s4lib, 'S4'), (s3lib, 'S3'), (s2lib, 'S2'), (stlib, 'S1'), (cslib, 'TMP')):
    setattr(_m, _a, CACHE)
assert cslib.DEADLINE_FIX is False and cslib.INS15_MASK is False and cslib.EXRIGHT_FIX is False
cslib.EXRIGHT_FIX = True

_SRC0 = inspect.getsource(s10lib.simulate_s10)
_REPL = [
    ("def simulate_s10(", "def simulate_r64("),
    ("exec_mode='open', trail=None, trail_daily=True, requal=None):",
     "exec_mode='open', trail=None, trail_daily=True, requal=None, stop=None, gapx=None, start_t=None, fee_rate=None, min_odd=None, odd_from='hist'):"),
    # ---- S62：起跑列、手續費率、零股／0050 低消、零股制度
    ("    T, t0 = P.T, P.t0\n", "    T = P.T; t0 = P.t0 if start_t is None else int(start_t)\n"),
    ("    fee, tax, minfee = FEE * cm, TAX * cm, MINFEE * cm\n",
     "    fee, tax, minfee = (FEE if fee_rate is None else fee_rate) * cm, TAX * cm, MINFEE * cm\n"
     "    mo = minfee if min_odd is None else min_odd * cm\n"),
    ("    tick = cslib.tick; _fee = stlib._fee; etf_tick = stlib.etf_tick\n",
     "    tick = cslib.tick; etf_tick = stlib.etf_tick\n"
     "    def _fee(shares, px, rate, mf):\n"
     "        if rate == 0:\n"
     "            return 0.0\n"
     "        rnd = (shares // 1000) * 1000; odd = shares - rnd; f = 0.0\n"
     "        if rnd > 0:\n"
     "            f += max(mf, rnd * px * rate)\n"
     "        if odd > 0:\n"
     "            f += max(mo, odd * px * rate)\n"
     "        return f\n"),
    ("    odd_t = int(np.searchsorted(P.IDX.values, np.datetime64(cslib.ODD_DATE), side='left'))\n",
     "    odd_t = int(np.searchsorted(P.IDX.values, np.datetime64(cslib.ODD_DATE), side='left'))\n"
     "    assert odd_from in ('hist', 'all', 'never')\n"
     "    if odd_from == 'all':\n"
     "        odd_t = 0\n"
     "    elif odd_from == 'never':\n"
     "        odd_t = T + 1\n"
     "    stopped = False; peak = float(capital); stop_t = -1; stop_ev = []\n"),
    ("    def e50_net(po):\n        nonlocal cash, v50\n",
     "    def e50_net(po):\n        nonlocal cash, v50\n        minfee = mo\n"),
    ("- (minfee if sat50 > 0 else 0.0)", "- (mo if sat50 > 0 else 0.0)"),
    ("- (minfee if v50 > 0 else 0.0)", "- (mo if v50 > 0 else 0.0)"),
    # ---- S64：開盤漲超過 x 不買
    ("""            if o >= P.LIMUP[t, j] - 1e-9:
                st['blk_limup'] += 1; continue
""",
     """            if o >= P.LIMUP[t, j] - 1e-9:
                st['blk_limup'] += 1; continue
            if gapx is not None and o / REF[t, j] - 1 > gapx:
                st['blk_gap'] = st.get('blk_gap', 0) + 1; continue
"""),
    # ---- S64：停手（halt＝前一日收盤的狀態）
    ("        sold_today = set()\n", "        sold_today = set(); halt = stopped\n"),
    ("        if sat_w < 1.0 and t in exec_days:\n            rebalance(t)\n",
     "        if sat_w < 1.0 and t in exec_days and not halt:\n            rebalance(t)\n"),
    ("        if cand:\n            buys(t, cand, po_o)\n", "        if cand and not halt:\n            buys(t, cand, po_o)\n"),
    ("        nav[t] = cash + v50 + tot\n",
     """        nav[t] = cash + v50 + tot
        if stop is not None:
            if not stopped:
                if nav[t] > peak:
                    peak = nav[t]
                if nav[t] < peak * (1.0 - stop[0]):
                    stopped = True; stop_t = t
                    for p in pos.values():
                        if not p['flag']:
                            p['flag'] = True; p['why'] = 'stop'
            else:
                for p in pos.values():
                    if not p['flag']:
                        p['flag'] = True; p['why'] = 'stop'
                r_hi = nav[t] > peak
                r_tm = P.IDX[t] >= P.IDX[stop_t] + pd.DateOffset(months=stop[1])
                if r_hi or r_tm:
                    stopped = False; peak = nav[t]
                    stop_ev.append((int(stop_t), int(t), 'time' if r_tm and not r_hi else 'newhigh'))
"""),
    ("    st['base_sold'] = bsold;", "    st['stop_ev'] = stop_ev; st['stopped_end'] = (int(stop_t) if stopped else None)\n    st['base_sold'] = bsold;"),
]
_SRC = _SRC0
for _a, _b in _REPL:
    assert _SRC.count(_a) == 1, f'替換點不唯一或不存在：{_a[:50]}'
    _SRC = _SRC.replace(_a, _b)
_G = dict(s10lib.__dict__)
exec(compile(_SRC, '<r64 simulate_r64>', 'exec'), _G)
simulate_r64 = _G['simulate_r64']


def load():
    """照 s56 run.py 的載入段：STD SURGE 在兩個期限開關關閉下建，之後開啟，A10 換股日用 rev_avail_days。"""
    CACHE.mkdir(parents=True, exist_ok=True)
    P = s20lib.load()
    assert P.exright_fix
    INNOV, _ = s4lib.innov_panel(P)
    U = stlib.universe(P, 'twse') & ~INNOV
    del INNOV
    F = s17lib.rev_family(P, defs=['STD'])
    s17lib.e50_fix_suspension(P)
    AMT20 = None
    for a in ('H', 'L', 'VOL', 'CHG', 'HF', 'VOLF', 'adjH', 'HI240', 'MA60', 'nbar', 'F_HI', 'IDXC', 'AO', 'listed', 'NEARHI', 'V60', 'liq'):
        if hasattr(P, a):
            delattr(P, a)
    S0 = F['S']['STD']; months = list(S0.index)
    del F; gc.collect()
    cslib.DEADLINE_FIX = True; cslib.INS15_MASK = True
    AV = cslib.rev_avail_days(P, months, True)
    return P, U, dict(S=S0, av=AV)


def build_sig(P, U, RT, shift):
    S = s10lib.make_signal_s10(P, U, RT, shift=shift)
    return dict(sig=S['sig'], rank=S['rank'], X=S['XV']['E0'], SD=None, max_hold=None, info=S['info'], sigrows=S['sigrows'], RQ=S['RQ'])


def cagr(nav, a, b):
    s = nav.loc[a:b].dropna()
    return float((s.iloc[-1] / s.iloc[0]) ** (365.25 / (s.index[-1] - s.index[0]).days) - 1)


def maxdd(nav, a, b):
    s = nav.loc[a:b].dropna()
    return float((s / s.cummax() - 1).min())
