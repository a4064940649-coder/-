#!/usr/bin/env python3
"""S62 共用程式庫：小資金散戶可行性（描述性；選股、進出場、配置規則一律不動，只改「本金、起跑日、手續費／低消、零股規則與零股滑價」）。
鏈：s20lib → s17lib → s10lib（全部不改）；正式設定 cslib.EXRIGHT_FIX、DEADLINE_FIX、INS15_MASK 全開（load() 之前設）。
simulate_s62：用 inspect 取 s10lib.simulate_s10 原始碼，做字串替換後 exec（原檔不動，寫法同 s47lib）。新參數（全部預設＝原行為）：
    start_t  ：起跑列號（None＝P.t0＝2015-07-01）；本題小資金一律 2018 年最後交易日起跑（全部現金，次日起照規則）。
    fee_rate ：手續費率（None＝stlib.FEE＝0.1425%×0.28）。
    min_odd  ：零股委託（不足一張的那一筆）與 0050 買賣的單筆最低手續費（None＝20 元）；整張那一筆的最低手續費固定 20 元。
    odd_from ：'hist'（原行為：2020-10-26 起才可買零股）、'all'（全期可買零股＝現行制度回推）、'never'（全期只買整張＝整股買不起就放棄）。
零股額外滑價用原引擎既有的 cost['oddx']（不足一張的部分每邊再加），不必改碼。sweep_min（閒錢掃進 0050 的門檻）由呼叫端給。
全部預設時與 simulate_s10 逐位相同（驗收 A1）。一律 python3 -I -B。"""
import sys, inspect, hashlib
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
AG = HERE.parent
TMP = Path('/tmp/claude-0/-home-user/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/s62')
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

ACC = dict(since2019=0.4875617247721724, full=0.3439657562340761, maxdd=-0.32119049020243395, n_closed=381)   # deadline_engine_fix ALL（現行 100 萬）
FEE_FULL = 0.001425
CAPS = (5e4, 1e5, 2e5, 3e5, 5e5)                           # 題目本金
REF_CAP = 1e6                                              # 參考：100 萬（同一套新起跑口徑）
MULTS = (0.8, 0.9, 1.0, 1.1, 1.2)                          # 20 條路徑的 5 個鄰近本金＝本金 × 這些倍數
SHIFTS = (0, 1, 2, 3)
SWEEP_FRAC = 0.02                                          # 閒錢門檻＝本金 × 2%（100 萬時＝現行 2 萬）

# 情境：disc＝手續費折扣；min_odd＝零股與 0050 單筆低消；oddx＝零股每邊額外滑價；odd＝零股規則
SCEN = {
    'R0_現行嚴成本':   dict(disc=0.28, min_odd=20.0, oddx=0.0,   odd='all'),
    'A_樂觀':         dict(disc=0.28, min_odd=1.0,  oddx=0.003, odd='all'),
    'B_基準':         dict(disc=0.6,  min_odd=1.0,  oddx=0.003, odd='all'),
    'C_保守':         dict(disc=1.0,  min_odd=20.0, oddx=0.006, odd='all'),
    'D_只買整張':      dict(disc=0.6,  min_odd=20.0, oddx=0.0,   odd='never'),
    'B_歷史零股制度':   dict(disc=0.6,  min_odd=1.0,  oddx=0.003, odd='hist'),
    'B_門檻固定2萬':   dict(disc=0.6,  min_odd=1.0,  oddx=0.003, odd='all', sweep_fixed=20000.0),
}
MAIN = 'B_基準'
GRID = [dict(disc=d, min_odd=m, oddx=x, odd=o) for d in (1.0, 0.6, 0.28) for m in (20.0, 1.0) for x in (0.0, 0.003, 0.006) for o in ('all', 'never')]   # 36 格（只跑題目本金、位移 0）


def gid(g):
    return f"折{g['disc']}_低消{int(g['min_odd'])}_零股滑價{g['oddx']*100:.1f}%_{'零股' if g['odd'] == 'all' else '只整張'}"


_SRC0 = inspect.getsource(s10lib.simulate_s10)
_REPL = [
    ("def simulate_s10(", "def simulate_s62("),
    ("exec_mode='open', trail=None, trail_daily=True, requal=None):",
     "exec_mode='open', trail=None, trail_daily=True, requal=None, start_t=None, fee_rate=None, min_odd=None, odd_from='hist'):"),
    ("    T, t0 = P.T, P.t0\n",
     "    T, t0 = P.T, (P.t0 if start_t is None else int(start_t))\n"),
    ("    fee, tax, minfee = FEE * cm, TAX * cm, MINFEE * cm\n",
     "    fee, tax, minfee = (FEE if fee_rate is None else fee_rate) * cm, TAX * cm, (MINFEE if min_odd is None else min_odd) * cm\n"),
    ("    tick = cslib.tick; _fee = stlib._fee; etf_tick = stlib.etf_tick\n",
     "    tick = cslib.tick; etf_tick = stlib.etf_tick\n"
     "    if min_odd is None:\n"
     "        _fee = stlib._fee\n"
     "    else:\n"
     "        def _fee(shares, px, rate, mf):\n"
     "            if rate == 0:\n"
     "                return 0.0\n"
     "            rnd = (shares // 1000) * 1000; odd = shares - rnd; f = 0.0\n"
     "            if rnd > 0:\n"
     "                f += max(MINFEE * cm, rnd * px * rate)\n"
     "            if odd > 0:\n"
     "                f += max(mf, odd * px * rate)\n"
     "            return f\n"),
    ("    odd_t = int(np.searchsorted(P.IDX.values, np.datetime64(cslib.ODD_DATE), side='left'))\n",
     "    odd_t = int(np.searchsorted(P.IDX.values, np.datetime64(cslib.ODD_DATE), side='left'))\n"
     "    assert odd_from in ('hist', 'all', 'never')\n"
     "    if odd_from == 'all':\n"
     "        odd_t = 0\n"
     "    elif odd_from == 'never':\n"
     "        odd_t = T + 1\n"),
]
_SRC = _SRC0
for _a, _b in _REPL:
    assert _SRC.count(_a) == 1, f'替換點不唯一或不存在：{_a[:40]}'
    _SRC = _SRC.replace(_a, _b)
_G = dict(s10lib.__dict__)
exec(compile(_SRC, '<s62 simulate_s62>', 'exec'), _G)
simulate_s62 = _G['simulate_s62']
SRC_SHA = dict(simulate_s10=hashlib.sha256(_SRC0.encode()).hexdigest(), simulate_s62=hashlib.sha256(_SRC.encode()).hexdigest())


def load():
    CACHE.mkdir(parents=True, exist_ok=True)
    P = s20lib.load()
    assert P.exright_fix is True
    return P


def setup():
    """載入面板、股池、營收訊號原料（同 S47 run.py），回傳 P, U, RT。"""
    import gc
    P = load()
    assert (P.T, P.N, str(P.IDX[-1].date())) == (2876, 2142, '2026-10-22'), 'P0 失敗'
    INNOV, _ = s4lib.innov_panel(P)
    U = stlib.universe(P, 'twse') & ~INNOV
    del INNOV
    F = s17lib.rev_family(P, defs=['STD'])
    RT = dict(S=F['S']['STD'], av=F['av'])
    del F
    s17lib.e50_fix_suspension(P)
    for a in ('H', 'L', 'VOL', 'CHG', 'HF', 'VOLF', 'adjH', 'HI240', 'MA60', 'nbar', 'F_HI', 'IDXC', 'AO', 'listed', 'NEARHI', 'V60', 'liq'):
        if hasattr(P, a):
            delattr(P, a)
    gc.collect()
    return P, U, RT


def sim_official(P, S, U, cap=1e6):
    """現行正式口徑（原引擎、2015-07-01 起跑、嚴成本）。"""
    S2, ekw = s10lib.variant_S(S, 'E0')
    return s10lib.simulate_s10(P, S2, U, cost=COSTS['strict'], capital=cap, cap_pct=0.05, sat_w=0.7, sizing='W10', lock=False, **ekw)


def sim(P, S, U, cap, sc, start_t, defaults=False):
    """散戶情境（引擎副本）。defaults=True：全部新參數用預設（驗收 A1 用）。"""
    S2, ekw = s10lib.variant_S(S, 'E0')
    kw = dict(capital=cap, cap_pct=0.05, sat_w=0.7, sizing='W10', lock=False, **ekw)
    if defaults:
        return simulate_s62(P, S2, U, cost=COSTS['strict'], **kw)
    cost = dict(cmult=1.0, xslip=0.003, oddx=sc['oddx'])
    return simulate_s62(P, S2, U, cost=cost, start_t=start_t, fee_rate=FEE_FULL * sc['disc'], min_odd=sc['min_odd'], odd_from=sc['odd'],
                        sweep_min=sc.get('sweep_fixed', SWEEP_FRAC * cap), **kw)
