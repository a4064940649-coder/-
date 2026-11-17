#!/usr/bin/env python3
"""S51 共用程式庫：「每檔多押一點」訓練→凍結→驗證一次（候選事前固定：C1＝S70_D8、C2＝S80_D10、C3＝S100_D12；現行 S70_D10）。
鏈：s47lib（不改；其下 s20lib → s17lib → s10lib，全部不改）。s47lib 匯入時已把正式開關 EXRIGHT_FIX、DEADLINE_FIX、INS15_MASK 全開。
本檔只做：
  (1) 把各層快取目錄改到 scratchpad/s51/cache（s47lib 的 CACHE 常數不用）；
  (2) 以 s47lib 已替換好的 simulate_s47 原始碼（L._SRC）再做「一處」字串替換 → simulate_s51：
      在買進迴圈的 5% 成交值上限檢查之前加一個計數器 st['n_sized']（＝S37 口徑的「有算出股數、進到上限檢查」的買進嘗試數），
      配合引擎原有的 st['blk_cap']（被 5% 上限縮量的次數）得到 S37 的 capped_rate＝blk_cap ÷ n_sized。計數器不影響任何計算（驗收 A3 逐位相同）。
  (3) sim()：一律用 simulate_s51（ndiv、idle='e50'），可帶 block（V3 禁買）、cap_pct、cost。
中間檔放 scratchpad/s51/cache。一律 python3 -I -B。"""
import sys, hashlib
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
AG = HERE.parent
TMP = Path('/tmp/claude-0/-home-user/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/s51')
CACHE = TMP / 'cache'
sys.path.insert(0, str(AG / 's47_allocation_family'))
import s47lib as L47                                       # noqa: E402
from s47lib import s20lib, s17lib, s10lib, s9lib, s8lib, s7lib, s6lib, s5lib, s4lib, s3lib, s2lib, stlib, cslib, COSTS   # noqa: E402,F401
for _m, _a in ((s20lib, 'S20'), (s17lib, 'S17'), (s10lib, 'S10'), (s9lib, 'S9'), (s8lib, 'S8'), (s7lib, 'S7'), (s6lib, 'S6'), (s5lib, 'S5'),
               (s4lib, 'S4'), (s3lib, 'S3'), (s2lib, 'S2'), (stlib, 'S1'), (cslib, 'TMP')):
    setattr(_m, _a, CACHE)
assert cslib.EXRIGHT_FIX is True and cslib.DEADLINE_FIX is True and cslib.INS15_MASK is True, '正式開關必須全開'

CUR = 'S70_D10'
CANDS = {'C1': 'S70_D8', 'C2': 'S80_D10', 'C3': 'S100_D12'}
CELLS = {'S70_D10': dict(sat_w=0.7, div=10), 'S70_D8': dict(sat_w=0.7, div=8), 'S80_D10': dict(sat_w=0.8, div=10), 'S100_D12': dict(sat_w=1.0, div=12)}
CAPS = (6e5, 8e5, 1e6, 1.5e6, 2e6)
SHIFTS = (0, 1, 2, 3)
ACC = dict(since2019=0.4875617247721724, full=0.3439657562340761, maxdd=-0.32119049020243395, n_closed=381)
TRAIN_LAST_MONTH = s20lib.TRAIN_LAST_MONTH                 # 營收月份 ≤ 2021-11（訊號日 2016-02～2021-12）
VALID_FIRST_MONTH = s20lib.VALID_FIRST_MONTH               # 營收月份 ≥ 2021-12（訊號日 2022-01 起）
fresh_from = s20lib.fresh_from
SEGV = ('2022', '2024-07~2025-04')
DROP = ('H', 'L', 'VOL', 'CHG', 'HF', 'VOLF', 'adjH', 'HI240', 'MA60', 'nbar', 'F_HI', 'IDXC', 'AO', 'listed', 'NEARHI', 'V60', 'liq')

_A = """            if cap_pct and np.isfinite(AMT[t, j]):
                mx = int(cap_pct * AMT[t, j] / o)"""
_B = """            st['n_sized'] = st.get('n_sized', 0) + 1
            if cap_pct and np.isfinite(AMT[t, j]):
                mx = int(cap_pct * AMT[t, j] / o)"""
_SRC0 = L47._SRC
assert _SRC0.count(_A) == 1, '替換點不唯一'
_SRC = _SRC0.replace(_A, _B).replace('def simulate_s47(', 'def simulate_s51(')
assert _SRC.count('def simulate_s51(') == 1
_G = dict(s10lib.__dict__)
exec(compile(_SRC, '<s51 simulate_s51>', 'exec'), _G)
simulate_s51 = _G['simulate_s51']
SRC_SHA = dict(L47.SRC_SHA, simulate_s51=hashlib.sha256(_SRC.encode()).hexdigest())


def load():
    CACHE.mkdir(parents=True, exist_ok=True)
    P = s20lib.load()
    assert P.exright_fix is True
    return P


def base(P):
    INNOV, _ = s4lib.innov_panel(P)
    U = stlib.universe(P, 'twse') & ~INNOV
    F = s17lib.rev_family(P, defs=['STD'])
    RT = dict(S=F['S']['STD'], av=F['av'])
    return U, RT


def slim(P):
    for a in DROP:
        if hasattr(P, a):
            delattr(P, a)


def sim(P, S, U, cell, cap=1e6, cost='strict', block=None, cap_pct=0.05, fresh=False, orig=False):
    """cell＝CELLS 的鍵。F1＋E0、嚴成本（預設）、W10 框架下每檔＝衛星 ÷ div、空位 0050。orig=True 走原引擎 simulate_s10（只用於驗收）。"""
    c = CELLS[cell]
    S1 = fresh_from(S, VALID_FIRST_MONTH) if fresh else S
    S2, ekw = s10lib.variant_S(S1, 'E0')
    kw = dict(cost=COSTS[cost], capital=cap, cap_pct=cap_pct, sat_w=c['sat_w'], sizing='W10', lock=False, block=block, **ekw)
    if orig:
        assert c['div'] == 10
        return s10lib.simulate_s10(P, S2, U, **kw)
    return simulate_s51(P, S2, U, ndiv=c['div'], idle='e50', **kw)


def capped_rate(st):
    n = int(st.get('n_sized', 0))
    return dict(n_sized=n, n_capped=int(st['blk_cap']), capped_rate=(float(st['blk_cap'] / n) if n else 0.0), n_buy=int(st['n_buy']),
                sell_partial_cap=int(st['sell_partial_cap']))


def cagr_ex(nav, a, b, year):
    """去掉某一曆年的年化報酬：窗 (a, b] 的日報酬中剔除該年各日，其餘連乘；年數＝(窗內曆日數 − 該年在窗內的曆日數) ÷ 365.25。"""
    s = nav.loc[a:b].dropna()
    r = s.pct_change().dropna()
    keep = r.index.year != year
    g = float(np.prod(1 + r[keep].to_numpy()))
    days = (s.index[-1] - s.index[0]).days
    y0, y1 = max(pd.Timestamp(f'{year}-01-01'), s.index[0]), min(pd.Timestamp(f'{year}-12-31'), s.index[-1])
    dex = max((y1 - y0).days + 1, 0) if y1 >= y0 else 0
    yrs = (days - dex) / 365.25
    return float(g ** (1 / yrs) - 1)


def maxdd_ex(nav, a, b, year):
    """去掉某一曆年的最大回撤：剔除該年日報酬後連乘出的序列之最大回撤。"""
    s = nav.loc[a:b].dropna()
    r = s.pct_change().dropna()
    r = r[r.index.year != year]
    x = np.concatenate([[1.0], np.cumprod(1 + r.to_numpy())])
    return float((x / np.maximum.accumulate(x) - 1).min())
