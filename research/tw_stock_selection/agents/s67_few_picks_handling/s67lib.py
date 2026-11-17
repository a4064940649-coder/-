#!/usr/bin/env python3
"""S67 共用程式庫：「名單只有 1～2 檔怎麼辦」訓練→凍結→驗證一次。
候選事前固定（「名單」＝當月訊號名單＝L_MA 依 SURGE 前 10 名中 RET20 > 10% 者，檔數 n_tgt；「名單太少月份」＝n_tgt ≤ 2，含 0 檔）：
  C1＝名單太少月份（1～2 檔），該月新買進每檔目標金額 ×2，上限每檔占開盤估總資產 14%（已持有的不加碼）；
  C2＝名單太少月份不買新股（該訊號列的買進名單清空；已持有照 E0 出場）；
  C3＝名單太少月份把 RET20 門檻從 10% 放寬到 5%：同一批前 10 名（L_MA 依 SURGE）中 5% < RET20 ≤ 10% 者，依 SURGE 由高到低補進，補到名單滿 3 檔為止（不夠就有幾檔補幾檔）。
鏈：s47lib（不改；其下 s20lib → s17lib → s10lib 全部不改）。s47lib 匯入時已把正式開關 EXRIGHT_FIX、DEADLINE_FIX、INS15_MASK 全開。
本檔只做：
  (1) 各層快取目錄改到 scratchpad/s67/cache；
  (2) 以 s47lib._SRC（simulate_s10 原始碼的字串替換副本）再做三處字串替換 → simulate_s67（原檔不動）：
      a. 新參數 mult（{訊號列 k: 倍數}）、mult_cap（每檔上限占開盤估總資產比例）；
      b. u = sv / div 之後：該訊號列有倍數時 u = min(u × 倍數, mult_cap × eq)；
      c. 5% 成交值上限檢查前加計數器 st['n_sized']（S37／S51 口徑的 capped_rate 分母，只計數）。
      mult=None 時與 simulate_s10 逐位相同（驗收 A3）。
  (3) apply_cand()：把候選套到 make_signal_s10 的輸出（C2／C3 改 sig／rank；C1 回傳 mult）。
一律 python3 -I -B。"""
import sys, hashlib
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
AG = HERE.parent
TMP = Path('/tmp/claude-0/-home-user/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/s67')
CACHE = TMP / 'cache'
sys.path.insert(0, str(AG / 's47_allocation_family'))
import s47lib as L47                                       # noqa: E402
from s47lib import s20lib, s17lib, s10lib, s9lib, s8lib, s7lib, s6lib, s5lib, s4lib, s3lib, s2lib, stlib, cslib, COSTS   # noqa: E402,F401
for _m, _a in ((s20lib, 'S20'), (s17lib, 'S17'), (s10lib, 'S10'), (s9lib, 'S9'), (s8lib, 'S8'), (s7lib, 'S7'), (s6lib, 'S6'), (s5lib, 'S5'),
               (s4lib, 'S4'), (s3lib, 'S3'), (s2lib, 'S2'), (stlib, 'S1'), (cslib, 'TMP')):
    setattr(_m, _a, CACHE)
assert cslib.EXRIGHT_FIX is True and cslib.DEADLINE_FIX is True and cslib.INS15_MASK is True, '正式開關必須全開'

CUR = 'CUR'
CANDS = ('C1', 'C2', 'C3')
FEW = 2                                                    # 名單太少：n_tgt ≤ 2
C1_MULT, C1_CAP = 2.0, 0.14
C3_LO, C3_FILL = 0.05, 3
DEPTH, MOM_LO, NSEL = 10, 0.10, 10                         # 現行（make_signal_s10 預設）
CAPS = (6e5, 8e5, 1e6, 1.5e6, 2e6)
SHIFTS = (0, 1, 2, 3)
ACC = dict(since2019=0.4875617247721724, full=0.3439657562340761, maxdd=-0.32119049020243395, n_closed=381)
VALID_FIRST_MONTH = s20lib.VALID_FIRST_MONTH               # 營收月份 ≥ 2021-12（訊號日 2022-01 起）
fresh_from = s20lib.fresh_from
SEGV = ('2022', '2024-07~2025-04')
DROP = ('H', 'L', 'VOL', 'CHG', 'HF', 'VOLF', 'adjH', 'HI240', 'MA60', 'nbar', 'F_HI', 'IDXC', 'AO', 'listed', 'NEARHI', 'V60', 'liq')

_REPL = [
    ("def simulate_s47(", "def simulate_s67("),
    ("requal=None, ndiv=None, idle='e50'):", "requal=None, ndiv=None, idle='e50', mult=None, mult_cap=None):"),
    ("""        u = sv / div
""",
     """        u = sv / div
        _mm = mult.get(k) if mult else None
        if _mm is not None:
            u = min(u * _mm, mult_cap * eq) if mult_cap else u * _mm
            st['n_mult_rows'] = st.get('n_mult_rows', 0) + 1
"""),
    ("""            if cap_pct and np.isfinite(AMT[t, j]):
                mx = int(cap_pct * AMT[t, j] / o)""",
     """            st['n_sized'] = st.get('n_sized', 0) + 1
            if cap_pct and np.isfinite(AMT[t, j]):
                mx = int(cap_pct * AMT[t, j] / o)"""),
]
_SRC = L47._SRC
for _a, _b in _REPL:
    assert _SRC.count(_a) == 1, f'替換點不唯一或不存在：{_a[:40]}'
    _SRC = _SRC.replace(_a, _b)
_G = dict(s10lib.__dict__)
exec(compile(_SRC, '<s67 simulate_s67>', 'exec'), _G)
simulate_s67 = _G['simulate_s67']
SRC_SHA = dict(L47.SRC_SHA, simulate_s67=hashlib.sha256(_SRC.encode()).hexdigest())


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


def pool_of(P, U, RT, x):
    """重建某訊號列的前 10 名（L_MA 依 SURGE，逐行同 make_signal_s10）。回傳 (pool, s, r20)。"""
    m = pd.Period(x['month'], 'M'); k = x['k']
    s = RT['S'].loc[m].to_numpy()
    s = cslib.ins15_mask_row(P, m, k, s)
    fin = U[k] & np.isfinite(s)
    ok_ma = fin & P.F_MA60[k]
    o_ma = np.flatnonzero(ok_ma)[np.argsort(-s[ok_ma], kind='stable')]
    return o_ma[:DEPTH], s, P.RET20[k]


def apply_cand(P, U, RT, S, cand):
    """回傳 (S', mult, 記錄)。cand＝'CUR'／'C1'／'C2'／'C3'。記錄 rec＝[dict(month, k, n_tgt, added=[代號…])]（只列名單太少月份）。"""
    few = [x for x in S['info'] if x['n_tgt'] <= FEW]
    rec = [dict(month=x['month'], k=int(x['k']), n_tgt=int(x['n_tgt']), added=[]) for x in few]
    if cand == 'CUR':
        return S, None, rec
    if cand == 'C1':
        return S, {int(x['k']): C1_MULT for x in few if x['n_tgt'] >= 1}, rec
    S2 = dict(S); sig = S['sig'].copy(); rank = S['rank'].copy()
    if cand == 'C2':
        for x in few:
            sig[x['k'], :] = False; rank[x['k'], :] = np.nan
    elif cand == 'C3':
        for x, r in zip(few, rec):
            k = x['k']
            pool, s, r20 = pool_of(P, U, RT, x)
            tgt = pool[r20[pool] > MOM_LO][:NSEL]
            assert set(int(j) for j in tgt) == set(int(j) for j in np.flatnonzero(S['sig'][k])), '名單重建不符'
            with np.errstate(invalid='ignore'):
                ex = pool[(r20[pool] > C3_LO) & ~(r20[pool] > MOM_LO)]
            add = ex[:max(C3_FILL - len(tgt), 0)]
            sig[k, add] = True; rank[k, add] = s[add]
            r['added'] = [str(P.COLS[j]) for j in add]
    else:
        raise ValueError(cand)
    S2['sig'] = sig; S2['rank'] = rank
    return S2, None, rec


def sim(P, S, U, mult=None, cap=1e6, cost='strict', block=None, cap_pct=0.05, fresh=False, orig=False):
    """F1＋E0、嚴成本（預設）、W10_70、空位 0050。orig=True 走原引擎 simulate_s10（只用於驗收）。"""
    S1 = fresh_from(S, VALID_FIRST_MONTH) if fresh else S
    S2, ekw = s10lib.variant_S(S1, 'E0')
    kw = dict(cost=COSTS[cost], capital=cap, cap_pct=cap_pct, sat_w=0.7, sizing='W10', lock=False, block=block, **ekw)
    if orig:
        assert mult is None
        return s10lib.simulate_s10(P, S2, U, **kw)
    return simulate_s67(P, S2, U, mult=mult, mult_cap=C1_CAP, **kw)


def run(P, U, RT, S, cand, **kw):
    S2, mult, _ = apply_cand(P, U, RT, S, cand)
    return sim(P, S2, U, mult=mult, **kw)


def capped_rate(st):
    n = int(st.get('n_sized', 0))
    return dict(n_sized=n, n_capped=int(st['blk_cap']), capped_rate=(float(st['blk_cap'] / n) if n else 0.0), n_buy=int(st['n_buy']),
                sell_partial_cap=int(st['sell_partial_cap']))
