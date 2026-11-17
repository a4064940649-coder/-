#!/usr/bin/env python3
"""S50 共用程式庫：S48 事後候選出場（C1／C2／C3）對現行 E0 的訓練→凍結→驗證一次。
載入鏈與自建出場矩陣逐行照 agents/s48_exit_rule_family/run.py（其現行格已逐位重現 deadline_engine_fix ALL：48.76%／381 筆）：
s20lib→s17lib→s10lib；EXRIGHT_FIX（load 前）；SURGE（STD）在 DEADLINE_FIX／INS15_MASK 關閉下建，之後兩個開關打開、rev_avail_days 重算可得日；
e50_fix_suspension；股池 twse & ~創新板；make_signal_s10(depth=10, mom_lo=0.10, nsel=10)（F1 進場不動）；
出場矩陣 X(N, M) 本檔自建；simulate_s10(strict, sat_w=0.7, W10, lock=False)。不改任何既有檔案。中間檔放 scratchpad/s50/cache。"""
import sys, gc
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
AG = HERE.parent
TMP = Path('/tmp/claude-0/-home-user/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/s50')
CACHE = TMP / 'cache'
sys.path.insert(0, str(AG / 's20_persistent_growth'))
import s20lib                                              # noqa: E402
from s20lib import s17lib, s10lib, s9lib, s8lib, s7lib, s6lib, s5lib, s4lib, s3lib, s2lib, stlib, cslib, COSTS   # noqa: E402,F401
for _m, _a in ((s20lib, 'S20'), (s17lib, 'S17'), (s10lib, 'S10'), (s9lib, 'S9'), (s8lib, 'S8'), (s7lib, 'S7'), (s6lib, 'S6'), (s5lib, 'S5'),
               (s4lib, 'S4'), (s3lib, 'S3'), (s2lib, 'S2'), (stlib, 'S1'), (cslib, 'TMP')):
    setattr(_m, _a, CACHE)

TRAIN_LAST_MONTH = s20lib.TRAIN_LAST_MONTH                 # 營收月份 ≤ 2021-11（訊號日 2016-02～2021-12）
VALID_FIRST_MONTH = s20lib.VALID_FIRST_MONTH               # 營收月份 ≥ 2021-12（訊號日 2022-01 起）
fresh_from = s20lib.fresh_from
CAPS = (6e5, 8e5, 1e6, 1.5e6, 2e6)
CELL = dict(sat_w=0.7, sizing='W10', lock=False)
MAS = (20, 40, 60, 120)
# 候選集合（事前固定）：(排名門檻 N, 出場均線 M)，None＝不用
CANDS = {'CUR': (20, 60), 'C1': (20, 120), 'C2': (None, 20), 'C3': (20, None)}
CAND_DESC = {'CUR': '現行 E0：跌出 L_MA 前 20 名、或換股日還原收盤 ≤ 60 日均線就賣',
             'C1': '前 20 名＋120 日均線：跌出 L_MA 前 20 名、或換股日還原收盤 ≤ 120 日均線就賣',
             'C2': '不用排名＋20 日均線：只要股池內、SURGE 有值且換股日還原收盤 > 20 日均線就續抱',
             'C3': '前 20 名＋不用均線：只看 L_MA 前 20 名（S10 的 MANO 寫法）'}


def setup():
    """回傳 (P, U, RT, FMA)。三個正式開關全開。"""
    CACHE.mkdir(parents=True, exist_ok=True)
    assert cslib.DEADLINE_FIX is False and cslib.INS15_MASK is False
    cslib.EXRIGHT_FIX = True
    P = s20lib.load()
    assert P.exright_fix
    assert (P.T, P.N, str(P.IDX[-1].date())) == (2876, 2142, '2026-10-22'), '面板不符'
    FMA = {}
    with np.errstate(invalid='ignore'):
        for M in MAS:
            FMA[M] = P.G > stlib.rmean(P.G, M)
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
    return P, U, dict(S=S0, av=AV), FMA


def build_exits(P, U, RT, FMA, sg, shift, cl):
    """逐行照 S48 build_exits：依 make_signal_s10 同一迴圈重建每個訊號列的續抱條件，回傳 {(N, M): X}。"""
    X = {c: np.zeros((P.T, P.N), bool) for c in cl}
    ks = []
    for m in RT['S'].index:
        k0 = RT['av'][m]
        if k0 >= P.T or k0 < 61:
            continue
        s = RT['S'].loc[m].to_numpy()
        if int((U[k0] & np.isfinite(s)).sum()) < 50:
            continue
        k = k0 + shift
        if k >= P.T:
            continue
        s = cslib.ins15_mask_row(P, m, k, s)
        fin = U[k] & np.isfinite(s)
        ok_ma = fin & P.F_MA60[k]
        o_ma = np.flatnonzero(ok_ma)[np.argsort(-s[ok_ma], kind='stable')]
        sn = np.nan_to_num(s, nan=-np.inf)
        ks.append(int(k))
        for (n, mm) in cl:
            keep = fin.copy()
            if n is not None:
                thr = s[o_ma[n - 1]] if len(o_ma) >= n else -np.inf
                keep &= sn >= thr
            if mm is not None:
                keep &= FMA[mm][k]
            X[(n, mm)][k, :] = ~keep
    assert ks == [int(x['k']) for x in sg['info']], '訊號列不符'
    return X


def signals(P, U, RT, FMA, shift):
    """位移 shift 的訊號與四個設定的 S（X 換成自建矩陣）；回傳 ({cfg: S}, A1 ok)。"""
    sg = s10lib.make_signal_s10(P, U, RT, shift=shift, depth=10, mom_lo=0.10, nsel=10)
    XS = build_exits(P, U, RT, FMA, sg, shift, list(CANDS.values()))
    a1 = bool(np.array_equal(XS[CANDS['CUR']], sg['XV']['E0']))
    base = dict(sg); del base['XV']
    out = {}
    for c, nm in CANDS.items():
        S2 = dict(base); S2['X'] = XS[nm]
        out[c] = S2
    return out, a1


def sim(P, U, S, cap=1e6, cost='strict', fresh=False, block=None):
    if fresh:
        S = fresh_from(S, VALID_FIRST_MONTH)
    return s10lib.simulate_s10(P, S, U, cost=COSTS[cost], capital=cap, block=block, **CELL)
