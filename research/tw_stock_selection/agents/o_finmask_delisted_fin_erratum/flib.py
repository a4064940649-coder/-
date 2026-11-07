#!/usr/bin/env python3
"""本題引擎包裝：只 import main_v2/v2_core 與 o_rev_trunc_2009_erratum/erlib（都不改）。
在每個訊號日 d 把 v2_core.FINMASK 換成待測遮罩 mask(d)（V2.y3 與 erlib.VU 的 el0 都讀這個模組屬性），其他參數不動。
只產名單、合格集合與股池，不計算任何報酬。"""
import sys, contextlib
from pathlib import Path
HERE = Path(__file__).resolve().parent
AG = HERE.parent
sys.path.insert(0, str(AG / 'o_rev_trunc_2009_erratum'))
import erlib as E
sys.path.pop(0)
import numpy as np, pandas as pd
V, K = E.V, E.K
COLS, CI, N = E.COLS, E.CI, E.N
C0MASK = V.FINMASK.copy()
_SURV0, _CAUSE0 = E.SURV, E.cause

@contextlib.contextmanager
def patched(mask=None, surv=None, drop=None):
    """暫時替換 v2_core.FINMASK、erlib.SURV、erlib.cause（drop 內代號的原因改為 DROPPED＝不放回）。"""
    old = V.FINMASK
    try:
        if mask is not None: V.FINMASK = mask
        if surv is not None: E.SURV = tuple(surv)
        if drop: E.cause = (lambda c, m, _d=frozenset(drop): 'DROPPED' if c in _d else _CAUSE0(c, m))
        yield
    finally:
        V.FINMASK = old; E.SURV = _SURV0; E.cause = _CAUSE0

class VM(E.VU):
    """mode：'base'＝C0 型名單；'opt'＝ER2 U_opt。maskfn(d)→bool 陣列（True＝金融，排除）。"""
    def __init__(self, maskfn, mode='base', which='SURV', surv=None, drop=None):
        self.maskfn, self.surv, self.drop = maskfn, surv, drop
        super().__init__(mode, which)

    def cross_section(self, d):
        if d in self._cs: return self._cs[d]
        with patched(self.maskfn(d), self.surv, self.drop):
            return super().cross_section(d)

def top300(d):
    t = K.IDX.get_loc(d)
    S52, V60 = V.rolling_signals('gapfix')
    v60 = V60[t]; px = K.Cn[t]
    base = np.isfinite(v60) & (v60 >= 1e7) & np.isfinite(px) & (px >= 10)
    vv = np.where(base, v60, -np.inf)
    order = np.argsort(-vv, kind='stable')
    top = np.zeros(N, bool); top[order[:min(300, base.sum())]] = True
    return top & base

def run(maskfn, mode='base', surv=None, drop=None):
    S = VM(maskfn, mode, 'SURV', surv, drop)
    pl = S.plan(10, 'official')
    out = {}
    for e, pk, el, d in pl:
        if E.LO <= d <= E.HI:
            cs = S.cross_section(d)
            od = S.order(cs, 'official')[:int(cs['el'].sum())]
            out[str(d.date())] = dict(picks=[COLS[j] for j in pk], aff=list(S.aff.get(d, [])), el=cs['el'].copy(),
                                      rank={COLS[j]: i + 1 for i, j in enumerate(od)}, e=e)
    return out

def sym_slots(A, B):
    """每日 |A △ B| / 2 的總和與明細。"""
    tot = 0; rows = []
    for d in sorted(A):
        a, b = set(A[d]['picks']), set(B[d]['picks'])
        ch = len(a ^ b) / 2
        if ch: rows.append(dict(sig=d, out=sorted(a - b), in_=sorted(b - a), slots=ch))
        tot += ch
    return tot, rows

def inserted(R):
    return [(d, c) for d in sorted(R) for c in R[d]['picks'] if c in set(R[d]['aff'])]
