#!/usr/bin/env python3
"""複製員共用：main_v2 正式引擎（只讀 import v2_core）＋我自建的營收版本、界限分析（B05/B10）、價格修補（PX）。"""
import sys, json, math
from pathlib import Path
H = Path(__file__).resolve().parent
AG = H.parent.parent
sys.path.insert(0, str(AG / 'main_v2'))
import numpy as np, pandas as pd
import v2_core as V
sys.path.pop(0)
K = V.K
IDX, COLS, N = K.IDX, K.COLS, K.N
import os
FOR = set(json.loads((H / 'my_foreign.json').read_text())) | set(filter(None, os.environ.get('EXTRA_FOREIGN', '').split(',')))
FORMASK = np.array([c in FOR for c in COLS])
LASTT = np.where(K.TRADED.any(0), len(IDX) - 1 - np.argmax(K.TRADED[::-1], 0), -1)
import os as _os
B_START = pd.Timestamp(_os.environ.get('B_START', '2013-01-01'))

def load_yoy(ver):
    Y = pd.read_parquet(H / f'my_yoy_{ver}.parquet')
    Y.index = pd.PeriodIndex(Y.index, freq='M')
    return Y

# ---------------- 價格修補 ----------------
PX30 = [c for c in COLS if LASTT[K.CI[c]] >= 0 and IDX[LASTT[K.CI[c]]] == pd.Timestamp('2019-05-16')]

def apply_price_patch():
    X = pd.read_parquet(H / 'my_tse_px.parquet')
    info = {}
    pos = {d: i for i, d in enumerate(IDX)}
    Vn = K.V.to_numpy().copy()
    for nm in ('Cn', 'On', 'Hn', 'Rn', 'Fn'): setattr(K, nm, getattr(K, nm).copy())
    for c in PX30:
        j = K.CI[c]
        x = X[X.code == c].sort_values('date').reset_index(drop=True)
        prev = np.nan; prev_traded = False; nfill = 0; last = None
        for _, r in x.iterrows():
            tr = np.isfinite(r['close'])
            if r['date'] > pd.Timestamp('2019-05-16') and r['date'] in pos and tr:
                t = pos[r['date']]
                ch = r['chg']
                if np.isfinite(ch):
                    cands = np.array([r['close'] + abs(ch), r['close'] - abs(ch)])
                    ref = cands[np.argmin(np.abs(cands - prev))] if np.isfinite(prev) else r['close'] - ch
                    ret = r['close'] / ref - 1
                else:
                    # 'X'：前一日沒成交 → 對上一筆收盤；否則視為除權息日 ret 0（＝面板 flag 1 慣例）
                    ret = (r['close'] / prev - 1) if (np.isfinite(prev) and not prev_traded) else 0.0
                K.Cn[t, j] = r['close']; K.On[t, j] = r['open']; K.Hn[t, j] = r['high']; K.Rn[t, j] = ret
                K.Fn[t, j] = 0; Vn[t, j] = r['value']
                nfill += 1; last = r['date']
            if tr: prev = r['close']
            prev_traded = tr
        info[c] = {'n_days': nfill, 'last': str(last.date()) if last is not None else None}
    # 衍生陣列重算
    K.TRADED = np.isfinite(K.Cn)
    K.GROW = np.where(K.TRADED, 1.0 + np.nan_to_num(K.Rn, nan=0.0), 1.0)
    K.MYADJ = pd.DataFrame(np.where(K.TRADED, np.cumprod(K.GROW, axis=0), np.nan), index=IDX, columns=COLS)
    K.V = pd.DataFrame(Vn, index=IDX, columns=COLS)
    K.C = pd.DataFrame(K.Cn, index=IDX, columns=COLS)
    K.PREVC = K.C.ffill().shift(1).to_numpy()
    K.NOBS = K.TRADED.sum(axis=1); K.GOOD = K.NOBS >= 300
    V._RS.clear()
    global LASTT
    LASTT = np.where(K.TRADED.any(0), len(IDX) - 1 - np.argmax(K.TRADED[::-1], 0), -1)
    return info

# ---------------- 界限分析 ----------------
class V2B(V.V2):
    """bound=None：一般；bound=(p, scope)：營收以外都合格但缺營收（2013-01 起）的股票放回，營收百分位固定 p。
    scope：'DEL'（只國內）或 'ALL'（含 KY/DR）。delisted_only：只放回後來下市者（面板最後交易日早於 2026-10）。"""
    def __init__(self, *a, bound=None, delisted_only=False, **kw):
        super().__init__(*a, **kw)
        self.bound = bound; self.delisted_only = delisted_only

    def base_parts(self, d):
        t = IDX.get_loc(d); e = t + 1
        v60 = self.V60[t]; px = K.Cn[t]
        base = np.isfinite(v60) & (v60 >= 1e7) & np.isfinite(px) & (px >= 10)
        vv = np.where(base, v60, -np.inf)
        order = np.argsort(-vv, kind='stable')
        top = np.zeros(N, bool); top[order[:min(300, base.sum())]] = True
        top &= base
        s52 = self.S52[t]
        ok = top & np.isfinite(s52) & ~K.disposal_mask(d) & np.isfinite(K.On[e]) & K.TRADED[e]
        return t, e, ok

    def missing_set(self, d, y3, scope):
        t, e, ok = self.base_parts(d)
        miss = ok & ~np.isfinite(y3)
        if self.exfin: miss &= ~V.FINMASK
        if scope == 'DEL': miss &= ~FORMASK
        if self.delisted_only: miss &= LASTT < IDX.get_loc(IDX[IDX >= pd.Timestamp('2026-10-01')][0])
        return miss

    def cross_section(self, d):
        if self.bound is None or d < B_START: return super().cross_section(d)
        if d in self._cs: return self._cs[d]
        cs = dict(super().cross_section(d))
        p, scope = self.bound
        add = self.missing_set(d, cs['y3'], scope)
        el0 = cs['el']; el = el0 | add
        def pr(x, m):
            return pd.Series(np.where(m, x, np.nan)).rank(pct=True).to_numpy()
        r52 = pr(cs['s52'], el)
        rrv = pr(cs['y3'], el0)
        rrv = np.where(add, p, rrv)
        sc = 0.5 * r52 + 0.5 * rrv
        sc[~el] = np.nan
        cs.update(el=el, sc=sc, nadd=int(add.sum()))
        self._cs[d] = cs
        return cs
