#!/usr/bin/env python3
"""o_rev_survivorship 核心：只讀 import main_v2/v2_core.py（再只讀 import independent_audit/ia_core.py），
策略規格與 main_v2 正式版完全相同（gapfix 窗、正式次排序、排除金融、AB、成本、期間、統計）。只換營收資料，另加：
  * 營收版本：ORIG（現行 tw_revenue）、R_DEL（補國內缺口）、R_ALL（另補 KY／DR）— 讀 build_panel.py 凍結的 parquet，斷言 sha256
  * 界限分析：仍缺營收的池內股票（gap_stockmonths_filled.csv 中未補到者），營收百分位固定為 p（0.5 或 1.0）後放回合格池
  * 價格修補（診斷 d）：在 K 的陣列上套 price_patch.parquet（另一個行程執行）
"""
import sys, json, hashlib
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'main_v2'))
import numpy as np, pandas as pd
import v2_core as V
K = V.K
sys.path.pop(0)

FREEZE = json.loads((HERE / 'data_freeze.json').read_text(encoding='utf-8'))
PP_META = json.loads((HERE / 'price_patch_meta.json').read_text(encoding='utf-8'))
FOREIGN = set(FREEZE['foreign_codes'])

def sha(f): return hashlib.sha256(Path(f).read_bytes()).hexdigest()

def load_yl(version):
    YL, Y5, Y5f = V.load_revenue()
    if version == 'ORIG': return YL, Y5
    o = FREEZE['outputs'][version]
    f = HERE / o['file']; assert sha(f) == o['sha256'], f'{f} 與凍結的 sha256 不符'
    Y = pd.read_parquet(f); Y.index = pd.PeriodIndex(Y.index, freq='M')
    return Y, Y5

def gapsets(base):
    """回傳 {訊號日 Timestamp: set(code)}：修補後仍缺營收的池內股票（2013-01 起）。
    base='R_DEL'：只算國內公司（KY 視為股池外，與現行規格相同）；base='R_ALL'：國內＋KY。"""
    G = pd.read_csv(HERE / 'gap_stockmonths_filled.csv', dtype={'code': str})
    if base == 'R_DEL': G = G[~G['foreign'] & ~G['fill_del']]
    elif base == 'R_ALL': G = G[~G['fill_all']]
    else: raise ValueError(base)
    out = {}
    for s, g in G.groupby('sig'): out[pd.Timestamp(s)] = set(g['code'])
    return out

class V2B(V.V2):
    """V2 加界限分析：gap（訊號日→代碼集合）中的股票，若營收以外的條件都合格，營收百分位固定為 pfix。"""
    def __init__(self, *a, gap=None, pfix=None, **kw):
        super().__init__(*a, **kw)
        self.gap, self.pfix = gap, pfix
        self.ngap_used = {}

    def cross_section(self, d):
        if self.pfix is None or self.gap is None: return super().cross_section(d)
        if d in self._cs: return self._cs[d]
        t = K.IDX.get_loc(d); e = t + 1
        v60 = self.V60[t]; px = K.Cn[t]
        base = np.isfinite(v60) & (v60 >= 1e7) & np.isfinite(px) & (px >= 10)
        vv = np.where(base, v60, -np.inf)
        order = np.argsort(-vv, kind='stable')
        top = np.zeros(K.N, bool); top[order[:min(300, base.sum())]] = True
        top &= base
        y3 = self.y3(d)
        nold = 0
        if self.unavail is not None:
            U = self.unavail(d)
            if U is not None and U.any():
                y3 = np.where(U, self.y3(d, 1), y3)
                nold = int((U & top).sum())
        s52 = self.S52[t]
        ok_other = top & np.isfinite(s52) & ~K.disposal_mask(d) & np.isfinite(K.On[e]) & K.TRADED[e]
        el_rev = ok_other & np.isfinite(y3)
        gm = np.zeros(K.N, bool)
        for c in self.gap.get(d, ()):
            j = K.CI.get(c)
            if j is not None: gm[j] = True
        if self.exfin: gm &= ~V.FINMASK
        gm &= ok_other & ~np.isfinite(y3)
        el = el_rev | gm
        self.ngap_used[d] = int(gm.sum())
        pr52 = pd.Series(np.where(el, s52, np.nan)).rank(pct=True).to_numpy()
        prr = pd.Series(np.where(el_rev, y3, np.nan)).rank(pct=True).to_numpy()
        prr = np.where(gm, self.pfix, prr)
        sc = 0.5 * pr52 + 0.5 * prr
        sc[~el] = np.nan
        cs = dict(t=t, e=e, el=el, sc=sc, s52=s52, y3=y3, nold=nold, gm=gm)
        self._cs[d] = cs
        return cs

def apply_price_patch():
    """把 price_patch.parquet 寫進 K 的原始表，重算衍生陣列（必須在建立任何 V2 物件之前呼叫）。"""
    f = HERE / 'price_patch.parquet'; assert sha(f) == PP_META['sha256']
    P = pd.read_parquet(f)
    n = 0
    for (dt, c), r in P.set_index(['date', 'code']).iterrows():
        if dt not in K.IDX or c not in K.CI: continue
        if np.isfinite(K.C.at[dt, c]): continue
        K.C.at[dt, c] = r['close']; K.O.at[dt, c] = r['open']; K.H.at[dt, c] = r['high']
        K.V.at[dt, c] = r['value']; K.RET.at[dt, c] = r['ret']; K.FLAG.at[dt, c] = 0
        n += 1
    K.Cn = K.C.to_numpy(); K.On = K.O.to_numpy(); K.Rn = K.RET.to_numpy(); K.Fn = K.FLAG.to_numpy(); K.Hn = K.H.to_numpy()
    K.TRADED = np.isfinite(K.Cn)
    K.GROW = np.where(K.TRADED, 1.0 + np.nan_to_num(K.Rn, nan=0.0), 1.0)
    K.MYADJ = pd.DataFrame(np.where(K.TRADED, np.cumprod(K.GROW, axis=0), np.nan), index=K.IDX, columns=K.COLS)
    K.NOBS = K.TRADED.sum(axis=1); K.GOOD = K.NOBS >= 300
    K.PREVC = K.C.ffill().shift(1).to_numpy()
    V._RS.clear(); K._RS.clear()
    return n
