#!/usr/bin/env python3
"""V7X 可執行口徑的名單產生（獨立行程；v7env 把 ia_core 換成 V7 面板）。只讀 import：
  o_v7_listing_exec_engine/v7env → o_v4_revenue_patch_official/v4lib（V4 類別、R_FIX、官方處置 trade、GAP）、
  o_stagger_tranche/st_core（j 定義與 AB 抽樣）、o_rev_avail_ds_pit_count/pclib（缺值原因 cause、GOV）與 avail（V6 證據，A_UB）。
變體：
  'orig'：與 o_v7 s4_jscan7 的 AB_V7_B10 逐字相同（R_FIX、DMT、MC unavail(seed)、GAP＋pfix 1.0）→ 須重現 data/plans_j7.csv。
  'REAL_<fill>'（fill＝opt｜p90｜carry）＝ALLPIT_REAL：
    營收：R_FIX，但「tw_revenue 本身缺值」的格（存活者缺口補值）改成本題過閘的補回值（rev_backfill_<B>），沒有補回值者設為缺；
    unavail：MC(seed) ∪ A_UB（D=S／D>S 日 V6 非 AVAIL；前一個月在資料裡算不出 y3 的資料邊界日不套）；處置：DMT ∪ 官方全歷史清單（≤ 2021-09-23，訊號日或成交日）；
    放回集合：GAP（B10 原定義）∪ GOV 缺值原因（pclib.cause，2006 起）；放回值：opt＝百分位 1.0、p90＝0.9、carry＝沿用最後已知年增並正常排名。"""
import sys, json
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent; AG = HERE.parent
sys.path.insert(0, str(AG / 'o_v7_listing_exec_engine'))
import v7env
L = v7env.load('v7')
sys.path.pop(0)
K, V = L.K, L.V
XR = L.setup_px()
YFX, Y5F = L.load_rfix(); DMT = L.disp_mask_fn('trade'); GAP = L.gap_dict()
sys.path.insert(0, str(AG / 'o_stagger_tranche')); import st_core as STC; sys.path.pop(0)
sys.path.insert(0, str(AG / 'o_rev_avail_ds_pit_count')); import pclib as P; import avail as AV; sys.path.pop(0)
assert P.K is K
COLS = list(K.COLS); N = len(COLS); IDX = K.IDX; CI = {c: j for j, c in enumerate(COLS)}
OFF = L.V4(YFX, Y5F, DMT)
ANCH = [pd.Period(d, 'M') for d in OFF.sig_days]

def days_for(j):
    out = []
    for m in ANCH:
        d = STC.day_for(m, j)
        if d is None: continue
        t = K.IDX.get_loc(d)
        if not K.GOOD[t + 1]: continue
        out.append((m, d))
    de = STC.day_for(pd.Period('2026-10', 'M'), j)
    if de is None or de > K.IDX[-1]: de = K.IDX[-1]
    return [(m, d) for m, d in out if d < de], de

# ---------------- ALLPIT 元件
E6 = None; _uc = {}
def a_ub(d):
    global E6
    if E6 is None: E6 = AV.Evidence('sig')
    if d in _uc: return _uc[d]
    u = np.zeros(N, bool)
    if V.deadline_case(d) in ('D=S', 'D>S'):
        ym = str(pd.Period(d, 'M') - 1)
        for j, c in enumerate(COLS): u[j] = E6.classify(pd.Timestamp(d), c, ym, 'V6')[0] != 'AVAIL'
    _uc[d] = u; return u
DU = pd.read_csv(AG / 'o_rev_avail_ds_pit_count/raw/disposal_official_universe.csv', dtype={'code': str}, parse_dates=['start', 'end'])
_dc = {}
def dm_all(d):
    if d in _dc: return _dc[d]
    m = DMT(d).copy()
    if d <= pd.Timestamp('2021-09-23'):
        t = IDX.get_loc(d); e = IDX[t + 1] if t + 1 < len(IDX) else d
        x = DU[((DU.start <= d) & (DU.end >= d)) | ((DU.start <= e) & (DU.end >= e))]
        for c in x.code:
            if c in CI: m[CI[c]] = True
    _dc[d] = m; return m
def real_tables(bver):
    """R_FIX 的存活者缺口格（tw_revenue YL 缺值、R_FIX 有值）→ 改成本題補回值或缺；其餘不動。Y5F 同法（對 rev_5y 缺值格）。"""
    RB = pd.read_csv(HERE / f'data/rev_backfill_{bver}.csv', dtype={'code': str}); RB = RB[np.isfinite(RB.yoy_pct)]
    rb = {(c, pd.Period(y, 'M')): float(v) for c, y, v in zip(RB.code, RB.ym, RB.yoy_pct)}
    YL0 = P.YL.reindex(index=YFX.index, columns=YFX.columns)
    Y = YFX.copy(); info = dict(rfix_gapcells=0, replaced_by_mine=0, dropped=0, added=0, agree_with_rfix_0p5pp=0, compared=0)
    gapm = YL0.isna().to_numpy() & np.isfinite(YFX.to_numpy())
    ii, jj = np.where(gapm); info['rfix_gapcells'] = int(len(ii))
    for i, j in zip(ii, jj):
        p, c = Y.index[i], Y.columns[j]; v = rb.get((c, p))
        if v is None: Y.iat[i, j] = np.nan; info['dropped'] += 1
        else:
            info['compared'] += 1; info['agree_with_rfix_0p5pp'] += int(abs(v - YFX.iat[i, j]) <= 0.5)
            Y.iat[i, j] = v; info['replaced_by_mine'] += 1
    for (c, p), v in rb.items():
        if p in Y.index:
            if c not in Y.columns: Y[c] = np.nan
            if not np.isfinite(Y.at[p, c]): Y.at[p, c] = v; info['added'] += 1
    Y5 = Y5F.copy() if Y5F is not None else None
    if Y5 is not None:
        Y50 = P.Y5.reindex(index=Y5.index, columns=Y5.columns)
        g5 = Y50.isna().to_numpy() & np.isfinite(Y5.to_numpy()); i5, j5 = np.where(g5)
        info['y5_gapcells'] = int(len(i5)); info['y5_dropped'] = 0; info['y5_replaced'] = 0
        for i, j in zip(i5, j5):
            p, c = Y5.index[i], Y5.columns[j]; v = rb.get((c, p))
            if v is None: Y5.iat[i, j] = np.nan; info['y5_dropped'] += 1
            else: Y5.iat[i, j] = v; info['y5_replaced'] += 1
        for (c, p), v in rb.items():
            if p in Y5.index:
                if c not in Y5.columns: Y5[c] = np.nan
                if not np.isfinite(Y5.at[p, c]): Y5.at[p, c] = v
    return Y, Y5, info

class JV7R(L.V4):
    """JV7（s4_jscan7）＋ALLPIT 放回。putback：'B10'（原 gm）或 'UNION'（GAP ∪ GOV）；fill：opt｜p90｜carry。"""
    def __init__(self, YL, Y5, dmask, putback='B10', fill='opt', **kw):
        self.putback, self.fill = putback, fill
        super().__init__(YL, Y5, dmask, **kw)
        self.ins = {}
    def set_days(self, days, end_day):
        self.anchor = {d: m for m, d in days}; self.sig_days = [d for m, d in days]; self.end_day = end_day; self._cs = {}
    def y3(self, d, back=0):
        m = self.anchor[d] - back
        Y = self.Y5 if d >= V.SWITCH else self.YL
        y = K.yoy3_for_month(Y, m)
        if self.exfin: y = np.where(V.FINMASK, np.nan, y)
        return y
    def cross_section(self, d):
        if self.putback == 'B10' and self.fill == 'opt':
            cs = super().cross_section(d); self.ins[d] = list(cs['gm'].nonzero()[0]); return cs
        if d in self._cs: return self._cs[d]
        cs = super().cross_section(d)          # 先算 V4（gap/pfix 設成 None：見建構）
        ok = cs['ok_other']; y3 = cs['y3'].copy(); s52 = cs['s52']
        miss = ok & ~np.isfinite(y3)
        A = np.zeros(N, bool)
        for j in np.where(miss)[0]:
            c = COLS[j]
            if (c in GAP and np.datetime64(d) <= np.datetime64(GAP[c])) or P.cause(c, d) in P.GOV: A[j] = True
        if self.fill == 'carry':
            yb = {}
            for j in np.where(A)[0]:
                yy = np.nan
                for b in range(1, 25):
                    if b not in yb: yb[b] = self.y3(d, b)
                    z = yb[b][j]
                    if np.isfinite(z): yy = float(z); break
                y3[j] = yy
            A &= np.isfinite(y3)
            el = (ok & np.isfinite(y3))
            pr52 = pd.Series(np.where(el, s52, np.nan)).rank(pct=True).to_numpy()
            prr = pd.Series(np.where(el, y3, np.nan)).rank(pct=True).to_numpy(); y3o = np.where(el, y3, np.nan)
        else:
            el_rev = ok & np.isfinite(y3); el = el_rev | A
            pr52 = pd.Series(np.where(el, s52, np.nan)).rank(pct=True).to_numpy()
            prr = pd.Series(np.where(el_rev, y3, np.nan)).rank(pct=True).to_numpy()
            pv = 1.0 if self.fill == 'opt' else 0.9
            prr = np.where(A, pv, prr)
            tie = np.inf if self.fill == 'opt' else (np.nanpercentile(y3[el_rev], 90) if el_rev.any() else np.nan)
            y3o = np.where(A, tie, y3)
        sc = 0.5 * pr52 + 0.5 * prr; sc[~el] = np.nan
        cs2 = dict(cs, el=el, sc=sc, y3=np.where(el, y3o, np.nan), gm=A)
        self._cs[d] = cs2; self.ins[d] = list(np.where(A)[0])
        return cs2

def make(variant, j, seed, bver='B1', _cache={}):
    days, de = days_for(j)
    if variant == 'orig':
        S = JV7R(YFX, Y5F, DMT, unavail=None, gap=GAP, pfix=1.0)
        S.unavail = (lambda d, _s=seed: STC.unavail_for(S.anchor[d], d, _s))
    else:
        fill = variant.split('_', 1)[1]
        if 'tab' not in _cache: _cache['tab'] = real_tables(bver)
        Y, Y5, _ = _cache['tab']
        S = JV7R(Y, Y5, dm_all, putback='UNION', fill=fill, unavail=None, gap=None, pfix=None)
        def _u(d, _s=seed):
            u = STC.unavail_for(S.anchor[d], d, _s); ub = a_ub(d)
            # 資料邊界：前一個月在資料裡根本算不出 y3（< 100 檔有值，例：2006-04 首月、2021-03 rev_5y 切換月）→ 該日不套 A_UB（無處可回退）
            if ub.any() and np.isfinite(S.y3(d, 1)).sum() < 100: return u
            return u | ub
        S.unavail = _u
    S.set_days(days, de)
    pl = S.plan(10, 'official')
    rows = [(variant, j, seed, str(p[3].date()), str(K.IDX[p[0]].date()), int(p[0]), ','.join(COLS[jj] for jj in p[1]),
             ','.join(COLS[jj] for jj in p[1] if jj in set(S.ins.get(p[3], [])))) for p in pl]
    return rows, str(de.date())
