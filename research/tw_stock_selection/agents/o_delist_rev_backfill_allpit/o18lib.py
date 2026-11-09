#!/usr/bin/env python3
"""o18 題 1 紙上引擎接線（只讀 import；不改任何他人檔案）：
  o_rev_avail_ds_pit_count/pclib（→ main_v2/v2_core → independent_audit/ia_core）、avail（V6 證據分類器，A_UB 用）、
  o_engine_delist_consideration/cz_engine（對價引擎）＋ events_v2.csv（sha 66e80725…）。
變體＝(fill, lo, rev, dsp, patch)：
  fill：'base'（C0）｜'opt'｜'p90'｜'carry'（未覆蓋的 GOV 缺營收股怎麼放回；pclib.VU 的定義）
  lo：放回期間起點（'2013-01-01'＝U13；'2006-01-01'＝UFULL）
  rev：None｜'UB'（A_UB：D=S／D>S 訊號日 V6 判定非 AVAIL 者營收回退一個月）
  dsp：False｜True（官方全歷史處置期濾網，≤ 2021-09-23，訊號日或成交日涵蓋；與 ia_core 原遮罩聯集）
  patch：None｜rev_backfill 版本（把補回的月年增填進 YL／Y5 的缺值格；只填缺值，不覆蓋）
用法：import o18lib as O；O.build(kind) → dict(plan, t0, t1, sd, aff, nold, base_missing)；O.sim(b, cost, engine)。"""
import sys, json, hashlib, math
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent
AG = HERE.parent
RA = AG / 'o_rev_avail_ds_pit_count'
CZ = AG / 'o_engine_delist_consideration'
sys.path.insert(0, str(RA))
import pclib as P
import avail as AV
sys.path.pop(0)
sys.path.insert(0, str(CZ))
import cz_engine as CE
sys.path.pop(0)
V, K = P.V, P.K
assert CE.K is K
COLS, N, IDX, CI = P.COLS, P.N, P.IDX, P.CI
def sha(f): return hashlib.sha256(Path(f).read_bytes()).hexdigest()

EVF = CZ / 'events_v2.csv'
assert sha(EVF) == '66e80725a4e48387690ded34fc1ebb5b4b7a71ca2b501448c81b30f583b0afbe'
EV = pd.read_csv(EVF, dtype={'code': str, 'successor': str})
R2J = json.loads((CZ / 'results.json').read_text(encoding='utf-8'))
DD_ = R2J['PREREG']['events_v2']['summary']['d']['d_used']
ST = {'cons_main': CE.build_settlement(EV, 'main', d=DD_), 'cons_lo': CE.build_settlement(EV, 'lo', d=DD_),
      'cons_hi': CE.build_settlement(EV, 'hi', d=DD_)}
OLD = ('freeze', 'nofreeze', 'dzero')

# ---------------- A_UB（與 integrate_o17 相同）
E6 = AV.Evidence('sig')
_uc = {}
def _unav_ub(d):
    if d in _uc: return _uc[d]
    u = np.zeros(N, bool)
    if V.deadline_case(d) in ('D=S', 'D>S'):
        ym = str(pd.Period(d, 'M') - 1)
        for j, c in enumerate(COLS):
            u[j] = E6.classify(pd.Timestamp(d), c, ym, 'V6')[0] != 'AVAIL'
    _uc[d] = u
    return u
UNAV = {'UB': _unav_ub}

# ---------------- 官方處置（與 integrate_o17 相同）
DUF = RA / 'raw' / 'disposal_official_universe.csv'
DU = pd.read_csv(DUF, dtype={'code': str}, parse_dates=['start', 'end'])
DSP_HI = pd.Timestamp('2021-09-23')
_orig_dm = K.disposal_mask
_dc = {}
def dm_official(d):
    d = pd.Timestamp(d)
    if d in _dc: return _dc[d]
    m = _orig_dm(d).copy()
    if d <= DSP_HI:
        t = IDX.get_loc(d); e = IDX[t + 1] if t + 1 < len(IDX) else d
        x = DU[((DU.start <= d) & (DU.end >= d)) | ((DU.start <= e) & (DU.end >= e))]
        for c in x.code:
            if c in CI: m[CI[c]] = True
    _dc[d] = m
    return m

# ---------------- 補回值
def patched(ver):
    """回傳 (YLp, Y5p, n_filled)。只填 YL／Y5 原為 NaN 的格。"""
    RB = pd.read_csv(HERE / f'data/rev_backfill_{ver}.csv', dtype={'code': str})
    RB = RB[np.isfinite(RB.yoy_pct)]
    YLp, Y5p = P.YL.copy(), P.Y5.copy()
    nf = {'YL': 0, 'Y5': 0}
    for c, ym, y in zip(RB.code, RB.ym, RB.yoy_pct):
        p = pd.Period(ym, 'M')
        for nm, Y in (('YL', YLp), ('Y5', Y5p)):
            if p not in Y.index: continue
            if c not in Y.columns: Y[c] = np.nan
            if not np.isfinite(Y.at[p, c]): Y.at[p, c] = float(y); nf[nm] += 1
    return YLp, Y5p, nf
_PC = {}
def _pt(ver):
    if ver not in _PC: _PC[ver] = patched(ver)
    return _PC[ver]

_orig_cause = P.cause
def build(kind, excl=()):
    """kind＝(fill, lo, rev, dsp, patch)。excl：這些代碼不放回（缺值原因改判 'EXCL'，等同 C0 的排除）。回傳名單與描述（不算報酬）。"""
    fill, lo, rev, dsp, patch = kind
    K.disposal_mask = dm_official if dsp else _orig_dm
    ex = set(excl)
    if ex: P.cause = (lambda c, d: 'EXCL' if c in ex else _orig_cause(c, d))
    try:
        YLx, Y5x = (P.YL, P.Y5) if patch is None else _pt(patch)[:2]
        if fill == 'base':
            S = V.V2('gapfix', YLx, Y5x, exfin=True, unavail=(UNAV[rev] if rev else None)); aff = {}
        else:
            S = P.VU(fill, lo=lo, hi='2026-12-31', which='GOV', YLx=YLx, Y5x=Y5x)
            if rev: S.unavail = UNAV[rev]
        pl = S.plan(10, 'official')
        t0, t1, sd = S.bounds()
        aff = {d: list(v) for d, v in getattr(S, 'aff', {}).items()}
        # 「原本缺營收」的放回／補回股：基準（無補值）下 y3 缺、且原因屬 GOV 的池內股
        bm = {}
        if patch is not None and fill != 'base':
            for e, pk, el, d in pl:
                if not (pd.Timestamp(lo) <= d): continue
                y0 = K.yoy3_for_month(P.Y5 if d >= V.SWITCH else P.YL, pd.Period(d, 'M'))
                if rev:
                    U = UNAV[rev](d)
                    if U.any(): y0 = np.where(U, K.yoy3_for_month(P.Y5 if d >= V.SWITCH else P.YL, pd.Period(d, 'M') - 1), y0)
                y0 = np.where(V.FINMASK, np.nan, y0)
                cs = S._cs[d]
                cand = [j for j in np.where(cs['el'])[0] if not np.isfinite(y0[j])]
                bm[d] = [COLS[j] for j in cand if (COLS[j] in aff.get(d, []) or P.cause(COLS[j], d) in P.GOV)]
    finally:
        K.disposal_mask = _orig_dm; P.cause = _orig_cause
    return dict(plan=pl, t0=t0, t1=t1, sd=list(sd), aff=aff, base_missing=bm,
                nold={str(d.date()): int(S._cs[d].get('nold', 0)) for d in S._cs} if rev else {})

COST = {'normal': K.rates(0.001), 'stress': K.rates('stress')}
def sim(b, cost, eng):
    pl = [(p[0], p[1]) for p in b['plan']]; bb, ss = COST[cost]
    if eng in OLD: return CE.simulate(pl, bb, ss, b['t0'], b['t1'], mode=eng)
    return CE.simulate(pl, bb, ss, b['t0'], b['t1'], mode='consideration', settle=ST[eng])

# ---------------- 統計（與 integrate_o17 相同口徑：訊號日收盤→下一訊號日收盤窗，對 0050 含息）
sys.path.insert(0, str(AG / 'o_forward_sequential_v2'))
import fslib as F
sys.path.pop(0)
E50 = F.e50()
END = pd.Timestamp(V.END1012)
def win(nav, SD):
    SD = pd.DatetimeIndex(SD)
    pts = list(SD[SD < END]) + [END]
    rows = [(pd.Period(a, 'M'), F._at(nav, b) / F._at(nav, a) - 1, F._at(E50, b) / F._at(E50, a) - 1, a.year) for a, b in zip(pts[:-1], pts[1:])]
    W = pd.DataFrame(rows, columns=['per', 'r', 'r50', 'yr']).set_index('per')
    return W[~W.index.duplicated(keep='last')].dropna()
PERS = {'ALL': (None, None), 'SEG1': ('2006-01', '2012-12'), 'SEG2': ('2013-01', '2018-12'), 'SEG3': ('2019-01', None),
        'SEL': (None, '2019-04'), 'VAL': ('2019-05', None), 'HOLD': ('2024-01', None), 'Y1318': ('2013-01', '2018-12')}
def pmask(idx, k):
    a, b = PERS[k]; m = np.ones(len(idx), bool)
    if a: m &= idx >= pd.Period(a, 'M')
    if b: m &= idx <= pd.Period(b, 'M')
    return m
def tt(x):
    x = np.asarray(x, float); s = x.std(ddof=1)
    return float(x.mean() / s * math.sqrt(len(x))) if len(x) > 2 and s > 1e-15 else None
def ann(r): r = np.asarray(r, float); return float(np.prod(1 + r) ** (12 / len(r)) - 1) * 100
def mdd(r):
    nav = np.cumprod(1 + np.asarray(r, float)); pk_ = np.maximum.accumulate(np.concatenate([[1.0], nav]))[1:]
    return float((nav / pk_ - 1).min()) * 100
def corr(a, b): return float(np.corrcoef(a, b)[0, 1]) if np.std(a) > 0 and np.std(b) > 0 else None
def stats_one(rv, r50, rM, idx, yrs):
    o = {}
    for k in PERS:
        m = pmask(idx, k); a, b, c = rv[m], r50[m], rM[m]
        o[k] = dict(n=int(m.sum()), ann=ann(a), ann_0050=ann(b), gex=ann(a) - ann(b), mdd=mdd(a), t_vs_0050=tt(a - b),
                    mean_ex_pp=float(np.mean(a - b) * 100), diff_vs_ref_pp=float(np.mean(a - c) * 100), t_diff_vs_ref=tt(a - c),
                    ann_diff_vs_ref=ann(a) - ann(c), corr_ref=corr(a, c), corr_0050=corr(a, b))
    s = pd.Series(rv - r50).groupby(np.asarray(yrs)).sum()
    b3 = list(s.sort_values(ascending=False).index[:3]); m = ~np.isin(yrs, b3)
    o['ALL']['t_drop_best3'] = tt(rv[m] - r50[m]); o['ALL']['best3'] = [int(x) for x in b3]
    ya = pd.Series(rv).groupby(np.asarray(yrs)).apply(lambda x: float(np.prod(1 + x) - 1) * 100)
    yb = pd.Series(rM).groupby(np.asarray(yrs)).apply(lambda x: float(np.prod(1 + x) - 1) * 100)
    o['yearly_abs_diff_vs_ref_pp'] = {int(y): float(ya[y] - yb[y]) for y in ya.index}
    return o
def stats_pair(Wk, Wref):
    J = Wref.join(Wk[['r']], rsuffix='_k', how='inner')
    return stats_one(J['r_k'].to_numpy(), J['r50'].to_numpy(), J['r'].to_numpy(), J.index, J['yr'].to_numpy())
