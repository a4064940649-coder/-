#!/usr/bin/env python3
"""本題共用：營收缺值原因分類（與 s1_explore.py 同定義，凍結於 ERRATUM2_SPEC）、C0 股池、主策略名單重建與 U 口徑排序。
只 import main_v2/v2_core（不改）。不計算任何報酬。"""
import sys, csv, json, hashlib
from pathlib import Path
HERE = Path(__file__).resolve().parent
AG = HERE.parent
sys.path.insert(0, str(AG / 'main_v2'))
import numpy as np, pandas as pd
import v2_core as V
sys.path.pop(0)
K = V.K
COLS = list(K.COLS); CI = {c: j for j, c in enumerate(COLS)}; N = len(COLS)

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()

# ---------- 營收水準 ----------
_d = pd.read_csv(K.LONG, dtype={'code': str, 'market': str})
_d['p'] = pd.PeriodIndex(pd.to_datetime(_d['month']), freq='M')
_d['pri'] = (_d['market'] != 'sii').astype(int)
_d = _d.sort_values(['p', 'code', 'pri']).drop_duplicates(['p', 'code'])
LV = _d.pivot(index='p', columns='code', values='revenue_k_twd')
LV = LV.reindex(pd.period_range(LV.index.min(), LV.index.max(), freq='M'))
LASTREV = {c: LV[c].dropna().index.max() for c in LV.columns}
FIRSTREV = {c: LV[c].dropna().index.min() for c in LV.columns}
NAME = {}
for c, nm in zip(_d['code'], _d['name']): NAME.setdefault(c, nm)
for r in csv.DictReader(open(K.X.parent / 'stock_list.csv', encoding='utf-8')): NAME.setdefault(r['code'], r['name'])
SNAPN = {}
for f in ('grs_568e6685_tools_twse_list.csv', 'grs_893c8772_grs_stock_no.csv', 'goristock_c3c718ab_grs_stock_no.csv'):
    for r in csv.reader(open(AG / 'o_pit_shares_industry_2006_2013/raw/industry_snap' / f, encoding='utf-8', errors='replace')):
        r = [x.strip().strip('"') for x in r]
        if len(r) >= 2 and r[0].isdigit() and len(r[0]) == 4: SNAPN.setdefault(r[0], set()).add(r[1])
LM = {r['code']: r for r in csv.DictReader(open(AG / 'o_v7_listing_exec_engine/listing_master.csv', encoding='utf-8'))}
LT = {}
for j, c in enumerate(COLS):
    w = np.where(K.TRADED[:, j])[0]
    LT[c] = K.IDX[w[-1]] if len(w) else pd.NaT

def is_foreign(c):
    nms = {NAME.get(c, '') or ''} | SNAPN.get(c, set())
    return c[:2] == '91' or any(('KY' in n.upper()) or n.endswith('DR') or n.startswith('F-') for n in nms)

def month_cause(c, x):
    lr, fr = LASTREV[c], FIRSTREV[c]
    ltm = pd.Period(LT[c], 'M')
    if x > lr:
        if pd.Period('2009-12', 'M') <= ltm <= pd.Period('2012-12', 'M'): return 'TRUNC'
        if x >= pd.Period('2013-01', 'M') and lr >= pd.Period('2012-12', 'M'): return 'IFRS13'
        return 'PRE12'
    if x < fr: return 'NEWLIST'
    return 'HOLE'

PRIO = ['TRUNC', 'IFRS13', 'PRE12', 'HOLE', 'NEWLIST', 'BASE_LE0']
SURV = ('TRUNC', 'IFRS13', 'PRE12', 'HOLE', 'NEVER_DOM')

def cause(c, m):
    if is_foreign(c): return 'FOREIGN'
    if c not in LV.columns: return 'NEVER_DOM'
    ks = set()
    for x in [m - k for k in (1, 2, 3)] + [m - k for k in (13, 14, 15)]:
        v = LV[c].get(x, np.nan)
        if not np.isfinite(v): ks.add(month_cause(c, x))
        elif v <= 0 and x <= m - 13: ks.add('BASE_LE0')
    for k in PRIO:
        if k in ks: return k
    return 'UNKNOWN'

# ---------- 主策略（C0） ----------
YL, Y5, _ = V.load_revenue()
LO, HI = pd.Timestamp('2006-01-01'), pd.Timestamp('2013-12-31')

def last_known_y3(c, m, maxback=24):
    for b in range(1, maxback + 1):
        y = K.yoy3_for_month(YL, m - b)[CI[c]]
        if np.isfinite(y): return float(y)
    return np.nan

class VU(V.V2):
    """mode：'base'（＝正式 C0）、'opt'（缺值股營收百分位＝1.0）、'p90'（＝0.9）、'carry'（沿用最後已知 y3 重排）。
    which：受影響集合（'SURV' 判定口徑；'ALL' 所有缺值原因，描述用）。"""
    def __init__(self, mode='base', which='SURV'):
        self.mode, self.which = mode, which
        self.aff = {}
        super().__init__('gapfix', YL, Y5, exfin=True)

    def cross_section(self, d):
        if d in self._cs: return self._cs[d]
        cs = super().cross_section(d)
        if self.mode == 'base' or not (LO <= d <= HI):
            self.aff[d] = []
            return cs
        t, e = cs['t'], cs['e']
        v60 = self.V60[t]; px = K.Cn[t]
        base = np.isfinite(v60) & (v60 >= 1e7) & np.isfinite(px) & (px >= 10)
        vv = np.where(base, v60, -np.inf)
        order = np.argsort(-vv, kind='stable')
        top = np.zeros(N, bool); top[order[:min(300, base.sum())]] = True; top &= base
        s52 = self.S52[t]
        el0 = top & np.isfinite(s52) & ~K.disposal_mask(d) & np.isfinite(K.On[e]) & K.TRADED[e] & ~V.FINMASK
        y3 = cs['y3'].copy()
        m = pd.Period(d, 'M')
        A = np.zeros(N, bool)
        for j in np.where(el0 & ~np.isfinite(y3))[0]:
            k = cause(COLS[j], m)
            if self.which == 'ALL' or k in SURV: A[j] = True
        if self.mode == 'carry':
            for j in np.where(A)[0]:
                y3[j] = last_known_y3(COLS[j], m)
            A &= np.isfinite(y3)
        self.aff[d] = [COLS[j] for j in np.where(A)[0]]
        el = cs['el'] | A
        n = el.sum()
        s52p = pd.Series(np.where(el, s52, np.nan)).rank(pct=True).to_numpy()
        if self.mode == 'carry':
            rp = pd.Series(np.where(el, y3, np.nan)).rank(pct=True).to_numpy()
            ytie = y3
        else:
            norm = el & ~A
            rp = pd.Series(np.where(norm, y3, np.nan)).rank().to_numpy() / n
            rp[A] = 1.0 if self.mode == 'opt' else 0.9
            ytie = np.where(A, (np.inf if self.mode == 'opt' else np.nan), y3)
            if self.mode == 'p90':   # 同分次序：缺值股 y3 以池內 p90 代入
                ytie = np.where(A, np.nanpercentile(y3[norm], 90), y3)
        sc = 0.5 * s52p + 0.5 * rp
        sc[~el] = np.nan
        ytie = np.where(el, ytie, np.nan)
        cs2 = dict(cs, el=el, sc=sc, y3=ytie, A=A)
        self._cs[d] = cs2
        return cs2

def lists(S):
    pl = S.plan(10, 'official')
    out = {}
    for e, pk, el, d in pl:
        if LO <= d <= HI:
            cs = S.cross_section(d)
            out[str(d.date())] = dict(picks=[COLS[j] for j in pk], aff=list(S.aff.get(d, [])),
                                      s52=cs['s52'], el=cs['el'])
    return out
