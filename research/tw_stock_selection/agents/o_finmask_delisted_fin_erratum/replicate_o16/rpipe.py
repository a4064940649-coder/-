"""o16 獨立複製：名單管線（自寫截面、排序、遞補；只 import v2_core／ia_core 的面板與滾動訊號原語）。"""
import sys
from pathlib import Path
sys.path.insert(0, '/home/user/-/research/tw_stock_selection/agents/main_v2')
import numpy as np, pandas as pd
import v2_core as V
K = V.K
sys.path.pop(0)
import rcls

COLS = list(V.COLS); N = len(COLS); CI = {c: j for j, c in enumerate(COLS)}
YL, Y5, _ = V.load_revenue()
_v = V.V2('gapfix', YL, Y5, exfin=True, unavail=None)       # 只取 sig_days／S52／V60
S52, V60 = _v.S52, _v.V60
DAYS = [d for d in _v.sig_days if pd.Timestamp('2006-01-01') <= d <= pd.Timestamp('2013-12-31')]
assert len(DAYS) == 93 and _v.sig_days[0] == DAYS[0]
GOV = ('TRUNC', 'IFRS13', 'PRE12', 'HOLE', 'NEVER_DOM')
PRI = ['TRUNC', 'IFRS13', 'PRE12', 'HOLE', 'NEWLIST', 'BASE_LE0']

S = rcls.load_sources()
SL = S['sl26']
C0 = np.array([(SL['industry'].get(c) == '金融保險業') or c[:2] == '28' or c in ('5876', '5880') for c in COLS])
assert (C0 == V.FINMASK).all()

# ---------- 營收水準（tw_revenue，同代碼同月重複時取 sii） ----------
_tr = pd.read_csv(rcls.TWREV, dtype={'code': str, 'market': str})
_tr['p'] = pd.PeriodIndex(pd.to_datetime(_tr['month']), freq='M')
_tr['pri'] = (_tr['market'] != 'sii').astype(int)
_tr = _tr.sort_values(['p', 'code', 'pri']).drop_duplicates(['p', 'code'])
LV = _tr.pivot(index='p', columns='code', values='revenue_k_twd')
LV = LV.reindex(pd.period_range(LV.index.min(), LV.index.max(), freq='M'))
REVC = set(_tr['code'])
FIRST = {c: LV[c].first_valid_index() for c in LV.columns}
LAST = {c: LV[c].last_valid_index() for c in LV.columns}
TR = K.TRADED
LASTTRADE = {}
for j, c in enumerate(COLS):
    w = np.where(TR[:, j])[0]
    LASTTRADE[c] = pd.Period(K.IDX[w[-1]], 'M') if len(w) else None

def foreign(c):
    if c[:2] == '91': return True
    for n in S['names'].get(c, ()):
        if 'KY' in n or 'DR' in n or n.startswith('F-'): return True
    return False

def lv(c, x):
    if c not in LV.columns or x not in LV.index: return np.nan
    return LV.at[x, c]

def cause(c, m):
    if foreign(c): return 'FOREIGN'
    if c not in REVC: return 'NEVER_DOM'
    f, l = FIRST[c], LAST[c]; lt = LASTTRADE[c]
    found = set()
    for k in (1, 2, 3):
        for x in (m - k, m - k - 12):
            v = lv(c, x)
            if np.isnan(v):
                if x > l and lt is not None and pd.Period('2009-12', 'M') <= lt <= pd.Period('2012-12', 'M'): found.add('TRUNC')
                elif x >= pd.Period('2013-01', 'M') and l >= pd.Period('2012-12', 'M'): found.add('IFRS13')
                elif x > l: found.add('PRE12')
                elif x < f: found.add('NEWLIST')
                else: found.add('HOLE')
        b = lv(c, m - k - 12)
        if not np.isnan(lv(c, m - k)) and not np.isnan(b) and b <= 0: found.add('BASE_LE0')
    for p in PRI:
        if p in found: return p
    return 'OTHER'

# ---------- 截面 ----------
_base = {}
def base(d):
    if d in _base: return _base[d]
    t = K.IDX.get_loc(d); e = t + 1
    v60 = V60[t]; px = K.Cn[t]
    ok = np.isfinite(v60) & (v60 >= 1e7) & np.isfinite(px) & (px >= 10)
    vv = np.where(ok, v60, -np.inf)
    o = np.argsort(-vv, kind='stable')
    top = np.zeros(N, bool); top[o[:min(300, int(ok.sum()))]] = True; top &= ok
    y3 = K.yoy3_for_month(YL, pd.Period(d, 'M'))
    s52 = S52[t]
    pre = top & np.isfinite(s52) & ~K.disposal_mask(d) & np.isfinite(K.On[e]) & K.TRADED[e]
    gu = K.gap_up_open(e)
    _base[d] = dict(t=t, e=e, top=top, y3=y3, s52=s52, pre=pre, gu=gu)
    return _base[d]

_cause = {}
def cause_cached(c, d):
    k = (c, d)
    if k not in _cause: _cause[k] = cause(c, pd.Period(d, 'M'))
    return _cause[k]

def rank_avg(x, m):
    """在 m 內的 average rank（1..n），其他 NaN"""
    s = pd.Series(np.where(m, x, np.nan))
    return s.rank(method='average').to_numpy()

def section(d, mask, mode='C0', gov=GOV, drop=(), relabel=None):
    """回傳 (排序後候選 index, el′ 大小, A 集合 index)"""
    b = base(d)
    y3m = np.where(mask, np.nan, b['y3'])
    el = b['pre'] & np.isfinite(y3m)
    A = np.zeros(N, bool)
    if mode == 'U':
        cand = np.where(b['pre'] & ~mask & ~np.isfinite(b['y3']))[0]
        for j in cand:
            c = COLS[j]
            if c in drop: continue
            cs = cause_cached(c, d)
            if relabel and c in relabel: cs = relabel[c]
            if cs in gov: A[j] = True
    elp = el | A
    n = int(elp.sum())
    p52 = rank_avg(b['s52'], elp) / n
    rk = rank_avg(y3m, el) / n
    prev = np.where(A, 1.0, rk)
    sc = 0.5 * p52 + 0.5 * prev
    sc = np.where(elp, sc, np.nan)
    scr = np.round(np.nan_to_num(sc, nan=-9), 10)
    yk = np.where(A, np.inf, np.nan_to_num(y3m, nan=-1e18))
    order = np.lexsort((np.arange(N), -yk, -scr))[:n]
    return order, n, set(np.where(A)[0]), el

def run(maskfn, mode='C0', **kw):
    """maskfn(d)→bool 陣列。回傳 {d: [codes 依選入順序]}, {d: A codes}"""
    held = set(); out = {}; aff = {}
    for d in DAYS:
        order, n, A, el = section(d, maskfn(d), mode, **kw)
        if el.sum() < 50: continue
        gu = base(d)['gu']
        pick = []
        for j in order:
            if gu[j] and j not in held: continue
            pick.append(j)
            if len(pick) == 10: break
        held = set(pick)
        out[d] = [COLS[j] for j in pick]; aff[d] = {COLS[j] for j in A}
    return out, aff
