#!/usr/bin/env python3
"""G6 複製員 r4：紙上 ia 口徑重算（不讀研究員 *.py；只 import 中心共用模組 pclib／avail／cz_engine／fslib，作法照凍結規格與 integrate_o17 的公開定義）。
補回表＝本複製員 r3 產生的 my_backfill.csv（只填 YL／Y5 原本缺值的格；KY 四檔不在表內）。
python3 -I -B r4_paper.py [mine|theirs]   （theirs＝用研究員 rev_backfill_B1.csv 當補回表，只作差異歸因）"""
import sys, json, math, time
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent
TOP = HERE.parent
AG = TOP.parent
RA = AG / 'o_rev_avail_ds_pit_count'; CZ = AG / 'o_engine_delist_consideration'
TAB = sys.argv[1] if len(sys.argv) > 1 else 'mine'
T0 = time.time()
def log(*a): print(f'[{time.time() - T0:6.0f}s]', *a, flush=True)
sys.path.insert(0, str(RA)); import pclib as P; import avail as AV; sys.path.pop(0)
sys.path.insert(0, str(CZ)); import cz_engine as CE; sys.path.pop(0)
sys.path.insert(0, str(AG / 'o_forward_sequential_v2')); import fslib as F; sys.path.pop(0)
V, K = P.V, P.K
COLS, N, IDX = P.COLS, P.N, P.IDX

# ---------------- 補回表 → YL／Y5 缺值格
BF = pd.read_csv(HERE / 'my_backfill.csv' if TAB == 'mine' else TOP / 'data' / 'rev_backfill_B1.csv', dtype={'code': str, 'ym': str})
BF = BF[BF.yoy_pct.notna()]
def patch(Y):
    Y = Y.copy(); n = 0
    for c, ym, v in zip(BF.code, BF.ym, BF.yoy_pct):
        p = pd.Period(ym, 'M')
        if p not in Y.index: continue
        if c not in Y.columns: Y[c] = np.nan
        if not np.isfinite(Y.at[p, c]): Y.at[p, c] = v; n += 1
    return Y, n
YLp, nL = patch(P.YL); Y5p, n5 = patch(P.Y5)
log('patch cells', nL, n5)

# ---------------- A_UB（avail V6）、官方處置
E6 = AV.Evidence('sig'); _uc = {}
def UB(d):
    if d in _uc: return _uc[d]
    u = np.zeros(N, bool)
    if V.deadline_case(d) in ('D=S', 'D>S'):
        ym = str(pd.Period(d, 'M') - 1)
        for j, c in enumerate(COLS): u[j] = E6.classify(pd.Timestamp(d), c, ym, 'V6')[0] != 'AVAIL'
    _uc[d] = u; return u
DU = pd.read_csv(RA / 'raw' / 'disposal_official_universe.csv', dtype={'code': str}, parse_dates=['start', 'end'])
_odm = K.disposal_mask; _dc = {}
def dm_off(d):
    d = pd.Timestamp(d)
    if d in _dc: return _dc[d]
    m = _odm(d).copy()
    if d <= pd.Timestamp('2021-09-23'):
        t = IDX.get_loc(d); e = IDX[t + 1] if t + 1 < len(IDX) else d
        for c in DU[((DU.start <= d) & (DU.end >= d)) | ((DU.start <= e) & (DU.end >= e))].code:
            if c in P.CI: m[P.CI[c]] = True
    _dc[d] = m; return m

_cause0 = P.cause
def build(fill, lo, rev, dsp, real, excl=()):
    K.disposal_mask = dm_off if dsp else _odm
    P.cause = (lambda c, d: 'EXCL' if c in excl else _cause0(c, d)) if excl else _cause0
    try:
        YLx, Y5x = (YLp, Y5p) if real else (P.YL, P.Y5)
        if fill == 'base':
            S = V.V2('gapfix', YLx, Y5x, exfin=True)
        else:
            S = P.VU(fill, lo=lo, hi='2026-12-31', which='GOV', YLx=YLx, Y5x=Y5x)
        if rev: S.unavail = UB
        pl = S.plan(10, 'official'); t0, t1, sd = S.bounds()
        aff = {d: set(v) for d, v in getattr(S, 'aff', {}).items()}
    finally:
        K.disposal_mask = _odm; P.cause = _cause0
    return dict(plan=pl, t0=t0, t1=t1, sd=list(sd), aff=aff)

VARS = {'C0': ('base', None, False, False, False),
        'ALLPIT_opt_o17': ('opt', '2006-01-01', True, True, False),
        'ALLPIT_REAL_opt': ('opt', '2006-01-01', True, True, True),
        'ALLPIT_REAL_p90': ('p90', '2006-01-01', True, True, True),
        'ALLPIT_REAL_carry': ('carry', '2006-01-01', True, True, True),
        'U13_REAL_opt': ('opt', '2013-01-01', False, False, True),
        'U13_REAL_p90': ('p90', '2013-01-01', False, False, True),
        'U13_REAL_carry': ('carry', '2013-01-01', False, False, True),
        'UFULL_REAL_opt': ('opt', '2006-01-01', False, False, True)}
SENS = {'ALLPIT_REAL_opt_X2311': ('2311',), 'ALLPIT_REAL_opt_Xtop3': ('2325', '2311', '1704'), 'ALLPIT_REAL_opt_KYX': ('1262', '1592', '6422', '6452')}
B = {k: build(*v) for k, v in VARS.items()}
for k, ex in SENS.items(): B[k] = build(*VARS['ALLPIT_REAL_opt'], excl=ex)
log('lists built')
assert all(B[k]['sd'] == B['C0']['sd'] for k in B)
SD = pd.DatetimeIndex(B['C0']['sd'])
PK = {k: {p[3]: [COLS[j] for j in p[1]] for p in B[k]['plan']} for k in B}
# 未補值下的「缺 y3」判斷：用無補值同口徑名單的 aff；真實補回槽＝放回候選（原本缺值）因補值而自然入選
B0 = {'U13': build('opt', '2013-01-01', False, False, False), 'ALLPIT': B['ALLPIT_opt_o17']}
def counts(k):
    base = 'ALLPIT' if k.startswith('ALLPIT') else ('U13' if k.startswith('U13') else 'UFULL')
    if base == 'UFULL': B0.setdefault('UFULL', build('opt', '2006-01-01', False, False, False))
    ins = [(d, c) for d in PK[k] for c in PK[k][d] if c in B[k]['aff'].get(d, set())]
    real = [(d, c) for d in PK[k] for c in PK[k][d] if c not in B[k]['aff'].get(d, set()) and c in B0[base]['aff'].get(d, set())]
    D13 = pd.Timestamp('2013-01-01'); D18 = pd.Timestamp('2018-12-31')
    return dict(fill_2013on=sum(d >= D13 for d, _ in ins), fill_0612=sum(d < D13 for d, _ in ins),
                real_2013on=sum(d >= D13 for d, _ in real), real_slots=[(str(d.date()), c) for d, c in real if d >= D13],
                ins_2013on=sum(d >= D13 for d, _ in ins) + sum(d >= D13 for d, _ in real),
                ins_1318=sum(D13 <= d <= D18 for d, _ in ins) + sum(D13 <= d <= D18 for d, _ in real),
                changed_vs_C0=int(sum(len(set(PK['C0'][d]) - set(PK[k][d])) for d in PK['C0'])))
CNT = {k: counts(k) for k in B if k != 'C0'}
log('counts', json.dumps({k: {a: b for a, b in v.items() if a != 'real_slots'} for k, v in CNT.items()}))

# ---------------- 模擬
EV = pd.read_csv(CZ / 'events_v2.csv', dtype={'code': str, 'successor': str})
ST = {b: CE.build_settlement(EV, b.split('_')[1], d=0.057) for b in ('cons_main', 'cons_lo', 'cons_hi')}
COST = {'normal': K.rates(0.001), 'stress': K.rates('stress')}
RUN = [('C0', 'normal', 'freeze'), ('C0', 'stress', 'freeze'), ('ALLPIT_opt_o17', 'normal', 'cons_main'),
       ('ALLPIT_REAL_opt', 'normal', 'cons_main'), ('ALLPIT_REAL_opt', 'stress', 'cons_main'),
       ('ALLPIT_REAL_opt', 'normal', 'cons_lo'), ('ALLPIT_REAL_opt', 'normal', 'cons_hi'), ('ALLPIT_REAL_opt', 'normal', 'freeze'),
       ('ALLPIT_REAL_p90', 'normal', 'cons_main'), ('ALLPIT_REAL_p90', 'stress', 'cons_main'),
       ('ALLPIT_REAL_carry', 'normal', 'cons_main'), ('ALLPIT_REAL_carry', 'stress', 'cons_main'),
       ('U13_REAL_opt', 'normal', 'cons_main'), ('U13_REAL_p90', 'normal', 'cons_main'), ('U13_REAL_carry', 'normal', 'cons_main'),
       ('UFULL_REAL_opt', 'normal', 'cons_main')] + [(k, 'normal', 'cons_main') for k in SENS]
NAV, LG = {}, {}
for k, c, eg in RUN:
    pl = [(p[0], p[1]) for p in B[k]['plan']]; bb, ss = COST[c]
    if eg == 'freeze': nav, lg = CE.simulate(pl, bb, ss, B[k]['t0'], B[k]['t1'], mode='freeze')
    else: nav, lg = CE.simulate(pl, bb, ss, B[k]['t0'], B[k]['t1'], mode='consideration', settle=ST[eg])
    NAV[(k, c, eg)] = nav
    if c == 'normal':
        LG[f'{k}|{eg}'] = dict(n_settled_codes=len({s['code'] for s in lg.get('settled', [])}), settled_codes=sorted({s['code'] for s in lg.get('settled', [])}),
                               frozen_codes=sorted({x for _, cs, _ in lg['frozen'] for x in cs}), frozen_max_share=max([x[2] for x in lg['frozen']] or [0.0]))
log('sims done')
OFF = pd.read_parquet(AG / 'main_v2' / 'navs_gapfix.parquet')
C0chk = {c: float(np.nanmax(np.abs(NAV[('C0', c, 'freeze')].reindex(OFF[f'EXFIN|n10|{c}'].dropna().index).to_numpy() - OFF[f'EXFIN|n10|{c}'].dropna().to_numpy()))) for c in COST}

E50 = F.e50(); END = pd.Timestamp(V.END1012)
def win(nav):
    pts = list(SD[SD < END]) + [END]
    rows = [(pd.Period(a, 'M'), F._at(nav, b) / F._at(nav, a) - 1, F._at(E50, b) / F._at(E50, a) - 1, a.year) for a, b in zip(pts[:-1], pts[1:])]
    W = pd.DataFrame(rows, columns=['per', 'r', 'r50', 'yr']).set_index('per')
    return W[~W.index.duplicated(keep='last')].dropna()
PERS = {'ALL': (None, None), 'SEG1': ('2006-01', '2012-12'), 'SEG2': ('2013-01', '2018-12'), 'SEG3': ('2019-01', None),
        'SEL': (None, '2019-04'), 'VAL': ('2019-05', None), 'HOLD': ('2024-01', None)}
def tt(x):
    x = np.asarray(x, float); s = x.std(ddof=1)
    return float(x.mean() / s * math.sqrt(len(x))) if len(x) > 2 and s > 1e-15 else None
def ann(r): r = np.asarray(r, float); return float(np.prod(1 + r) ** (12 / len(r)) - 1) * 100
def mdd(r):
    nav = np.cumprod(1 + np.asarray(r, float)); pk_ = np.maximum.accumulate(np.concatenate([[1.0], nav]))[1:]
    return float((nav / pk_ - 1).min()) * 100
W = {k: win(v) for k, v in NAV.items()}
def stats(key, ref=('C0', None, 'freeze')):
    ref = (ref[0], key[1], ref[2]) if ref[1] is None else ref
    J = W[ref].join(W[key][['r']], rsuffix='_k', how='inner')
    rv, r50, rM, idx, yrs = J.r_k.to_numpy(), J.r50.to_numpy(), J.r.to_numpy(), J.index, J.yr.to_numpy()
    o = {}
    for p, (a, b) in PERS.items():
        m = np.ones(len(idx), bool)
        if a: m &= idx >= pd.Period(a, 'M')
        if b: m &= idx <= pd.Period(b, 'M')
        o[p] = dict(n=int(m.sum()), ann=ann(rv[m]), ann_0050=ann(r50[m]), gex=ann(rv[m]) - ann(r50[m]), t=tt(rv[m] - r50[m]))
    s = pd.Series(rv - r50).groupby(yrs).sum(); b3 = list(s.sort_values(ascending=False).index[:3]); m = ~np.isin(yrs, b3)
    o['ALL'].update(t_drop_best3=tt(rv[m] - r50[m]), best3=[int(x) for x in b3], mdd_win=mdd(rv), mdd_daily=float((NAV[key] / NAV[key].cummax() - 1).min() * 100),
                    diff_vs_ref_pp=float(np.mean(rv - rM) * 100), t_diff_vs_ref=tt(rv - rM), corr_ref=float(np.corrcoef(rv, rM)[0, 1]))
    return o
ST_ = {f'{k}|{c}|{eg}': stats((k, c, eg)) for k, c, eg in RUN}
ST_['ALLPIT_REAL_opt|normal|cons_main|vs_o17'] = stats(('ALLPIT_REAL_opt', 'normal', 'cons_main'), ('ALLPIT_opt_o17', 'normal', 'cons_main'))['ALL']
out = dict(table=TAB, patch_cells=dict(YL=nL, Y5=n5), C0_vs_main_v2=C0chk, counts=CNT, logs=LG, stats=ST_)
(HERE / f'r4_paper_{TAB}.json').write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str), encoding='utf-8')
def f(x, n=2): return 'NA' if x is None else f'{x:.{n}f}'
for k, v in ST_.items():
    if 'ALL' in v:
        a = v['ALL']; print(k.ljust(42), 'ann', f(a['ann']), 't', f(a['t'], 3), 'seg', f(v['SEG1']['gex']), f(v['SEG2']['gex']), f(v['SEG3']['gex']), 'hold', f(v['HOLD']['gex']),
                            'tdrop3', f(a['t_drop_best3']), 'mdd', f(a['mdd_win'], 1), f(a['mdd_daily'], 1), 'n', a['n'], 'SEL', f(v['SEL']['gex']), 'VAL', f(v['VAL']['gex']),
                            't2', f(v['SEG2']['t']), 't3', f(v['SEG3']['t']), 'dC0', f(a['diff_vs_ref_pp'], 3), 'tC0', f(a['t_diff_vs_ref']), 'corr', f(a['corr_ref'], 3))
print('vs o17', ST_['ALLPIT_REAL_opt|normal|cons_main|vs_o17'])
print('C0chk', C0chk); log('done')
