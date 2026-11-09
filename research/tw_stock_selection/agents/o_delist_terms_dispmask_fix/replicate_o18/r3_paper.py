#!/usr/bin/env python3
"""o18 題 2 獨立複製（步驟 3）：紙上 C0／U_opt × 處置遮罩（複製員自寫 orig／union／replace／union_noovl／union14）× 引擎 freeze／o17 cons／v3。
名單：o_finmask_delisted_fin_erratum/flib.VM（C0＝'base'、U_opt＝'opt'，與 o17 s4_plans 同一名單生成器）；處置遮罩以 monkeypatch ia_core.disposal_mask。
v3 結算表：r2_v7x 同一套規則（以 ia 面板陣列重算 lt／lc，另加 cz 的價格缺口規則）。統計：integrate_o17 win() 口徑（245 窗、END 2026-10-12）自寫。"""
import sys, json, math, hashlib
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
TOP = HERE.parent; AG = TOP.parent
sys.path.insert(0, str(AG / 'o_finmask_delisted_fin_erratum'))
import flib as FL
sys.path.pop(0)
sys.path.insert(0, str(AG / 'o_engine_delist_consideration'))
import cz_engine as CE
sys.path.pop(0)
K, V = FL.K, FL.V
assert CE.K is K
COLS, N, IDX = FL.COLS, FL.N, K.IDX
R1 = json.loads((HERE / 'r1_classify.json').read_text())

# ---------------- 處置遮罩 ----------------
DUF = AG / 'o_rev_avail_ds_pit_count' / 'raw' / 'disposal_official_universe.csv'
assert hashlib.sha256(DUF.read_bytes()).hexdigest().startswith('2842f45d')
DU = pd.read_csv(DUF, dtype={'code': str}, parse_dates=['start', 'end'])
ORIG = K.disposal_mask
def off_cover(d, rows, which='both'):
    t = IDX.get_loc(d); e = IDX[t + 1] if t + 1 < len(IDX) else d
    cs = rows[(rows.start <= d) & (rows.end >= d)] if which in ('both', 'sig') else rows.iloc[:0]
    ce = rows[(rows.start <= e) & (rows.end >= e)] if which in ('both', 'trd') else rows.iloc[:0]
    m = np.zeros(N, bool)
    for c in pd.concat([cs.code, ce.code]):
        if c in FL.CI: m[FL.CI[c]] = True
    return m
HI = pd.Timestamp('2021-09-23')
def mk(kind):
    cache = {}
    rows = DU[DU.overlap_proxy == False] if kind == 'union_noovl' else DU
    lo = pd.Timestamp('2014-01-01') if kind in ('replace', 'union14') else pd.Timestamp('2006-01-01')
    def f(d):
        d = pd.Timestamp(d)
        if d in cache: return cache[d]
        if lo <= d <= HI:
            m = off_cover(d, rows) if kind == 'replace' else (ORIG(d) | off_cover(d, rows))
        else: m = ORIG(d)
        cache[d] = m; return m
    return f
MASKS = {'orig': ORIG, 'union': mk('union'), 'replace': mk('replace'), 'union_noovl': mk('union_noovl'), 'union14': mk('union14')}

C0M = FL.C0MASK.copy()
def build(mode, dm):
    K.disposal_mask = dm
    try:
        S = FL.VM(lambda d: C0M, mode, 'SURV')
        pl = S.plan(10, 'official'); t0, t1, sd = S.bounds()
    finally:
        K.disposal_mask = ORIG
    return dict(plan=[(int(p[0]), np.array(p[1], int)) for p in pl], sig=[pd.Timestamp(p[3]) for p in pl], t0=t0, t1=t1, sd=pd.DatetimeIndex(sd))

LISTS = {}
for lk in MASKS: LISTS[('C0', lk)] = build('base', MASKS[lk])
for lk in ('orig', 'union'): LISTS[('U_opt', lk)] = build('opt', MASKS[lk])
print('lists built', flush=True)
SD = LISTS[('C0', 'orig')]['sd']
for k, B in LISTS.items(): assert (B['sd'] == SD).all(), k

# 名單核對與槽位
PK = {k: {s: set(COLS[j] for j in p[1]) for s, p in zip(B['sig'], B['plan'])} for k, B in LISTS.items()}
def changed(k, a='2000-01-01', b='2100-01-01'):
    base = PK[(k[0], 'orig')]
    return int(sum(len(base[d] - PK[k][d]) for d in base if pd.Timestamp(a) <= d <= pd.Timestamp(b)))
slots = {}
c0 = PK[('C0', 'orig')]
for which in ('sig', 'trd', 'both'):
    hit = sorted(f'{d.date()} {c}' for d in c0 if pd.Timestamp('2014-01-01') <= d <= HI for c in c0[d] if off_cover(d, DU, which)[FL.CI[c]] and not ORIG(d)[FL.CI[c]])
    slots[which] = hit
chg = {f'{k[0]}|{k[1]}': changed(k) for k in LISTS if k[1] != 'orig'}
chg_detail = {f'{k[0]}|{k[1]}': sorted(f'{d.date()} -{sorted(PK[(k[0], "orig")][d] - PK[k][d])} +{sorted(PK[k][d] - PK[(k[0], "orig")][d])}'
                                         for d in PK[k] if PK[k][d] != PK[(k[0], 'orig')][d]) for k in LISTS if k[1] != 'orig'}
# 對照 main_v2 C0
OFF = pd.read_parquet(AG / 'main_v2' / 'navs_gapfix.parquet')

# ---------------- v3 結算表（ia 面板） ----------------
E2 = pd.read_csv(AG / 'o_engine_delist_consideration' / 'events_v2.csv', dtype=str)
for c in ('cash', 'ratio', 'pref_ratio', 'pref_value'): E2[c] = pd.to_numeric(E2[c])
O17V = set(E2[E2.verified == '已核實（新聞）'].code); MINE = set(R1['text']) | O17V
TR, Cn, On, DATES = K.TRADED, K.Cn, K.On, IDX.values
def _idx(d): return int(np.searchsorted(DATES, np.datetime64(pd.Timestamp(d))))
def _sf(sj, d):
    w = np.flatnonzero(TR[:, sj] & np.isfinite(On[:, sj])); w = w[w >= _idx(d)]
    return int(w[0]) if len(w) else None
def v3_settle(bound):
    out = {}
    for r in E2[E2.code.isin(MINE)].itertuples():
        j = FL.CI.get(r.code)
        if j is None: continue
        ltU = pd.Timestamp(r.last_trade)
        term = pd.Timestamp(r.terminate_date) if isinstance(r.terminate_date, str) else ltU + pd.Timedelta(days=14)
        w = np.flatnonzero(TR[:, j]); w = w[IDX[w] <= max(term, ltU)]
        if not len(w): continue
        lt = int(w[-1]); lc = float(Cn[lt, j]); ti = _idx(term); pay = min(ti + 20, len(IDX) - 1)
        if lt + 1 >= len(IDX): continue
        de = R1['delta_pre2013'] if ltU < pd.Timestamp('2013-01-01') else R1['delta']
        db = {'main': 0.0, 'lo': -de, 'hi': de}[bound]
        ru = dict(code=r.code, lt=lt, trig=lt + 1, lc=lc, pay=pay, term=ti, cash_0050=False, parts=[]); P = ru['parts']
        if IDX[lt] < ltU - pd.Timedelta(days=7):
            ru['type'] = 'price_gap'; P.append(dict(kind='fixed_sell', per_share=lc, rel=lt + 1)); out.setdefault(j, []).append(ru); continue
        sj = FL.CI.get(r.successor) if isinstance(r.successor, str) else None
        sf = _sf(sj, r.succ_first_trade) if (sj is not None and isinstance(r.succ_first_trade, str)) else None
        if r.code in O17V:
            ru['type'] = 'v_' + r.type
            if r.type == 'cash': P.append(dict(kind='cash', per_share=float(r.cash), rel=pay))
            else:
                if sf is None: P.append(dict(kind='cash', per_share=lc, rel=pay))
                else:
                    if np.isfinite(r.ratio): P.append(dict(kind='stock', succ=sj, ratio=float(r.ratio), rel=sf, per_share_window=lc))
                    if np.isfinite(r.cash): P.append(dict(kind='cash', per_share=float(r.cash), rel=pay))
                    if np.isfinite(r.pref_ratio): P.append(dict(kind='fixed_sell', per_share=float(r.pref_ratio * r.pref_value), rel=sf))
            out.setdefault(j, []).append(ru); continue
        T = R1['text'][r.code]; st = T['status']; t = T['terms']; mkt = R1['market'][r.code]
        if st == 'VERIFIED_TEXT':
            ru['type'] = 'VERIFIED_' + t['kind']
            if t['kind'] == 'cash': P.append(dict(kind='cash', per_share=t['cash'], rel=pay))
            else:
                if sj is None:
                    sj = FL.CI.get(t['succ']); sf = _sf(sj, term) if sj is not None else None
                if sf is None: P.append(dict(kind='cash', per_share=lc, rel=pay))
                else:
                    P.append(dict(kind='stock', succ=sj, ratio=t['ratio'], rel=sf, per_share_window=lc))
                    if t['kind'] == 'mixed': P.append(dict(kind='cash', per_share=t['cash'], rel=pay))
                    if t['kind'] == 'mixed_pref' and bound == 'hi': P.append(dict(kind='fixed_sell', per_share=t['pref'] * 10.0, rel=sf))
        elif t['kind'] == 'conv_newsucc':
            sj = FL.CI['3705']; sf = _sf(sj, '2011-01-03'); ru['type'] = 'CONV_NEWSUCC'
            P.append(dict(kind='stock', succ=sj, ratio=lc * (1 + db) / float(On[sf, sj]), rel=sf, per_share_window=lc))
        elif mkt['market'] == 'MARKET_TRACK' and sf is not None:
            ru['type'] = st + '+TRACK'; P.append(dict(kind='stock', succ=sj, ratio=mkt['ratio'] * (1 + db), rel=sf, per_share_window=lc))
        elif st == 'SEARCHED_NOT_FOUND' and mkt['market'] == 'MARKET_NONE':
            ru['type'] = 'SNF'
            P.append(dict(kind='fixed_sell', per_share=lc, rel=lt + 1, zero=True) if bound == 'lo' else dict(kind='cash', per_share=lc, rel=pay))
        else:
            ru['type'] = 'fixed'; P.append(dict(kind='cash', per_share=lc * (1 + db), rel=pay))
        out.setdefault(j, []).append(ru)
    rest = CE.build_settlement(E2[~E2.code.isin(MINE)], 'main', d=0.057)
    for j, L in rest.items():
        for ru in L: ru['type'] = 'REST_' + ru['type']
        out.setdefault(j, []).extend(L)
    return out
ST = {'cons_main': CE.build_settlement(E2, 'main', d=0.057), 'cons_lo': CE.build_settlement(E2, 'lo', d=0.057), 'cons_hi': CE.build_settlement(E2, 'hi', d=0.057)}
for b in ('main', 'lo', 'hi'): ST['v3_' + b] = v3_settle(b)

COST = {'normal': K.rates(0.001), 'stress': K.rates('stress')}
def sim(k, c, eg):
    B = LISTS[k]; bb, ss = COST[c]
    if eg == 'freeze': return CE.simulate(B['plan'], bb, ss, B['t0'], B['t1'], mode='freeze')
    return CE.simulate(B['plan'], bb, ss, B['t0'], B['t1'], mode='consideration', settle=ST[eg])

E50r = pd.read_csv('/home/user/jonatw/finlab-txd/data/raw/etf_0050.csv', parse_dates=['date']).set_index('date')['adj'].sort_index()
E50r = E50r[~E50r.index.duplicated(keep='last')]
END = pd.Timestamp('2026-10-12')
def at(s, d):
    s = s.dropna(); k = s.index.searchsorted(d, side='right') - 1
    return float(s.iloc[k]) if k >= 0 else np.nan
def win(nav):
    pts = list(SD[SD < END]) + [END]
    rows = [(pd.Period(a, 'M'), at(nav, b) / at(nav, a) - 1, at(E50r, b) / at(E50r, a) - 1, a.year) for a, b in zip(pts[:-1], pts[1:])]
    W = pd.DataFrame(rows, columns=['per', 'r', 'r50', 'yr']).set_index('per')
    return W[~W.index.duplicated(keep='last')].dropna()
def ann(r): r = np.asarray(r, float); return float(np.prod(1 + r) ** (12 / len(r)) - 1) * 100
def tt(x): x = np.asarray(x, float); return float(x.mean() / x.std(ddof=1) * math.sqrt(len(x)))
SEG = {'ALL': (None, None), 'SEG1': ('2006-01', '2012-12'), 'SEG2': ('2013-01', '2018-12'), 'SEG3': ('2019-01', None)}
def stats(W):
    o = {'n': len(W)}
    for s, (a, b) in SEG.items():
        m = np.ones(len(W), bool)
        if a: m &= W.index >= pd.Period(a, 'M')
        if b: m &= W.index <= pd.Period(b, 'M')
        x = W[m]; o[s] = dict(ann=ann(x.r), gex=ann(x.r) - ann(x.r50), t=tt(x.r - x.r50))
    return o

RUNS = [(('C0', lk), c, 'freeze') for lk in MASKS for c in COST]
RUNS += [(('C0', 'orig'), 'normal', e) for e in ('v3_main', 'v3_lo', 'v3_hi', 'cons_main')]
RUNS += [(('U_opt', lk), c, e) for lk in ('orig', 'union') for e in ('freeze', 'cons_main', 'cons_lo', 'cons_hi', 'v3_main', 'v3_lo', 'v3_hi') for c in (('normal', 'stress') if e in ('freeze', 'v3_main') else ('normal',))]
NAV = {}; RES = {}; SETL = {}
for k, c, e in RUNS:
    nav, lg = sim(k, c, e); NAV[(k, c, e)] = nav
    W = win(nav); RES[f'{k[0]}|{k[1]}|{c}|{e}'] = stats(W)
    SETL[f'{k[0]}|{k[1]}|{c}|{e}'] = sorted({(s['code'], s['type']) for s in lg.get('settled', [])})
ref = OFF['EXFIN|n10|normal'].dropna(); n0 = NAV[(('C0', 'orig'), 'normal', 'freeze')]
c0diff = float(np.nanmax(np.abs(n0.reindex(ref.index).to_numpy() - ref.to_numpy())))
# 逐年差（union 對 orig，C0 一般 freeze）：每年窗報酬連乘差
Wo = win(n0); Wu = win(NAV[(('C0', 'union'), 'normal', 'freeze')])
yo = Wo.groupby('yr').r.apply(lambda x: np.prod(1 + x) - 1) * 100; yu = Wu.groupby('yr').r.apply(lambda x: np.prod(1 + x) - 1) * 100
yd = (yu - yo); ys = (Wu.r - Wo.r).groupby(Wo.yr).sum() * 100
out = dict(c0_vs_main_v2=c0diff, slots_2014_2021=dict(n_sig=len(slots['sig']), n_trd=len(slots['trd']), n_both=len(slots['both']), sig=slots['sig'], trd=slots['trd']),
           changed=chg, changed_detail=chg_detail, stats=RES, settled=SETL,
           yearly_union_vs_orig=dict(compound={int(a): float(b) for a, b in yd.items() if abs(b) > 1e-9}, sum=float(yd.sum()), abssum=float(yd.abs().sum()),
                                     winsum={int(a): float(b) for a, b in ys.items() if abs(b) > 1e-9}, winsum_sum=float(ys.sum()), winsum_abs=float(ys.abs().sum())))
(HERE / 'r3_paper.json').write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str))
print('c0 vs main_v2', c0diff, 'slots', len(slots['sig']), len(slots['trd']), 'changed', chg)
for k, v in RES.items(): print(k, round(v['ALL']['ann'], 3), round(v['ALL']['t'], 3), [round(v[s]['gex'], 2) for s in ('SEG1', 'SEG2', 'SEG3')])
print('yearly', out['yearly_union_vs_orig'])
