#!/usr/bin/env python3
"""integrate_o4 回測：照 results.json 的 PREREG 重跑 AB、組合、統計、SEL 機械選版、判定，寫回 RESULTS（只寫一次）。
只讀 import o_v4_revenue_patch_official/v4lib.py（→ main_v2/v2_core、o_infra_v3_rebuild/o3_core、o_disposal_rejudge/drj_core）。
成本、成交、漲停遞補、停牌凍結、X 出清全部用 v4lib／o3_core，不重寫；本檔只新寫袖子混合（固定比例或曝險表）與統計彙整。"""
import sys, json, time, math, hashlib
from pathlib import Path
HERE = Path(__file__).resolve().parent
AG = HERE.parent
V4D = AG / 'o_v4_revenue_patch_official'
sys.path.insert(0, str(V4D))
import v4lib as L
sys.path.pop(0)
import numpy as np, pandas as pd
K, V = L.K, L.V
T0_ = time.time()
def log(*a): print(f'[{time.time() - T0_:6.0f}s]', *a, flush=True)

RJ = HERE / 'results.json'
RES = json.loads(RJ.read_text(encoding='utf-8'))
assert hashlib.sha256(json.dumps(RES['PREREG'], ensure_ascii=False, sort_keys=True).encode()).hexdigest() == RES['PREREG_sha256'], 'PREREG 被改過'
assert 'RESULTS' not in RES, 'RESULTS 已存在：只跑一次'
for k, f in (('navs_v4.parquet_sha256', V4D / 'navs_v4.parquet'), ('v4lib.py_sha256', V4D / 'v4lib.py'), ('data_freeze_v4.json_sha256', V4D / 'data_freeze_v4.json'),
             ('main_v2_navs_gapfix_sha256', AG / 'main_v2' / 'navs_gapfix.parquet'), ('largecap_navs_sha256', AG / 'o_largecap_leg' / 'navs.parquet')):
    assert L.sha(f) == RES['PREREG']['inputs'][k], k
L.check_freeze()
XR = L.setup_px()
YFX, Y5F = L.load_rfix()
DMT, DMS = L.disp_mask_fn('trade'), L.disp_mask_fn('signal')
DMU = lambda d: DMT(d) | DMS(d)
NAV4 = pd.read_parquet(V4D / 'navs_v4.parquet')
V4R = json.loads((V4D / 'results.json').read_text(encoding='utf-8'))['RESULTS']
A_, B_ = '2006-04-01', '2026-10-12'
SLIPS = [round(0.0010 + 0.0002 * i, 4) for i in range(16)]

def mk(dm=DMT, **kw): return L.V4(YFX, Y5F, dm, **kw)
SPEC = {'V4': DMT, 'F_SIG': DMS, 'F_DUNION': DMU}

# ---------------- BASE 計畫 ----------------
BASEOBJ, BASEPL, BASENAV, BASELOG = {}, {}, {}, {}
t0 = t1 = sd = None
for k, dm in SPEC.items():
    S = mk(dm); ta, tb, s_ = S.bounds(); pl = S.plan(10, 'official')
    if t0 is None: t0, t1, sd = ta, tb, s_
    assert (ta, tb) == (t0, t1) and s_ == sd
    BASEOBJ[k], BASEPL[k] = S, pl
    for c in ('normal', 'stress'):
        BASENAV[f'{k}|{c}'], BASELOG[f'{k}|{c}'] = L.sim_plan(pl, t0, t1, c, XR)
    assert [p[0] for p in pl] == [p[0] for p in BASEPL['V4']], k
EXEC = [p[0] for p in BASEPL['V4']]
VAL = {}
for k, ref in (('V4', 'V4|normal'), ('F_SIG', 'V4_SIG|normal'), ('V4s', 'V4|stress')):
    a = BASENAV[(k[:-1] if k == 'V4s' else k) + ('|stress' if k == 'V4s' else '|normal')]
    r = NAV4[ref].reindex(a.index)
    VAL[f'BASE_{k}_maxrel'] = float((a / r - 1).abs().max())
    assert VAL[f'BASE_{k}_maxrel'] <= 1e-9, (k, VAL)
log('base ok', VAL)

# ---------------- AB（mc 40）----------------
def ab(dm):
    costs = ['normal', 'stress'] + [('slip', s) for s in SLIPS]
    rets = {str(c): [] for c in costs}; ewr = []; pls = []
    for seed in range(40):
        S = mk(dm, unavail=V.make_unavail('mc', seed)); pl = S.plan(10, 'official'); pls.append(pl)
        for c in costs:
            nav, _ = L.sim_plan(pl, t0, t1, c, XR); rets[str(c)].append(nav.pct_change())
        ew, _ = K.simulate([(p[0], p[2]) for p in pl], 0, 0, t0, t1); ewr.append(ew.pct_change())
    mean_nav = lambda L_: (1 + pd.concat(L_, axis=1).mean(axis=1).fillna(0)).cumprod()
    return {c: mean_nav(v) for c, v in rets.items()}, mean_nav(ewr), pls
ABN, ABEW, ABPL = {}, {}, {}
for k, dm in SPEC.items():
    ABN[k], ABEW[k], ABPL[k] = ab(dm)
    log('AB', k, round(L.seg(ABN[k]['normal'], ABEW[k], sd, A_, B_)['CAGR'], 3))
for c in ('normal', 'stress', 'EW'):
    a = ABN['V4'][c] if c != 'EW' else ABEW['V4']
    r = NAV4[f'AB_V4|{c}'].reindex(a.index)
    VAL[f'AB_V4_{c}_maxrel'] = float((a / r - 1).abs().max())
    assert VAL[f'AB_V4_{c}_maxrel'] <= 1e-9, VAL
st = L.seg(ABN['V4']['normal'], ABEW['V4'], sd, A_, B_)
assert abs(st['CAGR'] - V4R['table']['AB_V4|normal']['ALL']['CAGR']) < 1e-6 and abs(st['t_0050'] - V4R['table']['AB_V4|normal']['ALL']['t_0050']) < 1e-6
log('AB ok', VAL)

# ---------------- 袖子混合 ----------------
IX = K.IDX[t0:t1 + 1]
EXI = set(i - t0 for i in EXEC if t0 <= i <= t1)
R_E50 = K.E50.reindex(IX).pct_change().fillna(0).to_numpy()
J2330 = K.CI['2330']
R_2330 = (K.GROW[t0:t1 + 1, J2330] - 1.0).copy(); R_2330[0] = 0.0
R_CASH = np.zeros(len(IX))
def sleeve(navA, rB, wA_fn, cost, kindB, ix=IX, exi=EXI):
    """navA：主策略腿淨值（已扣自身成本）；rB：另一腿日報酬；wA_fn(i)→成交日 i 主策略腿目標比例；cost＝(買, 賣) 股票費率；
    kindB：'etf'（賣方＝買方＋0.1%）、'stock'（同股票費率）、'cash'（0）。成交日收盤調回比例，移轉扣成本。"""
    buy, sell = cost
    bB, sB = {'etf': (buy, buy + 0.001), 'stock': (buy, sell), 'cash': (0.0, 0.0)}[kindB]
    ra = navA.reindex(ix).pct_change().fillna(0).to_numpy()
    w0 = wA_fn(0); a, b = w0, 1 - w0; out = np.empty(len(ix)); tc = 0.0
    for i in range(len(ix)):
        if i > 0: a *= 1 + ra[i]; b *= 1 + rB[i]
        if i in exi and i > 0:
            w = wA_fn(i); tot = a + b; ta = w * tot
            if a < ta: m = ta - a; c = m * (sB + buy)
            else: m = a - ta; c = m * (sell + bB)
            tc += c / tot; tot -= c; a, b = w * tot, (1 - w) * tot
        out[i] = a + b
    return pd.Series(out, index=ix)

# 驗證：重現 o_largecap_leg 的 MAIN70_0050_30|normal
LC = pd.read_parquet(AG / 'o_largecap_leg' / 'navs.parquet')
lcm = LC['MAIN|normal'].dropna()
ixl = lcm.index; it0 = K.IDX.get_loc(ixl[0])
exl = set(i - it0 for i in EXEC if it0 <= i <= it0 + len(ixl) - 1)
rE = K.E50.reindex(ixl).pct_change().fillna(0).to_numpy()
mix_lc = sleeve(lcm, rE, lambda i: 0.7, K.rates(0.001), 'etf', ix=ixl, exi=exl)
ref = LC['MAIN70_0050_30|normal'].reindex(ixl)
VAL['sleeve_vs_largecap_MAIN70_0050_30_final_rel'] = float(mix_lc.iloc[-1] / ref.iloc[-1] - 1)
assert abs(VAL['sleeve_vs_largecap_MAIN70_0050_30_final_rel']) <= 1e-6, VAL
r23 = (K.GROW[it0:it0 + len(ixl), J2330] - 1.0).copy(); r23[0] = 0
mix23 = sleeve(lcm, r23, lambda i: 0.7, K.rates(0.001), 'stock', ix=ixl, exi=exl)
VAL['sleeve2330_vs_largecap_TSMC30_single_account'] = {'final_rel': float(mix23.iloc[-1] / LC['TSMC30|normal'].reindex(ixl).iloc[-1] - 1),
    'CAGR_sleeve': L.seg(mix23, mix23, sd, A_, B_)['CAGR'], 'CAGR_single': L.seg(LC['TSMC30|normal'].reindex(ixl), mix23, sd, A_, B_)['CAGR']}
log('sleeve ok', VAL)

# 新高家數 B（訊號日 d，前 300 流動性池中 52 週比 ≥ 0.95 的檔數）
S0 = BASEOBJ['V4']
BREADTH = {}
for p in BASEPL['V4']:
    d = p[3]; cs = S0.cross_section(d); t = cs['t']
    BREADTH[p[0]] = int((cs['top'] & np.isfinite(S0.S52[t]) & (S0.S52[t] >= 0.95)).sum())
EXPO = {i - t0: min(1.0, BREADTH[i] / 30.0) for i in EXEC}
def w_breadth(i):
    if i == 0: return EXPO.get(0, 1.0)
    return EXPO[i]

LEGS = {'L_E50_20': (0.8, R_E50, 'etf'), 'L_E50_30': (0.7, R_E50, 'etf'), 'L_E50_50': (0.5, R_E50, 'etf'),
        'L_2330_20': (0.8, R_2330, 'stock'), 'L_2330_30': (0.7, R_2330, 'stock'), 'L_2330_50': (0.5, R_2330, 'stock')}
def build(name, mainnav, cost):
    if name in ('V4', 'F_SIG', 'F_DUNION'): return mainnav[name]
    if name == 'O_BREADTH30': return sleeve(mainnav['V4'], R_CASH, w_breadth, cost, 'cash')
    w, rB, kind = LEGS[name]
    return sleeve(mainnav['V4'], rB, lambda i: w, cost, kind)
ORDER = ['V4', 'F_SIG', 'F_DUNION', 'O_BREADTH30', 'L_E50_20', 'L_E50_30', 'L_E50_50', 'L_2330_20', 'L_2330_30', 'L_2330_50']
NAV = {}
for c in ('normal', 'stress'):
    mn = {k: ABN[k][c] for k in SPEC}
    for nm in ORDER: NAV[f'{nm}|{c}'] = build(nm, mn, L.cost_rates(c))
    mb = {k: BASENAV[f'{k}|{c}'] for k in SPEC}
    for nm in ORDER: NAV[f'BASE_{nm}|{c}'] = build(nm, mb, L.cost_rates(c))
SLIPNAV = {}
for s in SLIPS:
    mn = {k: ABN[k][str(('slip', s))] for k in SPEC}
    for nm in ORDER: SLIPNAV[(nm, s)] = build(nm, mn, L.cost_rates(('slip', s)))
log('combos built')

# ---------------- 統計 ----------------
E50 = K.E50
EWB = ABEW['V4']
MV2 = pd.read_parquet(AG / 'main_v2' / 'navs_gapfix.parquet')['EXFIN|n10|normal'].reindex(IX)
O3R = NAV4['O3REP|normal'].reindex(IX)
def xs(nav, a=A_, b=B_): return L.excess(nav, E50, sd, a, b)
def wwin(nav, a, b):
    w = L.win_ret(nav, sd); return w[(w.index >= pd.Timestamp(a)) & (w.index <= pd.Timestamp(b))].dropna()
def paired(a, b, x0=A_, x1=B_):
    d = (L.win_ret(a, sd) - L.win_ret(b.reindex(a.index), sd)); d = d[(d.index >= pd.Timestamp(x0)) & (d.index <= pd.Timestamp(x1))].dropna()
    nz = d[d.abs() > 1e-12]
    o = dict(mean_pp=float(d.mean() * 100), t=L.tstat(d), n=int(len(d)), n_nonzero=int(len(nz)))
    if len(nz) >= 2:
        big = d.abs().idxmax(); o['drop_maxabs_t'] = L.tstat(d.drop(big)); o['maxabs_month'] = str(big.date()); o['maxabs_pp'] = float(d[big] * 100)
    return o
def corr(a, b, x0=A_, x1=B_, excess=False):
    ra, rb = wwin(a, x0, x1), wwin(b.reindex(a.index), x0, x1)
    if excess:
        e = wwin(E50.reindex(a.index), x0, x1); ra, rb = ra - e, rb - e
    z = pd.concat([ra, rb], axis=1).dropna()
    return float(z.iloc[:, 0].corr(z.iloc[:, 1]))
def cal_t(nav):
    me = nav.groupby(nav.index.to_period('M')).last(); be_ = E50.reindex(nav.index).groupby(nav.index.to_period('M')).last()
    r = (me.pct_change() - be_.pct_change()).dropna()
    return L.tstat(r[(r.index >= pd.Period('2006-04', 'M')) & (r.index <= pd.Period('2026-10', 'M'))].to_numpy())
def endpoint(nav):
    x = xs(nav); o = {'t_all': L.tstat(x), 'drop_last1': L.tstat(x.iloc[:-1])}
    xsrt = x.sort_values(ascending=False)
    for k in (1, 3, 5): o[f'drop_best{k}'] = L.tstat(x.drop(xsrt.index[:k]))
    yr = x.groupby(x.index.year).sum(); by = int(yr.idxmax())
    o['best_year'] = by; o['drop_best_year'] = L.tstat(x[x.index.year != by])
    o['to_2025_end'] = L.tstat(x[x.index <= pd.Timestamp('2025-12-31')]); o['cal_month_t'] = cal_t(nav)
    return o
def yearly(nav):
    r = nav.pct_change(); r50 = E50.reindex(nav.index).pct_change(); msk = r.index >= pd.Timestamp(A_)
    return {int(y): {'s': float((1 + g.fillna(0)).prod() - 1) * 100, '0050': float((1 + r50[g.index].fillna(0)).prod() - 1) * 100} for y, g in r[msk].groupby(r[msk].index.year)}
def breakeven(nm):
    pts = [(s, L.seg(SLIPNAV[(nm, s)], EWB, sd, A_, B_)['t_0050']) for s in SLIPS]
    def cross(th):
        for (s1, a), (s2, b) in zip(pts, pts[1:]):
            if a >= th > b: return s1 + (s2 - s1) * (a - th) / (a - b)
        return '<0.10%（一般成本已低於門檻）' if pts[0][1] < th else float('inf')
    return {'slip_at_t2.3': cross(2.3), 'slip_at_t2.0': cross(2.0), 't_at_slip0.10': pts[0][1], 't_at_slip0.20': pts[5][1]}
PK = ('S1', 'S2', 'S3', 'HOLD', 'SEL', 'VAL', 'ALL')
def stats(nm, full=True):
    nav = NAV[f'{nm}|normal']; o = {'periods': {}}
    for p in PK:
        a, b = L.PER[p]; s_ = L.seg(nav, EWB, sd, a, b)
        s_['TE_ann'] = s_['sd_pp'] * math.sqrt(12)
        s_['vs_AB_V4'] = paired(nav, NAV['V4|normal'], a, b); s_['vs_MAIN_V2'] = paired(nav, MV2, a, b)
        s_['corr_AB_V4'] = corr(nav, NAV['V4|normal'], a, b); s_['corr_excess_AB_V4'] = corr(nav, NAV['V4|normal'], a, b, excess=True)
        o['periods'][p] = s_
    o['stress'] = {p: L.seg(NAV[f'{nm}|stress'], EWB, sd, *L.PER[p]) for p in ('SEL', 'VAL', 'HOLD', 'ALL')}
    o['BASE'] = {p: L.seg(NAV[f'BASE_{nm}|normal'], NAV4['V4|EW'].reindex(IX), sd, *L.PER[p]) for p in ('SEL', 'VAL', 'HOLD', 'ALL')}
    o['BASE_vs_V4'] = paired(NAV[f'BASE_{nm}|normal'], NAV['BASE_V4|normal'])
    if full:
        o['endpoint'] = endpoint(nav)
        tA = o['periods']['ALL']['t_0050']; p1 = L.p_one_sided(tA)
        o['p_one_sided'] = p1; o['bonferroni_m10'] = min(1.0, 10 * p1); o['bonferroni_m19'] = min(1.0, 19 * p1)
        o['breakeven'] = breakeven(nm)
        y = yearly(nav); o['yearly'] = y; o['years_lose_to_0050'] = [k for k, v in y.items() if v['s'] < v['0050']]
        e50 = o['periods']['ALL']['E50_CAGR']; o['live_expectation_ALL'] = e50 + 0.5 * (o['periods']['ALL']['CAGR'] - e50)
    return o
ST = {nm: stats(nm) for nm in ORDER}
REF = {'MAIN_V2': {p: L.seg(MV2, NAV4['V4|EW'].reindex(IX), sd, *L.PER[p]) for p in PK},
       'O3REP': {p: L.seg(O3R, NAV4['O3REP|EW'].reindex(IX), sd, *L.PER[p]) for p in PK},
       'AB_V4_vs_MAIN_V2': {p: paired(NAV['V4|normal'], MV2, *L.PER[p]) for p in PK},
       'AB_V4_corr_MAIN_V2': corr(NAV['V4|normal'], MV2), 'BASE_V4_vs_MAIN_V2': paired(NAV['BASE_V4|normal'], MV2)}
log('stats done')

# ---------------- 處置口徑名單差 ----------------
def picks(pl): return {p[3]: set(K.COLS[j] for j in p[1]) for p in pl}
PB = {k: picks(BASEPL[k]) for k in SPEC}
DIFF = {}
for k in ('F_SIG', 'F_DUNION'):
    mm = [d for d in PB['V4'] if PB[k][d] != PB['V4'][d]]
    DIFF[k] = {'BASE_months_changed': len(mm), 'BASE_slots_changed': int(sum(len(PB[k][d] - PB['V4'][d]) for d in mm)),
               'detail': {str(d.date()): {'in': sorted(PB[k][d] - PB['V4'][d]), 'out': sorted(PB['V4'][d] - PB[k][d])} for d in mm},
               'AB_seed_months_changed_mean': float(np.mean([sum(set(a[1].tolist()) != set(b[1].tolist()) for a, b in zip(ABPL[k][s], ABPL['V4'][s])) for s in range(len(ABPL[k]))]))}
BR = pd.Series({str(K.IDX[i].date()): BREADTH[i] for i in EXEC})
BRS = {'mean_expo': float(np.mean(list(EXPO.values()))), 'n_months_expo_lt1': int(sum(v < 1 for v in EXPO.values())),
       'n_months_expo_lt_0.5': int(sum(v < 0.5 for v in EXPO.values())), 'min_B': int(BR.min()),
       'expo_lt1_by_year': pd.Series([s[:4] for s, v in zip(BR.index, BR.values) if v < 30]).value_counts().sort_index().to_dict()}

# ---------------- 選版（只用 SEL）----------------
b_sel = ST['V4']['periods']['SEL']
elig = []
for nm in ('F_SIG', 'F_DUNION', 'O_BREADTH30', 'L_2330_20', 'L_2330_30', 'L_2330_50'):
    s_ = ST[nm]['periods']['SEL']
    ok = (s_['t_0050'] > b_sel['t_0050']) and (b_sel['CAGR'] - s_['CAGR'] <= 1.0)
    elig.append({'name': nm, 'SEL_t': s_['t_0050'], 'SEL_CAGR_loss_pp': b_sel['CAGR'] - s_['CAGR'], 'eligible': bool(ok)})
ok_ = [e for e in elig if e['eligible']]
chosen = max(ok_, key=lambda e: e['SEL_t'])['name'] if ok_ else 'AB_V4（維持）'

# ---------------- 判定 ----------------
bA = ST['V4']['periods']
def judge(nm):
    if nm.startswith('L_E50'): return '配置選項'
    P_ = ST[nm]['periods']; pa = P_['ALL']['vs_AB_V4']
    A = pa['t'] >= 1.0 and P_['SEL']['vs_AB_V4']['mean_pp'] > 0 and P_['VAL']['vs_AB_V4']['mean_pp'] > 0
    B = (P_['ALL']['t_0050'] - bA['ALL']['t_0050'] >= 0.2 and P_['SEL']['t_0050'] > bA['SEL']['t_0050'] and P_['VAL']['t_0050'] > bA['VAL']['t_0050']
         and bA['ALL']['CAGR'] - P_['ALL']['CAGR'] <= 1.0 and P_['ALL']['maxDD'] >= bA['ALL']['maxDD'])
    if A or B: raw = '可用'
    elif pa['t'] > -1 and bA['ALL']['CAGR'] - P_['ALL']['CAGR'] <= 2.0 and (P_['SEL']['t_0050'] > bA['SEL']['t_0050'] or P_['VAL']['t_0050'] > bA['VAL']['t_0050']): raw = '待觀察'
    else: raw = '不可用'
    return {'raw': raw, 'path_A': bool(A), 'path_B': bool(B), 'final': '待觀察' if raw == '可用' else raw}
JUD = {nm: judge(nm) for nm in ORDER if nm != 'V4'}

def r(o):
    if isinstance(o, dict): return {str(k): r(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)): return [r(v) for v in o]
    if isinstance(o, (float, np.floating)): return (None if np.isnan(o) else ('inf（格點 0.40% 內未跌破）' if o > 0 else '-inf')) if not np.isfinite(o) else round(float(o), 4)
    if isinstance(o, (np.integer,)): return int(o)
    return o
RES['RESULTS'] = r({'written_at': pd.Timestamp.now(tz='UTC').isoformat(), 'validation': VAL, 'stats': ST, 'reference': REF, 'disposal_diff': DIFF,
                    'breadth': BRS, 'selection': {'candidates': elig, 'chosen': chosen}, 'judgement': JUD,
                    'turnover_BASE': {k: BASELOG[f'{k}|normal']['turnover'] for k in SPEC}})
RJ.write_text(json.dumps(RES, ensure_ascii=False, indent=1), encoding='utf-8')
NV = pd.DataFrame({k: v for k, v in NAV.items()})
NV['MAIN_V2|normal'] = MV2
NV.to_parquet(HERE / 'navs.parquet')
log('done', chosen, {k: (v['final'] if isinstance(v, dict) else v) for k, v in JUD.items()})
