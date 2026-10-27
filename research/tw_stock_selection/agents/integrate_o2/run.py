#!/usr/bin/env python3
"""integrate_o2 回測：照 results.json 的 PREREG 跑全部組合、統計、SEL 機械選版、判定，寫回 results.json 的 RESULTS（只寫一次）。
只讀 import o_rev_survivorship/osurv_core.py 與 o_capacity_tsmc_gap/ocore.py。"""
import sys, json, time, hashlib
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'o_rev_survivorship'))
import osurv_core as OS
sys.path.pop(0)
sys.path.insert(0, str(HERE.parent / 'o_capacity_tsmc_gap'))
import ocore as O
sys.path.pop(0)
import numpy as np, pandas as pd
V, K = OS.V, OS.K
assert O.K is K and O.V is V
T0 = time.time()
def log(*a): print(f'[{time.time()-T0:5.0f}s]', *a, flush=True)

RJ = HERE / 'results.json'
R = json.loads(RJ.read_text(encoding='utf-8'))
assert hashlib.sha256(json.dumps(R['PREREG'], ensure_ascii=False, sort_keys=True).encode()).hexdigest() == R['PREREG_sha256'], 'PREREG 被改過'
assert 'RESULTS' not in R, 'RESULTS 已存在：只跑一次'
FSET = OS.FOREIGN
J = O.J2330

# ---------------- 股池與名單 ----------------
YO, Y5 = OS.load_yl('ORIG'); YD, _ = OS.load_yl('R_DEL'); YA, _ = OS.load_yl('R_ALL')
SO = OS.V2B('gapfix', YO, Y5, exfin=True)
SR = OS.V2B('gapfix', YD, Y5, exfin=True)
SN = OS.V2B('gapfix', YD, Y5, exfin=True, gap=OS.gapsets('R_DEL'), pfix=0.5)
SA = OS.V2B('gapfix', YA, Y5, exfin=True)
for S in (SO, SN, SA): assert S.sig_days == SR.sig_days and S.end_day == SR.end_day
t0, t1, sd = SR.bounds()
P = O.periods(sd)
lastfix = max(d for d in sd if d <= pd.Timestamp('2019-12-31')); firstfix = min(d for d in sd if d >= pd.Timestamp('2015-04-01'))
P['FIX'] = ('2015-04-01', '2019-12-31', str((firstfix + pd.Timedelta(days=1)).date()), str(sd[sd.index(lastfix) + 1].date()))
KEYS = ('SEL', 'VAL', 'VAL_A', 'HOLD', 'S1', 'S2', 'S3', 'FIX', 'ALL')

def plan_cap(S, n, cap):
    out = []; held = set(); nskip = 0
    for d in S.sig_days:
        cs = S.cross_section(d); el = cs['el']
        if el.sum() < 50: continue
        e = cs['e']; gu = K.gap_up_open(e)
        cand = S.order(cs, 'official')[:int(el.sum())]
        pick = []; nky = 0
        for j in cand:
            if gu[j] and j not in held: continue
            ky = K.COLS[j] in FSET
            if ky and nky >= cap: nskip += 1; continue
            pick.append(j); nky += int(ky)
            if len(pick) == n: break
        out.append((e, np.array(pick, int), np.where(el)[0], d))
        held = set(pick)
    return out, nskip

PL = {'MAIN_O': SO.plan(10, 'official'), 'MAIN_R': SR.plan(10, 'official'), 'NEUTRAL05': SN.plan(10, 'official'), 'KY_ALL': SA.plan(10, 'official')}
PL['KY_CAP2'], cap_skips = plan_cap(SA, 10, 2)
for k, v in PL.items(): assert [p[0] for p in v] == [p[0] for p in PL['MAIN_R']], k
log('plans', {k: len(v) for k, v in PL.items()})

def eqw(pl): return [(p[0], np.asarray(p[1], int), np.full(len(p[1]), 1.0 / len(p[1]))) for p in pl]
def mix(pa, pb, wa):
    out = []
    for a, b in zip(pa, pb):
        assert a[0] == b[0]
        idx = np.concatenate([a[1], b[1]]); w = np.concatenate([wa / len(a[1]) * np.ones(len(a[1])), (1 - wa) / len(b[1]) * np.ones(len(b[1]))])
        out.append((a[0], idx, w))
    return out
WP = {k: eqw(v) for k, v in PL.items()}
WP['MIX80_20_KY'] = mix(PL['MAIN_R'], PL['KY_ALL'], 0.8)
WP['MIX50_50_KY'] = mix(PL['MAIN_R'], PL['KY_ALL'], 0.5)
WP['TSMC30_R'] = O.tsmc_plan(PL['MAIN_R'], 0.30)
ORDER = ['MAIN_O', 'MAIN_R', 'NEUTRAL05', 'KY_ALL', 'KY_CAP2', 'MIX80_20_KY', 'MIX50_50_KY', 'TSMC30_R']
EWSRC = {'MAIN_O': 'MAIN_O', 'MAIN_R': 'MAIN_R', 'NEUTRAL05': 'NEUTRAL05', 'KY_ALL': 'KY_ALL', 'KY_CAP2': 'KY_ALL', 'MIX80_20_KY': 'MAIN_R', 'MIX50_50_KY': 'MAIN_R', 'TSMC30_R': 'MAIN_R'}
EW = {}
for k in ('MAIN_O', 'MAIN_R', 'NEUTRAL05', 'KY_ALL'):
    EW[k], _ = K.simulate([(p[0], p[2]) for p in PL[k]], 0, 0, t0, t1)

NAV, NAVS, TOV = {}, {}, {}
for k in ORDER:
    NAV[k], lg = O.simulate_w(WP[k], *O.NORMAL, t0, t1); TOV[k] = lg['turnover']
    NAVS[k], _ = O.simulate_w(WP[k], *O.STRESS, t0, t1)
    log('sim', k, round(float(NAV[k].iloc[-1]), 3))

# ---------------- 驗證 ----------------
I1 = json.loads((HERE.parent / 'integrate_o1' / 'results.json').read_text(encoding='utf-8'))
mo = O.segstat(NAV['MAIN_O'], EW['MAIN_O'], sd, *P['ALL'])['CAGR']
val = {'MAIN_O_ALL_CAGR': mo, 'integrate_o1_MAIN_ALL_CAGR': I1['RESULTS']['combos']['MAIN']['stats']['ALL']['CAGR']}
assert abs(val['MAIN_O_ALL_CAGR'] - val['integrate_o1_MAIN_ALL_CAGR']) <= 0.05, val
for k, f in (('MAIN_R', 'R_DEL'), ('NEUTRAL05', 'B05_DEL'), ('KY_ALL', 'R_ALL'), ('MAIN_O', 'ORIG')):
    ref = pd.read_parquet(HERE.parent / 'o_rev_survivorship' / f'navs_{f}.parquet')['EXFIN|n10|normal'].dropna()
    a, b = float(NAV[k].loc[ref.index[-1]] / NAV[k].iloc[0]), float(ref.iloc[-1] / ref.iloc[0])
    val[f'{k}_vs_osurv_{f}_final_rel'] = round(a / b - 1, 6)
    assert abs(a / b - 1) <= 0.001, (k, a, b)
log('validate ok', val)

# ---------------- 統計 ----------------
B = K.E50
W = {k: O.window_returns(v, sd) for k, v in NAV.items()}
WB = O.window_returns(B, sd)
def win(x, a, b): return x[(x.index >= pd.Timestamp(a)) & (x.index <= pd.Timestamp(b))]
def corr(x, y, a, b):
    z = pd.concat([win(x, a, b), win(y, a, b)], axis=1).dropna()
    return round(float(z.iloc[:, 0].corr(z.iloc[:, 1])), 4)
def strip(st): return {k: v for k, v in st.items() if k != 'yearly'}
def yearly(nav):
    n = nav.dropna(); g = n.groupby(n.index.year).last()
    prev = g.shift(1); prev.iloc[0] = n.iloc[0]
    return g / prev - 1
Y50 = yearly(B.reindex(NAV['MAIN_R'].index))

def picks_info(k):
    if k not in PL: return None
    pl = PL[k]; base = PL['MAIN_R']
    nky = [sum(K.COLS[j] in FSET for j in p[1]) for p in pl]
    diff = [str(p[3].date()) for p, q in zip(pl, base) if set(p[1].tolist()) != set(q[1].tolist())]
    slots = sum(len(set(p[1].tolist()) - set(q[1].tolist())) for p, q in zip(pl, base))
    kys = {}
    for p in pl:
        for j in p[1]:
            c = K.COLS[j]
            if c in FSET: kys[c] = kys.get(c, 0) + 1
    return {'ky_mean': round(float(np.mean(nky)), 3), 'ky_max': int(max(nky)), 'months_ky_ge3': int(sum(x >= 3 for x in nky)),
            'n_signals_differ_vs_MAIN_R': len(diff), 'slots_differ_vs_MAIN_R': int(slots), 'first_last_differ': [diff[0], diff[-1]] if diff else None,
            'differ_by_year': pd.Series([d[:4] for d in diff]).value_counts().sort_index().to_dict() if diff else {},
            'top_ky': sorted(kys.items(), key=lambda x: -x[1])[:10]}

out = {'validation': val, 'periods': {k: list(v) for k, v in P.items()}, 'kycap2_skips': cap_skips, 'combos': {}}
for k in ORDER:
    ew = EW[EWSRC[k]]
    d = {'stats': {}, 'stress': {}, 'paired_vs_MAIN_R': {}, 'paired_vs_MAIN_O': {}, 'corr_vs_MAIN_R': {}, 'corr_vs_MAIN_O': {}, 'corr_ex0050_vs_MAIN_R': {},
         'turnover': round(TOV[k], 4), 'picks': picks_info(k)}
    for p in KEYS:
        a, b, ad, bd = P[p]
        d['stats'][p] = strip(O.segstat(NAV[k], ew, sd, a, b, ad, bd))
        d['stress'][p] = strip(O.segstat(NAVS[k], ew, sd, a, b, ad, bd))
        d['paired_vs_MAIN_R'][p] = O.paired_t(W[k], W['MAIN_R'], a, b)
        d['paired_vs_MAIN_O'][p] = O.paired_t(W[k], W['MAIN_O'], a, b)
        d['corr_vs_MAIN_R'][p] = corr(W[k], W['MAIN_R'], a, b)
        d['corr_vs_MAIN_O'][p] = corr(W[k], W['MAIN_O'], a, b)
        d['corr_ex0050_vs_MAIN_R'][p] = corr(W[k] - WB, W['MAIN_R'] - WB, a, b)
    y = yearly(NAV[k])
    d['yearly_pct'] = {int(i): round(float(v) * 100, 1) for i, v in y.items()}
    d['years_beating_0050'] = int((y > Y50).sum()); d['n_years'] = int(len(y))
    s = d['stats']['ALL']
    d['expectation_ALL'] = round(s['E50_CAGR'] + 0.5 * (s['CAGR'] - s['E50_CAGR']), 2)
    d['expectation_HOLD'] = round(d['stats']['HOLD']['E50_CAGR'] + 0.5 * (d['stats']['HOLD']['CAGR'] - d['stats']['HOLD']['E50_CAGR']), 2)
    out['combos'][k] = d
    log(k, {p: (d['stats'][p]['CAGR'], d['stats'][p]['maxDD'], d['stats'][p]['t_0050'], d['paired_vs_MAIN_R'][p]['t'], d['corr_vs_MAIN_R'][p]) for p in ('SEL', 'VAL', 'HOLD', 'FIX', 'ALL')})
out['yearly_0050_pct'] = {int(i): round(float(v) * 100, 1) for i, v in Y50.items()}
# KY 腿本身：KY_ALL 減 MAIN_R 的差異只在哪裡
out['legs'] = {p: {'corr_KY_ALL_vs_MAIN_R': out['combos']['KY_ALL']['corr_vs_MAIN_R'][p],
                   'corr_KYex_vs_MAINRex': corr(W['KY_ALL'] - W['MAIN_R'], W['MAIN_R'] - WB, *P[p][:2])} for p in KEYS}

# ---------------- 選版（只用 SEL）與判定 ----------------
C = out['combos']; m = C['MAIN_R']['stats']
cands = ['NEUTRAL05', 'KY_CAP2', 'MIX80_20_KY', 'MIX50_50_KY', 'KY_ALL', 'TSMC30_R']
ok = [c for c in cands if m['SEL']['CAGR'] - C[c]['stats']['SEL']['CAGR'] <= 1.0 and C[c]['stats']['SEL']['t_0050'] > m['SEL']['t_0050']]
pick = max(ok, key=lambda c: C[c]['stats']['SEL']['t_0050']) if ok else 'MAIN_R'
sel_tab = {c: {'SEL_t_0050': C[c]['stats']['SEL']['t_0050'], 'SEL_CAGR_loss_pp': round(m['SEL']['CAGR'] - C[c]['stats']['SEL']['CAGR'], 2), 'eligible': c in ok} for c in cands}
CAP = {c: '待觀察' for c in cands}
RANK = {'可用': 2, '待觀察': 1, '不可用': 0}
judg = {}
for c in cands:
    s = C[c]['stats']; pr = C[c]['paired_vs_MAIN_R']
    A = pr['ALL']['t'] >= 1.0 and pr['SEL']['mean_pp'] > 0 and pr['VAL']['mean_pp'] > 0
    Bc = (s['ALL']['t_0050'] - m['ALL']['t_0050'] >= 0.2 and s['SEL']['t_0050'] > m['SEL']['t_0050'] and s['VAL']['t_0050'] > m['VAL']['t_0050']
          and m['ALL']['CAGR'] - s['ALL']['CAGR'] <= 1.0 and s['ALL']['maxDD'] >= m['ALL']['maxDD'])
    if A or Bc: raw = '可用'
    elif pr['ALL']['t'] > -1 and m['ALL']['CAGR'] - s['ALL']['CAGR'] <= 2.0 and any(s[p]['t_0050'] > m[p]['t_0050'] for p in ('SEL', 'VAL', 'FIX')): raw = '待觀察'
    else: raw = '不可用'
    fin = raw if RANK[raw] <= RANK[CAP[c]] else CAP[c]
    judg[c] = {'path_A_return': bool(A), 'path_B_tracking': bool(Bc), 'ALL_paired_t': pr['ALL']['t'], 'ALL_CAGR_loss_pp': round(m['ALL']['CAGR'] - s['ALL']['CAGR'], 2),
               'ALL_dt_0050': round(s['ALL']['t_0050'] - m['ALL']['t_0050'], 3), 'raw': raw, 'cap': CAP[c], 'verdict': fin}
out['selection'] = {'table': sel_tab, 'picked': pick}
out['judgement'] = judg
log('pick', pick, sel_tab); log('judg', judg)

pd.DataFrame(NAV).to_parquet(HERE / 'navs.parquet')
R['RESULTS'] = out
R['RESULTS_written_at'] = pd.Timestamp.now(tz='UTC').isoformat()
RJ.write_text(O.dumps(R), encoding='utf-8')
log('done')
