#!/usr/bin/env python3
import sys, json, pickle, math
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import numpy as np, pandas as pd
import r4_core as C
K = C.K
P = pickle.load(open(HERE / 'plans.pkl', 'rb'))
PL, sd, t0, t1 = P['PL'], P['sd'], P['t0'], P['t1']
FEE = K.FEE
def rates(slip, fee=FEE): return fee + slip, fee + 0.003 + slip
COST = {'normal': rates(0.001), 'stress': rates(0.004), 'gross': (0.0, 0.0), 'flat': (0.005, 0.005), 'fullfee': rates(0.001, 0.001425)}
S = C.make_S()
V60sig = {}
for p in PL['BASE']:
    V60sig[p['e']] = S.V60[p['e'] - 1]
RAWV60 = pd.DataFrame(K.V.to_numpy()).rolling(60, min_periods=20).mean().to_numpy()
def cap_fn(base_b, base_s):
    def f(t, htr, tgt, Vtot):
        v = S.V60[t - 1]
        v = np.where(np.isfinite(v) & (v > 0), v, RAWV60[t - 1])
        v = np.where(np.isfinite(v) & (v > 0), v, 1e7)
        amt = np.abs(tgt - htr) / Vtot * 1e7
        ex = 0.025 * np.sqrt(amt / v)
        return base_b + ex, base_s + ex
    return f

def plan_of(v): return [(p['e'], p['picks']) for p in PL[v]]
NAV = {}; TOV = {}
def run(v, c):
    if (v, c) in NAV: return NAV[(v, c)]
    if c == 'cap10m': nav, lg = C.sim(plan_of(v), 0, 0, t0, t1, cost_fn=cap_fn(*COST['normal']))
    elif isinstance(c, tuple): nav, lg = C.sim(plan_of(v), *c, t0, t1)
    else: nav, lg = C.sim(plan_of(v), *COST[c], t0, t1)
    NAV[(v, c)] = nav; TOV[(v, c)] = lg
    return nav

E50 = C.E50
def stat(nav, a, b):
    cg, dd = C.cagr_dd(nav, a, b)
    x = C.sel(C.winret(nav, sd) - C.winret(E50.reindex(nav.index), sd), a, b)
    return dict(CAGR=cg, maxDD=dd, t=C.tstat(x), ex_pp=float(x.mean() * 100), n=int(len(x)), E50=C.cagr_dd(E50.reindex(nav.index), a, b)[0],
                t_cal=C.cal_t(nav, a, b))
def paired(na, nb, a, b, drop=None):
    x = C.sel(C.winret(na, sd) - C.winret(nb, sd), a, b)
    return dict(pp=float(x.mean() * 100), t=C.tstat(x), n=len(x))

out = {}
for c in ('normal', 'stress', 'gross', 'flat', 'cap10m', 'fullfee'):
    for v in ('BASE', 'A20'):
        nav = run(v, c)
        out[f'{v}|{c}'] = {k: stat(nav, a, b) for k, (a, b) in C.PER.items()}
    out[f'PAIR|{c}'] = {k: paired(run('A20', c), run('BASE', c), a, b) for k, (a, b) in C.PER.items()}
    print(c, 'BASE', {k: (round(x['CAGR'], 2), round(x['t'], 3), round(x['maxDD'], 1)) for k, x in out[f'BASE|{c}'].items()}, flush=True)
    print(c, 'A20 ', {k: (round(x['CAGR'], 2), round(x['t'], 3), round(x['maxDD'], 1)) for k, x in out[f'A20|{c}'].items()}, flush=True)
    print(c, 'PAIR', {k: (round(x['pp'], 3), round(x['t'], 2)) for k, x in out[f'PAIR|{c}'].items()}, flush=True)
print('E50', {k: round(x['E50'], 2) for k, x in out['BASE|normal'].items()})
# 周轉
def tov(v, a, b):
    lg = TOV[(v, 'normal')]
    ds = [p['d'] for p in PL[v]]
    s = pd.Series(lg['ts'], index=pd.DatetimeIndex(ds)).iloc[1:]
    bb = pd.Series(lg['tb'], index=pd.DatetimeIndex(ds)).iloc[1:]
    return float(C.sel(s, a, b).mean() * 100), float(C.sel(bb, a, b).mean() * 100)
out['turnover'] = {v: {k: tov(v, a, b) for k, (a, b) in C.PER.items()} for v in ('BASE', 'A20')}
print('turnover', out['turnover'])
# A 家族
for v in ('A15', 'A25', 'A30'):
    run(v, 'normal')
    out[f'PAIR_A|{v}'] = {c: paired(run(v, c), run('BASE', c), *C.PER['SEL']) for c in ('normal', 'stress')}
    out['turnover'][v] = {k: tov(v, a, b) for k, (a, b) in C.PER.items()}
out['PAIR_A|A20'] = {c: out[f'PAIR|{c}']['SEL'] for c in ('normal', 'stress')}
for v in ('A15', 'A20', 'A25', 'A30'):
    print(v, 'turn SEL %.1f' % out['turnover'][v]['SEL'][0], {c: (round(x['pp'], 3), round(x['t'], 2)) for c, x in out[f'PAIR_A|{v}'].items()})

# T3 損益兩平滑價
def be(v, fn, target):
    lo, hi = 0.0, 0.03
    for _ in range(30):
        m = 0.5 * (lo + hi)
        if fn(run(v, rates(m))) > target: lo = m
        else: hi = m
    return 0.5 * (lo + hi) * 100
ex_sel = lambda nav: stat(nav, *C.PER['SEL'])['ex_pp']
t_all = lambda nav: stat(nav, *C.PER['ALL'])['t']
out['T3'] = {v: be(v, ex_sel, 0.0) for v in ('BASE', 'A20')}
out['T3_alt'] = {v: be(v, t_all, 2.3) for v in ('BASE', 'A20')}
print('T3', out['T3'], 'T3alt', out['T3_alt'], flush=True)

# T4：挑選期拿掉最好 3 個月（多種定義）
def wr(v, c='normal'): return C.winret(run(v, c), sd)
rb, ra, r5 = wr('BASE'), wr('A20'), C.winret(E50.reindex(run('BASE', 'normal').index), sd)
dd = C.sel(ra - rb, *C.PER['SEL'])
T4 = {}
for lab, key in (('best_A20', C.sel(ra, *C.PER['SEL'])), ('best_BASE', C.sel(rb, *C.PER['SEL'])), ('best_diff', dd),
                 ('best_A20_ex', C.sel(ra - r5, *C.PER['SEL'])), ('best_BASE_ex', C.sel(rb - r5, *C.PER['SEL'])), ('best_0050', C.sel(r5, *C.PER['SEL']))):
    drop = key.sort_values().index[-3:]
    x = dd.drop(drop)
    T4[lab] = (float(x.mean() * 100), C.tstat(x))
out['T4'] = T4
print('T4', T4, flush=True)

# 端點敏感度（ALL 對 0050 t）
def endp(v):
    r = wr(v); x = C.sel(r - r5, *C.PER['ALL'])
    o = {'none': C.tstat(x), 'drop_last': C.tstat(x.iloc[:-1])}
    for k in (1, 3, 5):
        o[f'drop_best{k}_ex'] = C.tstat(x.drop(x.sort_values().index[-k:]))
        rr = C.sel(r, *C.PER['ALL'])
        o[f'drop_best{k}_raw'] = C.tstat(x.drop(rr.sort_values().index[-k:]))
    yrs = x.groupby(x.index.year).sum()
    o['best_year'] = int(yrs.idxmax())
    o['drop_2021'] = C.tstat(x[x.index.year != 2021])
    o['to_2025'] = C.tstat(x[x.index <= pd.Timestamp('2025-12-31')])
    o['cal_t'] = C.cal_t(run(v, 'normal'), *C.PER['ALL'])
    return o
out['endpoint'] = {v: endp(v) for v in ('BASE', 'A20')}
print('endpoint', out['endpoint'], flush=True)
from math import erf, sqrt
p1 = lambda t: 0.5 * (1 - erf(t / sqrt(2)))
out['bonf'] = {v: {'m30': 30 * p1(out[f'{v}|normal']['ALL']['t']), 'm10': 10 * p1(out[f'{v}|normal']['ALL']['t'])} for v in ('BASE', 'A20')}
print('bonf', out['bonf'])
# 損益兩平也看 ALL 對 0050 t 在不同滑價
(HERE / 'r_results.json').write_text(json.dumps(out, ensure_ascii=False, indent=1, default=float), encoding='utf-8')
pickle.dump({k: v for k, v in NAV.items() if not isinstance(k[1], tuple)}, open(HERE / 'navs.pkl', 'wb'))
