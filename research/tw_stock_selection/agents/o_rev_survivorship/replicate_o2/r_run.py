#!/usr/bin/env python3
"""複製員主回測：python3 -I -B r_run.py <VERSION> [delisted_only=0|1] [nmc=40]
VERSION ∈ ORIG R_DEL R_ALL B05_DEL B10_DEL B05_ALL B10_ALL PX_ORIG PX_DEL。輸出 my_run_<VERSION>[_dl].json、my_navs_<...>.parquet、my_plans_<...>.json"""
import sys, json, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from r_lib import *
import r_lib as RL

VER = sys.argv[1]
DL = len(sys.argv) > 2 and sys.argv[2] == '1'
NMC = int(sys.argv[3]) if len(sys.argv) > 3 else 40
import os
TAG = VER + ('_dl' if DL else '') + os.environ.get('TAGX', '')
T0 = time.time()
def log(*a): print(f'[{TAG} {time.time() - T0:5.0f}s]', *a, flush=True)

px_info = None
if VER.startswith('PX_'):
    px_info = apply_price_patch()
    rv = {'PX_ORIG': 'ORIG', 'PX_DEL': 'R_DEL'}[VER]; bound = None
elif VER.startswith('B'):
    p = {'B05': 0.5, 'B10': 1.0}[VER[:3]]; scope = VER[4:]
    rv = {'DEL': 'R_DEL', 'ALL': 'R_ALL'}[scope]; bound = (p, scope)
else:
    rv = VER; bound = None
YL = load_yoy(rv)
_, Y5, _ = V.load_revenue()
COST = {'normal': K.rates(0.001), 'stress': K.rates('stress')}
SEGS = dict(V.SEGS)
SEGS.update({'SEL': ('2006-04-01', '2019-04-30', '2006-04-01'), 'VAL': ('2019-05-01', '2023-12-31', '2019-05-01'),
             'HOLD': ('2024-01-01', '2026-10-12', '2024-01-01')})

def r4(o): return {k: (round(float(v), 4) if isinstance(v, (float, np.floating)) else v) for k, v in o.items()}
def segstats(nav, ew, sd):
    return {k: r4(V.seg_stats(nav, ew, a, b, sd, ad=ad)) for k, (a, b, ad) in SEGS.items()}
def winret(nav, sd):
    D = pd.DatetimeIndex([d for d in sd if d in nav.index]); x = nav.reindex(D)
    return (x.shift(-1) / x - 1).iloc[:-1]

res = {'version': VER, 'rev': rv, 'bound': bound, 'delisted_only': DL, 'px': px_info}
navs = {}; plans = {}
for univ in ('ALL', 'EXFIN'):
    exfin = univ == 'EXFIN'
    S = V2B('gapfix', YL, Y5, exfin=exfin, bound=bound, delisted_only=DL)
    t0, t1, sd = S.bounds()
    ew = None
    for n in (10, 30):
        pl = S.plan(n, 'official')
        plans[f'{univ}|n{n}'] = [(str(p[3].date()), [COLS[j] for j in p[1]]) for p in pl]
        if ew is None:
            ew, _ = K.simulate([(p[0], p[2]) for p in pl], 0, 0, t0, t1); navs[f'{univ}|EW'] = ew
        for c, (bb, ss) in COST.items():
            nav, lg = K.simulate([(p[0], p[1]) for p in pl], bb, ss, t0, t1)
            navs[f'{univ}|n{n}|{c}'] = nav
            o = segstats(nav, ew, sd)
            w = winret(nav, sd); b5 = winret(K.E50.reindex(nav.index), sd)
            ex = (w - b5)
            ex = ex[(ex.index >= pd.Timestamp('2006-04-01')) & (ex.index <= pd.Timestamp('2026-10-12'))]
            o['ALL']['ex50_m_mean_pp'] = float(ex.mean() * 100); o['ALL']['ex50_m_sd_pp'] = float(ex.std(ddof=1) * 100)
            res.setdefault(univ, {}).setdefault(f'n{n}', {})[c] = o
            log(univ, n, c, {k: (o[k]['CAGR'], o[k]['t_0050']) for k in ('ALL', 'S2', 'S3', 'P1')})
        if bound is not None:
            res[univ][f'n{n}']['mean_added'] = float(np.mean([S.cross_section(d).get('nadd', 0) for d in S.sig_days if d >= RL.B_START]))
    res[univ]['sd'] = [str(d.date()) for d in sd]

# ---------- AB（排除金融，MC） ----------
res['AB'] = {}
for n in (10, 30):
    rets = {c: [] for c in COST}; ewr = []
    for seed in range(NMC):
        Sa = V2B('gapfix', YL, Y5, exfin=True, unavail=V.make_unavail('mc', seed), bound=bound, delisted_only=DL)
        ta, tb, sda = Sa.bounds()
        pl = Sa.plan(n, 'official')
        ewa, _ = K.simulate([(p[0], p[2]) for p in pl], 0, 0, ta, tb); ewr.append(ewa.pct_change())
        for c, (bb, ss) in COST.items():
            nav, _ = K.simulate([(p[0], p[1]) for p in pl], bb, ss, ta, tb); rets[c].append(nav.pct_change())
    ew_m = (1 + pd.concat(ewr, axis=1).mean(axis=1).fillna(0)).cumprod()
    for c in COST:
        nav_m = (1 + pd.concat(rets[c], axis=1).mean(axis=1).fillna(0)).cumprod()
        navs[f'AB|n{n}|{c}'] = nav_m
        o = segstats(nav_m, ew_m, sda)
        for k in o:
            o[k]['live_exp'] = o[k]['E50_CAGR'] + 0.5 * (o[k]['CAGR'] - o[k]['E50_CAGR'])
        res['AB'][f'n{n}'] = res['AB'].get(f'n{n}', {}); res['AB'][f'n{n}'][c] = o
        log('AB', n, c, {k: (o[k]['CAGR'], o[k]['t_0050'], round(o[k]['live_exp'], 2)) for k in ('ALL', 'P1', 'S3', 'S4')})

(H / f'my_run_{TAG}.json').write_text(json.dumps(res, ensure_ascii=False, default=lambda x: x.item() if hasattr(x, 'item') else str(x)), encoding='utf-8')
pd.DataFrame(navs).to_parquet(H / f'my_navs_{TAG}.parquet')
(H / f'my_plans_{TAG}.json').write_text(json.dumps(plans), encoding='utf-8')
log('done')
