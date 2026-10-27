#!/usr/bin/env python3
"""主回測：python3 -I -B run.py <VERSION>（ORIG／R_DEL／R_ALL／B05_DEL／B10_DEL／B05_ALL／B10_ALL／PX_ORIG／PX_DEL）
輸出 run_<VERSION>.json、navs_<VERSION>.parquet、plans_<VERSION>.json（逐期持股與持有期報酬）。
環境變數 OSURV_SMOKE=1：只跑 ORIG、MC 1 次（只用來確認能重現 main_v2，不寫正式檔）。"""
import sys, os, time, json, hashlib
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import numpy as np, pandas as pd
import osurv_core as O
V, K = O.V, O.K

VER = sys.argv[1]
CFG = {'ORIG': ('ORIG', None, None, False), 'R_DEL': ('R_DEL', None, None, False), 'R_ALL': ('R_ALL', None, None, False),
       'B05_DEL': ('R_DEL', 'R_DEL', 0.5, False), 'B10_DEL': ('R_DEL', 'R_DEL', 1.0, False),
       'B05_ALL': ('R_ALL', 'R_ALL', 0.5, False), 'B10_ALL': ('R_ALL', 'R_ALL', 1.0, False),
       'PX_ORIG': ('ORIG', None, None, True), 'PX_DEL': ('R_DEL', None, None, True)}
rev, gapbase, pfix, px = CFG[VER]
SMOKE = bool(os.environ.get('OSURV_SMOKE'))
RES = json.loads((HERE / 'results.json').read_text(encoding='utf-8')) if (HERE / 'results.json').exists() else {}
if not SMOKE:
    assert RES.get('PREREG'), '先執行 prereg.py'
    assert hashlib.sha256(json.dumps(RES['PREREG'], ensure_ascii=False, sort_keys=True).encode()).hexdigest() == RES['PREREG_sha256'], 'PREREG 被改過'
    assert RES['PREREG']['data']['freeze']['R_DEL_sha256'] == O.FREEZE['outputs']['R_DEL']['sha256']
assert hashlib.md5((K.P / 'value.parquet').read_bytes()).hexdigest().startswith('8272e253'), 'value.parquet 不是修正版'
T0 = time.time()
def log(*a): print(f'[{VER} {time.time() - T0:6.0f}s]', *a, flush=True)

npatch = O.apply_price_patch() if px else 0
YL, Y5 = O.load_yl(rev)
GAP = O.gapsets(gapbase) if gapbase else None
COST = {'normal': K.rates(0.001), 'stress': K.rates('stress')}
NMC = 1 if SMOKE else 40
CONF = [('ALL', 10), ('ALL', 30), ('EXFIN', 10), ('EXFIN', 30)]
if SMOKE: CONF = [('EXFIN', 10)]

def r2(o): return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in o.items()}
def segstats(nav, ew, sd, yearly=False):
    return {k: r2(V.seg_stats(nav, ew, a, b, sd, yearly=yearly, ad=ad)) for k, (a, b, ad) in V.SEGS.items()}
def mk(exfin, unavail=None):
    return O.V2B('gapfix', YL, Y5, exfin=exfin, unavail=unavail, gap=GAP, pfix=pfix)

res = {'version': VER, 'cfg': {'rev': rev, 'gapbase': gapbase, 'pfix': pfix, 'price_patch': px, 'n_patched_cells': npatch}, 'BASE': {}, 'AB': {}}
navs = {}; plans = {}
for univ in ('ALL', 'EXFIN'):
    exfin = univ == 'EXFIN'
    ns = [n for u, n in CONF if u == univ]
    if not ns: continue
    S = mk(exfin)
    t0, t1, sd = S.bounds()
    ew = None
    for n in ns:
        pl = S.plan(n, 'official')
        if ew is None:
            ew, _ = K.simulate([(p[0], p[2]) for p in pl], 0, 0, t0, t1)
            navs[f'{univ}|EW'] = ew
            res['BASE'][f'{univ}|EW'] = {k: r2(K.short(K.stats(ew, ad, b, sd))) for k, (a, b, ad) in V.SEGS.items()}
            res['BASE'][f'{univ}|EW_yearly'] = K.stats(ew, '2006-04-01', '2026-10-12', sd)['yearly']
            res['BASE'][f'{univ}|pool_mean'] = float(np.mean([p[2].size for p in pl]))
            if GAP is not None:
                res['BASE'][f'{univ}|gap_used_mean'] = float(np.mean(list(S.ngap_used.values()))) if S.ngap_used else 0.0
        o = {}
        for c, (bb, ss) in COST.items():
            nav, lg = K.simulate([(p[0], p[1]) for p in pl], bb, ss, t0, t1)
            o[c] = segstats(nav, ew, sd, yearly=(c == 'normal'))
            o[c + '_turnover'] = lg['turnover']; o[c + '_n_frozen'] = len(lg['frozen'])
            if c == 'normal':
                navs[f'{univ}|n{n}|normal'] = nav
                o['contrib'] = {k: V.top_contrib(pl, t1, a, b) for k, (a, b, ad) in V.SEGS.items()}
                con, per = V.stock_contrib(pl, t1)
                plans[f'{univ}|n{n}'] = [{'sig': str(d.date()), 'codes': [K.COLS[j] for j in pk], 'ret': [round(float(x), 6) for x in r]} for d, pk, r in per]
                o['frozen'] = lg['frozen'][:30]
            else:
                navs[f'{univ}|n{n}|stress'] = nav
        o['blocked'] = S.nblk
        res['BASE'][f'{univ}|n{n}'] = o
        log('BASE', univ, n, {k: (v['CAGR'], v['t_0050'], v['t_EW']) for k, v in o['normal'].items()})
    # ---- AB ----
    for n in ns:
        rets = {c: [] for c in COST}; ewr = []; cg = {c: [] for c in COST}
        for seed in range(NMC):
            Sa = mk(exfin, V.make_unavail('mc', seed))
            ta, tb, sda = Sa.bounds()
            pl = Sa.plan(n, 'official')
            ewa, _ = K.simulate([(p[0], p[2]) for p in pl], 0, 0, ta, tb)
            ewr.append(ewa.pct_change())
            for c, (bb, ss) in COST.items():
                nav, _ = K.simulate([(p[0], p[1]) for p in pl], bb, ss, ta, tb)
                rets[c].append(nav.pct_change())
                cg[c].append({k: V.seg_stats(nav, ewa, a, b, sda, ad=ad)['CAGR'] for k, (a, b, ad) in V.SEGS.items()})
        assert ta == t0 and tb == t1
        ew_m = (1 + pd.concat(ewr, axis=1).mean(axis=1).fillna(0)).cumprod()
        out = {}
        for c in COST:
            nav_m = (1 + pd.concat(rets[c], axis=1).mean(axis=1).fillna(0)).cumprod()
            out[c] = segstats(nav_m, ew_m, sd, yearly=(c == 'normal'))
            out[c + '_mc_p05_p50_p95'] = {k: [round(float(np.percentile([x[k] for x in cg[c]], q)), 2) for q in (5, 50, 95)] for k in V.SEGS}
            navs[f'AB_{univ}|n{n}|{c}'] = nav_m
        res['AB'][f'{univ}|n{n}'] = out
        log('AB', univ, n, {k: (v['CAGR'], v['t_0050']) for k, v in out['normal'].items()})

tag = 'smoke_' if SMOKE else ''
(HERE / f'run_{tag}{VER}.json').write_text(json.dumps(res, ensure_ascii=False, default=lambda x: x.item() if hasattr(x, 'item') else str(x)), encoding='utf-8')
if not SMOKE:
    pd.DataFrame(navs).to_parquet(HERE / f'navs_{VER}.parquet')
    (HERE / f'plans_{VER}.json').write_text(json.dumps(plans, ensure_ascii=False), encoding='utf-8')
log('done')
