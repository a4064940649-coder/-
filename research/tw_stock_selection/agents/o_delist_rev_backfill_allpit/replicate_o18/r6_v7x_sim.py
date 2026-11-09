#!/usr/bin/env python3
"""G6 複製員 r6：V7X 模擬（cz_v7x.CSim，K5T、GROW；20 seeds 日報酬平均成一條淨值；x_stats 算 j 中位數與 p10）。
名單＝r5 產生的 plans_r5{tag}.csv。不讀研究員 *.py；cfg 照 o_engine_delist_consideration/s10_v7x.py（公開程式）。
python3 -I -B r6_v7x_sim.py [--tag=] [--cells=all|main]"""
import sys, json, time
from pathlib import Path
from multiprocessing import Pool
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent
AG = HERE.parent.parent
SCR = Path('/tmp/claude-0/-home-user--/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/rep_o18')
OPT = dict(a[2:].split('=') for a in sys.argv[1:] if a.startswith('--'))
TAG = OPT.get('tag', ''); CELLSEL = OPT.get('cells', 'all')
sys.path.insert(0, str(AG / 'o_engine_delist_consideration'))
import cz_v7x as CV, cz_engine as CE
sys.path.pop(0)
D = CV.D
sys.path.insert(0, str(CV.V7X)); import x_stats as S; sys.path.pop(0)
VR = json.loads((CV.V7X / 'results.json').read_text())
BASE = dict(VR['PREREG']['orig_cfg']); BASE.update(dict(f1=True, f2=True, f3=0.5, f4=True))
E = pd.read_csv(AG / 'o_engine_delist_consideration' / 'events_v2.csv', dtype={'code': str, 'successor': str})
kw = dict(TR=D.TR, IDX=D.IDX, Cn=D.C, CI=D.CI, On=D.O)
ST = {b: CE.build_settlement(E, b.split('_')[1], d=0.057, **kw) for b in ('cons_main', 'cons_lo', 'cons_hi')}
TIERS = {'100萬': 1e6, '300萬': 3e6, '1000萬': 1e7}
CELLS = [(c, t) for c in ('normal', 'stress') for t in TIERS] if CELLSEL == 'all' else [('normal', '100萬')]
ENG = {'orig': ['orig', 'cons_main'], 'REAL_p90': ['cons_main'], 'REAL_carry': ['cons_main'], 'REAL_opt': ['cons_main', 'cons_lo', 'cons_hi'] if CELLSEL == 'all' else ['cons_main']}
PL = pd.read_csv(SCR / f'plans_r5{TAG}.csv', dtype={'picks': str})
PLANS = {}
for (v, j, s), g in PL.groupby(['strat', 'j', 'seed'], sort=False):
    PLANS[(v, int(j), int(s))] = sorted([(int(e), np.array([D.CI[c] for c in pk.split(',')], int), D.TPOS[pd.Timestamp(sg)]) for sg, e, pk in zip(g.signal, g.exec_idx, g.picks)], key=lambda x: x[0])

def seg3(nav, j):
    e = S.end_day('AB_V7_B10', j); nav = nav.loc[:e].dropna(); a = nav.index[0]
    b12 = nav.index[nav.index <= '2012-12-31'][-1]; b18 = nav.index[nav.index <= '2018-12-31'][-1]
    o = {}
    for k, (x, y) in {'S0612': (a, b12), 'S1318': (b12, b18), 'S19on': (b18, e)}.items():
        o['exc_' + k] = S.cagr(nav, x, y) - S.cagr(S.E50, x, y)
    return o

def job(key):
    v, j, s = key; pl = PLANS[key]; out = {}; info = {}
    for c, t in CELLS:
        cfg = dict(tier=TIERS[t], mode='GROW', model='K5T', cost=c); cfg.update(BASE)
        engs = ENG[v] if (c, t) == ('normal', '100萬') else ENG[v][:1] + (['cons_main'] if v == 'orig' else [])
        for eg in dict.fromkeys(engs):
            sm = CV.CSim(pl, cfg, settle=ST.get(eg), cmode='orig' if eg == 'orig' else 'consideration')
            out[(c, t, eg)] = sm.run()
            if (c, t) == ('normal', '100萬') and eg != 'orig':
                info[eg] = sorted({e['code'] for e in sm.log})
    return key, out, info

if __name__ == '__main__':
    T0 = time.time()
    keys = sorted(PLANS)
    navs = {}; infos = {}
    with Pool(4) as pool:
        for n, (key, out, info) in enumerate(pool.imap_unordered(job, keys, chunksize=1)):
            for ck, nv in out.items(): navs.setdefault((key[0], key[1]) + ck, []).append((key[2], nv))
            infos[f'{key[0]}|J{key[1]}|s{key[2]}'] = info
            if n % 40 == 0: print(n, round(time.time() - T0), 's', flush=True)
    cols = {}
    for (v, j, c, t, eg), Lst in navs.items():
        cols[f'{v}|J{j}|{c}|{t}|{eg}'] = S.avg_nav([nv for s, nv in sorted(Lst, key=lambda x: x[0])])
    df = pd.DataFrame(cols).sort_index(axis=1)
    df.to_parquet(SCR / f'navs_r6{TAG}.parquet')
    R4 = pd.read_parquet(CV.V7X / 'out' / 'navs_oos_R4.parquet')
    rep = {}
    for j in range(13):
        for c, t in CELLS:
            k = f'orig|J{j}|{c}|{t}|orig'
            if k in df: rep[f'J{j}|{c}|{t}'] = float(np.nanmax(np.abs(df[k].to_numpy() - R4[f'B10|J{j}|K5T|{c}|{t}|GROW'].reindex(df.index).to_numpy())))
    TAB = {}
    for v in sorted({k[0] for k in PLANS}):
        for c, t in CELLS:
            for eg in ENG[v] + (['cons_main'] if v == 'orig' else []):
                ks = [f'{v}|J{j}|{c}|{t}|{eg}' for j in range(13)]
                if not all(k in df for k in ks): continue
                per = [dict(S.stats(df[k], 'AB_V7_B10', j), **seg3(df[k], j)) for j, k in enumerate(ks)]
                o = S.jsumm(per)
                TAB[f'{v}|{eg}|{c}|{t}'] = {x: o[x] for x in ('exc_med', 'exc_p10', 't_0050_med', 'exc_Y19on_med', 'CAGR_med', 'maxDD_med', 'exc_S0612_med', 'exc_S1318_med', 'exc_S19on_med', 'exc_HOLD_med') if x in o}
                TAB[f'{v}|{eg}|{c}|{t}']['CAGR_by_j'] = o['CAGR_by_j']
    sett = sorted({x for v in infos.values() for x in v.get('cons_main', [])})
    res = dict(tag=TAG, runtime_s=time.time() - T0, reproduce_R4_max=max(rep.values()) if rep else None, table=TAB,
               settled_codes_REAL_opt=sorted({x for k, v in infos.items() if k.startswith('REAL_opt') for x in v.get('cons_main', [])}))
    (HERE / f'r6_v7x_sim{TAG}.json').write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding='utf-8')
    for k, v in TAB.items(): print(k, {a: round(b, 3) for a, b in v.items() if a != 'CAGR_by_j'})
    print('R4 max', res['reproduce_R4_max'], 'runtime', round(res['runtime_s']))
