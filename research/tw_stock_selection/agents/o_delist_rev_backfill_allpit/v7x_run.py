#!/usr/bin/env python3
"""V7X 模擬（正式；只跑一次）：讀 scratch/plans_v7x.csv，以 o_engine_delist_consideration/cz_v7x.CSim（只讀 import）跑
K5T × GROW × {一般, 壓力} × {100萬, 300萬, 1000萬}；引擎見 PREREG.v7x.engines。seeds 日報酬平均成一條淨值（x_stats.avg_nav）。
orig 名單 × orig 引擎必須重現 o_exec_engine_rewrite_v7x/out/navs_oos_R4.parquet（差 ≤ 1e−12）。python3 -I -B v7x_run.py"""
import sys, json, hashlib, time
from pathlib import Path
from multiprocessing import Pool
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent; AG = HERE.parent
def sha(f): return hashlib.sha256(Path(f).read_bytes()).hexdigest()
R = json.loads((HERE / 'results.json').read_text(encoding='utf-8'))
assert 'PREREG' in R and hashlib.sha256(json.dumps(R['PREREG'], ensure_ascii=False, sort_keys=True).encode()).hexdigest() == R['PREREG_sha256']
PR = R['PREREG']
for f, h in PR['file_sha256'].items(): assert sha(HERE / f) == h, f
assert 'RESULTS_V7X_PLANS' in R and R['RESULTS_V7X_PLANS']['orig_reproduces_plans_j7'] and 'RESULTS_V7X' not in R
PF = Path(R['RESULTS_V7X_PLANS']['file']); assert sha(PF) == R['RESULTS_V7X_PLANS']['sha256']
sys.path.insert(0, str(AG / 'o_engine_delist_consideration')); import cz_v7x as CV; import cz_engine as CE; sys.path.pop(0)
D, X = CV.D, CV.X
sys.path.insert(0, str(CV.V7X)); import x_stats as S; sys.path.pop(0)
VR = json.loads((CV.V7X / 'results.json').read_text())
BASE = dict(VR['PREREG']['orig_cfg']); BASE.update(dict(f1=True, f2=True, f3=0.5, f4=True))
E = pd.read_csv(AG / 'o_engine_delist_consideration/events_v2.csv', dtype={'code': str, 'successor': str})
assert sha(AG / 'o_engine_delist_consideration/events_v2.csv').startswith('66e80725')
R2 = json.loads((AG / 'o_engine_delist_consideration/results.json').read_text(encoding='utf-8'))
dd = R2['PREREG']['events_v2']['summary']['d']['d_used']
kw = dict(TR=D.TR, IDX=D.IDX, Cn=D.C, CI=D.CI, On=D.O)
ST = {'cons_main': CE.build_settlement(E, 'main', d=dd, **kw), 'cons_lo': CE.build_settlement(E, 'lo', d=dd, **kw), 'cons_hi': CE.build_settlement(E, 'hi', d=dd, **kw)}
TIERS = {'100萬': 1e6, '300萬': 3e6, '1000萬': 1e7}
CELLS = [(c, t) for c in ('normal', 'stress') for t in TIERS]
ENG = PR['v7x']['engines']          # {variant: [engines]}
PL = pd.read_csv(PF, dtype={'picks': str, 'inserted': str})
def to_plan(g): return sorted([(int(e), np.array([D.CI[c] for c in pk.split(',')], int), D.TPOS[pd.Timestamp(sg)]) for e, pk, sg in zip(g.exec_idx, g.picks, g.signal)], key=lambda x: x[0])
def job(a):
    v, j, s, pl = a
    out = {}; info = {}
    for c, t in CELLS:
        cfg = dict(tier=TIERS[t], mode='GROW', model='K5T', cost=c); cfg.update(BASE)
        for eg in ENG[v]:
            sm = CV.CSim(pl, cfg, settle=ST.get(eg), cmode='orig' if eg == 'orig' else 'consideration')
            out[(c, t, eg)] = sm.run()
            if c == 'normal' and t == '100萬' and eg == 'cons_main': info = dict(n_settled=sm.acc['n_settled'], n_fallback=sm.acc['n_fallback'], ev=sorted({e['code'] for e in sm.log}))
    return v, j, s, out, info
if __name__ == '__main__':
    t0 = time.time()
    jobs = [(v, int(j), int(s), to_plan(g)) for (v, j, s), g in PL.groupby(['variant', 'j', 'seed'])]
    navs = {}; infos = {}
    with Pool(4) as P:
        for n, (v, j, s, out, info) in enumerate(P.imap_unordered(job, jobs, chunksize=1)):
            for ck, nv in out.items(): navs.setdefault((v, j) + ck, []).append((s, nv))
            infos[f'{v}|J{j}|s{s}'] = info
            if n % 50 == 0: print(n, len(jobs), round(time.time() - t0), flush=True)
    cols = {}
    for (v, j, c, t, eg), L in navs.items():
        cols[f'{v}|J{j}|{c}|{t}|{eg}'] = S.avg_nav([nv for s, nv in sorted(L, key=lambda x: x[0])])
    df = pd.DataFrame(cols).sort_index(axis=1)
    df.to_parquet(Path(PR['scratch']) / 'navs_v7x.parquet')
    R4 = pd.read_parquet(CV.V7X / 'out' / 'navs_oos_R4.parquet')
    rep = {}
    for j in range(13):
        for c, t in CELLS:
            a = df[f'orig|J{j}|{c}|{t}|orig']; b = R4[f'B10|J{j}|K5T|{c}|{t}|GROW'].reindex(a.index)
            rep[f'J{j}|{c}|{t}'] = float(np.nanmax(np.abs(a.to_numpy() - b.to_numpy())))
    def segx(nav, j, a, b):
        e = S.end_day('AB_V7_B10', j); nav = nav.loc[:e].dropna()
        x = nav[(nav.index >= a) & (nav.index <= b)]
        if len(x) < 20: return np.nan
        prev = nav[nav.index < x.index[0]]; s0 = prev.index[-1] if len(prev) else x.index[0]
        return S.cagr(nav, s0, x.index[-1]) - S.cagr(S.E50, s0, x.index[-1])
    TAB = {}
    for v in ENG:
        for eg in ENG[v]:
            for c, t in CELLS:
                per = [S.stats(df[f'{v}|J{j}|{c}|{t}|{eg}'], 'AB_V7_B10', j) for j in range(13)]
                T = S.jsumm(per)
                for nm, (a, b) in {'S0612': ('2006-01-01', '2012-12-31'), 'S1318': ('2013-01-01', '2018-12-31'), 'S19on': ('2019-01-01', '2026-12-31')}.items():
                    xs = np.array([segx(df[f'{v}|J{j}|{c}|{t}|{eg}'], j, a, b) for j in range(13)])
                    T[f'exc_{nm}_med'] = float(np.nanmedian(xs)); T[f'exc_{nm}_p10'] = float(np.nanpercentile(xs, 10))
                TAB[f'{v}|{eg}|{c}|{t}'] = T
    yd = {}
    for v in ENG:
        if v == 'orig': continue
        Y = pd.DataFrame({j: pd.Series(S.yearly_exc(df[f'{v}|J{j}|normal|100萬|cons_main'], 'AB_V7_B10', j)) - pd.Series(S.yearly_exc(df[f'orig|J{j}|normal|100萬|orig'], 'AB_V7_B10', j)) for j in range(13)})
        yd[v] = {int(y): float(x) for y, x in Y.median(axis=1).items()}
    R = json.loads((HERE / 'results.json').read_text(encoding='utf-8'))
    assert 'RESULTS_V7X' not in R
    R['RESULTS_V7X'] = dict(utc=pd.Timestamp.now('UTC').isoformat(), runtime_s=round(time.time() - t0, 1), reproduce_R4_max=max(rep.values()), reproduce_R4=rep,
                            table=TAB, yearly_delta_vs_official_med_100m_normal=yd, settle_info_sample={k: v for k, v in list(infos.items())[:40]},
                            navs_file=str(Path(PR['scratch']) / 'navs_v7x.parquet'))
    (HERE / 'results.json').write_text(json.dumps(R, ensure_ascii=False, indent=1, default=str), encoding='utf-8')
    F = ['exc_med', 'exc_p10', 't_0050_med', 'exc_Y19on_med', 't_Y19on_med', 'exc_S1318_med']
    for k, T in TAB.items():
        if '|normal|100萬' in k or k.endswith('|cons_main|stress|100萬') or ('cons_main|normal' in k): print(k, {f: round(T[f], 3) for f in F})
    print('R4 reproduce max', max(rep.values()), round(time.time() - t0))
