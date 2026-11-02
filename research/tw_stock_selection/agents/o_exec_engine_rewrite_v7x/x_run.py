#!/usr/bin/env python3
"""V7X 批次模擬：每個 (j, seed) 名單跑全部格子，格內 seeds 日報酬平均成一條淨值。"""
import sys, json, time
from pathlib import Path
from multiprocessing import Pool
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
import x_data as D, x_exec as X, x_stats as S

OUT = Path(__file__).resolve().parent / 'out'
TIERS = {'30萬': 3e5, '100萬': 1e6, '300萬': 3e6, '1000萬': 1e7}
LAB = {'AB_V7_B10': 'B10', 'AB_V7': 'AB'}


def cells(models=('K5T', 'TICK'), costs=('normal', 'stress'), tiers=tuple(TIERS), modes=('FIX', 'GROW')):
    return [(m, c, t, v) for m in models for c in costs for t in tiers for v in modes]


_G = {}


def _job(args):
    key, pl = args
    out = {}; accs = {}
    for (m, c, t, v) in _G['cells']:
        cfg = dict(tier=TIERS[t], mode=v, model=m, cost=c); cfg.update(_G['extra'])
        nav, acc = X.run_one(pl, cfg) if _G['t1'] is None else _run_t1(pl, cfg, _G['t1'])
        out[(m, c, t, v)] = nav; accs[(m, c, t, v)] = acc
    return key, out, accs


def _run_t1(pl, cfg, t1):
    s = X.Sim(pl, cfg); nav = s.run(t1=t1)
    return nav, s.acc


def run_grid(tag, strat='AB_V7_B10', js=range(13), seeds=range(20), cl=None, extra=None, t1=None, official=False, save=True):
    t0 = time.time()
    cl = cl or cells(); extra = extra or {}
    _G.update(cells=cl, extra=extra, t1=t1)
    P = D.plans(strat, js=None if official else list(js), seeds=list(seeds), official=official)
    jobs = sorted(P.items(), key=lambda kv: (str(kv[0][0]), kv[0][1]))
    navs = {}; accs = {}
    with Pool(4) as pool:
        for key, out, ac in pool.imap_unordered(_job, jobs, chunksize=1):
            for ck, n in out.items():
                navs.setdefault((key[0],) + ck, []).append(n); accs.setdefault((key[0],) + ck, []).append(ac[ck])
    cols = {}; acc_avg = {}
    for k, L in navs.items():
        j, m, c, t, v = k
        name = f'{LAB[strat]}|{"OFF" if j == "OFF" else "J" + str(j)}|{m}|{c}|{t}|{v}'
        cols[name] = S.avg_nav(L)
        A = accs[k]
        acc_avg[name] = {kk: float(np.nanmean([a[kk] for a in A])) for kk in A[0]}
    df = pd.DataFrame(cols).sort_index(axis=1)
    if save:
        df.to_parquet(OUT / f'navs_{tag}.parquet')
        (OUT / f'acc_{tag}.json').write_text(json.dumps(acc_avg, ensure_ascii=False, indent=0))
    print(tag, 'done', len(cols), 'cells', round(time.time() - t0, 1), 's', flush=True)
    return df, acc_avg


def table(df, strat='AB_V7_B10', sel_only=False):
    """每格 j 統計與 j 中位數。"""
    lab = LAB[strat]; per = {}
    for c in df.columns:
        st, j, m, co, t, v = c.split('|')
        jj = 'OFF' if j == 'OFF' else int(j[1:])
        if sel_only:
            nav = df[c].dropna(); a = nav.index[0]; e = nav.index[nav.index <= '2018-12-31'][-1]
            s = dict(CAGR_SEL=S.cagr(nav, a, e), E50_CAGR_SEL=S.cagr(S.E50, a, e))
            x = S.windows(nav, strat, jj); s['t_SEL'] = S.tstat(x[(x.index <= '2018-12-31')])
        else:
            s = S.stats(df[c], strat, jj)
        per.setdefault((m, co, t, v), {})[jj] = s
    out = {}
    for k, d in per.items():
        js = sorted(x for x in d if x != 'OFF')
        if js:
            out['|'.join(k)] = S.jsumm([d[x] for x in js])
        if 'OFF' in d:
            out['OFF|' + '|'.join(k)] = d['OFF']
    return out
