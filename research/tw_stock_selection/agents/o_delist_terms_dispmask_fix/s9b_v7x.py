#!/usr/bin/env python3
"""正式執行（只跑一次）：V7X 可執行（AB_V7_B10 × 13 j × 20 seeds × K5T × GROW；R4 設定＝orig_cfg＋f1,f2,f3=0.5,f4）。
名單：orig／union／replace（v7plans.py 產出、sha 已凍結）。引擎：orig（現行 A16）、v3_main／lo／hi（cz3）、o17 cons_main（只 100 萬一般，對照）。
格：一般 100 萬、壓力 100 萬、一般 300 萬、一般 1,000 萬。斷言 PREREG 存在且雜湊相符、RESULTS_V7X 尚不存在。python3 -I -B s9b_v7x.py"""
import sys, json, hashlib, time
from pathlib import Path
from multiprocessing import Pool
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent; AG = HERE.parent
SC = Path('/tmp/claude-0/-home-user--/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/o18_delistterms')
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
RJ = HERE / 'results.json'
R = json.loads(RJ.read_text(encoding='utf-8'))
assert hashlib.sha256(json.dumps(R['PREREG'], ensure_ascii=False, sort_keys=True).encode()).hexdigest() == R['PREREG_sha256']
assert 'RESULTS_V7X' not in R, '只跑一次'
IS = R['PREREG']['input_sha256']
for f in ('events_v3.csv', 'cz3.py', 'dispmask_official.py', 'v7plans.py'): assert sha(HERE / f) == IS[f], f
for k in ('plans_orig.csv', 'plans_union.csv', 'plans_replace.csv'): assert sha(SC / k) == IS[k], k
sys.path.insert(0, str(AG / 'o_engine_delist_consideration')); import cz_v7x as CV, cz_engine as CE; sys.path.pop(0)
D = CV.D
for k in ('x_exec.py', 'x_data.py', 'x_stats.py', 'cz_v7x.py', 'cz_engine.py', 'events_v2.csv'):
    p = {'cz_v7x.py': AG / 'o_engine_delist_consideration' / 'cz_v7x.py', 'cz_engine.py': AG / 'o_engine_delist_consideration' / 'cz_engine.py',
         'events_v2.csv': AG / 'o_engine_delist_consideration' / 'events_v2.csv'}.get(k, CV.V7X / k)
    assert sha(p) == IS[k], k
sys.path.insert(0, str(CV.V7X)); import x_stats as S; sys.path.pop(0)
sys.path.insert(0, str(HERE)); import cz3; sys.path.pop(0)
VR = json.loads((CV.V7X / 'results.json').read_text())
BASE = dict(VR['PREREG']['orig_cfg']); BASE.update(dict(f1=True, f2=True, f3=0.5, f4=True))
E2, E3 = cz3.load_E()
kw = dict(TR=D.TR, IDX=D.IDX, Cn=D.C, CI=D.CI, On=D.O)
ST = {'v3_main': cz3.build(E2, E3, 'main', **kw), 'v3_lo': cz3.build(E2, E3, 'lo', **kw), 'v3_hi': cz3.build(E2, E3, 'hi', **kw),
      'o17_cons_main': CE.build_settlement(E2, 'main', d=0.057, **kw)}
COMBO = {'orig': ['orig', 'v3_main', 'v3_lo', 'v3_hi', 'o17_cons_main'], 'union': ['orig', 'v3_main', 'v3_lo', 'v3_hi'], 'replace': ['orig', 'v3_main']}
TIERS = {'100萬': 1e6, '300萬': 3e6, '1000萬': 1e7}
CELLS = [('normal', '100萬'), ('stress', '100萬'), ('normal', '300萬'), ('normal', '1000萬')]

def load_plans(f):
    d = pd.read_csv(f, dtype={'picks': str}); out = {}
    for (j, s), g in d.groupby(['j', 'seed'], sort=False):
        out[(int(j), int(s))] = sorted([(int(e), np.array([D.CI[c] for c in pk.split(',')], int), D.TPOS[pd.Timestamp(sg)]) for sg, e, pk in zip(g.signal, g.exec_idx, g.picks)], key=lambda x: x[0])
    return out
PL = {k: load_plans(SC / f'plans_{k}.csv') for k in COMBO}

def job(key):
    o = {}; info = {}
    for pk, egs in COMBO.items():
        pl = PL[pk][key]
        for c, t in CELLS:
            cfg = dict(tier=TIERS[t], mode='GROW', model='K5T', cost=c); cfg.update(BASE)
            for eg in egs:
                if eg == 'o17_cons_main' and (c, t) != ('normal', '100萬'): continue
                sm = CV.CSim(pl, dict(cfg), settle=ST.get(eg), cmode='orig' if eg == 'orig' else 'consideration')
                o[(pk, eg, c, t)] = sm.run().pct_change()
                if (c, t) == ('normal', '100萬') and eg != 'orig':
                    info[f'{pk}|{eg}'] = dict(n_settled=sm.acc['n_settled'], n_fallback=sm.acc['n_fallback'],
                                             ev=[(e['code'], e['type'], e['date'], round(e['value_lastclose'], 1), [(p[0], p[1], round(p[-1], 1)) for p in e['parts']]) for e in sm.log])
    return key, o, info

if __name__ == '__main__':
    t0 = time.time()
    keys = sorted(PL['orig'])
    SUM = {}; CNT = {}; INFO = {}
    with Pool(4) as pool:
        for n, (key, o, info) in enumerate(pool.imap_unordered(job, keys, chunksize=1)):
            for ck, r in o.items():
                kk = (key[0],) + ck
                SUM[kk] = r if kk not in SUM else SUM[kk].add(r, fill_value=0.0)
                CNT[kk] = r.notna().astype(float) if kk not in CNT else CNT[kk].add(r.notna().astype(float), fill_value=0.0)
            INFO[f'J{key[0]}|s{key[1]}'] = info
            if n % 26 == 0: print(n, round(time.time() - t0), 's', flush=True)
    NAV = {kk: (1 + (SUM[kk] / CNT[kk].replace(0, np.nan)).fillna(0)).cumprod() for kk in SUM}
    pd.DataFrame({'|'.join(map(str, k)): v for k, v in NAV.items()}).to_parquet(SC / 'navs_v7x_o18.parquet')
    R4 = pd.read_parquet(CV.V7X / 'out' / 'navs_oos_R4.parquet')
    rep = max(float(np.nanmax(np.abs(NAV[(j, 'orig', 'orig', c, t)].to_numpy() - R4[f'B10|J{j}|K5T|{c}|{t}|GROW'].reindex(NAV[(j, 'orig', 'orig', c, t)].index).to_numpy()))) for j in range(13) for c, t in CELLS)
    E50 = pd.Series(D.E50, index=D.IDX)
    def segx(nav, a, b):
        n = nav.loc[a:b].dropna(); e = E50.reindex(n.index)
        yrs = (n.index[-1] - n.index[0]).days / 365.25
        return ((n.iloc[-1] / n.iloc[0]) ** (1 / yrs) - (e.iloc[-1] / e.iloc[0]) ** (1 / yrs)) * 100
    TAB = {}; YR = {}
    FIELDS = ('CAGR', 'exc', 't_0050', 'exc_Y19on', 't_Y19on', 'maxDD')
    for pk, egs in COMBO.items():
        for eg in egs:
            for c, t in CELLS:
                if eg == 'o17_cons_main' and (c, t) != ('normal', '100萬'): continue
                per = []
                for j in range(13):
                    nv = NAV[(j, pk, eg, c, t)]; s = S.stats(nv, 'AB_V7_B10', j)
                    e = S.end_day('AB_V7_B10', j)
                    s['exc_0612'] = segx(nv, '2006-01-01', '2012-12-31'); s['exc_1318'] = segx(nv, '2013-01-01', '2018-12-31')
                    s['exc_1419'] = segx(nv, '2014-01-01', '2019-12-31'); s['exc_19e'] = segx(nv.loc[:e], '2019-01-01', str(e.date()))
                    per.append({k: s[k] for k in FIELDS + ('exc_0612', 'exc_1318', 'exc_1419', 'exc_19e')})
                    YR[(pk, eg, c, t, j)] = S.yearly_exc(nv, 'AB_V7_B10', j)
                TAB[f'{pk}|{eg}|{c}|{t}'] = S.jsumm(per)
    def ydiff(a, b, c='normal', t='100萬'):
        Y = pd.DataFrame({j: pd.Series(YR[a + (c, t, j)]) - pd.Series(YR[b + (c, t, j)]) for j in range(13)})
        med = Y.median(axis=1)
        return dict(med={int(y): float(v) for y, v in med.items()}, sum_med=float(med.sum()), sum_abs_med=float(med.abs().sum()))
    PAIRS = {'v3_vs_orig': (('orig', 'v3_main'), ('orig', 'orig')), 'v3lo_vs_orig': (('orig', 'v3_lo'), ('orig', 'orig')), 'v3hi_vs_orig': (('orig', 'v3_hi'), ('orig', 'orig')),
             'v3_vs_o17cons': (('orig', 'v3_main'), ('orig', 'o17_cons_main')),
             'union_vs_orig(orig)': (('union', 'orig'), ('orig', 'orig')), 'replace_vs_orig(orig)': (('replace', 'orig'), ('orig', 'orig')),
             'union+v3_vs_orig': (('union', 'v3_main'), ('orig', 'orig')), 'union+v3lo_vs_orig': (('union', 'v3_lo'), ('orig', 'orig'))}
    YD = {k: ydiff(a, b) for k, (a, b) in PAIRS.items()}
    YD['union_vs_orig(orig)|stress'] = ydiff(('union', 'orig'), ('orig', 'orig'), 'stress', '100萬')
    R = json.loads(RJ.read_text(encoding='utf-8')); assert 'RESULTS_V7X' not in R
    R['RESULTS_V7X'] = dict(utc=pd.Timestamp.now('UTC').isoformat(), runtime_s=round(time.time() - t0, 1), reproduce_R4_max=rep, table=TAB, yearly=YD,
                            settle_info=INFO, navs_file=str(SC / 'navs_v7x_o18.parquet'), navs_sha256=sha(SC / 'navs_v7x_o18.parquet'))
    RJ.write_text(json.dumps(R, ensure_ascii=False, indent=1, default=str), encoding='utf-8')
    print('done', round(time.time() - t0), 's reproduce', rep)
