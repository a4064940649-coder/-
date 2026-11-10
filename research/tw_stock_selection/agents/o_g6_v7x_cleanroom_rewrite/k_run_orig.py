# K：原引擎（cz_v7x.CSim，只讀 import）重跑 4 種名單 × {100 萬一般／壓力、300 萬、1,000 萬、30 萬標準}，seed 平均淨值與每 sim 會計彙總存密封區
import sys, json, time, pickle
from pathlib import Path
from multiprocessing import Pool
import numpy as np, pandas as pd
AG = Path('/home/user/-/research/tw_stock_selection/agents')
SCR = '/tmp/claude-0/-home-user--/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/'
sys.path.insert(0, str(AG / 'o_engine_delist_consideration')); import cz_v7x as CV; import cz_engine as CE; sys.path.pop(0)
D = CV.D
sys.path.insert(0, str(CV.V7X)); import x_stats as S; sys.path.pop(0)
VR = json.loads((CV.V7X / 'results.json').read_text())
BASE = dict(VR['PREREG']['orig_cfg']); BASE.update(dict(f1=True, f2=True, f3=0.5, f4=True))
E = pd.read_csv(AG / 'o_engine_delist_consideration/events_v2.csv', dtype={'code': str, 'successor': str})
R2 = json.loads((AG / 'o_engine_delist_consideration/results.json').read_text(encoding='utf-8'))
dd = R2['PREREG']['events_v2']['summary']['d']['d_used']
kw = dict(TR=D.TR, IDX=D.IDX, Cn=D.C, CI=D.CI, On=D.O)
ST = {'cons_main': CE.build_settlement(E, 'main', d=dd, **kw)}
CELLS = {'N100': ('normal', 1e6), 'S100': ('stress', 1e6), 'N300': ('normal', 3e6), 'N1000': ('normal', 1e7), 'O30': ('normal', 3e5)}
KD = pickle.load(open(SCR + 'g6K_lists/k_v7x_plans.pkl', 'rb'))
def to_plan(pl):
    return sorted([(int(D.TPOS[pd.Timestamp(e)]), np.array([D.CI[c] for c in pk], int), D.TPOS[pd.Timestamp(sg)]) for sg, e, pk in pl], key=lambda x: x[0])
def job(a):
    v, j, s = a; pl = to_plan(KD['variants'][v][(j, s)]); out = {}
    eg = 'orig' if v == 'orig' else 'cons_main'
    for cn, (c, t) in CELLS.items():
        cfg = dict(tier=t, mode='GROW', model='K5T', cost=c); cfg.update(BASE)
        sm = CV.CSim(pl, cfg, settle=ST.get(eg), cmode='orig' if eg == 'orig' else 'consideration')
        nav = sm.run(); out[cn] = (nav, {k: v_ for k, v_ in sm.acc.items() if isinstance(v_, (int, float))})
    return v, j, s, out
if __name__ == '__main__':
    t0 = time.time()
    jobs = [(v, j, s) for v in KD['variants'] for (j, s) in KD['variants'][v]]
    navs = {}; acc = {}
    with Pool(4) as P:
        for n, (v, j, s, out) in enumerate(P.imap_unordered(job, jobs, chunksize=2)):
            for cn, (nav, a) in out.items():
                navs.setdefault((v, j, cn), []).append((s, nav)); acc[(v, j, s, cn)] = a
            if n % 100 == 0: print(n, len(jobs), round(time.time() - t0), flush=True)
    avg = {k: S.avg_nav([nv for s, nv in sorted(L, key=lambda x: x[0])]) for k, L in navs.items()}
    ends = {j: str(S.end_day('AB_V7_B10', j).date()) for j in range(13)}
    pickle.dump({'avg': avg, 'acc': acc, 'ends': ends}, open(SCR + 'g6_sealed/k_orig_engine.pkl', 'wb'))
    print('done', round(time.time() - t0))
