#!/usr/bin/env python3
"""凍結前煙霧測試（V7X）：orig 與 o17 cons_main（events_v2）重現 o17 公開數字（100 萬一般 j 中位數全期超額 1.6096→1.5720、t 0.5462→0.5341）；
處置遮罩 dmode='orig' 名單＝plans_j7.csv 逐筆相同（已由 v7plans.py 比對，這裡再斷言檔案）。不跑 v3、不跑新遮罩報酬。"""
import sys, json, time
from pathlib import Path
from multiprocessing import Pool
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent; AG = HERE.parent
sys.path.insert(0, str(AG / 'o_engine_delist_consideration')); import cz_v7x as CV, cz_engine as CE; sys.path.pop(0)
D = CV.D
sys.path.insert(0, str(CV.V7X)); import x_stats as S; sys.path.pop(0)
VR = json.loads((CV.V7X / 'results.json').read_text())
BASE = dict(VR['PREREG']['orig_cfg']); BASE.update(dict(f1=True, f2=True, f3=0.5, f4=True))
E2 = pd.read_csv(AG / 'o_engine_delist_consideration' / 'events_v2.csv', dtype={'code': str, 'successor': str})
ST = {'cons_main': CE.build_settlement(E2, 'main', d=0.057, TR=D.TR, IDX=D.IDX, Cn=D.C, CI=D.CI, On=D.O)}
cfg0 = dict(tier=1e6, mode='GROW', model='K5T', cost='normal'); cfg0.update(BASE)
def job(a):
    key, pl = a
    return key, {'orig': CV.CSim(pl, dict(cfg0), cmode='orig').run(), 'cons_main': CV.CSim(pl, dict(cfg0), settle=ST['cons_main'], cmode='consideration').run()}
if __name__ == '__main__':
    SC = Path('/tmp/claude-0/-home-user--/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/o18_delistterms')
    a = pd.read_csv(SC / 'plans_orig.csv', dtype={'picks': str}); b = pd.read_csv(AG / 'o_v7_listing_exec_engine' / 'data' / 'plans_j7.csv', dtype={'picks': str})
    b = b[(b.strat == 'AB_V7_B10') & (b.seed < 20)].reset_index(drop=True)
    assert a[['j', 'seed', 'signal', 'exec_idx', 'picks']].equals(b[['j', 'seed', 'signal', 'exec_idx', 'picks']])
    P = D.plans('AB_V7_B10', js=list(range(13)), seeds=list(range(20)))
    navs = {}
    with Pool(4) as pool:
        for key, o in pool.imap_unordered(job, sorted(P.items()), chunksize=2):
            for eg, nv in o.items(): navs.setdefault((key[0], eg), []).append((key[1], nv))
    R4 = pd.read_parquet(CV.V7X / 'out' / 'navs_oos_R4.parquet')
    TAB = {}; rep = 0.0
    for eg in ('orig', 'cons_main'):
        per = []
        for j in range(13):
            nv = S.avg_nav([x for s, x in sorted(navs[(j, eg)], key=lambda x: x[0])])
            if eg == 'orig': rep = max(rep, float(np.nanmax(np.abs(nv.to_numpy() - R4[f'B10|J{j}|K5T|normal|100萬|GROW'].reindex(nv.index).to_numpy()))))
            per.append(S.stats(nv, 'AB_V7_B10', j))
        TAB[eg] = {k: v for k, v in S.jsumm(per).items() if k in ('exc_med', 't_0050_med')}
    out = dict(utc=pd.Timestamp.now('UTC').isoformat(), table=TAB, reproduce_R4_max=rep,
               ok=bool(abs(TAB['orig']['exc_med'] - 1.6096) < 1e-3 and abs(TAB['cons_main']['exc_med'] - 1.5720) < 1e-3 and abs(TAB['orig']['t_0050_med'] - 0.5462) < 1e-3 and abs(TAB['cons_main']['t_0050_med'] - 0.5341) < 1e-3))
    assert out['ok'], out
    (HERE / 'smoke_v7x.json').write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding='utf-8'); print(out)
