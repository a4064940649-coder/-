#!/usr/bin/env python3
"""事後敏感度（不得推翻判定）：把 V7X 結算事件中「只靠 WebSearch 摘要」核實的 6 起（2432、3009、3012、2394、2473、8078）降級為市場證據
（TRACK→市場比例×(1±δ_evt)、LOCK→lc×(1±δ_evt)），重算 V7X 100 萬一般 j 中位數全期超額 main／lo／hi。python3 -I -B s12b_posthoc_wsdown.py"""
import sys, json, time
from pathlib import Path
from multiprocessing import Pool
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent; AG = HERE.parent
sys.path.insert(0, str(AG / 'o_engine_delist_consideration')); import cz_v7x as CV; sys.path.pop(0)
D = CV.D
sys.path.insert(0, str(CV.V7X)); import x_stats as S; sys.path.pop(0)
sys.path.insert(0, str(HERE)); import cz3; sys.path.pop(0)
VR = json.loads((CV.V7X / 'results.json').read_text())
BASE = dict(VR['PREREG']['orig_cfg']); BASE.update(dict(f1=True, f2=True, f3=0.5, f4=True))
E2, E3 = cz3.load_E()
WS = {'2432': 'track', '3009': 'track', '3012': 'track', '2394': 'track', '2473': 'fixed', '8078': 'fixed'}
E4 = E3.copy()
for c, s in WS.items():
    i = E4.index[E4.code == c][0]; E4.at[i, 'settle'] = s; E4.at[i, 'text_status'] = 'WS_DOWNGRADED'
kw = dict(TR=D.TR, IDX=D.IDX, Cn=D.C, CI=D.CI, On=D.O)
ST = {b: cz3.build(E2, E4, b, **kw) for b in ('main', 'lo', 'hi')}
cfg0 = dict(tier=1e6, mode='GROW', model='K5T', cost='normal'); cfg0.update(BASE)
def job(a):
    key, pl = a
    return key, {b: CV.CSim(pl, dict(cfg0), settle=ST[b], cmode='consideration').run() for b in ST}
if __name__ == '__main__':
    P = D.plans('AB_V7_B10', js=list(range(13)), seeds=list(range(20))); navs = {}
    with Pool(4) as pool:
        for key, o in pool.imap_unordered(job, sorted(P.items()), chunksize=2):
            for b, nv in o.items(): navs.setdefault((key[0], b), []).append((key[1], nv))
    TAB = {}
    for b in ST:
        per = [S.stats(S.avg_nav([nv for s, nv in sorted(navs[(j, b)], key=lambda x: x[0])]), 'AB_V7_B10', j) for j in range(13)]
        TAB[b] = {k: v for k, v in S.jsumm(per).items() if k in ('exc_med', 't_0050_med', 'exc_p10')}
    R = json.loads((HERE / 'results.json').read_text(encoding='utf-8'))
    R.setdefault('POSTHOC_DESC', {})['websearch_downgraded_V7X_100m_normal'] = dict(events=WS, table=TAB, note='事後敏感度：只有 o17 逐字新聞核實的 4 起（2311、2325、6286、3474）依條款，其餘全部用市場證據 ±δ_evt')
    (HERE / 'results.json').write_text(json.dumps(R, ensure_ascii=False, indent=1, default=str), encoding='utf-8')
    print(TAB)
