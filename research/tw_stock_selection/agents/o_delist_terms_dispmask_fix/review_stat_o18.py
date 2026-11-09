#!/usr/bin/env python3
"""反方審查（資料與統計）o18 題 2：不修改研究員檔案，只讀 import cz3／cz_v7x。
情境（V7X 100 萬一般成本 13 j × 20 seeds，j 中位數全期超額）：
  A_draft3 ：SPEC_draft 嚴格讀法（v2 新增的中間狀態不存在）：TYPE_TEXT+MARKET_NONE（3063、1716、2452）→ unknown（lo 歸零）
  B_draft5 ：A ＋ TENDER_TEXT+MARKET_NONE（3007、6280）→ unknown
  C_strict ：B ＋ 6 起 WebSearch 摘要核實降為市場證據（同 s12b）
python3 -I -B review_stat_o18.py"""
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

def mod(E, zero=(), ws=False):
    E = E.copy()
    for c in zero:
        i = E.index[E.code == c][0]; E.at[i, 'settle'] = 'unknown'; E.at[i, 'text_status'] = 'REV_ZERO'
    if ws:
        for c, s in {'2432': 'track', '3009': 'track', '3012': 'track', '2394': 'track', '2473': 'fixed', '8078': 'fixed'}.items():
            i = E.index[E.code == c][0]; E.at[i, 'settle'] = s; E.at[i, 'text_status'] = 'WS_DOWN'
    return E

Z3 = ('3063', '1716', '2452'); Z5 = Z3 + ('3007', '6280')
kw = dict(TR=D.TR, IDX=D.IDX, Cn=D.C, CI=D.CI, On=D.O)
ST = {'A_draft3_lo': cz3.build(E2, mod(E3, Z3), 'lo', **kw),
      'B_draft5_lo': cz3.build(E2, mod(E3, Z5), 'lo', **kw),
      'C_strict_lo': cz3.build(E2, mod(E3, Z5, ws=True), 'lo', **kw),
      'R_v3_lo': cz3.build(E2, E3, 'lo', **kw)}
cfg0 = dict(tier=1e6, mode='GROW', model='K5T', cost='normal'); cfg0.update(BASE)

def job(a):
    key, pl = a
    return key, {eg: CV.CSim(pl, dict(cfg0), settle=ST[eg], cmode='consideration').run() for eg in ST}

if __name__ == '__main__':
    t0 = time.time()
    v7 = json.loads((HERE / 'SPEC.json').read_text())['PREREG']['event_list']['V7X_settled_20']
    ev = E3.set_index('code')
    ver = [c for c in v7 if (c in ev.index and ev.at[c, 'text_status'] == 'VERIFIED_TEXT') or c in ('2311', '2325', '3474', '6286')]
    verb = [c for c in ver if c in ('2311', '2325', '3474', '6286') or ev.at[c, 'src_kind'] != 'websearch_summary']
    print('V7X verified', len(ver), ver, '| excl stale-ratio 2394:', len([c for c in ver if c != '2394']), '| verbatim only:', len(verb))
    P = D.plans('AB_V7_B10', js=list(range(13)), seeds=list(range(20)))
    navs = {}
    with Pool(4) as pool:
        for key, o in pool.imap_unordered(job, sorted(P.items()), chunksize=2):
            for eg, nv in o.items(): navs.setdefault((key[0], eg), []).append((key[1], nv))
    TAB = {}
    for eg in ST:
        per = [S.stats(S.avg_nav([nv for s, nv in sorted(navs[(j, eg)], key=lambda x: x[0])]), 'AB_V7_B10', j) for j in range(13)]
        sm = S.jsumm(per)
        TAB[eg] = {k: round(float(sm[k]), 4) for k in ('exc_med', 't_0050_med', 'exc_p10', 'exc_SEL_med', 'exc_VAL_med', 'exc_Y19on_med') if k in sm}
        TAB[eg]['exc_by_j'] = [round(float(p['exc']), 3) for p in per]
    print(json.dumps(TAB, ensure_ascii=False)); print('runtime', round(time.time() - t0), 's')
