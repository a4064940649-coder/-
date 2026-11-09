#!/usr/bin/env python3
"""事前可行性檢查（SPEC 第 8 步要求；凍結前唯一允許的報酬計算，只算 V7X 100 萬一般成本 j 中位數全期超額的 lo）：
 best＝未核實事件一律 lc×(1−δ_evt) 固定請求權（不歸零、不承擔空窗期換股風險）；worst＝非 VERIFIED_TEXT 一律歸零（lo）。
 若 best_lo ≤ 0 →「下界回到 > 0 無法觸發」。python3 -I -B s6_feasibility.py"""
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
kw = dict(TR=D.TR, IDX=D.IDX, Cn=D.C, CI=D.CI, On=D.O)
ST = {'best_lo': cz3.build(E2, E3, 'lo', scenario='best', **kw), 'worst_lo': cz3.build(E2, E3, 'lo', scenario='worst', **kw)}
cfg0 = dict(tier=1e6, mode='GROW', model='K5T', cost='normal'); cfg0.update(BASE)
def job(a):
    key, pl = a
    return key, {eg: CV.CSim(pl, dict(cfg0), settle=ST[eg], cmode='consideration').run() for eg in ST}
if __name__ == '__main__':
    t0 = time.time()
    P = D.plans('AB_V7_B10', js=list(range(13)), seeds=list(range(20)))
    navs = {}
    with Pool(4) as pool:
        for key, o in pool.imap_unordered(job, sorted(P.items()), chunksize=2):
            for eg, nv in o.items(): navs.setdefault((key[0], eg), []).append((key[1], nv))
    TAB = {}
    for eg in ST:
        per = [S.stats(S.avg_nav([nv for s, nv in sorted(navs[(j, eg)], key=lambda x: x[0])]), 'AB_V7_B10', j) for j in range(13)]
        TAB[eg] = {k: v for k, v in S.jsumm(per).items() if k in ('exc_med', 't_0050_med', 'exc_p10')}
    out = dict(utc=pd.Timestamp.now('UTC').isoformat(), runtime_s=round(time.time() - t0), table=TAB,
               rule='best_lo ≤ 0 → 下界 > 0 無法觸發', triggerable=bool(TAB['best_lo']['exc_med'] > 0),
               note='o17 已公開：cons_lo +0.058pp（研究員分類）、−0.69pp（2432 判 unknown）；本檢查用 δ_evt（0.096／2013 前 0.12）')
    (HERE / 'feasibility.json').write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding='utf-8')
    print(json.dumps(out, ensure_ascii=False, indent=1))
