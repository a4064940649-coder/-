#!/usr/bin/env python3
"""第 2 步：模型迭代（只用挑選期 2006-04～2019-04 訊號窗；模擬在 2019-05 底截止）。最多 8 個變體，全部寫入 iterations.json。
事前寫死的挑選準則：
  主版本（合併池）＝100 萬 GROW、K5T、一般成本下，對上市版主策略（TW 前 300、n10、同 j）配對 t 的 j 中位數最高者；
    並要求「池大小 ±100」鄰近變體的配對差同號（不翻號），否則改選次佳。
  衛星版本（只上櫃池）＝30 萬 GROW（100 萬 × 30%）下對 0050 年化超額 j 中位數最高者。
不可調：分數權重、52 週與 3 月營收定義、換股日、成本模型、V7X 正式設定（F1～F4、K5T）。"""
import sys, json, time
from pathlib import Path
from multiprocessing import Pool
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import otlib as L
D = L.D
SELA, SELB = pd.Timestamp(L.SEL[0]), pd.Timestamp(L.SEL[1])
T1 = int(np.searchsorted(D.IDX, pd.Timestamp('2019-06-15')))
BASE = dict(pool='TW', K=300, n=10)
VARIANTS = {
    'v1_ALL300_n10': dict(pool='ALL', K=300, n=10),
    'v2_ALL400_n10': dict(pool='ALL', K=400, n=10),
    'v3_ALL500_n10': dict(pool='ALL', K=500, n=10),
    'v4_ALL400_n10_cap50': dict(pool='ALL', K=400, n=10, cap=0.5),
    'v5_ALL400_n10_floor2k': dict(pool='ALL', K=400, n=10, floor=2e7),
    'v6_ALL400_n15': dict(pool='ALL', K=400, n=15),
    'v7_TP200_n10': dict(pool='TP', K=200, n=10),
    'v8_TP300_n10': dict(pool='TP', K=300, n=10),
}
CELLS = [('100萬', 'GROW'), ('300萬', 'GROW'), ('1000萬', 'GROW'), ('30萬', 'GROW')]


def job(a):
    name, v, j = a
    pl = L.gen_plan(v, j)
    out = {}
    for t, m in CELLS:
        nav, acc = L.sim(pl, L.cfg(t, m), t1=T1)
        out[(t, m)] = (nav, acc['part_median'])
    return name, j, out


if __name__ == '__main__':
    t0 = time.time()
    pre = {'created': pd.Timestamp.utcnow().isoformat(), 'rule': __doc__, 'variants': VARIANTS, 'baseline': BASE}
    (HERE / 'iterations.json').write_text(json.dumps({'PLAN': pre}, ensure_ascii=False, indent=1))
    jobs = [('BASE', BASE, j) for j in range(13)] + [(k, v, j) for k, v in VARIANTS.items() for j in range(13)]
    R = {}
    with Pool(4) as P:
        for name, j, out in P.imap_unordered(job, jobs):
            R[(name, j)] = out
    res = {}
    for name in ['BASE'] + list(VARIANTS):
        rr = {}
        for t, m in CELLS:
            per = []
            for j in range(13):
                nav, pm = R[(name, j)][(t, m)]
                W = L.windows(nav, j); s = L.seg_stats(W, SELA, SELB)
                d = {}
                if name != 'BASE':
                    nb = R[('BASE', j)][(t, m)][0]
                    Wb = L.windows(nav, j, ref=nb); sb = L.seg_stats(Wb, SELA, SELB)
                    d = dict(pair_mean_pp=sb.get('mean_pp'), pair_t=sb.get('t'), pair_ann=sb.get('ann_x'))
                per.append(dict(j=j, ann=s.get('ann'), ann_x=s.get('ann_x'), t=s.get('t'), part_median=pm, **d))
            key = f'{t}|{m}'
            rr[key] = dict(ann_x=L.jsumm([p['ann_x'] for p in per]), t=L.jsumm([p['t'] for p in per]), ann=L.jsumm([p['ann'] for p in per]),
                           part_median=L.jsumm([p['part_median'] for p in per]))
            if name != 'BASE':
                rr[key].update(pair_t=L.jsumm([p['pair_t'] for p in per]), pair_mean_pp=L.jsumm([p['pair_mean_pp'] for p in per]),
                               pair_ann=L.jsumm([p['pair_ann'] for p in per]))
            rr[key]['by_j'] = per
        res[name] = rr
        print(name, {k: (round(v['ann_x']['median'], 2), round(v['t']['median'], 2), round(v.get('pair_t', {}).get('median') or 0, 2)) for k, v in rr.items()}, flush=True)
    # 挑選
    main_c = [k for k in VARIANTS if VARIANTS[k]['pool'] == 'ALL']
    sc = {k: res[k]['100萬|GROW']['pair_t']['median'] for k in main_c}
    order = sorted(sc, key=lambda k: -sc[k])
    nb = {'v1_ALL300_n10': ['v2_ALL400_n10'], 'v2_ALL400_n10': ['v1_ALL300_n10', 'v3_ALL500_n10'], 'v3_ALL500_n10': ['v2_ALL400_n10']}
    chosen = None
    for k in order:
        sgn = np.sign(res[k]['100萬|GROW']['pair_mean_pp']['median'])
        ok = all(np.sign(res[n]['100萬|GROW']['pair_mean_pp']['median']) == sgn for n in nb.get(k, []))
        if ok: chosen = k; break
    sat_c = [k for k in VARIANTS if VARIANTS[k]['pool'] == 'TP']
    sat = max(sat_c, key=lambda k: res[k]['30萬|GROW']['ann_x']['median'])
    doc = json.loads((HERE / 'iterations.json').read_text())
    doc['RESULTS'] = res; doc['selection'] = dict(main_scores_pair_t_median=sc, order=order, chosen_main=chosen, chosen_satellite=sat,
                                                  n_variants=len(VARIANTS), runtime_s=time.time() - t0)
    (HERE / 'iterations.json').write_text(json.dumps(doc, ensure_ascii=False, indent=1, default=str))
    print('chosen', chosen, sat, sc)
