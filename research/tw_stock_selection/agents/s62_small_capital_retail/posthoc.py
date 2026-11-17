#!/usr/bin/env python3
"""S62 事後補充（PREREG 之後才想到，不進判定，只描述）：python3 -I -B posthoc.py → results_posthoc.json
問題：主表從 2018 年底起跑後淨值每年長四成多，「5 萬」很快不是 5 萬，固定低消的負擔被稀釋。散戶關心的是「本金就停在這個水位」時每年被多吃多少。
做法：在 2018～2024 每年最後交易日各新起跑一次（本金 X 與 100 萬），只看起跑後第一年的報酬，X 減 100 萬＝該本金水位的一年摩擦差。
      7 個起跑年 × 位移 0～3＝28 個觀察值；情境 R0、A、B、C、D。另列第一年對 0050 的超額、主跑的買不到筆數明細。"""
import sys, json, time, resource
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
import s62lib as L
from s62lib import s10lib, stlib

H = Path(__file__).resolve().parent
t_all = time.time()
P, U, RT = L.setup()
lastb = lambda d: P.IDX[P.IDX < pd.Timestamp(d)][-1]
YS = list(range(2018, 2025))
DS = {y: lastb(f'{y+1}-01-01') for y in YS + [2025]}
row = lambda d: int(np.searchsorted(P.IDX.values, np.datetime64(d)))
bench = pd.Series(P.E50, index=P.IDX)
NM = ('R0_現行嚴成本', 'A_樂觀', 'B_基準', 'C_保守', 'D_只買整張')
ck = lambda c: f'{int(round(c / 1e4))}萬'
obs = {k: {ck(c): [] for c in L.CAPS + (L.REF_CAP,)} for k in NM}
b1 = {y: float(bench.loc[DS[y + 1]] / bench.loc[DS[y]] - 1) for y in YS}
for s in L.SHIFTS:
    sg = s10lib.make_signal_s10(P, U, RT, shift=s)
    for k in NM:
        for y in YS:
            for c in L.CAPS + (L.REF_CAP,):
                nav, lg, st = L.sim(P, sg, U, c, L.SCEN[k], row(DS[y]))
                obs[k][ck(c)].append(dict(y=y, s=s, r1=float(nav.loc[DS[y + 1]] / nav.iloc[0] - 1)))
out = dict(note='事後補充，不進判定', start_years=YS, bench_first_year=b1, cells={})
for k in NM:
    out['cells'][k] = {}
    ref = np.array([o['r1'] for o in obs[k]['100萬']])
    bb = np.array([b1[o['y']] for o in obs[k]['100萬']])
    for c in L.CAPS + (L.REF_CAP,):
        r = np.array([o['r1'] for o in obs[k][ck(c)]])
        d = r - ref
        out['cells'][k][ck(c)] = dict(first_year_mean=float(r.mean()), drag_vs_100萬_mean=float(d.mean()), drag_median=float(np.median(d)), drag_min=float(d.min()), drag_max=float(d.max()),
                                     drag_by_year={y: float(np.mean([dd for dd, o in zip(d, obs[k][ck(c)]) if o['y'] == y])) for y in YS},
                                     excess_vs_0050_mean=float((r - bb).mean()), excess_by_year={y: float(np.mean([rr - b1[y] for rr, o in zip(r, obs[k][ck(c)]) if o['y'] == y])) for y in YS},
                                     n=len(r), beat_0050=int((r > bb).sum()))
out['secs'] = round(time.time() - t_all); out['peak_mb'] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
json.dump(out, open(H / 'results_posthoc.json', 'w'), ensure_ascii=False, indent=1)
for k in NM:
    print(k)
    for c, v in out['cells'][k].items():
        print(f"  {c:>5} 第一年平均 {v['first_year_mean']*100:6.2f}% 對100萬 {v['drag_vs_100萬_mean']*100:6.2f}（中位 {v['drag_median']*100:6.2f}，{v['drag_min']*100:6.2f}～{v['drag_max']*100:6.2f}） 對0050 {v['excess_vs_0050_mean']*100:6.2f} 勝 {v['beat_0050']}/{v['n']} 逐年超額 "
              + ' '.join(f"{y%100}:{x*100:.0f}" for y, x in v['excess_by_year'].items()))
