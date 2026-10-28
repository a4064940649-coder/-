#!/usr/bin/env python3
"""主版本 F1_K5_N20_sqrt：建橫截面 → 權重 → 引擎 → 統計。"""
import sys, json, pickle
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
import rlib as L, rstrat as S

DATES = L.DATES
xs_path = L.OUT / 'xs.pkl'
if xs_path.exists():
    XS = pickle.load(open(xs_path, 'rb'))
else:
    XS = {}
    for d in DATES[:-1]:
        XS[d] = S.XS(d)
    pickle.dump(XS, open(xs_path, 'wb'))

def run_f1(K=5, N=20, wmode='sqrt'):
    W = {}; held = set(); info = []
    for d in DATES[:-1]:
        x = XS[d]
        w, pick, nskip = S.f1_weights(x, held, K=K, N=N, wmode=wmode)
        W[d] = w; held = set(np.where(w > 0)[0]); info.append((d, len(pick), nskip, int(x.el.sum())))
    return W, info

if __name__ == '__main__':
    W, info = run_f1()
    REPW = {d: XS[d].R for d in DATES[:-1]}
    res = {}
    navs = {}
    for kind in ('normal', 'stress'):
        nav, turn = S.simulate(W, kind); navs['F|' + kind] = nav
        if kind == 'normal': res['turnover_1side'] = float(np.mean([t[1] for t in turn])) * 100; res['buy_turn'] = float(np.mean([t[2] for t in turn])) * 100
    nav_rep, _ = S.simulate(REPW, 'normal'); navs['REP|normal'] = nav_rep
    nav_panel, _ = S.simulate(W, 'normal', gapfix=False); navs['F|panelret'] = nav_panel
    pd.DataFrame(navs).to_parquet(L.OUT / 'navs_main.parquet')
    for k in navs:
        res[k] = S.stats(navs[k], DATES)[0]
    res['F_vs_REP'] = S.stats(navs['F|normal'], DATES, other=nav_rep)[0]
    res['REP_vs_0050'] = res['REP|normal']
    res['info_head'] = [(str(a.date()), b, c, e) for a, b, c, e in info[:3]] + [(str(a.date()), b, c, e) for a, b, c, e in info[-3:]]
    res['n_el_min'] = int(min(i[3] for i in info))
    res['nskip_total'] = int(sum(i[2] for i in info))
    json.dump(res, open(L.OUT / 'r1_main.json', 'w'), ensure_ascii=False, indent=1, default=float)
    for k in ('F|normal', 'F|stress', 'REP|normal', 'F_vs_REP', 'F|panelret'):
        print('==', k)
        for p, v in res[k].items():
            print(f"  {p:9s} n={v['n']:3d} ex={v['ex_pp']:+.2f}pp t={v['t']:+.2f} IR={v['IR']:+.2f} TE={v['TE']:.1f}% ann={v['ann']:.1f}/{v['ann_b']:.1f} (cal {v['ann_cal']:.1f}/{v['ann_b_cal']:.1f}) mdd={v['mdd']:.1f}/{v['mdd_b']:.1f}")
    print('turnover', res['turnover_1side'], res['buy_turn'], 'nskip', res['nskip_total'], 'nel_min', res['n_el_min'])
