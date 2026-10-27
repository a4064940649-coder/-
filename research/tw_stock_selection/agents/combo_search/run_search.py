#!/usr/bin/env python3
"""步驟 1：129 組合 × n30/n10 × 開關有無，滑價 0.1%，2006-04～2026-10 全部模擬；只存檔（月窗報酬、日頻淨值），不輸出任何統計。
統計分析在 analyze_search.py（挑選／驗證）與 holdout.py（保留期，最後才跑）。"""
import sys, time
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent)); sys.dont_write_bytecode = True
import clib as K
L = K.L
OUT = K.HERE

def main():
    t = time.time()
    S = L.Setup(L.yoy_long(), start='2005-01-01')
    comps = K.components(S, K.rev_level())
    el = S.t300 & S.s52.notna() & ~S.inD & S.yoy3.notna()
    prk = K.pct_ranks(comps, el)
    t0 = L.pos(S.dates[0]); t1 = L.pos(S.dates[-1])
    b, s = L.cost_rates(0.001)
    navs, mon = {}, {}
    ev, pools = K.select(S, K.score(prk, K.MAIN), el, 30)
    ew = K.sim(pools, 0, 0, t0, t1); navs['EW'] = ew; mon['EW'] = K.monthly(ew, S.DATES)
    e50 = L.E50.iloc[t0:t1 + 1]; navs['0050'] = e50; mon['0050'] = K.monthly(e50, S.DATES)
    for c in K.COMBOS:
        sc = K.score(prk, c)
        for n in (30, 10):
            for sw in (False, True):
                ev, _ = K.select(S, sc, el, n, switch=sw)
                nav = K.sim(ev, b, s, t0, t1)
                key = f'{K.cname(c)}|n{n}|{"sw" if sw else "raw"}'
                navs[key] = nav; mon[key] = K.monthly(nav, S.DATES)
        print(K.cname(c), round(time.time() - t, 1), flush=True)
    pd.DataFrame(navs).astype('float32').to_parquet(OUT / 'search_nav.parquet')
    pd.DataFrame(mon).to_parquet(OUT / 'search_monthly.parquet')
    # 元件覆蓋率（股池內有值比例）
    cov = {k: round(float(v.where(el).notna().sum(axis=1).div(el.sum(axis=1)).mean()), 4) for k, v in comps.items()}
    pd.Series(cov).to_json(OUT / '_coverage.json')
    print('done', round(time.time() - t, 1))

if __name__ == '__main__':
    main()
