#!/usr/bin/env python3
"""迭代（只用挑選期 2006-07～2013-12）：python3 -I -B iterate.py → iterations.json

版本空間（研究計畫事前固定，全部跑、全部記錄）：營收訊號 3 種 × 52 週百分位區間 4 種 × 持股 10／30 檔 = 24 版。
模擬的最後一天是 2013-12-31（程式斷言），沒有任何 2014 年以後的價格進入。
選擇規則（之後由 prereg.py 套用）：挑選期對同池等權 t（訊號窗、一般成本）最高，且和主策略 n10（排除金融）
月超額（對同池等權）的相關 < 0.6；沒有版本 t ≥ 1.5 → 判不可用、不打開後兩段。
"""
import sys, json, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
import olib as O
V, K = O.V, O.K
T0 = time.time()

R = O.revenue()
S = V.V2('gapfix', R['YL'], R['Y5'], exfin=True)
L = O.Leg(S)
d0 = O.leg_start(L)
days = [d for d in S.sig_days if d0 <= d <= O.SEL_END]
t1 = K.IDX.get_loc(K.IDX[K.IDX <= O.SEL_END][-1])
assert K.IDX[t1] <= O.SEL_END
sd = days + [K.IDX[t1]]
COST = {'normal': K.rates(0.001), 'stress': K.rates('stress')}

# 主策略 n10（排除金融、gapfix、正式次排序）同期
plm = [p for p in S.plan(10, 'official') if p[3] in set(days)]
t0 = plm[0][0]
ew, _ = K.simulate([(p[0], p[2]) for p in plm], 0, 0, t0, t1)
navm = {c: K.simulate([(p[0], p[1]) for p in plm], b, s, t0, t1)[0] for c, (b, s) in COST.items()}
wr_ew = O.window_returns(ew, sd); wr_50 = O.window_returns(K.E50.reindex(ew.index), sd)
wr_m = O.window_returns(navm['normal'], sd)
exm = (wr_m - wr_ew).to_numpy(); exm50 = (wr_m - wr_50).to_numpy()
y13 = np.array([d.year != 2013 for d in wr_m.index])
main_stats = O.seg(navm['normal'], ew, O.SEL[0], O.SEL_END, sd, yearly=True)
main_stats['t_EW_ex2013'] = O.tstat(exm[y13])

it = {'space': {'rev': O.REVS, 'bands': O.BANDS, 'n': O.NS}, 'n_versions': len(O.ALL_VERSIONS), 'period': [str(days[0].date()), str(K.IDX[t1].date())],
      'n_windows': len(days), 'main_n10_exfin_same_period': main_stats, 'versions': {}}
X = []
for rev in O.REVS:
    for band in O.BANDS:
        for n in O.NS:
            v = O.vname(rev, band, n)
            pl = L.plan(rev, band, n, days)
            assert all(p[0] <= t1 for p in pl) and [p[0] for p in pl] == [p[0] for p in plm]
            r = {}
            for c, (b, s) in COST.items():
                nav, lg = K.simulate([(p[0], p[1]) for p in pl], b, s, t0, t1)
                st = O.seg(nav, ew, O.SEL[0], O.SEL_END, sd, yearly=(c == 'normal'))
                wr = O.window_returns(nav, sd)
                ex = (wr - wr_ew).to_numpy(); ex50 = (wr - wr_50).to_numpy()
                st['corr_main_exEW'] = O.corr(ex, exm); st['corr_main_ex0050'] = O.corr(ex50, exm50)
                st['t_EW_ex2013'] = O.tstat(ex[y13]); st['mean_ex_EW_pct_m'] = float(np.mean(ex) * 100)
                st['turnover'] = lg['turnover']
                r[c] = st
                if c == 'normal': X.append(ex)
            r['mean_holdings'] = float(np.mean([len(p[1]) for p in pl]))
            r['overlap_with_main_n10'] = float(np.mean([len(set(p[1]) & set(q[1])) / max(len(p[1]), 1) for p, q in zip(pl, plm)]))
            r['blocked_gapup'] = L.nblk
            it['versions'][v] = O.r4(r)
            print(f'{v:14s} CAGR {r["normal"]["CAGR"]:6.1f} tEW {r["normal"]["t_EW"]:5.2f} t50 {r["normal"]["t_0050"]:5.2f} corr {r["normal"]["corr_main_exEW"]:.2f} '
                  f'ex13 {r["normal"]["t_EW_ex2013"]:.2f} stress tEW {r["stress"]["t_EW"]:.2f} ovl {r["overlap_with_main_n10"]:.2f}', flush=True)
X = np.column_stack(X)
it['maxt_bootstrap_selection'] = O.maxt_bootstrap(X, B=5000, mean_block=6, seed=0)
it['maxt_bootstrap_selection']['versions_order'] = O.ALL_VERSIONS
it['runtime_s'] = round(time.time() - T0, 1)
(O.HERE / 'iterations.json').write_text(json.dumps(O.r4(it), ensure_ascii=False, indent=1), encoding='utf-8')
print('main', {k: main_stats[k] for k in ('CAGR', 't_EW', 't_0050', 'maxDD', 't_EW_ex2013')})
print(it['maxt_bootstrap_selection'])
