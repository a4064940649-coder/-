#!/usr/bin/env python3
"""事後診斷（不參與判定）：n10／n30 被出場的股票，從賣出開盤到下一個換股日開盤的報酬（＝持有不動會得到的），以及同窗口原版組合報酬。結果寫到 results.json DIAG_AFTER_EXIT。"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.dont_write_bytecode = True
import xlib as X
L = X.L
res = json.loads((HERE / 'results.json').read_text(encoding='utf-8'))
assert 'HOLDOUT' in res
b, s = L.cost_rates(0.001)
SEG = {'SEL': ('2006-04-01', '2013-12-31'), 'VAL': ('2014-01-01', '2020-12-31'), 'HOLD': ('2021-03-01', '2026-10-31')}
out = {}
for n in (10, 30):
    items, _ = X.main_items(n); T0 = items[0]['t']; T1 = len(X.IDX) - 1
    rebs = np.array(sorted(it['t'] for it in items if it['t'] >= T0))
    base, _ = X.simulate(items, 1, b, s, T0, T1)
    for rn in ['TP20', 'TP30', 'TP50', 'TP100', 'TR10', 'TR15', 'TR20', 'TR25', 'TIME10', 'MKT', 'SL15']:
        rule = X.RULES.get(rn, dict(sl=0.15))
        lg = []; X.simulate(items, 1, b, s, T0, T1, rule=rule, mode='cash', log=lg)
        rows = []
        for t, _, j, e in lg:
            k = np.searchsorted(rebs, t, side='right')
            if k >= len(rebs): continue
            tn = rebs[k]
            fwd = X.AO2[tn, j] / X.AO2[t, j] - 1
            pb = base.loc[X.IDX[tn - 1]] / base.loc[X.IDX[t - 1]] - 1   # 原版組合同窗口（近似：前一日收盤到換股前一日收盤）
            rows.append((X.IDX[t], fwd, pb, X.AO2[t, j] / e - 1, tn - t))
        df = pd.DataFrame(rows, columns=['date', 'fwd_to_next_reb', 'base_port_same_window', 'ret_at_exit', 'days_to_reb'])
        o = {}
        for sk, (a, bb) in SEG.items():
            d = df[(df.date >= a) & (df.date <= bb)]
            if len(d) == 0: o[sk] = None; continue
            o[sk] = {'n': int(len(d)), 'mean_fwd_pct': round(float(d.fwd_to_next_reb.mean() * 100), 2), 'median_fwd_pct': round(float(d.fwd_to_next_reb.median() * 100), 2),
                     'share_fwd_pos': round(float((d.fwd_to_next_reb > 0).mean()), 3), 'base_port_same_window_pct': round(float(d.base_port_same_window.mean() * 100), 2),
                     'mean_ret_at_exit_pct': round(float(d.ret_at_exit.mean() * 100), 1), 'mean_days_to_reb': round(float(d.days_to_reb.mean()), 1)}
        out[f'n{n}|{rn}|cash'] = o
        print(n, rn, {k: (v['n'], v['mean_fwd_pct'], v['median_fwd_pct'], v['share_fwd_pos'], v['base_port_same_window_pct'], v['mean_ret_at_exit_pct']) if v else None for k, v in o.items()}, flush=True)
res['DIAG_AFTER_EXIT'] = {'說明': '事後診斷，不參與判定。fwd＝被出場股票從賣出開盤到下一換股日開盤的報酬（若不出場會拿到的部分，未扣成本）；base_port＝同窗口原版組合報酬；ret_at_exit＝賣出開盤價相對進場價。', 'rows': out}
(HERE / 'results.json').write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str), encoding='utf-8')
