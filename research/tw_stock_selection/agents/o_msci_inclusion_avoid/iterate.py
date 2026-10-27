#!/usr/bin/env python3
"""迭代（只用挑選期：訊號窗 2010-11～2018-12；模擬只跑到 2019 年第一個訊號日收盤，之後的資料完全不進入計算）。
選版規則（寫在跑之前，iterations.json 'selection_rule'）：
  1. 主指標：排除金融 n10、一般成本、挑選期受影響月的配對差（改版 − 原版，訊號窗）平均 > 0，且挑選期年化 ≥ 原版 − 0.3pp。
  2. 通過者中，取受影響月最多者；同數時取 n10 與 n30 配對差平均較大者；再同取較簡單者（A1, A2, B1, B2, C1, C2 順序）。
  3. 都不通過：取 n10＋n30 挑選期年化差平均最大者凍結，並標記「挑選期未通過選版規則」。
  4. 題目門檻（受影響事件月 ≥ 8）另行檢查並報告；不足時寫明檢定力不足。"""
import sys, os, json, datetime
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
import oa_lib as O
import vers as W
K, IDX = O.K, O.IDX
RULE = __doc__
A, B = '2010-11-01', '2018-12-31'
t1 = IDX.get_loc([d for d in W.SD if d >= pd.Timestamp('2019-01-01')][0])
assert IDX[t1] < pd.Timestamp('2019-01-20')
ew = W.ew_nav(10, t1)
res = {'selection_rule': RULE, 'versions': W.VERSIONS, 'pressure_median_select': W.P_MED, 'n_gate_events': len(W.GATE),
       'window': [A, B], 'sim_end': str(IDX[t1].date()), 'note_engine_fix': '第 1 次執行發現：同一 E 日多檔賣出時，遞補金額被合併買進 1 檔；修正為每筆賣出各自遞補一檔後重跑（只影響 n30 的 B1/C1/C2；版本定義與選版規則不變）', 'created': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'results': {}}
summary = []
for n in (10, 30):
    for cost in ('normal', 'stress'):
        nb, plb = W.base(n, cost, t1)
        sb = W.seg(nb, ew, A, B)
        res['results'][f'BASE|n{n}|{cost}'] = {'select': sb}
        for v in W.VERSIONS:
            r = W.run(v, n, cost, t1)
            aff = W.affected_periods(r, plb)
            sm = W.seg(r['nav'], ew, A, B)
            pr = W.paired(r['nav'], nb, W.SD, aff, A, B)
            o = {'select': sm, 'paired': pr, 'dCAGR_pp': round(sm['CAGR'] - sb['CAGR'], 3), 'dMaxDD_pp': round(sm['maxDD'] - sb['maxDD'], 3),
                 'skips': [x for x in r['skips'] if x[0] <= B], 'sells': [x for x in r['log'] if x[0] in ('sell_E', 'postpone', 'buy_repl', 'buy_0050')]}
            res['results'][f'{v}|n{n}|{cost}'] = o
            summary.append((v, n, cost, o['dCAGR_pp'], pr['n_affected_months'], pr['mean_affected%'], pr['t_affected'], o['dMaxDD_pp']))
            print(v, n, cost, 'dCAGR', o['dCAGR_pp'], 'aff', pr['n_affected_months'], pr['mean_affected%'], pr['t_affected'], 'dDD', o['dMaxDD_pp'], pr['affected_list'], flush=True)
# 選版
R = res['results']
def ok(v):
    x = R[f'{v}|n10|normal']; return (x['paired']['mean_affected%'] or -1) > 0 and x['dCAGR_pp'] >= -0.3
order = list(W.VERSIONS)
passing = [v for v in order if ok(v)]
def key(v):
    a = R[f'{v}|n10|normal']['paired']; b = R[f'{v}|n30|normal']['paired']
    return (a['n_affected_months'], np.nanmean([a['mean_affected%'] or 0, b['mean_affected%'] or 0]), -order.index(v))
if passing:
    chosen = max(passing, key=key); how = 'rule 1-2'
else:
    chosen = max(order, key=lambda v: (np.mean([R[f'{v}|n10|normal']['dCAGR_pp'], R[f'{v}|n30|normal']['dCAGR_pp']]), -order.index(v))); how = 'rule 3（無版本通過）'
res['passing'] = passing; res['chosen'] = chosen; res['chosen_how'] = how
res['threshold_affected_months_ge_8'] = {v: R[f'{v}|n10|normal']['paired']['n_affected_months'] >= 8 for v in order}
res['n_tests'] = '6 版 × n10/n30 × 一般／壓力成本 = 24 組挑選期結果'
json.dump(res, open(O.HERE / 'iterations.json', 'w'), ensure_ascii=False, indent=1, default=str)
print('passing', passing, 'chosen', chosen, how)
