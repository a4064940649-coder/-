# -*- coding: utf-8 -*-
"""反方審查（成本與可成交）：只讀研究員檔案，不寫入任何研究員檔案。"""
import sys, json, math
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import numpy as np, pandas as pd
import rlib as L
N = L.NAVS
# 1. 0050 分割（2025-06）處理
r50 = N['E50'].pct_change()
print('E50 min/max daily ret', r50.min(), r50.idxmin(), r50.max(), r50.idxmax())
print('A2330 min/max', N['A2330'].pct_change().min(), N['A2330'].pct_change().max())
# 2. 主策略腿：資金可行性（30萬/100萬/1000萬，w=0 與 w=0.3）與 1000 萬參與率
MP = L.MP
rows = []
for p in MP['plan']:
    for pr, v in zip(p['open_raw'], p['v60']):
        rows.append(dict(y=int(p['exec'][:4]), d=p['exec'], px=pr, v60=v))
df = pd.DataFrame(rows)
print('v60 quantiles', df.v60.quantile([0, .05, .1, .5]).to_dict())
for cap in (300_000, 1_000_000, 10_000_000):
    for w in (0.0, 0.3):
        alloc = cap * (1 - w) / 10
        lot = df.px * 1000
        below = alloc < lot
        whole = np.floor(alloc / lot) * lot / alloc
        part = alloc / df.v60
        g = {'all': df.index, '<=2018': df.index[df.y <= 2018], '2019-23': df.index[(df.y >= 2019) & (df.y <= 2023)], '2024+': df.index[df.y >= 2024], 'pre2020-10-26': df.index[df.d < '2020-10-26']}
        print(cap, w, {k: (round(float(below[i].mean()), 3), round(float(whole[i].mean()), 3)) for k, i in g.items()},
              'part p50/p95/max %', [round(float(part.quantile(q)) * 100, 3) for q in (.5, .95, 1.0)], 'part>1%', round(float((part > 0.01).mean()), 4))
# 3. 小資金：每月再平衡 0050 移轉含最低手續費 20 元
r = L.sim(0.3, 'M', *L.PER['ALL'])
wb = r['wb']; dev = (wb - 0.3).abs()
for cap in (300_000, 1_000_000, 10_000_000):
    amt = dev * cap
    fee_model = amt * 0.001425 * 0.6
    fee_real = np.maximum(fee_model, 20.0)
    extra = (fee_real - fee_model).sum() / cap / ((L.IX[-1] - L.IX[0]).days / 365.25) * 100
    print('cap', cap, 'median transfer', round(float(amt.median())), 'extra minfee drag pp/yr (0050 side only)', round(float(extra), 4))
# 4. 主策略腿的最低手續費（每月 62% 周轉 -> 約 12.4 筆/月）
for cap in (300_000, 1_000_000):
    for w in (0.0, 0.3):
        slot = cap * (1 - w) / 10
        f = slot * 0.001425 * 0.6
        print('main leg slot', cap, w, round(slot), 'fee/order model', round(f, 1), 'floor binds', f < 20,
              'extra drag pp/yr', round(max(0, 20 - f) * 12.4 * 12 / cap * 100, 3))
# 5. 壓力成本下 2024～2026 與 POST：70/30 每月對 0050
res = json.loads((HERE / 'results.json').read_text(encoding='utf-8'))
T = res['OOS']['trio']
for c in ('normal', 'stress'):
    for P in ('HOLD', 'POST', 'ALL'):
        x = T[c][P]
        print(c, P, 'M70_30', round(x['MAIN70_0050_30_M']['CAGR'], 2), round(x['MAIN70_0050_30_M']['t_vs0050'], 2), 'E50', round(x['E50']['CAGR'], 2), 'A2330 alone', round(x['A2330']['CAGR'], 2))
# 6. 再平衡成本：移轉成本歸零 vs 一般（w=30 每月，ALL）
L.C['zero'] = dict(stock_buy=0.0, stock_sell=0.0, etf_buy=0.0, etf_sell=0.0)
orig_leg = L.leg_returns
def leg2(cost='normal', B='E50'): return orig_leg('normal', B)
L.leg_returns = leg2
z = L.sim(0.3, 'M', *L.PER['ALL'], cost='zero')['nav']
L.leg_returns = orig_leg
print('rebal cost drag pp/yr (ALL, w30 M)', round((L.cagr(z) - L.cagr(r['nav'])) * 100, 4))
# 7. 小資金頻率比較：在每筆 0050 移轉加 20 元固定成本時，每月 vs 每季（ALL/POST），30 萬
def sim_fixed(w, rule, a, b, cap):
    out = L.sim(w, rule, a, b)
    n = out['n_trades']; yrs = (out['nav'].index[-1] - out['nav'].index[0]).days / 365.25
    return L.cagr(out['nav']) * 100 - n * 20 / cap / yrs * 100 * 1.0
for P in ('ALL', 'POST'):
    a, b = L.PER[P]
    print('30萬', P, {ru: round(sim_fixed(0.3, ru, a, b, 300_000) - sim_fixed(0.3, 'M', a, b, 300_000), 3) for ru in ('Q', 'H', 'A', 'B5', 'B10')})
# 8. 執行點：exec 日收盤記帳對主選版判定（壓力）
for cost in ('normal', 'stress'):
    a, b = L.PER['SEL']
    n1 = L.sim(0.3, 'NONE', a, b, cost=cost)['nav']; n0 = L.sim(0.3, 'M', a, b, cost=cost, exec_mode='exec')['nav']
    print('exec-mode', cost, 'SEL diff', round((L.cagr(n1) - L.cagr(n0)) * 100, 3), 'drop2015', round(L.ann_diff_drop_year(n1, n0, 2015), 3))
