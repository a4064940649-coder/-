# -*- coding: utf-8 -*-
"""凍結與樣本外之後的描述性補充（不改任何規格、不參與判定）：
(1) 落後 0050 的「深度」：相對淨值（組合／0050）最大回撤、最差滾動 36 個訊號窗相對報酬；(2) 損益兩平滑價 t=2.3；(3) 全額手續費；
(4) 多重檢定帳；(5) 各 w 的 SEL 選版在 2019+ 的表現（w=20/40）。寫進 results.json 的 POSTHOC。"""
import sys, json, math
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import numpy as np, pandas as pd
import rlib as L
res = json.loads((HERE / 'results.json').read_text(encoding='utf-8'))
assert 'PREREG' in res and 'OOS' in res
PRE = res['PREREG']; O = res['OOS']
PH = {}
def depth(nav):
    b = L.NAVS['E50'].reindex(nav.index); b = b / b.iloc[0]
    r = L.win(nav); rb = L.win(b)
    rel = (1 + r).cumprod() / (1 + rb).cumprod(); rel = pd.concat([pd.Series([1.0], index=[nav.index[0]]), rel])
    cr = (1 + r).cumprod(); cb = (1 + rb).cumprod()
    s36 = ((cr / cr.shift(36)) - (cb / cb.shift(36))).dropna()
    s12 = ((cr / cr.shift(12)) - (cb / cb.shift(12))).dropna()
    return dict(max_rel_dd_pct=float((rel / rel.cummax() - 1).min() * 100), worst_roll36_shortfall_pp=float(s36.min() * 100) if len(s36) else None,
                worst_roll12_shortfall_pp=float(s12.min() * 100) if len(s12) else None)
dp = {}
for ru in ('M', 'NONE'):
    for w in (0.0, 0.1, 0.2, 0.3, 0.4, 0.5):
        for P in ('S1', 'S2', 'SEL', 'VAL', 'HOLD', 'POST', 'ALL'):
            dp[f'{ru}|{int(w*100)}|{P}'] = depth(L.sim(w, ru if w > 0 else 'M', *L.PER[P])['nav'])
for P in ('S1', 'S2', 'SEL', 'VAL', 'HOLD', 'POST', 'ALL'):
    dp[f'2330M|30|{P}'] = depth(L.sim(0.3, 'M', *L.PER[P], B='A2330')['nav'])
PH['lag_depth'] = dp
# 損益兩平 t=2.3
cg = O['cost_grid']
def interp(key, field, target):
    cols = [c for c in cg if c.startswith('slip')]
    xs = [float(c[4:]) for c in cols]; ys = [cg[c][key][field] for c in cols]
    for i in range(len(xs) - 1):
        if (ys[i] - target) * (ys[i + 1] - target) <= 0 and ys[i] != ys[i + 1]:
            return xs[i] + (target - ys[i]) * (xs[i + 1] - xs[i]) / (ys[i + 1] - ys[i])
    return None
PH['breakeven_t2.3'] = {k: interp(k, 't', 2.3) for k in ('MAIN100|ALL', 'M70_30_M|ALL', 'M70_30_SEL|ALL')}
PH['fullfee'] = cg['fullfee_slip0.0010']
PH['cost_grid_ALL'] = {c: {k: (round(v['CAGR'], 2), round(v['t'], 2)) for k, v in cg[c].items() if k.endswith('|ALL')} for c in cg}
# 多重檢定帳
it = json.loads((HERE / 'iterations.json').read_text(encoding='utf-8'))
from math import erf, sqrt
def p2(t): return 2 * (1 - 0.5 * (1 + erf(abs(t) / sqrt(2))))
allt = []
for x in it['iterations']:
    if x['it'] in ('it1', 'it3') or x['it'].startswith('it4'):
        for k, v in x['cells'].items():
            if 'paired_vs_M' in v and v['paired_vs_M']['t'] == v['paired_vs_M']['t']: allt.append((x['it'] + ':' + k, v['paired_vs_M']['t']))
m = it['meta']['n_distinct_rule_variants_tested']
best = max(allt, key=lambda z: abs(z[1]))
sel_t = res['OOS']['judgment']['normal']['SEL']['t']
PH['multiple_testing'] = dict(m_rule_variants=m, n_iterations=it['meta']['n_iterations'], best_SEL_t=best, best_p=p2(best[1]), best_p_bonf=min(1, p2(best[1]) * m),
                              selected_SEL_t=sel_t, selected_p=p2(sel_t), selected_p_bonf=min(1, p2(sel_t) * m),
                              note='主策略本身已經過 6 輪以上改版（o3 5.3 節第 8 點）；本題的 m 只算配置規則')
# w=20／40 的 SEL 選版在 POST
PH['other_w_selected_POST'] = {w: O['grid'][f'normal|POST|{w}|{PRE["selected"][str(w)]}']['paired_vs_M'] for w in (20, 30, 40)}
PH['other_w_selected_POST_stress'] = {w: O['grid'][f'stress|POST|{w}|{PRE["selected"][str(w)]}']['paired_vs_M'] for w in (20, 30, 40)}
res['POSTHOC'] = PH
(HERE / 'results.json').write_text(json.dumps(res, ensure_ascii=False, indent=1, default=float), encoding='utf-8')
for k in ('S1', 'S2', 'VAL', 'HOLD', 'ALL'):
    print('DEPTH', k, {w: (round(dp[f'M|{w}|{k}']['max_rel_dd_pct'], 1), None if dp[f'M|{w}|{k}']['worst_roll36_shortfall_pp'] is None else round(dp[f'M|{w}|{k}']['worst_roll36_shortfall_pp'], 1), round(dp[f'M|{w}|{k}']['worst_roll12_shortfall_pp'], 1)) for w in (0, 10, 20, 30, 40, 50)},
          '2330', round(dp[f'2330M|30|{k}']['max_rel_dd_pct'], 1))
print(PH['breakeven_t2.3'], PH['fullfee'], PH['multiple_testing'])
print(PH['cost_grid_ALL'])
print({w: (round(v['ann_diff_pp'], 2), round(v['t'], 2)) for w, v in PH['other_w_selected_POST'].items()}, {w: (round(v['ann_diff_pp'], 2), round(v['t'], 2)) for w, v in PH['other_w_selected_POST_stress'].items()})
