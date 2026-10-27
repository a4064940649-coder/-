#!/usr/bin/env python3
"""彙整：全期（2021-10～2026-10-12）連續淨值、B1 整張／零股可行性、判定。寫入 results.json（不動 PREREG）。"""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
import sbl_lib as L

H = L.HERE
R = json.loads((H / 'results.json').read_text(encoding='utf-8'))
assert 'PREREG' in R
K, IDX, COLS = L.K, L.IDX, L.COLS
S = L.strategy(True)
a, b = L.SEL[0], L.VAL[1]
t_end = IDX.get_loc(S.end_day)
base = L.main_plan(S, a, b); sl = L.sleeve_plan(S, 'lvl_size', a, b); c2 = L.main_plan(S, a, b, third='lvl')
SD = [p[3] for p in base] + [S.end_day]
out = {}
navs = {}
for c in L.COST:
    nm = L.run_plan(base, c, t_end)[0]; ns = L.run_plan(sl, c, t_end)[0]; n2 = L.run_plan(c2, c, t_end)[0]
    navs[c] = {'main': nm, 'B1': ns, 'C2': n2, 'C1': L.combine(nm, ns, base, 0.8)}
ew = L.ew_nav(base, t_end)
full = {}
for c in L.COST:
    full[c] = {}
    for k, nv in navs[c].items():
        o = L.cagr_dd(nv, IDX[base[0][0]], S.end_day)
        o['t_vs_0050'] = L.summ(L.monthly_t(nv, K.E50, SD, a, b))
        o['t_vs_EW'] = L.summ(L.monthly_t(nv, ew, SD, a, b))
        if k != 'main': o['paired_vs_main'] = L.summ(L.monthly_t(nv, navs[c]['main'], SD, a, b))
        full[c][k] = o
full['EW'] = L.cagr_dd(ew, IDX[base[0][0]], S.end_day); full['0050'] = L.cagr_dd(K.E50, IDX[base[0][0]], S.end_day)
out['full_2021-10_2026-10'] = full

# B1 可行性：每期每檔分配金額 vs 一張（1000 股 × 成交日開盤）
feas = {}
for cap in (3e5, 1e6):
    need_odd = [];
    for p in sl:
        alloc = cap / max(len(p[1]), 1)
        lot = 1000 * K.On[p[0], p[1]]
        need_odd.append(float(np.mean(lot > alloc)))
    feas[f'{int(cap/1e4)}萬'] = {'share_of_picks_needing_odd_lot_mean': round(float(np.mean(need_odd)), 3),
                                'per_stock_alloc': round(cap / 30)}
out['B1_lot_feasibility'] = feas
out['B1_avg_names_changed_per_month'] = round(float(np.mean([len(set(p[1]) - set(q[1])) for p, q in zip(sl[1:], sl[:-1])])), 2)

oos = json.loads((H / 'results_oos.json').read_text(encoding='utf-8'))
v = oos['res']['VAL_ALL_2024_2026']['normal']
it = json.loads((H / 'iterations.json').read_text(encoding='utf-8'))
b1v = v['B1_sleeve']['vs_EW']; corr = oos['res']['VAL_ALL_2024_2026']['corr_B1_excess_vs_main_excess(vs_EW)']
ws = oos['B1_within_size_vs_group_EW']
g3 = np.mean([ws[g]['VAL_ALL_2024_2026']['mean%'] for g in ('small', 'mid', 'large')])
crit1 = {'selection_t>=2': it['versions']['B1']['normal']['vs_EW']['t'] >= 2, 'val_mean>0': b1v['mean%'] > 0, 'val_t>=1.0': b1v['t'] >= 1.0,
         'corr<0.5': corr < 0.5, 'within_size_avg>0': bool(g3 > 0)}
sel = it['versions']['C2']['normal']; selm = it['main_n10']['normal']
c2v, mv = v['C2_third_score'], v['main']
crit2 = {'paired_both_periods_t>=1': bool(sel['vs_main']['t'] >= 1 and c2v['vs_main']['t'] >= 1),
         'val_DD_TE_alt': bool(c2v['CAGR%'] >= mv['CAGR%'] - 1 and c2v['maxDD%'] > mv['maxDD%'] and c2v['TE_vs_0050_ann%'] < mv['TE_vs_0050_ann%'])}
st = json.loads((H / 'results_oos_selftest.json').read_text(encoding='utf-8'))['res']['SEL']['normal']
crit2['sel_DD_TE_alt'] = bool(st['C2_third_score']['CAGR%'] >= st['main']['CAGR%'] - 1 and st['C2_third_score']['maxDD%'] > st['main']['maxDD%']
                              and st['C2_third_score']['TE_vs_0050_ann%'] < st['main']['TE_vs_0050_ann%'])
out['criteria_H3b'] = {'(1)_B1': crit1, 'B1_within_size_three_group_avg_val%': round(float(g3), 3), '(2)_C2': crit2}
out['verdict'] = {
 'H3a': '無法測試（借券賣出餘額只有 2026-07-30 起；已登記 2026-11 起前瞻追蹤，2028-11 評估）',
 'H3b_B1_second_leg': '不可用（驗證期 t 0.98 < 1.0；市值組內效果從小型股移到大型股；去掉前 5 大貢獻股後消失）',
 'H3b_C2_third_score': '待觀察（兩期都符合「年化不低於主策略、回撤較淺、追蹤誤差較低」的替代條件，但配對差 t 只有 0.54／0.15，保留段回撤反而較深；資料只有 2021-10 起，上限待觀察）',
 'overall': '待觀察'}
R['results'] = out
R['oos_file'] = 'results_oos.json'; R['h3a_pilot_file'] = 'h3a_pilot.json'; R['data_audit_file'] = 'results_data_audit.json'
(H / 'results.json').write_text(json.dumps(R, ensure_ascii=False, indent=1, default=str), encoding='utf-8')
print(json.dumps(out, ensure_ascii=False, indent=1, default=str))
