#!/usr/bin/env python3
"""彙整：讀 results.json 的 RESULTS_*，產生 SUMMARY（判定、關鍵數字、預測逐條對照）並印出 summary.txt。"""
import sys, json
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import fam_common as C
RES = C.check_prereg()

def g(d, *ks, default=None):
    for k in ks:
        if not isinstance(d, dict) or k not in d: return default
        d = d[k]
    return d

def main():
    SPA = RES['RESULTS_SPA']; BR = RES['RESULTS_BREAKS']; HO = RES['RESULTS_HOLD']; RO = RES.get('RESULTS_ROBUST', {}); CC = RES.get('CROSSCHECK', {})
    lines = []
    P = lambda *a: lines.append(' '.join(str(x) for x in a))
    P('== 族群規模 =='); P(json.dumps(SPA['family_sizes'], ensure_ascii=False))
    P('== 有效試驗數 ==')
    for k, v in SPA['eff_trials'].items(): P(k, 'N', v['N'], 'LiJi', round(v['M_eff_LiJi'], 1), 'Nyholt', round(v['M_eff_Nyholt'], 1))
    P('== SPA 格點（對 0050）==')
    for k, v in SPA['grid'].items():
        if not k.endswith('x0050'): continue
        P(k, 'rank', v['focus_rank'], 't', round(v['focus_t'], 2), 'max_t', round(v['max_t'], 2), v['argmax_member'][:60],
          '| RC', v['p_RC'], 'RCf', round(v['p_RC_focus'], 4), 'SPA l/c/u', v['p_SPA_l'], v['p_SPA_c'], v['p_SPA_u'],
          '| focus l/c/u', v['p_SPA_l_focus'], v['p_SPA_c_focus'], v['p_SPA_u_focus'], '| StepM', round(v['p_StepM_focus'], 4), 'single', v['p_single_focus'])
    P('== SPA 格點（對同池等權）==')
    for k, v in SPA['grid'].items():
        if not k.endswith('xEW'): continue
        P(k, 'rank', v['focus_rank'], 't', round(v['focus_t'], 2), 'max_t', round(v['max_t'], 2), 'SPA_c', v['p_SPA_c'], 'focus c', v['p_SPA_c_focus'], 'StepM', round(v['p_StepM_focus'], 4))
    P('== DSR =='); [P(k, {a: round(b['p_DSR'], 4) for a, b in v.items()}) for k, v in SPA['dsr'].items()]
    P('== 合成成員 =='); [P(k, 'n_syn', v['n_syn'], 'rank', v['focus_rank'], 'focus c', v['p_SPA_c_focus'], 'SPA_c', v['p_SPA_c'], 'StepM', round(v['p_StepM_focus'], 4)) for k, v in SPA['synthetic'].items()]
    P('== 子期間 =='); [P(k, 'n', v['n_windows'], 'rank', v['focus_rank'], 't', round(v['focus_t'], 2), 'focus c', v['p_SPA_c_focus'], 'SPA_c', v['p_SPA_c']) for k, v in SPA['subperiod'].items()]
    P('== 判定 =='); P(json.dumps(SPA['decision_value'], ensure_ascii=False), SPA['decision_grade'])
    P('== 斷點 ==')
    for nm, o in BR['series'].items():
        bp = o['BP_HAC']
        P(nm, 'mean', round(o['mean_pp'], 3), 't', round(o['t'], 2), '| supF', round(bp['supF1'], 2), 'p', bp['p_supF1'], bp['supF1_date'], '| UDmax p', bp['p_UDmax'],
          '| m BIC/LWZ/seq', bp['m_BIC'], bp['m_LWZ'], bp['m_seq5'], bp['sel_BIC']['break_dates'], bp['sel_BIC']['ci90_dates'],
          '| CUSUM p', round(o['OLS_CUSUM_HAC']['p'], 3), 'rec', o['REC_CUSUM']['crosses_5pct'],
          '| Chow19', round(o['Chow_HAC']['2019-05']['diff'] * 100, 3), round(o['Chow_HAC']['2019-05']['p'], 3), 'MDE', round(o['Chow_HAC']['2019-05']['MDE80_two_sided'] * 100, 2),
          '| Chow24', round(o['Chow_HAC']['2024-01']['diff'] * 100, 3), round(o['Chow_HAC']['2024-01']['p'], 3), 'MDE', round(o['Chow_HAC']['2024-01']['MDE80_two_sided'] * 100, 2),
          '| iid supF p', o['BP_iid']['p_supF1'], 'm_seq', o['BP_iid']['m_seq5'])
    P(json.dumps(BR['power_sim'], ensure_ascii=False)); P(json.dumps(BR['decision'], ensure_ascii=False))
    P('== 保留期 ==')
    for nm, o in HO['predictive'].items():
        P(nm, json.dumps(o['actual']), 'pre mean', round(o['pre_mean_arith_pp'], 3), 'L6', json.dumps(o['L6']), 'L3 p', o['L3']['p_lower_cumlog'], 'L12 p', o['L12']['p_lower_cumlog'], 'MDE', json.dumps(o['MDE80_one_sided_pp']), 'MDEboot', round(o['MDE80_boot_pp'], 2))
    P(json.dumps(HO['decomp'], ensure_ascii=False)); P(json.dumps(HO['concentration_regression'], ensure_ascii=False))
    if RO: P('== 穩健性 =='); [P(k, v) for k, v in RO['variants'].items()]; P('yearly', RO['yearly_excess_pp']); P('cal t', RO['calendar_month_t'])
    if CC: P('== 交叉核對 =='); [P(k, v) for k, v in CC.get('spa', {}).items()]; P(CC.get('ols_cusum'))
    (HERE / 'summary.txt').write_text('\n'.join(lines), encoding='utf-8')
    print('\n'.join(lines))

if __name__ == '__main__':
    main()
