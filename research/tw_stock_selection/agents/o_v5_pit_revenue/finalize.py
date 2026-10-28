#!/usr/bin/env python3
"""最終判定（依 PREREG 寫定的規則，不改規格）：級別、資料等級、預測核對、與主策略比較、SOP 實盤預期修正。寫進 results.json 的 FINAL。"""
import json, hashlib
from pathlib import Path
HERE = Path(__file__).resolve().parent
import numpy as np
res = json.loads((HERE / 'results.json').read_text(encoding='utf-8'))
assert hashlib.sha256(json.dumps(res['PREREG'], ensure_ascii=False, sort_keys=True).encode()).hexdigest() == res['PREREG_sha256']
R, J = res['RESULTS'], res['RESULTS_JSCAN']
rep = json.loads((HERE / 'rep' / 'rep_v5.json').read_text()) if (HERE / 'rep' / 'rep_v5.json').exists() else None
it = json.loads((HERE / 'iterations.json').read_text(encoding='utf-8'))
E = json.loads((HERE / 'explore_fv.json').read_text(encoding='utf-8'))
T = R['table']; ab = T['AB_V5|normal']; ep = ab['endpoint']
t = ab['ALL']['t_0050']
three = {'drop_best1': ep['drop_best1'], 'start_2008': ep['start_2008-01']['t'], 'stress': T['AB_V5|stress']['ALL']['t_0050']}
nlow = sum(v < 2.0 for v in three.values())
sd = R['seed_dist']['AB_V5|normal']
if t >= 2.58: grade = '第 1 級'
elif t >= 2.30:
    grade = '第 2 級' if (nlow >= 2 or sd['n_seed_t_below_2.30'] > 20) else '第 1 級邊緣'
else: grade = '第 2 級'
cons_key = 'AB_V5_B10|normal'          # 缺口閘門未過 → AB＋B10（V5 資料）
cons = T[cons_key]['ALL']
e50 = ab['ALL']['E50_CAGR']
def live(c): return e50 + (c - e50) * 0.5
use = cons if grade == '第 2 級' else ab['ALL']
F = {'grade': grade, 'AB_V5_t': t, 'three_checks': three, 'n_three_below_2.0': nlow, 'n_seed_below_2.30': sd['n_seed_t_below_2.30'],
     'conservative_column': cons_key, 'conservative': {'CAGR': cons['CAGR'], 't': cons['t_0050']},
     'SOP_live_expectation_rule': '0050 年化 ＋（策略年化 − 0050）× 0.5；第 2 級用保守口徑（AB_V5_B10）',
     'SOP_live_expectation_pct': live(use['CAGR']), 'SOP_live_expectation_prev_V4_pct': 21.0,
     'SOP_live_expectation_change_pp': live(use['CAGR']) - 21.0,
     'data_grade': {'first_value_panel': 'A−（只有單一新聞來源、前 30 名覆蓋 0.35%、首次值閘門未過；其餘以界限處理）',
                    'gap_proxy': '不適用（閘門未過：2006～2012 缺口格 0／490 有 PIT 合法代理；以 AB_V5_B10 結案）'},
     'vs_main': {'AB_V5_minus_AB_V4': R['paired']['AB_V5_minus_AB_V4']['ALL'], 'AB_V5_minus_main_v2': R['paired']['AB_V5_minus_main_v2']},
     'replication': rep and {k: rep[k] for k in ('V5|normal_maxreldiff', 'AB_V5|normal_maxreldiff', 'seed_nav_maxreldiff', 'seed_t_maxabsdiff', 'jscan_t_maxabsdiff', 'pass_1e-9')}}
# 預測核對
pc = R['paired']['AB_V5_minus_AB_V4']['ALL']
dcagr = ab['ALL']['CAGR'] - T['AB_V4REF|normal']['ALL']['CAGR']
P = {}
P['P1a_disagree_rate_0.5_1.0pct'] = {'confirmed_F3': 36 / E['n_news_cells_prec_ok'], 'raw': E['all']['disagree_rate'],
                                     'holds': bool(0.005 <= 36 / E['n_news_cells_prec_ok'] <= 0.010)}
P['P1b_top30_ratio_ge_1.5（<1.2 即否定機制）'] = {'raw_ratio': E['ratio_top30_vs_all'],
    'confirmed_ratio': (E['top30_cur']['rate_disagree_consensus_or_internal'] / E['all']['rate_disagree_consensus_or_internal']),
    'holds': bool(E['ratio_top30_vs_all'] >= 1.5), 'mechanism_falsified': bool(E['ratio_top30_vs_all'] < 1.2)}
P['P1c_top10_changed_stockmonths_le_15'] = {'nonAB': len(R['fv_pick_changes']['nonAB_months_changed']), 'AB_per_seed_mean': R['fv_pick_changes']['AB_mean_months_changed_per_seed'],
                                            'holds': bool(R['fv_pick_changes']['AB_mean_months_changed_per_seed'] <= 15)}
P['P1d_paired_-0.05_0pp_and_abs_dCAGR_le_0.3'] = {'paired_mean_pp': pc['mean_pp'], 't': pc['t'], 'dCAGR_pp': dcagr,
                                                  'holds': bool(-0.05 <= pc['mean_pp'] <= 0 and abs(dcagr) <= 0.3)}
P['P2_gap_proxy'] = {'holds': None, 'note': '閘門未過，缺口代理未建，無法檢定；B10 入選缺口股見 RESULTS.B10_gap_picks 與 mechanism'}
P['P3_AB_V5_25.5_26.8_t_2.25_2.48'] = {'CAGR': ab['ALL']['CAGR'], 't': t, 'holds': bool(25.5 <= ab['ALL']['CAGR'] <= 26.8 and 2.25 <= t <= 2.48),
                                       'flag_gt_AB_V4': bool(ab['ALL']['CAGR'] > T['AB_V4REF|normal']['ALL']['CAGR']), 'flag_lt_AB_B10': bool(ab['ALL']['CAGR'] < cons['CAGR'])}
seg = R['paired']['AB_V5_minus_AB_V4']
P['P4_bias_in_S1_S3S4_zero'] = {k: seg[k]['mean_pp'] for k in ('S1', 'S2', 'S3', 'S4')}
P['P4_bias_in_S1_S3S4_zero']['holds'] = bool(abs(seg['S1']['mean_pp']) >= max(abs(seg['S3']['mean_pp']), abs(seg['S4']['mean_pp'])) and abs(seg['S1']['mean_pp']) > 0)
P['P4b_nonelec_gt_elec'] = {'swaps': R['fv_swaps_by_industry_group（AB 逐 seed 合計名額）'], 'explore_disagree_elec': E['disagree_electronics'], 'explore_disagree_nonelec': E['disagree_non_electronics']}
P['P5_start2008_t_lt_2.0'] = {'t': three['start_2008'], 'holds': bool(three['start_2008'] < 2.0)}
F['predictions'] = P
F['jscan'] = {k: J[k] for k in ('t_median', 't_p10', 't_min', 't_max', 'n_ge_2.30', 'n_ge_2.0')}
F['n_versions'] = {'data_rule_versions': it['n_versions_total'], 'strategy_spec_versions': 1}
res['FINAL'] = F
(HERE / 'results.json').write_text(json.dumps(res, ensure_ascii=False, indent=1, default=lambda o: float(o) if isinstance(o, (np.floating, np.integer)) else str(o)), encoding='utf-8')
print(json.dumps(F, ensure_ascii=False, indent=1, default=str))
