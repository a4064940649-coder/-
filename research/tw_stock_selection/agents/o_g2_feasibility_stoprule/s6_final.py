#!/usr/bin/env python3
"""彙整：iterations.json（生成器變體）、最綁門檻、停損規則自動判定、事前預測核對、可用條件、主策略同表、DSR 可達性分析。
寫入 results.json 的 FINAL 與 iterations.json。"""
import json, math, sys
import numpy as np, pandas as pd
from scipy import stats
sys.path.insert(0, __import__('os').path.dirname(__file__))
import g2lib as G

r = G.check_prereg(need2=True)
FAM = r['FAMILY']; FUN = r['FUNDS']
S = {v: json.loads((G.DATA / f'sim_{v}.json').read_text()) for v in ['V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7'] if (G.DATA / f'sim_{v}.json').exists()}
V1 = S['V1']
GATES = ['G1', 'G2a', 'G2b', 'G3a', 'G3b', 'G4a', 'G4b', 'G4c', 'G7']
IRS = list(V1['by_IR'].keys())

def binding(res):
    single = min(GATES, key=lambda g: res[g])
    others = [g for g in GATES if g != 'G3b']
    single_ex = min(others, key=lambda g: res[g])
    marg = max(GATES, key=lambda g: res['joint_without'][g] - res['joint_G1_G7'])
    return {'single_lowest': single, 'single_lowest_excl_G3b': single_ex, 'marginal': marg,
            'single_ranked': sorted(GATES, key=lambda g: res[g])[:4]}

tab = {}
for ir in IRS:
    x = V1['by_IR'][ir]
    tab[ir] = {'joint_G1_G7': x['joint_G1_G7'], 'joint_without_DSR': x['joint_without_G3b'], **{g: x[g] for g in GATES},
               'A': x['A'], 'A_plus_G1_G7': x['A_plus_G1_G7'], 'B': x['B'], 'B_without_DSR': x['B_without_G3b'], 'C': x['C'], 'C_without_DSR': x['C_without_G3b'],
               'power_seq24': x['power_seq24'], 'power_fixed24': x['power_fixed24'], 'median_tA_jmed': x['median_tA'], 'median_tY_jmed': x['median_tY'], **binding(x)}

# 迭代紀錄
it = {'n_variants': len(S), 'note': '只改生成器實作細節；門檻與停損規則不變。V1 為事前指定主變體。', 'variants': {}}
for v, o in S.items():
    it['variants'][v] = {'desc': r['PREREG']['variants_iterations'].get('V1_main' if v == 'V1' else v), 'L': o['L'], 'sigma_monthly': o['sigma_monthly'],
                         'joint': {ir: o['by_IR'][ir]['joint_G1_G7'] for ir in o['by_IR']}, 'joint_without_DSR': {ir: o['by_IR'][ir]['joint_without_G3b'] for ir in o['by_IR']},
                         'G2': {ir: o['by_IR'][ir]['G2'] for ir in o['by_IR']}, 'G7': {ir: o['by_IR'][ir]['G7'] for ir in o['by_IR']},
                         'A': {ir: o['by_IR'][ir]['A'] for ir in o['by_IR']}, 'binding_0.5': binding(o['by_IR']['0.5']), 'binding_0.66': binding(o['by_IR']['0.66']),
                         'spa_approx_vs_full': o.get('spa_approx_vs_full_200')}
(G.HERE / 'iterations.json').write_text(json.dumps(it, ensure_ascii=False, indent=1, default=float))
spread = {ir: [min(S[v]['by_IR'][ir]['joint_without_G3b'] for v in S), max(S[v]['by_IR'][ir]['joint_without_G3b'] for v in S)] for ir in IRS}

# DSR 可達性：在 center 的 DSR 定義下需要的 IR
dsr = FAM['DSR_family']
sr0 = dsr['SR0_monthly_rawN']; n = 245
need_m = sr0 + 1.645 / math.sqrt(n - 1)
dsr_reach = {'SR0_monthly': sr0, 'SR0_annual': sr0 * math.sqrt(12), 'IR_needed_for_DSR95_at_median': need_m * math.sqrt(12),
             'SR0_monthly_LiJi': dsr['SR0_monthly_LiJi'], 'IR_needed_LiJi': (dsr['SR0_monthly_LiJi'] + 1.645 / math.sqrt(n - 1)) * math.sqrt(12)}
# 家族限定 DSR（主策略改版鏈 F_main）
F = pd.read_parquet(G.DATA / 'family_ir.parquet')
fm = F[F.F_main & F.SR_m.notna()]
vm = float(np.var(fm.SR_m, ddof=1)); emc = 0.5772156649
for lab, Nn in [('F_main_rawN', len(fm)), ('F_main_N10', 10)]:
    s0 = math.sqrt(vm) * ((1 - emc) * stats.norm.ppf(1 - 1 / Nn) + emc * stats.norm.ppf(1 - 1 / (Nn * math.e)))
    dsr_reach[lab] = {'N': Nn, 'var_SR': vm, 'SR0_monthly': s0, 'IR_needed': (s0 + 1.645 / math.sqrt(n - 1)) * math.sqrt(12)}
# 前瞻延長下的 DSR 下限
var_true_m = (FAM['deconvolution']['main_folder_dedup_iid']['prior_sd'] / math.sqrt(12)) ** 2
dsr_reach['SR0_annual_asymptotic_T_inf'] = math.sqrt(var_true_m) * ((1 - emc) * stats.norm.ppf(1 - 1 / 2335) + emc * stats.norm.ppf(1 - 1 / (2335 * math.e))) * math.sqrt(12)

# 停損規則
p05 = V1['by_IR']['0.5']['joint_G1_G7']
fund_top = FUN['survivors_ge15y_general']['p90_top10pct']
dec_share = FAM['deconvolution']['main_folder_dedup_iid']['share_true_ge_0.5']
a = p05 < 0.05; b = fund_top < 0.5; c = dec_share < 0.05
stop = {'S1_a_P_joint_IR05': p05, 'S1_a': a, 'S1_b_fund_top10_IR_upper': fund_top, 'S1_b': b, 'S1_c_share_true_ge05_NPMLE': dec_share, 'S1_c': c,
        'S1_triggered': bool(a and b and c),
        'sensitivity': {'a_without_DSR': V1['by_IR']['0.5']['joint_without_G3b'], 'a_without_DSR_lt_5pct': V1['by_IR']['0.5']['joint_without_G3b'] < 0.05,
                        'a_all_variants_max_without_DSR': max(S[v]['by_IR']['0.5']['joint_without_G3b'] for v in S),
                        'b_all_active_domestic_p90': FUN['survivors_ge15y_all_active_domestic']['p90_top10pct'],
                        'c_normal_EB_folder_dedup': FAM['deconvolution_normal_EB_sensitivity']['folder_dedup']['share_true_ge_0.5'],
                        'c_note': '參數式常態先驗會把左偏族群（大量扣成本後大負的短線規則）的右尾灌大；事前登記的 c 用 NPMLE。若改用常態 EB，(c) 會變成不成立（6.0% ≥ 5%），停損不觸發——列為敏感度並在建議中揭露。'}}

# 主策略同表
mp = V1['main_posterior_points']
def row(lab):
    x = mp[lab]
    return {'IR': x['IR'], 'joint_G1_G7': x['joint_G1_G7'], 'joint_without_DSR': x['joint_without_G3b'], 'G2': x['G2'], 'G7': x['G7'], 'A': x['A'], 'A_plus_G1_G7': x['A_plus_G1_G7'],
            'B': x['B'], 'B_without_DSR': x['B_without_G3b'], 'C': x['C'], 'C_without_DSR': x['C_without_G3b']}
main_tab = {'AB_V4_measured_IR(ideal)': row('AB_V4_measured'), 'AB_V4_posterior_mean': row('AB_V4_post_mean'), 'V7X_posterior_mean(executable)': row('V7X_post_mean')}

# 預測核對
fg = FUN['survivors_ge15y_general']
pred = {
 'P1': {'top10_IR': fg['p90_top10pct'], 'in_0.15_0.50': 0.15 <= fg['p90_top10pct'] <= 0.50, 'median_IR': fg['median'], 'median_lt_0.10': fg['median'] < 0.10,
        'falsified_ge_0.66': fg['p90_top10pct'] >= 0.66, 'Y19_median': fg['IR_Y19']['median'], 'P0618_median': fg['IR_P0618']['median'],
        'sub_prediction_Y19_lower': fg['IR_Y19']['median'] < fg['IR_P0618']['median'], 'verdict': None},
 'P2': {'P_joint_0.5': p05, 'lt_10pct': p05 < 0.10, 'P_joint_0.66': V1['by_IR']['0.66']['joint_G1_G7'], 'in_15_40pct': 0.15 <= V1['by_IR']['0.66']['joint_G1_G7'] <= 0.40,
        'P_joint_0.3': V1['by_IR']['0.3']['joint_G1_G7'], 'lt_1pct': V1['by_IR']['0.3']['joint_G1_G7'] < 0.01,
        'binding_0.5': tab['0.5']['single_lowest'], 'binding_excl_DSR_0.5': tab['0.5']['single_lowest_excl_G3b'], 'binding_excl_DSR_0.66': tab['0.66']['single_lowest_excl_G3b']},
 'P3': {'share_true_gt_0.3': FAM['deconvolution']['main_folder_dedup_iid']['share_true_gt_0.3'], 'lt_10pct': FAM['deconvolution']['main_folder_dedup_iid']['share_true_gt_0.3'] < 0.10,
        'posterior_max': FAM['deconvolution']['main_folder_dedup_iid']['posterior_of_max']['post_mean'], 'max_lt_0.5': FAM['deconvolution']['main_folder_dedup_iid']['posterior_of_max']['post_mean'] < 0.5,
        'AB_V4_shrunk': FAM['posterior_main']['AB_V4_ideal']['post_mean'], 'AB_V4_lt_0.45': FAM['posterior_main']['AB_V4_ideal']['post_mean'] < 0.45},
 'P4': {'ideal_minus_exec_IR_100w': FAM['P4_ideal_vs_exec']['100萬']['ALL']['median_diff'], 'ge_0.15': FAM['P4_ideal_vs_exec']['100萬']['ALL']['median_diff'] >= 0.15,
        'subsample': FAM['P4_ideal_vs_exec']['subsample_small_highturnover']}}
pred['P1']['verdict'] = '部分支持：前 10% 與中位數落在預測區間；2019 年起反而較高（子預測被否證）' if (pred['P1']['in_0.15_0.50'] and pred['P1']['median_lt_0.10'] and not pred['P1']['sub_prediction_Y19_lower']) else ('支持' if pred['P1']['in_0.15_0.50'] and pred['P1']['median_lt_0.10'] else '不支持')

# 可用條件
rep = json.loads((G.HERE / 'replicate' / 'rep_results.json').read_text()) if (G.HERE / 'replicate' / 'rep_results.json').exists() else None
rep_cmp = None
if rep:
    rep_cmp = {}
    for ir in IRS:
        a_ = V1['by_IR'][ir]; b_ = rep['by_IR'][ir]
        rep_cmp[ir] = {'joint_main': a_['joint_G1_G7'], 'joint_rep': b_['joint'], 'diff_pp': (a_['joint_G1_G7'] - b_['joint']) * 100,
                       'joint_noDSR_main': a_['joint_without_G3b'], 'joint_noDSR_rep': b_['joint_without_G3b'], 'diff_noDSR_pp': (a_['joint_without_G3b'] - b_['joint_without_G3b']) * 100,
                       'binding_main': tab[ir]['single_lowest'], 'binding_rep': b_['binding_single'],
                       'binding_excl_DSR_main': tab[ir]['single_lowest_excl_G3b'], 'binding_excl_DSR_rep': min([g for g in GATES if g != 'G3b'], key=lambda g: b_[g]),
                       'binding_tie_aware_same': bool(b_[tab[ir]['single_lowest']] <= min(b_[g] for g in GATES) + 1e-12),
                       **{f'{g}_diff_pp': (a_[g] - b_[g]) * 100 for g in GATES}}
    rep_ok = all(abs(v['diff_pp']) <= 2 for v in rep_cmp.values()) and all(v['binding_tie_aware_same'] for v in rep_cmp.values())
    rep_noDSR_maxdiff = max(abs(v['diff_noDSR_pp']) for v in rep_cmp.values())
else:
    rep_ok = False
usable = {'iii_noDSR_max_abs_diff_pp(非登記條件，MC 標準誤約 1.4pp)': rep_noDSR_maxdiff if rep else None, 'i': True, 'ii': True, 'iii_pre_replication_by_researcher': rep_ok, 'iii_note': '研究員 clean-room 預複製；正式獨立複製由工作流複製員執行，未完成前 (iii) 視為暫時成立', 'iv': True, 'v': True}
FINAL = {'table_V1': tab, 'variant_spread_joint_without_DSR': spread, 'dsr_reachability': dsr_reach, 'stop_rule': stop, 'main_strategy_table': main_tab,
         'predictions': pred, 'replication_compare': rep_cmp, 'usable': usable,
         'years_needed': V1.get('years_needed'), 'forward_extension': V1.get('forward_extension'), 'clean_length': V1.get('clean_length'), 'C_pocock': V1['C_pocock']}
G.save_key('FINAL', json.loads(json.dumps(FINAL, default=lambda o: bool(o) if isinstance(o, np.bool_) else float(o))))
print(json.dumps({'stop': stop, 'dsr': dsr_reach, 'pred': pred, 'usable': usable, 'years': V1.get('years_needed')}, ensure_ascii=False, indent=1, default=float))
print(json.dumps(main_tab, ensure_ascii=False, indent=1, default=float))
for ir in IRS: print(ir, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in tab[ir].items()})
print('spread', spread)
if rep_cmp: print(json.dumps(rep_cmp, ensure_ascii=False, default=float)[:3000])
