#!/usr/bin/env python3
"""彙整事前登記判定與預測對照（只讀正式結果）→ results.json 的 FINAL。用法：python3 -I -B finalize.py"""
import sys, json
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import sc_lib as S

def main():
    S.check_prereg()
    R = S.load_res()
    P, A, O, RB = R['RESULTS_POSTERIOR'], R['RESULTS_ALLOC'], R['RESULTS_OOS'], R['RESULTS_ROBUST']
    mp = P['families']['MAIN']['post']['G2']
    m = A['main']
    def pw(c, hy, x): return m[c][hy][x]['P_win']
    pred = {
        'prior_center_le_0': P['families']['MAIN']['priors']['G2']['mean'] <= 0,
        'tau_0.2_0.5': (P['families']['MAIN']['priors']['G1']['tau'], 0.2 <= P['families']['MAIN']['priors']['G1']['tau'] <= 0.5),
        'AB_post_0.25_0.60': (mp['AB']['mean'], 0.25 <= mp['AB']['mean'] <= 0.60), 'AB_P_0.70_0.92': (mp['AB']['p_pos'], 0.70 <= mp['AB']['p_pos'] <= 0.92),
        'B10_post_0.15_0.45': (mp['B10']['mean'], 0.15 <= mp['B10']['mean'] <= 0.45), 'B10_P_0.60_0.85': (mp['B10']['p_pos'], 0.60 <= mp['B10']['p_pos'] <= 0.85),
        'EXEC_post_0.05_0.35': (mp['EXEC']['mean'], 0.05 <= mp['EXEC']['mean'] <= 0.35), 'EXEC_P_0.55_0.80': (mp['EXEC']['p_pos'], 0.55 <= mp['EXEC']['p_pos'] <= 0.80),
        'EXEC_J_post_0.05_0.35': (mp['EXEC_J']['mean'], 0.05 <= mp['EXEC_J']['mean'] <= 0.35), 'EXEC_J_P_0.55_0.80': (mp['EXEC_J']['p_pos'], 0.55 <= mp['EXEC_J']['p_pos'] <= 0.80),
        'LOCO_stronger': {c: (P['families']['LOCO']['post']['G2'][c]['mean'], P['families']['LOCO']['post']['G2'][c]['mean'] < mp[c]['mean']) for c in ('AB', 'B10', 'EXEC', 'EXEC_J')},
        'falsify_post_ge_80pct_raw': P['falsify_post_ge_80pct_raw'],
        'EXEC_J_x100_Pwin_10y_0.52_0.66': (pw('EXEC_J', '10y', '100'), 0.52 <= pw('EXEC_J', '10y', '100') <= 0.66),
        'EXEC_J_x100_Pwin_20y_0.54_0.72': (pw('EXEC_J', '20y', '100'), 0.54 <= pw('EXEC_J', '20y', '100') <= 0.72),
        'EXEC_x100_Pwin_10y_0.52_0.66': (pw('EXEC', '10y', '100'), 0.52 <= pw('EXEC', '10y', '100') <= 0.66),
        'Pwin_spread_x20_100_le_6pp_10y': {c: (round((max(pw(c, '10y', x) for x in ('20', '30', '40', '50', '60', '70', '80', '90', '100')) - min(pw(c, '10y', x) for x in ('20', '30', '40', '50', '60', '70', '80', '90', '100'))) * 100, 2)) for c in ('AB', 'B10', 'EXEC', 'EXEC_J')},
        'EXEC_J_P_ratio_lt_0.75_x100_15_30pct': m['EXEC_J']['10y']['100']['P_ratio_lt_0.75'], 'EXEC_J_P_ratio_lt_0.75_x30_0_5pct': m['EXEC_J']['10y']['30']['P_ratio_lt_0.75'],
        'mdd20y_median_x100_-60_-70': m['EXEC_J']['20y']['100']['mdd_p50'], 'mdd20y_median_x50_-55_-62': m['EXEC_J']['20y']['50']['mdd_p50'],
        'rel_mdd_median_x100_-35_-50': (m['EXEC_J']['10y']['100']['rel_mdd_p50'], m['EXEC_J']['20y']['100']['rel_mdd_p50']),
        'x_star': {c: A['kelly_grid'][c]['G2|L6|10y']['x_star'] for c in ('AB', 'B10', 'EXEC', 'EXEC_J')},
    }
    final = {'decision': A['decision'], 'predictions': pred,
             'n_tests_disclosure': {'model_versions': '3（v1、v2、v3）＋ v3 格點修正重跑 1 次＋ v3 定義前合成診斷 6 組＋冒煙 1 次（皆只用合成資料）',
                                    'prior_models': 3, 'families': 4, 'external_priors': 2, 'flat_control': 1,
                                    'focus_series': '主 4（AB、B10、EXEC、EXEC_J）＋敏感度 8', 'block_lengths': 2, 'rebalance': 2, 'horizons': 2,
                                    'robust_cases': len(RB['cases']), 'decision_column': 'G2×MAIN×EXEC_J×L6×每窗×10 年（事前登記單一欄）'},
             'oos_summary': {mo: {p: {'MSE_raw': O['models'][mo][p]['all']['MSE_raw'], 'MSE_post': O['models'][mo][p]['all']['MSE_post'],
                                      'slope_real_on_raw': O['models'][mo][p]['all']['slope_real_on_raw'], 'cover80': O['models'][mo][p]['all']['cover80_post'],
                                      'top5pct_raw_post_real': (O['models'][mo][p]['all']['top5pct_raw_mean'], O['models'][mo][p]['all']['top5pct_post_mean'], O['models'][mo][p]['all']['top5pct_real_mean'])}
                                  for p in ('VAL', 'HOLD')} for mo in ('G1', 'G2', 'G3')}}
    S.save_key('FINAL', final)
    print(json.dumps(final, ensure_ascii=False, indent=1, default=str))

if __name__ == '__main__':
    main()
