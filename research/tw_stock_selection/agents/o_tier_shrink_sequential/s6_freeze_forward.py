#!/usr/bin/env python3
"""步驟 4＋6：凍結前瞻序貫判定規則（PREREG_FORWARD）並寫 prereg_forward.json（含雜湊）。
門檻取步驟 5（STEP5_SEQSIM）登記內調整後的最終值；先驗密度 data/forward_priors.npz（由 RESULTS 產生）；
配置（L1 有利格／不利格）取 tier_table.json 的寫入值（範圍取下界）。
凍結後不得引用 2026-10 以前的報酬調整任何參數；歷史只透過先驗進入。
用法：python3 -I -B s6_freeze_forward.py
"""
import sys, json, hashlib, re
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import tlib as T

TIERS = ['30萬', '100萬', '300萬', '1000萬']; MODES = ['FIX', 'GROW']

def low(w):
    return int(re.findall(r'\d+', w)[0])

def main():
    res = T.check_block('PREREG')
    for k in ('RESULTS', 'STEP5_SEQSIM'): assert k in res, k
    assert 'PREREG_FORWARD' not in res, '已凍結'
    R = res['RESULTS']; S5 = res['STEP5_SEQSIM']
    tt = json.loads((HERE / 'tier_table.json').read_text(encoding='utf-8'))
    FP = R['forward_priors']
    thr = dict(S5['thresholds_final']); thr['dead_zone_pp'] = 20; thr['max_changes_per_year'] = 1
    def key(n): return n.replace('|', '__').replace('@', '_AT_')
    L1v = {}
    for t in TIERS:
        for m in MODES:
            nm = f'L1_MAIN_B10_EXEC@{t}|{m}'
            rn = next(r for r in tt['rows'] if r['tier'] == t and r['mode'] == m and r['cost'] == 'normal')
            rs = next(r for r in tt['rows'] if r['tier'] == t and r['mode'] == m and r['cost'] == 'stress')
            L1v[nm] = {'prior_key': key(nm), 'sigma_pp': FP[nm]['sigma_pp'], 'alpha_min': FP[nm]['alpha_min'],
                       'prior_mean_pp': FP[nm]['mean'], 'prior_P_pos': FP[nm]['p_pos'],
                       'allocation': {'favorable': low(rn['x_star_write']), 'unfavorable': low(rs['x_star_write']),
                                      'favorable_written': rn['x_star_write'], 'unfavorable_written': rs['x_star_write']}}
    lines = {'L1_MAIN_B10_EXEC': {'data_key': 'L1', 'desc': '主策略 B10 可執行（實際帳戶；級距／模式由 --tier --mode 指定）', 'allocation': True}}
    desc = {'L2_MAIN80_DISP20': '主策略 80%＋處置 20%（需 ret_disp）', 'L3_MAIN70_E50_30': '主策略 70%＋0050 30%',
            'L4_MAIN70_2330_30': '主策略 70%＋2330 30%（需 ret_2330；事後觀察，不得升級）', 'L5_CAPM_F': '市值匹配選股凍結版 F1_K5_N20_sqrt',
            'L6_CAPM_C50': '0050 50%＋市值匹配 50%', 'L7_CAPM_C70': '0050 70%＋市值匹配 30%', 'L8_REGIME': '權值集中輪動 V5（對 0050）',
            'L9_REGIME_CORE50': '0050 50%＋輪動 50%', 'L10_REGIME_D': '輪動 − 同平均權重靜態混合（o_concentration_regime 登記焦點）'}
    for nm, d in desc.items():
        lines[nm] = {'data_key': nm, 'desc': d, 'prior_key': key(nm), 'sigma_pp': FP[nm]['sigma_pp'], 'alpha_min': FP[nm]['alpha_min'],
                     'prior_mean_pp': FP[nm]['mean'], 'prior_P_pos': FP[nm]['p_pos']}
    body = {'topic': 'o_tier_shrink_sequential 前瞻盲測序貫判定規則', 'forward_start': '2026-11-01',
            'priors_file': 'data/forward_priors.npz', 'lines': lines, 'lines_L1_variants': L1v, 'thresholds': thr,
            'rules_text': {
                'upgrade': f"E_up ≥ {thr['E_up']:g} 且前瞻 ≥ {thr['up_min_months']} 個月 → 主策略第 3 級回第 2 級；新策略由不可用／待觀察升一級的必要條件。2330 30% 線（L4）不得升級（o6 教訓 20）。",
                'down': f"E_down ≥ {thr['E_down']:g}，或（更新後 P(α>0) < {thr['P_down']} 且前瞻 ≥ {thr['P_down_min_months']} 個月）→ 該線配置降到配置表不利格（L1：同級距、同模式的壓力成本格）。",
                'review': f"財富比 < {thr['ratio_review']} 或 E_down ≥ {thr['E_review']:g} → 觸發總監審查（不自動停用）。",
                'constraints': f"任何配置變更：前瞻 ≥ {thr['change_min_months']} 個月、±{thr['dead_zone_pp']}pp 死區、每年最多一次。",
                'stats': 'x_t＝線窗報酬 − 0050 窗報酬（pp）；E_up／E_down 為以登記先驗截斷混合的常態概似比（σ 固定為歷史長期 sd）；P(α>0) 為格點先驗的常態更新。',
                'no_refit': '凍結後不得引用 2026-10 以前的報酬調整任何參數；V7 交件後先驗可依 PREREG 的 V7 條款只換輸入重算一次，σ、門檻不變。'},
            'data_sources': {'MAIN': 'live_tracking/main_forward/forward_returns.csv（start,end,ret[,ret_0050][,ret_disp][,ret_2330]）；缺則 live_tracking/o7_regime/main_leg_returns.csv',
                             'TRADES': 'live_tracking/main_forward/trades.csv（date,code,side,shares,fill_price,open_price,ticks_from_open,pct_open_volume）',
                             'CAPM': 'live_tracking/o7_capmatched/paper_nav.tsv', 'REGIME': 'live_tracking/o7_regime/paper_nav.csv'},
            'simulation_summary': S5['prediction4'],
            'hashes': {f: T.sha(HERE / f) for f in ('forward_monitor.py', 'seqlib.py', 'data/forward_priors.npz', 'tier_table.json', 's5_seqsim.py', 's6_freeze_forward.py')},
            'source_PREREG_sha256': res['PREREG_sha256'], 'utc': T.now_utc()}
    h = hashlib.sha256(json.dumps(body, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
    (HERE / 'prereg_forward.json').write_text(json.dumps({'registration': body, 'registration_sha256': h}, ensure_ascii=False, indent=1), encoding='utf-8')
    T.save_key('PREREG_FORWARD', body); T.save_key('PREREG_FORWARD_sha256', T.canon_sha(body))
    print('prereg_forward', h, body['utc'])
    print(json.dumps({k: v['allocation'] for k, v in L1v.items()}, ensure_ascii=False))

if __name__ == '__main__':
    main()
