#!/usr/bin/env python3
"""彙整 G1～G8 與最終判定 → results.json 的 FINAL（只讀前面各段結果）。"""
import sys, json, datetime
sys.dont_write_bytecode = True
from pathlib import Path
HERE = Path(__file__).resolve().parent
RF = HERE / 'results.json'
r = json.loads(RF.read_text(encoding='utf-8'))
R, P, V = r['RESULTS'], r['POSTHOC_FIX'], r['REVIEW']
foc = P['sleeves']['C1xCL|100萬|FIX|normal']
G = {
 'G1': {'pass': R['G1_summary']['FULL|exec']['p10_mean_pct'] > 0 and P['timing_sleeve_100萬FIX_normal']['ex_pp_p10'] > 0,
        'event_exec_FULL_median_p10_pct': [R['G1_summary']['FULL|exec']['median_mean_pct'], R['G1_summary']['FULL|exec']['p10_mean_pct']],
        'event_exec_stress_FULL_median_p10_pct': [R['G1_summary']['FULL|exec_stress']['median_mean_pct'], R['G1_summary']['FULL|exec_stress']['p10_mean_pct']],
        'sleeve_100萬FIX_ex_pp_median_p10': [P['timing_sleeve_100萬FIX_normal']['ex_pp_median'], P['timing_sleeve_100萬FIX_normal']['ex_pp_p10']],
        'sleeve_stress_ex_pp_median_p10': [P['timing_sleeve_100萬FIX_stress']['ex_pp_median'], P['timing_sleeve_100萬FIX_stress']['ex_pp_p10']]},
 'G2': {'pass': (P['G2']['FULL_t'] or 0) >= 3.0 and (P['G2']['2019+_t'] or 0) >= 2.0, 'FULL_t': P['G2']['FULL_t'], '2019+_t': P['G2']['2019+_t'],
        'timing_median_t': P['timing_sleeve_100萬FIX_normal']['t_full_median'], 'prereg_bugged_FIX_t': R['G2']['FULL_t']},
 'G3': {'pass': P['G3']['SPA_c_focus_p'] < 0.05 and P['G3']['DSR_LiJi']['DSR'] >= 0.95, 'SPA_c_p': P['G3']['SPA_c_focus_p'],
        'rank': P['G3']['focus_rank'], 'family': P['G3']['family_size'], 'DSR': P['G3']['DSR_LiJi']['DSR'], 'best_variant_SPA_p': P['G3']['best_this_SPA_c_p']},
 'G4': {'pass': False, 'segments_ex_pp': [foc['SEG_2006_2012'].get('ex_month_mean_pp'), foc['SEG_2013_2018'].get('ex_month_mean_pp'), foc['SEG_2019_2026'].get('ex_month_mean_pp')],
        'remove_best3_t': P['G2']['remove_best3_years']['t'], 'neighborhood_sign_flip': False},
 'G5': {'pass': True, 'note': '公告日一律早於處置第 1 日（2,218 筆，領先 1～2 交易日）；含下市股（P 下市價），OOS 缺價 4 筆最壞界限 +2.10%、t 1.95；0050 贖回時點有輕微日內前視（cashbuffer 版另報）'},
 'G6': {'pass': False, 'numeric_match': V['stat']['G6'], 'note': '程式獨立重寫、只讀凍結規格，數字一致（差 0.001pp、t 0.002）；但複製者與研究員為同一代理，不符「另一位研究員」→ 待外部複製'},
 'G7': {'pass': False, 'GROW_ex_pp_t': {k: [P['sleeves'][f'C1xCL|{k}|GROW|normal']['FULL'].get('ex_month_mean_pp'), P['sleeves'][f'C1xCL|{k}|GROW|normal']['FULL'].get('t_vs_0050')] for k in ('100萬', '300萬', '1000萬')},
        'GROW_stress_ex_pp': {k: P['sleeves'][f'C1xCL|{k}|GROW|stress']['FULL'].get('ex_month_mean_pp') for k in ('100萬', '300萬', '1000萬')},
        'note': '三級距月超額都 > 0 但都不顯著（t 1.67／1.39／0.91），壓力成本下 1,000 萬約 0'},
 'G8': {'pass': None, 'note': '前瞻 12 個月由題 3 帳本自 2026-11 起追蹤；未開始'},
}
r['FINAL'] = {'time_utc': datetime.datetime.now(datetime.UTC).replace(tzinfo=None).isoformat(timespec='seconds') + 'Z', 'G': G,
              'verdict': '待觀察', 'mechanism_supported': '部分支持',
              'sop': '處置衛星上限維持 20%（P1 未失敗，「降為 0」規則未觸發），不得上調；最高標準 G2、G3、G4、G6、G7 不通過，不可投資；列入前瞻帳本'}
RF.write_text(json.dumps(r, ensure_ascii=False, indent=1, default=str), encoding='utf-8')
print(json.dumps(G, ensure_ascii=False, default=str))
