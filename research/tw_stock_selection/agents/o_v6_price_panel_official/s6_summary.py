#!/usr/bin/env python3
"""彙整（PREREG 之後；只讀已產出的淨值與結果，不改任何規格）：
(1) 共同基期的年化比較（V4 淨值在 2024-01 沒有日期，HOLD 一律以 2023-12-29 收盤為基期，避免兩邊起點不同）；
(2) 獨立複製比對（replicate/）；(3) 事前預測計分；(4) 事前登記的最終判定。寫入 results.json 的 SUMMARY（標註為事後描述）。"""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
import v6common as C, v6stats as ST
RES = C.check_prereg()
NV = pd.read_parquet(C.HERE / 'navs_v6.parquet'); N4 = pd.read_parquet(C.V4DIR / 'navs_v4.parquet'); e = C.e50()
def cagr(nav, a, b):
    s = nav.dropna(); s = s[~s.index.duplicated()]
    a, b = pd.Timestamp(a), pd.Timestamp(b)
    va = s[s.index <= a].iloc[-1]; vb = s[s.index <= b].iloc[-1]; da = s[s.index <= a].index[-1]; db = s[s.index <= b].index[-1]
    return float((vb / va) ** (365.25 / (db - da).days) - 1) * 100
out = {'note': '事後描述（非事前登記的判定量）；判定一律用 RESULTS_RUN／RESULTS_HOLD_SPA 的事前登記欄位'}
cb = {}
for k6, k4 in (('AB_V6|normal', 'AB_V4|normal'), ('AB_V6_B10|normal', 'AB_V4_B10|normal'), ('V6|normal', 'V4|normal'), ('AB_V6|stress', 'AB_V4|stress')):
    cb[k6] = {}
    for lab, a, b in (('ALL', '2006-04-11', '2026-10-12'), ('HOLD_from_2023-12-29', '2023-12-29', '2026-10-12'), ('2024-01-02~2024-02-15', '2023-12-29', '2024-02-15'),
                      ('2024-02-15~2026-10-12', '2024-02-15', '2026-10-12')):
        cb[k6][lab] = {'V6': cagr(NV[k6], a, b), 'V4': cagr(N4[k4], a, b), '0050': cagr(e, a, b)}
        cb[k6][lab]['delta_V6_minus_V4'] = cb[k6][lab]['V6'] - cb[k6][lab]['V4']
out['common_base_CAGR'] = cb
# 2024-01 兩窗（V6 實際執行 2024-01-11 換股）
R = RES['RESULTS_HOLD_SPA']
out['gap_windows'] = R['tsmc_decomp']['AB_V6|windows_2023_12_to_2024_02']
# 複製
rc = json.loads((C.HERE / 'replicate' / 'rep_compare.json').read_text())
rs = json.loads((C.HERE / 'replicate' / 'rep_stats.json').read_text())
REP = pd.read_parquet(C.HERE / 'replicate' / 'out_rep.parquet')
nd = {}
for a, b in (('AB_V6', 'AB_V6|normal'), ('AB_V6_B10', 'AB_V6_B10|normal'), ('V6', 'V6|normal')):
    x = NV[b].dropna(); y = REP[a].reindex(x.index); y = y / y.iloc[0] * x.iloc[0]
    nd[a] = float((y / x - 1).abs().max())
sp = R['spa']
out['replication'] = {
    'panel_cell_agree': {k: rc[k]['agree'] for k in ('adj', 'open', 'close_raw')}, 'adj_daily_ratio_agree': rc['adj']['daily_ratio_agree'],
    'panel_note': 'adj 的 260 格差異全部來自 1225 在 2025-09-24（窗後第一筆）一格 ret：凍結規格只寫「依新前收重算」，研究員程式對 V4 已標 flag 2（除權息）的窗後第一筆保留原 ret；low 欄複製員沒做窗外補值（非比對欄）',
    'nav_max_rel_diff': nd,
    'spa_p_focus_L6': {'AB_V6': {'researcher': sp['AB_V6|L6']['p_SPA_c_focus'], 'replicator': rs['AB_V6|normal|L6']['p_SPA_c_focus']},
                       'AB_V6_B10': {'researcher': sp['AB_V6_B10|L6']['p_SPA_c_focus'], 'replicator': rs['AB_V6_B10|normal|L6']['p_SPA_c_focus']}},
    'spa_family_note': '複製員用 replicate_o5 自己建的族群（2,387 條）與自寫 SPA；研究員用 o_main_spa_breaks 的 2,319 條',
    'hold_p_lower_L6': {'AB_V6': {'researcher': R['hold']['AB_V6']['x50_calendar_v6grid']['L6']['p_lower_cumlog'], 'replicator': rs['hold|AB_V6']['p_lower_L6']},
                        'AB_V6_B10': {'researcher': R['hold']['AB_V6_B10']['x50_calendar_v6grid']['L6']['p_lower_cumlog'], 'replicator': rs['hold|AB_V6_B10']['p_lower_L6']}}}
r = out['replication']
rs2 = json.loads((C.HERE / 'replicate' / 'rep_stats2.json').read_text())
r['spa_same_family_replicator_impl_L6'] = {'AB_V6': rs2['AB_V6|normal|L6']['p_SPA_c_focus'], 'AB_V6_B10': rs2['AB_V6_B10|normal|L6']['p_SPA_c_focus']}
r['spa_same_family_diff'] = {k: abs(sp[f'{k}|L6']['p_SPA_c_focus'] - v) for k, v in r['spa_same_family_replicator_impl_L6'].items()}
r['pass'] = {'panel>=99.9%': all(v >= 0.999 for v in r['panel_cell_agree'].values()), 'nav<=1e-8': all(v <= 1e-8 for v in nd.values()),
             'spa_diff<=0.02': {k: abs(v['researcher'] - v['replicator']) for k, v in r['spa_p_focus_L6'].items()},
             'hold_diff<=0.02': {k: abs(v['researcher'] - v['replicator']) for k, v in r['hold_p_lower_L6'].items()}}
# 預測計分
RR = RES['RESULTS_RUN']; T = RR['table']; J = RES.get('RESULTS_JSCAN', {})
v4t = json.loads((C.V4DIR / 'results.json').read_text(encoding='utf-8'))['RESULTS']['table']
dA = T['AB_V6|normal']['ALL']['CAGR'] - v4t['AB_V4|normal']['ALL']['CAGR']; tA = T['AB_V6|normal']['ALL']['t_0050']
b6 = json.loads((C.DATA / 'blk_v6.json').read_text()); b4 = json.loads((C.DATA / 'blk_v4.json').read_text())
def nb(b, k): return sum(len(v) for d, v in b[k].items() if '2024-02-01' <= d <= '2025-09-30')
ex = json.loads((C.HERE / 'explore.json').read_text(encoding='utf-8'))['approx_error_src3_vs_official']
hc = R['hold']['AB_V6']['x50_calendar_v6grid']
out['predictions'] = {
    'P2_close_abs_median_pct': ex['close']['all']['abs_median_pct'], 'P2_ret_corr': ex['daily_ret_corr'],
    'P2_subgroups': {g: ex['close'][g]['abs_median_pct'] for g in ('close_near_limit(|r|>=9%)', 'not_near_limit', 's52_ge_0.95', 's52_lt_0.95', 'V60_rank_1_50', 'V60_rank_201_250', 'V60_rank_251_300')},
    'P4_dCAGR_ALL': dA, 'P4_t': tA, 'P4_nonzero_windows': RR['paired_V6_minus_V4']['AB_V6|normal − V4']['ALL']['n_nonzero'],
    'P4_dCAGR_HOLD_common_base': cb['AB_V6|normal']['HOLD_from_2023-12-29']['delta_V6_minus_V4'],
    'P4_blocked_2024_02_to_2025_09': {'V6': nb(b6, 'V'), 'V4': nb(b4, 'V'), 'AB_s0_V6': nb(b6, 'AB_s0'), 'AB_s0_V4': nb(b4, 'AB_s0')},
    'P5_hold_windows': hc['actual']['n_hold'], 'P5_p_lower_L6': hc['L6']['p_lower_cumlog'], 'P5_range': hc['p_lower_range'],
    'P6_spa': R['decision']['SPA_focus_p_L6'], 'P7_tsmc_share': R['decision']['tsmc_gap_over_shortfall_calendar']}
# 最終判定（事前登記規則）
gate_ok = RES['GATE']['PASS']; rep_ok = r['pass']['panel>=99.9%'] and r['pass']['nav<=1e-8'] and max(r['pass']['hold_diff<=0.02'].values()) <= 0.02
out['decision'] = {
    'data_version': 'V6 取代 V4 成為正式資料版本' if (gate_ok and rep_ok) else 'V4 照舊（複製或閘門未過）',
    'sop_numbers_rule': f'|Δ年化| = {abs(dA):.3f}pp ' + ('≤ 1pp → SOP 正式數字直接換成 V6' if abs(dA) <= 1 else '> 1pp → 需拆解'),
    'grade': R['decision']['grade'], 'hold': R['decision']['hold_judgement'],
    'live_expect_rule': 'AB_V6_B10 可執行版 j 分布中位數（見 RESULTS_JSCAN.AB_V6_B10|normal.exec_j）'}
C.save_key('SUMMARY_posthoc', out)
print(json.dumps(out, ensure_ascii=False, indent=1, default=float)[:6000])
