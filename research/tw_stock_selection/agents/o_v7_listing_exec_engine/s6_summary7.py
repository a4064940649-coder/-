#!/usr/bin/env python3
"""彙整（PREREG 之後；只讀已產出的淨值與結果，不改任何規格）：
(1) 正式數字表（V7／AB_V7／B10、一般與壓力）與 V7−V4、V7−V6；(2) j 分布；(3) 可執行實盤預期表；(4) 處置期賣出與併購延遲拆解；
(5) 保留期 p_lower 三版、SPA 焦點；(6) 0050 核心＋主策略衛星（描述）；(7) 事前預測計分；(8) 獨立複製比對（超過容許值取不利方向）；
(9) 事前登記規則下的判定。寫入 results.json 的 SUMMARY_posthoc（標註為事後彙整）。"""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
import v7common as C, v7stats as ST
RES = C.check_prereg()
RR, RJ, RE, RH = RES['RESULTS_RUN'], RES['RESULTS_JSCAN'], RES['RESULTS_EXEC'], RES['RESULTS_HOLD_SPA']
NV = pd.read_parquet(C.HERE / 'navs_v7.parquet'); N6 = pd.read_parquet(C.V6DIR / 'navs_v6.parquet'); N4 = pd.read_parquet(C.V4DIR / 'navs_v4.parquet')
e = C.e50()
T = RR['table']
out = {'note': '事後彙整（非新的判定量）；判定一律用事前登記欄位'}
def row(k):
    t = T[k]
    return {'CAGR': t['ALL']['CAGR'], 't_0050': t['ALL']['t_0050'], 'maxDD': t['ALL']['maxDD'], 't_EW': t['ALL']['t_EW'], 't_cal': t['ALL']['t_cal'],
            'SEL_t': t['SEL']['t_0050'], 'VAL_t': t['VAL']['t_0050'], 'HOLD_t': t['HOLD']['t_0050'], 'Y19on_t': t['Y19on']['t_0050'], 'POST2019_05_t': t['POST2019_05']['t_0050'],
            'Y19on_CAGR': t['Y19on']['CAGR'], 'Y19on_E50': t['Y19on']['E50_CAGR'], 'HOLD_CAGR': t['HOLD']['CAGR'], 'HOLD_E50': t['HOLD']['E50_CAGR'],
            'HOLD_cal_mean_pp': t['HOLD_cal']['mean_pp'], 'HOLD_cal_t': t['HOLD_cal']['t_0050'], 'HOLD_cal_cumlog': t['HOLD_cal']['cum_log_vs_0050'],
            'E50_CAGR': t['ALL']['E50_CAGR'], 'EW_CAGR': t['ALL']['EW_CAGR']}
main = {k: row(k) for k in ('V7|normal', 'AB_V7|normal', 'V7_B10|normal', 'AB_V7_B10|normal', 'V7|stress', 'AB_V7|stress', 'V7_B10|stress', 'AB_V7_B10|stress') if k in T}
out['main'] = main
out['endpoint'] = {k: T[k].get('endpoint') for k in ('AB_V7|normal', 'AB_V7_B10|normal', 'AB_V7|stress', 'AB_V7_B10|stress', 'V7|normal', 'V7_B10|normal') if k in T}
out['seed_t'] = RR['seed_t']; out['seed_t_Y19on'] = RR.get('seed_t_Y19on'); out['breakeven'] = RR['breakeven']
out['yearly'] = {k: T[k].get('yearly') for k in ('AB_V7|normal', 'AB_V7_B10|normal')}
# 對 V4、V6
def cagr(nav, a, b):
    s = nav.dropna(); s = s[~s.index.duplicated()]
    a, b = pd.Timestamp(a), pd.Timestamp(b)
    va = s[s.index <= a].iloc[-1]; vb = s[s.index <= b].iloc[-1]; da = s[s.index <= a].index[-1]; db = s[s.index <= b].index[-1]
    return float((vb / va) ** (365.25 / (db - da).days) - 1) * 100
cb = {}
for k7 in ('AB_V7|normal', 'AB_V7_B10|normal', 'V7|normal', 'AB_V7|stress'):
    cb[k7] = {}
    for lab, a, b in (('ALL', '2006-04-11', '2026-10-12'), ('HOLD_from_2023-12-29', '2023-12-29', '2026-10-12'), ('2019-01-02~', '2018-12-28', '2026-10-12')):
        cb[k7][lab] = {'V7': cagr(NV[k7], a, b), 'V6': cagr(N6[k7.replace('V7', 'V6')], a, b), 'V4': cagr(N4[k7.replace('V7', 'V4')], a, b), '0050': cagr(e, a, b)}
out['common_base_CAGR'] = cb
pr4 = RR['paired_V7_minus_V4']; pr6 = RR['paired_V7_minus_V6']
out['paired'] = {'V7−V4': {k: {p: {kk: v[p][kk] for kk in ('mean_pp', 't', 'n_nonzero') if kk in v[p]} for p in ('ALL', 'HOLD')} | {'CAGR_diff_ALL': v['CAGR_diff_ALL'], 'largest': v['ALL'].get('largest_month'), 'largest_pp': v['ALL'].get('largest_pp'), 't_drop_largest': v['ALL'].get('t_drop_largest')} for k, v in pr4.items()},
                 'V7−V6': {k: {p: {kk: v[p][kk] for kk in ('mean_pp', 't', 'n_nonzero') if kk in v[p]} for p in ('ALL', 'HOLD')} | {'CAGR_diff_ALL': v['CAGR_diff_ALL'], 'largest': v['ALL'].get('largest_month'), 'largest_pp': v['ALL'].get('largest_pp'), 't_drop_largest': v['ALL'].get('t_drop_largest'), 'nonzero_windows_pp': v.get('nonzero_windows_pp')} for k, v in pr6.items()}}
out['picks_diff_V7_vs_V6'] = RR.get('picks_diff_V7_vs_V6')
out['jscan'] = {k: {kk: v[kk] for kk in ('t_median', 't_p10', 't_min', 't_max', 'CAGR_median', 'CAGR_p10', 'n_ge_2.30', 'n_ge_2.0', 't_Y19on_median')} for k, v in RJ.items() if isinstance(v, dict) and 't_median' in v}
out['exec_live_table'] = RE['live_expect_table']; out['exec_decomp_zero_impact'] = RE['decomp_zero_impact']
out['exec_identity'] = RE.get('ideal_identity_maxrel')
js = RE['jsumm']
out['exec_jsumm_main'] = {k: {kk: v[kk] for kk in ('CAGR_median', 'CAGR_p10', 't_median', 't_p10', 'excess_vs_0050_median_pp', 'E50_CAGR_median', 'maxDD_median', 't_Y19on_median', 'CAGR_HOLD_median', 'E50_CAGR_HOLD_median')} for k, v in js.items()}
out['exec_official_day'] = {k: {kk: v[kk] for kk in ('CAGR', 't_0050', 'maxDD', 'CAGR_minus_0050', 't_Y19on', 'seed_t') if kk in v} | {'cost_pp_per_year': v['acc']['cost_pp_per_year'], 'n_disp_sells': v['acc']['n_disp_sells'], 'n_cap_hit': v['acc']['n_cap_hit'], 'deferred_days': v['acc']['deferred_days']}
                            for k, v in RE['table'].items() if '|OFF|' in k}
# 保留期、SPA
hd = RH['decision']
out['hold'] = {'AB': hd['hold_p_lower_L6_three_versions'], 'B10': hd['hold_p_lower_L6_three_versions_B10'], 'judgement_raw': hd['hold_judgement'],
               'MDE_pp': hd['hold_MDE_pp'], 'p_lower_range_raw': hd['hold_p_lower_range_calendar']}
out['spa'] = {'focus_L6': hd['SPA_focus_p_L6'], 'all': {k: v for k, v in RH['spa'].items() if '|L' in k}, 'eff_trials': RH['spa'].get('eff_trials_F_long_V6')}
out['tsmc'] = {k: v for k, v in RH['tsmc_decomp'].items() if k.endswith('calendar')}
# 0050 核心＋主策略衛星（描述；每個訊號日再平衡，零衝擊）
sd = [pd.Timestamp(x) for x in pd.read_csv(C.HERE / 'plan_V7.csv')['signal']] + [pd.Timestamp(ST.B0)]
def mix(nav, w):
    r = ST.win(nav, sd).r; rb = ST.win(e, sd).r
    m = (1 - w) * rb + w * r
    navm = (1 + m).cumprod()
    yrs = (sd[-1] - sd[0]).days / 365.25
    dd = float((navm / navm.cummax() - 1).min() * 100)
    x = m - rb
    return {'CAGR_windows': float(navm.iloc[-1] ** (1 / yrs) - 1) * 100, 'maxDD_monthly': dd, 't_vs_0050': ST.tstat(x.to_numpy()), 'TE_pp_annual': float(x.std() * np.sqrt(12) * 100)}
out['core_satellite'] = {f'{int((1 - w) * 100)}%0050+{int(w * 100)}%{k}': mix(NV[k], w) for k in ('AB_V7|normal', 'AB_V7_B10|normal') for w in (0.3, 0.5, 0.7, 1.0)}
out['core_satellite']['0050'] = mix(NV['AB_V7|normal'], 0.0)
# 預測計分
G = RES['GATE']
a7 = main['AB_V7|normal']
pr = {'P1': {'gate_cov': G['coverage_conservative']['official_share'], 'n_hole_in_top300_range': G['prediction1']['range'], 'n_pick_changes': G['prediction1']['n_pick_changes_total_2024_2025']},
      'P2': {'dCAGR_V7_V4': pr4['AB_V7|normal − V4']['CAGR_diff_ALL'], 'paired_t_V7_V4': pr4['AB_V7|normal − V4']['ALL']['t'], 'dCAGR_V7_V6': pr6['AB_V7|normal − V6']['CAGR_diff_ALL'],
             'paired_t_V7_V6': pr6['AB_V7|normal − V6']['ALL']['t'], 'AB_V7_t': a7['t_0050'], 'j_median_t_AB': RJ['AB_V7|normal']['t_median']},
      'P3': {'Y19on_t_AB': a7['Y19on_t'], 'Y19on_t_B10': main['AB_V7_B10|normal']['Y19on_t'], 'HOLD_cal_mean_pp': a7['HOLD_cal_mean_pp']},
      'P4': {k: {m: RE['decomp_zero_impact'][k][m]['dCAGR_pp'] for m in ('disp', 'mdelay', 'both')} for k in ('AB|OFF', 'B10|OFF')},
      'P5': RE['live_expect_table'], 'P6': hd['SPA_focus_p_L6']}
out['predictions'] = pr
# 複製
rep = {}
for f in ('rep_compare.json', 'rep_stats.json', 'rep_stats2.json', 'rep_exec.json', 'rep_nav.json'):
    p = C.HERE / 'replicate' / f
    if p.exists(): rep[f] = json.loads(p.read_text(encoding='utf-8'))
out['replication_raw_keys'] = list(rep)
C.save_key('SUMMARY_posthoc', out)
print(json.dumps(C.conv({k: out[k] for k in ('main', 'paired', 'jscan', 'exec_live_table', 'hold', 'spa', 'predictions', 'core_satellite')}), ensure_ascii=False, indent=0, default=str)[:12000])
