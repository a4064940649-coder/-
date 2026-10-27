#!/usr/bin/env python3
"""合併 explore／iterations／OOS／V3 → results.json 的 DECISION 與摘要；explore.json＝A 與 A+P 的探索合併。"""
import json, hashlib, datetime
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
L = lambda f: json.loads((HERE / f).read_text(encoding='utf-8'))
res = L('results.json'); assert 'PREREG_sha256' in res
exA, exAP = L('explore_A.json'), L('explore_AP.json')
(HERE / 'explore.json').write_text(json.dumps({'A': exA, 'AP': exAP}, ensure_ascii=False, indent=1), encoding='utf-8')
it = L('iterations.json'); oA, oAP = L('oos_A.json'), L('oos_AP.json'); v3 = L('v3_results.json'); fz = L('data_freeze.json')
SEL = res['PREREG']['selected_version']
def g(o, v, per, c='normal'): return o['paired'][f'{v}|n10|{c}'][per]
chk = {}
for lab, o, ex in (('A', oA, exA), ('AP', oAP, exAP)):
    p1 = g(o, 'E0', 'SEL'); pl = ex['d_placebo']
    P1 = p1['mean_pp'] > 0 and p1['t'] >= 1.5
    P4_ok = abs(pl['excess']['mean_pp']) < 0.5 * abs(pl['disposal_excess']['mean_pp'])
    s_sel, s_e1 = o['runs'][f'{SEL}|n10|normal']['stats']['ALL'], o['runs']['E1|n10|normal']['stats']['ALL']
    g2 = dict(post2021_diff_pos=g(o, SEL, 'POST')['mean_pp'] > 0,
              cross_cycle_cagr_ge_E1=s_sel['CAGR'] >= s_e1['CAGR'], cross_cycle_t_ge_E1=s_sel['vs0050']['t'] >= s_e1['vs0050']['t'],
              stress_SEL_diff_pos=g(o, SEL, 'SEL', 'stress')['mean_pp'] > 0,
              drop_top3_SEL_diff_pos=o['P5_right_tail_SEL']['paired_SEL_after_drop_top3']['mean_pp'] > 0)
    chk[lab] = dict(P1_pass=bool(P1), P1=p1, P4_placebo_not_as_large=bool(P4_ok), P4=dict(placebo=pl['excess'], disposal=pl['disposal_excess']),
                    gate2=g2, gate2_all=bool(all(g2.values())), P5_top3_share=o['P5_right_tail_SEL']['top3_share_of_total'], P5_top3=o['P5_right_tail_SEL']['top3'],
                    affected_slots_SEL_n10=ex['a_affected_slots']['n10']['total_slots'], affected_slots_SEL_n30=ex['a_affected_slots']['n30']['total_slots'])
v3s = {k: v['stats']['ALL'] for k, v in v3['runs'].items()}
g2v3 = dict(cagr_ge_E1=v3s[f'ORIG|{SEL}']['CAGR'] >= v3s['ORIG|E1']['CAGR'], t_ge_E1=v3s[f'ORIG|{SEL}']['vs0050']['t'] >= v3s['ORIG|E1']['vs0050']['t'])
mde = it['A']['MDE_E0_minus_E1_pp_per_month']
DEC = dict(
  verdict='不可用', mechanism='不支持',
  reason=['P1 挑選期 E0−E1 配對差 +%.3fpp／月、t %.2f（A）／+%.3fpp、t %.2f（A+P），低於 1.5 門檻 → 依事前規則判不可用，SOP 維持排除。'
          % (chk['A']['P1']['mean_pp'], chk['A']['P1']['t'], chk['AP']['P1']['mean_pp'], chk['AP']['P1']['t']),
          '挑選期 n10 受影響名額只有 %d 個（n30 %d 個），< 60 → 檢定力不足；MDE ≈ %.2fpp／月。' % (chk['A']['affected_slots_SEL_n10'], chk['A']['affected_slots_SEL_n30'], mde['mde_80pow_5pct']),
          '選中版本 %s 的閘門 (2)：剔除貢獻最大 3 檔後挑選期配對差轉負（%.3fpp）→ 不通過；P5：前 3 檔貢獻 %.0f%% → 上限待觀察。'
          % (SEL, oA['P5_right_tail_SEL']['paired_SEL_after_drop_top3']['mean_pp'], 100 * chk['A']['P5_top3_share'])],
  checks=chk, gate2_on_V3=g2v3,
  sop_recommendation='維持「刪掉換股日當天在處置期間的股票，由備取遞補」（E1）。不改成 E0／E2／E5。正式回測從此全期套用歷史處置名單（本題 disposal_hist_rejudge.csv），跨循環正式數字由 28.15% 改為 27.47%（t 2.64）。',
  live_tracking='2026-11 起每個換股日記錄：入選前 10 名中是否為處置股（公告日、起訖、第幾次、剩餘天數）、E1 與 E5／E0 的紙上名單與淨值差；新制滿 12 個月（2027-08）重評，紙上差 ≥ 0 且受影響名額 ≥ 15 才重開此題。',
)
res['DECISION'] = DEC
res['SUMMARY'] = dict(
  selection_A=it['A']['selection'], selection_AP=it['AP']['selection'], MDE=mde,
  cross_cycle={k: {v: oA['runs'][f'{v}|n10|normal']['stats'][k] for v in ('E1', 'E0', SEL)} for k in ('SEL', 'SEL_ex2021', 'VAL', 'HOLD', 'POST', 'ALL')},
  official_main_v2=oA['official_main_v2_stats'], paired_A={v: oA['paired'][f'{v}|n10|normal'] for v in ('E0', 'E2', 'E3', 'E4', 'E5')},
  paired_A_stress={v: oA['paired'][f'{v}|n10|stress'] for v in ('E0', 'E2', 'E3', 'E4', 'E5')},
  paired_A_n30={v: oA['paired'][f'{v}|n30|normal'] for v in ('E0', 'E2', 'E3', 'E4', 'E5')},
  v3=v3, list_agreement=dict(disposal_5y=fz['agree_disposal_5y'], infra_v3_proxy=v3['agree_with_infra_v3_list']),
  t_decomposition=oA['t_decomposition_vs_0050'], yearly=oA['yearly'], yearly_diff=oA['yearly_diff_vs_E1_pp'], affected=oA['affected'],
  P5=oA['P5_right_tail_SEL'], P5_E0=oA['P5_right_tail_E0_SEL'], post2021_event_level=oA['post2021_event_level_report_only'], overlap=oA['overlap_with_disposal_sleeve'])
res['FILES_sha256'] = {f: hashlib.sha256((HERE / f).read_bytes()).hexdigest() for f in ['disposal_hist_rejudge.csv', 'explore.json', 'iterations.json', 'oos_A.json', 'oos_AP.json', 'v3_results.json', 'oos.py', 'v3run.py', 'finalize.py']}
res['FINAL_time_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds')
(HERE / 'results.json').write_text(json.dumps(res, ensure_ascii=False, indent=1, default=float), encoding='utf-8')
print(json.dumps(DEC, ensure_ascii=False, indent=1, default=float)[:4000])
