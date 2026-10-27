"""步驟 10：彙整各步結果成 results_replica.json（關鍵數字攤平）。"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import rlib as L
b, ex, it, oo, ph, sl, p4 = (L.load(f) for f in ['build_events.json', 'explore.json', 'iterate.json', 'oos.json', 'posthoc.json', 'sleeve.json', 'p4.json'])
c = ex['classes']
R = {
 'gate.msci_del_valid': b['gate']['msci_del_ex_src3_in_panel'], 'gate.eff_stated_share': b['gate']['msci_eff_stated_share'], 'gate.ftse': b['gate']['ftse_rows'],
 'gate.n_dates': b['gate']['review_dates_ex_src3'], 'gate.vr_min': b['gate']['vr_median_range_ex_src3'][0], 'gate.vr_max': b['gate']['vr_median_range_ex_src3'][1],
 'gate.rows': b['n_rows'], 'gate.in_panel': b['in_panel_n'],
 'exp.n': ex['n_explorable_select'],
 'exp.std_del.n': c['MSCI_STD_DEL']['n'], 'exp.std_del.abE': c['MSCI_STD_DEL']['abE'], 'exp.std_del.abE_t': c['MSCI_STD_DEL']['abE_t_evt'],
 'exp.std_del.vr': c['MSCI_STD_DEL']['vr_median'], 'exp.std_del.abE1': c['MSCI_STD_DEL']['abE1'], 'exp.std_del.car1_20': c['MSCI_STD_DEL']['sc20'],
 'exp.sc_del.n': c['MSCI_SC_DEL']['n'], 'exp.sc_del.abE': c['MSCI_SC_DEL']['abE'], 'exp.sc_del.abE_t': c['MSCI_SC_DEL']['abE_t_evt'],
 'exp.sc_del.car5': c['MSCI_SC_DEL']['sc5'], 'exp.sc_del.car20': c['MSCI_SC_DEL']['sc20'],
 'exp.std_add.n': c['MSCI_STD_ADD']['n'], 'exp.std_add.pre10': c['MSCI_STD_ADD']['sp10'], 'exp.std_add.abE': c['MSCI_STD_ADD']['abE'], 'exp.std_add.car5': c['MSCI_STD_ADD']['sc5'],
 **{f'exp.ter_{k}.vr': v['vr_median'] for k, v in ex['tertiles'].items()},
 **{f'exp.ter_{k}.abE': v['abE'] for k, v in ex['tertiles'].items()},
 **{f'exp.ter_{k}.car20': v['car20_gross'] for k, v in ex['tertiles'].items()},
 'exp.cut_median': ex['pressure_median_select_del_explorable'],
 'v6.sel.n': it['V6']['high']['n'], 'v6.sel.months': it['V6']['high']['n_months'], 'v6.sel.mean': it['V6']['high']['mean'],
 'v6.sel.median': it['V6']['high']['median'], 'v6.sel.t': it['V6']['high']['t_month'], 'v6.sel.trim5': it['V6']['high']['trim5'],
 'v6.sel.HL': it['V6']['HL']['paired'], 'v6.sel.HL_t': it['V6']['HL']['t_paired'], 'v6.sel.stress': it['V6']['high_stress']['mean'],
 'oos.val.n': oo['validate']['high']['n'], 'oos.val.mean': oo['validate']['high']['mean'], 'oos.val.median': oo['validate']['high']['median'], 'oos.val.t': oo['validate']['high']['t_month'],
 'oos.hold.n': oo['holdout']['high']['n'], 'oos.hold.mean': oo['holdout']['high']['mean'],
 'oos.all.n': oo['oos']['high']['n'], 'oos.all.months': oo['oos']['high']['n_months'], 'oos.all.mean': oo['oos']['high']['mean'], 'oos.all.median': oo['oos']['high']['median'],
 'oos.all.t': oo['oos']['high']['t_month'], 'oos.all.trim5': oo['oos']['high']['trim5'], 'oos.all.stress': oo['oos']['high_stress']['mean'], 'oos.all.vs0050': oo['oos']['high_vs0050']['mean'],
 'oos.low.n': oo['oos']['low']['n'], 'oos.low.mean': oo['oos']['low']['mean'],
 'oos.HL.simple': oo['oos']['HL']['simple'], 'oos.HL.paired': oo['oos']['HL']['paired'], 'oos.HL.t': oo['oos']['HL']['t_paired'],
 'oos.seg1920.mean': oo['oos_segments']['2019_2020']['mean'], 'oos.seg1920.n': oo['oos_segments']['2019_2020']['n'],
 'oos.seg21.mean': oo['oos_segments']['2021+']['mean'], 'oos.seg21.n': oo['oos_segments']['2021+']['n'],
 'oos.verdict_i': oo['verdict']['i'], 'oos.verdict_ii': oo['verdict']['ii'], 'oos.verdict_iii': oo['verdict']['iii'],
}
for k, (a, b2) in {'s1': ('2010-11_2018', None), 's2': ('2019_2026-10-12', None)}.items():
    s = sl[a]
    R.update({f'sl.{k}.sleeve': s['sleeve_CAGR'], f'sl.{k}.0050': s['0050_CAGR'], f'sl.{k}.main': s['main_CAGR'], f'sl.{k}.EW': s['EW_CAGR'],
              f'sl.{k}.dd': s['sleeve_maxDD_daily'], f'sl.{k}.t50': s['t_vs_0050'], f'sl.{k}.mex': s['mean_monthly_ex_0050']})
R.update({'sl.s2.tmain': sl['2019_2026-10-12']['t_vs_main'], 'sl.s2.stress': sl['2019_2026-10-12']['stress_CAGR'],
          'sl.slots': sl['avg_slots_used'], 'sl.days': sl['share_days_holding'], 'sl.corr_main': sl['corr']['sleeve_main'],
          'sl.corr_disp': sl['corr']['sleeve_disposal'], 'sl.corr_ex': sl['corr']['excess_sleeve_vs_main_excess']})
A = sl['alloc']
R.update({'al.721.cagr': A['2021-10_2026-10|70/20/10']['CAGR_days'], 'al.721.t': A['2021-10_2026-10|70/20/10']['t_vs_0050'], 'al.721.dd': A['2021-10_2026-10|70/20/10']['maxDD_monthly'],
          'al.82.cagr': A['2021-10_2026-10|80/20']['CAGR_days'], 'al.82.t': A['2021-10_2026-10|80/20']['t_vs_0050'], 'al.82.dd': A['2021-10_2026-10|80/20']['maxDD_monthly'],
          'al.91.cagr': A['2010-12_2026-10|90/10']['CAGR_days'], 'al.91.t': A['2010-12_2026-10|90/10']['t_vs_0050'],
          'al.100.cagr': A['2010-12_2026-10|100']['CAGR_days'], 'al.100.t': A['2010-12_2026-10|100']['t_vs_0050']})
P = oo['P1_P3']
R.update({'p1.sel.mean': P['select_all_del']['mean'], 'p1.sel.t': P['select_all_del']['t_month'], 'p1.sel.median': P['select_all_del']['median'],
          'p1.oos.mean': P['oos_all_del']['mean'], 'p1.oos.median': P['oos_all_del']['median'],
          'p1.sc.mean': P['oos_by_class']['MSCI_SC_DEL']['mean'], 'p1.sc.t': P['oos_by_class']['MSCI_SC_DEL']['t_month'],
          'p1.std.mean': P['oos_by_class']['MSCI_STD_DEL']['mean'], 'p1.std.t': P['oos_by_class']['MSCI_STD_DEL']['t_month'],
          'p1.tw50.mean': P['oos_by_class']['TW50_DEL']['mean'], 'p1.hd.mean': P['oos_by_class_HD_pooled']['mean'],
          'p3.low.mean': P['oos_lowADV_tertile']['mean'], 'p3.low.t': P['oos_lowADV_tertile']['t_month'], 'p3.high.mean': P['oos_highADV_tertile']['mean']})
R.update({'ph.std_del.abE': ph['oos']['MSCI_STD_DEL']['abE'], 'ph.sc_del.abE': ph['oos']['MSCI_SC_DEL']['abE'],
          'ph.std_del.vr': ph['oos']['MSCI_STD_DEL']['vr_median'], 'ph.sc_del.vr': ph['oos']['MSCI_SC_DEL']['vr_median'],
          'ph.del_pre10_msci': ph['oos']['msci_del']['pre10_sum'], 'ph.std_add_pre10': ph['oos']['MSCI_STD_ADD']['pre10_sum'],
          'ph.adv_sel': ph['select']['all_del']['adv20_median_e8'], 'ph.adv_oos': ph['oos']['all_del']['adv20_median_e8']})
R.update({'p4.n_all': p4['n_all'], 'p4.n_main': p4['n_ex_src3'], 'p4.mean_cl': p4['main']['mean_month_cluster'], 'p4.t': p4['main']['t_month'],
          'p4.mean_evt': p4['main']['mean_events'], 'p4.median': p4['main']['median_events'],
          'p4.seg1': p4['seg_2006_2015']['mean_month_cluster'], 'p4.seg1_m': p4['seg_2006_2015']['n_months'],
          'p4.seg2': p4['seg_2016_2026']['mean_month_cluster'], 'p4.seg2_t': p4['seg_2016_2026']['t_month'],
          'p4.std': p4['by_class']['MSCI_STD_ADD']['mean_month_cluster'], 'p4.std_t': p4['by_class']['MSCI_STD_ADD']['t_month'],
          'p4.sc': p4['by_class']['MSCI_SC_ADD']['mean_month_cluster'], 'p4.sc_t': p4['by_class']['MSCI_SC_ADD']['t_month'],
          'p4.tw50': p4['by_class']['TW50_ADD']['mean_month_cluster'],
          'p4.base_cagr': p4['nav']['2006-04-11']['base'][0], 'p4.mod_cagr': p4['nav']['2006-04-11']['sell_on_add_ex_src3'][0],
          'p4.base_dd': p4['nav']['2006-04-11']['base'][1], 'p4.mod_dd': p4['nav']['2006-04-11']['sell_on_add_ex_src3'][1],
          'p4.delta_pp': round(p4['nav']['2006-04-11']['sell_on_add_ex_src3'][0] - p4['nav']['2006-04-11']['base'][0], 2), 'p4.adopt': p4['adopt']})
L.dump(R, 'results_replica.json'); print(len(R), 'keys')
