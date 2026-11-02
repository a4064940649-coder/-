#!/usr/bin/env python3
"""s7（PREREG 之後的事後比對；2026-11-02 補寫，不在 PREREG 雜湊清單內——已揭露）：
把研究員結果與 replicate/ 的獨立複製逐項比對，依事前登記的容許值判定；超過容許值一律採不利方向的值（o6 教訓 22），
淨值複製與統計實作差異分開寫。最後套用事前登記的判定規則（frozen_spec 第五節），寫入 results.json 的 VERDICT_posthoc。
只讀既有結果，不重算任何淨值。"""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
import v7common as C
RES = C.check_prereg()
RP = C.HERE / 'replicate'
rc = json.loads((RP / 'rep_compare.json').read_text(encoding='utf-8'))
rn = json.loads((RP / 'rep_nav.json').read_text(encoding='utf-8'))
rs = json.loads((RP / 'rep_stats.json').read_text(encoding='utf-8'))
rs2 = json.loads((RP / 'rep_stats2.json').read_text(encoding='utf-8'))
rx = json.loads((RP / 'rep_exec.json').read_text(encoding='utf-8'))
RH = RES['RESULTS_HOLD_SPA']; RE = RES['RESULTS_EXEC']; RR = RES['RESULTS_RUN']
out = {'note': '事後比對（PREREG 後新增腳本，只讀結果）；容許值與「超過取不利方向」均為事前登記'}

# (1) 面板與上市矩陣逐格
a = pd.read_parquet(C.DATA / 'low.parquet'); b = pd.read_parquet(RP / 'data' / 'low.parquet')
w = (a.index >= C.W_A) & (a.index <= C.W_B)
A, B = a[w], b[w].reindex_like(a[w])
anyc = A.notna() | B.notna(); eq = A.notna() & B.notna() & np.isclose(A.fillna(0), B.fillna(0), rtol=1e-9, atol=1e-9)
low_win = float(eq.to_numpy().sum() / anyc.to_numpy().sum())
cells = {k: rc[k]['agree'] for k in ('adj', 'open', 'close_raw', 'high', 'value', 'ret', 'flag')}
cells['low_window(V7 只重建窗內；窗外沿用 V6、不在本題重建範圍)'] = low_win
cells['low_all(參考)'] = rc['low']['agree']
cells['listing_ALL'] = rc['listing_ALL']['agree']; cells['listing_2019-2026'] = rc['listing_2019-2026']['agree']; cells['listing_window'] = rc['listing_window']['agree']
pass_cells = all(v >= 0.999 for k, v in cells.items() if '參考' not in k)
out['cells'] = {'agree': cells, 'threshold': 0.999, 'pass': pass_cells}
# (2) 淨值
navd = {k: v['max_rel_diff'] for k, v in rn.items() if isinstance(v, dict)}
out['nav'] = {'max_rel_diff': navd, 'threshold': 1e-8, 'pass': max(navd.values()) <= 1e-8, 'signal_days_identical': rn.get('signal_days_identical')}
# (3) SPA（統計實作差異：同族群 F_long 2,319 條＝rep_stats2；族群差異：複製員自建族群 2,387 條＝rep_stats）
spa = {}
for foc, rk in (('AB', 'AB_V7'), ('B10', 'AB_V7_B10')):
    for L in (3, 6, 12):
        mine = RH['spa'][f'{rk}|L{L}']['p_SPA_c_focus']; same = rs2[f'{rk}|normal|L{L}']['p_SPA_c_focus']; own = rs[f'{rk}|normal|L{L}']['p_SPA_c_focus']
        d = abs(mine - same)
        spa[f'{foc}|L{L}'] = {'researcher': mine, 'replicator_same_family': same, 'replicator_own_family': own, 'diff_same_family': d,
                             'pass(≤0.02)': d <= 0.02, 'adopted(不利＝較大 p)': max(mine, same) if d > 0.02 else mine,
                             't_researcher': RH['spa'][f'{rk}|L{L}']['focus_t'], 't_replicator': rs2[f'{rk}|normal|L{L}']['t_main']}
out['spa'] = spa
# (4) 保留期 p_lower（L3/6/12 × 三版）
hold = {}
for foc, rk in (('AB', 'AB_V7'), ('B10', 'AB_V7_B10')):
    H = RH['hold'][rk]['x50_calendar_v6grid']; R = rs[f'hold|{rk}']
    for L in (3, 6, 12):
        for ver, km, kr in (('raw', 'p_lower_cumlog', 'p_lower_L'), ('shrink50', 'p_lower_cumlog_shrink50', 'p_lower_shrink50_L'), ('zero', 'p_lower_cumlog_zero', 'p_lower_zero_L')):
            m = H[f'L{L}'][km]; r = R[f'{kr}{L}']; d = abs(m - r)
            hold[f'{foc}|L{L}|{ver}'] = {'researcher': m, 'replicator': r, 'diff': d, 'pass(≤0.02)': d <= 0.02,
                                         'adopted(對策略不利＝較小 p)': min(m, r) if d > 0.02 else m}
    hold[f'{foc}|cumlog'] = {'researcher': RH['hold'][rk]['x50_calendar_v6grid']['actual']['cumlog'], 'replicator': R['cumlog']}
out['hold'] = hold
# (5) 可執行表（B10、一般成本、j＝0～12 中位數年化）
ex = {}
for k, v in rx.items():
    model, tier, ver = k.split('|')
    mine = RE['jsumm'].get(f'B10|{model}|normal|{tier}|{ver}')
    if mine is None: continue
    nj = len(v['by_j'])
    m = mine['CAGR_median']; r = v['CAGR_median']; d = abs(m - r)
    byj = {jj: abs(mine['CAGR_by_j'][int(jj[1:])] - x['CAGR']) for jj, x in v['by_j'].items()}
    ex[k] = {'researcher_j_median': m, 'replicator_j_median': r, 'diff_pp': d, 'n_j_replicated': nj, 'max_diff_by_j_pp': max(byj.values()),
             'pass(≤0.2pp)': d <= 0.2, 'adopted(不利＝較低年化)': min(m, r) if d > 0.2 else m}
out['exec'] = ex
out['exec_complete'] = len(ex) == 16 and all(x['n_j_replicated'] == 13 for x in ex.values())

# (6) 事前登記判定
G = RES['GATE']
gate_pass = bool(G['PASS'])
data_rep_pass = pass_cells and out['nav']['pass']
cb = RES['SUMMARY_posthoc']['common_base_CAGR'] if 'SUMMARY_posthoc' in RES else None
d_all = RR['paired_V7_minus_V4']['AB_V7|normal − V4']['CAGR_diff_ALL']
spa_L6 = {f: spa[f'{f}|L6']['adopted(不利＝較大 p)'] for f in ('AB', 'B10')}
dec = {'gate_pass': gate_pass, 'data_replication_pass(逐格＋淨值)': data_rep_pass,
       'stat_replication_all_within_tol': all(x['pass(≤0.02)'] for x in spa.values()) and all(x['pass(≤0.02)'] for k, x in hold.items() if 'cumlog' not in k),
       'exec_replication_all_within_tol': all(x['pass(≤0.2pp)'] for x in ex.values()) if ex else None,
       'dCAGR_ALL_AB_V7_minus_V4_pp': d_all,
       'V7_replaces_V4': gate_pass and data_rep_pass,
       'SOP_numbers': ('直接換成 V7' if abs(d_all) <= 1 else '先拆解到逐月逐檔') if (gate_pass and data_rep_pass) else 'V4 照舊',
       'SPA_focus_L6_adopted': spa_L6 | {'stress': RH['spa']['AB_V7_stress|L6']['p_SPA_c_focus']},
       'grade': '可提議回第 2 級' if (spa_L6['AB'] <= 0.10 and spa_L6['B10'] <= 0.10) else '維持第 3 級'}
# 實盤預期表：主欄 B10 j 中位數（一般成本），兩模型；複製超差取不利值
tbl = {}
for key, row in RE['live_expect_table'].items():
    tier, ver = key.split('|')
    u = ex.get(f'K5T|{tier}|{ver}'); l = ex.get(f'TICK|{tier}|{ver}')
    cu = u['adopted(不利＝較低年化)'] if u else row['CAGR_K5T(上界成本)']; cl = l['adopted(不利＝較低年化)'] if l else row['CAGR_TICK(下界成本)']
    e50 = row['0050_CAGR']
    wu, wl = cu - e50 > 0, cl - e50 > 0
    tbl[key] = {'CAGR_K5T': cu, 'CAGR_TICK': cl, '0050': e50, 'excess_K5T_pp': cu - e50, 'excess_TICK_pp': cl - e50,
                'p10_K5T': row['CAGR_p10_K5T'], 'p10_TICK': row['CAGR_p10_TICK'], 'stress_K5T': row['stress_CAGR_K5T'], 'stress_TICK': row['stress_CAGR_TICK'],
                'beat_0050': '贏' if (wu and wl) else ('輸' if (not wu and not wl) else '無法判斷')}
dec['live_expect_table_adopted'] = tbl
out['decision'] = dec
C.save_key('VERDICT_posthoc', out)
print(json.dumps(C.conv(out), ensure_ascii=False, indent=1, default=str)[:15000])
