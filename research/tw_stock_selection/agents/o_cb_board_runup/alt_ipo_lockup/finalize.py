# -*- coding: utf-8 -*-
"""彙整：判定逐項、t 拆解，寫回 results.json（保留 PREREG），並寫上層 o_cb_board_runup/results.json。"""
import os, sys, json, hashlib
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pathlib import Path
HERE = Path(__file__).resolve().parent
R = json.loads((HERE / 'results.json').read_text(encoding='utf-8'))
assert hashlib.sha256(R['PREREG'].encode('utf-8')).hexdigest() == R['PREREG_sha256']
OUT = {m: json.loads((HERE / f'oos_{m}.json').read_text(encoding='utf-8')) for m in ('v2', 'v3')}
EX = json.loads((HERE / 'explore.json').read_text(encoding='utf-8'))
IT = json.loads((HERE / 'iterations.json').read_text(encoding='utf-8'))

def judge(o):
    a = o['runs']['A5|normal']; p = o['runs']['PLACEBO_transfer|normal']; s = o['runs']['A5|stress']; b = o['runs']['base|normal']
    c = {
        '1_sel_paired_gt0': (a['SEL']['paired']['mean_pp'], a['SEL']['paired']['mean_pp'] > 0),
        '1_val_paired_gt0': (a['VAL']['paired']['mean_pp'], a['VAL']['paired']['mean_pp'] > 0),
        '2_all_cagr_ge_base': ((a['ALL']['CAGR'], b['ALL']['CAGR']), a['ALL']['CAGR'] >= b['ALL']['CAGR']),
        '3_val_ex_top3_gt0': (o['robust']['VAL']['paired_ex_top3']['mean_pp'], o['robust']['VAL']['paired_ex_top3']['mean_pp'] > 0),
        '4_placebo_lt_half': ((p['VAL']['paired']['mean_pp'], a['VAL']['paired']['mean_pp']), abs(p['VAL']['paired']['mean_pp']) < 0.5 * abs(a['VAL']['paired']['mean_pp']) and a['VAL']['paired']['mean_pp'] > 0),
        '5_stress_val_gt0': (s['VAL']['paired']['mean_pp'], s['VAL']['paired']['mean_pp'] > 0),
    }
    return dict(checks=c, all_pass=all(v[1] for v in c.values()))
J = {m: judge(o) for m, o in OUT.items()}
decomp = {}
for m, o in OUT.items():
    for sg in ('SEL', 'VAL', 'HOLD', 'ALL'):
        b = o['runs']['base|normal'][sg]; a = o['runs']['A5|normal'][sg]
        decomp[f'{m}|{sg}'] = dict(base=dict(mean=b['ex0050_mean_pp'], sd=b['ex0050_sd_pp'], t=b['t_0050']), A5=dict(mean=a['ex0050_mean_pp'], sd=a['ex0050_sd_pp'], t=a['t_0050']))
R['GATE_main_topic'] = json.loads((HERE.parent / 'gate.json').read_text(encoding='utf-8'))
R['explore_key'] = {k: EX[k] for k in ('sample', 'P1_resid', 'P1_fm', 'P1_matched_decile', 'P2', 'P2_oldrule_2to5y', 'P3', 'age_cells_resid_vs_old5', 'P1_resid_by_era', 'P4_picks')}
R['iterations'] = IT['versions']; R['n_versions'] = IT['n_versions']
R['OOS'] = {m: dict(runs={k: {sg: {x: v[x] for x in ('CAGR', 'maxDD', 't_0050', 't_EW', 'E50_CAGR', 'EW_CAGR', 'ex0050_mean_pp', 'ex0050_sd_pp') if x in v} | ({'paired': v['paired']} if 'paired' in v else {})
                                    for sg, v in r.items()} for k, r in o['runs'].items()},
                    robust=o['robust'], paired_by_year=o['paired_by_year_normal'], young_share=o['young_share_picks'], patched_cells=o['patched_cells'])
            for m, o in OUT.items()}
R['t_decomp'] = decomp
R['JUDGE'] = J
R['VERDICT'] = '不可用' if not (J['v2']['all_pass'] and J['v3']['all_pass']) else '待觀察'
(HERE / 'results.json').write_text(json.dumps(R, ensure_ascii=False, indent=1, default=str), encoding='utf-8')
top = dict(topic='CB 發行前推升（董事會錨點）', gate=R['GATE_main_topic'], build_summary=json.loads((HERE.parent / 'build_summary.json').read_text(encoding='utf-8')),
           alt_topic='alt_ipo_lockup：新上市股的內部人鎖定期供給', alt_verdict=R['VERDICT'], alt_results='alt_ipo_lockup/results.json')
(HERE.parent / 'results.json').write_text(json.dumps(top, ensure_ascii=False, indent=1, default=str), encoding='utf-8')
print(json.dumps(J, ensure_ascii=False, indent=1)); print(json.dumps(decomp, ensure_ascii=False)); print(R['VERDICT'])
