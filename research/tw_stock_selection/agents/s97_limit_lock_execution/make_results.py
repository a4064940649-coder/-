#!/usr/bin/env python3
"""S97：把 out_official.json、out_cont.json、spot_check.csv 彙整成 results.json（只整理、不重算回測）。"""
import json, hashlib
from pathlib import Path
import pandas as pd
H = Path(__file__).resolve().parent
o = json.loads((H / 'out_official.json').read_text()); c = json.loads((H / 'out_cont.json').read_text())
pp = lambda x: round(x * 100, 4)
SC = ['BASE', 'NONE', 'A_i', 'A_ii', 'A_iii', 'B_i', 'B_ii', 'B_iii']
def row_o(s):
    m = o['scen'][s]; p = o['paths_summary'][s]
    return dict(cagr_since2019_pct=pp(m['cagr_since2019']), cagr_full_pct=pp(m['cagr_full']), maxdd_pct=pp(m['maxdd']), n_closed=m['n_closed'], n_all=m['n_all'],
                ex0050_since2019_pp=round(m['ex_since2019_pp'], 4), ex0050_full_pp=round(m['ex_full_pp'], 4),
                BASE_minus_scen_since2019_pp=round(-(m.get('d_cagr_since2019_vs_BASE_pp', 0.0)), 4), BASE_minus_scen_full_pp=round(-(m.get('d_cagr_full_vs_BASE_pp', 0.0)), 4),
                blk_limup=m['blk_limup'], sell_defer_limdn=m['sell_defer_limdn'], events=m.get('events'),
                paths20=dict(cagr19_median_pct=pp(p['cagr19_median']), cagr19_min_pct=pp(p['cagr19_min']), cagr19_max_pct=pp(p['cagr19_max']), cagr_full_median_pct=pp(p['cagr_full_median']),
                             maxdd_median_pct=pp(p['maxdd_median']), maxdd_worst_pct=pp(p['maxdd_worst']),
                             BASE_minus_scen_median_of_medians_pp=round((o['paths_summary']['BASE']['cagr19_median'] - p['cagr19_median']) * 100, 4),
                             scen_minus_BASE_paired_pp=p['d19_vs_BASE_pp'], scen_minus_BASE_paired_full_pp=p['dfull_vs_BASE_pp'],
                             blk_limup_total_20paths=p['blk_limup_total'], sell_defer_total_20paths=p['sell_defer_total']))
def row_c(s):
    m = c['scen'][s]
    return dict(cagr_full_pct=pp(m['cagr_full']), cagr_2008_2015H1_pct=pp(m['cagr_2008_2015H1']), cagr_since2015_07_pct=pp(m['cagr_since2015_07']), cagr_since2019_pct=pp(m['cagr_since2019']),
                maxdd_pct=pp(m['maxdd']), dd_trough=m['dd_trough'], final_nav=round(m['final_nav']), n_closed=m['n_closed'], n_all=m['n_all'], ex0050_full_pp=round(m['ex_full_pp'], 4),
                BASE_minus_scen_full_pp=round(-(m.get('d_cagr_full_vs_BASE_pp', 0.0)), 4), blk_limup=m['blk_limup'], sell_defer_limdn=m['sell_defer_limdn'], events=m.get('events'))
strip = lambda d: {k: ({a: b for a, b in v.items() if a != 'rows'} if isinstance(v, dict) else v) for k, v in d.items()}
sp = pd.read_csv(H / 'spot_check.csv', dtype={'code': str})
res = dict(
 id='S97', prereg_sha256=(H / 'PREREG.sha256').read_text().split()[0], prereg_sha256_recomputed=hashlib.sha256((H / 'PREREG.json').read_bytes()).hexdigest(),
 重要更正='正式引擎（s10lib.simulate_s10）本來就不假設開盤一定成交：開盤≥漲停價不買不遞補、開盤≤跌停價順延。現行基準＝口徑 A＋處理 (i)。NONE 才是「一定成交」的對照。',
 acceptance=dict(official=o['acceptance_base'], official_plug_off=o['acceptance_plug_off'], official_A_i_equals_BASE=o['scen']['A_i']['nav_max_abs_diff_vs_BASE'] == 0,
                 cont=c['acceptance_base'], cont_plug_off=c['acceptance_plug_off'], B_iii_equals_B_i=dict(official=o['check_B_iii_equals_B_i'], cont=c['check_B_iii_equals_B_i'])),
 official=dict(window=[o['start'], o['end']], e50=dict(cagr_since2019_pct=pp(o['scen']['BASE']['e50_cagr_since2019']), cagr_full_pct=pp(o['scen']['BASE']['e50_cagr_full'])), scen={s: row_o(s) for s in SC}),
 cont=dict(window=[c['start'], c['end']], 註='BASE 在 2015-06-01 前用 10% 判漲跌停（S75 原樣）；情境一律用當時的 7%。只跑標準路徑。',
           e50=dict(cagr_full_pct=pp(c['scen']['BASE']['e50_cagr_full']), cagr_2008_2015H1_pct=pp(c['scen']['BASE']['e50_2008_2015H1'])), scen={s: row_c(s) for s in SC}),
 desc=dict(official=dict(D1=strip(o['desc']['D1']), D1_rows=o['desc']['D1']['2015-07～2026-10']['rows'], **{k: v for k, v in o['desc'].items() if k != 'D1'}),
           cont=dict(D1=strip(c['desc']['D1']), D1_rows=c['desc']['D1']['2008-01～2026-10']['rows'], **{k: v for k, v in c['desc'].items() if k != 'D1'})),
 spot_check=dict(n=int(len(sp)), population=len(c['desc']['D1']['2008-01～2026-10']['rows']), all_match=bool(sp['match'].all()), ref_equals_prev_close_all=bool(sp['ref_equals_prev_close'].all())),
 resources=dict(official=dict(peak_mb=o['peak_mb'], secs=o['secs']), cont=dict(peak_mb=c['peak_mb'], secs=c['secs'])))
(H / 'results.json').write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str))
for s in SC:
    r = res['official']['scen'][s]; p = r['paths20']
    print('正式', s, r['cagr_since2019_pct'], r['cagr_full_pct'], r['maxdd_pct'], r['n_closed'], r['ex0050_since2019_pp'], r['ex0050_full_pp'], '|', r['BASE_minus_scen_since2019_pp'], r['BASE_minus_scen_full_pp'], '| 20路徑', p['cagr19_median_pct'], p['cagr19_min_pct'], p['cagr19_max_pct'], p['BASE_minus_scen_median_of_medians_pp'], {k: round(v, 3) for k, v in p['scen_minus_BASE_paired_pp'].items()}, p['maxdd_median_pct'], p['maxdd_worst_pct'], p['blk_limup_total_20paths'], p['sell_defer_total_20paths'], r['events'])
for s in SC:
    r = res['cont']['scen'][s]
    print('連續', s, r['cagr_full_pct'], r['cagr_2008_2015H1_pct'], r['cagr_since2015_07_pct'], r['maxdd_pct'], r['dd_trough'], r['final_nav'], r['n_closed'], r['n_all'], r['ex0050_full_pp'], '|', r['BASE_minus_scen_full_pp'], r['blk_limup'], r['events'])
print(res['official']['e50'], res['cont']['e50'], res['prereg_sha256'] == res['prereg_sha256_recomputed'])
