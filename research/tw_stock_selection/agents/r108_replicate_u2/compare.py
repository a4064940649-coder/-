#!/usr/bin/env python3
"""R108：彙整我的數字（step0.json、valid.json、cont2008.json）→ my_numbers.json；算完後才與 S102 results.json／results_2008.json 比對 → compare.json。
容忍：年化（差）≤ 0.3 個百分點；判定（V1～V4、R2、總判定）必須相同。用法：python3 -I -B compare.py"""
import json
from pathlib import Path
HERE = Path(__file__).resolve().parent
ORIG = HERE.parent / 's102_u1_u2'
s0 = json.loads((HERE / 'step0.json').read_text()); v = json.loads((HERE / 'valid.json').read_text()); c8 = json.loads((HERE / 'cont2008.json').read_text())
keys = list(v['paths']['CUR'])
PP = {k: (v['paths']['FZ'][k]['valid'] - v['paths']['CUR'][k]['valid']) * 100 for k in keys}
pv = v['pairs']; fz = v['main']['FZ']; rbl = fz['rebuys']
my = dict(
    baseline=dict(full=s0['std']['full'], since2019=s0['std']['since2019'], n_closed=s0['std']['n_closed'], paths_median=s0['paths_median'],
                  nav_max_abs_diff_vs_navs_official=s0['std']['nav_max_abs_diff'], engine_identical_20paths=s0['A1_engine_identical_20paths']),
    valid=dict(diff_median_pp=pv['valid']['median'], npos=pv['valid']['npos'], min_pp=pv['valid']['min'], max_pp=pv['valid']['max'], per_path_pp=PP,
               y2022_2023_median_pp=pv['y2022_2023']['median'], y2022_2023_npos=pv['y2022_2023']['npos'], since2024_median_pp=pv['since2024']['median'], since2024_npos=pv['since2024']['npos'],
               V3_blk_median_pp=pv['blk']['median'], V3_blk_npos=pv['blk']['npos'], seg_2022_median_pp=pv['seg']['2022']['median'], seg_2024_07_2025_04_median_pp=pv['seg']['2024-07~2025-04']['median'],
               R2_vs_scaled_median_pp=pv['R2_vs_scaled']['median'], R2_vs_scaled_npos=pv['R2_vs_scaled']['npos'], exposure_full_diff_pp=v['exposure_full_std']['diff_pp'],
               main_CUR=v['main']['CUR']['cagr']['valid'], main_FZ=fz['cagr']['valid'], rb_flag=fz['rb_flag'], rb_trig=fz['rb_trig'], rb_buy=fz['rb_buy'],
               rebuys=dict(n=rbl['n'], win=rbl['win'], mean_ret=rbl['mean_ret'], median_ret=rbl['median_ret'], mean_excess=rbl['mean_excess'], pnl_sum=rbl['pnl_sum'], list=rbl['list']),
               rb_buy_per_path=v['rb_buy_per_path'], verdict=v['verdict']),
    cont2008_V_BRIDGE_std=dict(CUR_full=c8['V_BRIDGE']['CUR']['full'], FZ_full=c8['V_BRIDGE']['FZ']['full'], diff_pp=c8['V_BRIDGE']['diff_pp'], rebuys=c8['V_BRIDGE']['rebuys'],
                               baseline=c8['V_BRIDGE']['baseline']),
    cont2008_V_MASK_std=dict(CUR_full=c8['V_MASK']['CUR']['full'], FZ_full=c8['V_MASK']['FZ']['full']))
(HERE / 'my_numbers.json').write_text(json.dumps(my, ensure_ascii=False, indent=1))

# ---- 以下才讀原作者結果
R = json.loads((ORIG / 'results.json').read_text()); R8 = json.loads((ORIG / 'results_2008.json').read_text())
o = R['topics']['U2']; op = o['pairs']; P = R['paths']
TOL = 0.3
rows = []


def add(name, mine, orig, unit='pp', tol=TOL):
    if unit == 'rate':
        d = (mine - orig) * 100
    elif unit in ('pp',):
        d = mine - orig
    elif unit == 'count':
        d = mine - orig
    else:
        d = None
    ok = abs(d) <= tol if unit in ('rate', 'pp') else (mine == orig)
    rows.append(dict(item=name, mine=mine, orig=orig, diff=d, diff_unit='個百分點' if unit in ('rate', 'pp') else unit, within_tol=bool(ok)))


add('基準 2015-07 起全期年化', my['baseline']['full'], R['cont_std_2015']['CUR']['full'], 'rate')
add('基準 2019 起年化', my['baseline']['since2019'], R['cont_std_2015']['CUR']['since2019'], 'rate')
add('驗證期年化差 中位數', pv['valid']['median'], op['valid']['median'])
add('驗證期年化差 >0 路徑數', pv['valid']['npos'], op['valid']['npos'], 'count')
add('驗證期年化差 最小', pv['valid']['min'], op['valid']['min'])
add('驗證期年化差 最大', pv['valid']['max'], op['valid']['max'])
for k in keys:
    od = (P['FZ_U2|fresh'][k]['valid'] - P['CUR|fresh'][k]['valid']) * 100
    add(f'驗證期年化差 路徑 {k}', PP[k], od)
add('2022–23 年化差 中位數', pv['y2022_2023']['median'], op['y2022_2023']['median'])
add('2024 起 年化差 中位數', pv['since2024']['median'], op['since2024']['median'])
add('V3 禁買聯集後 中位數', pv['blk']['median'], op['blk']['median'])
add('V4 2022 段回撤差 中位數', pv['seg']['2022']['median'], op['seg']['2022']['median'])
add('V4 2024-07～2025-04 段回撤差 中位數', pv['seg']['2024-07~2025-04']['median'], op['seg']['2024-07~2025-04']['median'])
add('R2 對縮放現行 中位數', pv['R2_vs_scaled']['median'], op['R2_vs_scaled']['median'])
add('全期曝險差（R2 觸發依據）', v['exposure_full_std']['diff_pp'], o['exposure_full_period_std']['diff_pp'])
add('主跑 現行 驗證期年化', v['main']['CUR']['cagr']['valid'], o['main']['CUR']['cagr']['valid'], 'rate')
add('主跑 U2 驗證期年化', fz['cagr']['valid'], o['main']['FZ']['cagr']['valid'], 'rate')
ox = o['main']['FZ']['extra_trades']
add('主跑 買回筆數', rbl['n'], ox['n'], 'count')
add('主跑 買回平均報酬', rbl['mean_ret'], ox['mean_ret'], 'rate')
add('主跑 買回對 0050 平均超額', rbl['mean_excess'], ox['mean_ex_0050'], 'rate')
mine_list = [(x['code'], x['ent'], x['exit'], round(x['ret'], 8)) for x in rbl['list']]
orig_list = [(x['code'], x['ent'], x['exit'], round(x['ret'], 8)) for x in ox['list']]
add('主跑 買回逐筆（代號、進出日、報酬）完全相同', mine_list == orig_list, True, 'bool')
for vk in ('V1', 'V2', 'V3', 'V4', 'R2_triggered', 'R2'):
    add(f'判定 {vk}', v['verdict'][vk], o['verdict'][vk], 'bool')
add('判定 overall', v['verdict']['overall'], o['verdict']['overall'], 'str')
ob = R8['versions']['V_BRIDGE']['std']
add('2008 v3 V_BRIDGE 標準路徑 現行全期年化', c8['V_BRIDGE']['CUR']['full'], ob['CUR']['full'], 'rate')
add('2008 v3 V_BRIDGE 標準路徑 U2 全期年化', c8['V_BRIDGE']['FZ']['full'], ob['FZ_U2']['full'], 'rate')
add('2008 v3 V_BRIDGE 標準路徑 U2 2019 起年化', c8['V_BRIDGE']['FZ']['since2019'], ob['FZ_U2']['since2019'], 'rate')
add('2008 v3 V_BRIDGE 標準路徑 U2 交易筆數', c8['V_BRIDGE']['n_all']['FZ'], ob['FZ_U2']['n_all'], 'count')
om = R8['versions']['V_MASK']['std']
add('2008 v3 V_MASK 標準路徑 U2 全期年化', c8['V_MASK']['FZ']['full'], om['FZ_U2']['full'], 'rate')
maxabs = max(abs(r['diff']) for r in rows if r['diff_unit'] == '個百分點')
out = dict(tolerance='年化與年化差 ≤ 0.3 個百分點；筆數、逐筆、判定須完全相同', n_items=len(rows), n_within=int(sum(r['within_tol'] for r in rows)),
           max_abs_diff_pp=maxabs, all_same=bool(all(r['within_tol'] for r in rows)), rows=rows)
(HERE / 'compare.json').write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str))
print(out['n_items'], out['n_within'], out['max_abs_diff_pp'], out['all_same'])
for r in rows:
    if not r['within_tol'] or 'diff' in r and r['diff_unit'] == '個百分點' and abs(r['diff']) > 1e-9:
        print(r)
