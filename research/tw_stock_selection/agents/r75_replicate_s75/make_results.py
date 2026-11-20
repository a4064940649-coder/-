#!/usr/bin/env python3
"""R75 最後一步：彙整 results.json、compare.csv（宣稱值取自 S75 的 FINDINGS.md／任務書；最後才讀 S75 的 nav_2008_2026.csv.gz、yearly.csv 逐日／逐年對照）。"""
import json
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent
TMP = Path('/tmp/claude-0/-home-user/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/r75')
S75 = HERE.parent / 's75_continuous_2008_2026'
J = lambda f: json.loads((TMP / f).read_text())
s1, run, s3, aa, ab = J('step1.json'), J('run_cache_full.json'), J('step3.json'), J('audit_a.json'), J('audit_bcd.json')
pan = {k: J(f'panel_{k}.json') for k in ('cache_full', 'cache_seam07', 'cache_twse')}
B, M, E = run['V_BRIDGE'], run['V_MASK'], run['bench']
rows = []
def add(item, claim, mine, tol, unit, kind='abs'):
    diff = (mine / claim - 1) * 100 if kind == 'rel' else mine - claim
    rows.append(dict(item=item, claimed=claim, r75=round(mine, 4), diff=round(diff, 4), tolerance=tol, unit=unit, ok=bool(abs(diff) <= tol + 1e-9)))
add('正式引擎 2019 起年化 %', 48.76, s1['since2019'] * 100, 0.3, '百分點'); add('正式引擎 已平倉筆數', 381, s1['n_closed'], 0, '筆')
add('V_BRIDGE 期末淨值（萬）', 5308, B['final'] / 1e4, 3, '%', 'rel'); add('V_BRIDGE 年化 %', 23.52, B['cagr'] * 100, 0.3, '百分點'); add('V_BRIDGE 最大回撤 %', -49.21, B['maxdd'] * 100, 0.5, '百分點')
add('V_MASK 期末淨值（萬）', 3883, M['final'] / 1e4, 3, '%', 'rel'); add('V_MASK 年化 %', 21.48, M['cagr'] * 100, 0.3, '百分點')
add('0050 期末淨值（萬）', 1291, E['final'] / 1e4, 3, '%', 'rel'); add('0050 年化 %', 14.57, E['cagr'] * 100, 0.3, '百分點'); add('0050 最大回撤 %', -52.38, E['maxdd'] * 100, 0.5, '百分點')
CY = {2008: (-36.60, -42.04), 2009: (85.11, 75.00), 2010: (18.73, 13.13), 2011: (-17.93, -15.81), 2012: (0.32, 11.92), 2013: (19.78, 11.68), 2014: (14.27, 16.66), 2015: (-2.77, -6.28),
      2016: (17.50, 19.65), 2017: (23.23, 18.14), 2018: (4.55, -4.94), 2019: (30.26, 33.52), 2020: (106.60, 31.14), 2021: (45.50, 21.92), 2022: (-17.40, -21.37), 2023: (56.67, 27.52),
      2024: (64.40, 48.64), 2025: (56.84, 36.83), 2026: (75.79, 67.19)}
for y, (a, b) in CY.items():
    add(f'{y} V_BRIDGE 年報酬 %', a, B['yearly'][str(y)] * 100, 1.0, '百分點'); add(f'{y} 0050 年報酬 %', b, E['yearly'][str(y)] * 100, 1.0, '百分點')
add('2013 V_MASK 年報酬 %', 9.08, M['yearly']['2013'] * 100, 1.0, '百分點')
for k, (ex, t, cs, cb) in {'2008~2012': (1.34, 0.51, 2.79, 1.58), '2013~2015': (2.91, 1.55, 9.97, 6.87), '2016~2018': (4.71, 0.80, 14.86, 10.38), '2019~2023': (19.07, 2.55, 38.32, 16.43),
                           '2024~2026': (12.42, 1.63, 71.08, 54.48), 'OOS_2008-01~2015-06': (2.10, 1.08, 7.32, 5.20), 'IS_2015-07~': (12.74, 2.91, 35.57, 21.23)}.items():
    p = B['periods'][k]
    add(f'{k} 年化超額', ex, p['ex_ann_pp'], 0.3, '百分點'); add(f'{k} NW t', t, p['t_nw6'], 0.05, 't'); add(f'{k} V_BRIDGE 年化 %', cs, p['cagr_cal'] * 100, 0.3, '百分點'); add(f'{k} 0050 年化 %', cb, p['cagr_cal_0050'] * 100, 0.3, '百分點')
add('全期月超額年化', 8.50, B['periods']['ALL']['ex_ann_pp'], 0.3, '百分點'); add('全期 NW t', 2.93, B['periods']['ALL']['t_nw6'], 0.05, 't')
add('已平倉筆數', 624, B['trades']['n_closed'], 0, '筆'); add('勝率 %', 48.1, B['trades']['win_rate'] * 100, 0.1, '百分點')
add('連續帳戶 2015-07-01 重新基準化後 2019 起年化 %', 49.32, B['rebased']['since2019'] * 100, 0.3, '百分點')
add('20 條路徑年化中位數 %', 22.83, s3['paths_summary']['median'] * 100, 0.3, '百分點'); add('20 條路徑年化最小 %', 21.79, s3['paths_summary']['min'] * 100, 0.3, '百分點'); add('20 條路徑年化最大 %', 24.25, s3['paths_summary']['max'] * 100, 0.3, '百分點')
for k, (a, b) in {'2008': (-49.21, -52.38), '2011': (-24.64, -23.01), '2015': (-21.12, -20.96), '2018': (-22.30, -15.99), '2020H1': (-26.98, -28.22), '2022': (-27.55, -33.96), '2024-07~2025-04': (-32.04, -27.48)}.items():
    add(f'空頭 {k} V_BRIDGE 段內跌幅 %', a, s3['bears'][k]['strat'] * 100, 0.5, '百分點'); add(f'空頭 {k} 0050 段內跌幅 %', b, s3['bears'][k]['e0050'] * 100, 0.5, '百分點')
add('勝過 0050 年數（19 年中）', 14, s3['beat_years'], 0, '年')
cmp = pd.DataFrame(rows); cmp.to_csv(HERE / 'compare.csv', index=False)
# 逐日／逐年對照 S75 存檔（最後才讀）
a = pd.read_csv(TMP / 'nav_cache_full.csv', index_col=0, parse_dates=True); s = pd.read_csv(S75 / 'nav_2008_2026.csv.gz', index_col=0, parse_dates=True)
navcmp = dict(same_dates=bool(a.index.equals(s.index)), n_days=int(len(a)), max_rel_diff={c: float((a[c] / s[c] - 1).abs().max()) for c in ('V_BRIDGE', 'V_MASK', '0050')},
              max_abs_diff_twd={c: float((a[c] - s[c]).abs().max()) for c in ('V_BRIDGE', 'V_MASK', '0050')})
y = pd.read_csv(S75 / 'yearly.csv').set_index('year')
ycmp = dict(max_abs_diff_pp=dict(V_BRIDGE=float(max(abs(y.at[int(k), 'V_BRIDGE'] - v) for k, v in B['yearly'].items()) * 100), V_MASK=float(max(abs(y.at[int(k), 'V_MASK'] - v) for k, v in M['yearly'].items()) * 100),
                                 e0050=float(max(abs(y.at[int(k), 'e0050'] - v) for k, v in E['yearly'].items()) * 100)))
a.round(2).to_csv(HERE / 'nav_r75.csv.gz')
ab['d']['trades_post2016'].pop('only_bridge_note', None)
res = dict(task='R75：獨立重現 S75（未讀 s75lib.py／run.py／build_panel.py／results.json）', verdict='重現成功' if cmp['ok'].all() else '重現有不符',
           n_items=int(len(cmp)), n_ok=int(cmp['ok'].sum()), not_ok=cmp[~cmp['ok']].to_dict('records'),
           setup=dict(engine='s47lib.sim → s10lib.simulate_s10（W10_70、E0、strict、100 萬）；股池 stlib.universe(twse) & ~創新板；SURGE＝s71lib.rev_rule（B2／MASK）；F1 make_signal_s10(shift=0)；三個正式開關 True；e50_fix_suspension',
                      runtime_overrides='stlib.START＝2008-01-02、s17lib.FIRST_MONTH＝2008-01、各層快取改到 scratchpad/r75（不呼叫 git）',
                      panel='2015-01-05 前＝R55 主做法（Asoul 長表＋data_panel 補格）；2015-01-05 起＝官方面板原值；欄＝pre 926 ∪ 官方全部 2142＝2200（不丟欄）',
                      seam='2015-01-05（官方日 K 第一天；官方資料有的日子一律用官方）。敏感度：接縫改 2015-07-01、或只留曾上市欄（1222 欄，S75 的做法）→ 三條淨值逐日完全相同'),
           step1_official=s1, panel=pan, run=run, extras=s3, vs_S75_saved=dict(nav=navcmp, yearly=ycmp),
           sensitivity=dict(seam_2015_07_01={v: J('run_cache_seam07.json')[v]['final'] for v in ('V_BRIDGE', 'V_MASK')}, cols_twse_only={v: J('run_cache_twse.json')[v]['final'] for v in ('V_BRIDGE', 'V_MASK')}),
           audit_a_0050=aa, audit_b_2008=ab['b'], audit_held_days_beyond_limit=ab['held_days_beyond_limit'], audit_trades_vs_data_panel=ab['trades_vs_data_panel_pre2015H1'], audit_c_seam=ab['c'], audit_d_bridge_vs_mask=ab['d'],
           audit_limit7=dict(note='2015-06-01 前實際漲跌停 7%：開盤 ≥ +6.85% 的買進 V_BRIDGE 5 筆（2478、1516、6120、8201、4426；淨報酬 −6.8%、+5.1%、+8.4%、−20.5%、+7.5%），開盤 ≤ −6.85% 的賣出 0 筆；2008-10-15（跌幅 3.5% 限制期間）賣出的 3 檔開盤 −2.1%、+2.2%、−1.1%，都不在跌停'),
           maxrss_mb=dict(step1=s1['maxrss_mb'], run_full=run['maxrss_mb'], audit=ab['maxrss_mb'], paths=s3['maxrss_mb']))
(HERE / 'results.json').write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str))
print(res['verdict'], res['n_ok'], '/', res['n_items'], res['not_ok']); print(navcmp, ycmp)
