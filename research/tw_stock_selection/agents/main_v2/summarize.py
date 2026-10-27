#!/usr/bin/env python3
"""合併 run_<win>.json、results_tool_compare.json 進 results.json（保留 PREREG），算正式數字與區間，印出 FINDINGS 用的表。"""
import sys, json
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
R = json.loads((HERE / 'results.json').read_text(encoding='utf-8'))
assert 'PREREG_sha256' in R
W = {w: json.loads((HERE / f'run_{w}.json').read_text(encoding='utf-8')) for w in ('strict', 'gapfix', 'lenient')}
TC = json.loads((HERE / 'results_tool_compare.json').read_text(encoding='utf-8')) if (HERE / 'results_tool_compare.json').exists() else {}
SEGS = ['S1', 'S2', 'S3', 'S4', 'P1', 'ALL']
MET = ['CAGR', 'maxDD', 'Calmar', 't_0050', 't_EW', 't_0050_cal', 't_EW_cal', 'ex_0050', 'ex_EW', 'E50_CAGR', 'EW_CAGR']

def get(w, u, n, c, tie, seg, m):
    return W[w]['grid'][u][str(n)][c][tie][seg][m]

OFF = {}; RNG = {}
for u in ('ALL', 'EXFIN'):
    for n in (10, 30):
        for c in ('normal', 'stress'):
            for seg in SEGS:
                for m in MET:
                    k = f'{u}|n{n}|{c}|{seg}|{m}'
                    OFF[k] = get('gapfix', u, n, c, 'official', seg, m)
                    allv = [get(w, u, n, c, t, seg, m) for w in W for t in W[w]['grid'][u][str(n)][c]]
                    tiev = [get('gapfix', u, n, c, t, seg, m) for t in W['gapfix']['grid'][u][str(n)][c]]
                    winv = [get(w, u, n, c, 'official', seg, m) for w in W]
                    RNG[k] = {'all93': [min(allv), max(allv)], 'tie31_gapfix': [min(tiev), max(tiev)], 'win3_official': winv, 'n': len(allv)}

AB = {}
for u in ('ALL', 'EXFIN'):
    for n in (10, 30):
        for c in ('normal', 'stress'):
            for seg in SEGS:
                for m in MET:
                    v = {w: W[w]['AB'][u][str(n)][c][seg][m] for w in W}
                    AB[f'{u}|n{n}|{c}|{seg}|{m}'] = {'gapfix': v['gapfix'], 'win3': [v['strict'], v['gapfix'], v['lenient']]}
                mc = {w: W[w]['AB'][u][str(n)][c + '_mc_p05_p95'][seg] for w in W}
                AB[f'{u}|n{n}|{c}|{seg}|CAGR_mc'] = mc

# 實盤預期：EXFIN＋AB、gapfix、正式次排序；超額打五折
EXP = {}
for n in (10, 30):
    for c in ('normal', 'stress'):
        for seg in ('ALL', 'P1', 'S3', 'S4'):
            a = W['gapfix']['AB']['EXFIN'][str(n)][c][seg]
            ex = a['CAGR'] - a['E50_CAGR']
            exs = [W[w]['AB']['EXFIN'][str(n)][c][seg]['CAGR'] - W[w]['AB']['EXFIN'][str(n)][c][seg]['E50_CAGR'] for w in W]
            mc = W['gapfix']['AB']['EXFIN'][str(n)][c + '_mc_p05_p95'][seg]
            lo = min(min(exs), mc[0] - a['E50_CAGR']); hi = max(max(exs), mc[2] - a['E50_CAGR'])
            EXP[f'n{n}|{c}|{seg}'] = {'backtest_CAGR': a['CAGR'], 'E50_CAGR': a['E50_CAGR'], 'excess': ex, 'excess_range': [lo, hi],
                                     'expected_excess': 0.5 * ex, 'expected_excess_range': [0.5 * lo, 0.5 * hi],
                                     'expected_CAGR': a['E50_CAGR'] + 0.5 * ex, 'expected_CAGR_range': [a['E50_CAGR'] + 0.5 * lo, a['E50_CAGR'] + 0.5 * hi],
                                     'maxDD': a['maxDD'], 't_0050': a['t_0050'], 't_EW': a['t_EW']}

R['RUN'] = {w: {k: v for k, v in W[w].items() if k != 'grid'} for w in W}
R['GRID'] = {w: W[w]['grid'] for w in W}
R['OFFICIAL'] = OFF; R['RANGES'] = RNG; R['AB_SUMMARY'] = AB; R['EXPECTATION'] = EXP; R['TOOL_COMPARE'] = TC
(HERE / 'results.json').write_text(json.dumps(R, ensure_ascii=False, indent=0), encoding='utf-8')

# ---------------- 印表 ----------------
f1 = lambda x: f'{x:.1f}'; f2 = lambda x: f'{x:.2f}'
def cell(k, m, fmt=f1):
    v = OFF[k + '|' + m]; lo, hi = RNG[k + '|' + m]['all93']
    return f'{fmt(v)}（{fmt(lo)}～{fmt(hi)}）'
print('## 正式數字（gapfix＋正式次排序＋一般成本；括號＝93 組 同分×滾動窗 範圍）')
for u in ('ALL', 'EXFIN'):
    for n in (10, 30):
        print(f'\n### {u} n{n}')
        print('| 期間 | 年化% | 最大回撤% | Calmar | t 對 0050 | t 對等權 | 超額 0050 pp | 超額等權 pp | 0050 年化 | 等權年化 |')
        print('|---|---|---|---|---|---|---|---|---|---|')
        for seg in SEGS:
            k = f'{u}|n{n}|normal|{seg}'
            print(f'| {seg} | {cell(k, "CAGR")} | {cell(k, "maxDD")} | {cell(k, "Calmar", f2)} | {cell(k, "t_0050", f2)} | {cell(k, "t_EW", f2)} | {cell(k, "ex_0050")} | {cell(k, "ex_EW")} | {f1(OFF[k + "|E50_CAGR"])} | {f1(OFF[k + "|EW_CAGR"])} |')
print('\n## 細分區間（年化；同分 31 組 in gapfix ／ 三版窗 strict,gapfix,lenient 正式次排序）')
for u in ('ALL', 'EXFIN'):
    for n in (10, 30):
        row = []
        for seg in SEGS:
            r = RNG[f'{u}|n{n}|normal|{seg}|CAGR']
            row.append(f'{seg} 同分 {r["tie31_gapfix"][0]:.1f}～{r["tie31_gapfix"][1]:.1f}／窗 ' + ','.join(f'{x:.1f}' for x in r['win3_official']))
        print(f'- {u} n{n}: ' + '；'.join(row))
print('\n## t 細分（t_0050：同分範圍 in gapfix／三版窗）')
for u in ('ALL', 'EXFIN'):
    for n in (10, 30):
        row = []
        for seg in SEGS:
            r = RNG[f'{u}|n{n}|normal|{seg}|t_0050']
            row.append(f'{seg} {r["tie31_gapfix"][0]:.2f}～{r["tie31_gapfix"][1]:.2f}／' + ','.join(f'{x:.2f}' for x in r['win3_official']))
        print(f'- {u} n{n}: ' + '；'.join(row))
print('\n## 日曆月 t（附表；gapfix 正式）')
for u in ('ALL', 'EXFIN'):
    for n in (10, 30):
        print(f'- {u} n{n}: ' + '；'.join(f'{seg} {OFF[f"{u}|n{n}|normal|{seg}|t_0050_cal"]:.2f}/{OFF[f"{u}|n{n}|normal|{seg}|t_EW_cal"]:.2f}' for seg in SEGS))
print('\n## 壓力成本（gapfix 正式；括號 93 組範圍）')
for u in ('ALL', 'EXFIN'):
    for n in (10, 30):
        print(f'- {u} n{n}: ' + '；'.join(f'{seg} {cell(f"{u}|n{n}|stress|{seg}", "CAGR")} t {OFF[f"{u}|n{n}|stress|{seg}|t_0050"]:.2f}/{OFF[f"{u}|n{n}|stress|{seg}|t_EW"]:.2f}' for seg in SEGS))
print('\n## 除權息近似（gapfix 正式，一般成本）：年化 原樣→近似，t0050 原樣→近似')
for u in ('ALL', 'EXFIN'):
    for n in (10, 30):
        e = W['gapfix']['exdiv'][u][str(n)]
        print(f'- {u} n{n}: ' + '；'.join(f'{seg} {OFF[f"{u}|n{n}|normal|{seg}|CAGR"]:.1f}→{e[seg]["CAGR"]:.1f}, t {OFF[f"{u}|n{n}|normal|{seg}|t_0050"]:.2f}→{e[seg]["t_0050"]:.2f}, t_EW {OFF[f"{u}|n{n}|normal|{seg}|t_EW"]:.2f}→{e[seg]["t_EW"]:.2f}' for seg in SEGS))
print(f'  平均跳空 {W["gapfix"]["exdiv_gap_mean"]:.5f}，flag1 股票日 {W["gapfix"]["n_flag1_days"]}')
print('\n## 到最新資料日 2026-10-22（一般成本；exec_oct＝執行 10-12 訊號；hold_sep＝9 月持股抱著）')
for w in W:
    for u in ('ALL', 'EXFIN'):
        for n in (10, 30):
            L = W[w]['latest'][u][str(n)]
            print(f'- {w} {u} n{n}: ' + '；'.join(f'{seg} 執行 {L["exec_oct|normal"][seg]["CAGR"]:.1f}(t {L["exec_oct|normal"][seg]["t_0050"]:.2f}) 不換 {L["hold_sep"][seg]["CAGR"]:.1f}' for seg in ('S3', 'S4', 'ALL')))
print('\n## AB（只用已公告營收，MC 40 平均淨值；gapfix；括號＝三版窗 strict,gapfix,lenient；MC 5/50/95）')
for u in ('ALL', 'EXFIN'):
    for n in (10, 30):
        print(f'\n### AB {u} n{n}')
        print('| 期間 | 年化% | 三版窗 | MC 5/50/95（gapfix） | 最大回撤% | Calmar | t 對 0050 | t 對等權 | 壓力年化 | 最壞年化 | 原版（BASE）年化 |')
        print('|---|---|---|---|---|---|---|---|---|---|---|')
        for seg in SEGS:
            a = W['gapfix']['AB'][u][str(n)]['normal'][seg]
            w3 = ','.join(f'{W[w]["AB"][u][str(n)]["normal"][seg]["CAGR"]:.1f}' for w in W)
            mc = W['gapfix']['AB'][u][str(n)]['normal_mc_p05_p95'][seg]
            st = W['gapfix']['AB'][u][str(n)]['stress'][seg]['CAGR']; wo = W['gapfix']['AB'][u][str(n)]['worst|normal'][seg]['CAGR']
            print(f'| {seg} | {a["CAGR"]:.1f} | {w3} | {mc[0]:.1f}/{mc[1]:.1f}/{mc[2]:.1f} | {a["maxDD"]:.1f} | {a["Calmar"]:.2f} | {a["t_0050"]:.2f} | {a["t_EW"]:.2f} | {st:.1f} | {wo:.1f} | {OFF[f"{u}|n{n}|normal|{seg}|CAGR"]:.1f} |')
        print(f'  平均每期前 300 名中被判未公告的檔數 {W["gapfix"]["AB"][u][str(n)]["mean_unavail_in_top300"]:.1f}')
        if 'latest_exec_oct|normal' in W['gapfix']['AB'][u][str(n)]:
            L = W['gapfix']['AB'][u][str(n)]['latest_exec_oct|normal']
            print('  到 10-22（執行 10 月）：' + '；'.join(f'{seg} {L[seg]["CAGR"]:.1f}' for seg in ('S3', 'S4', 'ALL')))
print('\n## 實盤預期（EXFIN＋AB、gapfix；超額打五折）')
print('| 版本 | 期間 | 成本 | 回測年化 | 0050 | 回測超額 | 超額範圍 | 預期超額（×0.5） | 預期年化 | 預期年化範圍 | 回撤 |')
print('|---|---|---|---|---|---|---|---|---|---|---|')
for k, e in EXP.items():
    n, c, seg = k.split('|')
    print(f'| {n} | {seg} | {c} | {e["backtest_CAGR"]:.1f} | {e["E50_CAGR"]:.1f} | {e["excess"]:+.1f} | {e["excess_range"][0]:+.1f}～{e["excess_range"][1]:+.1f} | {e["expected_excess"]:+.1f} | {e["expected_CAGR"]:.1f} | {e["expected_CAGR_range"][0]:.1f}～{e["expected_CAGR_range"][1]:.1f} | {e["maxDD"]:.1f} |')
print('\n## 逐年（gapfix 正式，一般成本）')
yrs = sorted({int(y) for y in W['gapfix']['grid']['ALL']['10']['normal']['official']['ALL']['yearly']})
cols = [('ALL', 10), ('ALL', 30), ('EXFIN', 10), ('EXFIN', 30)]
print('| 年 | ' + ' | '.join(f'{u} n{n}' for u, n in cols) + ' | AB EXFIN n10 | AB EXFIN n30 | 0050 |')
print('|---' * (len(cols) + 4) + '|')
import pandas as pd
sys.path.insert(0, str(HERE)); import v2_core as V
nav = pd.read_parquet(HERE / 'navs_gapfix.parquet')
e50 = V.K.E50.reindex(nav.index)
e50y = {int(y): round(float((1 + g).prod() - 1) * 100, 1) for y, g in e50.pct_change()[(nav.index >= '2006-04-11') & (nav.index <= '2026-10-12')].groupby(nav.index[(nav.index >= '2006-04-11') & (nav.index <= '2026-10-12')].year)}
for y in yrs:
    row = [W['gapfix']['grid'][u][str(n)]['normal']['official']['ALL']['yearly'][str(y)] for u, n in cols]
    ab = [W['gapfix']['AB']['EXFIN'][str(n)]['normal']['ALL']['yearly'][str(y)] for n in (10, 30)]
    print(f'| {y} | ' + ' | '.join(f'{x:.1f}' for x in row + ab) + f' | {e50y.get(y, float("nan")):.1f} |')
print('\n## 最大單股貢獻（gapfix 正式，一般成本，毛報酬近似）')
for u in ('ALL', 'EXFIN'):
    for n in (10, 30):
        C = W['gapfix']['official'][u][str(n)]['contrib']
        print(f'- {u} n{n}: ' + '；'.join(f'{seg} {c["top_code"]}{c["top_name"]} {c["top_contrib_pp"]:.1f}pp／總 {c["sum_contrib_pp"]:.0f}pp（{c["top_share"]*100:.1f}%），剔除後年化 −{c["ex_top_drop_pp"]:.1f}pp' for seg, c in C.items()))
print('\n## 其他（gapfix 正式）')
for u in ('ALL', 'EXFIN'):
    for n in (10, 30):
        o = W['gapfix']['official'][u][str(n)]
        print(f'- {u} n{n}: 周轉 {o["normal"]["turnover"]:.3f}，開盤漲停跳過 {o["normal"]["blocked"]} 次，停牌凍結 {o["normal"]["n_frozen"]} 次，換股 {o["normal"]["n_signals"]} 次（{o["normal"]["first_exec"]}～{o["normal"]["last_exec"]}），金融名額 {o["fin_slots_share"]*100:.1f}%，平均池 {W["gapfix"]["official"][u]["pool_mean"]:.0f}')
print('\n## 10-12 訊號名單')
for w in W:
    for u in ('ALL', 'EXFIN'):
        for n in (10, 30):
            L = W[w]['lists_1012'][u][str(n)]
            print(f'- {w} {u} n{n}: 池 {L["pool"]}，買 {L["picks"]}，隨機同分下不同名單數 {L["rand_tie_lists_distinct"]}')
    for u in ('ALL', 'EXFIN'):
        if 'AB_' + u in W[w]['lists_1012']:
            for n in (10, 30):
                L = W[w]['lists_1012']['AB_' + u][str(n)]
                print(f'- {w} AB {u} n{n}: {L["picks"]}（MC 不同名單數 {L["n_distinct_over_mc"]}）')
print('\n## 工具比較')
for k, v in TC.items():
    if not v['same_set']:
        print(f'- {k}: v2 多 {v["only_v2"]}；工具多 {v["only_tool"]}；工具跳過：{v["tool_skipped_lines"]}')
print(f'- 共 {len(TC)} 組，相同 {sum(v["same_set"] for v in TC.values())} 組')
