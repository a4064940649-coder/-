#!/usr/bin/env python3
"""合併各階段結果進 results.json（不動 PREREG），並寫入最終判定。"""
import sys, json, hashlib, datetime
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import olib as L
H = L.HERE
PR, SH, PT = L.assert_prereg()
doc = json.loads((H / 'results.json').read_text(encoding='utf-8'))
J = lambda f: json.loads((H / f).read_text(encoding='utf-8'))
E, I, P4S, P4O, OOS, ROB, FULL, DQ = (J(f) for f in ('explore.json', 'iterations.json', 'p4_sel.json', 'p4_oos.json', 'results_oos.json', 'p4_robust.json', 'p4_full.json', 'dq.json'))
assert OOS['PREREG_sha256'] == SH
v4b = P4O['ex_src3']['buy']; nv = P4O['nav']
c1 = v4b['imp_close_mean_pct'] > 0.20 and v4b['imp_close_t_month'] >= 1.0
c2 = nv['normal|buy_close|sell_open|ex_src3']['paired_vs_open']['mean_pct'] > 0 and nv['stress|buy_close|sell_open|ex_src3']['paired_vs_open']['mean_pct'] > 0
c3 = P4O['ex_src3_ex_limitclose']['buy']['imp_close_mean_pct'] > 0
p4_primary = all(P[k]['pooled'][f'imp_{x}_mean_pct'] > 0.20 for P, k in ((P4S, 'all'), (P4O, 'ex_src3')) for x in ('close',)) or \
             all(P[k]['pooled'][f'imp_{x}_mean_pct'] > 0.20 for P, k in ((P4S, 'all'), (P4O, 'ex_src3')) for x in ('twap',))
doc['SUMMARY'] = {
 'time': datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds'),
 'verdict_combo': '不可用（凍結前判定：8 版配對差全部為負，沒有 t ≥ 1.0；驗證期未計算結合版）',
 'P1_verdict': OOS['P1_verdict'], 'P1_primary': {k: {p: L.r4({x: OOS['P1'][k][p][x] for x in ('n', 'hml_pct', 't', 't_nw6')}) for p in ('selection', 'validation_main', 'validation_keep_src3')} for k in ('primary_L252_S21|IMOM', 'primary_L252_S21|OMOM', 'secondary_L126_S0|IMOM', 'secondary_L126_S0|OMOM')},
 'standalone_leg': OOS['standalone_leg'], 'corr_IMOM_vs_main_excess': OOS['corr_IMOM_spread_vs_main_excess_0050'],
 'P2_L252_S0_share': OOS['P2_L252_S0_share'], 'P5_triggered': '是（挑選期主策略前 30 名占比分組 t 0.10 < 1；驗證期 OMOM 也不反轉）',
 'P4_primary_pass': bool(p4_primary), 'P4_primary_verdict': '維持隔日開盤（買賣合併每筆改善：挑選期 close +0.05%、TWAP +0.02%；驗證期 close +0.13%、TWAP +0.05%，都 < 0.20%）',
 'P4b_criteria': {'c1_buy_imp_gt_0.2_t_ge_1': bool(c1), 'c2_nav_paired_pos_normal_stress': bool(c2), 'c3_ex_limitclose_pos': bool(c3)},
 'P4b_verdict': '可用（買進改隔日收盤、賣出維持隔日開盤）' if (c1 and c2 and c3) else '未通過',
 'P4b_full_period': {k: L.r4({kk: vv for kk, vv in v.items() if not kk.startswith('yearly')}) for k, v in FULL.items()},
 'P4b_robust': ROB,
 'src3_split_error': DQ['src3_error'],
 'iterations': {v: {'spec': o['spec'], 'combo': o['combo'], 'SEL': L.r4({x: o['normal']['SEL'][x] for x in ('CAGR', 'maxDD', 't_0050', 't_EW')}),
                    'paired_vs_main': L.r4(o['normal']['paired_vs_main']['SEL'])} for v, o in I['results']['versions'].items()},
 'main_selection': L.r4({x: I['results']['main']['normal']['SEL'][x] for x in ('CAGR', 'maxDD', 't_0050', 't_EW')}),
 'files_sha256': {f: hashlib.sha256((H / f).read_bytes()).hexdigest() for f in sorted(p.name for p in H.glob('*') if p.suffix in ('.py', '.json') and p.name != 'results.json')},
}
(H / 'results.json').write_text(json.dumps(doc, ensure_ascii=False, indent=1), encoding='utf-8')
L.assert_prereg()
print(json.dumps({k: doc['SUMMARY'][k] for k in ('verdict_combo', 'P1_verdict', 'P4_primary_pass', 'P4b_criteria', 'P4b_verdict')}, ensure_ascii=False, indent=1))
