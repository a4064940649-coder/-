#!/usr/bin/env python3
"""S4c 判定格：嚴格（＝寬鬆）、posthoc 對 o18 新正式數字（題 1 RESULTS_V7X、RESULTS_PAPER）逐格取 opt／p90／carry 最差；Δ 門檻 0.3pp／t 0.05；30 萬兩模式。
python3 -I -B s14_decide.py → results.json DECISION"""
import json, hashlib
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent; AG = HERE.parent
RJ = HERE / 'results.json'; R = json.loads(RJ.read_text(encoding='utf-8'))
for k in ('PREREG', 'PREREG_ADDENDUM'): assert hashlib.sha256(json.dumps(R[k], ensure_ascii=False, sort_keys=True).encode()).hexdigest() == R[k + '_sha256']
T1 = json.loads((AG / 'o_delist_rev_backfill_allpit/results.json').read_text(encoding='utf-8'))['RESULTS_V7X']['table']
TB = R['RESULTS_V7X']['table']; V = ('REAL_opt', 'REAL_p90', 'REAL_carry')
CELL = {'100萬一般 全期超額': ('normal', '100萬', 'exc_med'), '100萬一般 t': ('normal', '100萬', 't_0050_med'), '100萬一般 j p10': ('normal', '100萬', 'exc_p10'),
        '2019 起超額（100萬一般）': ('normal', '100萬', 'exc_Y19on_med'), '2019 起年化（100萬一般）': ('normal', '100萬', 'CAGR_Y19on_med'),
        '100萬壓力 全期超額': ('stress', '100萬', 'exc_med'), '300萬一般 全期超額': ('normal', '300萬', 'exc_med'), '1000萬一般 全期超額': ('normal', '1000萬', 'exc_med')}
def get(src, v, c, t, f, dv):
    if src == 'off': return T1[f'{v}|cons_main|{c}|{t}'][f]
    key = f'{dv}|{v}|cons_main|{c}|{t}|std'
    if key in TB: return TB[key][f]
    if dv == 'posthoc' and R['RESULTS_V7X']['posthoc_plans_changed'] == 0: return T1[f'{v}|cons_main|{c}|{t}'][f]
    if dv == 'strict': return T1[f'{v}|cons_main|{c}|{t}'][f]   # 嚴格名單 sha 與官方相同（s11a 斷言計數＋ends；100 萬一般重現差 0）→ 其餘格＝官方
    raise KeyError(key)
worst = lambda xs: min(xs)   # 所有欄都是「越大越好」，最差＝最小
tab = {}
for nm, (c, t, f) in CELL.items():
    off = worst([get('off', v, c, t, f, None) for v in V])
    st = worst([get('new', v, c, t, f, 'strict') for v in V])
    ph = worst([get('new', v, c, t, f, 'posthoc') for v in V])
    adverse = min(st, ph)
    thr = 0.05 if f.startswith('t_') else 0.3
    tab[nm] = dict(official_worst=round(off, 4), strict_worst=round(st, 4), posthoc_worst=round(ph, 4), adverse_strict_vs_posthoc=round(adverse, 4),
                   delta_strict=round(st - off, 4), delta_posthoc=round(ph - off, 4), threshold=thr,
                   exceeds_strict=abs(st - off) > thr, exceeds_posthoc=abs(ph - off) > thr,
                   by_variant_posthoc={v: round(get('new', v, c, t, f, 'posthoc') - get('off', v, c, t, f, None), 4) for v in V})
# 30 萬
t30 = {}
for mode in ('std', 'wholelot'):
    for c in ('normal', 'stress'):
        ks = [f'strict|{v}|cons_main|{c}|30萬|{mode}' for v in V]
        if not all(k in TB for k in ks): continue
        t30[f'{c}|{mode}'] = {f: dict(by_variant={v: round(TB[k][f], 3) for v, k in zip(V, ks)}, worst=round(min(TB[k][f] for k in ks), 3))
                              for f in ('exc_med', 'exc_p10', 't_0050_med', 'exc_Y19on_med', 'CAGR_Y19on_med', 'exc_S0612_med', 'exc_S1318_med', 'exc_S19on_med', 'CAGR_PREODD_med', 'exc_PREODD_med', 'exc_POSTODD_med', 'maxDD_med') if f in TB[ks[0]]}
# 紙上
PS = R['RESULTS_PAPER']['stats']; P1 = json.loads((AG / 'o_delist_rev_backfill_allpit/results.json').read_text(encoding='utf-8'))['RESULTS_PAPER']['stats']
paper = {}
for f, lab in (('ann', '紙上全期年化'), ('t_vs_0050', '紙上全期 t')):
    off = min(P1[f'ALLPIT_{v}|normal|cons_main|vs_C0']['ALL'][f] for v in V)
    st = min(PS[f'strict|ALLPIT_{v}|normal']['ALL'][f] for v in V)
    paper[lab] = dict(official_worst=off, strict_worst=st, posthoc_worst=st, delta=st - off, note='posthoc 紙上名單 0 槽變動')
any_replace = any(x['exceeds_strict'] for x in tab.values()) or any(abs(x['delta']) > (0.05 if 't' in k else 0.3) for k, x in paper.items())
any_replace_posthoc = any(x['exceeds_posthoc'] for x in tab.values())
M = R['RESULTS_MODEL']
DEC = dict(cells=tab, paper=paper, tier_30w=t30, replace_strict=any_replace, replace_if_posthoc_counted=any_replace_posthoc,
           verdict_numbers='加註（不取代）：嚴格＝寬鬆補列 0、公告時點 P1 成立 → 主策略正式數字 Δ＝0（名單 sha 逐位相同、重現差 0）。posthoc（範圍外單一來源 2344）各格 |Δ| 見 cells，未超過門檻者不改判。',
           model=dict(y2019_O=M['y2019']['O'], E=M['y2019']['E'], PI95_primary=M['y2019']['PI95'], PI95_calibrated=M['y2019']['PI95_calibrated'], z=M['y2019']['z'],
                      verdict_primary=M['y2019']['verdict_primary'], years_below_primary_PI=M['years_below_PI'], phi=M['phi_annual']),
           P_status={'P1': '成立（計數 2218/2218、0 列 start≤announce；公開證據：ELD 盤後 14:32 文章、2026 年「明起處置」報導；計數部分凍結前已知）',
                     'P2': '部分不成立：2019 觀察 31 低於主 PI95 [42,89]（z −2.65），但在年層級校準區間 [27.5,98.3] 內；嚴格補列 0 < 5 → 依規則不得改判 H_inc；歸因見 FINDINGS',
                     'P3': '無法檢驗（無缺漏可分布）',
                     'P4': '成立：嚴格受影響槽 0；posthoc 影響 V7X 2020-12 一個訊號月（4 個 j 日）、紙上 0 槽',
                     'P5': '成立：1 條規則（事件紀錄路徑 /frozen、/rows 排除）解釋 18/18'})
R['DECISION'] = DEC
RJ.write_text(json.dumps(R, ensure_ascii=False, indent=1, default=str), encoding='utf-8')
for k, v in tab.items(): print(k, v['official_worst'], v['strict_worst'], v['posthoc_worst'], v['delta_posthoc'], v['exceeds_posthoc'], v['by_variant_posthoc'])
print(json.dumps(paper, ensure_ascii=False)); print(any_replace, any_replace_posthoc)
for k, v in t30.items(): print(k, {f: v[f]['worst'] for f in v})
