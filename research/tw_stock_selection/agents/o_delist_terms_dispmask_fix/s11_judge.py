#!/usr/bin/env python3
"""步驟 10：依 PREREG.decision 判定（不改規格）＋預測對照＋G1～G8 欄；寫入 results.json 的 VERDICT。python3 -I -B s11_judge.py"""
import json, hashlib
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent
RJ = HERE / 'results.json'; R = json.loads(RJ.read_text(encoding='utf-8'))
assert hashlib.sha256(json.dumps(R['PREREG'], ensure_ascii=False, sort_keys=True).encode()).hexdigest() == R['PREREG_sha256']
IA, VX = R['RESULTS_IA'], R['RESULTS_V7X']; T = VX['table']; S = IA['stats']
PK = json.loads((HERE / 'pklscan.json').read_text(encoding='utf-8'))['summary']
CELLS = ['normal|100萬', 'stress|100萬', 'normal|300萬', 'normal|1000萬']
F = ['CAGR_med', 'exc_med', 't_0050_med', 'exc_p10', 'exc_Y19on_med', 't_Y19on_med', 'exc_Y19on_p10', 'exc_0612_med', 'exc_1318_med', 'exc_19e_med', 'exc_1419_med']
def dtab(a, b):
    rows = []
    for c in CELLS:
        for f in F:
            ka, kb = f'{a}|{c}', f'{b}|{c}'
            if ka not in T or kb not in T: continue
            d = T[ka][f] - T[kb][f]; lim = 0.05 if f.startswith('t_') else 0.3
            rows.append(dict(cell=c, field=f, new=T[ka][f], ref=T[kb][f], delta=d, exceed=bool(abs(d) > lim)))
    return rows
D = {k: dtab(a, b) for k, (a, b) in {'v3_main': ('orig|v3_main', 'orig|orig'), 'v3_lo': ('orig|v3_lo', 'orig|orig'), 'v3_hi': ('orig|v3_hi', 'orig|orig'),
                                     'o17_cons_main': ('orig|o17_cons_main', 'orig|orig'), 'union': ('union|orig', 'orig|orig'), 'replace': ('replace|orig', 'orig|orig'),
                                     'union+v3': ('union|v3_main', 'orig|orig'), 'union+v3_lo': ('union|v3_lo', 'orig|orig'), 'union+v3_hi': ('union|v3_hi', 'orig|orig')}.items()}
g = lambda k, c='normal|100萬', f='exc_med': T[f'{k}|{c}'][f]
lo, hi, mn, o0 = g('orig|v3_lo'), g('orig|v3_hi'), g('orig|v3_main'), g('orig|orig')
tlo, thi, tmn, t0 = g('orig|v3_lo', f='t_0050_med'), g('orig|v3_hi', f='t_0050_med'), g('orig|v3_main', f='t_0050_med'), g('orig|orig', f='t_0050_med')
E3 = pd.read_csv(HERE / 'events_v3.csv', dtype={'code': str})
# 界限由哪些事件決定：v3_lo 下非 VERIFIED_TEXT 事件 lt 市值 × δ_evt 的合計（260 份名單平均）
contrib = {}
for k, x in VX['settle_info'].items():
    for e in x.get('orig|v3_lo', {}).get('ev', []):
        r = E3[E3.code == e[0]]
        if len(r) and r.text_status.iloc[0] != 'VERIFIED_TEXT':
            contrib[e[0]] = contrib.get(e[0], 0.0) + e[3] * r.delta_evt.iloc[0] / 260
contrib = dict(sorted(contrib.items(), key=lambda kv: -kv[1]))
a_ok = bool(lo > 0 and hi > 0)
ann_a = (f'下市對價口徑差：主口徑 {mn:+.2f}pp（對現行 {o0:+.2f}pp，Δ{mn - o0:+.2f}）、界限 {lo:+.2f}～{hi:+.2f}pp，均 > 0；t {tmn:.3f}（界限 {tlo:.3f}～{thi:.3f}）' if a_ok else
         f'下市對價口徑差：主口徑全期超額 {mn:+.2f}pp（Δ{mn - o0:+.2f}）、t {tmn:.3f}；界限 {lo:+.2f}～{hi:+.2f}pp、t {tlo:.3f}～{thi:.3f}，下界由 ' + '、'.join(list(contrib)[:5]) + ' 等未核實事件的 δ 決定；NOT_SEARCHED 0 起')
# (b) 處置遮罩
c0u = S['C0|union|freeze|normal']; c0 = S['C0|orig|freeze|normal']; c0us = S['C0|union|freeze|stress']; c0s = S['C0|orig|freeze|stress']
pap = {f'{seg}': dict(d_gex=c0u[seg]['gex'] - c0[seg]['gex'], d_t=c0u[seg]['t_vs_0050'] - c0[seg]['t_vs_0050']) for seg in ('ALL', 'SEG1', 'SEG2', 'SEG3', 'Y1419')}
pap['ALL_ann'] = c0u['ALL']['ann'] - c0['ALL']['ann']; pap['stress_ALL_ann'] = c0us['ALL']['ann'] - c0s['ALL']['ann']
pap['t_drop_best3'] = c0u['ALL']['t_drop_best3'] - c0['ALL']['t_drop_best3']
b_paper = any(abs(v['d_gex']) > 0.3 or abs(v['d_t']) > 0.05 for k, v in pap.items() if isinstance(v, dict)) or abs(pap['ALL_ann']) > 0.3
b_v7x = any(r['exceed'] for r in D['union'] if r['field'] in ('CAGR_med', 'exc_med', 't_0050_med', 'exc_p10', 'exc_Y19on_med', 't_Y19on_med', 'exc_Y19on_p10'))
yd = VX['yearly']['union_vs_orig(orig)']['med']; top_y = sorted(yd.items(), key=lambda kv: -abs(kv[1]))[:3]
desc = IA['slot_desc']['C0|union']
ann_b = (f"處置遮罩官方化（2006～2021-09-23 官方處置期涵蓋訊號日或成交日者排除，＝C_DSP）：紙上 C0 全期年化 Δ{pap['ALL_ann']:+.2f}pp、t Δ{pap['ALL']['d_t']:+.3f}，2019 起超額 Δ{pap['SEG3']['d_gex']:+.2f}pp，"
         f"集中在 {'、'.join(d['sig'] for d in desc['top_dates'][:2])} 訊號日；V7X 100 萬一般 j 中位數全期超額 Δ{[r for r in D['union'] if r['cell']=='normal|100萬' and r['field']=='exc_med'][0]['delta']:+.2f}pp、"
         f"2019 起 Δ{[r for r in D['union'] if r['cell']=='normal|100萬' and r['field']=='exc_Y19on_med'][0]['delta']:+.2f}pp（逐年最大：" + '、'.join(f'{y} {v:+.2f}' for y, v in top_y) + '）')
# (c) 分級
v_share = 10 / 20; verb_share = 4 / 20
grade = '待觀察'
grade_note = 'VERIFIED_TEXT 10／20＝50%（≥50% 成立）、NOT_SEARCHED 0（成立）；第三條「獨立複製一致」本子代理無法執行 → 依規則待觀察；複製通過即升為可用。另：新核實 6 起全部是 WebSearch 摘要，只計逐字原文時 4／20＝20%'
# 預測對照
y = lambda pair, a, b: sum(v for k, v in VX['yearly'][pair]['med'].items() if a <= int(k) <= b)
du = {c: {f: [r for r in D['union'] if r['cell'] == c and r['field'] == f][0]['delta'] for f in ('exc_med', 'exc_Y19on_med', 'exc_1419_med')} for c in CELLS}
PRED = dict(
  P1=R['PREREG']['predictions']['P1']['verdict'], P2=R['PREREG']['predictions']['P2']['verdict'],
  P3=dict(main=mn, main_in_range=bool(1.30 <= mn <= 1.65), lo=lo, lo_in_range=bool(-0.50 <= lo <= 0.10), lo_gt0=bool(lo > 0),
          text=f"主口徑 {mn:.3f}（區間 1.30～1.65：{'成立' if 1.30 <= mn <= 1.65 else '不成立'}）；下界 {lo:.3f}（區間 −0.50～+0.10：{'成立' if -0.5 <= lo <= 0.1 else '不成立'}）；下界 > 0：{'是（與「機率 ≤30%」的事前判斷方向相反）' if lo > 0 else '否'}"),
  P4=dict(slots='成立（11／10，逐槽相同）', mean_r_out=desc['mean_r_out'], in_minus_out_pp=desc['sum_in_minus_out_pp'], in_minus_out_2019_pp=desc['sum_in_minus_out_pp_2019on'],
          port_by_year=desc['by_year'], union_C0=pap['ALL_ann'], union_2019=pap['SEG3']['d_gex'], replace_minus_union=S['C0|replace|freeze|normal']['ALL']['ann'] - c0u['ALL']['ann'],
          v7x=du, v7x_full_in=all(-1.6 <= du[c]['exc_med'] <= 0 for c in CELLS[:1]), v7x_19_in=bool(-4.5 <= du['normal|100萬']['exc_Y19on_med'] <= 0), v7x_1419_abs=abs(du['normal|100萬']['exc_1419_med']),
          same_dir_300_1000=bool(np.sign(du['normal|300萬']['exc_med']) == np.sign(du['normal|100萬']['exc_med']) == np.sign(du['normal|1000萬']['exc_med']))),
  P5=dict(overlap_share=0.605, d_noovl=S['C0|union_noovl|freeze|normal']['ALL']['ann_diff_vs_ref'], d_union=pap['ALL_ann'],
          text='前提不成立（overlap_proxy 不是代理推得的處置期；全部列起訖都是官方公告）；機械上：排除 overlap 列後 |Δ| 由 0.68 縮為 0.25pp'),
  P6=dict(n_with_lists=PK['n_with_lists'], n_exec_mdelay=PK['n_exec_mdelay_pos'], v7x_linked_frozen_max=PK['v7x_runs_linked_lists']['frozen_share_max'], census=PK['census_total_after'],
          text=f"不成立：26 個 pkl 都沒有名單或持股（o5 14 個只有淨值，無法判定；V7X 7 個以 acc 的併購延遲計數判定有持有到下市的部位；其餘 5 個非組合）；V7X 對應名單凍結資產最大 {PK['v7x_runs_linked_lists']['frozen_share_max']:.1%}（≥5% 成立）；名單層普查完整數 {PK['census_total_after']} 組（o17 文字 2,075 組，同一份 census.json 重數為 2,093，差 18 組未能解釋）"))
G = dict(G1='✘：V7X 100 萬一般 j p10 ' + f"{g('orig|v3_main', f='exc_p10'):+.2f}pp（v3 主口徑）／壓力 {g('orig|v3_main', 'stress|100萬'):+.2f}pp",
         G2=f"✘：V7X 全期 t {tmn:.3f}（v3）、2019 起 t {g('orig|v3_main', f='t_Y19on_med'):.3f}；紙上 C0 union t {c0u['ALL']['t_vs_0050']:.3f}",
         G3='✘：沒有新候選（S1），族群數不變', G4=f"✘：V7X 三段 j 中位數超額（v3）2006～2012 {g('orig|v3_main', f='exc_0612_med'):+.2f}、2013～2018 {g('orig|v3_main', f='exc_1318_med'):+.2f}、2019 起 {g('orig|v3_main', f='exc_19e_med'):+.2f}pp",
         G5='✘（處置欄更新）：處置遮罩官方化後紙上 2019 起 −2.65pp；可執行名單原已用 V4 官方名單（成交日口徑），新增訊號日口徑影響見 b；營收前視／2013 後存活者偏誤仍未排除（題 1）',
         G6='✘：本題未完成不同子代理的獨立複製', G7=f"✘：1,000 萬 v3 全期超額 {g('orig|v3_main', 'normal|1000萬'):+.2f}pp", G8='✘：前瞻 0 個月')
R['VERDICT'] = dict(a=dict(rewrite=a_ok, text=ann_a, lo=lo, hi=hi, main=mn, orig=o0, t=(tmn, tlo, thi, t0), lo_contrib_value_x_delta=contrib),
                    b=dict(annotate=bool(b_paper or b_v7x), paper=pap, v7x_rows=[r for r in D['union'] if r['exceed']], text=ann_b),
                    c=dict(grade=grade, note=grade_note), deltas=D, predictions=PRED, G=G)
RJ.write_text(json.dumps(R, ensure_ascii=False, indent=1, default=str), encoding='utf-8')
print(json.dumps({k: v for k, v in R['VERDICT'].items() if k != 'deltas'}, ensure_ascii=False, indent=1, default=str))
