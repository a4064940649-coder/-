#!/usr/bin/env python3
"""探索與迭代（非閘門診斷）：建構規則三版 V1／V2／V3 只在「校準窗」2014Q1～2020Q4（閘門窗 2006～2013 之外）比較，
診斷＝(a) 全錨點留一（季末）、(b) 稀疏錨點留一（只留 12 月錨點，模擬下市股只有 goodinfo 年底股本的處境）。
選版規則（事前定）：(b) 合併 |對數誤差| 中位數最小者；差 < 0.1pp 時看 (a)，再相同取較簡單（V3＞V1＞V2）。
輸出 iterations.json（每版本結果、程式 sha、時間）。不讀 adj、ret；不算任何報酬。"""
import sys, json, time, datetime
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
import plib as P, pbuild as B
t0 = time.time()
A = B.load_anchor_rows()          # 含 o8 來源的新版
M = B.merge_anchors(A)
tlo, thi = P.ti('2005-06-01'), P.ti('2021-06-30')
mult = B.event_multipliers(tlo, thi)
ADV = P.adv120()
cps = [P.ti(q) for q in pd.date_range('2014-03-31', '2020-12-31', freq='QE')]
dec_only = lambda K: K[K.date.dt.month == 12]
out = dict(rule=__doc__, created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
           code_sha={'plib.py': P.sha(P.HERE / 'plib.py'), 'pbuild.py': P.sha(P.HERE / 'pbuild.py'), 's2_iterate.py': P.sha(__file__)},
           n_event_mult=int(sum(len(v) for v in mult.values())), versions={})
for v in ('V1', 'V2', 'V3'):
    Sf, _ = B.build(M, tlo, thi, mult, v)
    ra, da = B.loo_eval(M, cps, mult, v, Sf, ADV)
    rb, db = B.loo_eval(M, cps, mult, v, Sf, ADV, anc_filter=dec_only)
    def summ(r, d):
        ov = np.array([x['overlap'] for x in r]); e = np.array([x['abs_log'] for x in d])
        return dict(pooled_med_abs_log=round(float(np.median(e)), 5), pooled_p90=round(float(np.percentile(e, 90)), 4), overlap_med=float(np.median(ov)), overlap_min=int(ov.min()))
    out['versions'][v] = dict(a_all_anchors=summ(ra, da), b_dec_only=summ(rb, db),
                              b_by_q=[{k: x[k] for k in ('d', 'overlap', 'med_abs_log', 'n_truthless_in_top')} for x in rb])
    print(v, out['versions'][v]['a_all_anchors'], out['versions'][v]['b_dec_only'], round(time.time() - t0), flush=True)
V = out['versions']
order1 = sorted(V, key=lambda k: (round(V[k]['b_dec_only']['pooled_med_abs_log'], 3), V[k]['a_all_anchors']['pooled_med_abs_log'], {'V3': 0, 'V1': 1, 'V2': 2}[k]))
order2 = sorted(V, key=lambda k: (V[k]['b_dec_only']['pooled_p90'], V[k]['a_all_anchors']['pooled_p90'], {'V3': 0, 'V1': 1, 'V2': 2}[k]))
out['rounds'] = [
 dict(round=1, rule='(b) 中位數（取到 0.1pp）→ (a) 中位數 → 較簡單', selected=order1[0],
      finding='三版 (a)(b) 中位數全部 = 0（2014～2020 大型股多數一年內沒有股數變動），規則退化成「取最簡單」→ V3（純內插）。V3 在 (b) 的 p90 最差（3.18%），且 2006～2008 配股頻繁，純內插不適用。'),
 dict(round=2, rule='中位數退化 → 改看 (b) p90 → (a) p90（只用校準窗、非閘門診斷；屬於看過第 1 輪結果後的規則修改，照實揭露）', selected=order2[0])]
out['selected'] = order2[0]
out['n_versions_tried'] = 3
out['n_rounds'] = 2
out['note'] = '3 個建構版本、2 輪選版規則；校準窗 2014～2020 不與閘門窗 2006～2013 重疊。'
(P.HERE / 'iterations.json').write_text(P.dumps(out), encoding='utf-8')
print('selected', out['selected'], round(time.time() - t0), 's')
