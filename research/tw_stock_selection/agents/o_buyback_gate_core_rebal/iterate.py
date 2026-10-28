# -*- coding: utf-8 -*-
"""迭代（只用 SEL）：it1 主格點 27 格；it2 轉移時點改成交日收盤；it3 帶狀「回到帶邊」；it4 動能回看 6／24；
it5 年／半年／季的錨定月份；it6 壓力成本；it7 曝險對齊（漂移版平均權重下的每月版）。結果寫 iterations.json。"""
import sys, json
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import numpy as np, pandas as pd
import rlib as L
a, b = L.PER['SEL']
W = [0.2, 0.3, 0.4]
def cell(w, rule, **kw):
    r = L.sim(w, rule, a, b, **kw); s = L.stats(r['nav'])
    return r, dict(CAGR=s['CAGR'], MDD=s['MDD'], TE=s['TE'], t_vs0050=s['t_vs0050'], longest_lag_m=s['longest_lag_m'],
                   roll36_lag_frac=s['roll36_lag_frac'], n_trades=r['n_trades'], turnover_sum=r['turnover'],
                   wB_mean=float(r['wb'].mean()) if len(r['wb']) else w, wB_end=float(r['wb'].iloc[-1]) if len(r['wb']) else w)
IT = []
def grid(tag, desc, rules, ws=W, **kw):
    out = {}
    for w in ws:
        base, _ = None, None
        mref = L.sim(w, 'M', a, b, **{k: v for k, v in kw.items() if k in ('cost', 'exec_mode', 'B')})['nav']
        for ru in rules:
            r, s = cell(w, ru, **kw)
            s['paired_vs_M'] = L.paired(r['nav'], mref)
            out[f'{int(w*100)}|{ru}'] = s
    IT.append(dict(it=tag, desc=desc, period='SEL', cells=out))
    print(tag, desc)
    for k, v in out.items():
        print('  ', k, round(v['CAGR'], 2), 'TE', round(v['TE'], 2), 'MDD', round(v['MDD'], 1), 'n', v['n_trades'], 'vsM', round(v['paired_vs_M']['ann_diff_pp'], 3), round(v['paired_vs_M']['t'], 2), 'lag', v['longest_lag_m'])
    return out
g1 = grid('it1', '主格點：w∈{20,30,40}×9 條規則，一般成本，訊號日收盤轉移', L.RULES)
g2 = grid('it2', '轉移改在換股成交日收盤記帳（引擎敏感度，非新規則）', L.RULES, exec_mode='exec')
g3 = grid('it3', '帶狀超出時只回到帶邊（新規則 3×3＝9 格）', ['B5', 'B10', 'B15'], band_to='edge')
o4 = {}
for n in (6, 24):
    o4[n] = grid(f'it4_mom{n}', f'動能條件回看 {n} 個訊號窗（新規則 3 格）', ['MOM12'], mom_n=n)
# it5 錨定月份
anc = {}
for w in W:
    mref = L.sim(w, 'M', a, b)['nav']
    for ru, sets in (('A', [{m} for m in range(1, 13)]), ('H', [{m, m + 6} for m in range(1, 7)]), ('Q', [{m, m + 3, m + 6, m + 9} for m in range(1, 4)])):
        vals = []
        for ms in sets:
            r = L.sim(w, ru, a, b, months=ms)
            vals.append(dict(months=sorted(ms), ann_diff_vs_M=L.paired(r['nav'], mref)['ann_diff_pp'], CAGR=L.cagr(r['nav']) * 100))
        anc[f'{int(w*100)}|{ru}'] = dict(each=vals, min=min(v['ann_diff_vs_M'] for v in vals), max=max(v['ann_diff_vs_M'] for v in vals),
                                         median=float(np.median([v['ann_diff_vs_M'] for v in vals])))
        print('anchor', w, ru, [round(v['ann_diff_vs_M'], 3) for v in vals])
IT.append(dict(it='it5', desc='年／半年／季再平衡的錨定月份全部 12／6／3 種（新規則 21 格；正式只用 1 月錨）', period='SEL', cells=anc))
g6 = grid('it6', '壓力成本（主策略 stress3 淨值＋移轉每邊再加 0.3%）', L.RULES, cost='stress')
# it7 曝險對齊：NONE 的平均 0050 權重下的每月版
ex = {}
for w in W:
    rN = L.sim(w, 'NONE', a, b)
    A = L.bench('MAIN|normal', a, b); B = L.bench('E50', a, b)
    x = (1 - w) * A; y = w * B; wb = (y / (x + y))
    wbar = float(wb.mean())
    rM = L.sim(wbar, 'M', a, b)
    ex[int(w * 100)] = dict(NONE_wB_daily_mean=wbar, NONE_CAGR=L.cagr(rN['nav']) * 100, M_at_mean_w_CAGR=L.cagr(rM['nav']) * 100,
                            diff_pp=(L.cagr(rN['nav']) - L.cagr(rM['nav'])) * 100, NONE_TE=L.stats(rN['nav'])['TE'], M_at_mean_w_TE=L.stats(rM['nav'])['TE'],
                            NONE_MDD=L.mdd(rN['nav']) * 100, M_at_mean_w_MDD=L.mdd(rM['nav']) * 100)
    print('exposure-matched', w, ex[int(w * 100)])
IT.append(dict(it='it7', desc='曝險對齊診斷：NONE 對「每月、權重＝NONE 的期間平均 0050 權重」（非新規則，機制檢查）', period='SEL', cells=ex))
n_rules = 27 + 9 + 6 + 21
meta = dict(n_iterations=len(IT), n_distinct_rule_variants_tested=n_rules, note='it2、it6、it7 是同規則的引擎／成本敏感度與機制診斷，不另計規則數；正式選版只在 plan.json 登記的 9 條規則中進行')
(HERE / 'iterations.json').write_text(json.dumps(dict(meta=meta, iterations=IT), ensure_ascii=False, indent=1, default=float), encoding='utf-8')
