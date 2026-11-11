#!/usr/bin/env python3
"""integrate_o20 判定（機械套用 PREREG D1、D1_lookahead、D3；只跑一次）。python3 -I -B decide.py"""
import json, hashlib
from pathlib import Path
import pandas as pd
HERE = Path(__file__).resolve().parent; RJ = HERE / 'results.json'; R = json.loads(RJ.read_text(encoding='utf-8'))
assert hashlib.sha256(json.dumps(R['PREREG'], ensure_ascii=False, sort_keys=True).encode()).hexdigest() == R['PREREG_sha256']
assert 'DECISION' not in R
P, X = R['RESULTS_PAPER'], R['RESULTS_V7X']
FP = ['ALLPIT_REAL_opt', 'ALLPIT_REAL_p90', 'ALLPIT_REAL_carry']; FX = ['REAL_opt', 'REAL_p90', 'REAL_carry']
def pw(dv, f, c='normal', per='ALL'):
    v = {k: P['stats'][f'{dv}|{k}|{c}'][per][f] for k in FP}; k = min(v, key=v.get); return v[k], k
def xw(dv, cell, f):
    v = {k: X['table'][f'{dv}|{k}|{cell}'][f] for k in FX}; k = min(v, key=v.get); return v[k], k
CELLS = {'紙上全期年化': lambda dv: pw(dv, 'ann'), '紙上全期 t': lambda dv: pw(dv, 't_vs_0050'),
         '紙上壓力全期年化': lambda dv: pw(dv, 'ann', 'stress'), '紙上壓力全期 t': lambda dv: pw(dv, 't_vs_0050', 'stress'),
         '紙上 2019 起超額': lambda dv: pw(dv, 'gex', 'normal', 'SEG3'),
         'V7X 100萬一般 全期超額': lambda dv: xw(dv, 'normal|100萬|std', 'exc_med'), 'V7X 100萬一般 t': lambda dv: xw(dv, 'normal|100萬|std', 't_0050_med'),
         'V7X 100萬一般 j p10': lambda dv: xw(dv, 'normal|100萬|std', 'exc_p10'), 'V7X 2019 起年化': lambda dv: xw(dv, 'normal|100萬|std', 'CAGR_Y19on_med'),
         'V7X 2019 起超額': lambda dv: xw(dv, 'normal|100萬|std', 'exc_Y19on_med'), 'V7X 100萬壓力 全期超額': lambda dv: xw(dv, 'stress|100萬|std', 'exc_med'),
         'V7X 300萬一般 全期超額': lambda dv: xw(dv, 'normal|300萬|std', 'exc_med'), 'V7X 1000萬一般 全期超額': lambda dv: xw(dv, 'normal|1000萬|std', 'exc_med'),
         'V7X 30萬一般（標準）全期超額': lambda dv: xw(dv, 'normal|30萬|std', 'exc_med'), 'V7X 30萬整股限制 全期超額': lambda dv: xw(dv, 'normal|30萬|wholelot', 'exc_med'),
         'V7X 30萬壓力 全期超額': lambda dv: xw(dv, 'stress|30萬|std', 'exc_med')}
TBL = {}
for nm, fn in CELLS.items():
    b, kb = fn('M1' if nm.startswith('紙上') else 'base'); m, km = fn('M2')
    thr = 0.05 if nm.endswith(' t') else 0.1
    TBL[nm] = dict(base_worst=round(b, 4), base_worst_variant=kb, M2_worst=round(m, 4), M2_worst_variant=km, delta=round(m - b, 4), threshold=thr,
                   exceeds=abs(m - b) > thr, worst_variant_changed=kb != km)
LA = R.get('LOOKAHEAD_CHECK2', {})
la_affects = bool((LA.get('n_affect_paper') or 0) + (LA.get('n_affect_v7x') or 0))
D1 = dict(cells=TBL, n_exceed=sum(v['exceeds'] for v in TBL.values()),
          verdict=('統一遮罩改變部分格（> 門檻）：該格並列 M2 統一口徑版並附歸因；不取代正式數字（現行遮罩無影響名單的前視；G6 角色分離不成立）。是否改採 M2 交研究中心核定。'
                   if any(v['exceeds'] for v in TBL.values()) and not la_affects else
                   ('前視且影響名單 → 原引擎錯，Δ > 0.1pp 的格以 M2 取代（待不同子代理覆核）' if la_affects else '統一遮罩不改變任何格（在門檻內）')),
          lookahead=dict(n_rows=LA.get('n_lookahead'), n_affect_paper=LA.get('n_affect_paper'), n_affect_v7x=LA.get('n_affect_v7x'), affects_lists=la_affects))
corrs_p = [P['blends'][k]['ALL']['corr_ref'] for k in P['blends']]
corrs_x = [X['blends'][k]['vs_main']['ALL']['corr']['med'] for k in X['blends']]
D3 = dict(min_corr_paper=min(corrs_p), min_corr_v7x=min(corrs_x), verdict='不構成第二條腿（自混），不推薦任何比例' if min(corrs_p + corrs_x) >= 0.99 else '相關 < 0.99：另看混合超額是否落在兩腿之間')
D4 = dict(topic_verdict='待觀察（不變）', G1_G8='全部不過（不變）', G6='✘ 維持（題目 A／B／C 由同一子代理擔任，PREREG (b)）', investable=False)
R = json.loads(RJ.read_text(encoding='utf-8')); R['DECISION'] = dict(utc=pd.Timestamp.now('UTC').isoformat(), D1=D1, D3=D3, D4=D4)
RJ.write_text(json.dumps(R, ensure_ascii=False, indent=1, default=str), encoding='utf-8')
print(json.dumps(R['DECISION'], ensure_ascii=False, indent=1))
