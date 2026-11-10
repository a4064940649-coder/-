#!/usr/bin/env python3
"""integrate_o19 判定（機械套用 PREREG D1～D3；只跑一次）。python3 -I -B decide.py"""
import json, hashlib
from pathlib import Path
import pandas as pd
HERE = Path(__file__).resolve().parent; RJ = HERE / 'results.json'; R = json.loads(RJ.read_text(encoding='utf-8'))
assert hashlib.sha256(json.dumps(R['PREREG'], ensure_ascii=False, sort_keys=True).encode()).hexdigest() == R['PREREG_sha256']
assert 'DECISION' not in R
P, X = R['RESULTS_PAPER'], R['RESULTS_V7X']
FILLS_P = ['ALLPIT_REAL_opt', 'ALLPIT_REAL_p90', 'ALLPIT_REAL_carry']; FILLS_X = ['REAL_opt', 'REAL_p90', 'REAL_carry']
def pw(dv, f, c='normal', per='ALL'): return min(P['stats'][f'{dv}|{k}|{c}'][per][f] for k in FILLS_P)
def xw(dv, cell, f): return min(X['table'][f'{dv}|{v}|{cell}'][f] for v in FILLS_X)
CELLS = {'紙上全期年化': lambda dv: pw(dv, 'ann'), '紙上全期 t': lambda dv: pw(dv, 't_vs_0050'),
         '紙上壓力全期年化': lambda dv: pw(dv, 'ann', 'stress'), '紙上 2019 起超額': lambda dv: pw(dv, 'gex', 'normal', 'SEG3'),
         'V7X 100萬一般 全期超額': lambda dv: xw(dv, 'normal|100萬|std', 'exc_med'), 'V7X 100萬一般 t': lambda dv: xw(dv, 'normal|100萬|std', 't_0050_med'),
         'V7X 100萬一般 j p10': lambda dv: xw(dv, 'normal|100萬|std', 'exc_p10'), 'V7X 2019 起年化': lambda dv: xw(dv, 'normal|100萬|std', 'CAGR_Y19on_med'),
         'V7X 2019 起超額': lambda dv: xw(dv, 'normal|100萬|std', 'exc_Y19on_med'), 'V7X 100萬壓力 全期超額': lambda dv: xw(dv, 'stress|100萬|std', 'exc_med'),
         'V7X 300萬一般 全期超額': lambda dv: xw(dv, 'normal|300萬|std', 'exc_med'), 'V7X 1000萬一般 全期超額': lambda dv: xw(dv, 'normal|1000萬|std', 'exc_med'),
         'V7X 30萬一般（標準）全期超額': lambda dv: xw(dv, 'normal|30萬|std', 'exc_med'), 'V7X 30萬整股限制 全期超額': lambda dv: xw(dv, 'normal|30萬|wholelot', 'exc_med'),
         'V7X 30萬壓力 全期超額': lambda dv: xw(dv, 'stress|30萬|std', 'exc_med')}
TBL = {}
for nm, fn in CELLS.items():
    base = fn('strict'); row = dict(strict_worst=round(base, 4))
    for dv in ('posthoc', 'I3a', 'I3b', 'I3c'):
        row[f'{dv}_delta'] = round(fn(dv) - base, 4)
    row['threshold'] = 0.05 if ' t' in nm else 0.3
    TBL[nm] = row
off_same = all(P['lists_sha'][f'strict|{k}'] == P['lists_sha'][f'official|{k}'] for k in ['C0'] + FILLS_P) and X['plans_changed'].get('strict', 0) == 0
D1_delta0 = all(abs(P['stats'][f'strict|{k}|{c}']['ALL']['diff_vs_ref_pp']) < 1e-12 for k in FILLS_P for c in ('normal', 'stress'))
D1 = dict(lists_identical=off_same, paper_all_delta_zero=D1_delta0, v7x_strict_is_reference=True,
          result='整合＝主策略本身，正式數字不變' if off_same and D1_delta0 else '檢查')
ex = {nm: abs(r['I3a_delta']) > r['threshold'] for nm, r in TBL.items() if not nm.startswith('V7X 30萬')}
D2 = dict(I3a_exceeds=ex, triggered=any(ex.values()),
          result=('加註「2019 處置清單不完整的可能影響」＝' + '、'.join(f'{k} {TBL[k]["I3a_delta"]:+.2f}' for k, v in ex.items() if v)) if any(ex.values()) else '加註「2019 不完整的合理影響 ≤ 0.3pp」',
          extreme_I3b_max_abs=max(abs(r['I3b_delta']) for r in TBL.values()), extreme_I3c_max_abs=max(abs(r['I3c_delta']) for r in TBL.values()))
bc = {}
for k, s in P['blends'].items(): bc['paper|' + k] = s['ALL']['corr_ref']
for k, s in X['blends'].items(): bc['v7x|' + k] = s['vs_main']['ALL']['corr']['med']
D3 = dict(min_corr=min(v for v in bc.values() if v is not None), corr=bc,
          result='不構成第二條腿（自混），不推薦任何比例' if min(v for v in bc.values() if v is not None) >= 0.99 else '檢查')
R = json.loads(RJ.read_text(encoding='utf-8')); assert 'DECISION' not in R
R['DECISION'] = dict(utc=pd.Timestamp.now('UTC').isoformat(), cells=TBL, D1=D1, D2=D2, D3=D3,
                     D4='題目維持「待觀察」；主策略 G1～G8 判定不變（不可投資）')
RJ.write_text(json.dumps(R, ensure_ascii=False, indent=1, default=str), encoding='utf-8')
print(json.dumps(R['DECISION'], ensure_ascii=False, indent=1)[:6000])
