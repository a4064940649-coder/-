#!/usr/bin/env python3
"""S85 程式自我檢查（不產生新結果）：(1) B 結構把候選換成 0050 本身、κ＝0 → 必須等於現行；(2) A 結構 x＝1 → 等於現行；
(3) A 結構候選＝現行自己 → 等於現行（κ＝0）；(4) 驗證格重算與 valid_results.json 一致；(5) 事前預測對照。寫 checks.json。"""
import sys, json
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
import s85lib as L
sd, bench, C, th = L.load()
s = sd['nav'].loc[L.START:]; s = s / s.iloc[0]
o = {}
o['B_cand_is_0050_kappa0_maxrel'] = float((L.blend_B(sd, bench, bench, 1.0, kappa=0.0) / s - 1).abs().max())
o['A_x1_maxrel'] = float((L.blend_A(sd['nav'], C['DISP_D'], 1.0) / s - 1).abs().max())
o['A_self_kappa0_maxrel'] = float((L.blend_A(sd['nav'], sd['nav'], 0.5, kappa=0.0) / s - 1).abs().max())
R = json.loads((L.H / 'valid_results.json').read_text())
k = 'DISP_D|A70'; v = L.stats(L.blend_A(sd['nav'], C['DISP_D'], 0.7), bench, th, L.VALID)
o['recompute_DISP_D_A70_valid_cagr_diff'] = float(abs(v['cagr'] - R['cells'][k]['valid']['cagr']))
T = json.loads((L.H / 'train_results.json').read_text())
o['train_vs_validate_same_train_numbers_maxdiff'] = float(max(abs(T['cells'][c]['cagr'] - R['cells'][c]['train']['cagr']) for c in T['cells']))
tr = sd.copy(); tr = pd.read_csv(L.H / 'surge_trades.csv.gz')
o['top5pct_trades_by_exit_year'] = tr[tr.top5pct].assign(y=lambda d: d.exit_date.str[:4]).groupby('y').size().to_dict()
o['top5pct_in_train_window_exits'] = int((tr[tr.top5pct].exit_date <= '2021-12-31').sum())
cc = R['corr']
o['predictions'] = {
 'P1 相關 0.2～0.6': {c: round(cc[c]['corr_full'], 3) for c in L.COMPARABLE},
 'P2 A 結構年化都低超過 2 點（訓練／驗證 d_cagr）': {c: [round((R['cells'][c]['train']['cagr'] - R['current']['train']['cagr']) * 100, 2), round((R['cells'][c]['valid']['cagr'] - R['current']['valid']['cagr']) * 100, 2)] for c in R['cells'] if '|A' in c and R['cells'][c]['group'] == 'comparable'},
 'P3 DISP_D B 訓練合格': {c: R['cells'][c]['train']['passed'] for c in ('DISP_D|B50', 'DISP_D|B100')},
 'P4 禁買前 5 趴後各格驗證段年化（現行 ' + str(round(R['current']['valid']['ex_top5pct_cagr'] * 100, 2)) + '）': {c: round(R['cells'][c]['valid']['ex_top5pct_cagr'] * 100, 2) for c in R['cells'] if R['cells'][c]['group'] == 'comparable'}}
(L.H / 'checks.json').write_text(json.dumps(o, ensure_ascii=False, indent=1)); print(json.dumps(o, ensure_ascii=False, indent=1))
