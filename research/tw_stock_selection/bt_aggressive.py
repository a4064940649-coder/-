#!/usr/bin/env python3
"""積極配置（使用者接受大回撤、目標打敗大盤）：選股規則 30 檔／10 檔 × 處置股策略的組合，每月換股日再平衡。
2021-03～2026-10；處置股策略 2021-10 前視為現金。輸出 output/bt_aggressive.json"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
import bt_final_rule as fr
from btlib import ann, yearly, tstat
OUT = Path(__file__).resolve().parent / 'output'
d = pd.read_csv(OUT / 'disposal_daily_returns.csv', index_col=0, parse_dates=True)['0.005_cash']
eq = (1 + d).cumprod()
v = eq.reindex(pd.DatetimeIndex(fr.dates), method='ffill').fillna(1.0)
disp = (v.shift(-1) / v - 1).iloc[:-1]
r30, _, _, _ = fr.run(fr.yoy3, n=30); r10, _, _, _ = fr.run(fr.yoy3, n=10); B = fr.B
idx = r30.index.intersection(disp.index)
res = {}
for name, parts in {'0050': {'b': 1}, '選股30': {'r30': 1}, '選股10': {'r10': 1}, '處置股': {'disp': 1},
                    '選股30 50%＋處置 50%': {'r30': .5, 'disp': .5}, '選股10 50%＋處置 50%': {'r10': .5, 'disp': .5},
                    '選股30 70%＋處置 30%': {'r30': .7, 'disp': .3}, '選股10 70%＋處置 30%': {'r10': .7, 'disp': .3},
                    '0050 40%＋選股30 40%＋處置 20%': {'b': .4, 'r30': .4, 'disp': .2}}.items():
    src = {'b': B, 'r30': r30, 'r10': r10, 'disp': disp}
    r = sum(w * src[k].reindex(idx).fillna(0) for k, w in parts.items())
    a = ann(r, fr.dates); ex = r - B.reindex(idx)
    eqr = (1 + r).cumprod()
    res[name] = {'CAGR': a['CAGR'], 'maxDD': a['maxDD'], 'vol': a['vol'], 'excess_pct_m': round(float(ex.mean() * 100), 2), 't_vs_0050': tstat(ex),
                 'months_beat_0050': round(float((ex > 0).mean()), 3), 'worst_month_pct': round(float(r.min() * 100), 1),
                 '1M_becomes_NTD': round(float(eqr.iloc[-1] * 1e6)), 'yearly': yearly(r, fr.dates)}
(OUT / 'bt_aggressive.json').write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding='utf-8')
for k, x in res.items(): print(f"{k:28s} CAGR {x['CAGR']:.3f} MDD {x['maxDD']:.3f} vol {x['vol']:.3f} t {x['t_vs_0050']} beat {x['months_beat_0050']} worstM {x['worst_month_pct']} 100萬→{x['1M_becomes_NTD']:,}", x['yearly'])
