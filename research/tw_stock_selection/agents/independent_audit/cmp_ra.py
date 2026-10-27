#!/usr/bin/env python3
"""rev_announce navs.parquet 用本稽核的 stats 在同一窗口重算，與本稽核重建（新/舊 value）比較。"""
import sys, json; from pathlib import Path; sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd, ia_core as K
ra = pd.read_parquet(K.HERE.parent / 'rev_announce/navs.parquet')
lh = pd.read_parquet(K.HERE.parent / 'revenue_longhistory/oos_nav.parquet')
S = K.Strategy(K.yoy_from_long(), '2005-01-01'); sd = S.sig_days + [S.end_day]
SEG = {'P1': ('2006-04-01', '2020-12-31'), 'P2': ('2021-01-01', '2026-10-12'), 'V24': ('2024-01-01', '2026-10-12')}
out = {}
for k in ['BASE_n10|slip0.001', 'BASE_n30|slip0.001', 'AB_n10|slip0.001', 'AB_n30|slip0.001', 'AB_worst_n10|slip0.001', 'AB_worst_n30|slip0.001', 'A_exFIN_n10|slip0.001', 'B_pit_worst_n10|slip0.001']:
    out['RA|' + k] = {s: K.short(K.stats(ra[k].dropna(), *w, sd, K.E50)) for s, w in SEG.items()}
for k in ['M2_n10_slip0.001', 'M2_n30_slip0.001']:
    out['LH|' + k] = {s: K.short(K.stats(lh[k].dropna(), *w, sd, K.E50)) for s, w in SEG.items()}
print('RA BASE n10 vs LH n10 max daily diff', float((ra['BASE_n10|slip0.001'].pct_change() - lh['M2_n10_slip0.001'].reindex(ra.index).pct_change()).abs().max()))
print('RA last days', ra['BASE_n10|slip0.001'].loc['2026-10-09':].round(3).to_dict())
for k, v in out.items(): print(k, {s: (x['CAGR'], x['t_0050']) for s, x in v.items()})
(K.HERE / 'results_cmp_ra.json').write_text(json.dumps(out, ensure_ascii=False, indent=1, default=float), encoding='utf-8')
