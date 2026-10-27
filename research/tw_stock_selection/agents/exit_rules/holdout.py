#!/usr/bin/env python3
"""exit_rules 第 2 階段（判定寫入後才執行）：保留期 2021-03～2026-10 統計、2021 起逐年、2022 空頭、最終建議；另附 2006～2020 合併段（補充）。"""
import json, sys, pickle
from pathlib import Path
import pandas as pd
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.dont_write_bytecode = True
import xlib as X
L = X.L

res = json.loads((HERE / 'results.json').read_text(encoding='utf-8'))
assert 'RUN' in res and all('VERDICT' in s for s in res['RUN']['strategies'].values()), '判定尚未寫入'
ND = pd.read_parquet(HERE / 'navs.parquet')
INF = pickle.loads((HERE / 'infos.pkl').read_bytes())
HS = {'HOLD': ('2021-03-01', '2026-10-31'), 'SUPP_SEL+VAL': ('2006-04-01', '2020-12-31')}
EWCOL = {'D_weekly_reversal': ('w3_weekly_reversal/navs.parquet', 'EW_pool'),
         'D_high_volume': ('w3_high_volume_premium/navs.parquet', 'main_GKM_weekly_20d_EW'),
         'D_revenue_sue': ('w3_revenue_sue/nav.parquet', 'S_main_EW')}
out = {'strategies': {}}
for sn, S in res['RUN']['strategies'].items():
    ew = pd.read_parquet(X.AG / EWCOL[sn][0])[EWCOL[sn][1]] if sn in EWCOL else None
    hs = dict(HS)
    if sn == 'D_revenue_sue': hs['SUPP_SEL+VAL'] = ('2008-03-01', '2020-12-31')
    base = {ck: ND[f'{sn}|NONE|{ck}'].dropna() for ck in ('slip0.001', 'stress')}
    o = {'variants': {}}
    for v in S['variants']:
        w = {'segments': {}, 'paired_vs_NONE': {}, 'paired_vs_NONE_stress': {}, 'counts': {}}
        for sk, (a, b) in hs.items():
            for ck in ('slip0.001', 'stress'):
                w['segments'][f'{sk}|{ck}'] = X.seg_stats(ND[f'{sn}|{v}|{ck}'].dropna(), a, b, ew=ew)
            if v != 'NONE':
                w['paired_vs_NONE'][sk] = X.paired(ND[f'{sn}|{v}|slip0.001'].dropna(), base['slip0.001'], a, b)
                w['paired_vs_NONE_stress'][sk] = X.paired(ND[f'{sn}|{v}|stress'].dropna(), base['stress'], a, b)
            w['counts'][sk] = X.counts(INF[f'{sn}|{v}'], a, b)
        nv = ND[f'{sn}|{v}|slip0.001'].dropna()
        w['yearly_pct_2021plus'] = {y: x for y, x in X.yearly(nv).items() if y >= 2021}
        w['bear_2022'] = X.bear_rets(nv)['2022']
        o['variants'][v] = w
    sel = S['SELECTED']; hp = o['variants'][sel]['paired_vs_NONE']['HOLD']
    o['SELECTED'] = sel; o['SELECTED_HOLD'] = hp
    if sn.startswith('main'):
        usable = S['VERDICT'].startswith('可用')
        o['FINAL'] = f'建議採用 {sel}' if (usable and hp['diff_pct_m'] >= 0) else ('可用但保留期變差，不建議採用' if usable else '原版（不設停損／停利）')
    else:
        o['FINAL'] = S['VERDICT'] + f'；保留期配對 {hp["diff_pct_m"]}%/月（t {hp["t_paired_m"]}），只報告'
    out['strategies'][sn] = o
    print(sn, sel, hp, o['FINAL'], flush=True)
e = L.E50.reindex(ND.index)
out['0050'] = {sk: X.R.seg_stats(e.dropna(), a, b) for sk, (a, b) in HS.items()}
out['0050_yearly_2021plus'] = {y: x for y, x in X.yearly(e.loc['2006-01-01':]).items() if y >= 2021}
out['0050_bear_2022'] = X.bear_rets(e.loc['2006-01-01':])['2022']
res['HOLDOUT'] = out
(HERE / 'results.json').write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str), encoding='utf-8')
print('holdout done')
