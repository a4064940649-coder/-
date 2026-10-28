#!/usr/bin/env python3
"""步驟 0（o5 教訓 1）：清點 V6 面板、V6 淨值（含 j-scan）、0050、各 raw 來源「連續 > 5 個交易日沒有資料」的區段 → explore_gaps.json。不看報酬。"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
import v7common as C
cal = C.taiex_cal()
cl = pd.read_parquet(C.V6DATA / 'close_raw.parquet')
nobs = cl.notna().sum(axis=1)
out = {'calendar': 'TAIEX（finlab taiex_twii.csv）交易日', 'rule': '連續缺 ≥ 6 個交易日（> 5）'}
out['taiex_calendar_contains_typhoon_days'] = {str(d.date()): bool(d in set(cal)) for d in C.TYPHOON}
out['panel_v6'] = {'range': [str(cl.index[0].date()), str(cl.index[-1].date())], 'n_dates': len(cl),
                   'gaps_gt5': C.gaps(cl.index, cal, '2004-01-02', cl.index[-1]),
                   'panel_dates_not_in_taiex': [str(d.date()) for d in cl.index if d not in set(cal)],
                   'days_nobs_lt300': {str(d.date()): int(n) for d, n in nobs[nobs < 300].items()}}
nv = pd.read_parquet(C.V6DIR / 'navs_v6.parquet')
out['navs_v6'] = {k: C.gaps(nv[k].dropna().index, cal, nv[k].dropna().index[0], nv[k].dropna().index[-1]) for k in ('AB_V6|normal', 'AB_V6_B10|normal', 'V6|normal')}
nj = pd.read_parquet(C.V6DIR / 'navs_jscan_v6.parquet')
out['navs_jscan_v6_any_gap'] = {k: C.gaps(nj[k].dropna().index, cal, nj[k].dropna().index[0], nj[k].dropna().index[-1]) for k in nj.columns if C.gaps(nj[k].dropna().index, cal, nj[k].dropna().index[0], nj[k].dropna().index[-1])}
e = C.e50()
out['e50'] = {'range': [str(e.index[0].date()), str(e.index[-1].date())], 'gaps_gt5': C.gaps(e.index, cal, '2004-01-02', e.index[-1])}
# per-stock long gaps inside window (listing holes): stocks with data both before and after but missing ≥ 6 trading days in between
w = cl.loc['2023-10-01':'2025-12-31']
holes = {}
for c in w.columns:
    x = w[c].dropna()
    if len(x) < 2: continue
    g = C.gaps(x.index, cal, x.index[0], x.index[-1])
    if g: holes[c] = g
out['panel_v6_per_stock_gaps_2023_10_2025_12'] = {'n_codes': len(holes), 'total_days': int(sum(x['n_trading_days'] for v in holes.values() for x in v)), 'detail': holes}
tdi = pd.DatetimeIndex(sorted(pd.Timestamp(p.name[4:14]) for p in (C.TASI / 'kbar').glob('kbar*.csv')))
out['tasi_kbar'] = {'n_files': len(tdi), 'weekend_files': int((tdi.dayofweek >= 5).sum()), 'gaps_gt5': C.gaps(tdi[tdi.dayofweek < 5], cal, tdi.min(), tdi.max())}
ndi = pd.DatetimeIndex(sorted({pd.Timestamp(p.name[:8]) for p in (C.TASI / '淨值').glob('*市淨值.csv')}))
out['tasi_pbr_twse_daily'] = {'n_files': len(ndi), 'range': [str(ndi.min().date()), str(ndi.max().date())], 'gaps_gt5': C.gaps(ndi, cal, ndi.min(), ndi.max())}
h = pd.read_csv(C.RAW / 'hschi' / 'stock_dayK' / '2330.csv', usecols=['date'], parse_dates=['date'])['date']
out['hschi_2330'] = {'range': [str(h.min().date()), str(h.max().date())], 'gaps_gt5': C.gaps(h, cal, h.min(), h.max())}
hh = pd.read_csv(C.HIST, usecols=['date'], parse_dates=['date'])['date'].drop_duplicates()
out['history_csv'] = {'range': [str(hh.min().date()), str(hh.max().date())], 'gaps_gt5': C.gaps(hh, cal, hh.min(), hh.max())}
C.jdump(out, C.HERE / 'explore_gaps.json')
print({k: (v.get('gaps_gt5') if isinstance(v, dict) else v) for k, v in out.items() if k != 'panel_v6_per_stock_gaps_2023_10_2025_12'})
print('per-stock holes', out['panel_v6_per_stock_gaps_2023_10_2025_12']['n_codes'], out['panel_v6_per_stock_gaps_2023_10_2025_12']['total_days'])
print(out['navs_v6'], list(out['navs_jscan_v6_any_gap'])[:5])
