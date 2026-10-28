#!/usr/bin/env python3
"""步驟 0（o5 教訓 1）：任何計算前，先清點 V4 面板、V4 淨值、0050 序列中「連續 > 5 個交易日沒有資料」的區段，
以及面板每日有值檔數 < 300 的日子 → explore_gaps.json。不看任何報酬。"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
import v6common as C

cal = C.taiex_cal()
P = C.load_panel(fields=['close_raw', 'src'])
cl = P['close_raw']
nobs = cl.notna().sum(axis=1)
navs = pd.read_parquet(C.V4DIR / 'navs_v4.parquet')
e = C.e50()
out = {'calendar': 'TAIEX（finlab taiex_twii.csv）交易日', 'rule': '連續缺 ≥ 6 個交易日（> 5）'}
out['panel_v4'] = {'range': [str(cl.index[0].date()), str(cl.index[-1].date())], 'n_dates': len(cl),
                   'gaps_gt5': C.gaps(cl.index, cal, '2004-01-02', cl.index[-1]),
                   'panel_dates_not_in_taiex': [str(d.date()) for d in cl.index if d not in set(cal)],
                   'days_nobs_lt300': {str(d.date()): int(n) for d, n in nobs[nobs < 300].items()}}
# V4 淨值（主欄）
nv = navs['AB_V4|normal'].dropna()
out['navs_v4_AB'] = {'range': [str(nv.index[0].date()), str(nv.index[-1].date())], 'gaps_gt5': C.gaps(nv.index, cal, nv.index[0], nv.index[-1])}
out['e50'] = {'range': [str(e.index[0].date()), str(e.index[-1].date())], 'gaps_gt5': C.gaps(e.index, cal, '2004-01-02', e.index[-1]),
              'has_2024_01': int(((e.index >= '2024-01-02') & (e.index <= '2024-02-14')).sum()),
              'taiex_days_2024_01_02_to_02_14': int(((cal >= '2024-01-02') & (cal <= '2024-02-14')).sum())}
# 每日有值檔數（逐月最小值）在 2023-10～2025-12
nm = nobs['2023-10':'2025-12']
out['panel_nobs_monthly_min_2023_10_2025_12'] = {str(k): int(v) for k, v in nm.groupby(nm.index.to_period('M')).min().items()}
# 各來源的跳日（只看日期，不看價格）
tasi_days = sorted(pd.Timestamp(p.name[4:14]) for p in C.TASI.glob('kbar*.csv'))
tdi = pd.DatetimeIndex(tasi_days)
out['tasi_kbar'] = {'n_files': len(tasi_days), 'weekend_files': int((tdi.dayofweek >= 5).sum()),
                    'gaps_gt5': C.gaps(tdi[tdi.dayofweek < 5], cal, tdi.min(), tdi.max())}
h = pd.read_csv(C.HSCHI / '2330.csv', usecols=['date'], parse_dates=['date'])['date']
out['hschi_2330'] = {'range': [str(h.min().date()), str(h.max().date())], 'gaps_gt5': C.gaps(h, cal, h.min(), h.max())}
C.jdump(out, C.HERE / 'explore_gaps.json')
print({k: (v['gaps_gt5'] if isinstance(v, dict) and 'gaps_gt5' in v else '') for k, v in out.items()})
print('nobs<300', out['panel_v4']['days_nobs_lt300'])
