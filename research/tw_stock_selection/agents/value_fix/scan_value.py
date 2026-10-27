#!/usr/bin/env python3
"""全期掃描面板 value 的市場層級異常：
(a) 每日截面中位數 log(V_t ÷ 該股前後各 20 日中位數)（不含 t），列出 |偏離| 大的日子；
(b) 2020-10 起與當沖還原官方量逐日比對（每日截面中位數「面板量 ÷ 官方量」）。
用法：python3 -I -B scan_value.py [value.parquet 路徑]（預設原始面板）"""
import sys, json
from pathlib import Path
import numpy as np, pandas as pd

S = Path('/tmp/claude-0/-home-user--/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad')
P = S / 'panel'
OUT = Path(__file__).resolve().parent
vp = Path(sys.argv[1]) if len(sys.argv) > 1 else P / 'value.parquet'
tag = 'fixed' if 'fixed' in vp.name else 'orig'
V = pd.read_parquet(vp); C = pd.read_parquet(P / 'close_raw.parquet'); SRC = pd.read_parquet(P / 'src.parquet')
GOOD = V.notna().sum(axis=1) >= 300
Vg = V.loc[GOOD].where(V.loc[GOOD] > 0)
L = np.log(Vg)
# 前後各 20 日中位數（不含 t）
before = L.shift(1).rolling(20, min_periods=10).median()
after = L[::-1].shift(1).rolling(20, min_periods=10).median()[::-1]
nb = (before + after) / 2
J = (L - nb).median(axis=1)
src_mode = SRC.loc[GOOD].mode(axis=1)[0]
res = {'file': str(vp)}
thr = np.log(2.2)
fl = J[(J.abs() > thr)]
res['xsec_jump_days_ratio>2.2_or<0.455'] = {str(d.date()): {'ratio': round(float(np.exp(v)), 3), 'src': int(src_mode[d]), 'n': int(Vg.loc[d].notna().sum())} for d, v in fl.items()}
res['xsec_jump_quantiles_by_src'] = {int(s): {q: round(float(np.exp(J[src_mode == s].quantile(q))), 3) for q in (0.01, 0.5, 0.99)} for s in sorted(src_mode.dropna().unique())}
# 當沖比對
dt = pd.read_csv(S / 'scr_extra/daytrade_5y.csv.gz', dtype={'code': str}, parse_dates=['date'])
dt = dt[(dt['dt_vol'] >= 10) & (dt['dt_ratio'] > 0)].drop_duplicates(['date', 'code'])
dt['date'] = dt['date'].astype('datetime64[us]')
dt['off'] = dt['dt_vol'] / (dt['dt_ratio'] / 100) * 1000
pv = (V / C).where(V > 0).stack(); pv.index.names = ['date', 'code']
ss = SRC.stack(); ss.index.names = ['date', 'code']
m = dt.merge(pv.rename('pv').reset_index(), on=['date', 'code']).merge(ss.rename('src').reset_index(), on=['date', 'code'])
m['r'] = m['pv'] / m['off']
dr = m.groupby('date')['r'].median(); dsrc = m.groupby('date')['src'].agg(lambda s: int(s.mode()[0]))
res['daytrade_daily_median_ratio_by_src'] = {int(s): {'n_days': int((dsrc == s).sum()), 'p1': round(float(dr[dsrc == s].quantile(.01)), 3), 'median': round(float(dr[dsrc == s].median()), 3),
                                                     'p99': round(float(dr[dsrc == s].quantile(.99)), 3)} for s in sorted(dsrc.unique())}
res['daytrade_days_outside_0.85_1.15'] = {str(d.date()): [round(float(v), 3), int(dsrc[d])] for d, v in dr[(dr > 1.15) | (dr < 0.85)].items()}
# src 2 期間（k66）value 是官方成交金額（非 量×收盤），比值離散較大屬正常；另報 src 2 每檔每日的分布
res['row_level_ratio_by_src'] = {int(s): {'n': int(len(g)), 'p10': round(float(g['r'].quantile(.1)), 3), 'median': round(float(g['r'].median()), 3), 'p90': round(float(g['r'].quantile(.9)), 3)}
                                 for s, g in m.groupby('src')}
(OUT / f'results_scan_{tag}.json').write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding='utf-8')
print(json.dumps(res, ensure_ascii=False)[:6000])
