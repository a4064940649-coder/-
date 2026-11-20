#!/usr/bin/env python3
"""稽核獨立重建（二）：0050 含息序列＝官方日 K 原始收盤 × 事件日累乘倍數。
事件：官方日 K 倉庫 exright（除息，倍數＝前收÷(前收−息值)＝前收÷參考價）＋ etfsplit（分割，倍數＝四捨五入到整數的 前收÷參考價）。
另以 TWT49U 封存交叉核對 0050 除息事件。輸出 data/e50_tr_mine.csv（date,close,factor,adj）、data/e50_compare_daily.csv、結果印出並寫 data/e50_build.json。"""
import json, glob
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent; D = HERE / 'data'
ENG = HERE.parent / 'eng_audit_fixes_2026-11' / 'e50'
JON = '/home/user/jonatw/finlab-txd/data/raw/etf_0050.csv'
out = {}
a = pd.read_csv(D / 'e50_raw_raw.csv', parse_dates=['date']).set_index('date')
b = pd.read_csv(D / 'e50_raw_tsd.csv', parse_dates=['date']).set_index('date')
ov = a.index.intersection(b.index)
out['raw_two_repos'] = dict(n_raw=len(a), n_tsd=len(b), overlap=len(ov), close_diff_days=int((a.loc[ov, 'close'] != b.loc[ov, 'close']).sum()),
                            open_diff_days=int(((a.loc[ov, 'open'] != b.loc[ov, 'open']) & ~(a.loc[ov, 'open'].isna() & b.loc[ov, 'open'].isna())).sum()),
                            only_tsd=[str(x.date()) for x in b.index.difference(a.index)][:30])
px = pd.concat([a, b[~b.index.isin(a.index)]]).sort_index()
px = px[px['close'].notna() & (px['close'] > 0)]
ex = pd.read_csv(D / 'ev_exright_tsd.csv', parse_dates=['date'])
sp = pd.read_csv(D / 'ev_etfsplit_tsd.csv', parse_dates=['date']); sp = sp[sp['stock_id'].astype(str) == '0050']
# TWT49U 封存核對
tw = pd.concat([pd.read_csv(f, dtype={'code': str}) for f in sorted(glob.glob('/home/user/-/research/tw_stock_selection/raw/twt49u/twse-exright/*/*.csv.gz'))])
tw = tw[tw['code'] == '0050'].copy(); tw['date'] = pd.to_datetime(tw['date']); tw = tw.drop_duplicates('date', keep='last').set_index('date').sort_index()
tw15 = tw.loc['2015-01-01':]
m = ex.set_index('date')
out['events_vs_twt49u'] = dict(n_repo=len(m), n_archive_2015on=len(tw15), only_repo=[str(x.date()) for x in m.index.difference(tw15.index)],
                               only_archive=[str(x.date()) for x in tw15.index.difference(m.index)],
                               field_diff=int(sum((abs(m.loc[d, 'pre_close'] - tw15.loc[d, 'prev_close']) > 1e-9) or (abs(m.loc[d, 'ref_price'] - tw15.loc[d, 'ref_price']) > 1e-9) or (abs(m.loc[d, 'value'] - tw15.loc[d, 'value']) > 1e-9)
                                              for d in m.index.intersection(tw15.index))))
# 事件用「兩來源聯集」：倉庫讀不到的日子由封存補
evs = {}
for d, r in tw15.iterrows(): evs[d] = ('息', float(r['prev_close']), float(r['ref_price']), float(r['value']))
for d, r in m.iterrows(): evs[d] = ('息', float(r['pre_close']), float(r['ref_price']), float(r['value']))
for _, r in sp.iterrows(): evs[r['date']] = ('分割', float(r['pre_close']), float(r['ref_price']), np.nan)
fac = pd.Series(1.0, index=px.index); rows = []
for d in sorted(evs):
    k, pc, ref, val = evs[d]
    prev = px['close'].loc[:d - pd.Timedelta(days=1)]
    pc_k = float(prev.iloc[-1]) if len(prev) else np.nan
    f = float(round(pc / ref)) if k == '分割' else pc / (pc - val)
    rows.append(dict(date=str(d.date()), kind=k, pre_close=pc, ref=ref, value=val, k_prev_close=pc_k, prev_date=str(prev.index[-1].date()) if len(prev) else None,
                     pre_close_matches_k=bool(abs(pc - pc_k) < 1e-9), ref_eq_pre_minus_value=(bool(abs(pc - val - ref) < 0.0051) if k == '息' else None), factor=f, in_k=bool(d in px.index)))
    if d in fac.index: fac.loc[d] = f
out['events'] = rows
px['factor'] = fac; px['adj'] = px['close'] * fac.cumprod()
px[['open', 'close', 'factor', 'adj']].to_csv(D / 'e50_tr_mine.csv')
# 逐日比對
jon = pd.read_csv(JON, parse_dates=['date']).set_index('date')['adj']
eng = pd.read_csv(ENG / 'e50_tr_official.csv', parse_dates=['date']).set_index('date')
out['eng_cols'] = list(eng.columns)
ecol = [c for c in eng.columns if 'tr' in c.lower() or 'adj' in c.lower()][0]
S, E_ = pd.Timestamp('2015-07-01'), pd.Timestamp('2026-10-22')
cmp_ = pd.DataFrame({'close': px['close'], 'mine': px['adj'], 'eng': eng[ecol], 'jon': jon}).loc[S:E_]
out['index_check'] = dict(n=len(cmp_), mine_nan=int(cmp_['mine'].isna().sum()), eng_nan=int(cmp_['eng'].isna().sum()), jon_nan=int(cmp_['jon'].isna().sum()),
                          jon_days_not_in_k=[str(x.date()) for x in jon.loc[S:E_].index.difference(px.index)])
c = cmp_.dropna(subset=['mine']).copy()
for k in ('mine', 'eng', 'jon'):
    c['r_' + k] = c[k].ffill().pct_change()
c['cum_jon_vs_mine_pct'] = (c['jon'].ffill() / c['mine'] / (c['jon'].iloc[0] / c['mine'].iloc[0]) - 1) * 100
c['cum_eng_vs_mine_pct'] = (c['eng'] / c['mine'] / (c['eng'].iloc[0] / c['mine'].iloc[0]) - 1) * 100
c['dr_jon_bp'] = (c['r_jon'] - c['r_mine']) * 1e4; c['dr_eng_bp'] = (c['r_eng'] - c['r_mine']) * 1e4
c.round(6).to_csv(D / 'e50_compare_daily.csv')
big = c[c['dr_jon_bp'].abs() > 2]
out['mine_vs_eng'] = dict(max_abs_cum_pct=float(c['cum_eng_vs_mine_pct'].abs().max()), max_abs_daily_bp=float(c['dr_eng_bp'].abs().max()))
out['jon_vs_mine'] = dict(days_daily_gt_2bp=int(len(big)), first=str(big.index[0].date()) if len(big) else None,
                          list=[dict(date=str(d.date()), close=float(r['close']), dr_bp=round(float(r['dr_jon_bp']), 2), cum_pct=round(float(r['cum_jon_vs_mine_pct']), 4),
                                     implied_jon_close=round(float(c['close'].shift(1).loc[d] * (1 + r['r_jon']) / (1.0 if d not in evs else fac.loc[d])), 3)) for d, r in big.iterrows()],
                          cum_pct_at={d: round(float(c['cum_jon_vs_mine_pct'].loc[:d].iloc[-1]), 4) for d in ('2024-12-31', '2025-06-17', '2025-07-18', '2025-07-21', '2025-12-31', '2026-07-20', '2026-07-21', '2026-08-11', '2026-08-12', '2026-10-22')},
                          max_abs_cum_by_year={int(y): round(float(g.abs().max()), 5) for y, g in c['cum_jon_vs_mine_pct'].groupby(c.index.year)})
def yearly(s):
    yr = s.groupby(s.index.year).last(); p = s.iloc[0]; o = {}
    for y in yr.index: o[int(y)] = round(float(yr[y] / p - 1) * 100, 4); p = yr[y]
    return o
out['yearly_pct'] = dict(mine=yearly(c['mine']), jon=yearly(c['jon'].ffill()), eng=yearly(c['eng'].ffill()))
json.dump(out, open(D / 'e50_build.json', 'w'), ensure_ascii=False, indent=1, default=str)
print(json.dumps(out, ensure_ascii=False, indent=1, default=str))
