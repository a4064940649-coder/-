#!/usr/bin/env python3
"""第 0 步：資料閘門（不看任何報酬）。用法：python3 -I -B s0_gate.py → data_gate.json"""
import sys, json, datetime
sys.dont_write_bytecode = True
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import numpy as np, pandas as pd
import core as K

L = K.load_list()
U = K.universe(L)
out = {'time_utc': datetime.datetime.utcnow().isoformat(timespec='seconds') + 'Z'}
out['list'] = {'rows_all': int(len(L)), 'rows_common_stock': int(len(U)),
               'excluded_non_common(權證/ETF/DR/特別股)': int((~L['is_common']).sum())}
# 逐年（依 start 年）
U['year'] = U['start'].dt.year
U['has_px'] = [bool(r.in_D and r.s_i < K.T and K.TR[r.s_i, K.CI[r.code]]) for r in U.itertuples()]
U['has_any_px'] = [bool(r.in_D and K.TR[max(r.s_i - 5, 0):min(r.e_i + 1, K.T), K.CI[r.code]].any()) if r.in_D else False for r in U.itertuples()]
yr = U.groupby('year').agg(n=('code', 'size'), in_panel=('in_D', 'sum'), px_at_start=('has_px', 'sum'),
                           px_in_window=('has_any_px', 'sum'), first=('second30', lambda s: int((~s).sum())),
                           prepay_only=('regime', lambda s: int((s == 'prepay_only').sum())))
out['by_year'] = {int(k): {kk: int(vv) for kk, vv in v.items()} for k, v in yr.iterrows()}
def cov(a, b):
    m = (U['start'] >= pd.Timestamp(a)) & (U['start'] <= pd.Timestamp(b))
    x = U[m]
    return {'n': int(len(x)), 'cov_px_in_window': round(float(x['has_any_px'].mean()), 4) if len(x) else None,
            'cov_px_at_start': round(float(x['has_px'].mean()), 4) if len(x) else None}
out['coverage'] = {'OOS_2006-04_2021-09': cov('2006-04-01', '2021-09-30'), 'PREPAY_2004_2006-03': cov('2004-01-01', '2006-03-31'),
                   '2003(面板前)': cov('2002-12-01', '2003-12-31'), 'SEL_2021-10_2023': cov('2021-10-01', '2023-12-31'),
                   'DESC_2024_2026': cov('2024-01-01', '2026-10-22')}
out['coverage_gate_95pct_OOS'] = out['coverage']['OOS_2006-04_2021-09']['cov_px_in_window'] >= 0.95
miss = U[~U['has_any_px'] & (U['start'] >= '2004-01-01')]
out['missing_px_events'] = miss[['start', 'code', 'name', 'kind', 'in_D']].astype(str).to_dict('records')
# 公告日在處置第 1 日之前（PIT）
d = (U['s_i'] - U['ann_i'])
out['announce_lead_tradingdays'] = {str(k): int(v) for k, v in d.value_counts().sort_index().items()}
out['announce_not_before_start'] = int((U['ann_i'] >= U['s_i']).sum())
# 主規則下的事件數（不看報酬）
for name, per in (('OOS', 'OOS_2006-04_2021-09'), ('PREPAY', 'PREPAY_2004_2006-03'), ('SEL', 'SEL_2021-10_2023'), ('DESC', 'DESC_2024_2026-10')):
    df, st = K.build({}, U, per)
    out[f'rule_counts_{name}'] = {'kept_in_period': int(len(df)), 'by_year': {int(k): int(v) for k, v in df.groupby(df['date'].dt.year).size().items()},
                                  'second': int(df['second'].sum()) if len(df) else 0, 'delayed': int(df['delayed'].sum()) if len(df) else 0,
                                  'delisted_in_hold': int(df['delisted'].sum()) if len(df) else 0}
    if name == 'OOS':
        out['rule_filter_counts_all_years'] = st
        dl = df[df['delisted']]
        out['delisted_in_hold_events_OOS'] = dl[['date', 'exit_date', 'code']].astype(str).to_dict('records')
        # 2019-05～2023 期間下市股事件（逐筆列）：面板最後有價日在 2019-05～2023
        last = pd.Series({c: K.IDX[K.LASTV[j]] if K.LASTV[j] >= 0 else pd.NaT for c, j in K.CI.items()})
        U2 = U[(U['start'] >= '2019-05-01') & (U['start'] <= '2023-12-31') & U['in_D']].copy()
        U2['last_px'] = U2['code'].map(last)
        dd = U2[U2['last_px'] < pd.Timestamp('2026-10-01')]
        out['events_2019-05_2023_of_later_delisted_stocks'] = dd[['start', 'end', 'code', 'name', 'last_px']].astype(str).to_dict('records')
# 與 disposal_bear_longhistory 近似事件重疊（那批報酬已看過）
PX = pd.read_csv(K.PROXY_F, dtype={'code': str}, parse_dates=['start', 'end'])
pxg = {c: g for c, g in PX.groupby('code')}
def ov(r):
    g = pxg.get(r.code)
    if g is None: return False
    return bool(((g['start'] <= r.end) & (g['end'] >= r.start)).any())
U['overlap_proxy'] = [ov(r) for r in U.itertuples()]
m = (U['start'] >= '2006-04-01') & (U['start'] <= '2021-09-30')
out['overlap_with_proxy_OOS'] = {'n_overlap': int(U.loc[m, 'overlap_proxy'].sum()), 'n_total': int(m.sum()),
                                 'share': round(float(U.loc[m, 'overlap_proxy'].mean()), 3)}
U[['code', 'ordr', 'start', 'end', 'overlap_proxy', 'second30', 'take_R1']].to_csv(HERE / 'universe_flags.csv', index=False)
# 資料凍結
files = {'list': K.LIST_F, 'proxy': K.PROXY_F, 'x_cache': K.D.CACHE, 'taiex': K.TAIEX_F, 'e50': K.E50_F,
         'model_spec7': K.AG / 'o_v7_listing_exec_engine' / 'data' / 'model_spec7.json', 'core.py': HERE / 'core.py'}
out['data_freeze_sha256'] = {k: K.sha(v) for k, v in files.items()}
(HERE / 'data_gate.json').write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str), encoding='utf-8')
print(json.dumps({k: out[k] for k in ('list', 'coverage', 'coverage_gate_95pct_OOS', 'announce_lead_tradingdays', 'announce_not_before_start',
                                       'overlap_with_proxy_OOS', 'rule_filter_counts_all_years')}, ensure_ascii=False, indent=1))
for n in ('OOS', 'PREPAY', 'SEL', 'DESC'): print(n, out[f'rule_counts_{n}'])
print('delisted OOS', len(out['delisted_in_hold_events_OOS']), 'events of later-delisted 2019-05..2023', len(out['events_2019-05_2023_of_later_delisted_stocks']))
print('missing', len(out['missing_px_events']))
