#!/usr/bin/env python3
"""0050 含息：自建（官方日 K＋事件）對現用（jonatw adj），逐日比對＋對正式數字的影響。只讀正式檔，輸出到 e50/。
python3 -I -B compare_e50.py"""
import json, gzip
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent; OUT = HERE / 'e50'
AG = HERE.parent; SITE = Path('/home/user/-/xq_scripts/site')
R = {}
new = pd.read_csv(OUT / 'e50_tr_official.csv', parse_dates=['date']).set_index('date')
old = pd.read_csv('/home/user/jonatw/finlab-txd/data/raw/etf_0050.csv', parse_dates=['date']).set_index('date')['adj'].sort_index()
nav = pd.read_csv(SITE / 'navs_official.csv.gz', parse_dates=['date']).set_index('date')
IDX = nav.index; D0, DE = IDX[0], IDX[-1]
R['period'] = [str(D0.date()), str(DE.date())]; R['jonatw_last'] = str(old.index.max().date()); R['jonatw_dup_dates'] = int(old.index.duplicated().sum())
R['jonatw_non_trading_rows_in_period'] = [str(d.date()) for d in old.loc[D0:DE].index.difference(new.index)]
# TWT49U 封存交叉核對 0050 事件
ev = pd.concat([pd.read_csv(f, dtype={'code': str}) for f in sorted((AG.parent / 'raw/twt49u/twse-exright').glob('*/*.csv.gz'))])
e49 = ev[(ev['code'] == '0050') & (ev['date'] >= '2015-07-01')]
mine = json.loads((OUT / 'e50_events.json').read_text())
m_div = {e['date']: (e['pre_close'], e['ref_price']) for e in mine['events'] if e['kind'] == 'exright'}
a_div = {r.date: (r.prev_close, r.ref_price) for r in e49.itertuples()}
R['events_vs_twt49u_archive'] = dict(n_mine=len(m_div), n_archive=len(a_div), identical=m_div == a_div,
                                     only_mine=sorted(set(m_div) - set(a_div)), only_archive=sorted(set(a_div) - set(m_div)))
# 引擎口徑：對交易日軸 ffill
E_old = old.reindex(IDX.union(old.index)).ffill().reindex(IDX)
E_new = new['tr_close'].reindex(IDX)
assert E_new.notna().all() and E_old.notna().all()
a = E_old / E_old.iloc[0]; b = E_new / E_new.iloc[0]
rel = (a / b - 1)                                      # 現用 ÷ 自建 − 1（累計）
dr = (a.pct_change() - b.pct_change()).fillna(0)       # 當日報酬差
cmp = pd.DataFrame({'jonatw_norm': a, 'official_norm': b, 'cum_diff_pct': rel * 100, 'daily_ret_diff_pp': dr * 100,
                    'nav0050_now': nav['0050'], 'nav0050_new': (b * 1e6).round(2)})
cmp.index.name = 'date'; cmp.to_csv(OUT / 'e50_daily_compare.csv.gz', float_format='%.8f')
big = dr[dr.abs() > 2e-4]
R['daily'] = dict(n_days=int(len(IDX)), max_abs_cum_diff_pct=float(rel.abs().max() * 100), cum_diff_end_pct=float(rel.iloc[-1] * 100),
                  n_days_ret_diff_gt_0p02pp=int(len(big)), first_such_day=str(big.index[0].date()) if len(big) else None,
                  by_year={int(y): dict(max_abs_cum_diff_pct=round(float(g.abs().max() * 100), 5), n_days_ret_diff_gt_0p02pp=int((dr[g.index].abs() > 2e-4).sum()),
                                        max_abs_daily_ret_diff_pp=round(float(dr[g.index].abs().max() * 100), 5)) for y, g in rel.groupby(rel.index.year)},
                  days_ret_diff_gt_0p02pp=[dict(date=str(d.date()), diff_pp=round(float(v * 100), 4), jonatw_ret_pct=round(float(a.pct_change()[d] * 100), 4), official_ret_pct=round(float(b.pct_change()[d] * 100), 4)) for d, v in big.items()])
R['cum_diff_pct_at'] = {d: round(float(rel.loc[:d].iloc[-1] * 100), 4) for d in ('2024-12-31', '2025-12-31', '2026-08-11', '2026-08-12', '2026-10-22')}


def yearly(s):
    ye = s.groupby(s.index.year).last(); out = {}; prev = s.iloc[0]
    for y in ye.index:
        out[int(y)] = float(ye.loc[y] / prev - 1); prev = ye.loc[y]
    return out


def cagr(s, d0=None):
    s = s if d0 is None else s.loc[d0:]; return float((s.iloc[-1] / s.iloc[0]) ** (365.25 / (s.index[-1] - s.index[0]).days) - 1)


yo, yn = yearly(a), yearly(b)
Y = pd.DataFrame({'現用_jonatw_%': pd.Series(yo) * 100, '自建_官方_%': pd.Series(yn) * 100}); Y['差_百分點'] = Y.iloc[:, 1] - Y.iloc[:, 0]
Y.index.name = 'year'; Y.round(4).to_csv(OUT / 'e50_yearly_compare.csv')
prof = json.loads((SITE / 'profit.json').read_text())
D19 = IDX[IDX < pd.Timestamp('2019-01-01')][-1]
R['yearly'] = {int(y): dict(now=round(yo[y] * 100, 4), new=round(yn[y] * 100, 4), diff_pp=round((yn[y] - yo[y]) * 100, 4), site_profit_json=prof['years']['etf'].get(str(y))) for y in yo}
R['navs_official_0050_col'] = dict(end_now=float(nav['0050'].iloc[-1]), end_new=float(round(b.iloc[-1] * 1e6, 2)), end_diff=float(round(b.iloc[-1] * 1e6 - nav['0050'].iloc[-1], 2)),
                                   rebuilt_now_matches_file_max_abs=float((a * 1e6 - nav['0050']).abs().max()),
                                   n_days_changed_gt_1=int(((b * 1e6 - nav['0050']).abs() > 1).sum()), n_days_changed_gt_1000=int(((b * 1e6 - nav['0050']).abs() > 1000).sum()),
                                   cagr_full_now=cagr(a) * 100, cagr_full_new=cagr(b) * 100, cagr_2019_now=cagr(a, D19) * 100, cagr_2019_new=cagr(b, D19) * 100,
                                   maxdd_now=float((a / a.cummax() - 1).min() * 100), maxdd_new=float((b / b.cummax() - 1).min() * 100))
# S75：2008-01-02 起的 0050 欄；2015-07-01 以前沒有官方日 K → 沿用現用序列，之後接自建
s75 = pd.read_csv(AG / 's75_continuous_2008_2026/nav_2008_2026.csv.gz', parse_dates=['date']).set_index('date')['0050']
t = s75.index[s75.index <= D0][-1]
s75_new = pd.concat([s75.loc[:t], s75.loc[t] * (E_new.reindex(s75.loc[t:].index) / E_new.loc[t]).iloc[1:]])
R['s75_0050'] = dict(splice_day=str(t.date()), end_day=str(s75.index[-1].date()), end_now=float(s75.iloc[-1]), end_new=float(s75_new.iloc[-1]), diff=float(s75_new.iloc[-1] - s75.iloc[-1]),
                     cagr_now=cagr(s75) * 100, cagr_new=cagr(s75_new) * 100, year_2026_now=yearly(s75)[2026] * 100, year_2026_new=yearly(s75_new)[2026] * 100,
                     note='2008-01-02～2015-07-01 沒有官方日 K 與事件檔，沿用現用序列（單一來源，稽核已註記無法驗證）')
pd.DataFrame({'0050_now': s75, '0050_new': s75_new.round(2)}).loc['2015-07-01':].to_csv(OUT / 's75_0050_compare.csv.gz')
(OUT / 'e50_compare.json').write_text(json.dumps(R, ensure_ascii=False, indent=1))
print(json.dumps({k: v for k, v in R.items() if k != 'daily'}, ensure_ascii=False, indent=1))
d = R['daily']; print({k: v for k, v in d.items() if k not in ('days_ret_diff_gt_0p02pp',)}); print(d['days_ret_diff_gt_0p02pp'][:40])
