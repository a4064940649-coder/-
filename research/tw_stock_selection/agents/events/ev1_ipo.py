#!/usr/bin/env python3
"""檢定 1：新上市股（事前固定規則）。
* 新上市日＝面板中該股第一筆有價日期；限 2005-01-01 之後；排除第一筆日期恰好是資料源起點／缺口後重啟日
  （2010-01-04、2019-05-17、2024-02-15、2025-09-24）。
* 「轉上市」辨識：Asoul 原始行情（含櫃買）在面板首日前已有 ≥ 20 筆交易，或 history.csv（上市＋上櫃）在面板首日前已有資料
  → 標為轉上市（之前在櫃買交易）；其餘標為「初次上市」。兩者都報告。
* 進場：上市第 5 個交易日收盤（前 5 日無漲跌幅限制，不做漲停排除）；持有 20／60／120／250 個交易日。
* 異常報酬：對「所有上市股等權」（主要）與「流動性合格等權」（輔助），另報對 0050。
* 流動性：上市前 5 日平均成交值 ≥ 1,000 萬且第 5 日收盤 ≥ 10 元（60 日均量在新股無法計算）。
* 上市第 1 年：首日收盤→第 250 日（非可成交基準，抽籤價不在資料中，僅描述）與第 5 日收盤→第 250 日（可成交）。
* 次要分組（事前固定）：上市首 4 日漲幅（第 1 日收盤→第 5 日收盤）高於／低於挑選期中位數。
* 日曆時間組合：每檔第 5 日收盤買、持有 60／250 日，等權，閒置持現金，來回成本 0.5%。
挑選期：上市日 2005～2018；驗證期：2019～2026。
"""
import glob, sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
import evlib as L

D = L.load(); idx = D['idx']; C = D['close_raw']; V = D['value']
SPL = {'sel_2005_2018': ('2005-01-01', '2018-12-31'), 'val_2019_2026': ('2019-01-01', '2026-12-31')}
ART = {pd.Timestamp(x) for x in ['2010-01-04', '2019-05-17', '2024-02-15', '2025-09-24']}
first = C.apply(lambda x: x.first_valid_index())
cand = first[(first >= pd.Timestamp('2005-01-01')) & (~first.isin(ART))]
print('candidates', len(cand))

# 轉上市辨識
hist = pd.read_csv('/home/user/stevenhsiehsheng-ctrl/tw-stock-screener/data/history.csv.gz', usecols=['date', 'code'], dtype={'code': str}, parse_dates=['date'])
hist_first = hist.groupby('code')['date'].min()
def asoul_rows_before(code, d):
    f = L.S / 'tsec_all' / f'{code}.csv'
    if not f.exists(): return 0
    s = pd.read_csv(f, header=None, usecols=[0], dtype=str)[0].str.strip()
    p = s.str.split('/', expand=True)
    try:
        dt = pd.to_datetime(dict(year=p[0].astype(int) + 1911, month=p[1].astype(int), day=p[2].astype(int)), errors='coerce')
    except Exception:
        dt = pd.to_datetime(s, errors='coerce')
    return int((dt < d - pd.Timedelta(days=3)).sum())
rows = []
car_all = L.Car(D, 'mkt_all'); car_liq = L.Car(D, 'mkt_liq')
pos = {d: i for i, d in enumerate(idx)}
for code, d in cand.items():
    i_f = pos[d]; i0 = i_f + 4
    if i0 >= len(idx): continue
    transfer = asoul_rows_before(code, d) >= 20 or (code in hist_first.index and hist_first[code] < d - pd.Timedelta(days=3))
    v5 = V[code].iloc[i_f:i0 + 1].mean(); c5 = C[code].iat[i0]
    run4 = D['adj'][code].iat[i0] / D['adj'][code].iat[i_f] - 1 if np.isfinite(D['adj'][code].iat[i_f]) else np.nan
    r = {'code': code, 'list_date': d, 'date': idx[i0], 'transfer': bool(transfer), 'liquid': bool(v5 >= 1e7 and c5 >= 10), 'run4': run4}
    for h in (20, 60, 120, 250):
        raw, ab, ab50 = car_all(code, i0, i0 + h); r[f'raw{h}'] = raw; r[f'ab{h}'] = ab; r[f'ab50_{h}'] = ab50
        r[f'abliq{h}'] = car_liq(code, i0, i0 + h)[1]
    raw, ab, ab50 = car_all(code, i_f, i_f + 250); r['raw_y1_from_d1'] = raw; r['ab_y1_from_d1'] = ab
    rows.append(r)
ev = pd.DataFrame(rows)
print(ev.groupby([ev['list_date'].dt.year, 'transfer']).size().unstack(fill_value=0))
res = {'n_candidates': int(len(cand)), 'n_events': int(len(ev)), 'n_transfer': int(ev['transfer'].sum()),
       'by_year_counts': {int(y): {'ipo': int((~g['transfer']).sum()), 'transfer': int(g['transfer'].sum())} for y, g in ev.groupby(ev['list_date'].dt.year)}}
med_run = ev[(ev['date'] <= '2018-12-31')]['run4'].median(); res['sel_median_run4'] = round(float(med_run), 4)

def mk(df, h, key='ab'):
    return pd.DataFrame({'date': df['date'], 'raw': df[f'raw{h}'], 'ab': df[f'{key}{h}'], 'ab50': df[f'ab50_{h}']})
groups = {'all': ev, 'ipo_only': ev[~ev['transfer']], 'transfer_only': ev[ev['transfer']], 'liquid_all': ev[ev['liquid']],
          'liquid_ipo_only': ev[ev['liquid'] & ~ev['transfer']],
          'run4_high': ev[ev['run4'] > med_run], 'run4_low': ev[ev['run4'] <= med_run]}
res['event'] = {}
for gname, g in groups.items():
    res['event'][gname] = {f'h{h}': L.by_period(mk(g, h), SPL) for h in (20, 60, 120, 250)}
    res['event'][gname]['h250_vs_liquidEW'] = L.by_period(mk(g, 250, 'abliq'), SPL)
y1 = pd.DataFrame({'date': ev['date'], 'raw': ev['raw_y1_from_d1'], 'ab': ev['ab_y1_from_d1'], 'ab50': np.nan})
res['first_year_from_day1_close'] = {k: L.by_period(y1[m], SPL) for k, m in [('all', ev.index == ev.index), ('ipo_only', ~ev['transfer'])]}
res['yearly_ab250_all'] = L.yearly_ab(mk(ev, 250)); res['yearly_ab60_all'] = L.yearly_ab(mk(ev, 60))
# 日曆時間組合
res['calendar'] = {}
for h in (60, 250):
    for gname in ('all', 'ipo_only', 'liquid_all'):
        g = groups[gname]
        tr = [(c, pos[d], min(pos[d] + h, len(idx) - 1)) for c, d in zip(g['code'], g['date'])]
        for pname, (a, b) in SPL.items():
            res['calendar'][f'{gname}_h{h}_{pname}'] = L.calendar_portfolio(D, tr, a, min(pd.Timestamp(b), idx[-1]))
L.save('1_ipo', res)
for g in ('all', 'ipo_only', 'liquid_all'):
    for h in (20, 60, 120, 250):
        e = res['event'][g][f'h{h}']; print(g, h, {k: (v.get('n'), v.get('mean_ab_pct'), v.get('median_ab_pct'), v.get('t_month')) for k, v in e.items()})
for k, v in res['calendar'].items(): print(k, v['strategy'], v['0050'], v['t_vs_0050'], v['years_beat_0050'])
