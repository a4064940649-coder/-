#!/usr/bin/env python3
"""項目 2 補充：(a) 持有期間內 TWT49U 事件對官方前收、對第二來源（stevenhsieh exdiv_5y，2021-10 起）；(b) 持有期間每日漲跌是否在漲跌幅限制內；(c) 指名大贏家價格明細。"""
import sys, json; from pathlib import Path; sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
import alib as A
T = A.trades(); D = A.Data(set(T['code'])); pos = {d: i for i, d in enumerate(D.IDX)}
ex5 = pd.read_csv('/home/user/stevenhsiehsheng-ctrl/tw-stock-screener/data/extras/exdiv_5y.csv.gz', dtype={'code': str}, parse_dates=['date']).drop_duplicates(['date', 'code'], keep='last').set_index(['date', 'code'])
ca = pd.read_csv('/home/user/stevenhsiehsheng-ctrl/tw-stock-screener/data/extras/corp_actions.csv', dtype={'code': str}, parse_dates=['date'])
out = {}
ev = []; lim = []; resets = []; offcal = []
for r in T.drop_duplicates(['code', 'ent_date', 'exit_date']).itertuples():
    f = D.frame(r.code); i0, i1 = pos[r.ent_date], pos[r.exit_date]
    sub = f.iloc[i0 + 1:i1 + 1]
    if r.code in D.evd:
        e = D.evd[r.code]; e = e[(e.index > r.ent_date) & (e.index <= r.exit_date)]
        for d, x in e.iterrows():
            if d not in f.index:
                offcal.append(dict(code=r.code, date=str(d.date()), type=x['type'], prev_close=x['prev_close'], ref=x['ref_price'])); continue
            k = (d, r.code)
            ev.append(dict(code=r.code, date=d, type=x['type'], prev_close=x['prev_close'], ref=x['ref_price'], value=x['value'], off_prevC=f['prevC'].loc[d], fix=f['fix'].loc[d],
                           raw_chg=float(D.off[r.code]['change'].get(d, np.nan)) if r.code in D.off else np.nan,
                           ex5_ref=float(ex5.at[k, 'ref_price']) if k in ex5.index else np.nan, ex5_prev=float(ex5.at[k, 'prev_close']) if k in ex5.index else np.nan))
    for d, x in sub.iterrows():
        if not x['has']: continue
        pct = 0.10 if d >= A.LIM10 else 0.07
        m = x['C'] / x['REF'] - 1
        if abs(m) > pct + 0.005:
            lim.append(dict(code=r.code, date=str(d.date()), prevC=x['prevC'], REF=x['REF'], C=x['C'], move=round(m, 4), xflag=int(x['xflag']), fix=x['fix']))
        if np.isfinite(x['prevC']) and abs(x['REF'] - x['prevC']) > 1e-6 and not (r.code in D.evd and d in D.evd[r.code].index) and x['xflag'] == 0:
            c2 = ca[(ca['code'] == r.code) & (ca['date'] == d)]
            resets.append(dict(code=r.code, date=str(d.date()), prevC=x['prevC'], REF=round(x['REF'], 4), ratio=round(x['REF'] / x['prevC'], 4), halted_before=bool(not f['has'].shift(1).loc[d]),
                               second_source=(c2.iloc[0][['kind', 'prev_close', 'ref_price']].to_dict() if len(c2) else None)))
E = pd.DataFrame(ev).drop_duplicates(['code', 'date'])
E['prev_ok'] = (E['prev_close'] - E['off_prevC']).abs() < 0.011
E['ex5_ok'] = (E['ex5_ref'] - E['ref']).abs() < 0.011
out['twt49u_in_holdings'] = dict(n=int(len(E)), prev_close_matches_engine_prev_close=int(E['prev_ok'].sum()), prev_mismatch=E[~E['prev_ok']].assign(date=lambda x: x['date'].astype(str)).to_dict('records'),
                                 fix=E['fix'].value_counts().to_dict(), n_with_second_source=int(E['ex5_ref'].notna().sum()), second_source_ref_match=int(E['ex5_ok'].sum()),
                                 second_source_mismatch=E[E['ex5_ref'].notna() & ~E['ex5_ok']].assign(date=lambda x: x['date'].astype(str)).to_dict('records'),
                                 by_year=E.groupby(E['date'].dt.year).size().to_dict(), n_raw_chg_nonzero_on_exday=int((E['raw_chg'].fillna(0) != 0).sum()))
out['twt49u_events_on_non_trading_day_in_holdings'] = offcal
allev = D.ev[D.ev['code'].str.fullmatch(r'[1-9]\d{3}')]
nt = allev[~allev['date'].isin(D.IDX) & (allev['date'] <= A.END)]
out['twt49u_all_events_on_non_trading_day'] = dict(n=int(len(nt)), dates=nt.groupby(nt['date'].astype(str)).size().to_dict(), n_all=int(len(allev)))
out['moves_beyond_limit_in_holdings'] = lim
out['ref_resets_non_twt49u'] = resets
# (c) 指名大贏家價格明細
det = {}
for c, e, x in [('2408', '2025-09-12', '2026-04-14'), ('3653', '2026-08-12', '2026-10-14'), ('3017', '2025-05-14', '2026-01-14'), ('2344', '2026-05-13', '2026-07-15'), ('2059', '2026-06-12', '2026-10-14'), ('1325', '2020-03-12', '2020-08-12')]:
    f = D.frame(c).loc[e:x]
    mv = (f['C'] / f['REF'] - 1)
    det[c] = dict(ent=e, exit=x, ent_open=float(f['O'].iloc[0]), ent_close=float(f['C'].iloc[0]), exit_open=float(f['O'].iloc[-1]), max_close=float(f['C'].max()), min_close=float(f['C'].min()),
                  n_days=int(len(f)), n_limit_up_close=int((mv > 0.095).sum()), n_limit_dn_close=int((mv < -0.095).sum()), max_daily_move=float(mv.max()), min_daily_move=float(mv.min()),
                  n_notrade=int((~f['has']).sum()), name=str(f['name'].iloc[0]), month_end_closes={str(k): float(v) for k, v in f['C'].groupby(f.index.to_period('M')).last().items()},
                  mean_daily_amount_e8=float(f['AMT'].mean() / 1e8))
out['winners_detail'] = det
(A.HERE / 'results_extra.json').write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str))
print(json.dumps(out, ensure_ascii=False, indent=1, default=str)[:9000])
