#!/usr/bin/env python3
"""S16 事後分析（標「事後」，不影響判定）：只讀 results.json、trades.csv.gz，不重算面板。
(1) ALL 主跑損益來自上市／上櫃各多少（報酬貢獻口徑＋金額口徑對照），全期、2019 起、2019～2023、2024 起；
(2) 上櫃大贏家（ALL、OTC 主跑報酬貢獻前 10 筆上櫃交易）；ALL 對 CUR 換掉了哪些交易；
(3) 單筆 ≤ 當日成交值 5% 上限：分市場觸發次數（主跑與 20 條路徑合計）、被綁住的買進金額 ÷ 目標；
(4) 流動性：買進時 60 日均成交值與買進金額 ÷ 60 日均成交值（分市場）；
(5) 分市場每筆期望值；(6) 逐年；(7) 資料版本 ALL_RAW；(8) 持有期間轉上市的交易。
python3 -I -B posthoc.py → results_posthoc.json"""
import json
from pathlib import Path
import numpy as np, pandas as pd

H = Path(__file__).resolve().parent
R = json.loads((H / 'results.json').read_text())
tr = pd.read_csv(H / 'trades.csv.gz', dtype={'code': str}, parse_dates=['sig_date', 'ent_date', 'exit_date'])
out = {}
PER = ('full', 'since2019', 'y2019_2023', 'since2024')

# (1) 損益來源
src = {}
for run in ('ALL', 'ALL_O20', 'CUR', 'OTC'):
    x = tr[tr['run'] == run]
    nc = R['cells'][run]['nav_contrib']
    d = {}
    for p in PER:
        c = x.groupby('market')[f'contrib_{p}'].sum()
        tot = float(c.sum())
        d[p] = dict(stock_contrib_pct=round(tot * 100, 2), nav_contrib_pct=round(nc[p] * 100, 2), etf_and_cash_pct=round((nc[p] - tot) * 100, 2),
                    **{f'{m}_contrib_pct': round(float(c.get(m, 0.0)) * 100, 2) for m in ('twse', 'tpex')},
                    **{f'{m}_share_of_stock': (round(float(c.get(m, 0.0)) / tot, 3) if tot else None) for m in ('twse', 'tpex')})
    # 金額口徑（依進場日歸期間，對照用）
    amt = {}
    for p, (a, b) in dict(full=('2000', '2100'), since2019=('2019-01-01', '2100'), y2019_2023=('2019-01-01', '2023-12-31'), since2024=('2024-01-01', '2100')).items():
        y = x[(x['ent_date'] >= a) & (x['ent_date'] <= b)]
        g = y.groupby('market')['pnl'].sum(); t = float(g.sum())
        amt[p] = {f'{m}_share': (round(float(g.get(m, 0.0)) / t, 3) if t else None) for m in ('twse', 'tpex')}
    src[run] = dict(return_contribution=d, money_by_entry_date=amt, n_trades=x.groupby('market').size().to_dict())
out['1_pnl_source（報酬貢獻＝逐日損益 ÷ 前一日總淨值後加總，單位：百分點）'] = src

# (2) 上櫃大贏家；ALL 對 CUR 換掉了哪些
cols = ['code', 'ent_date', 'exit_date', 'ret', 'contrib_full', 'contrib_since2024', 'pnl', 'why']
fmt = lambda df: [dict(code=r.code, ent=str(r.ent_date.date()), exit=str(r.exit_date.date()), ret_pct=round(r.ret * 100, 1), contrib_pp=round(r.contrib_full * 100, 2),
                       pnl=round(r.pnl), why=r.why) for r in df.itertuples()]
for run in ('ALL', 'OTC'):
    x = tr[(tr['run'] == run) & (tr['market'] == 'tpex')].sort_values('contrib_full', ascending=False)
    out[f'2_top10_tpex_in_{run}'] = fmt(x.head(10))
    out[f'2_worst5_tpex_in_{run}'] = fmt(x.tail(5).iloc[::-1])
    out[f'2_{run}_tpex_top10_contrib_pp'] = round(float(x.head(10)['contrib_full'].sum()) * 100, 2)
    out[f'2_{run}_tpex_total_contrib_pp'] = round(float(x['contrib_full'].sum()) * 100, 2)
cur = tr[tr['run'] == 'CUR']; al = tr[tr['run'] == 'ALL']
kc = set(zip(cur['code'], cur['ent_date'])); ka = set(zip(al['code'], al['ent_date']))
only_cur = cur[[k not in ka for k in zip(cur['code'], cur['ent_date'])]].sort_values('contrib_full', ascending=False)
only_all = al[[k not in kc for k in zip(al['code'], al['ent_date'])]].sort_values('contrib_full', ascending=False)
codes_cur = set(cur['code']);
out['2_ALL_vs_CUR_trades'] = dict(
    n_cur=int(len(cur)), n_all=int(len(al)), n_same_code_and_entry=int(len(kc & ka)), n_only_cur=int(len(only_cur)), n_only_all=int(len(only_all)),
    only_all_tpex=int((only_all['market'] == 'tpex').sum()),
    only_cur_contrib_pp={p: round(float(only_cur[f'contrib_{p}'].sum()) * 100, 2) for p in PER},
    only_all_contrib_pp={p: round(float(only_all[f'contrib_{p}'].sum()) * 100, 2) for p in PER},
    only_all_tpex_contrib_pp={p: round(float(only_all.loc[only_all['market'] == 'tpex', f'contrib_{p}'].sum()) * 100, 2) for p in PER},
    only_all_twse_contrib_pp={p: round(float(only_all.loc[only_all['market'] == 'twse', f'contrib_{p}'].sum()) * 100, 2) for p in PER},
    top10_only_cur=fmt(only_cur.head(10)), top10_only_all=fmt(only_all.head(10)),
    cur_big_winners_missing_in_all=[c for c in cur.sort_values('contrib_full', ascending=False).head(10)['code'] if c not in set(al['code'])])
# CUR 最賺 10 筆在 ALL 裡有沒有買到（同代號、進場日 ±5 個交易日內 → 用 ±10 天近似）
t10 = cur.sort_values('contrib_full', ascending=False).head(10)
chk = []
for r in t10.itertuples():
    m = al[(al['code'] == r.code) & ((al['ent_date'] - r.ent_date).abs() <= pd.Timedelta(days=10))]
    chk.append(dict(code=r.code, ent=str(r.ent_date.date()), cur_contrib_pp=round(r.contrib_full * 100, 2),
                    in_all=bool(len(m)), all_contrib_pp=(round(float(m['contrib_full'].sum()) * 100, 2) if len(m) else 0.0)))
out['2_CUR_top10_in_ALL'] = chk

# (3) 5% 成交值上限
capd = {}
for run in ('CUR', 'ALL', 'OTC', 'ALL_O20'):
    c = R['cells'][run]
    x = tr[tr['run'] == run]
    g = x.groupby('market')
    p20 = c['paths20']
    capd[run] = dict(main_cap_bound_by_mkt={m: dict(buys=int(len(y)), cap_bound=int(y['cap_bound'].sum()), share=round(float(y['cap_bound'].mean()), 4)) for m, y in g},
                     main_engine_blk_cap=c['engine']['blk_cap'], main_sell_partial_cap=c['engine']['sell_partial_cap'],
                     paths20_cap_tpex=p20['cap_tpex_total'], paths20_buys_tpex=p20['buys_tpex_total'], paths20_cap_twse=p20['cap_twse_total'], paths20_buys_twse=p20['buys_twse_total'],
                     by_capital={cap: dict(cap_tpex=int(sum(v['cap_by_mkt'].get('tpex', {}).get('cap_bound', 0) for k, v in c['paths'].items() if k.startswith(cap + '萬'))),
                                           cap_twse=int(sum(v['cap_by_mkt'].get('twse', {}).get('cap_bound', 0) for k, v in c['paths'].items() if k.startswith(cap + '萬'))),
                                           sell_partial=int(sum(v['sell_partial_cap'] for k, v in c['paths'].items() if k.startswith(cap + '萬'))))
                                 for cap in ('60', '80', '100', '150', '200')})
out['3_cap5pct'] = capd

# (4) 流動性
liq = {}
for run in ('CUR', 'ALL', 'OTC', 'ALL_O20'):
    x = tr[tr['run'] == run].copy()
    x['amt_over_v60'] = x['buy_amt'] / (x['v60_e8'] * 1e8)
    liq[run] = {m: dict(v60_median_e8=round(float(y['v60_e8'].median()), 3), v60_p10_e8=round(float(y['v60_e8'].quantile(0.1)), 3),
                        buy_over_v60_median_pct=round(float(y['amt_over_v60'].median()) * 100, 3), buy_over_v60_p90_pct=round(float(y['amt_over_v60'].quantile(0.9)) * 100, 3))
                for m, y in x.groupby('market')}
out['4_liquidity（60 日均成交值單位：億元；買進金額 ÷ 60 日均成交值，%）'] = liq

# (5) 分市場每筆期望值
out['5_trades_by_mkt'] = {run: {m: {k: (round(v, 4) if isinstance(v, float) else v) for k, v in s.items() if k != 'why'} for m, s in R['cells'][run]['trades_by_mkt'].items()}
                          for run in ('CUR', 'ALL', 'OTC', 'ALL_O20')}

# (6) 逐年
yrs = sorted(R['cells']['CUR']['yearly'].keys(), key=int)
out['6_yearly_pct'] = {y: dict(CUR=round(R['cells']['CUR']['yearly'][y]['strat'] * 100, 1), ALL=round(R['cells']['ALL']['yearly'][y]['strat'] * 100, 1),
                               OTC=round(R['cells']['OTC']['yearly'][y]['strat'] * 100, 1), ALL_O20=round(R['cells']['ALL_O20']['yearly'][y]['strat'] * 100, 1),
                               e0050=round(R['cells']['CUR']['yearly'][y]['e0050'] * 100, 1),
                               ALL_minus_CUR=round((R['cells']['ALL']['yearly'][y]['strat'] - R['cells']['CUR']['yearly'][y]['strat']) * 100, 1)) for y in yrs}
# 逐年：ALL 買進中上櫃占比與上櫃報酬貢獻
x = tr[tr['run'] == 'ALL']
out['6_ALL_tpex_share_by_entry_year'] = {int(y): dict(n=int(len(g)), tpex_share=round(float((g['market'] == 'tpex').mean()), 3),
                                                     tpex_mean_ret_pct=round(float(g.loc[g['market'] == 'tpex', 'ret'].mean()) * 100, 2) if (g['market'] == 'tpex').any() else None,
                                                     twse_mean_ret_pct=round(float(g.loc[g['market'] == 'twse', 'ret'].mean()) * 100, 2) if (g['market'] == 'twse').any() else None)
                                         for y, g in x.groupby(x['ent_date'].dt.year)}

# (7) 資料版本
pr = R['pairs']['ALL_RAW-ALL']
out['7_ALL_RAW_vs_ALL'] = {k: pr[k] for k in ('ret_npos', 'ret_median_pp', 'dd_median_pp', 'r1923_npos', 'r1923_median_pp', 'r24_median_pp')}
out['7_ALL_RAW_paths20_median'] = R['cells']['ALL_RAW']['paths20']['median']

# (8) 持有期間轉上市
sw = tr[(tr['run'] == 'ALL') & (tr['market'] == 'tpex') & (tr['mk_exit'] == 'twse')]
out['8_ALL_tpex_to_twse_during_hold'] = dict(n=int(len(sw)), contrib_pp=round(float(sw['contrib_full'].sum()) * 100, 2), trades=fmt(sw))

# 集中度：ALL 最賺 5 筆
out['9_block5_top'] = {run: R['cells'][run].get('block5_top') for run in ('CUR', 'ALL', 'ALL_O20', 'OTC')}
out['9_top5_contrib_share（報酬貢獻口徑：前 5 筆貢獻 ÷ 股票貢獻合計）'] = {}
for run in ('CUR', 'ALL', 'ALL_O20', 'OTC'):
    x = tr[tr['run'] == run]
    t = float(x['contrib_full'].sum())
    out['9_top5_contrib_share（報酬貢獻口徑：前 5 筆貢獻 ÷ 股票貢獻合計）'][run] = round(float(x.nlargest(5, 'contrib_full')['contrib_full'].sum()) / t, 3) if t else None

(H / 'results_posthoc.json').write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str))
for k, v in out.items():
    print(k, json.dumps(v, ensure_ascii=False, default=str)[:1500])
