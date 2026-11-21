#!/usr/bin/env python3
"""法人、融資融券、借券面板 對 研究中心既有的第三方來源（重疊期逐格比對）。
用法：python3 -I -B crosscheck.py <tsd_extra 資料夾> <結果 json>
"""
import sys, os, json
import pandas as pd, numpy as np
out, resf = sys.argv[1:3]
sys.path.insert(0, out)
from tsd_extra_load import load
R = '/home/user/-/research/tw_stock_selection/agents/o_sbl_foreign_level/raw'
res = {}


def cmp(a, b, cols, tol=0):
    m = a.merge(b, on=['date', 'code'], suffixes=('', '_o'))
    o = dict(rows_ours_in_range=len(a), rows_other=len(b), matched_keys=len(m), only_other=len(b) - len(m))
    for c in cols:
        d = (m[c] - m[c + '_o']).abs()
        o[c] = dict(equal=int((d <= tol).sum()), different=int((d > tol).sum()), max_abs_diff=float(d.max()) if len(d) else None)
        if (d > tol).any():
            bad = m[d > tol]
            o[c]['different_days'] = int(bad.date.nunique()); o[c]['sample'] = bad[['date', 'code', c, c + '_o']].head(5).astype(str).to_dict('records')
            o[c]['top_days'] = bad.date.astype(str).value_counts().head(5).to_dict()
    return o


def ours(name, s, e, cols):
    d = load(name, s, e, columns=['date', 'code'] + cols); d['date'] = d['date'].astype(str); return d

# 1. 法人：BlackBoxOP（股數；外資＝外資＋外資自營）
b = pd.read_parquet(f'{R}/bb_inst.parquet'); b = b.rename(columns={'Date': 'date', 'Ticker': 'code'})
b['foreign'] = b.Foreign_Net + b.Foreign_Dealer_Net; b['trust'] = b.Trust_Net; b['dealer'] = b.Dealer_Net
for mk, nm in (('TWSE', 'inst_twse'), ('TPEx', 'inst_tpex')):
    x = b[b.Market == mk][['date', 'code', 'foreign', 'trust', 'dealer']]
    if len(x):
        res[f'{nm}_vs_BlackBoxOP_bb_inst'] = dict(period=[x.date.min(), x.date.max()], unit='股', **cmp(ours(nm, x.date.min(), x.date.max(), ['foreign', 'trust', 'dealer']), x, ['foreign', 'trust', 'dealer']))
# 2. 法人：tw-stock-screener inst_5y（張，小數三位）2021-10 起
f = '/home/user/stevenhsiehsheng-ctrl/tw-stock-screener/data/extras/inst_5y.csv.gz'
if os.path.exists(f):
    s = pd.read_csv(f, dtype={'code': str})
    o = pd.concat([ours('inst_twse', s.date.min(), s.date.max(), ['foreign', 'trust', 'dealer']), ours('inst_tpex', s.date.min(), s.date.max(), ['foreign', 'trust', 'dealer'])])
    for c in ['foreign', 'trust', 'dealer']: o[c] = o[c] / 1000.0
    res['inst_twse_tpex_vs_screener_inst_5y'] = dict(period=[s.date.min(), s.date.max()], unit='張（我們的股數÷1000），容許差 0.0011', **cmp(o, s[['date', 'code', 'foreign', 'trust', 'dealer']], ['foreign', 'trust', 'dealer'], tol=0.0011))
# 3. 融資融券：BlackBoxOP（張）
b = pd.read_parquet(f'{R}/bb_margin.parquet').rename(columns={'Date': 'date', 'Ticker': 'code', 'Margin_Balance': 'm_balance', 'Short_Balance': 's_balance', 'Margin_Prev_Balance': 'm_prev'})
for mk, nm in (('TWSE', 'margin_twse'), ('TPEx', 'margin_tpex')):
    x = b[b.Market == mk][['date', 'code', 'm_balance', 's_balance', 'm_prev']]
    if len(x):
        res[f'{nm}_vs_BlackBoxOP_bb_margin'] = dict(period=[x.date.min(), x.date.max()], unit='張', **cmp(ours(nm, x.date.min(), x.date.max(), ['m_balance', 's_balance', 'm_prev']), x, ['m_balance', 's_balance', 'm_prev']))
# 4. 借券：BlackBoxOP TWT93U（股）
b = pd.read_parquet(f'{R}/bb_sbl_twt93u.parquet').rename(columns={'Date': 'date', 'Ticker': 'code', 'SBL_Balance': 'sbl_balance', 'SBL_Daily_Sell': 'sbl_sell', 'Margin_Short_Balance': 's_balance'})
x = b[['date', 'code', 'sbl_balance', 'sbl_sell', 's_balance']]
res['sbl_twse_vs_BlackBoxOP_twt93u'] = dict(period=[x.date.min(), x.date.max()], unit='股', **cmp(ours('sbl_twse', x.date.min(), x.date.max(), ['sbl_balance', 'sbl_sell', 's_balance']), x, ['sbl_balance', 'sbl_sell', 's_balance']))
# 5. 內部一致性
i = load('inst_twse'); res['inst_twse_identities'] = dict(rows=len(i), total_eq_sum=int((i.total == i.foreign + i.trust + i.dealer).sum()), dealer_eq_self_plus_hedge=int((i.dealer == i.dealer_self + i.dealer_hedge).sum()))
i = load('inst_tpex'); res['inst_tpex_identities'] = dict(rows=len(i), total_eq_sum=int((i.total == i.foreign + i.trust + i.dealer).sum()), dealer_eq_self_plus_hedge=int((i.dealer == i.dealer_self + i.dealer_hedge).sum()))
del i
m = load('margin_twse', columns=['date', 'code', 'm_buy', 'm_sell', 'm_ret', 'm_prev', 'm_balance', 's_balance']); s = load('sbl_twse', columns=['date', 'code', 's_balance', 'sbl_prev', 'sbl_sell', 'sbl_return', 'sbl_adj', 'sbl_balance'])
res['margin_twse_identity_balance'] = dict(rows=len(m), m_balance_eq_prev_plus_buy_minus_sell_minus_ret=int((m.m_balance == m.m_prev + m.m_buy - m.m_sell - m.m_ret).sum()))
res['sbl_twse_identity_balance'] = dict(rows=len(s), sbl_balance_eq_prev_plus_sell_minus_return_plus_adj=int((s.sbl_balance == s.sbl_prev + s.sbl_sell - s.sbl_return + s.sbl_adj).sum()))
j = m.merge(s, on=['date', 'code'], suffixes=('_m', '_s'))
res['margin_vs_sbl_short_balance_twse'] = dict(matched=len(j), equal_after_x1000=int((j.s_balance_m * 1000 == j.s_balance_s).sum()), note='融資融券表的融券餘額（張）×1000 對 借券表裡的融券餘額（股）')
json.dump(res, open(resf, 'w'), ensure_ascii=False, indent=1, default=str)
print(json.dumps(res, ensure_ascii=False, default=str)[:6000])
