#!/usr/bin/env python3
"""制度事件的「可成交版本」重測（上市股，流動性：60 日均成交值 ≥ NT$1,000 萬；異常報酬＝個股 − 上市等權）。
  E2' 注意股：公告為盤後 → 隔日收盤進場，持有 5／20 日
  E3' 處置股：公告為盤後 → 處置第一天收盤進場，持有到處置結束；處置結束後 20 日
  E5' 漲停：鎖漲停收盤買不到 → 隔日開盤買，持有到隔日收盤／5 日；另測「盤中漲停但收盤打開」收盤買、隔日收盤賣
  E1' 搶息組合：每天持有「10 個交易日內將除息」的流動股（等權），除息前一日收盤賣出，逐日計算組合報酬與周轉成本
輸出 output/bt_rule_events2.json
"""
import json
from pathlib import Path
import numpy as np, pandas as pd

S = Path('/tmp/claude-0/-home-user--/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad'); P = S / 'panel'
X = Path('/home/user/stevenhsiehsheng-ctrl/tw-stock-screener/data/extras'); OUT = Path(__file__).resolve().parent / 'output'
adj = pd.read_parquet(P / 'adj.parquet'); C = pd.read_parquet(P / 'close_raw.parquet'); O = pd.read_parquet(P / 'open.parquet'); H = pd.read_parquet(P / 'high.parquet')
R1 = pd.read_parquet(P / 'ret.parquet'); SRC = pd.read_parquet(P / 'src.parquet'); V = pd.read_parquet(P / 'value.parquet')
idx = adj.index; liq = V.rolling(60).mean() >= 1e7
mkt = R1.where(liq).mean(axis=1)
LA = np.log1p(R1.fillna(0)).cumsum(); LM = np.log1p(mkt.fillna(0)).cumsum()
def car(code, i0, i1):
    if i0 < 0 or i1 >= len(idx): return np.nan
    if not (np.isfinite(adj[code].iat[i0]) and np.isfinite(adj[code].iat[i1])): return np.nan
    return float(np.expm1(LA[code].iat[i1] - LA[code].iat[i0]) - np.expm1(LM.iat[i1] - LM.iat[i0]))
def summ(events, label, cost=0.005):
    s = pd.Series([v for _, v in events], index=pd.DatetimeIndex([d for d, _ in events])).dropna()
    if len(s) < 20: return {'label': label, 'n': int(len(s))}
    byd = s.groupby(level=0).mean(); t = byd.mean() / (byd.std(ddof=1) / np.sqrt(len(byd)))
    yr = s.groupby(s.index.year).mean()
    return {'label': label, 'n_events': int(len(s)), 'mean_event_pct': round(float(s.mean() * 100), 3), 'median_event_pct': round(float(s.median() * 100), 3),
            'mean_by_date_pct': round(float(byd.mean() * 100), 3), 't_by_date': round(float(t), 2), 'hit_rate': round(float((s > 0).mean()), 3),
            'net_event_after_cost_pct': round(float(s.mean() * 100 - cost * 100), 3), 'by_year_mean_pct': {int(k): round(float(v * 100), 2) for k, v in yr.items()}}
res = {}
at = pd.read_csv(X / 'attention_5y.csv.gz', dtype={'code': str}, parse_dates=['date']); at = at[at['code'].isin(adj.columns)].drop_duplicates(['date', 'code'])
for k in (5, 20):
    ev = []
    for d, c in zip(at['date'], at['code']):
        i = idx.searchsorted(d)
        if i < len(idx) and idx[i] == d and liq[c].iat[i]: ev.append((d, car(c, i + 1, i + 1 + k)))
    res[f"E2'_attention_lag1_{k}d"] = summ(ev, f'注意股：隔日收盤進場，持有 {k} 日')
dp = pd.read_csv(X / 'disposal_5y.csv.gz', dtype={'code': str}, parse_dates=['date', 'start', 'end']); dp = dp[dp['code'].isin(adj.columns)]
during, after = [], []
for _, r in dp.iterrows():
    i0 = idx.searchsorted(r['start']); i1 = idx.searchsorted(r['end'], 'right') - 1
    if i0 >= len(idx) or i1 <= i0: continue
    during.append((r['start'], car(r['code'], i0, i1))); after.append((r['end'], car(r['code'], i1, i1 + 20)))
res["E3'_disposal_from_first_day_close"] = summ(during, '處置第一天收盤進場→處置結束')
res["E3'_disposal_after_end_20d"] = summ(after, '處置結束收盤→後 20 日')
prevC = C.ffill().shift(1)
official = SRC.isin([1, 2, 4]) & pd.DataFrame(np.repeat((idx >= pd.Timestamp('2015-06-01'))[:, None], C.shape[1], axis=1), index=idx, columns=C.columns)
locked = (R1 >= 0.095) & (C >= H * 0.999) & liq & official
opened = (H / prevC >= 1.095) & (C < H * 0.99) & (R1 < 0.095) & liq & official
lo_oc, lo_o5, op_cc = [], [], []
for d, c in zip(*np.where(locked.to_numpy())):
    if d + 6 >= len(idx): continue
    code = adj.columns[c]; o = O.iat[d + 1, c]; cl1 = C.iat[d + 1, c]
    if not (np.isfinite(o) and o > 0 and np.isfinite(cl1)): continue
    m1 = mkt.iat[d + 1] if np.isfinite(mkt.iat[d + 1]) else 0.0
    lo_oc.append((idx[d], cl1 / o - 1 - m1 * 0))         # 日內（開→收）報酬，未扣大盤
    a5 = adj[code].iat[d + 5] / adj[code].iat[d + 1] * (cl1 / o) - 1
    m5 = np.expm1(LM.iat[d + 5] - LM.iat[d])
    lo_o5.append((idx[d], a5 - m5))
for d, c in zip(*np.where(opened.to_numpy())):
    if d + 1 < len(idx): op_cc.append((idx[d], car(adj.columns[c], d, d + 1)))
res["E5'_locked_limit_up_buy_next_open_to_close"] = summ(lo_oc, '鎖漲停→隔日開盤買、隔日收盤賣（原始日內報酬）', cost=0.0023)
res["E5'_locked_limit_up_buy_next_open_hold_5d"] = summ(lo_o5, '鎖漲停→隔日開盤買、持有至第 5 日收盤（異常）')
res["E5'_limit_touched_but_opened_close_to_next"] = summ(op_cc, '盤中觸及漲停但收盤打開→收盤買、隔日收盤賣（異常）', cost=0.0023)
# E1' 搶息組合（逐日）
ex = pd.read_csv(X / 'exdiv_5y.csv.gz', dtype={'code': str}, parse_dates=['date']); ex = ex[ex['code'].isin(adj.columns)]
W = pd.DataFrame(False, index=idx, columns=adj.columns)
for _, r in ex.iterrows():
    i = idx.searchsorted(r['date'])
    if i >= len(idx) or idx[i] != r['date'] or i < 11: continue
    W.iloc[i - 11:i - 1, W.columns.get_loc(r['code'])] = True     # 持有：除息前第 10 日收盤買 → 除息前 1 日收盤賣（報酬落在 i-10..i-1）
hold = W.shift(1, fill_value=False) & liq.shift(1, fill_value=False)   # 當日報酬屬於前一日收盤的持股
port = R1.where(hold).mean(axis=1)
n = hold.sum(axis=1)
w_prev = hold.shift(1, fill_value=False)
turn = ((hold & ~w_prev).sum(axis=1) / n.replace(0, np.nan)).fillna(0)
cost = turn * 0.005
sel = (n >= 5) & (idx >= pd.Timestamp('2021-11-01'))
pr = (port - cost).where(sel); mk = mkt.where(sel)
yrs = (idx[sel][-1] - idx[sel][0]).days / 365.25
def cagr_days(x):
    x = x.dropna(); return float(np.exp(np.log1p(x).sum() / yrs) - 1)
ex_d = (port - cost - mkt)[sel].dropna()
res["E1'_pre_exdiv_portfolio"] = {'label': '搶息組合：除息前 10 日買、前 1 日賣，只在有 ≥5 檔時持有，其餘時間不持有',
    'days_invested_share': round(float(sel.mean() if False else (sel[idx >= pd.Timestamp('2021-11-01')]).mean()), 3),
    'avg_names': round(float(n[sel].mean()), 1), 'daily_excess_bp': round(float(ex_d.mean() * 1e4), 2), 't_daily_excess': round(float(ex_d.mean() / (ex_d.std() / np.sqrt(len(ex_d)))), 2),
    'cum_return_invested_days_net': round(float(np.expm1(np.log1p((port - cost)[sel].dropna()).sum())), 4), 'cum_market_same_days': round(float(np.expm1(np.log1p(mkt[sel].dropna()).sum())), 4),
    'avg_daily_turnover_cost_bp': round(float(cost[sel].mean() * 1e4), 2),
    'by_year_excess_pct': {int(y): round(float(np.expm1(np.log1p((port - cost)[sel & (idx.year == y)].dropna()).sum()) - np.expm1(np.log1p(mkt[sel & (idx.year == y)].dropna()).sum())) * 100, 2) for y in range(2021, 2027)}}
(OUT / 'bt_rule_events2.json').write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding='utf-8')
for k, v in res.items(): print(k, v)
