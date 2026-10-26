#!/usr/bin/env python3
"""制度／規則造成的可交易現象（上市股）。異常報酬＝個股報酬 − 同日上市股等權平均；t 值以「事件日」為單位（同日事件先平均），避免同日相關高估。
  E1 除權息：除息前 10 日→前 1 日（搶息）、除息日→後 20 日（填息）
  E2 注意股：公告後 1／5／20 日
  E3 處置股：處置開始日→結束日、結束後 20 日
  E4 投信季底作帳：季末前 15 日投信買超最多 1/5 的股票，季末最後 10 日 vs 新季前 10 日
  E5 漲停隔日：收盤漲停（漲幅 ≥ 9.5% 且收在最高）→ 隔日開盤、隔日收盤、5 日（官方資料期間，2015-06 起 10% 漲跌幅）
  E6 月初效應與農曆年前（0050 含息，2004～2026）
來回成本：一般 0.5%；當沖 0.15% 稅＋手續費約 0.23%。輸出 output/bt_rule_events.json
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
    """日序 i0（不含）到 i1（含）的累積異常報酬（log 近似後轉回）。"""
    if i0 < 0 or i1 >= len(idx) or code not in LA.columns: return np.nan
    a = LA[code].iat[i1] - LA[code].iat[i0]; m = LM.iat[i1] - LM.iat[i0]
    if not np.isfinite(adj[code].iat[i0]) or not np.isfinite(adj[code].iat[i1]): return np.nan
    return float(np.expm1(a) - np.expm1(m))

def summarize(events, label, cost=0.005):
    """events: list of (date, value)。以日期平均後計算 t。"""
    s = pd.Series([v for _, v in events], index=pd.DatetimeIndex([d for d, _ in events])).dropna()
    if len(s) < 20: return {'label': label, 'n_events': int(len(s))}
    byd = s.groupby(level=0).mean()
    t = float(byd.mean() / (byd.std(ddof=1) / np.sqrt(len(byd))))
    return {'label': label, 'n_events': int(len(s)), 'n_dates': int(len(byd)), 'mean_abnormal_pct': round(float(s.mean() * 100), 3),
            'mean_by_date_pct': round(float(byd.mean() * 100), 3), 't_by_date': round(t, 2), 'hit_rate': round(float((s > 0).mean()), 3),
            'net_after_cost_pct': round(float(byd.mean() * 100 - cost * 100), 3)}

res = {}
pos = {d: i for i, d in enumerate(idx)}
# E1 除權息
ex = pd.read_csv(X / 'exdiv_5y.csv.gz', dtype={'code': str}, parse_dates=['date']); ex = ex[ex['code'].isin(adj.columns)]
pre, post = [], []
for _, r in ex.iterrows():
    i = idx.searchsorted(r['date'])
    if i >= len(idx) or idx[i] != r['date']: continue
    pre.append((r['date'], car(r['code'], i - 11, i - 1))); post.append((r['date'], car(r['code'], i - 1, i + 20)))
res['E1_exdiv_pre_10d'] = summarize(pre, '除息前 10 日（搶息）'); res['E1_exdiv_post_20d'] = summarize(post, '除息前一日收盤→除息後 20 日（含息）')
# E2 注意股
at = pd.read_csv(X / 'attention_5y.csv.gz', dtype={'code': str}, parse_dates=['date']); at = at[at['code'].isin(adj.columns)].drop_duplicates(['date', 'code'])
for k in (1, 5, 20):
    ev = []
    for d, c in zip(at['date'], at['code']):
        i = idx.searchsorted(d)
        if i < len(idx) and idx[i] == d: ev.append((d, car(c, i, i + k)))
    res[f'E2_attention_next_{k}d'] = summarize(ev, f'注意股公告收盤→後 {k} 日')
# E3 處置股
dp = pd.read_csv(X / 'disposal_5y.csv.gz', dtype={'code': str}, parse_dates=['date', 'start', 'end']); dp = dp[dp['code'].isin(adj.columns)]
during, after = [], []
for _, r in dp.iterrows():
    i0 = idx.searchsorted(r['start']) - 1; i1 = idx.searchsorted(r['end'], 'right') - 1
    if i0 < 0 or i1 <= i0: continue
    during.append((r['start'], car(r['code'], i0, i1))); after.append((r['end'], car(r['code'], i1, i1 + 20)))
res['E3_disposal_during'] = summarize(during, '處置開始前一日→處置結束'); res['E3_disposal_after_20d'] = summarize(after, '處置結束→後 20 日')
# E4 投信季底作帳
inst = pd.read_csv(X / 'inst_5y.csv.gz', dtype={'code': str}, parse_dates=['date']); inst = inst[inst['code'].isin(adj.columns)]
TR = inst.pivot_table(index='date', columns='code', values='trust', aggfunc='sum').reindex(idx).reindex(columns=adj.columns) * 1000
volsh = V / C
qe = [g.index[-1] for _, g in pd.Series(idx, index=idx).loc['2021-11-01':'2026-09-30'].groupby([idx[(idx >= '2021-11-01') & (idx <= '2026-09-30')].year, idx[(idx >= '2021-11-01') & (idx <= '2026-09-30')].quarter])]
last10, first10 = [], []
for d in qe:
    i = pos[d]
    sig = (TR.iloc[i - 25:i - 10].sum() / volsh.iloc[i - 25:i - 10].sum()).where(liq.iloc[i - 10])
    top = sig.dropna(); top = top[top >= top.quantile(0.8)].index
    for c in top:
        last10.append((d, car(c, i - 10, i))); first10.append((d, car(c, i, i + 10)))
res['E4_trust_window_dressing_last10'] = summarize(last10, '投信前期買超前 1/5：季末最後 10 日')
res['E4_trust_window_dressing_next10'] = summarize(first10, '同一批：新季前 10 日')
# E5 漲停隔日（官方資料期間，10% 制度）
lu = (R1 >= 0.095) & (C >= H * 0.999) & liq & SRC.isin([1, 2, 4]) & (pd.DataFrame(idx >= pd.Timestamp('2015-06-01'), index=idx, columns=['x'])['x'].to_numpy()[:, None])
on_open, on_close, d5 = [], [], []
Cn = C.shift(-1); On = O.shift(-1)
for d, c in zip(*np.where(lu.to_numpy())):
    dt = idx[d]; code = adj.columns[c]
    if d + 5 >= len(idx): continue
    m1 = mkt.iat[d + 1] if np.isfinite(mkt.iat[d + 1]) else 0
    o = On.iat[d, c]; cl = C.iat[d, c]
    if np.isfinite(o) and cl > 0: on_open.append((dt, o / cl - 1))
    on_close.append((dt, car(code, d, d + 1))); d5.append((dt, car(code, d, d + 5)))
res['E5_limit_up_next_open_raw'] = summarize(on_open, '漲停收盤→隔日開盤（原始報酬）', cost=0.0023)
res['E5_limit_up_next_close'] = summarize(on_close, '漲停收盤→隔日收盤（異常）')
res['E5_limit_up_next_5d'] = summarize(d5, '漲停收盤→5 日（異常）')
# E6 月初效應、農曆年前（0050 含息）
e50 = pd.read_csv('/home/user/jonatw/finlab-txd/data/raw/etf_0050.csv', parse_dates=['date']).set_index('date')['adj'].pct_change().dropna()
e = e50.to_frame('r'); e['ym'] = e.index.to_period('M')
e['rank_fwd'] = e.groupby('ym').cumcount(); e['rank_bwd'] = e.groupby('ym').cumcount(ascending=False)
tom = (e['rank_bwd'] == 0) | (e['rank_fwd'] <= 2)
res['E6_turn_of_month'] = {'tom_days_mean_bp': round(float(e.loc[tom, 'r'].mean() * 1e4), 2), 'other_days_mean_bp': round(float(e.loc[~tom, 'r'].mean() * 1e4), 2),
    't_diff': round(float((e.loc[tom, 'r'].mean() - e.loc[~tom, 'r'].mean()) / np.sqrt(e.loc[tom, 'r'].var() / tom.sum() + e.loc[~tom, 'r'].var() / (~tom).sum())), 2),
    'share_of_days': round(float(tom.mean()), 3),
    'tom_only_CAGR': round(float(np.exp(np.log1p(e.loc[tom, 'r']).sum() / ((e.index[-1] - e.index[0]).days / 365.25)) - 1), 4),
    'buy_hold_CAGR': round(float(np.exp(np.log1p(e['r']).sum() / ((e.index[-1] - e.index[0]).days / 365.25)) - 1), 4)}
gaps = e.index.to_series().diff().dt.days
lny_after = gaps[(gaps >= 5) & (e.index.month <= 2)].index
pre5 = []
for d in lny_after:
    i = e.index.get_loc(d)
    if i >= 6: pre5.append(float(np.expm1(np.log1p(e['r'].iloc[i - 6:i - 1]).sum())))
res['E6_pre_lunar_new_year_5d'] = {'n_years': len(pre5), 'mean_pct': round(float(np.mean(pre5) * 100), 2), 'hit_rate': round(float(np.mean(np.array(pre5) > 0)), 2),
    't': round(float(np.mean(pre5) / (np.std(pre5, ddof=1) / np.sqrt(len(pre5)))), 2), 'avg_5d_any_pct': round(float(np.expm1(np.log1p(e['r']).rolling(5).sum()).mean() * 100), 2)}
(OUT / 'bt_rule_events.json').write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding='utf-8')
for k, v in res.items(): print(k, v)
