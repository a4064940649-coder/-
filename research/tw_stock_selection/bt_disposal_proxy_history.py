#!/usr/bin/env python3
"""用公開的注意／處置規則重建 2005～2021 的處置股，延伸處置股策略的歷史檢驗。
近似規則（只用價格，證交所實際有多種注意標準，這裡只取最主要的漲幅條件）：
  注意：近 6 個營業日累積漲幅 ≥ 32% 且比加權指數多 ≥ 20 個百分點（原始收盤價）
  處置：連續 3 個營業日達注意 → 第 3 天為公告日，隔日起處置 10 個營業日（期間不重複觸發）
校對：2021-10～2026-09 與真實處置名單（disposal_5y，上市）比對：真實事件中有多少被近似規則在 ±3 日內抓到（召回率），近似事件中有多少是真的（精確率）。
回測：近似事件「處置第一天收盤買、第 10 天收盤賣」，2005～2021-09（真實名單之前）與 2021-10 之後分開報告；異常報酬＝個股 − 上市等權。
輸出 output/bt_disposal_proxy_history.json
"""
import json
from pathlib import Path
import numpy as np, pandas as pd
S = Path('/tmp/claude-0/-home-user--/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad'); P = S / 'panel'
X = Path('/home/user/stevenhsiehsheng-ctrl/tw-stock-screener/data/extras'); OUT = Path(__file__).resolve().parent / 'output'
adj = pd.read_parquet(P / 'adj.parquet'); C = pd.read_parquet(P / 'close_raw.parquet'); R1 = pd.read_parquet(P / 'ret.parquet'); V = pd.read_parquet(P / 'value.parquet')
idx = adj.index
tx = pd.read_csv('/home/user/jonatw/finlab-txd/data/raw/taiex_twii.csv', parse_dates=['date']).set_index('date')['close'].reindex(idx).ffill()
liq = V.rolling(60).mean() >= 1e7
mkt = R1.where(liq).mean(axis=1)
A6 = adj / adj.shift(6) - 1                       # 用還原價避免除權息造成假跌
m6 = tx / tx.shift(6) - 1
att = (A6 >= 0.32) & (A6.sub(m6, axis=0) >= 0.20)
trig = att & att.shift(1, fill_value=False) & att.shift(2, fill_value=False)
events = []
T = trig.to_numpy(); n_d, n_c = T.shape
for c in range(n_c):
    i = 0; col = T[:, c]
    while i < n_d:
        if col[i]:
            s, e = i + 1, i + 10
            if e < n_d: events.append((adj.columns[c], s, e))
            i = e + 1
        else:
            i += 1
ev = pd.DataFrame(events, columns=['code', 'i0', 'i1']); ev['start'] = idx[ev.i0]
# 校對
dp = pd.read_csv(X / 'disposal_5y.csv.gz', dtype={'code': str}, parse_dates=['start', 'end']); dp = dp[dp.code.isin(adj.columns)]
lo, hi = pd.Timestamp('2021-10-01'), pd.Timestamp('2026-09-30')
real = dp[(dp.start >= lo) & (dp.start <= hi)]; prox = ev[(ev.start >= lo) & (ev.start <= hi)]
def matched(a, b):
    bset = {}
    for c, d in zip(b.code, b.start): bset.setdefault(c, []).append(d)
    return np.mean([any(abs((d - x).days) <= 5 for x in bset.get(c, [])) for c, d in zip(a.code, a.start)]) if len(a) else np.nan
calib = {'real_events': int(len(real)), 'proxy_events': int(len(prox)), 'recall': round(float(matched(real, prox)), 3), 'precision': round(float(matched(prox, real)), 3)}
# 回測
LA = np.log1p(R1.fillna(0)).cumsum(); LM = np.log1p(mkt.fillna(0)).cumsum()
rows = []
for c, i0, i1 in zip(ev.code, ev.i0, ev.i1):
    a0, a1 = adj[c].iat[i0], adj[c].iat[i1]
    if not (np.isfinite(a0) and np.isfinite(a1)): continue
    raw = a1 / a0 - 1; m = np.expm1(LM.iat[i1] - LM.iat[i0])
    rows.append({'start': idx[i0], 'ret': raw, 'abn': raw - m})
d = pd.DataFrame(rows)
def stats(x):
    byd = x.groupby('start').abn.mean()
    yr = x.groupby(x.start.dt.year).ret.agg(['count', 'mean'])
    return {'events': int(len(x)), 'mean_pct': round(x.ret.mean() * 100, 2), 'median_pct': round(x.ret.median() * 100, 2), 'hit_rate': round(float((x.ret > 0).mean()), 3),
            'mean_abnormal_pct': round(x.abn.mean() * 100, 2), 't_abnormal_by_date': round(float(byd.mean() / (byd.std() / np.sqrt(len(byd)))), 2),
            'net_after_0.5pct_cost': round(x.ret.mean() * 100 - 0.5, 2),
            'by_year': {int(k): {'n': int(v['count']), 'mean_pct': round(v['mean'] * 100, 2)} for k, v in yr.iterrows()}}
res = {'calibration_2021-10_2026-09': calib,
       'proxy_2005_2015-05_7pct_limit': stats(d[(d.start >= '2005-01-01') & (d.start < '2015-06-01')]),
       'proxy_2015-06_2021-09': stats(d[(d.start >= '2015-06-01') & (d.start < '2021-10-01')]),
       'proxy_2021-10_2026-10': stats(d[d.start >= '2021-10-01'])}
(OUT / 'bt_disposal_proxy_history.json').write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding='utf-8')
for k, v in res.items(): print(k, {kk: vv for kk, vv in v.items() if kk != 'by_year'});
for k in list(res)[1:]: print(k, res[k]['by_year'])
