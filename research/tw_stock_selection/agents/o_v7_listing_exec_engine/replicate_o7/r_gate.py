#!/usr/bin/env python3
"""複製員 o7：閘門（自己的口徑）。只讀 V7／V6 面板、上市矩陣、exdiv_5y、研究員 V6 plan（資料）。"""
import sys, json
from pathlib import Path
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
import numpy as np, pandas as pd
O7 = HERE.parent; AG = O7.parent; D7 = O7 / 'data'; D6 = AG / 'o_v6_price_panel_official' / 'data'
res = {}
F = ['adj', 'close_raw', 'open', 'high', 'low', 'value', 'ret', 'flag', 'src']
# 1. 2006～2023 面板逐格對 V6
cut = pd.Timestamp('2024-01-01'); diff = {}
for k in F:
    a = pd.read_parquet(D7 / f'{k}.parquet'); b = pd.read_parquet(D6 / f'{k}.parquet')
    a = a[a.index < cut]; b = b[b.index < cut].reindex(index=a.index, columns=a.columns)
    x = a.to_numpy(float); y = b.to_numpy(float)
    neq = ~((x == y) | (np.isnan(x) & np.isnan(y)))
    diff[k] = int(neq.sum())
res['pre2024_cells_diff_vs_V6'] = diff
C = pd.read_parquet(D7 / 'close_raw.parquet'); S = pd.read_parquet(D7 / 'src.parquet').reindex_like(C); Vv = pd.read_parquet(D7 / 'value.parquet').reindex_like(C)
C6 = pd.read_parquet(D6 / 'close_raw.parquet').reindex(index=C.index, columns=C.columns)
W0, W1 = pd.Timestamp('2024-01-02'), pd.Timestamp('2025-09-23')
OFF = {1, 2, 4, 5, 7}
# 2. 覆蓋：2020-01 起每個 V7 官方訊號日，以上市矩陣中的股票、近 60 日 value 平均取前 300（近 60 日無資料的上市股保守代入）；
#    到下個訊號日前一日的股日；有收盤者中 src 屬官方（1/2/4/5/7）的比率；另報上市但無收盤者
plan = pd.read_csv(O7 / 'plan_V7.csv')
sig = pd.DatetimeIndex(plan['signal']); sig = sig[sig >= '2020-01-01'].append(pd.DatetimeIndex(['2026-10-12']))
v60 = Vv.rolling(60, min_periods=1).mean()
for nm in ('listed_twse', 'listed_twse_strong'):
    Lm = pd.read_parquet(D7 / f'{nm}.parquet').reindex(index=C.index, columns=C.columns).fillna(False).astype(bool)
    tot = nof = miss = 0; tot_w = nof_w = 0; jan = [0, 0]
    for a, b in zip(sig[:-1], sig[1:]):
        L0 = Lm.loc[a]; vv = v60.loc[a].where(L0)
        nodata = L0 & vv.isna()
        top = set(vv.dropna().sort_values(ascending=False).index[:300]) | set(nodata.index[nodata])
        cols = sorted(top)
        dd = C.index[(C.index >= a) & (C.index < b)]
        c = C.loc[dd, cols]; s = S.loc[dd, cols]; l = Lm.loc[dd, cols]
        has = c.notna() & l
        off = s.isin(OFF) & has
        tot += int(has.values.sum()); nof += int((has & ~off).values.sum()); miss += int((l & c.isna()).values.sum())
        inw = (dd >= W0) & (dd <= W1)
        tot_w += int(has.values[inw].sum()); nof_w += int((has & ~off).values[inw].sum())
        inj = (dd >= W0) & (dd <= pd.Timestamp('2024-02-14'))
        jan[0] += int(has.values[inj].sum()); jan[1] += int((has & ~off).values[inj].sum())
    res[f'coverage_{nm}'] = dict(stockdays_with_price=tot, non_official=nof, pct_official=100 * (1 - nof / tot), listed_no_price=miss,
                                  window_stockdays=tot_w, window_non_official=nof_w, jan_feb2024=dict(n=jan[0], non_official=jan[1]))
# 3. 颱風日事件：exdiv_5y 中 TWSE、事件日＝颱風停市日 → 面板下一個日期的隱含參考價 close/(1+ret) 對 ref_price
ex = pd.read_csv('/home/user/stevenhsiehsheng-ctrl/tw-stock-screener/data/extras/exdiv_5y.csv.gz', dtype={'code': str}, parse_dates=['date'])
TY = pd.DatetimeIndex(['2024-07-24', '2024-07-25', '2024-10-02', '2024-10-03', '2024-10-31'])
R = pd.read_parquet(D7 / 'ret.parquet').reindex_like(C)
e = ex[ex.date.isin(TY) & (ex.market == 'TWSE') & ex.code.isin(C.columns)]
rows = []
for r in e.itertuples():
    nd = C.index[C.index > r.date]
    nd = [d for d in nd[:10] if np.isfinite(C.at[d, r.code])]
    if not nd: rows.append((r.code, str(r.date.date()), None, r.ref_price, None)); continue
    d = nd[0]; imp = C.at[d, r.code] / (1 + R.at[d, r.code])
    rows.append((r.code, str(r.date.date()), str(d.date()), r.ref_price, float(imp)))
ok = [x for x in rows if x[4] is not None and abs(x[4] / x[3] - 1) < 0.003]
res['typhoon'] = dict(n_events=len(rows), n_match=len(ok), mismatches=[x for x in rows if x not in ok][:20])
# 4. 補回的上市空洞股：V7 前 300（自寫引擎）中、V6 該日無收盤者；V7 vs V6 入選差異
import rb_core as B
p6 = pd.read_csv(AG / 'o_v6_price_panel_official' / 'plan_V6.csv', dtype=str)
P6 = {pd.Timestamp(s): set(x.split(',')) for s, x in zip(p6.signal, p6.picks)}
pl, _ = B.plan(B.official_days())
holes = {}; chg = []
for p in pl:
    d = p['d']; t = B.TPOS[d]
    v = B.V60[t]; px = B.Cn[t]
    base = np.isfinite(v) & (v >= 1e7) & np.isfinite(px) & (px >= 10)
    o = np.argsort(-np.where(base, v, -np.inf), kind='stable')[:min(300, int(base.sum()))]
    top = [B.COLS[j] for j in o if base[j]]
    if d >= pd.Timestamp('2024-01-01'):
        h = [c for c in top if not np.isfinite(C6.at[d, c])]
        if h: holes[str(d.date())] = h
    mine = {B.COLS[j] for j in p['picks']}
    if d in P6 and mine != P6[d]: chg.append((str(d.date()), sorted(mine - P6[d]), sorted(P6[d] - mine)))
res['holes_in_top300'] = holes; res['pick_changes_vs_V6'] = chg
json.dump(res, open(HERE / 'out' / 'gate.json', 'w'), ensure_ascii=False, indent=1)
print(json.dumps({k: v for k, v in res.items() if k != 'holes_in_top300'}, ensure_ascii=False, indent=1))
print({k: len(v) for k, v in holes.items()})
