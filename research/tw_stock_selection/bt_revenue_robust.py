#!/usr/bin/env python3
"""月營收 × 52 週高點組合（R5）與月營收規則的穩健性檢定，上市股 2021-01～2026-10。

檢定項目（均為事前列出，全部報告）：
  1 子期間：2021-01～2023-12 vs 2024-01～2026-10
  2 換股日 11 日 vs 15 日（已在 bt_revenue）；此處另做「52 週訊號延後一天」
  3 漲停買不到：進場日收盤漲幅 ≥ 9.5% 的股票改選下一名
  4 檔數 10 / 20 / 30 / 50；流動性範圍前 200 / 300 / 500 名
  5 去極值：每期剔除報酬最高的 1 檔後重算（檢查是否靠少數飆股）
  6 成本：0.5% / 1% / 1.5% 來回
  7 容量：入選股 60 日均成交值中位數、單檔 5% 成交量上限推算的資金規模
  8 隨機對照：同一流動性範圍內隨機抽 30 檔 × 1,000 次，看 R5 落在分布哪裡
"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
from btlib import period_returns, bench_returns, ann, yearly, tstat

P = Path('/tmp/claude-0/-home-user--/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/panel')
OUT = Path(__file__).resolve().parent / 'output'
adj = pd.read_parquet(P / 'adj.parquet'); C = pd.read_parquet(P / 'close_raw.parquet'); V = pd.read_parquet(P / 'value.parquet'); R1d = pd.read_parquet(P / 'ret.parquet')
rev = pd.read_csv('/home/user/stevenhsiehsheng-ctrl/tw-stock-screener/data/extras/rev_5y.csv.gz', dtype={'code': str})
rev = rev[rev['code'].isin(adj.columns)]; rev['p'] = pd.PeriodIndex(rev['ym'], freq='M')
REV = rev.pivot_table(index='p', columns='code', values='revenue', aggfunc='last').sort_index()
YOY = rev.pivot_table(index='p', columns='code', values='yoy', aggfunc='last').sort_index()
full = pd.period_range(REV.index.min(), REV.index.max(), freq='M'); REV = REV.reindex(full); YOY = YOY.reindex(full)
YOY3 = YOY.rolling(3, min_periods=3).mean()
cnt = REV.notna().sum(axis=1); last_full = cnt[cnt >= 0.8 * cnt.median()].index.max()
e50 = pd.read_csv('/home/user/jonatw/finlab-txd/data/raw/etf_0050.csv', parse_dates=['date']).set_index('date')['adj']
S52 = adj / adj.rolling(252).max(); V60 = V.rolling(60).mean()

def rebalance_dates(day=11):
    idx = adj.index[adj.index >= pd.Timestamp('2021-01-01')]; s = pd.Series(idx, index=idx)
    ds = [g[g.dt.day >= day].iloc[0] for _, g in s.groupby([idx.year, idx.month]) if (g.dt.day >= day).any()]
    keep = [d for d in ds if YOY.index.min() <= (pd.Period(d, 'M') - 1) <= last_full]
    return keep + [d for d in ds if d > keep[-1]][:1]

dates = rebalance_dates(11)
R = period_returns(adj, dates)
B = bench_returns(e50, dates)
m = [pd.Period(d, 'M') - 1 for d in dates]
yoy3 = pd.DataFrame(YOY3.reindex(m).to_numpy(), index=pd.DatetimeIndex(dates), columns=YOY3.columns).reindex(columns=adj.columns)
v60 = V60.reindex(dates); px = C.reindex(dates); entry_ret = R1d.reindex(dates)
base = (v60 >= 1e7) & (px >= 10)
rank_liq = v60.where(base).rank(axis=1, ascending=False)

def combo_score(liq_n, s52_lag=0):
    s52 = (S52.shift(s52_lag) if s52_lag else S52).reindex(dates)
    el = base & (rank_liq <= liq_n) & yoy3.notna() & s52.notna()
    pr = lambda M: M.where(el).rank(axis=1, pct=True)
    return (pr(s52) + pr(yoy3)) / 2, el

def backtest(score, el, n=30, rt=0.005, no_limit_up=False, trim_best=0):
    port, uni, to, prev, names, medv = {}, {}, [], None, [], []
    for t in R.index:
        ok = el.loc[t] & score.loc[t].notna() & R.loc[t].notna()
        if ok.sum() < 50: continue
        s = score.loc[t][ok].sort_values(ascending=False)
        if no_limit_up:
            lu = entry_ret.loc[t].reindex(s.index) >= 0.095
            s = s[~lu.fillna(False)]
        pick = list(s.index[:n]); r = R.loc[t][pick].sort_values()
        if trim_best: r = r.iloc[:-trim_best]
        port[t] = r.mean(); uni[t] = R.loc[t][ok].mean(); names.append(pick); medv.append(float(v60.loc[t][pick].median()))
        if prev is not None: to.append(len(set(pick) - prev) / len(pick))
        prev = set(pick)
    g = pd.Series(port); u = pd.Series(uni); tov = float(np.mean(to)); net = g - tov * rt; b = B.reindex(net.index)
    return {'net': net, 'gross': g, 'uni': u, 'bench': b, 'turnover': tov, 'median_v60_of_picks': float(np.median(medv)), 'names': names}

def stats(x, start=None, end=None):
    sl = lambda s: s[(s.index >= pd.Timestamp(start or '1900')) & (s.index <= pd.Timestamp(end or '2100'))]
    net, u, b = sl(x['net']), sl(x['uni']), sl(x['bench'])
    a = ann(net, dates)
    return {'net_CAGR': a.get('CAGR'), 'maxDD': a.get('maxDD'), 'vol': a.get('vol'), 'universe_CAGR': ann(u, dates).get('CAGR'), '0050_CAGR': ann(b, dates).get('CAGR'),
            'net_minus_universe_pct_m': round(float((net - u).mean() * 100), 3), 't_vs_universe': tstat(net - u),
            'net_minus_0050_pct_m': round(float((net - b).mean() * 100), 3), 't_vs_0050': tstat(net - b), 'turnover': round(x['turnover'], 3), 'periods': int(len(net))}

res = {}
sc, el = combo_score(300)
base_bt = backtest(sc, el)
res['R5_base'] = stats(base_bt)
res['R5_base']['yearly_net'] = yearly(base_bt['net'], dates); res['R5_base']['yearly_0050'] = yearly(base_bt['bench'], dates)
res['1_sub_2021_2023'] = stats(base_bt, '2021-01-01', '2023-12-31')
res['1_sub_2024_2026'] = stats(base_bt, '2024-01-01', '2026-12-31')
sc1, el1 = combo_score(300, s52_lag=1); res['2_s52_lag1'] = stats(backtest(sc1, el1))
res['3_no_limit_up_entry'] = stats(backtest(sc, el, no_limit_up=True))
for n in [10, 20, 50]: res[f'4_n{n}'] = stats(backtest(sc, el, n=n))
for L in [200, 500]:
    scL, elL = combo_score(L); res[f'4_liq_top{L}'] = stats(backtest(scL, elL))
res['5_trim_best_1'] = stats(backtest(sc, el, trim_best=1))
res['5_trim_best_3'] = stats(backtest(sc, el, trim_best=3))
for rt in [0.01, 0.015]: res[f'6_cost_{rt}'] = stats(backtest(sc, el, rt=rt))
mv = base_bt['median_v60_of_picks']
res['7_capacity'] = {'median_60d_avg_traded_value_of_picks_NTD': round(mv), 'capacity_NTD_if_each_position_le_5pct_of_ADV_30_names': round(mv * 0.05 * 30),
                     'note': '每檔部位 ≤ 日均成交值 5%，一次進場；分批（3～5 日）可放大 3～5 倍'}
# 8 隨機對照
rng = np.random.default_rng(42); sims = []
okmat = {t: list((el.loc[t] & R.loc[t].notna())[lambda s: s].index) for t in base_bt['net'].index}
for _ in range(1000):
    rr = pd.Series({t: R.loc[t][list(rng.choice(okmat[t], size=30, replace=False))].mean() for t in base_bt['net'].index}) - base_bt['turnover'] * 0.005
    sims.append(ann(rr, dates)['CAGR'])
sims = np.array(sims)
res['8_random30_same_universe'] = {'sim_CAGR_median': round(float(np.median(sims)), 4), 'sim_CAGR_p95': round(float(np.quantile(sims, 0.95)), 4), 'sim_CAGR_max': round(float(sims.max()), 4),
                                   'R5_CAGR': res['R5_base']['net_CAGR'], 'share_of_random_ge_R5': round(float((sims >= res['R5_base']['net_CAGR']).mean()), 4)}
# 其他規則的子期間（R1 Q5、R3、R4 已在 bt_revenue；此處補 R3 子期間）
sc3 = yoy3.where(base & (rank_liq <= 300))
r3 = backtest(sc3, base & (rank_liq <= 300) & yoy3.notna())
res['R3_sub_2021_2023'] = stats(r3, '2021-01-01', '2023-12-31'); res['R3_sub_2024_2026'] = stats(r3, '2024-01-01', '2026-12-31')
s52only = S52.reindex(dates).where(base & (rank_liq <= 300))
r52 = backtest(s52only, base & (rank_liq <= 300) & S52.reindex(dates).notna())
res['R5b_sub_2021_2023'] = stats(r52, '2021-01-01', '2023-12-31'); res['R5b_sub_2024_2026'] = stats(r52, '2024-01-01', '2026-12-31')
# 最新一期入選名單（供讀者對照，非投資建議）
res['latest_picks'] = {'date': str(base_bt['net'].index[-1].date()), 'codes': base_bt['names'][-1]}
(OUT / 'bt_revenue_robust.json').write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding='utf-8')
for k, v in res.items():
    if k in ('latest_picks',): continue
    print(k, json.dumps({kk: vv for kk, vv in v.items() if not kk.startswith('yearly')}, ensure_ascii=False))
print('yearly', res['R5_base']['yearly_net'], res['R5_base']['yearly_0050'])
