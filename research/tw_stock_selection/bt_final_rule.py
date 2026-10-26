#!/usr/bin/env python3
"""最終規則（R5＋處置濾網）的風險輪廓、敏感度與核心／衛星配置，上市股 2021-01～2026-10。

規則：每月第一個 ≥ 11 日的交易日收盤，在 60 日均成交值前 300 名（且 ≥ NT$1,000 萬、股價 ≥ 10 元、非處置期間）中，
      依「收盤／52 週最高（還原）」與「近 3 個月營收年增率平均」的百分位平均分數，買前 30 檔、等權，持有一個月。
輸出：output/bt_final_rule.json
"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
from btlib import period_returns, bench_returns, ann, yearly, tstat

P = Path('/tmp/claude-0/-home-user--/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/panel')
X = Path('/home/user/stevenhsiehsheng-ctrl/tw-stock-screener/data/extras')
OUT = Path(__file__).resolve().parent / 'output'
adj = pd.read_parquet(P / 'adj.parquet'); C = pd.read_parquet(P / 'close_raw.parquet'); V = pd.read_parquet(P / 'value.parquet')
rev = pd.read_csv(X / 'rev_5y.csv.gz', dtype={'code': str}); rev = rev[rev['code'].isin(adj.columns)]; rev['p'] = pd.PeriodIndex(rev['ym'], freq='M')
YOY = rev.pivot_table(index='p', columns='code', values='yoy', aggfunc='last').sort_index()
YOY = YOY.reindex(pd.period_range(YOY.index.min(), YOY.index.max(), freq='M')); YOY3 = YOY.rolling(3, min_periods=3).mean()
cnt = YOY.notna().sum(axis=1); last_full = cnt[cnt >= 0.8 * cnt.median()].index.max()
ii = adj.index[adj.index >= pd.Timestamp('2021-01-01')]; s = pd.Series(ii, index=ii)
ds = [g[g.dt.day >= 11].iloc[0] for _, g in s.groupby([ii.year, ii.month]) if (g.dt.day >= 11).any()]
keep = [d for d in ds if YOY3.index.min() <= (pd.Period(d, 'M') - 1) <= last_full and YOY3.reindex([pd.Period(d, 'M') - 1]).notna().sum(axis=1).iloc[0] > 100]
dates = keep + [d for d in ds if d > keep[-1]][:1]
R = period_returns(adj, dates)
e50 = pd.read_csv('/home/user/jonatw/finlab-txd/data/raw/etf_0050.csv', parse_dates=['date']).set_index('date')['adj']; B = bench_returns(e50, dates)
mm = [pd.Period(d, 'M') - 1 for d in dates]
def bym(M): return pd.DataFrame(M.reindex(mm).to_numpy(), index=pd.DatetimeIndex(dates), columns=M.columns).reindex(columns=adj.columns)
yoy3, yoy1 = bym(YOY3), bym(YOY)
v60 = V.rolling(60).mean().reindex(dates); px = C.reindex(dates); base = (v60 >= 1e7) & (px >= 10)
t300 = base & (v60.where(base).rank(axis=1, ascending=False) <= 300)
s52 = (adj / adj.rolling(252).max()).reindex(dates)
dsp = pd.read_csv(X / 'disposal_5y.csv.gz', dtype={'code': str}, parse_dates=['start', 'end'])
inD = pd.DataFrame(False, index=pd.DatetimeIndex(dates), columns=adj.columns)
for _, r in dsp.iterrows():
    if r['code'] in inD.columns and pd.notna(r['start']) and pd.notna(r['end']):
        inD.loc[(inD.index >= r['start']) & (inD.index <= r['end']), r['code']] = True

def run(rev_sig, w52=0.5, n=30, rt=0.005):
    el = t300 & rev_sig.notna() & s52.notna() & ~inD
    sc = w52 * s52.where(el).rank(axis=1, pct=True) + (1 - w52) * rev_sig.where(el).rank(axis=1, pct=True)
    port, uni, to, prev, picks = {}, {}, [], None, {}
    for t in R.index:
        ok = el.loc[t] & R.loc[t].notna()
        if ok.sum() < 50: continue
        pick = list(sc.loc[t][ok].sort_values(ascending=False).index[:n])
        port[t] = R.loc[t][pick].mean(); uni[t] = R.loc[t][ok].mean(); picks[t] = pick
        if prev is not None: to.append(len(set(pick) - prev) / n)
        prev = set(pick)
    g = pd.Series(port); tov = float(np.mean(to))
    return g - tov * rt, pd.Series(uni), tov, picks

def profile(net, uni, tov):
    b = B.reindex(net.index); ex = net - b
    eq = (1 + net).cumprod(); eqb = (1 + b).cumprod()
    rel12 = (eq / eq.shift(12)) / (eqb / eqb.shift(12)) - 1
    return {'net': ann(net, dates), '0050': ann(b, dates), 'top300_EW': ann(uni, dates), 'turnover_per_rebalance': round(tov, 3),
            'annual_cost_drag_at_0.5pct_rt': round(tov * 12 * 0.005, 4), 'hit_rate_vs_0050': round(float((ex > 0).mean()), 3),
            'mean_excess_vs_0050_pct_m': round(float(ex.mean() * 100), 3), 't_vs_0050': tstat(ex), 't_vs_EW': tstat(net - uni),
            'excess_95ci_pct_m': [round(float((ex.mean() - 1.96 * ex.std() / np.sqrt(len(ex))) * 100), 2), round(float((ex.mean() + 1.96 * ex.std() / np.sqrt(len(ex))) * 100), 2)],
            'worst_period_pct': round(float(net.min() * 100), 1), 'worst_period_date': str(net.idxmin().date()), 'best_period_pct': round(float(net.max() * 100), 1),
            'worst_rolling12_relative_to_0050_pct': round(float(rel12.min() * 100), 1), 'tracking_error_ann': round(float(ex.std() * np.sqrt(12)), 3),
            'yearly_net_pct': yearly(net, dates), 'yearly_0050_pct': yearly(b, dates)}

def main():
    res = {}
    net, uni, tov, picks = run(yoy3)
    res['FINAL_R5_with_disposal_filter'] = profile(net, uni, tov)
    b = B.reindex(net.index)
    for w in [0.3, 0.5, 0.7]:
        bl = w * net + (1 - w) * b      # 每月再平衡的核心（0050）＋衛星（規則）
        res[f'BLEND_{int(w*100)}pct_rule_{int((1-w)*100)}pct_0050'] = {'ann': ann(bl, dates), 'yearly': yearly(bl, dates)}
    n1, u1, t1, _ = run(yoy1); res['SENS_single_month_yoy'] = profile(n1, u1, t1)
    for w in [0.33, 0.67]:
        nw, uw, tw, _ = run(yoy3, w52=w); res[f'SENS_weight_52w_{w}'] = profile(nw, uw, tw)
    for k in ['SENS_single_month_yoy', 'SENS_weight_52w_0.33', 'SENS_weight_52w_0.67']:
        res[k] = {kk: res[k][kk] for kk in ['net', 'mean_excess_vs_0050_pct_m', 't_vs_0050', 't_vs_EW', 'turnover_per_rebalance']}
    last = max(picks); res['latest_picks'] = {'entry_date': str(last.date()), 'codes': picks[last]}
    (OUT / 'bt_final_rule.json').write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding='utf-8')
    print(json.dumps(res, ensure_ascii=False, indent=1)[:6000])


if __name__ == '__main__':
    main()
