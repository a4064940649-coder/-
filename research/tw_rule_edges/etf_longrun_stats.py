#!/usr/bin/env python3
"""0050／0056／00631L 含息還原序列與加權指數的長期統計：年化、回撤、任意 N 年持有報酬分布、定期定額 vs 單筆、MA200 擇時含息版。"""
import json, math
from pathlib import Path
import numpy as np, pandas as pd
raw = Path('/home/user/jonatw/finlab-txd/data/raw'); out = Path('/home/user/-/research/tw_rule_edges/output'); out.mkdir(exist_ok=True)
def ser(name, col):
    df = pd.read_csv(raw / name, parse_dates=['date']).set_index('date').sort_index(); return df[col].astype(float)
res = {}
def stats(s, label):
    s = s.dropna(); r = s.pct_change().dropna(); yrs = (s.index[-1] - s.index[0]).days / 365.25
    cagr = (s.iloc[-1] / s.iloc[0]) ** (1 / yrs) - 1; dd = (s / s.cummax() - 1)
    o = {'start': str(s.index[0].date()), 'end': str(s.index[-1].date()), 'years': round(yrs, 1), 'CAGR': round(float(cagr), 4), 'vol': round(float(r.std() * math.sqrt(245)), 4),
         'maxDD': round(float(dd.min()), 4), 'maxDD_date': str(dd.idxmin().date())}
    # 年度報酬
    yr = s.groupby(s.index.year).last(); yr_ret = yr.pct_change().dropna(); o['yearly_pct'] = {int(k): round(float(v * 100), 1) for k, v in yr_ret.items()}
    # 任意 N 年持有（每日起點）
    for N in (1, 3, 5, 10):
        fwd = s.shift(-245 * N) / s - 1; fwd = fwd.dropna()
        if len(fwd) > 100:
            o[f'hold_{N}y'] = {'n_start_dates': int(len(fwd)), 'min_total_pct': round(float(fwd.min() * 100), 1), 'p10_pct': round(float(fwd.quantile(.1) * 100), 1), 'median_pct': round(float(fwd.median() * 100), 1),
                               'max_total_pct': round(float(fwd.max() * 100), 1), 'prob_loss_pct': round(float((fwd < 0).mean() * 100), 1), 'min_CAGR_pct': round(float(((1 + fwd.min()) ** (1 / N) - 1) * 100), 2)}
    # 定期定額（每月首個交易日投 1 單位）vs 同期單筆
    m = s.resample('MS').first().dropna(); units = (1 / m).cumsum(); value = units * m
    invested = pd.Series(np.arange(1, len(m) + 1), index=m.index)
    dca_mult = value / invested
    o['DCA_vs_lumpsum'] = {'DCA_final_multiple': round(float(dca_mult.iloc[-1]), 3), 'DCA_IRR_approx_pct': None, 'lump_sum_multiple': round(float(s.iloc[-1] / s.iloc[0]), 3)}
    # 近似 IRR（月現金流）
    cf = [-1.0] * len(m); total = float(value.iloc[-1])
    lo, hi = -0.99, 1.0
    for _ in range(100):
        mid = (lo + hi) / 2; npv = sum(c / (1 + mid) ** (i / 12) for i, c in enumerate(cf)) + total / (1 + mid) ** ((len(cf) - 1) / 12)
        if npv > 0: lo = mid
        else: hi = mid
    o['DCA_vs_lumpsum']['DCA_IRR_approx_pct'] = round(mid * 100, 2)
    # MA200 擇時（含息），成本 0.2%/次（ETF 稅 0.1% + 手續費）
    ma = s.rolling(200).mean(); pos = (s > ma).astype(int).shift(1).fillna(0); sw = pos.diff().abs().fillna(0)
    strat = (1 + pos * r.reindex(s.index).fillna(0) - sw * 0.002).cumprod()
    o['MA200_timing_vs_hold'] = {'timing_CAGR': round(float(strat.iloc[-1] ** (1 / yrs) - 1), 4), 'timing_maxDD': round(float((strat / strat.cummax() - 1).min()), 4), 'hold_CAGR': o['CAGR'], 'hold_maxDD': o['maxDD'], 'switches': int(sw.sum())}
    res[label] = o
stats(ser('etf_0050.csv', 'adj'), '0050_total_return')
stats(ser('etf_0056.csv', 'adj'), '0056_total_return')
stats(ser('etf_00631L.csv', 'adj'), '00631L_total_return')
tw = ser('taiex_twii.csv', 'close'); stats(tw.loc['2003-07-01':], 'TAIEX_price_since_2003_07'); stats(tw, 'TAIEX_price_since_1999')
# 0050 vs TAIEX 價格：隱含股息貢獻
a = res['0050_total_return']['CAGR']; b = res['TAIEX_price_since_2003_07']['CAGR']; res['implied_dividend_contribution_pp'] = round((a - b) * 100, 2)
(out / 'etf_longrun_stats.json').write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding='utf-8')
for k, v in res.items():
    if isinstance(v, dict): print(k, {kk: v[kk] for kk in ['start', 'end', 'CAGR', 'vol', 'maxDD', 'maxDD_date']}, '| hold10y', v.get('hold_10y'), '| DCA', v['DCA_vs_lumpsum'], '| MA200', v['MA200_timing_vs_hold'])
    else: print(k, v)
