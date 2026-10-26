#!/usr/bin/env python3
"""大範圍訊號搜尋（上市股），事前固定的「先挑選、再驗證」流程，避免資料探勘。

A 價格／技術／產業訊號（月底換股）：
  挑選期 2005-01～2019-04，驗證期 2019-05～2026-10。
  挑選規則：挑選期「前 300 名流動性中 30 檔」對同池等權 t ≥ 2 且全體五分位價差 t ≥ 2 的訊號 → 等權百分位合成。
B 基本面／籌碼訊號（月營收換股日，每月第一個 ≥ 11 日交易日）：
  挑選期 2021-03～2023-12，驗證期 2024-01～2026-10，同樣規則（挑選期 t ≥ 2）。
所有組合：前 300 名流動性中 30 檔、等權、扣 0.5% 來回成本；對手為 0050 含息。
輸出 output/bt_factor_search.json
"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
from btlib import month_ends, period_returns, bench_returns, run_sort, summarize

S = Path('/tmp/claude-0/-home-user--/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad')
P = S / 'panel'; X = Path('/home/user/stevenhsiehsheng-ctrl/tw-stock-screener/data')
OUT = Path(__file__).resolve().parent / 'output'
adj = pd.read_parquet(P / 'adj.parquet'); C = pd.read_parquet(P / 'close_raw.parquet'); V = pd.read_parquet(P / 'value.parquet'); R1 = pd.read_parquet(P / 'ret.parquet')
e50 = pd.read_csv('/home/user/jonatw/finlab-txd/data/raw/etf_0050.csv', parse_dates=['date']).set_index('date')['adj']
A = adj.ffill(limit=5)
V60 = V.rolling(60).mean()
# 產業別（現行分類）
lst = pd.read_csv(X / 'stock_list.csv', dtype=str); tw = pd.read_csv(S / 'piptest/twstock_wheel/twstock/codes/twse_equities.csv', dtype=str)
ind = dict(zip(tw['code'], tw['group'])); ind.update({c: i for c, i in zip(lst['code'], lst['industry']) if isinstance(i, str)})
IND = pd.Series({c: ind.get(c) for c in adj.columns})
def ind_avg(M):
    """每列：股票所屬產業的平均值（至少 3 檔）。"""
    out = pd.DataFrame(np.nan, index=M.index, columns=M.columns)
    for g, cols in IND.dropna().groupby(IND.dropna()).groups.items():
        cols = list(cols)
        if len(cols) < 3: continue
        m = M[cols].mean(axis=1, skipna=True); n = M[cols].notna().sum(axis=1)
        vals = np.where((n >= 3).to_numpy(), m.to_numpy(), np.nan)
        out[cols] = np.repeat(vals[:, None], len(cols), axis=1)
    return out.where(M.notna())

# 日頻訊號
ma = {n: A.rolling(n, min_periods=int(n * 0.8)).mean() for n in (20, 60, 120, 240)}
gain = R1.clip(lower=0).rolling(14).mean(); loss = (-R1.clip(upper=0)).rolling(14).mean()
mom6 = A.shift(21) / A.shift(126) - 1
daily = {
    'p_52w_high': adj / adj.rolling(252).max(),
    'p_mom_12_1': A.shift(21) / A.shift(252) - 1,
    'p_mom_6_1': mom6,
    'p_mom_3m': A / A.shift(63) - 1,
    'p_reversal_1m': -(A / A.shift(21) - 1),
    'p_low_vol_60d': -R1.rolling(60, min_periods=50).std(),
    'p_low_MAX_21d': -R1.rolling(21, min_periods=15).max(),
    't_ma_alignment': sum((A > ma[n]).astype(float) for n in (20, 60, 120, 240)) + (ma[20] > ma[60]).astype(float) + (ma[60] > ma[120]).astype(float),
    't_close_over_ma200': A / A.rolling(200, min_periods=160).mean(),
    't_breakout_20d': A / A.rolling(20).max(),
    't_volume_surge_20_120': V.rolling(20).mean() / V.rolling(120, min_periods=100).mean(),
    't_rsi14': 100 - 100 / (1 + gain / loss.replace(0, np.nan)),
    't_dist_52w_low': adj / adj.rolling(252).min(),
    'l_illiquidity_amihud': (R1.abs() / V).rolling(60, min_periods=50).mean(),
    'i_industry_mom_6_1': None, 'i_industry_52w': None,
}
daily['t_ma_alignment'] = daily['t_ma_alignment'].where(ma[240].notna())
daily['i_industry_mom_6_1'] = ind_avg(mom6)
daily['i_industry_52w'] = ind_avg(daily['p_52w_high'])
print('daily signals ready', flush=True)

def universe(dates):
    v = V60.reindex(dates); px = C.reindex(dates)
    base = (v >= 1e7) & (px >= 10)
    return base, base & (v.where(base).rank(axis=1, ascending=False) <= 300)

def evaluate(sigs, dates, R, B, windows, label_prefix):
    base, top300 = universe(dates); out = {}
    for k, s in sigs.items():
        q = run_sort(s, R, base & s.notna()); t = run_sort(s, R, top300 & s.notna(), top_n=30)
        for wn, (a, b) in windows.items():
            sq = summarize(q, dates, B, k, a, b); st = summarize(t, dates, B, k, a, b)
            out[f'{label_prefix}{k}|{wn}'] = {'Q5_Q1_pct_m': sq['Q5_minus_Q1_pct_per_period'], 't_Q5_Q1': sq['t_Q5_Q1'],
                'top30_net_CAGR': st['best_net'].get('CAGR'), 'top30_maxDD': st['best_net'].get('maxDD'), 'top300_EW_CAGR': st['universe'].get('CAGR'),
                '0050_CAGR': st['bench'].get('CAGR'), 't_top30_vs_EW': st['t_net_vs_universe'], 't_top30_vs_0050': st['t_net_vs_bench'], 'years_beat_0050': st.get('years_beat_bench')}
    return out

def composite(sigs, names, dates):
    base, top300 = universe(dates)
    el = top300.copy()
    for n in names: el &= sigs[n].notna()
    return sum(sigs[n].where(el).rank(axis=1, pct=True) for n in names) / len(names), el

res = {}
# ---------- A：月底、價格／技術／產業 ----------
me = month_ends(adj.index); R = period_returns(adj, me); B = bench_returns(e50, me)
sigA = {k: v.reindex(me) for k, v in daily.items()}
WA = {'select_2005_2019-04': ('2005-01-01', '2019-04-30'), 'validate_2019-05_2026-10': ('2019-05-01', '2026-10-31')}
resA = evaluate(sigA, me, R, B, WA, 'A:')
res.update(resA)
selA = [k for k in sigA if resA[f'A:{k}|select_2005_2019-04']['t_top30_vs_EW'] >= 2 and resA[f'A:{k}|select_2005_2019-04']['t_Q5_Q1'] >= 2]
print('A selected (select period only):', selA, flush=True)
if selA:
    cA, elA = composite(sigA, selA, me)
    t = run_sort(cA, R, elA, top_n=30)
    for wn, (a, b) in WA.items():
        s = summarize(t, me, B, 'A composite', a, b)
        res[f'A:COMPOSITE|{wn}'] = {'signals': selA, 'net_CAGR': s['best_net'].get('CAGR'), 'maxDD': s['best_net'].get('maxDD'), '0050_CAGR': s['bench'].get('CAGR'),
                                    'EW_CAGR': s['universe'].get('CAGR'), 't_vs_EW': s['t_net_vs_universe'], 't_vs_0050': s['t_net_vs_bench'], 'years_beat_0050': s.get('years_beat_bench'), 'yearly': s.get('yearly_net_pct'), 'yearly_0050': s.get('yearly_bench_pct')}

# ---------- B：營收換股日、基本面／籌碼（含價格訊號，讓組合可以混用） ----------
import bt_final_rule as fr
dB = fr.dates; RB = fr.R; BB = fr.B
mm = [pd.Period(d, 'M') - 1 for d in dB]
def bym(M): return pd.DataFrame(M.reindex(mm).to_numpy(), index=pd.DatetimeIndex(dB), columns=M.columns).reindex(columns=adj.columns)
YOY, YOY3 = fr.YOY, fr.YOY3
sigB = {k: v.reindex(dB) for k, v in daily.items()}
sigB['f_rev_yoy'] = bym(YOY); sigB['f_rev_yoy3'] = bym(YOY3)
sigB['f_rev_accel'] = bym(YOY - YOY.shift(3))
REV = fr.rev.pivot_table(index='p', columns='code', values='revenue', aggfunc='last').sort_index().reindex(YOY.index)
sigB['f_rev_12m_high_ratio'] = bym(REV / REV.shift(1).rolling(12, min_periods=12).max())
sigB['i_industry_rev_yoy3'] = ind_avg(sigB['f_rev_yoy3'])
pe = pd.read_csv(X / 'extras/pe_5y.csv.gz', dtype={'code': str}, parse_dates=['date'])
def snap(col):
    W = pe.pivot_table(index='date', columns='code', values=col, aggfunc='last').sort_index()
    return W.reindex(pd.DatetimeIndex(dB), method='ffill').reindex(columns=adj.columns)
PEm, PBm = snap('pe'), snap('pb')
sigB['v_earnings_yield'] = (1 / PEm).where(PEm > 0); sigB['v_book_to_price'] = (1 / PBm).where(PBm > 0); sigB['v_div_yield'] = snap('yield')
inst = pd.read_csv(X / 'extras/inst_5y.csv.gz', dtype={'code': str}, parse_dates=['date']); inst = inst[inst['code'].isin(adj.columns)]
idx = adj.index[(adj.index >= inst['date'].min()) & (adj.index <= inst['date'].max())]
volsh = (V / C).reindex(idx); vol20 = volsh.rolling(20, min_periods=15).sum()
for col, name in [('trust', 'c_trust_20d'), ('foreign', 'c_foreign_20d')]:
    M = inst.pivot_table(index='date', columns='code', values=col, aggfunc='sum').reindex(idx).reindex(columns=adj.columns) * 1000
    M = M.where(volsh.notna()).fillna(0).where(volsh.notna())
    sigB[name] = (M.rolling(20, min_periods=15).sum() / vol20).shift(1).reindex(pd.DatetimeIndex(dB), method='ffill')
dt = pd.read_csv(S / 'scr_extra/daytrade_5y.csv.gz', dtype={'code': str}, parse_dates=['date']); dt = dt[dt['code'].isin(adj.columns)]
DT = dt.pivot_table(index='date', columns='code', values='dt_ratio', aggfunc='last').reindex(columns=adj.columns)
sigB['c_low_daytrade_ratio_20d'] = (-DT.rolling(20, min_periods=10).mean().shift(1)).reindex(pd.DatetimeIndex(dB), method='ffill')
_dt = sigB['c_low_daytrade_ratio_20d']; sigB['c_low_daytrade_ratio_20d'] = _dt[_dt.index >= DT.index.min() + pd.Timedelta(days=40)].reindex(_dt.index)
for k in list(sigB):
    sigB[k] = sigB[k].where(~fr.inD)     # 處置股不可投資
WB = {'select_2021-03_2023-12': ('2021-03-01', '2023-12-31'), 'validate_2024-01_2026-10': ('2024-01-01', '2026-10-31'), 'full': ('2021-03-01', '2026-10-31')}
resB = evaluate(sigB, dB, RB, BB, WB, 'B:')
res.update(resB)
selB = [k for k in sigB if resB[f'B:{k}|select_2021-03_2023-12']['t_top30_vs_EW'] >= 2 and resB[f'B:{k}|select_2021-03_2023-12']['t_Q5_Q1'] >= 2]
print('B selected (select period only):', selB, flush=True)
def comp_report(names, tag):
    c, el = composite(sigB, names, dB)
    t = run_sort(c, RB, el, top_n=30)
    for wn, (a, b) in WB.items():
        s = summarize(t, dB, BB, tag, a, b)
        res[f'B:{tag}|{wn}'] = {'signals': names, 'net_CAGR': s['best_net'].get('CAGR'), 'maxDD': s['best_net'].get('maxDD'), '0050_CAGR': s['bench'].get('CAGR'),
                                'EW_CAGR': s['universe'].get('CAGR'), 't_vs_EW': s['t_net_vs_universe'], 't_vs_0050': s['t_net_vs_bench'],
                                'years_beat_0050': s.get('years_beat_bench'), 'yearly': s.get('yearly_net_pct'), 'yearly_0050': s.get('yearly_bench_pct')}
if selB: comp_report(selB, 'COMPOSITE_selected')
comp_report(['p_52w_high', 'f_rev_yoy3'], 'REFERENCE_final_rule')
(OUT / 'bt_factor_search.json').write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding='utf-8')
for k, v in res.items():
    if 'COMPOSITE' in k or 'REFERENCE' in k:
        print(k, {kk: vv for kk, vv in v.items() if kk not in ('yearly', 'yearly_0050')})
    else:
        print(f"{k:60s} Q5-Q1 {v['Q5_Q1_pct_m']:>7}%/m t={v['t_Q5_Q1']:>6} | top30 {v['top30_net_CAGR']} vsEW t={v['t_top30_vs_EW']} vs0050 t={v['t_top30_vs_0050']} | EW {v['top300_EW_CAGR']} 0050 {v['0050_CAGR']}")
