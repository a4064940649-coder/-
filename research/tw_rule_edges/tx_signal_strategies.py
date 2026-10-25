#!/usr/bin/env python3
"""台指期訊號策略評估（延續 tx_futures_tests）：外資淨未平倉變化、外資 20 日流量、動能控制、隔夜持有，含成本與子期間。"""
import json, math
from pathlib import Path
import numpy as np, pandas as pd

d = Path('/home/user/chijiun/tx-flow-trend-backtest/input'); out = Path('/home/user/-/research/tw_rule_edges/output'); out.mkdir(exist_ok=True)
allc = pd.read_csv(d / 'all_contracts.csv', encoding='utf-8-sig', parse_dates=['date']).sort_values(['contract', 'date'])
near = pd.read_csv(d / 'prices.csv', encoding='utf-8-sig', parse_dates=['date']).sort_values('date')
allc['prev_close'] = allc.groupby('contract')['close'].shift(1)
allc['cc'] = allc['close'] / allc['prev_close'] - 1
allc['overnight'] = allc['open'] / allc['prev_close'] - 1
m = allc.merge(near[['date', 'contract']], on=['date', 'contract']).set_index('date').sort_index().dropna(subset=['cc'])
inst = pd.read_csv(d / 'institutional.csv', encoding='utf-8-sig'); inst['date'] = pd.to_datetime(inst['日期'])
oi = inst.pivot_table(index='date', columns='身份別', values='多空未平倉口數淨額', aggfunc='sum').sort_index()
fl = inst.pivot_table(index='date', columns='身份別', values='多空交易口數淨額', aggfunc='sum').sort_index()
f_oi_chg = oi['外資及陸資'].reindex(m.index).diff()
f_flow20 = fl['外資及陸資'].reindex(m.index).rolling(20).sum()
cc = m['cc']; nxt = cc.shift(-1); same = cc  # 當日報酬（動能控制）
def tdiff(a, b):
    a, b = a.dropna(), b.dropna(); return float((a.mean() - b.mean()) / math.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b)))
def ann(r, cost_series=None):
    r = r.dropna(); eq = (1 + r).cumprod(); yrs = len(r) / 245; vol = r.std() * math.sqrt(245); cagr = eq.iloc[-1] ** (1 / yrs) - 1
    return {'CAGR': round(float(cagr), 4), 'vol': round(float(vol), 4), 'sharpe': round(float(cagr / vol), 2) if vol > 0 else None, 'maxDD': round(float((eq / eq.cummax() - 1).min()), 4), 'n_days': int(len(r))}
res = {}
# 1) 分位差異 t 值
for nm, s in [('外資淨未平倉變化', f_oi_chg), ('外資20日淨流量', f_flow20)]:
    z = s.dropna(); q = pd.qcut(z, 3, labels=['low', 'mid', 'high'])
    hi, lo = nxt.reindex(z.index)[q == 'high'], nxt.reindex(z.index)[q == 'low']
    res[nm] = {'H_minus_L_bps': round(float((hi.mean() - lo.mean()) * 1e4), 2), 't_diff': round(tdiff(hi, lo), 2), 'n_high': int(hi.count())}
    # 動能控制：在當日報酬分位內比較
    sq = pd.qcut(same.reindex(z.index), 3, labels=['dn', 'flat', 'up'])
    ctrl = {}
    for lab in ['dn', 'flat', 'up']:
        mask = (sq == lab)
        h2, l2 = nxt.reindex(z.index)[mask & (q == 'high')], nxt.reindex(z.index)[mask & (q == 'low')]
        ctrl[lab] = {'H_minus_L_bps': round(float((h2.mean() - l2.mean()) * 1e4), 2), 't': round(tdiff(h2, l2), 2), 'n': int(h2.count() + l2.count())}
    res[nm]['within_same_day_return_tercile'] = ctrl
# 2) 策略：訊號 >0 時次日持有多單，否則空手；成本每趟來回 3 點
cost_rt = 3.0 / m['prev_close']
strats = {}
for nm, s in [('外資淨未平倉變化>0', f_oi_chg > 0), ('外資20日淨流量>0', f_flow20 > 0), ('當日漲(動能)>0', same > 0), ('外資OI變化>0 且 當日漲', (f_oi_chg > 0) & (same > 0))]:
    pos = s.astype(float).reindex(m.index).fillna(0).shift(1).fillna(0)   # 訊號日盤後公布 → 次日持有
    sw = pos.diff().abs().fillna(0)
    r = pos * cc - sw * cost_rt / 2
    strats[nm] = {**ann(r), 'time_in_market': round(float(pos.mean()), 3), 'switches': int(sw.sum())}
    for lab, sl in [('2018-08~2021', slice('2018-08-01', '2021-12-31')), ('2022~2026', slice('2022-01-01', '2026-12-31'))]:
        strats[nm][lab] = ann(r.loc[sl])
strats['buy_and_hold_TX'] = ann(cc.loc['2018-08-01':])
for lab, sl in [('2018-08~2021', slice('2018-08-01', '2021-12-31')), ('2022~2026', slice('2022-01-01', '2026-12-31'))]:
    strats['buy_and_hold_TX'][lab] = ann(cc.loc[sl])
# 3) 隔夜持有策略（13:45 進、08:45 出）淨成本，與 B&H 同期比較
on = m['overnight']; on_net = on - cost_rt
strats['隔夜持有(同契約, 淨3點/日)'] = ann(on_net.loc['2016-10-06':])
strats['buy_and_hold_TX_全期'] = ann(cc)
res['strategies'] = strats
# 4) 現貨 TAIEX 同期隔夜 vs 日內（Yahoo ^TWII，供對照資料品質）
tw = pd.read_csv('/home/user/jonatw/finlab-txd/data/raw/taiex_twii.csv', parse_dates=['date']).set_index('date').sort_index().loc['2016-10-06':'2026-10-05']
res['TAIEX_cash_same_period'] = {'overnight_bps': round(float((tw['open'] / tw['close'].shift(1) - 1).mean() * 1e4), 2), 'intraday_bps': round(float((tw['close'] / tw['open'] - 1).mean() * 1e4), 2),
                                 'TX_overnight_bps': round(float(on.mean() * 1e4), 2), 'TX_intraday_bps': round(float((m['close'] / m['open'] - 1).mean() * 1e4), 2)}
(out / 'tx_signal_strategies.json').write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding='utf-8')
print(json.dumps(res, ensure_ascii=False, indent=1))
