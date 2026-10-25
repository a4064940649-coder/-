#!/usr/bin/env python3
"""台指期（TX）日線層級檢定：隔夜 vs 日內、趨勢過濾、三大法人期貨未平倉訊號。
資料：ChiJiun/tx-flow-trend-backtest（期交所每日行情與三大法人期貨部位）。
用法：python3 tx_futures_tests.py --dir <repo>/input --out ./output
成本：每趟來回預設 3 點（≈期交稅 0.002%×2×契約值/200 + 手續費）。
"""
import argparse, json, math
from pathlib import Path
import numpy as np, pandas as pd

def t(x):
    x = pd.Series(x).dropna(); return float(x.mean() / (x.std(ddof=1) / math.sqrt(len(x)))) if len(x) > 2 else float('nan')

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--dir', required=True); ap.add_argument('--out', default='./output'); ap.add_argument('--rt-cost-pts', type=float, default=3.0)
    a = ap.parse_args(); d = Path(a.dir); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    allc = pd.read_csv(d / 'all_contracts.csv', encoding='utf-8-sig', parse_dates=['date'])
    near = pd.read_csv(d / 'prices.csv', encoding='utf-8-sig', parse_dates=['date']).sort_values('date').set_index('date')
    # 同一契約的隔夜／日內報酬（避免換月基差污染）：用 all_contracts 計算每契約逐日
    allc = allc.sort_values(['contract', 'date'])
    allc['prev_close'] = allc.groupby('contract')['close'].shift(1)
    allc['overnight'] = allc['open'] / allc['prev_close'] - 1
    allc['intraday'] = allc['close'] / allc['open'] - 1
    allc['cc'] = allc['close'] / allc['prev_close'] - 1
    # 取近月：以 prices.csv 的契約為準
    near_c = near[['contract']].reset_index()
    m = allc.merge(near_c, on=['date', 'contract'], how='inner').set_index('date').sort_index()
    m = m.dropna(subset=['overnight'])
    res = {'rows': int(len(m)), 'start': str(m.index.min().date()), 'end': str(m.index.max().date())}
    def ann(r):
        r = r.dropna(); eq = (1 + r).cumprod(); yrs = len(r) / 245
        return {'CAGR': round(float(eq.iloc[-1] ** (1 / yrs) - 1), 4), 'vol': round(float(r.std() * math.sqrt(245)), 4), 'maxDD': round(float((eq / eq.cummax() - 1).min()), 4)}
    res['overnight_vs_intraday_TX'] = {
        'overnight_mean_bps': round(float(m['overnight'].mean() * 1e4), 2), 'overnight_t': round(t(m['overnight']), 2), 'overnight_ann': ann(m['overnight']),
        'intraday_mean_bps': round(float(m['intraday'].mean() * 1e4), 2), 'intraday_t': round(t(m['intraday']), 2), 'intraday_ann': ann(m['intraday']),
        'cc_mean_bps': round(float(m['cc'].mean() * 1e4), 2), 'cc_ann': ann(m['cc']),
        'by_period': {}}
    for lab, sl in [('2016-10~2019', slice('2016-10-01', '2019-12-31')), ('2020~2022', slice('2020-01-01', '2022-12-31')), ('2023~2026', slice('2023-01-01', '2026-12-31'))]:
        s = m.loc[sl]
        if len(s) > 50:
            res['overnight_vs_intraday_TX']['by_period'][lab] = {'n': int(len(s)), 'overnight_bps': round(float(s['overnight'].mean() * 1e4), 2), 'overnight_t': round(t(s['overnight']), 2),
                                                               'intraday_bps': round(float(s['intraday'].mean() * 1e4), 2), 'intraday_t': round(t(s['intraday']), 2)}
    # 隔夜策略成本：每天一趟來回 3 點 ÷ 指數水位
    cost_daily = a.rt_cost_pts / m['prev_close']
    on_net = m['overnight'] - cost_daily
    res['overnight_vs_intraday_TX']['overnight_net_of_cost_ann'] = ann(on_net)
    res['overnight_vs_intraday_TX']['overnight_net_mean_bps'] = round(float(on_net.mean() * 1e4), 2)
    # 趨勢過濾（近月收盤 vs 均線；換月日以同契約報酬計算，持倉跨月以 cc 近似）
    cc = m['cc']
    trend = {}
    for L in (20, 50, 100):
        ma = near['close'].rolling(L).mean().reindex(m.index)
        pos = (near['close'].reindex(m.index) > ma).astype(int).shift(1).fillna(0)
        sw = pos.diff().abs().fillna(0)
        strat = pos * cc - sw * (a.rt_cost_pts / 2) / m['prev_close']
        trend[f'MA{L}'] = {**ann(strat), 'switches': int(sw.sum()), 'time_in_market': round(float(pos.mean()), 3)}
    trend['buy_and_hold_TX_cc'] = ann(cc)
    res['trend_filter_TX'] = trend
    # 三大法人期貨未平倉訊號
    inst = pd.read_csv(d / 'institutional.csv', encoding='utf-8-sig')
    inst['date'] = pd.to_datetime(inst['日期'])
    piv = inst.pivot_table(index='date', columns='身份別', values='多空未平倉口數淨額', aggfunc='sum').sort_index()
    flow = inst.pivot_table(index='date', columns='身份別', values='多空交易口數淨額', aggfunc='sum').sort_index()
    sig = {}
    nxt = cc.shift(-1)  # 隔日報酬（訊號當日盤後公布 → 次日持有）
    for who in piv.columns:
        lvl = piv[who].reindex(m.index); chg = lvl.diff(); fl = flow[who].reindex(m.index)
        for nm, s in [('netOI_level', lvl), ('netOI_change', chg), ('daily_net_flow', fl), ('flow_20d_sum', fl.rolling(20).sum())]:
            z = s.dropna()
            q = pd.qcut(z, 3, labels=['low', 'mid', 'high'], duplicates='drop')
            grp = nxt.reindex(z.index).groupby(q)
            sig[f'{who}|{nm}'] = {k: {'n': int(v.count()), 'next_day_bps': round(float(v.mean() * 1e4), 2), 't': round(t(v), 2)} for k, v in grp}
            hi = nxt.reindex(z.index)[q == 'high']; lo = nxt.reindex(z.index)[q == 'low']
            sig[f'{who}|{nm}']['high_minus_low_bps'] = round(float((hi.mean() - lo.mean()) * 1e4), 2)
    res['institutional_signals_next_day'] = sig
    (out / 'tx_futures_tests.json').write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding='utf-8')
    print(json.dumps({k: res[k] for k in ['rows', 'start', 'end', 'overnight_vs_intraday_TX', 'trend_filter_TX']}, ensure_ascii=False, indent=1))
    for k, v in sig.items():
        print(k, '| H-L bps', v['high_minus_low_bps'], '| high', v.get('high'), '| low', v.get('low'))

if __name__ == '__main__':
    main()
