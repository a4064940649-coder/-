#!/usr/bin/env python3
"""S103 期貨層：台指期（TX）連續報酬序列、基差估計、加權指數（價格指數）。
資料（只讀、不上網）：
  - /home/user/chijiun/tx-flow-trend-backtest/input/all_contracts.csv（GitHub 公開存檔 ChiJiun/tx-flow-trend-backtest；
    tw_rule_edges/tx_futures_tests.py 用過；期交所每日行情，2016-10-05～2026-10-05，每契約日 OHLC）。
  - cslib.TWII_CSV（加權指數日線，價格指數，不含息）。
換月規則（事前寫死）：持有最近月；在最後交易日的前一個交易日收盤換到次月（避開結算日）。
日報酬＝同一契約 close_t ÷ close_{t−1} − 1（換月日用前一天持有的契約；新契約從換月日收盤起算）。
2016-10-05 以前與 2026-10-06 以後（無 TX 資料）：r_fut ＝ r_TWII ＋ g(m)，g(m)＝2016-10～2026-09 實測「TX 連續報酬 − 加權指數報酬」
  依曆月（1～12 月）平均的每日差（股息季節性：除息旺季 6～9 月逆價差收斂）。這是假設，結果檔明列各月 g。"""
from pathlib import Path
import numpy as np, pandas as pd

CJ = Path('/home/user/chijiun/tx-flow-trend-backtest/input')
TWII_CSV = Path('/home/user/jonatw/finlab-txd/data/raw/taiex_twii.csv')
REAL_START = pd.Timestamp('2016-10-05')


def twii():
    t = pd.read_csv(TWII_CSV, parse_dates=['date']).set_index('date')['close'].sort_index()
    return t[~t.index.duplicated()]


def tx_continuous():
    a = pd.read_csv(CJ / 'all_contracts.csv', encoding='utf-8-sig', parse_dates=['date'])
    px = a.pivot_table(index='date', columns='contract', values='close', aggfunc='last').sort_index()
    days = px.index
    last = a.groupby('contract')['date'].max()
    # 最後一個（尚未到期）契約的最後觀察日只是資料截止，不是最後交易日：以契約月第三個週三近似辨識
    cons = sorted(last.index, key=lambda c: c[2:])
    lpos = {c: int(days.get_loc(last[c])) for c in cons}
    end = len(days) - 1
    held = []
    for i in range(len(days)):
        h = None
        for c in cons:
            lp = lpos[c]
            if lp == end and c[2:] > days[-1].strftime('%Y%m'):
                lp = 10 ** 9                       # 資料截止時仍在交易的遠月
            elif lp == end:
                lp = 10 ** 9 if days[-1].day < 15 else lp
            if lp >= i + 2 and np.isfinite(px.at[days[i], c]):
                h = c; break
        held.append(h)
    held = pd.Series(held, index=days)
    r = pd.Series(np.nan, index=days); roll = pd.Series(False, index=days)
    for i in range(1, len(days)):
        c = held.iloc[i - 1]
        p0, p1 = px.at[days[i - 1], c], px.at[days[i], c]
        r.iloc[i] = p1 / p0 - 1
        roll.iloc[i] = held.iloc[i] != c
    lvl = pd.Series([px.at[d, c] for d, c in held.items()], index=days)
    return pd.DataFrame(dict(r=r, held=held, roll=roll, px=lvl))


def build(dates):
    """dates：策略交易日（DatetimeIndex）。回傳 DataFrame：r_twii、r_fut（實測或近似）、real（是否實測）、roll（換月日）、
    idx（指數水位，算契約值）、g_month（近似用的每日差）。"""
    tw = twii()
    tx = tx_continuous()
    rt = tw.pct_change()
    # 校準：2016-10-06～2026-09-30，r_fut − r_twii，依曆月平均（每日）
    j = pd.concat([tx['r'], rt], axis=1, keys=['f', 't']).dropna()
    j = j.loc['2016-10-06':'2026-09-30']
    gap = j['f'] - j['t']
    g_month = gap.groupby(gap.index.month).mean()
    df = pd.DataFrame(index=dates)
    df['twii'] = tw.reindex(dates).ffill()
    df['r_twii'] = df['twii'].pct_change().fillna(0.0)
    real = tx['r'].reindex(dates)
    approx = df['r_twii'] + pd.Series(dates.month, index=dates).map(g_month).to_numpy()
    df['real'] = real.notna() & (dates > REAL_START)
    df['r_fut'] = np.where(df['real'], real, approx)
    df.iloc[0, df.columns.get_loc('r_fut')] = 0.0
    rl = tx['roll'].reindex(dates).fillna(False).astype(bool)
    # 近似期間：每月第三個週三的前一交易日視為換月日
    m = pd.Series(dates, index=dates)
    third_wed = m.groupby([dates.year, dates.month]).apply(lambda s: s[(s.dt.weekday <= 2) & (s.dt.day >= 15)].min() if len(s) else pd.NaT)
    approx_roll = pd.Series(False, index=dates)
    for d in third_wed.dropna():
        k = dates.get_indexer([d])[0]
        if k > 0:
            approx_roll.iloc[k - 1] = True
    df['roll'] = np.where(df['real'], rl, approx_roll)
    stats = dict(calib_window=[str(j.index[0].date()), str(j.index[-1].date())], n_days=int(len(j)),
                 gap_ann_mean=float(gap.mean() * 245), gap_ann_by_year={int(y): float(v.mean() * 245) for y, v in gap.groupby(gap.index.year)},
                 g_month_daily_bps={int(k): round(float(v * 1e4), 3) for k, v in g_month.items()},
                 g_month_sum_ann_pct=float(sum(g_month[k] * (gap.index.month == k).sum() for k in g_month.index) / len(gap) * 245 * 100),
                 corr_fut_twii=float(j['f'].corr(j['t'])), tx_vs_twii_cum=dict(tx=float((1 + j['f']).prod() - 1), twii=float((1 + j['t']).prod() - 1)),
                 n_real_days=int(df['real'].sum()), real_first=str(df.index[df['real']][0].date()), real_last=str(df.index[df['real']][-1].date()),
                 n_rolls_real=int(df.loc[df['real'], 'roll'].sum()), n_rolls_approx=int(df.loc[~df['real'], 'roll'].sum()))
    return df, stats
