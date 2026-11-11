# -*- coding: utf-8 -*-
"""獨立重建 zheng2-signal 兩桶規則的回測引擎（不 import 作者程式）。
執行：python3 -I -B run.py
"""
import math
from statistics import NormalDist

import numpy as np
import pandas as pd

RAW = '/home/user/-/research/tw_stock_selection/agents/o_ext_zheng2_signal/raw/'

# 短利（% 年；隔夜拆款年均近似值，2003 起沿用 w4_lev2_half_rebalance；2000~2002 為近似，需查證）
RATE = {1999: 4.8, 2000: 4.7, 2001: 3.7, 2002: 2.0, 2003: 1.07, 2004: 1.03, 2005: 1.32, 2006: 1.55, 2007: 1.98, 2008: 1.85,
        2009: 0.11, 2010: 0.19, 2011: 0.34, 2012: 0.43, 2013: 0.39, 2014: 0.39, 2015: 0.35, 2016: 0.19, 2017: 0.18,
        2018: 0.18, 2019: 0.18, 2020: 0.10, 2021: 0.08, 2022: 0.26, 2023: 0.68, 2024: 0.82, 2025: 0.82, 2026: 0.82}
L_FEE = 0.012          # 00631L 內扣（經理費 1.0%＋保管 0.04%＋期貨交易成本等，取 1.2%／年；需查證）
BAD_DATES = ['2026-09-20']   # 週日壞列（0050 當日 −7.1%、隔日 +9.2%，加權僅 +0.4%）
L_START = '2014-10-31'


def load():
    tw = pd.read_csv(RAW + 'taiex_twii.csv', parse_dates=['date']).set_index('date')
    a = pd.read_csv(RAW + 'etf_0050.csv', parse_dates=['date']).set_index('date')['adj']
    l = pd.read_csv(RAW + 'etf_00631L.csv', parse_dates=['date']).set_index('date')['adj']
    bad = pd.to_datetime(BAD_DATES)
    tw, a, l = tw.drop(bad, errors='ignore'), a.drop(bad, errors='ignore'), l.drop(bad, errors='ignore')
    cal = tw.index
    df = pd.DataFrame(index=cal)
    df['tw_o'], df['tw_c'] = tw['open'], tw['close']
    df['p50'] = a.reindex(cal).ffill()
    df['pL'] = l.reindex(cal).ffill()
    df['rtw'] = df.tw_c.pct_change().fillna(0.0)
    df['gap'] = (df.tw_o / df.tw_c.shift() - 1).fillna(0.0)
    df['rf'] = [RATE[y] / 100 / 252 for y in cal.year]
    r50 = df.p50.pct_change()
    rL = df.pL.pct_change()
    df['r50_real'] = r50
    df['rL_real'] = rL
    # 0050 上市前（2003-07 以前）代理：加權指數價格報酬＋股息率（由 2003-07~2013 的 0050含息−加權 校準）
    m = (df.index >= '2003-07-02') & (df.index <= '2013-12-31')
    dy = float((np.log1p(r50[m]).sum() - np.log1p(df.rtw[m]).sum()) / m.sum())   # 每日對數差
    df['r50'] = r50.where(r50.notna(), np.expm1(np.log1p(df.rtw) + dy))
    # 合成 00631L：A = 2×0050含息 − (費用+短利)；B = 2×(加權價格報酬+股息率) − (費用+短利)
    df['rL_synA'] = 2 * df.r50 - (L_FEE / 252 + df.rf)
    df['rL_synB'] = 2 * np.expm1(np.log1p(df.rtw) + dy) - (L_FEE / 252 + df.rf)
    info = {'div_yield_proxy_ann_pct': round(dy * 252 * 100, 2)}
    return df, info


def add_open_split(df, col50, colL):
    """把 close-to-close 報酬拆成 隔夜（前收→開盤）與 盤中（開盤→收盤）。
    2020-01-02~2025-05-28 用 hschi 真實開盤價；其餘以加權指數開盤缺口 ×β 推估（β=1 / 2）。"""
    out = {}
    for nm, col, beta, f in (('50', col50, 1.0, 'hschi_0050_open_close.csv'), ('L', colL, 2.0, 'hschi_00631L_open_close.csv')):
        h = pd.read_csv(RAW + f, parse_dates=['date']).set_index('date')
        intr_real = (h['close'] / h['open'] - 1).reindex(df.index)
        ov_proxy = (beta * df.gap).clip(-0.15, 0.15)
        intr_proxy = (1 + df[col]) / (1 + ov_proxy) - 1
        intr = intr_real.where(intr_real.notna(), intr_proxy)
        ov = (1 + df[col]) / (1 + intr) - 1
        out['ov' + nm], out['in' + nm] = ov.values, intr.values
        out['real_open_' + nm] = intr_real.notna().values
        out['ov_proxy_' + nm] = ov_proxy.values
    return out


def replay(close, ladder=((0.20, 0.60), (0.30, 1.00)), high_win=252, ma_n=200, buf=0.03, bucket=0.5):
    """回傳每日收盤後目標 (wL, w50) 與狀態；邏輯逐行對照作者 replay()。"""
    n = len(close)
    s = pd.Series(close)
    hi = s.rolling(high_win, min_periods=1).max().values
    ma = s.rolling(ma_n).mean().values
    wL, w50, lv, onv = np.zeros(n), np.zeros(n), np.zeros(n, int), np.zeros(n, bool)
    lvl, on = 0, True
    for i in range(n):
        c = close[i]
        dd = c / hi[i] - 1
        if dd >= 0:
            lvl = 0
        for j, (th, _) in enumerate(ladder):
            if dd <= -th and j + 1 > lvl:
                lvl = j + 1
        if not np.isnan(ma[i]):
            if c > ma[i] * (1 + buf):
                on = True
            elif c < ma[i] * (1 - buf):
                on = False
        aL = ladder[lvl - 1][1] if lvl else 0.0
        wL[i] = round(bucket * aL + (1 - bucket) * (1.0 if on else 0.0), 4)
        w50[i] = round(bucket * (1 - aL), 4)
        lv[i], onv[i] = lvl, on
    return wL, w50, lv, onv


def simulate(rL, r50, rf, wL, w50, start, band=0.10, buy=0.0, sell=0.0, j=0, mode='close', osp=None, cash_rate=None):
    """訊號 T 日收盤 → T+1+j 日成交。mode='close'：收盤價成交；'open'：開盤價成交（需 osp）。
    回傳 dict：eq（自 start 起每日資產）、expo（日終曝險）、n_reb、turnover。"""
    n = len(rL)
    vL = v1 = 0.0
    cash = 1.0
    prev = None
    eq, expo = np.empty(n - start), np.empty(n - start)
    n_reb, turn, cost_sum = 0, 0.0, 0.0
    for i in range(start, n):
        cr = rf[i] if cash_rate is None else cash_rate / 252
        k = i - 1 - j
        pend = (wL[k], w50[k]) if k >= 0 else None
        if mode == 'open':
            vL *= 1 + osp['ovL'][i]
            v1 *= 1 + osp['ov50'][i]
        else:
            vL *= 1 + rL[i]
            v1 *= 1 + r50[i]
            cash *= 1 + cr
        if pend is not None:
            tot = vL + v1 + cash
            if prev != pend or abs(vL / tot - pend[0]) > band or abs(v1 / tot - pend[1]) > band:
                dL, d1 = pend[0] * tot - vL, pend[1] * tot - v1
                cost = sum((buy if x > 0 else sell) * abs(x) for x in (dL, d1))
                vL += dL
                v1 += d1
                cash -= dL + d1 + cost
                if abs(dL) + abs(d1) > 1e-12:
                    n_reb += 1
                    turn += (abs(dL) + abs(d1)) / tot
                    cost_sum += cost / tot
                prev = pend
        if mode == 'open':
            vL *= 1 + osp['inL'][i]
            v1 *= 1 + osp['in50'][i]
            cash *= 1 + cr
        tot = vL + v1 + cash
        eq[i - start] = tot
        expo[i - start] = (2 * vL + v1) / tot
    return {'eq': eq, 'expo': expo, 'n_reb': n_reb, 'turnover': turn, 'cost_sum': cost_sum}


def monthly(eq_s):
    m = eq_s.groupby([eq_s.index.year, eq_s.index.month]).last()
    first = eq_s.iloc[0]
    r = m.pct_change()
    r.iloc[0] = m.iloc[0] / first - 1
    return r


def tstat(x):
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    if len(x) < 3 or x.std(ddof=1) == 0:
        return float('nan')
    return float(x.mean() / (x.std(ddof=1) / math.sqrt(len(x))))


def stats(eq_s, bench_s=None, rf_s=None, lo=None, hi=None):
    """eq_s：日資產 Series（含起點）。切片 [lo, hi] 後計算。"""
    e = eq_s
    if lo is not None:
        # 以 lo 前一個交易日為基準
        idx = e.index.searchsorted(pd.Timestamp(lo))
        e = e.iloc[max(idx - 1, 0):]
    if hi is not None:
        e = e[:pd.Timestamp(hi)]
    if len(e) < 30:
        return None
    yrs = (e.index[-1] - e.index[0]).days / 365.25
    r = e.pct_change().dropna()
    cagr = (e.iloc[-1] / e.iloc[0]) ** (1 / yrs) - 1
    vol = r.std(ddof=1) * math.sqrt(252)
    rf = rf_s.reindex(r.index) if rf_s is not None else 0.0
    sharpe = (r - rf).mean() / r.std(ddof=1) * math.sqrt(252)
    mdd = float((e / e.cummax() - 1).min())
    out = {'start': str(e.index[0].date()), 'end': str(e.index[-1].date()), 'years': round(yrs, 2), 'cagr': round(cagr * 100, 2),
           'vol': round(vol * 100, 2), 'sharpe': round(float(sharpe), 3), 'mdd': round(mdd * 100, 2)}
    if bench_s is not None:
        b = bench_s.reindex(e.index)
        bc = (b.iloc[-1] / b.iloc[0]) ** (1 / yrs) - 1
        xm = monthly(e) - monthly(b)
        out.update({'bench_cagr': round(bc * 100, 2), 'excess_cagr': round((cagr - bc) * 100, 2),
                    't_month': round(tstat(xm.values), 2), 'n_month': int(xm.notna().sum()),
                    'mean_month_excess_pct': round(float(xm.mean()) * 100, 3),
                    'bench_mdd': round(float((b / b.cummax() - 1).min()) * 100, 2)})
    return out


def yearly(eq_s):
    y = eq_s.groupby(eq_s.index.year).last()
    r = y.pct_change()
    r.iloc[0] = y.iloc[0] / eq_s.iloc[0] - 1
    return {int(k): round(float(v) * 100, 1) for k, v in r.items()}


def deflated_sharpe(x, sr_trials):
    """x：月超額報酬；sr_trials：鄰域各試驗的月頻 SR（同一比較基準）。回傳 (SR, SR0, DSR, PSR0)。"""
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    T = len(x)
    sr = x.mean() / x.std(ddof=1)
    z = (x - x.mean()) / x.std(ddof=0)
    sk, ku = float((z ** 3).mean()), float((z ** 4).mean())
    N = len(sr_trials)
    V = float(np.var(sr_trials, ddof=1))
    nd = NormalDist()
    g = 0.5772156649
    sr0 = math.sqrt(V) * ((1 - g) * nd.inv_cdf(1 - 1 / N) + g * nd.inv_cdf(1 - 1 / (N * math.e)))
    den = math.sqrt(max(1e-12, 1 - sk * sr + (ku - 1) / 4 * sr ** 2))
    dsr = nd.cdf((sr - sr0) * math.sqrt(T - 1) / den)
    psr0 = nd.cdf(sr * math.sqrt(T - 1) / den)
    return {'sr_month': round(float(sr), 4), 'sr0_month': round(sr0, 4), 'dsr': round(dsr, 4), 'psr_vs0': round(psr0, 4),
            'n_trials': N, 'T': T, 'skew': round(sk, 2), 'kurt': round(ku, 2)}
