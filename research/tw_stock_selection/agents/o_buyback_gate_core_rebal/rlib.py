# -*- coding: utf-8 -*-
"""兩腿再平衡引擎與統計（自己實作；淨值來自 build_main.py 的 main_navs.parquet）。"""
import json, math
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent
PLAN = json.loads((HERE / 'plan.json').read_text(encoding='utf-8'))['PLAN']
NAVS = pd.read_parquet(HERE / 'main_navs.parquet')
MP = json.loads((HERE / 'main_plan.json').read_text(encoding='utf-8'))
IX = NAVS.index
SD_ALL = pd.DatetimeIndex(MP['sig_days'])          # 含評價終點
SIG = SD_ALL[:-1]                                   # 可檢查／換股的訊號日
EXEC = pd.DatetimeIndex([p['exec'] for p in MP['plan']])
SPLIT = [d for d in SIG if d.year == 2019][0]
END = IX[-1]
PER = {'SEL': (IX[0], SPLIT), 'VAL': (SPLIT, pd.Timestamp('2023-12-31')), 'HOLD': (pd.Timestamp('2024-01-01'), END),
       'POST': (SPLIT, END), 'ALL': (IX[0], END), 'S1': (IX[0], pd.Timestamp('2013-12-31')), 'S2': (pd.Timestamp('2014-01-01'), SPLIT)}
C = PLAN['cost']
RULES = list(PLAN['rules'].keys())
MONTHS = {'Q': {1, 4, 7, 10}, 'H': {1, 7}, 'A': {1}}

def leg_returns(cost='normal', B='E50'):
    A = NAVS['MAIN|normal' if cost == 'normal' else 'MAIN|stress3']
    return A.pct_change().fillna(0).to_numpy(), NAVS[B].pct_change().fillna(0).to_numpy()

def costs(cost='normal', B='E50'):
    c = C[cost]
    if B == 'E50': return c['stock_buy'], c['stock_sell'], c['etf_buy'], c['etf_sell']
    return c['stock_buy'], c['stock_sell'], c['stock_buy'], c['stock_sell']   # 2330 腿用個股成本

def sim(w, rule, start, end, cost='normal', B='E50', months=None, exec_mode='sig', band_to='target', mom_n=12, A_override=None):
    """w＝B（0050）目標權重。start：以 start 之前（不含）最後一個交易日收盤為期初（若 start＝IX[0] 則期初＝IX[0]）。
    回傳 dict(nav（日頻，期初＝1）, n_trades, turnover_sum, wB 序列（訊號日）)。"""
    ra, rb = leg_returns(cost, B)
    if A_override is not None: ra = A_override.reindex(IX).pct_change().fillna(0).to_numpy()
    ab, as_, bb, bs = costs(cost, B)
    i0 = 0 if start <= IX[0] else int(IX.searchsorted(start)) - 1
    i1 = int(IX.searchsorted(end, 'right')) - 1
    days = SIG if exec_mode == 'sig' else EXEC
    chk = {int(IX.get_loc(d)): d for d in days if IX[i0] < d <= IX[i1] and d in IX}
    if months is None: months = MONTHS.get(rule)
    A_nav = (NAVS['MAIN|normal' if cost == 'normal' else 'MAIN|stress3'] if A_override is None else A_override.reindex(IX)).to_numpy(); B_nav = NAVS[B].to_numpy()
    sig_pos = [int(IX.get_loc(d)) for d in days if d in IX]
    a, b = 1 - w, w
    out = np.empty(i1 - i0 + 1); out[0] = 1.0
    n_tr = 0; tov = 0.0; wbs = {}
    for i in range(i0 + 1, i1 + 1):
        a *= 1 + ra[i]; b *= 1 + rb[i]
        if i in chk and w > 0:
            d = chk[i]; tot = a + b; wb = b / tot
            if rule == 'M': do = True
            elif rule in ('Q', 'H', 'A'): do = d.month in months
            elif rule == 'NONE': do = False
            elif rule.startswith('B'): do = abs(wb - w) > float(rule[1:]) / 100 + 1e-12
            elif rule.startswith('MOM'):
                k = sig_pos.index(i) if i in sig_pos else None
                if k is None or k < mom_n: do = True
                else:
                    j = sig_pos[k - mom_n]
                    relA = A_nav[i] / A_nav[j] - B_nav[i] / B_nav[j]
                    over_A = wb < w
                    do = not ((over_A and relA > 0) or ((not over_A) and relA < 0))
            else: raise ValueError(rule)
            if do:
                tgt = w
                if rule.startswith('B') and band_to == 'edge':
                    bd = float(rule[1:]) / 100; tgt = w + bd if wb > w else w - bd
                tb = tgt * tot
                if b > tb: m = b - tb; c = m * (bs + ab)     # 賣 0050 買主策略
                else: m = tb - b; c = m * (as_ + bb)
                if m > 1e-12: n_tr += 1; tov += m / tot
                tot -= c; a, b = (1 - tgt) * tot, tgt * tot
            wbs[d] = wb
        out[i - i0] = a + b
    nav = pd.Series(out, index=IX[i0:i1 + 1])
    return dict(nav=nav, n_trades=n_tr, turnover=tov, wb=pd.Series(wbs, dtype=float))

def bench(name, start, end):
    s = NAVS[name]
    i0 = 0 if start <= IX[0] else int(IX.searchsorted(start)) - 1
    i1 = int(IX.searchsorted(end, 'right')) - 1
    x = s.iloc[i0:i1 + 1]
    return x / x.iloc[0]

def cagr(nav):
    yrs = (nav.index[-1] - nav.index[0]).days / 365.25
    return (nav.iloc[-1] / nav.iloc[0]) ** (1 / yrs) - 1

def mdd(nav): return float((nav / nav.cummax() - 1).min())

def win(nav):
    D = pd.DatetimeIndex([d for d in SD_ALL if nav.index[0] <= d <= nav.index[-1]])
    if D[0] != nav.index[0]: D = D.insert(0, nav.index[0])
    if D[-1] != nav.index[-1]: D = D.append(pd.DatetimeIndex([nav.index[-1]]))
    x = nav.reindex(D)
    return (x.shift(-1) / x - 1).iloc[:-1]

def tstat(x):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    if len(x) < 3: return float('nan')
    s = x.std(ddof=1)
    return float(x.mean() / (s / math.sqrt(len(x)))) if s > 0 else float('nan')

def nw_t(x, L=6):
    x = np.asarray(x, float); x = x[np.isfinite(x)]; n = len(x)
    if n < 10: return float('nan')
    e = x - x.mean(); g0 = (e @ e) / n; s = g0
    for l in range(1, L + 1):
        s += 2 * (1 - l / (L + 1)) * (e[l:] @ e[:-l]) / n
    return float(x.mean() / math.sqrt(s / n)) if s > 0 else float('nan')

def lag_stats(nav, b):
    r = win(nav); rb = win(b.reindex(nav.index))
    rel = (1 + r).cumprod() / (1 + rb).cumprod()
    rel = pd.concat([pd.Series([1.0], index=[nav.index[0]]), rel])
    under = rel < rel.cummax() - 1e-12
    longest = cur = 0
    for u in under:
        cur = cur + 1 if u else 0; longest = max(longest, cur)
    cr = (1 + r).cumprod(); cb = (1 + rb).cumprod()
    out = dict(longest_lag_m=int(longest))
    for k in (12, 36):
        if len(r) > k:
            ra = (cr / cr.shift(k)).dropna(); rbb = (cb / cb.shift(k)).dropna()
            out[f'roll{k}_lag_frac'] = float((ra < rbb).mean())
        else: out[f'roll{k}_lag_frac'] = None
    return out

def stats(nav, b='E50', ew=None):
    bn = NAVS[b].reindex(nav.index); bn = bn / bn.iloc[0]
    r = win(nav); rb = win(bn); d = r - rb
    o = dict(CAGR=cagr(nav) * 100, MDD=mdd(nav) * 100, B_CAGR=cagr(bn) * 100, ex_pp=(cagr(nav) - cagr(bn)) * 100,
             t_vs0050=tstat(d), nwt_vs0050=nw_t(d), TE=float(d.std(ddof=1) * math.sqrt(12) * 100), n_win=int(len(d)),
             mean_ex_m_pp=float(d.mean() * 100), sd_ex_m_pp=float(d.std(ddof=1) * 100))
    o.update(lag_stats(nav, bn))
    if ew is not None:
        e = NAVS['POOL_EW|gross'].reindex(nav.index); de = r - win(e)
        o['t_vsEW'] = tstat(de); o['EW_CAGR'] = cagr(e / e.iloc[0]) * 100
    return o

def paired(nav1, nav0):
    d = win(nav1) - win(nav0)
    return dict(ann_diff_pp=(cagr(nav1) - cagr(nav0)) * 100, mean_pp=float(d.mean() * 100), t=tstat(d), nwt=nw_t(d), n=int(len(d)),
                n_nonzero=int((d.abs() > 1e-12).sum()))

def yearly_diff(nav1, nav0):
    y1 = nav1.groupby(nav1.index.year).last(); y0 = nav0.groupby(nav0.index.year).last()
    p1 = pd.concat([pd.Series([nav1.iloc[0]], index=[0]), y1]); p0 = pd.concat([pd.Series([nav0.iloc[0]], index=[0]), y0])
    r1 = (p1 / p1.shift(1) - 1).iloc[1:]; r0 = (p0 / p0.shift(1) - 1).iloc[1:]
    return ((r1 - r0) * 100)

def ann_diff_drop_year(nav1, nav0, year):
    """拿掉某一日曆年的兩者日報酬後，重新連乘算年化差。"""
    r1 = nav1.pct_change().fillna(0); r0 = nav0.pct_change().fillna(0)
    m = nav1.index.year != year
    yrs = m.sum() / 252.0
    g1 = (1 + r1[m]).prod() ** (1 / yrs) - 1; g0 = (1 + r0[m]).prod() ** (1 / yrs) - 1
    return (g1 - g0) * 100
