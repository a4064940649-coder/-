#!/usr/bin/env python3
"""S105 共用：在 S75 v3 連續帳戶（引擎逐日 NAV＋stockfrac，取自 s103_futures_overlay/sleeves.npz）上，把部分 0050 底倉改成元大台灣50正2（00631L）。
帳戶四塊：股票 S（衛星，報酬 r_s）、0050 E、00631L 部位 L、現金 C（年利率 rf，主設定 0）。
r_s 用 s103 overlay.stock_returns 由引擎拆出（r_nav = w·r_s + (1−w)·r_0050；w＝前一日 stockfrac）。
每日（t−1 收盤 → t 收盤）：
  1. S×(1+r_s)、E×(1+r_other)、L×(1+r_L)、C×(1+rf/245)；A＝S+E+L+C。
  2. S＝sf_t·A（照引擎持股比例；股票↔0050 的移轉成本只算在引擎裡）。N＝A−S。
  3. 底倉再平衡：mode 'monthly' 在該位移的換股日（引擎交易日 k+1）收盤；mode 'band' 在 |L/A − Lw| > 5pp 的收盤。
     再平衡時 L→Lw·A、C→Cw·A，差額由 0050 進出；L 與 0050 各自計單邊成本。
  4. E＝N−L−C；E<0（股票漲多吃掉 0050）時先用 C，再賣 L（計「被迫賣 L」日數與成本）。
成本（每邊，占成交值）：買＝FEE＋slip；賣＝FEE＋證交稅 0.1%（ETF）＋slip；FEE＝0.1425%×0.28（同引擎 stlib.FEE），slip 主設定 0.1%。"""
import sys, math, json
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
AG = HERE.parent
S103 = AG / 's103_futures_overlay'
sys.path.insert(0, str(S103))
import overlay as O                                    # noqa: E402  （只用 stock_returns，不改檔）

L2_CSV = Path('/home/user/jonatw/finlab-txd/data/raw/etf_00631L.csv')   # 還原日序列（w4_lev2_half_rebalance 同檔），2014-10-31 起
FEE, TAX_ETF = 0.001425 * 0.28, 0.001
SYN_FEE = 0.01                                          # 合成期：r_L＝2×r_0050 − 1%/245
RATE_ON = {2008: 1.85, 2009: 0.11, 2010: 0.19, 2011: 0.34, 2012: 0.43, 2013: 0.39, 2014: 0.39, 2015: 0.35, 2016: 0.19, 2017: 0.18, 2018: 0.18,
           2019: 0.18, 2020: 0.10, 2021: 0.08, 2022: 0.26, 2023: 0.68, 2024: 0.82, 2025: 0.82, 2026: 0.82}   # 隔拆年均（% 年，同 w4 run.py）


def load(synth='main', gap_ann=None):
    z = np.load(S103 / 'sleeves.npz')
    d = pd.DatetimeIndex(z['dates'])
    e = z['e50']; r50 = np.r_[0.0, e[1:] / e[:-1] - 1]
    la = pd.read_csv(L2_CSV, parse_dates=['date']).set_index('date')['adj'].sort_index()
    # 對齊引擎交易日：非交易日列（2026-07-10、2026-09-20）丟掉；缺 2026-03-25～30 → ffill（報酬併入 03-31）
    lv = la.reindex(d).ffill()
    ra = lv.pct_change().to_numpy()
    actual = np.isfinite(ra)
    on = np.array([RATE_ON[y] for y in d.year]) / 100
    if synth == 'main':
        syn = 2 * r50 - SYN_FEE / 245
    elif synth == 'rate':                               # 再扣隔拆利率（期貨隱含融資成本）
        syn = 2 * r50 - SYN_FEE / 245 - on / 245
    elif synth == 'cal':                                # 用 2015～2021 實測「實際 − 每日2倍」年化落差校正（gap_ann 含費用）
        syn = 2 * r50 + gap_ann / 245
    else:
        raise ValueError(synth)
    rL = np.where(actual, ra, syn); rL[0] = 0.0
    return z, d, e, r50, rL, actual, on


def run(nav, sf, r50, rL, Lw, Cw, mode, rebal_mask, band=0.05, t_start=0, rf_d=None, slip=0.001):
    T = len(nav)
    sf = np.nan_to_num(np.asarray(sf, float), nan=0.0)
    rs, rn, ro = O.stock_returns(nav, sf, r50)
    rf_d = np.zeros(T) if rf_d is None else rf_d
    cb, csl = FEE + slip, FEE + TAX_ETF + slip
    side = lambda x: x * cb if x > 0 else -x * csl
    A = 1e6
    S = sf[t_start] * A; L = Lw * A; C = Cw * A
    st = dict(cost_L=side(L), cost_50=side(-L), traded_L=abs(L), traded_50=abs(L), n_rebal=0, forced_L_days=0, forced_L_val=0.0, cash_used_days=0)
    E = A - S - L - C - st['cost_L'] - st['cost_50']
    if E < 0:
        x = min(-E, C); C -= x; E += x
        if E < 0:
            L += E; E = 0.0
    out = np.full(T, np.nan); out[t_start] = S + E + L + C
    wL = np.full(T, np.nan); wE = np.full(T, np.nan); wC = np.full(T, np.nan); Asum = 0.0
    for t in range(t_start + 1, T):
        S *= 1 + rs[t]; E *= 1 + ro[t]; L *= 1 + rL[t]; C *= 1 + rf_d[t]
        A = S + E + L + C
        S = sf[t] * A; N = A - S
        reb = (Lw > 0 or Cw > 0) and (rebal_mask[t] if mode == 'monthly' else abs(L / A - Lw) > band)
        if reb:
            dL = Lw * A - L; dC = Cw * A - C
            c1, c2 = side(dL), side(-(dL + dC))
            st['cost_L'] += c1; st['cost_50'] += c2; st['traded_L'] += abs(dL); st['traded_50'] += abs(dL + dC); st['n_rebal'] += 1
            L += dL; C += dC; N -= c1 + c2
        E = N - L - C
        if E < 0:
            x = min(-E, C)
            if x > 0:
                C -= x; E += x; st['cash_used_days'] += 1
            if E < -1e-9:
                y = -E / (1 - csl)                      # 賣 L 補足（含賣出成本）
                y = min(y, L); L -= y; E += y * (1 - csl)
                st['cost_L'] += y * csl; st['traded_L'] += y; st['forced_L_days'] += 1; st['forced_L_val'] += y
        out[t] = S + E + L + C
        wL[t] = L / out[t]; wE[t] = E / out[t]; wC[t] = C / out[t]; Asum += out[t]
    n = T - t_start - 1
    avgA = Asum / max(n, 1); yrs = n / 245
    st.update(avg_wL=float(np.nanmean(wL)), avg_wE=float(np.nanmean(wE)), avg_wC=float(np.nanmean(wC)), avg_sf=float(np.mean(sf[t_start + 1:])),
              turnover_L_ann=st['traded_L'] / avgA / yrs, turnover_50_ann=st['traded_50'] / avgA / yrs,
              cost_ann_pct=(st['cost_L'] + st['cost_50']) / avgA / yrs * 100)
    st['avg_index_exposure_ex_stocks'] = st['avg_wE'] + 2 * st['avg_wL']          # 0050＋2×00631L（不含衛星股票）
    return out, st, wL


# ------------------------------------------------------------------ 統計（同 s103 run_s103.metrics 口徑：基底＝窗前最後交易日，Sharpe rf=0 ×√252）
def wslice(d, a, b):
    a, b = pd.Timestamp(a), pd.Timestamp(b)
    i0 = int(np.searchsorted(d.values, a.to_datetime64(), side='left')); i0 = max(i0 - 1, 0)
    i1 = int(np.searchsorted(d.values, b.to_datetime64(), side='right')) - 1
    return i0, i1


def metrics(eq, d, a, b, e50=None):
    i0, i1 = wslice(d, a, b)
    s = pd.Series(eq[i0:i1 + 1], index=d[i0:i1 + 1])
    yrs = (s.index[-1] - s.index[0]).days / 365.25
    r = s.pct_change().dropna(); dd = s / s.cummax() - 1
    yr = s.groupby(s.index.year).last(); prev = s.iloc[0]; yl = {}
    for y, v in yr.items():
        yl[int(y)] = float(v / prev - 1); prev = v
    first = d[i0 + 1].year if i0 > 0 else d[0].year
    yl = {y: v for y, v in yl.items() if y >= first}
    cg = float((s.iloc[-1] / s.iloc[0]) ** (1 / yrs) - 1); mdd = float(dd.min())
    wy = min(yl.items(), key=lambda x: x[1])
    o = dict(cagr=cg, maxdd=mdd, dd_peak=str(s.loc[:dd.idxmin()].idxmax().date()), dd_trough=str(dd.idxmin().date()),
             vol=float(r.std(ddof=1) * math.sqrt(252)), sharpe=float(r.mean() / r.std(ddof=1) * math.sqrt(252)),
             calmar=cg / abs(mdd) if mdd < 0 else None, worst_year=dict(year=wy[0], ret=wy[1]), yearly=yl)
    if e50 is not None:
        q = pd.Series(e50[i0:i1 + 1], index=s.index); o['cagr_0050'] = float((q.iloc[-1] / q.iloc[0]) ** (1 / yrs) - 1)
        o['excess_vs_0050_pp'] = (cg - o['cagr_0050']) * 100
    return o


def vs(m, b):
    return dict(cagr_pp=(m['cagr'] - b['cagr']) * 100, maxdd_pp=(m['maxdd'] - b['maxdd']) * 100, sharpe=m['sharpe'] - b['sharpe'])


def judge(m, b):
    c = m['cagr'] > b['cagr']; dd = m['maxdd'] >= b['maxdd'] - 0.05; sh = m['sharpe'] >= b['sharpe']
    return dict(cagr_gt=bool(c), mdd_ok=bool(dd), sharpe_ge=bool(sh), pass_all=bool(c and dd and sh))


BEARS = {'2008': ('2008-01-01', '2008-12-31'), '2011': ('2011-01-01', '2011-12-31'), '2015': ('2015-01-01', '2015-12-31'),
         '2018': ('2018-01-01', '2018-12-31'), '2020H1': ('2020-01-01', '2020-06-30'), '2022': ('2022-01-01', '2022-12-31')}


def bears(eq, d):
    o = {}
    for k, (a, b) in BEARS.items():
        i0, i1 = wslice(d, a, b); s = pd.Series(eq[i0:i1 + 1], index=d[i0:i1 + 1])
        o[k] = dict(ret=float(s.iloc[-1] / s.iloc[0] - 1), maxdd=float((s / s.cummax() - 1).min()))
    return o


def lev_drag(d, e, rL, actual, a, b):
    """00631L 對 2×0050 的落差（只用實際資料日）。cagr_L − 2×cagr_0050＝總落差；cagr_L − cagr(每日2倍)＝費用/融資/追蹤；cagr(每日2倍) − 2×cagr_0050＝純波動耗損。"""
    i0, i1 = wslice(d, a, b)
    sl = slice(i0 + 1, i1 + 1)
    assert actual[sl].all(), '窗內有非實際資料日'
    yrs = (d[i1] - d[i0]).days / 365.25
    r50 = e[i0:i1 + 1]; r50 = r50[1:] / r50[:-1] - 1
    g = lambda r: float(np.prod(1 + r) ** (1 / yrs) - 1)
    cL, c50, c2x = g(rL[sl]), g(r50), g(2 * r50)
    return dict(window=[str(d[i0].date()), str(d[i1].date())], cagr_00631L=cL, cagr_0050=c50, two_x_cagr_0050=2 * c50, cagr_daily2x_0050=c2x,
                total_gap_vs_2x_pp=(cL - 2 * c50) * 100, vol_drag_pp=(c2x - 2 * c50) * 100, fee_fin_tracking_gap_pp=(cL - c2x) * 100,
                te_daily_ann=float(np.std(rL[sl] - 2 * r50, ddof=1) * math.sqrt(252)))
