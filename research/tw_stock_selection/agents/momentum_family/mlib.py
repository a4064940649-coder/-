"""momentum_family 共用：長歷史換股日曆、價格訊號 P1～P6、M2 選股（漲跌幅依年代）、日頻統計。

事前固定（寫程式前就定好，見 FINDINGS「事前規則」）：
* 換股日曆：每月第一個 ≥ 11 日的交易日 d（與主策略相同）。訊號用 d 收盤（= 實際成交日 d+1 的前一日收盤）計算，
  d+1 開盤買（M2）；d+1 開盤漲停（含息跳空 ≥ 漲停 −0.6%）且未持有者視為買不到、由下一名遞補。
  漲停幅度：2015-06-01 前 7%、之後 10%。
* 股池：d 的 60 日均成交值前 300 名（且 ≥ NT$1,000 萬、收盤 ≥ 10 元）、d 非處置期間（名單 2021-09-24 起才有）、
  d+1 有開盤價、訊號 P1/P2/P3/P6 都有值（共同股池，保證 P1～P6 比的是同一群股票；P4 另需 ≥ 24 個 21 日區塊歷史）。
* 引擎與成本：直接用 exec_rebuild/exlib.py 的 sim_portfolio、cost_rates（滑價 0.1%、壓力來回 1%）。
"""
import math, sys
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'exec_rebuild'))
import exlib as L  # noqa: E402  （只讀取、不修改）

adj, C, V, IDX, COLS, N = L.adj, L.C, L.V, L.IDX, L.COLS, L.N
RET = L.RET
pos = L.pos

# ---------------- 長歷史日曆 ----------------
_s = pd.Series(IDX, index=IDX)
LD = [g[g.dt.day >= 11].iloc[0] for _, g in _s.groupby([IDX.year, IDX.month]) if (g.dt.day >= 11).any()]
LD = [d for d in LD if d >= pd.Timestamp('2005-01-01')]
LDATES = pd.DatetimeIndex(LD)
assert set(L.DATES) <= set(LDATES), '主策略日曆必須是長日曆的子集'

# ---------------- 漲停（依年代） ----------------
thr = np.where(IDX < pd.Timestamp('2015-06-01'), 0.07, 0.10)[:, None] - L.LTOL
GAP_UP = (L.AO_raw / L.AC_prev - 1) >= thr      # d+1 開盤漲停

# ---------------- 訊號 ----------------
A = adj.ffill(limit=10)
P1 = adj / adj.rolling(252, min_periods=240).max()
P2 = A.shift(21) / A.shift(252) - 1
P3 = A.shift(21) / A.shift(126) - 1
VOL252 = RET.rolling(252, min_periods=200).std()
P6 = P2 / VOL252
tx = pd.read_csv('/home/user/jonatw/finlab-txd/data/raw/taiex_twii.csv', parse_dates=['date']).set_index('date')['close'].sort_index()
TX = tx.reindex(IDX, method='ffill')

def resid_mom(dates, market='taiex', nblk=36, minblk=24):
    """21 日區塊報酬（k=1 最近），對市場回歸（含截距，至少 24 塊），分數 = Σ e_k(k=2..12) / sd(e_k, k=2..12)。"""
    An = A.to_numpy(); out = np.full((len(dates), N), np.nan)
    for r, d in enumerate(dates):
        i = pos(d); ends = [i - 21 * k for k in range(nblk + 1)]
        if ends[-1] < 0:
            ends = [e for e in ends if e >= 0]
        if len(ends) - 1 < minblk: continue
        Pm = An[ends]                                   # (K+1, N)
        R = Pm[:-1] / Pm[1:] - 1                         # (K, N)，row0 = 最近區塊
        if market == 'taiex':
            m = TX.to_numpy()[ends]; M = (m[:-1] / m[1:] - 1)[:, None] * np.ones((1, N))
        else:
            M = np.nanmean(np.where(np.isfinite(R), R, np.nan), axis=1)[:, None] * np.ones((1, N))
        ok = np.isfinite(R) & np.isfinite(M)
        cnt = ok.sum(0)
        Rz = np.where(ok, R, 0); Mz = np.where(ok, M, 0)
        with np.errstate(invalid='ignore', divide='ignore'):
            mx = Mz.sum(0) / cnt; my = Rz.sum(0) / cnt
            sxx = (np.where(ok, (M - mx) ** 2, 0)).sum(0); sxy = (np.where(ok, (M - mx) * (R - my), 0)).sum(0)
            beta = sxy / sxx; alpha = my - beta * mx
            E = np.where(ok, R - alpha - beta * M, np.nan)
            f = E[1:12]                                   # k = 2..12
            nf = np.isfinite(f).sum(0)
            sc = np.nansum(f, 0) / np.nanstd(f, 0, ddof=1)
        sc[(cnt < minblk) | (nf < 9) | ~np.isfinite(sc)] = np.nan
        # 需要最近區塊（k=1）的價格存在：避免已停牌股
        sc[~np.isfinite(An[i])] = np.nan
        out[r] = sc
    return pd.DataFrame(out, index=pd.DatetimeIndex(dates), columns=COLS)

# 產業別（stock_list.csv 現行分類；不在名單者依代碼前兩碼回推傳產，電子股無法回推 → 「未分類」群組）
PREFIX_IND = {'11': '水泥工業', '12': '食品工業', '13': '塑膠工業', '14': '紡織纖維', '15': '電機機械', '16': '電器電纜',
              '17': '化學工業', '18': '玻璃陶瓷', '19': '造紙工業', '20': '鋼鐵工業', '21': '橡膠工業', '22': '汽車工業',
              '25': '建材營造業', '26': '航運業', '27': '觀光餐旅', '28': '金融保險業', '29': '貿易百貨業', '99': '其他業'}
_sl = pd.read_csv('/home/user/stevenhsiehsheng-ctrl/tw-stock-screener/data/stock_list.csv', dtype={'code': str}).drop_duplicates('code').set_index('code')['industry']
IND = {}
IND_SRC = {'list': 0, 'prefix': 0, 'unclassified': 0}
for c in COLS:
    if c in _sl.index and isinstance(_sl[c], str): IND[c] = _sl[c]; IND_SRC['list'] += 1
    elif c[:2] in PREFIX_IND: IND[c] = PREFIX_IND[c[:2]]; IND_SRC['prefix'] += 1
    else: IND[c] = '未分類'; IND_SRC['unclassified'] += 1
IND = pd.Series(IND)

def at(df, dates):
    return pd.DataFrame(df.reindex(dates).to_numpy(), index=pd.DatetimeIndex(dates), columns=COLS)

# 2018-01-13（週六）面板只有 46 檔有資料，會讓嚴格 60 日均值讓其他股票失格 60 天 → 計算 60 日均值時剔除該日
_na = adj.notna().sum(axis=1); SPARSE = _na.index[(_na / _na.rolling(21, center=True, min_periods=5).median()) < 0.5]
V60 = V.drop(SPARSE).rolling(60).mean().reindex(IDX).ffill(limit=1)

def universe(dates):
    """d 收盤可知：前 300 流動性、≥1,000 萬、≥10 元、非處置；d+1 有開盤價。"""
    days = pd.DatetimeIndex(dates)
    v60 = V60.reindex(days); px = C.reindex(days); base = (v60 >= 1e7) & (px >= 10)
    t300 = base & (v60.where(base).rank(axis=1, ascending=False) <= 300)
    inD = L.in_disposal(days)
    nxt = [IDX[min(pos(d) + 1, len(IDX) - 1)] for d in days]
    trad = pd.DataFrame(np.isfinite(L.OPEN_RAW[[pos(d) for d in nxt]]) & np.isfinite(L.ADJ_RAW[[pos(d) for d in nxt]]), index=days, columns=COLS)
    return t300 & ~inD & trad, t300, inD

def ind_neutral(sig, el):
    """產業內百分位排名（產業 < 5 檔合格者併入「小產業合併」），同分以全市場排名 1e-6 權重打破。"""
    out = pd.DataFrame(np.nan, index=sig.index, columns=COLS)
    for d in sig.index:
        s = sig.loc[d][el.loc[d] & sig.loc[d].notna()]
        if len(s) == 0: continue
        g = IND.reindex(s.index)
        sz = g.map(g.value_counts()); g = g.where(sz >= 5, '小產業合併')
        r = s.groupby(g).rank(pct=True) + 1e-6 * s.rank(pct=True)
        out.loc[d, r.index] = r
    return out

# ---------------- 選股（M2） ----------------
def select(score, el, dates, n):
    events, pools = {}, {}; held = set(); nblk = 0
    for d in dates[:-1]:
        i = pos(d); ex = i + 1
        okb = (el.loc[d] & score.loc[d].notna()).to_numpy()
        trad = np.isfinite(L.OPEN_RAW[ex]) & np.isfinite(L.ADJ_RAW[ex])
        ok = okb & trad
        if ok.sum() < 50: continue
        order = score.loc[d].to_numpy().astype(float).copy(); order[~ok] = np.nan
        cand = np.argsort(-np.nan_to_num(order, nan=-9e9))[:int(ok.sum())]
        pick = []
        for j in cand:
            if GAP_UP[ex, j] and j not in held:
                nblk += 1; continue
            pick.append(j)
            if len(pick) == n: break
        events[ex] = ('open', np.array(pick)); pools[ex] = ('open', np.where(ok)[0])
        held = set(pick)
    select.nblk = nblk
    return events, pools

def sim(events, cost, dates):
    t0 = pos(dates[0]); t1 = pos(dates[-1])
    if cost == 'gross': b = s = 0.0
    else: b, s = L.cost_rates(cost)
    return L.sim_portfolio(events, b, s, t0, t1)

# ---------------- 統計 ----------------
def month_rets(nav, dates):
    nd = nav.reindex(pd.DatetimeIndex(dates)); return (nd.shift(-1) / nd - 1).iloc[:-1]

def stats(nav, a, b, dates, bench=None, ew=None):
    r = nav.pct_change(); m = (r.index >= pd.Timestamp(a)) & (r.index <= pd.Timestamp(b)) & r.notna().to_numpy()
    if m.sum() < 20: return None
    rr = r[m]; first = np.argmax(m)
    span = (rr.index[-1] - r.index[max(first - 1, 0)]).days / 365.25
    eq = (1 + rr).cumprod()
    o = {'CAGR': round(float(eq.iloc[-1] ** (1 / span) - 1), 4), 'vol': round(float(rr.std() * math.sqrt(252)), 4),
         'maxDD': round(float((eq / eq.cummax() - 1).min()), 4), 'years': round(span, 2)}
    pr = month_rets(nav, dates); pr = pr[(pr.index >= pd.Timestamp(a)) & (pr.index <= pd.Timestamp(b))].dropna()
    o['n_months'] = int(len(pr))
    for nm, bn in (('0050', bench), ('EW', ew)):
        if bn is None: continue
        bp = month_rets(bn, dates).reindex(pr.index); ex = pr - bp
        o[f'excess_vs_{nm}_pct_m'] = round(float(ex.mean() * 100), 2); o[f't_vs_{nm}_m'] = L.tstat(ex)
        bd = bn.pct_change().reindex(rr.index); eb = (1 + bd.fillna(0)).cumprod()
        o[f'{nm}_CAGR'] = round(float(eb.iloc[-1] ** (1 / span) - 1), 4)
        o[f'{nm}_maxDD'] = round(float((eb / eb.cummax() - 1).min()), 4)
    return o

def yearly(nav):
    r = nav.pct_change().dropna()
    return {int(y): round(float((1 + g).prod() - 1) * 100, 1) for y, g in r.groupby(r.index.year)}

def ic_series(score, el, dates):
    """d 的訊號 vs d+1 開盤 → 下一個 d+1 開盤 的報酬，Spearman。"""
    ex = [pos(d) + 1 for d in dates]
    out = {}
    for k in range(len(dates) - 1):
        d = dates[k]; p0 = L.AO[ex[k]]; p1 = L.AO[ex[k + 1]]
        fr = pd.Series(p1 / p0 - 1, index=COLS)
        ok = el.loc[d] & score.loc[d].notna() & fr.notna()
        if ok.sum() < 50: continue
        out[d] = score.loc[d][ok].rank().corr(fr[ok].rank())
    return pd.Series(out, dtype=float)
