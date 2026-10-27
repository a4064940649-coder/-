#!/usr/bin/env python3
"""industry_revenue_long 共用：產業營收指標（2005 起長期月營收）＋ M2 引擎（import revenue_longhistory/lhlib）。

產業別：stock_list.csv 的 industry（現行分類往回套）；面板中不在 stock_list 的已下市股，傳統產業代碼段依前兩碼回推
（同 agents/industry/ilib.PREFIX_IND），已下市電子股無法回推 → 無產業（中性處理，不排除）。
"""
import math, sys
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
AG = HERE.parent
sys.path.insert(0, str(AG / 'revenue_longhistory'))
import lhlib as L  # noqa: E402
from lhlib import tstat  # noqa: E402

PREFIX_IND = {'11': '水泥工業', '12': '食品工業', '13': '塑膠工業', '14': '紡織纖維', '15': '電機機械', '16': '電器電纜',
              '17': '化學工業', '18': '玻璃陶瓷', '19': '造紙工業', '20': '鋼鐵工業', '21': '橡膠工業', '22': '汽車工業',
              '25': '建材營造業', '26': '航運業', '27': '觀光餐旅', '28': '金融保險業', '29': '貿易百貨業', '99': '其他業'}
EXCLUDE_IND = {'金融保險業'}
ELEC = {'半導體業', '電腦及週邊設備業', '光電業', '通信網路業', '電子零組件業', '電子通路業', '資訊服務業', '其他電子業', '數位雲端'}
MIN_COS = 5

def industry_map(coarse=False):
    sl = pd.read_csv('/home/user/stevenhsiehsheng-ctrl/tw-stock-screener/data/stock_list.csv', dtype={'code': str})
    ind = sl.drop_duplicates('code').set_index('code')['industry']
    m = {}; npref = 0
    for c in L.COLS:
        if c in ind.index and isinstance(ind[c], str): m[c] = ind[c]
        elif c[:2] in PREFIX_IND: m[c] = PREFIX_IND[c[:2]]; npref += 1
    s = pd.Series(m)
    if coarse: s = s.where(~s.isin(ELEC), '電子工業')
    s = s[~s.isin(EXCLUDE_IND)]
    return s, npref

def revenue_levels():
    d = pd.read_csv(L.LONG, dtype={'code': str, 'market': str, 'name': str})
    d['p'] = pd.PeriodIndex(pd.to_datetime(d['month']), freq='M')
    d = d.sort_values(['p', 'market']).drop_duplicates(['p', 'code'], keep='first')
    d = d[d['code'].isin(L.COLS)]
    Lv = d.pivot_table(index='p', columns='code', values='revenue_k_twd', aggfunc='last').sort_index()
    return Lv.reindex(pd.period_range(Lv.index.min(), Lv.index.max(), freq='M'))

def industry_indicators(Lv, IND):
    """回傳 dict：YOY3／DIFF／ACC，皆為 (月 Period × 產業) 表。加總後算年增率；配對樣本＝6 個月都有正值的公司。"""
    Lv = Lv.where(Lv > 0)
    R3 = Lv.rolling(3, min_periods=3).sum(); R3p = R3.shift(12)
    ok = R3.notna() & R3p.notna()
    cols = [c for c in Lv.columns if c in IND.index]
    g = IND[cols]
    num = R3[cols].where(ok[cols]).T.groupby(g).sum(min_count=1).T
    den = R3p[cols].where(ok[cols]).T.groupby(g).sum(min_count=1).T
    n = ok[cols].T.groupby(g).sum().T
    yoy3 = (num / den - 1).where(n >= MIN_COS)
    pos = ((R3 / R3p) > 1).astype(float)[cols].where(ok[cols])
    n = n.astype(float)
    diff = (pos.T.groupby(g).sum().T / n).where(n >= MIN_COS)
    acc = yoy3 - yoy3.shift(3)
    return {'YOY3': yoy3, 'DIFF': diff, 'ACC': acc, 'N': n}

def ind_on_dates(tab, dates):
    mm = [pd.Period(d, 'M') - 1 for d in dates]
    return pd.DataFrame(tab.reindex(mm).to_numpy(), index=pd.DatetimeIndex(dates), columns=tab.columns)

def select_generic(S, sc, el, n, held_rule=True):
    """M2：照抄 lhlib.Setup.selection 的成交規則（訊號日收盤分數，隔日開盤成交，開盤漲停且未持有者跳過遞補）。"""
    GU = L.GAP_UP_OPEN_ERA
    events = {}; held = set(); nblk = 0
    for d in S.dates[:-1]:
        i = L.pos(d); ex = i + 1
        okb = el.loc[d].to_numpy()
        trad = np.isfinite(L.OPEN_RAW[ex]) & np.isfinite(L.ADJ_RAW[ex])
        s = sc.loc[d].to_numpy()
        ok = okb & trad & np.isfinite(s)
        if ok.sum() < 50: continue
        order = s.copy(); order[~ok] = np.nan
        cand = np.argsort(-np.nan_to_num(order, nan=-9))[:int(ok.sum())]
        pick = []
        for j in cand:
            if GU[ex, j] and j not in held: nblk += 1; continue
            pick.append(j)
            if len(pick) == n: break
        events[ex] = ('open', np.array(pick)); held = set(pick)
    return events, nblk

def select_rotation(S, indval, IND, within, n_ind=3, n_per=10, base=None, mainscore=None, min_names=5):
    """(A) 產業輪動：依產業指標選前 n_ind 產業，產業內前 n_per（流動性或主策略分數）等權；M2 成交規則。"""
    GU = L.GAP_UP_OPEN_ERA
    gcode = pd.Series(IND.reindex(L.COLS).to_numpy(), index=L.COLS)
    events = {}; held = set(); nblk = 0; chosen = {}
    v60 = L.v60_fix.reindex(S.DATES)
    for d in S.dates[:-1]:
        i = L.pos(d); ex = i + 1
        trad = np.isfinite(L.OPEN_RAW[ex]) & np.isfinite(L.ADJ_RAW[ex])
        okb = base.loc[d].to_numpy() & trad
        if within == 'main':
            ms = mainscore.loc[d].to_numpy(); okb = okb & np.isfinite(ms); key = ms
        else:
            key = v60.loc[d].to_numpy()
        iv = indval.loc[d].dropna()
        cnt = pd.Series(okb, index=L.COLS).groupby(gcode).sum()
        iv = iv[iv.index.isin(cnt[cnt >= min_names].index)]
        if len(iv) < 6: continue
        top = list(iv.sort_values(ascending=False).index[:n_ind]); chosen[str(d.date())] = top
        pick = []
        for gname in top:
            idx = np.where(okb & (gcode.to_numpy() == gname))[0]
            idx = idx[np.argsort(-key[idx])]
            k = 0
            for j in idx:
                if GU[ex, j] and j not in held: nblk += 1; continue
                pick.append(j); k += 1
                if k == n_per: break
        events[ex] = ('open', np.array(pick)); held = set(pick)
    return events, nblk, chosen

def monthly(nav, DATES):
    nd = nav.reindex(DATES); return (nd.shift(-1) / nd - 1).iloc[:-1]

def in_segs(idx, segs):
    m = np.zeros(len(idx), bool)
    for a, b in segs: m |= (idx >= pd.Timestamp(a)) & (idx <= pd.Timestamp(b))
    return m

def full_stats(nav, segs, DATES, ew, main=None):
    o = L.daily_stats(nav, segs, DATES, L.E50, ew)
    if o is None: return None
    if main is not None:
        a = monthly(nav, DATES); b = monthly(main, DATES)
        dlt = (a - b)[in_segs(a.index, segs)].dropna()
        o['excess_vs_main_pct_m'] = round(float(dlt.mean() * 100), 2); o['t_vs_main_m'] = tstat(dlt)
        o['hit_vs_main_m'] = round(float((dlt > 0).mean()), 3)
        o['corr_with_main_m'] = round(float(a[in_segs(a.index, segs)].corr(b[in_segs(b.index, segs)])), 3)
        mm = L.daily_stats(main, segs, DATES)
        o['main_CAGR'] = mm['CAGR']; o['main_maxDD'] = mm['maxDD']; o['main_yearly_pct'] = mm['yearly_pct']
    return o

def nw_t(y, x, lag, coef=1):
    """OLS y = a + b·x 的 Newey-West t（Bartlett 核）；x 可為多欄，回傳第 coef 個係數（1＝第一個解釋變數）的 t。"""
    y = np.asarray(y, float); x = np.asarray(x, float); n = len(y)
    X = np.column_stack([np.ones(n), x]); XtXi = np.linalg.inv(X.T @ X); b = XtXi @ X.T @ y; e = y - X @ b
    Z = X * e[:, None]; Sm = Z.T @ Z
    for l in range(1, lag + 1):
        w = 1 - l / (lag + 1); G = Z[l:].T @ Z[:-l]; Sm += w * (G + G.T)
    V = XtXi @ Sm @ XtXi
    return float(b[coef] / math.sqrt(V[coef, coef]))
