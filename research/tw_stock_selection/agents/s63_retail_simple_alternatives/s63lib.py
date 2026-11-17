#!/usr/bin/env python3
"""S63 共用程式庫：散戶「做法」比較（同一引擎、同一套嚴成本）。
引擎一律 import 既有程式（s47lib → s20lib → s17lib → s10lib → … → stlib → cslib），檔案不改；三個正式開關由 s47lib import 時全開。
本檔只在執行期改「資料位置與樣本窗」（同 S55／R55 的做法）：
  'is' ：官方面板快取（sha256 434f6422…），START 2015-07-01、FIRST_MONTH 2016-01（現行）；
  'oos'：S55 自建 2007～2015H1 面板（sha256 見 S55 SPEC），stlib.START＝2008-01-02、s17lib.FIRST_MONTH＝2008-01。
自建的東西只有：訊號／出場矩陣（D 季換股＝把訊號月份篩成 1／4／7／10 月；F 懶人版）、空訊號（A）、0050 定期定額現金流（B）。
一律 python3 -I -B。暫存 scratchpad/s63/（結束刪除）。"""
import sys, math, hashlib
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
AG = HERE.parent
SCR = Path('/tmp/claude-0/-home-user/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/s63')
DATA_SHA = dict(
    is_={'panel_cs.npz': '434f64222b8fd1f10e1e25c9ac4553e7e6f6d11305896a3ceabb3b212650dfab'},
    oos={'panel_cs.npz': 'd169a22afd8c634f78757222a9eddd9080d2440dbd3892ca0307d93db0a2576e',
         'e50_ohlc.csv': '11a22824034f73661ec56d8d403e6343d86bd930f29229ad5d01ae719722fb14',
         'innov.npz': '7eb99a187427432888772d5d49d9b5369e29e3893983cc3d1aba5f6a667f6868'})
sys.path.insert(0, str(AG / 's47_allocation_family'))
import s47lib as L                                         # noqa: E402
from s47lib import s20lib, s17lib, s10lib, s9lib, s8lib, s7lib, s6lib, s5lib, s4lib, s3lib, s2lib, stlib, cslib, COSTS   # noqa: E402

sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
CAPS = (6e5, 8e5, 1e6, 1.5e6, 2e6)
SHIFTS = (0, 1, 2, 3)
QMONTHS = (1, 4, 7, 10)
FEE, TAX_ETF, MINFEE = s10lib.FEE, s10lib.TAX_ETF, s10lib.MINFEE


def setup(which):
    cache = SCR / ('cache_is' if which == 'is' else 'cache_pre')
    for f, h in DATA_SHA['is_' if which == 'is' else 'oos'].items():
        assert sha(cache / f) == h, f'{which} {f} sha256 不符'
    for f in ('panel_cs.npz', 'e50_ohlc.csv', 'innov.npz'):
        assert (cache / f).exists()
    if which == 'oos':
        stlib.START = pd.Timestamp('2008-01-02')
        s17lib.FIRST_MONTH = pd.Period('2008-01', 'M')
    for m, a in ((s20lib, 'S20'), (s17lib, 'S17'), (s10lib, 'S10'), (s9lib, 'S9'), (s8lib, 'S8'), (s7lib, 'S7'), (s6lib, 'S6'), (s5lib, 'S5'),
                 (s4lib, 'S4'), (s3lib, 'S3'), (s2lib, 'S2'), (stlib, 'S1'), (cslib, 'TMP')):
        setattr(m, a, cache)
    L.CACHE = cache
    assert cslib.EXRIGHT_FIX and cslib.DEADLINE_FIX and cslib.INS15_MASK
    return cache


def build(which):
    """面板、股池、SURGE（STD）、月營收矩陣、訊號日。同 S55 s55lib.build（rev_family 在三個開關全開下建＝S47／S55 的現行格）。"""
    setup(which)
    P = L.load()
    INNOV, _ = s4lib.innov_panel(P)
    U = stlib.universe(P, 'twse') & ~INNOV
    F = s17lib.rev_family(P, defs=['STD'])
    RT = dict(S=F['S']['STD'], av=F['av'])
    fix = s17lib.e50_fix_suspension(P)
    with np.errstate(invalid='ignore'):
        amt = np.where(np.isfinite(P.AMT), P.AMT, np.nan)
    P.AMT20 = pd.DataFrame(amt).rolling(20, min_periods=10).mean().to_numpy()      # F 用：近 20 個交易日平均成交值
    return P, U, RT, F['Rv'], fix


def rt_quarterly(P, RT):
    """D：只留「訊號日（位移 0）落在 1／4／7／10 月」的營收月份；其餘月份完全不動作（不買、不賣、不拉回）。"""
    S, av = RT['S'], RT['av']
    keep = [m for m in S.index if av[m] < P.T and P.IDX[av[m]].month in QMONTHS]
    return dict(S=S.loc[keep], av=av)


def empty_signal(P):
    """A：沒有任何個股訊號 → 引擎第一個交易日開盤把現金整筆掃進 0050，之後不動。"""
    return dict(sig=np.zeros((P.T, P.N), bool), rank=np.full((P.T, P.N), np.nan), X=np.zeros((P.T, P.N), bool), SD=None, max_hold=None,
                info=[], RQ={}, sigrows=np.zeros(P.T, bool))


def make_signal_lazy(P, univ, RT, Rv, shift=0, nsel=10):
    """F 懶人版：訊號列與現行同一組（同一個 A10 訊號日、同樣的跳過條件、同 INS15 遮罩），只是選股換成：
    當月營收 ≥ 近 12 個月（含當月）最高、12 個月都有值（視窗含 2013 年 IFRS 斷點前後混用者缺值：營收月份在 2013 年一律不出訊號，同現行慣例）；
    且股池內、還原收盤 > 60 日均線（P.F_MA60）；依近 20 日平均成交值由大到小取前 nsel 檔。
    出場：訊號列上「不在股池／不是 12 個月新高／收盤 ≤ 60 日均線」任一成立就標記，次日開盤賣（不看成交值名次）。"""
    S, av = RT['S'], RT['av']
    hi12 = (Rv >= Rv.rolling(12, min_periods=12).max()) & Rv.notna()
    full12 = Rv.notna().rolling(12, min_periods=12).sum() == 12
    NH = (hi12 & full12).astype(float).where(full12)       # 1＝新高、0＝不是、NaN＝資料不足
    NH[NH.index.year == 2013] = np.nan
    sig = np.zeros((P.T, P.N), bool); rank = np.full((P.T, P.N), np.nan); X = np.zeros((P.T, P.N), bool)
    info = []
    for m in S.index:
        k0 = av[m]
        if k0 >= P.T or k0 < 61:
            continue
        s = S.loc[m].to_numpy()
        if int((univ[k0] & np.isfinite(s)).sum()) < 50:    # 與現行同一組訊號列
            continue
        k = k0 + shift
        if k >= P.T:
            continue
        nh = NH.loc[m].to_numpy() if m in NH.index else np.full(P.N, np.nan)
        nh = cslib.ins15_mask_row(P, m, k, nh)
        ok = univ[k] & (np.nan_to_num(nh, nan=0.0) > 0.5) & P.F_MA60[k]
        liq = np.where(ok, np.nan_to_num(P.AMT20[k], nan=0.0), -np.inf)
        cand = np.flatnonzero(ok)
        tgt = cand[np.argsort(-liq[cand], kind='stable')][:nsel]
        sig[k, tgt] = True; rank[k, tgt] = liq[tgt]
        X[k, :] = ~ok
        info.append(dict(month=str(m), k=int(k), sig_date=str(P.IDX[k].date()), n_ma=int(ok.sum()), n_tgt=int(len(tgt)),
                         targets=[str(P.COLS[j]) for j in tgt]))
    sigrows = np.zeros(P.T, bool); sigrows[[x['k'] for x in info]] = True
    return dict(sig=sig, rank=rank, X=X, SD=None, max_hold=None, info=info, RQ={}, sigrows=sigrows)


def sim(P, S, U, sat_w=0.7, div=10, slots=10, cap=1e6):
    """現行參數（÷10、10 檔）→ 原引擎 simulate_s10；E（÷5、5 檔）→ s47lib.simulate_s47（S47 的引擎副本，ndiv 參數）。"""
    kw = dict(cost=COSTS['strict'], capital=cap, cap_pct=0.05, sat_w=sat_w, sizing='W10', lock=False)
    if div == 10 and slots == 10:
        return s10lib.simulate_s10(P, S, U, **kw)
    return L.simulate_s47(P, S, U, ndiv=div, idle='e50', slots=slots, **kw)


# ---------------------------------------------------------------- B：0050 定期定額（同引擎 e50_net 的買進成本式）
def _buy50(P, t, x, cm):
    """t 日開盤花 x 元買 0050，回傳買到的市值（以 t 日開盤計）。成本式＝s10lib.simulate_s10.e50_net 的掃入：手續費 max(20, x×費率)、跳動點半格。"""
    po = P.E50O_raw[t] if np.isfinite(P.E50O_raw[t]) and P.E50O_raw[t] > 0 else 100.0
    sl50 = 0.5 * cm * stlib.etf_tick(po) / po
    f = max(MINFEE * cm, x * FEE * cm)
    return (x - f) / (1 + sl50)


def dca(P, a, b, monthly, shift=0, n_months=None, lump=0.0):
    """a～b（Timestamp；a 為起點基底日，之後每個日曆月的第 1＋shift 個交易日開盤買 monthly 元）。
    n_months＝只扣前 n 個月（None＝每月都扣到期末）；lump＝起點已到位的現金（放著不計息，逐月移入）。
    回傳 dict：日淨值（帳戶＝0050 市值＋未投入現金）、現金流、內部報酬率（年化，日頻折現）、期末倍數。外部現金流版（lump＝0）淨值只供參考。"""
    cm = COSTS['strict']['cmult']
    idx = P.IDX; ia = int(np.searchsorted(idx.values, np.datetime64(a), side='left')); ib = int(np.searchsorted(idx.values, np.datetime64(b), side='right')) - 1
    per = idx.to_period('M')
    units = 0.0; cash = float(lump); flows = []; nav = np.full(ib - ia + 1, np.nan); nav[0] = cash
    cnt = {}; nb = 0
    for t in range(ia + 1, ib + 1):
        p = per[t]; cnt[p] = cnt.get(p, 0) + 1
        if cnt[p] == 1 + shift and (n_months is None or nb < n_months):
            x = monthly
            if lump > 0:
                x = min(monthly, cash); cash -= x
            if x > 0:
                units += _buy50(P, t, x, cm) / P.E50O[t]; nb += 1
                if lump <= 0:
                    flows.append((t, x))
        nav[t - ia] = units * P.E50[t] + cash
    fv = units * P.E50[ib] + cash
    out = dict(nav=pd.Series(nav, index=idx[ia:ib + 1]), n_buys=nb, final=float(fv))
    if lump <= 0:
        tot = sum(x for _, x in flows)
        days = np.array([(idx[ib] - idx[t]).days for t, _ in flows], float); amt = np.array([x for _, x in flows])
        lo, hi = -0.99, 5.0
        for _ in range(200):
            mid = (lo + hi) / 2
            if (amt * (1 + mid) ** (days / 365.25)).sum() > fv:
                hi = mid
            else:
                lo = mid
        out.update(invested=float(tot), multiple=float(fv / tot), irr=float((lo + hi) / 2))
    return out


# ---------------------------------------------------------------- 統計（月超額同 S52／S55：日曆月底、第一個月以起日為基底、NW Bartlett 落後 6）
def month_points(s):
    s = s.dropna()
    me = s.groupby(s.index.to_period('M')).tail(1)
    return pd.concat([s.iloc[:1], me[me.index > s.index[0]]])


def monthly_returns(nav, bench):
    q = bench.reindex(nav.index)
    rs = month_points(nav).pct_change().dropna(); rb = month_points(q).pct_change().dropna()
    rs.index = rs.index.to_period('M'); rb.index = rb.index.to_period('M')
    return rs, rb


def nw_t(x, lag=6):
    x = np.asarray(x, float); T = len(x); m = x.mean(); e = x - m
    lr = (e * e).sum() / T
    for i in range(1, lag + 1):
        lr += 2 * (1 - i / (lag + 1)) * (e[i:] * e[:-i]).sum() / T
    return float(m / math.sqrt(lr / T)) if lr > 0 else float('nan')


def stats_block(nav, bench, a, b):
    """a～b 的日淨值（a 為基底日）：幾何年化（stlib.cagr 口徑）、最大回撤、Calmar、對 0050 幾何差與月超額（算術年化、NW t）。"""
    s = nav.loc[a:b].dropna(); q = bench.reindex(s.index)
    cg, cb = stlib.cagr(s), stlib.cagr(q)
    dd = float((s / s.cummax() - 1).min())
    rs, rb = monthly_returns(s, q)
    ex = (rs - rb).to_numpy(); n = len(ex)
    sd = ex.std(ddof=1) if n > 1 else float('nan')
    return dict(cagr=cg, cagr_0050=cb, geo_diff_pp=(cg - cb) * 100, maxdd=dd, calmar=(cg / abs(dd) if dd < 0 else float('nan')), months=int(n),
                ex_ann_pp=float(ex.mean() * 12 * 100), te_ann_pp=float(sd * math.sqrt(12) * 100) if n > 1 else float('nan'),
                t_nw6=nw_t(ex, 6) if n > 12 and sd > 0 else float('nan'), maxdd_0050=float((q / q.cummax() - 1).min()))
