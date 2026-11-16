#!/usr/bin/env python3
"""S30 共用程式庫：底倉與空位從 0050 換成「低 β＋高品質」防禦組合（Buffett's Alpha：Frazzini、Kabiller、Pedersen 2018）。
在 s20lib（不改；其下 s17lib→s10lib：面板、創新板、SURGE（STD，σ>0）、F1、E0、引擎 simulate_s10）之上，本檔只新增：
- 防禦組合（PIT，每季再平衡，計成本）的持股與日淨值（收盤指數 Dc、開盤指數 Do）：
  股池＝季末最後交易日 d：上市（MK＝1）、當日有成交、非 91 開頭、非創新板、PIT 市值（股數落後 63 交易日，同 s24）前 150 名、
        過去 252 個交易日中有成交且有報酬的日子 ≥ 200。
  β＝過去 252 日日報酬對 0050 含息日報酬的 OLS 斜率（只用股票有成交的日子；剔除 0050 分割停牌 2025-06-11～17）；
  VOL＝同窗日報酬標準差；ROE＝最近已公告（s24 的 PIT 可得日）TTM 歸屬母公司淨利 ÷ 最近一季歸屬母公司權益（權益 > 0、四季連續）；
  DY＝d 之前 365 天內 TWT49U「息」「權息」事件（權值＋息值）÷ 除權息前收盤 的合計（權息事件含股票股利部分；純除權事件不計）。
  LOWB  ：β 最低 30 檔等權
  LBQ   ：β 最低一半（75 檔）中 ROE 最高 30 檔等權（課程的「低 β ∩ 高品質」）
  HDLV  ：VOL 最低一半（75 檔）中 DY 最高 30 檔等權（同分依 VOL 低者）
  MIX   ：50% 0050＋50% LBQ，每月最後交易日收盤拉回 50／50
  成交：d 的下一個交易日收盤再平衡到等權；成本＝手續費（同引擎 0.1425% × 0.28）＋賣出證交稅 0.3%＋滑價（半個跳動點＋0.3%，同 strict）。
  2015-12-31 前（第一次形成前）底倉＝0050；第一次換倉另計「賣 0050（稅 0.1%）＋買組合」成本。成分股停牌或下市 → 報酬 0（價值凍結到下次再平衡）。
- base_swap()／sim()：最小包裝（不改共用引擎）：模擬期間把 P.E50／E50O／E50O_raw／E50C_raw 暫時換成防禦組合指數，並把引擎內
  0050 換手成本換成組合的成本（s10lib.TAX_ETF＝0.3%、stlib.etf_tick → 每邊滑價 SB），結束後全部還原。B＝None 時完全不動（＝現行）。
中間檔放 scratchpad/s30/。規格見 PREREG.json。一律 python3 -I -B。"""
import sys, math
from contextlib import contextmanager
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
AG = HERE.parent
S30 = Path('/tmp/claude-0/-home-user--/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/s30')
sys.path.insert(0, str(AG / 's20_persistent_growth'))
import s20lib                                              # noqa: E402
from s20lib import s17lib, s10lib, s9lib, s8lib, s7lib, s6lib, s5lib, s4lib, s3lib, s2lib, stlib, cslib, COSTS   # noqa: E402,F401
s20lib.S20 = S30; s17lib.S17 = S30; s10lib.S10 = S30; s9lib.S9 = S30; s8lib.S8 = S30; s7lib.S7 = S30; s6lib.S6 = S30; s5lib.S5 = S30; s4lib.S4 = S30
s3lib.S3 = S30; s2lib.S2 = S30; stlib.S1 = S30; cslib.TMP = S30
cslib.EXRIGHT_FIX = True

TRAIN_LAST_MONTH = s20lib.TRAIN_LAST_MONTH                 # 營收月份 ≤ 2021-11（訊號日 2016-02～2021-12）
VALID_FIRST_MONTH = s20lib.VALID_FIRST_MONTH               # 營收月份 ≥ 2021-12（訊號日 2022-01 起）
SEGMENTS = s17lib.SEGMENTS
fresh_from = s20lib.fresh_from
PEAD = AG / 'o_pead_margin_surprise_low_attention' / 'data'
SHARES = AG / 'o_lc_sector_rotation_pit' / 'data' / 'shares_full.parquet'
TOPN, NB, NHALF, WIN, MINOBS = 150, 30, 75, 252, 200
XSLIP = 0.003
FEE = s10lib.FEE                                           # 0.001425 × 0.28
TAX_STK, TAX_ETF0 = 0.003, s10lib.TAX_ETF
SB_BASKET = 0.004                                          # 組合整籃換手每邊滑價（0.3% 額外＋約 0.1% 半個跳動點）；引擎資金進出組合時使用
FLOW = {'LOWB': dict(sb=SB_BASKET, tax=TAX_STK), 'LBQ': dict(sb=SB_BASKET, tax=TAX_STK), 'HDLV': dict(sb=SB_BASKET, tax=TAX_STK),
        'MIX': dict(sb=0.5 * SB_BASKET + 0.5 * 0.0002, tax=0.5 * TAX_STK + 0.5 * TAX_ETF0)}
BASKETS = ('LOWB', 'LBQ', 'HDLV')


def load():
    S30.mkdir(parents=True, exist_ok=True)
    P = s20lib.load()
    assert P.exright_fix, '必須開 EXRIGHT_FIX'
    return P


def surge(P):
    F = s17lib.rev_family(P, defs=['STD'])
    return dict(S=F['S']['STD'], av=F['av'])


# ---------------------------------------------------------------- 形成日與 PIT 特徵
def formation_rows(P):
    """每季最後一個交易日（3、6、9、12 月），2015-12 起到 2026-09；回傳 [(形成列 t, 成交列 t+1)]。"""
    idx = pd.DatetimeIndex(P.IDX)
    s = pd.Series(np.arange(P.T), index=idx)
    me = s.groupby(idx.to_period('M')).max()
    out = []
    for m, t in me.items():
        if m.month in (3, 6, 9, 12) and m >= pd.Period('2015-12', 'M') and t + 1 < P.T:
            out.append((int(t), int(t + 1)))
    return out


def mcap(P):
    """同 s24：PIT 市值＝63 交易日前股數 × 收盤（股數檔沒有的股票 NaN）。"""
    sh = pd.read_parquet(SHARES)
    sh = sh.reindex(sh.index.union(P.IDX)).ffill().reindex(P.IDX).reindex(columns=P.COLS).to_numpy(np.float64)
    now = sh * P.CF
    LAG = 63
    MC = np.full((P.T, P.N), np.nan)
    MC[LAG:] = now[:-LAG] * (P.G[LAG:] / P.G[:-LAG])
    return np.where(np.isfinite(MC), MC, now)


def e50_ret(P):
    r = np.r_[np.nan, P.E50[1:] / P.E50[:-1] - 1]
    idx = pd.DatetimeIndex(P.IDX)
    r[(idx >= pd.Timestamp(s17lib.SUSP[0])) & (idx <= pd.Timestamp(s17lib.SUSP[1]))] = np.nan
    return r


def beta_vol(P, t, rm):
    """形成列 t：過去 252 列（t−251…t）。回傳 β、VOL、有效日數（長度 N）。"""
    a = max(0, t - WIN + 1)
    X = np.where(P.has[a:t + 1], P.R[a:t + 1], np.nan)
    y = rm[a:t + 1][:, None]
    ok = np.isfinite(X) & np.isfinite(y)
    n = ok.sum(0).astype(float)
    with np.errstate(invalid='ignore', divide='ignore'):
        xm = np.where(ok, X, 0).sum(0) / n
        ym = np.where(ok, y, 0).sum(0) / n
        dx = np.where(ok, X - xm, 0); dy = np.where(ok, y - ym, 0)
        beta = (dx * dy).sum(0) / (dy * dy).sum(0)
        vol = np.sqrt((dx * dx).sum(0) / (n - 1))
    beta[n < MINOBS] = np.nan; vol[n < MINOBS] = np.nan
    return beta, vol, n


def roe_at(P, dates):
    """dates：Timestamp 清單 → dict(date → 長度 N 的 ROE)。照 s24 的 PIT 可得日（A，缺值用法定期限）。"""
    f = pd.read_parquet(PEAD / 'fin_long.parquet', columns=['code', 'y', 'q', 'nip_s', 'equity_parent'])
    f = f[f['code'].str.fullmatch(r'[1-9]\d{3}')].copy()
    f['kq'] = f['y'] * 4 + f['q'] - 1
    a = pd.read_parquet(PEAD / 'avail_dates.parquet', columns=['code', 'y', 'q', 'A'])
    f = f.merge(a, on=['code', 'y', 'q'], how='left')
    dl = {1: (5, 15), 2: (8, 14), 3: (11, 14)}
    f['deadline'] = [pd.Timestamp(y + (q == 4), *(dl[q] if q < 4 else (3, 31))) for y, q in zip(f['y'], f['q'])]
    f['avail'] = pd.to_datetime(f['A']).fillna(f['deadline'])
    f = f.sort_values(['code', 'kq'])
    g = f.groupby('code')
    f['ttm'] = g['nip_s'].transform(lambda s: s.rolling(4, min_periods=4).sum())
    f['kq_lag3'] = g['kq'].shift(3)
    f.loc[f['kq'] - f['kq_lag3'] != 3, 'ttm'] = np.nan
    f['roe'] = np.where(f['equity_parent'] > 0, f['ttm'] / f['equity_parent'], np.nan)
    ci = {c: j for j, c in enumerate(P.COLS)}
    f = f[f['code'].isin(ci)]
    out = {}
    for d in dates:
        x = f[f['avail'] <= d]
        last = x.groupby('code').last()
        v = np.full(P.N, np.nan)
        js = np.array([ci[c] for c in last.index])
        v[js] = last['roe'].to_numpy(float)
        out[d] = v
    return out


def dy_at(P, dates):
    """d 之前 365 天（不含 d 之後）TWT49U 事件 value ÷ prev_close 合計；沒有事件＝0。"""
    tw = cslib.load_twt49u()
    tw = tw.copy()
    tw['date'] = pd.to_datetime(tw['date'])
    tw = tw[tw['code'].astype(str).isin(set(P.COLS))]
    tw = tw[tw['type'].astype(str).str.contains('息')]           # 只用「息」「權息」事件（排除純除權：股票股利、現增）
    tw['y'] = pd.to_numeric(tw['value'], errors='coerce') / pd.to_numeric(tw['prev_close'], errors='coerce')
    ci = {c: j for j, c in enumerate(P.COLS)}
    out = {}
    for d in dates:
        x = tw[(tw['date'] <= d) & (tw['date'] > d - pd.Timedelta(days=365))]
        s = x.groupby(x['code'].astype(str))['y'].sum()
        v = np.zeros(P.N)
        for c, y in s.items():
            if np.isfinite(y):
                v[ci[c]] = y
        out[d] = v
    return out


def holdings(P, INNOV, MC=None):
    """每個形成日的三組持股（欄號）與診斷。回傳 dict(name → {t_form: array}), info(list)。"""
    FR = formation_rows(P)
    if MC is None:
        MC = mcap(P)
    rm = e50_ret(P)
    dates = [pd.Timestamp(P.IDX[t]) for t, _ in FR]
    ROE = roe_at(P, dates); DY = dy_at(P, dates)
    H = {b: {} for b in BASKETS}; info = []
    for (t, te), d in zip(FR, dates):
        beta, vol, n = beta_vol(P, t, rm)
        mc = MC[t]
        elig = (P.MK[t] == 1) & P.has[t] & P.not91 & ~INNOV[t] & np.isfinite(mc) & (mc > 0) & np.isfinite(beta) & np.isfinite(vol)
        cand = np.flatnonzero(elig)
        top = cand[np.argsort(-mc[cand], kind='stable')][:TOPN]
        ob = top[np.lexsort((top, beta[top]))]                        # β 由低到高（同值依欄號）
        H['LOWB'][t] = np.sort(ob[:NB])
        lowhalf = ob[:NHALF]
        r = ROE[d][lowhalf]
        okr = lowhalf[np.isfinite(r)]
        H['LBQ'][t] = np.sort(okr[np.lexsort((okr, -ROE[d][okr]))][:NB])
        ov = top[np.lexsort((top, vol[top]))]
        lv = ov[:NHALF]
        H['HDLV'][t] = np.sort(lv[np.lexsort((vol[lv], -DY[d][lv]))][:NB])
        info.append(dict(form=str(d.date()), trade=str(P.IDX[te].date()), n_elig=int(elig.sum()), n_top=int(len(top)),
                         mc_cut=float(mc[top[-1]]) if len(top) else None, lbq_n_roe=int(len(okr)),
                         beta_med_top=float(np.median(beta[top])), **{f'n_{b}': int(len(H[b][t])) for b in BASKETS},
                         **{f'beta_{b}': float(np.mean(beta[H[b][t]])) for b in BASKETS},
                         roe_LBQ=float(np.nanmedian(ROE[d][H['LBQ'][t]])), dy_HDLV=float(np.median(DY[d][H['HDLV'][t]])),
                         dy_top=float(np.median(DY[d][top])), roe_top=float(np.nanmedian(ROE[d][top]))))
    return H, info, FR


# ---------------------------------------------------------------- 組合指數
def _slip(px):
    px = np.asarray(px, float)
    with np.errstate(invalid='ignore', divide='ignore'):
        s = 0.5 * cslib.tick(px) / px + XSLIP
    return np.where(np.isfinite(s), s, SB_BASKET)


def basket_index(P, Hb, FR):
    """等權、每季於成交列收盤再平衡。回傳 dict(Dc, Do, turnover（每次再平衡單邊換手率）, cost（每次成本占淨值）)。
    第一次成交列之前（含當天收盤前）＝0050（E50／E50O）。"""
    T = P.T
    Dc = np.full(T, np.nan); Do = np.full(T, np.nan)
    tes = {te: t for t, te in FR}
    first = FR[0][1]
    Dc[:first + 1] = P.E50[:first + 1]; Do[:first + 1] = P.E50O[:first + 1]
    W = float(P.E50[first])
    # 第一次換倉：賣 0050＋買組合
    hold = Hb[FR[0][0]]
    po = P.E50C_raw[first] if np.isfinite(P.E50C_raw[first]) and P.E50C_raw[first] > 0 else 100.0
    c_sell = FEE + TAX_ETF0 + 0.5 * stlib.etf_tick(po) / po
    sl = _slip(P.CF[first, hold])
    c_buy = FEE + float(np.mean(sl))
    W = W * (1 - c_sell) / (1 + c_buy)
    v = np.full(len(hold), W / len(hold))
    turn = [dict(date=str(P.IDX[first].date()), turnover=1.0, cost=1 - (1 - c_sell) / (1 + c_buy))]
    Dc[first] = W
    for t in range(first + 1, T):
        o = P.O[t, hold]; ref = P.REF[t, hold]
        ro = np.where(np.isfinite(o) & (o > 0) & np.isfinite(ref) & (ref > 0), o / ref, 1.0)
        tot0 = v.sum()
        Do[t] = Dc[t - 1] * float((v * ro).sum() / tot0)
        v = v * (1 + P.R[t, hold])
        Dc[t] = Dc[t - 1] * float(v.sum() / tot0)
        if t in tes:                                               # 收盤再平衡
            new = Hb[tes[t]]
            Wt = float(v.sum())
            cur = dict(zip(hold.tolist(), v.tolist()))
            tgt0 = Wt / len(new)
            allj = sorted(set(cur) | set(new.tolist()))
            sl = dict(zip(allj, _slip(P.CF[t, allj]).tolist()))
            newset = set(new.tolist())
            cost = 0.0
            for _ in range(3):
                tgt = (Wt - cost) / len(new)
                cost = 0.0; to = 0.0
                for j in allj:
                    d = (tgt if j in newset else 0.0) - cur.get(j, 0.0)
                    if d < 0:
                        cost += -d * (FEE + TAX_STK + sl[j]); to += -d
                    else:
                        cost += d * (FEE + sl[j])
            tgt = (Wt - cost) / len(new)
            hold = new; v = np.full(len(new), tgt)
            turn.append(dict(date=str(P.IDX[t].date()), turnover=to / Wt, cost=cost / Wt))
            Dc[t] = Dc[t] * (v.sum() / Wt)                         # v 的總和＝指數水準（扣成本後）
        else:
            v = v * (Dc[t] / v.sum())
    return dict(Dc=Dc, Do=Do, turns=turn)


def mix_index(P, A, B, wa=0.5):
    """A＝0050（E50／E50O）、B＝組合指數 dict；每月最後交易日收盤拉回 wa／1−wa，移轉金額計雙邊成本。"""
    T = P.T
    idx = pd.DatetimeIndex(P.IDX)
    me = set(pd.Series(np.arange(T), index=idx).groupby(idx.to_period('M')).max().tolist())
    Ec, Eo = P.E50, P.E50O
    Bc, Bo = B['Dc'], B['Do']
    Dc = np.full(T, np.nan); Do = np.full(T, np.nan)
    first = int(np.flatnonzero(np.isfinite(Bc))[0])
    t_start = P.t0
    Dc[:t_start + 1] = Ec[:t_start + 1]; Do[:t_start + 1] = Eo[:t_start + 1]
    a = wa * Ec[t_start]; b = (1 - wa) * Ec[t_start]                   # 起點即 50／50（B 在第一次形成前＝0050，等同 0050）
    costs = []
    for t in range(t_start + 1, T):
        tot0 = a + b
        Do[t] = Dc[t - 1] * (a * Eo[t] / Ec[t - 1] + b * Bo[t] / Bc[t - 1]) / tot0
        a *= Ec[t] / Ec[t - 1]; b *= Bc[t] / Bc[t - 1]
        Dc[t] = Dc[t - 1] * (a + b) / tot0
        if t in me:
            W = a + b; x = a - wa * W                                  # x > 0：0050 → 組合
            po = P.E50C_raw[t] if np.isfinite(P.E50C_raw[t]) and P.E50C_raw[t] > 0 else 100.0
            c50 = FEE + 0.5 * stlib.etf_tick(po) / po
            if x > 0:
                c = x * (c50 + TAX_ETF0) + x * (FEE + SB_BASKET)
            else:
                c = -x * (FEE + TAX_STK + SB_BASKET) + -x * c50
            W2 = W - c
            a = wa * W2; b = (1 - wa) * W2
            Dc[t] = Dc[t] * W2 / W
            costs.append(c / W)
        s = Dc[t] / (a + b); a *= s; b *= s
    return dict(Dc=Dc, Do=Do, turns=[dict(cost=float(np.sum(costs)), n=len(costs))])


def base_arrays(P, name, IDXS):
    """回傳 base_swap 用的 dict（E50、E50O、E50O_raw、E50C_raw、sb、tax）。name='E50' → None（現行，不換）。"""
    if name in (None, 'E50'):
        return None
    D = IDXS[name]
    f = FLOW[name]
    return dict(E50=np.array(D['Dc'], float), E50O=np.array(D['Do'], float), E50O_raw=np.full(P.T, 100.0), E50C_raw=np.full(P.T, 100.0),
                sb=f['sb'], tax=f['tax'])


@contextmanager
def base_swap(P, B):
    """B＝None：完全不動。B＝dict：暫時換 0050 陣列與成本，離開時還原（含例外）。
    B 也可只給陣列（sb＝None）：只換陣列、不換成本函數（驗收「包裝本身不改結果」用）。"""
    if B is None:
        yield; return
    keys = ('E50', 'E50O', 'E50O_raw', 'E50C_raw')
    old = {k: getattr(P, k) for k in keys}
    old_tick, old_tax = stlib.etf_tick, s10lib.TAX_ETF
    try:
        for k in keys:
            setattr(P, k, B[k])
        if B.get('sb') is not None:
            sb = float(B['sb'])
            stlib.etf_tick = lambda p, _sb=sb: 2.0 * _sb * p           # 引擎：sl50＝0.5 × cm × etf_tick(po) ÷ po＝cm × sb
            s10lib.TAX_ETF = float(B['tax'])
        yield
    finally:
        for k in keys:
            setattr(P, k, old[k])
        stlib.etf_tick = old_tick; s10lib.TAX_ETF = old_tax


CELL = dict(sat_w=0.7, sizing='W10', lock=False)


def sim(P, S1, U, B=None, cap=1e6, cost='strict', block=None, fresh=False):
    if fresh:
        S1 = fresh_from(S1, VALID_FIRST_MONTH)
    S2, ekw = s10lib.variant_S(S1, 'E0')
    with base_swap(P, B):
        return s10lib.simulate_s10(P, S2, U, cost=COSTS[cost], capital=cap, block=block, **CELL, **ekw)


def capm_beta(nav, bench, a=None, b=None, freq='D'):
    """日（或月）報酬對 0050 報酬的 OLS β 與 α（年化）。"""
    x = nav.loc[a:b].dropna(); y = bench.reindex(x.index)
    if freq == 'M':
        x = x.groupby(x.index.to_period('M')).last(); y = y.groupby(y.index.to_period('M')).last()
    rx = x.pct_change().dropna(); ry = y.pct_change().reindex(rx.index)
    ok = np.isfinite(rx.to_numpy()) & np.isfinite(ry.to_numpy()) & (ry.to_numpy() != 0 if freq == 'D' else True)
    X = ry.to_numpy()[ok]; Y = rx.to_numpy()[ok]
    bt = float(np.cov(X, Y)[0, 1] / np.var(X, ddof=1))
    al = float(np.mean(Y) - bt * np.mean(X)) * (252 if freq == 'D' else 12)
    return dict(beta=bt, alpha_ann=al, n=int(ok.sum()), corr=float(np.corrcoef(X, Y)[0, 1]))


def describe_index(Dc, bench, a, b):
    s = pd.Series(Dc, index=bench.index).loc[a:b]
    r = s.pct_change().dropna()
    return dict(cagr=stlib.cagr(s), vol=float(r.std() * np.sqrt(252)), maxdd=stlib.maxdd(s), capm=capm_beta(s, bench), seg=s17lib.seg_dd(s))
