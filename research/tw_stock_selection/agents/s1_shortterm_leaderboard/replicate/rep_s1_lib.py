# -*- coding: utf-8 -*-
"""S1 C1 月營收驚奇 SURGE：獨立複製員自寫函式庫。
只依 FINDINGS.md 第 4 節與 PREREG.json 的文字規格；未讀、未 import 研究員的 stlib.py／run.py／leaderboard_build.py／cslib.py。
import 的現成函式庫只有 o_ext_canslim_xs/replicate/rep_lib.py（前一位獨立複製員寫的：面板載入、跳動點、漲跌停價）。
"""
import sys, os, json, math
import numpy as np
import pandas as pd

sys.path.insert(0, "/home/user/-/research/tw_stock_selection/agents/o_ext_canslim_xs/replicate")
import rep_lib as RL  # noqa: E402

REV = "/home/user/niklavsd/bottleneck-signals/data/processed/tw_revenue.csv"
NAV_START = "2015-07-01"
ODD_START = "2020-10-26"
FEE = 0.001425 * 0.28
TAX = 0.003
tick, limit_up, limit_dn = RL.tick, RL.limit_up, RL.limit_dn


# ------------------------------------------------------------------ 面板
def build(scratch):
    P = RL.load_panel(scratch)
    T, N = P.T, P.N
    P.t0 = int(np.searchsorted(np.array(P.dstr), NAV_START))          # 第一個 ≥ 2015-07-01 的交易日
    P.t_odd = int(np.searchsorted(np.array(P.dstr), ODD_START))
    amt = np.where(P.inbar, P.AMT, 0.0)
    adv = pd.DataFrame(amt).rolling(60, min_periods=60).mean().values
    P.adv60 = np.where(P.inbar & (P.barno >= 60), adv, np.nan)         # 無成交日記 0、滿 60 根
    with np.errstate(invalid="ignore", divide="ignore"):
        P.refe = np.where(P.traded, P.Cf / (1.0 + P.ret), np.nan)      # 與含息日報酬一致的參考價
    r0 = np.where(np.isfinite(P.ret), P.ret, 0.0)
    P.G = np.where(P.inbar, np.cumprod(1.0 + r0, axis=0), np.nan)      # 含息累積指數
    P.cidx = {c: i for i, c in enumerate(P.codes)}
    P.names = json.load(open(os.path.join(scratch, "names.json")))
    P.is_twse = (P.M == 1)
    return P


def didx(P, d):
    return int(np.searchsorted(np.array(P.dstr), d))


def universe(P, s, twse_only=True):
    """訊號日 s 的股池：60 日均成交值 ≥ 1,000 萬、收盤 ≥ 10 元、當日有成交；主欄只上市。"""
    with np.errstate(invalid="ignore"):
        u = P.inbar[s] & P.traded[s] & (P.C[s] >= 10.0) & (P.adv60[s] >= 1e7)
    if twse_only:
        u &= P.is_twse[s]
    return u


# ------------------------------------------------------------------ 月營收
def load_rev(P):
    r = pd.read_csv(REV, dtype={"code": str})
    r["pri"] = (r.market != "sii").astype(int)                          # 同代碼優先上市
    r = r.sort_values(["month", "code", "pri"]).drop_duplicates(["month", "code"], keep="first")
    P.rev_codes_all = set(r.code)
    r = r[r.code.isin(P.cidx)]
    mi = r.month.str[:4].astype(int).values * 12 + r.month.str[5:7].astype(int).values - 1
    mi0 = int(mi.min()); nm = int(mi.max()) - mi0 + 1
    R = np.full((nm, P.N), np.nan)
    R[mi - mi0, r.code.map(P.cidx).values] = r.revenue_k_twd.values
    R[~(R > 0)] = np.nan                                                # 營收 ≤ 0 視為缺值
    return R, mi0


def surge(R, mi0, null2013=True):
    nm, N = R.shape
    D = np.full((nm, N), np.nan)
    D[12:] = R[12:] - R[:-12]
    if null2013:
        a = 2013 * 12 - mi0
        D[a:a + 12] = np.nan                                            # m 在 2013 年的 D 設缺值
    S = np.full((nm, N), np.nan)
    for m in range(24, nm):
        w = D[m - 24:m]
        ok = np.isfinite(w).all(axis=0) & np.isfinite(D[m])
        mu = np.where(ok, np.nan_to_num(w).mean(axis=0), np.nan)
        sd = np.where(ok, np.nan_to_num(w).std(axis=0, ddof=1), np.nan)
        with np.errstate(invalid="ignore", divide="ignore"):
            S[m] = np.where(ok & (sd > 0), (D[m] - mu) / sd, np.nan)
    return S, D


def rev_avail(P, mi):
    """營收月份 mi（y*12+m-1）→ (次月 10 日的日期字串, 期限日索引 j, 訊號日索引 j+1)。
    期限日＝第一個 ≥ 次月 10 日的交易日（非交易日順延）；訊號日＝期限日之後的第一個交易日。"""
    y, mo = divmod(mi, 12); mo += 1
    ny, nmo = (y + 1, 1) if mo == 12 else (y, mo + 1)
    d10 = "%04d-%02d-10" % (ny, nmo)
    j = int(np.searchsorted(np.array(P.dstr), d10, side="left"))
    return d10, j, j + 1


def c1_plans(P, S, mi0, shift=0, twse_only=True, m_first=None):
    """回傳 {訊號日索引: 依 SURGE 由大到小排好的股池內代碼索引陣列}、以及每月資訊。"""
    plans = {}; info = []
    for m in range(S.shape[0]):
        mi = mi0 + m
        d10, j, s = rev_avail(P, mi)
        s = s + shift
        if s >= P.T - 1 or s < P.t0:
            continue
        sv = S[m]
        if not np.isfinite(sv).any():
            continue
        u = universe(P, s, twse_only)
        ok = u & np.isfinite(sv)
        idx = np.where(ok)[0]
        order = idx[np.lexsort((np.array([P.codes[i] for i in idx]), -sv[idx]))]
        plans[s] = order
        info.append(dict(ym="%04d-%02d" % (mi // 12, mi % 12 + 1),
                         d10=d10, deadline=P.dstr[j] if j < P.T else None, signal=P.dstr[s], buy=P.dstr[s + 1],
                         n_univ=int(u.sum()), n_surge=int(ok.sum()), m=m, s=s))
    return plans, info


# ------------------------------------------------------------------ 引擎
def simulate(P, plans, nmax=10, capital=1_000_000.0, cmult=1.0, extra=0.0, block=None, block_span=None,
             hold_days=None, fallback=False, liq_cap=0.05, lot_rule=True):
    """訊號日收盤後決定、隔天開盤成交；先賣後買。
    plans[s]＝訊號日 s 的排序名單（前 nmax 名是目標名單）。
    cmult：手續費率、最低手續費、稅、半跳動點滑價的倍數；extra：每邊額外滑價比例。
    block：{(代號, 買進日)} 封鎖該筆進場；block_span：{代號: [(起日索引, 迄日索引)]} 期間內不得買進。
    hold_days：固定持有 N 個交易日（進場後第 N 個交易日開盤賣）；None＝掉出名單才賣。
    fallback：開盤漲停時遞補下一名（主策略用）。"""
    T = P.T; t0 = P.t0; codes = P.codes
    fee = FEE * cmult; minfee = 20.0 * cmult; tax = TAX * cmult
    cash = capital; pos = {}
    nav = np.full(T, np.nan); nav[t0] = capital
    npos = np.zeros(T); cashw = np.ones(T)
    trades = []; buys = []
    cnt = dict(cand=0, cand_pre_odd=0, buy=0, limit_up=0, no_open=0, lot_skip=0, no_cash=0, no_slot=0, liq_capped=0,
               blocked=0, sell_defer=0, sell_split=0, delist=0, fee=0.0, tax=0.0, slip=0.0, turnover_buy=0.0)

    def fees(val, sh):
        lf = ((sh // 1000) * 1000) / sh if sh > 0 else 1.0
        f = 0.0
        for fr in (lf, 1.0 - lf):
            if fr > 1e-12:
                f += max(minfee, fee * val * fr)                         # 整股與零股各算一筆
        return f

    for t in range(t0 + 1, T):
        # ---- 開盤賣出（含下市結清）
        for i in sorted(pos, key=lambda j: codes[j]):
            p = pos[i]
            if t > P.last[i]:
                cash += p["val"]; cnt["delist"] += 1
                p["recv"] += p["val"]; p["done"] = True; p["delist"] = 1
                continue
            if not p["pend"]:
                continue
            o = P.O[t, i]; ref = P.refe[t, i]
            if not (o == o) or not (ref == ref):
                cnt["sell_defer"] += 1; continue
            if o <= limit_dn(P.ref[t, i]) + 1e-9:
                cnt["sell_defer"] += 1; continue
            gross_full = p["val"] * (o / ref)
            frac = 1.0
            if liq_cap is not None and gross_full > liq_cap * P.AMT[t, i]:
                frac = liq_cap * P.AMT[t, i] / gross_full; cnt["sell_split"] += 1
            gross = gross_full * frac
            g2 = gross * ((o - tick(o) / 2 * cmult) / o) * (1.0 - extra)
            fe = fees(g2, p["sh"]); tx = tax * g2
            cash += g2 - fe - tx
            p["recv"] += g2 - fe - tx
            cnt["fee"] += fe; cnt["tax"] += tx; cnt["slip"] += gross - g2
            if frac >= 1.0:
                p["done"] = True
            else:
                p["val"] *= (1.0 - frac)
        for i in [j for j in pos if pos[j].get("done")]:
            p = pos.pop(i)
            trades.append(dict(code=codes[i], i=i, tb=p["tb"], ts=t, buy_date=P.dstr[p["tb"]], sell_date=P.dstr[t],
                               cost=p["cost"], proceeds=p["recv"], pnl=p["recv"] - p["cost"],
                               ret=p["recv"] / p["cost"] - 1.0, hold=t - p["tb"], delist=p.get("delist", 0),
                               buy_open=p["o"], part=p["part"], gap=p["gap"]))
        # ---- 開盤買進
        if (t - 1) in plans:
            ranked = plans[t - 1]
            eq = cash
            for i, p in pos.items():
                o = P.O[t, i]; ref = P.refe[t, i]
                eq += p["val"] * (o / ref) if (o == o and ref == ref) else p["val"]
            pool = ranked if fallback else ranked[:nmax]
            n_fail = 0                                                    # 非漲停原因買不到的不遞補
            for rk, i in enumerate(pool):
                i = int(i)
                if i in pos:
                    continue
                if fallback and (len(pos) + n_fail >= nmax):
                    break
                if (not fallback) and rk >= nmax:
                    break
                cnt["cand"] += 1
                if t < P.t_odd: cnt["cand_pre_odd"] += 1
                if len(pos) >= nmax:
                    cnt["no_slot"] += 1; continue
                if block and (codes[i], P.dstr[t]) in block:
                    cnt["blocked"] += 1; continue
                if block_span and any(a <= t <= b for a, b in block_span.get(codes[i], ())):
                    cnt["blocked"] += 1; continue
                o = P.O[t, i]
                if not (o == o):
                    cnt["no_open"] += 1; n_fail += 1; continue
                if o >= limit_up(P.ref[t, i]) - 1e-9:
                    cnt["limit_up"] += 1
                    continue                                              # fallback=True 時迴圈自然往下一名
                budget = min(eq / nmax, cash)
                if liq_cap is not None and budget > liq_cap * P.AMT[t, i]:
                    budget = liq_cap * P.AMT[t, i]; cnt["liq_capped"] += 1
                px = (o + tick(o) / 2 * cmult) * (1.0 + extra)
                lot_only = lot_rule and t < P.t_odd
                unit = 1000 if lot_only else 1
                sh = int(budget / (px * (1 + fee)) / unit) * unit
                while sh > 0 and sh * px + fees(sh * px, sh) > budget + 1e-9:
                    sh -= unit
                if sh <= 0:
                    cnt["lot_skip" if lot_only else "no_cash"] += 1
                    n_fail += 1
                    continue
                fe = fees(sh * px, sh)
                c = sh * px + fe
                cash -= c
                cnt["buy"] += 1; cnt["fee"] += fe; cnt["slip"] += sh * (px - o); cnt["turnover_buy"] += sh * o
                part = sh * o / P.AMT[t, i] if P.AMT[t, i] > 0 else np.nan
                gap = o / P.refe[t, i] - 1.0
                pos[i] = dict(tb=t, sh=sh, val=sh * o, cost=c, recv=0.0, pend=False, new=True, o=o, part=part, gap=gap)
                buys.append(dict(code=codes[i], i=i, tb=t, buy_date=P.dstr[t], sig=t - 1, rank=rk, open=o, sh=sh,
                                 amt=sh * o, part=part, gap=gap, adv60=P.adv60[t - 1, i], cost=c))
        # ---- 收盤：市值、淨值、出場旗標
        v = 0.0
        for i, p in pos.items():
            if p.pop("new", False):
                p["val"] = p["sh"] * P.Cf[t, i]
            else:
                r = P.ret[t, i]
                if r == r: p["val"] *= (1.0 + r)
            v += p["val"]
        nav[t] = cash + v; npos[t] = len(pos); cashw[t] = cash / nav[t]
        if hold_days is not None:
            for i, p in pos.items():
                if (t + 1) - p["tb"] >= hold_days:
                    p["pend"] = True
        elif t in plans:
            tgt = set(int(x) for x in plans[t][:nmax])
            for i, p in pos.items():
                p["pend"] = i not in tgt
    tr = pd.DataFrame(trades)
    return dict(nav=nav, trades=tr, buys=pd.DataFrame(buys), cnt=cnt, npos=npos, cashw=cashw,
                open_pos=[(codes[i], P.dstr[p["tb"]], p["cost"], p["val"]) for i, p in pos.items()])


# ------------------------------------------------------------------ 統計
def ann(P, x, i0, i1):
    yrs = (P.dates[i1] - P.dates[i0]).days / 365.25
    return float((x[i1] / x[i0]) ** (1.0 / yrs) - 1.0)


def maxdd(P, x, i0, i1):
    v = np.asarray(x[i0:i1 + 1], float)
    pk = np.maximum.accumulate(v)
    dd = v / pk - 1.0
    k = int(dd.argmin()); kp = int(v[:k + 1].argmax())
    return float(dd[k]), P.dstr[i0 + kp], P.dstr[i0 + k]


def yearly(P, x, i0=None):
    i0 = P.t0 if i0 is None else i0
    s = pd.Series(np.asarray(x[i0:], float), index=P.dates[i0:])
    ye = s.groupby(s.index.year).last()
    out = {}; prev = s.iloc[0]
    for y, v in ye.items():
        out[int(y)] = float(v / prev - 1.0); prev = v
    return out


def monthly(P, x, i0):
    s = pd.Series(np.asarray(x[i0:], float), index=P.dates[i0:])
    me = s.groupby(s.index.to_period("M")).last()
    r = me / me.shift(1) - 1.0
    r.iloc[0] = me.iloc[0] / s.iloc[0] - 1.0
    if s.index[0].to_period("M") == me.index[0] and (s.index.to_period("M") == me.index[0]).sum() == 1:
        r = r.iloc[1:]
    return r


def trade_stats(tr):
    if len(tr) == 0:
        return dict(n=0)
    w = tr.ret > 0
    return dict(n=int(len(tr)), win=float(w.mean()), avg_win=float(tr.ret[w].mean()),
                avg_loss=float(-tr.ret[~w].mean()), exp=float(tr.ret.mean()), hold_mean=float(tr.hold.mean()),
                hold_med=float(tr.hold.median()), n_delist=int(tr.delist.sum()))


def summary(P, res, e50=None):
    nav = res["nav"]; T = P.T
    i19 = didx(P, "2018-12-28"); ih = didx(P, "2021-02-19"); t0 = P.t0; e = T - 1
    out = dict(ann_2019=ann(P, nav, i19, e), ann_full=ann(P, nav, t0, e), ann_h1=ann(P, nav, t0, ih),
               ann_h2=ann(P, nav, ih, e), mdd_2019=maxdd(P, nav, i19, e), mdd_full=maxdd(P, nav, t0, e),
               yearly=yearly(P, nav), final_nav=float(nav[e]), trades=trade_stats(res["trades"]),
               trades_2019=trade_stats(res["trades"][res["trades"].tb > i19]) if len(res["trades"]) else {},
               cnt={k: (float(v) if isinstance(v, float) else int(v)) for k, v in res["cnt"].items()},
               avg_npos=float(res["npos"][t0 + 1:].mean()), avg_cashw=float(res["cashw"][t0 + 1:].mean()),
               n_open=len(res["open_pos"]))
    return out


def bench(P):
    e = P.e50.values; T = P.T
    i19 = didx(P, "2018-12-28"); ih = didx(P, "2021-02-19"); t0 = P.t0; x = T - 1
    return dict(ann_2019=ann(P, e, i19, x), ann_full=ann(P, e, t0, x), ann_h1=ann(P, e, t0, ih), ann_h2=ann(P, e, ih, x),
                mdd_2019=maxdd(P, e, i19, x), mdd_full=maxdd(P, e, t0, x), yearly=yearly(P, e))
