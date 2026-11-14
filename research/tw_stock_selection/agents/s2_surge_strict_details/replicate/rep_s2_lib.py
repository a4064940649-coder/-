# -*- coding: utf-8 -*-
"""S2（月營收驚奇 SURGE 嚴成本、D1 加季線）獨立複製：自寫函式庫。
只依 s2_surge_strict_details/FINDINGS.md 第 2 節與 PREREG.json 的文字規格。
未讀、未 import 研究員的 s2lib.py／run.py／posthoc.py／stlib.py／cslib.py。
import 的現成程式：s1_shortterm_leaderboard/replicate/rep_s1_lib.py（S1 獨立複製員寫的面板、SURGE、統計函式）。
本檔自己寫的部分：停牌排除的股池、D1 均線條件、名單、嚴成本（加法口徑）引擎、整段持有期禁買、股池等權、配對差。
"""
import sys, os
import numpy as np
import pandas as pd

sys.path.insert(0, "/home/user/-/research/tw_stock_selection/agents/s1_shortterm_leaderboard/replicate")
import rep_s1_lib as L  # noqa: E402

FEE, TAX = L.FEE, L.TAX
tick, limit_up, limit_dn = L.tick, L.limit_up, L.limit_dn


def build(scratch):
    P = L.build(scratch)
    z = np.load(os.path.join(scratch, "panel.npz"))
    pres = z["P"].astype(bool)
    del z
    # S2 規格：近 60 根都要有資料列，有缺列（停牌）的排除
    P.full60 = pd.DataFrame(pres.astype(np.float32)).rolling(60, min_periods=60).sum().values >= 60
    P.ma = {}
    return P


def ma(P, n):
    """還原收盤（含息累積指數 G）的 n 日均線，滿 n 根才有值。"""
    if n not in P.ma:
        P.ma[n] = pd.DataFrame(P.G).rolling(n, min_periods=n).mean().values
    return P.ma[n]


def universe(P, s):
    with np.errstate(invalid="ignore"):
        u = P.inbar[s] & P.traded[s] & (P.C[s] >= 10.0) & (P.adv60[s] >= 1e7) & P.full60[s] & P.is_twse[s]
    return u


def make_plans(P, S, mi0, ma_len=None, shift=0, min_n=50):
    """{訊號日列號: 依 SURGE 由大到小排好的代碼索引}；ma_len 給定時加「還原收盤 > n 日均線」。"""
    plans = {}; info = []
    for m in range(S.shape[0]):
        d10, j, s = L.rev_avail(P, mi0 + m)
        s += shift
        if s >= P.T - 1 or s < P.t0:
            continue
        sv = S[m]
        u = universe(P, s)
        ok = u & np.isfinite(sv)
        n_surge = int(ok.sum())
        if n_surge < min_n:                                   # 股池內 SURGE 有值不足 50 檔不產生訊號
            continue
        if ma_len is not None:
            mv = ma(P, ma_len)[s]
            with np.errstate(invalid="ignore"):
                ok = ok & (P.G[s] > mv)
        idx = np.where(ok)[0]
        order = idx[np.lexsort((np.array([P.codes[i] for i in idx]), -sv[idx]))]
        plans[s] = order
        info.append(dict(ym="%04d-%02d" % ((mi0 + m) // 12, (mi0 + m) % 12 + 1), signal=P.dstr[s], s=s, m=m,
                         n_univ=int(u.sum()), n_surge=n_surge, n_pass=int(ok.sum())))
    return plans, info


def simulate(P, plans, nmax=10, capital=1_000_000.0, cmult=1.0, extra=0.0, block_span=None, liq_cap=0.05):
    """訊號日收盤後決定、隔日開盤成交；先賣後買；開盤漲停不買不遞補；月中不出場。
    成本：手續費率 FEE×cmult（整張、零股各一筆，最低 20×cmult 元）、賣方稅 TAX×cmult、
    買進價＝開盤×(1＋半跳動點比率×cmult＋extra)、賣出價＝開盤×(1−半跳動點比率×cmult−extra)。
    成本加倍＝cmult 2、extra 0.006。block_span：{代碼: [(起列號, 迄列號)]} 期間內任何一天不得買進。"""
    T, t0, codes = P.T, P.t0, P.codes
    fee = FEE * cmult; minfee = 20.0 * cmult; tax = TAX * cmult
    cash = capital; pos = {}
    nav = np.full(T, np.nan); nav[t0] = capital
    npos = np.zeros(T)
    trades = []
    cnt = dict(cand=0, buy=0, limit_up=0, no_open=0, lot_skip=0, no_cash=0, blocked=0, liq_capped=0,
               sell_defer=0, sell_split=0, delist=0, cost_total=0.0, buy_amt=0.0)

    def fees(val, sh):
        lot = (sh // 1000) * 1000
        f = 0.0
        if lot > 0:
            f += max(minfee, fee * val * lot / sh)
        if sh - lot > 0:
            f += max(minfee, fee * val * (sh - lot) / sh)
        return f

    for t in range(t0 + 1, T):
        # ---------- 開盤賣
        closed = []
        for i in sorted(pos, key=lambda j: codes[j]):
            p = pos[i]
            if t > P.last[i]:                                  # 自日 K 消失：以最後收盤結清
                cash += p["val"]; p["recv"] += p["val"]; p["delist"] = 1; cnt["delist"] += 1
                closed.append(i); continue
            if not p["pend"]:
                continue
            o = P.O[t, i]; rf = P.refe[t, i]
            if not (o == o) or not (rf == rf) or o <= limit_dn(P.ref[t, i]) + 1e-9:
                cnt["sell_defer"] += 1; continue
            gross_full = p["val"] * (o / rf)
            frac = 1.0
            if liq_cap is not None and gross_full > liq_cap * P.AMT[t, i]:
                frac = liq_cap * P.AMT[t, i] / gross_full; cnt["sell_split"] += 1
            gross = gross_full * frac
            net = gross * (1.0 - tick(o) / 2.0 / o * cmult - extra)
            fe = fees(net, p["sh"]); tx = tax * net
            cash += net - fe - tx; p["recv"] += net - fe - tx
            cnt["cost_total"] += (gross - net) + fe + tx
            if frac >= 1.0:
                closed.append(i)
            else:
                p["val"] *= (1.0 - frac)
        for i in closed:
            p = pos.pop(i)
            trades.append(dict(code=codes[i], tb=p["tb"], ts=t, buy_date=P.dstr[p["tb"]], sell_date=P.dstr[t],
                               cost=p["cost"], proceeds=p["recv"], pnl=p["recv"] - p["cost"],
                               ret=p["recv"] / p["cost"] - 1.0, hold=t - p["tb"], delist=p.get("delist", 0),
                               sh=p["sh0"], rank=p["rank"]))
        # ---------- 開盤買
        if (t - 1) in plans:
            eq = cash
            for i, p in pos.items():
                o = P.O[t, i]; rf = P.refe[t, i]
                eq += p["val"] * (o / rf) if (o == o and rf == rf) else p["val"]
            sold_today = set(closed)
            for rk, i in enumerate(plans[t - 1][:nmax]):
                i = int(i)
                if i in pos or i in sold_today:                # 仍持有不重複買；當天賣的當天不買回
                    continue
                cnt["cand"] += 1
                if len(pos) >= nmax:
                    continue
                if block_span and any(a <= t <= b for a, b in block_span.get(codes[i], ())):
                    cnt["blocked"] += 1; continue
                o = P.O[t, i]
                if not (o == o):
                    cnt["no_open"] += 1; continue
                if o >= limit_up(P.ref[t, i]) - 1e-9:
                    cnt["limit_up"] += 1; continue
                budget = min(eq / nmax, cash)
                if liq_cap is not None and budget > liq_cap * P.AMT[t, i]:
                    budget = liq_cap * P.AMT[t, i]; cnt["liq_capped"] += 1
                px = o * (1.0 + tick(o) / 2.0 / o * cmult + extra)
                lot_only = t < P.t_odd
                unit = 1000 if lot_only else 1
                sh = int(budget / (px * (1 + fee)) / unit) * unit
                while sh > 0 and sh * px + fees(sh * px, sh) > budget + 1e-9:
                    sh -= unit
                if sh <= 0:
                    cnt["lot_skip" if lot_only else "no_cash"] += 1; continue
                fe = fees(sh * px, sh)
                c = sh * px + fe
                cash -= c
                cnt["buy"] += 1; cnt["cost_total"] += sh * (px - o) + fe; cnt["buy_amt"] += sh * o
                pos[i] = dict(tb=t, sh=sh, sh0=sh, val=sh * o, cost=c, recv=0.0, pend=False, new=True, rank=rk)
        # ---------- 收盤
        v = 0.0
        for i, p in pos.items():
            if p.pop("new", False):
                p["val"] = p["sh"] * P.Cf[t, i]
            else:
                r = P.ret[t, i]
                if r == r:
                    p["val"] *= (1.0 + r)
            v += p["val"]
        nav[t] = cash + v; npos[t] = len(pos)
        if t in plans:
            tgt = set(int(x) for x in plans[t][:nmax])
            for i, p in pos.items():
                p["pend"] = i not in tgt
    tr = pd.DataFrame(trades)
    return dict(nav=nav, trades=tr, cnt=cnt, npos=npos,
                open_pos=[(codes[i], P.dstr[p["tb"]], float(p["cost"]), float(p["val"])) for i, p in pos.items()])


def stats(P, res):
    nav = res["nav"]; e = P.T - 1
    i19 = L.didx(P, "2018-12-28"); i24 = L.didx(P, "2023-12-29")
    assert P.dstr[i19] == "2018-12-28" and P.dstr[i24] == "2023-12-29"
    md = L.maxdd(P, nav, P.t0, e)
    tr = res["trades"]
    return dict(ann_2019=L.ann(P, nav, i19, e), ann_full=L.ann(P, nav, P.t0, e), ann_2024=L.ann(P, nav, i24, e),
                ann_2019_2023=L.ann(P, nav, i19, i24), mdd_full=md[0], mdd_peak=md[1], mdd_trough=md[2],
                n_trades=int(len(tr)), n_trades_2019=int((tr.tb > i19).sum()) if len(tr) else 0,
                exp=float(tr.ret.mean()) if len(tr) else None, win=float((tr.ret > 0).mean()) if len(tr) else None,
                yearly=L.yearly(P, nav), final_nav=float(nav[e]), avg_npos=float(res["npos"][P.t0 + 1:].mean()),
                n_open=len(res["open_pos"]),
                cnt={k: (float(v) if isinstance(v, float) else int(v)) for k, v in res["cnt"].items()})


def span_of(tr, k):
    """已平倉交易中損益金額最大的 k 筆 → {代碼: [(進場列號, 出場列號)]}。"""
    top = tr.sort_values("pnl", ascending=False).head(k)
    sp = {}
    for c, a, b in zip(top.code, top.tb, top.ts):
        sp.setdefault(c, []).append((int(a), int(b)))
    return sp, [dict(code=c, buy=a, sell=b, pnl=float(d), ret=float(r)) for c, a, b, d, r in
                zip(top.code, top.buy_date, top.sell_date, top.pnl, top.ret)]


def month_ret(P, nav):
    s = pd.Series(np.asarray(nav[P.t0:], float), index=P.dates[P.t0:])
    me = s.groupby(s.index.to_period("M")).last()
    return (me / me.shift(1) - 1.0).dropna()


def paired(P, nav_a, nav_b, start):
    d = (month_ret(P, nav_a) - month_ret(P, nav_b))
    d = d[d.index >= pd.Period(start, "M")]
    n = len(d); mu = float(d.mean()); sd = float(d.std(ddof=1))
    return dict(n=n, mean=mu, t=mu / (sd / np.sqrt(n)), first=str(d.index[0]), last=str(d.index[-1]))


def eqw_nav(P, plans, hold=True):
    """股池等權：每個訊號日以當日只上市股池全部股票等權，隔日起收盤對收盤含息報酬，持有到下一個訊號日，不扣成本。
    hold=True：期初等權後不再平衡（買進持有）；False：每日再平衡成等權。"""
    sig = sorted(plans)
    nav = np.ones(P.T)
    for k, s in enumerate(sig):
        e = sig[k + 1] if k + 1 < len(sig) else P.T - 1
        mem = np.where(universe(P, s))[0]
        r = np.nan_to_num(P.ret[s + 1:e + 1][:, mem], nan=0.0)
        if hold:
            path = np.cumprod(1.0 + r, axis=0).mean(axis=1)
        else:
            path = np.cumprod(1.0 + r.mean(axis=1))
        nav[s + 1:e + 1] = nav[s] * path
        if k + 1 == len(sig):
            nav[e + 1:] = nav[e]
    return nav
