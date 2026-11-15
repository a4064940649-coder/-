# -*- coding: utf-8 -*-
"""S3（D1 加「前 20 日漲幅門檻」與「寬出場」）獨立複製：自寫函式庫。
只依 s3_winners_momentum/FINDINGS.md 第 2 節與 PREREG.json 的文字規格。
未讀、未 import 研究員的 s3lib.py／run.py／posthoc.py／s2lib.py／stlib.py／cslib.py。
import 的現成程式：s2_surge_strict_details/replicate/rep_s2_lib.py（S2 獨立複製員寫的股池、均線、統計）
與它依賴的 rep_s1_lib.py。
本檔自己寫的部分：L_MA／L_MOM 兩份名單、前 20 日漲幅、買進名單與續抱名單分開的引擎、股池排除（創新板）。
"""
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, "/home/user/-/research/tw_stock_selection/agents/s2_surge_strict_details/replicate")
import rep_s2_lib as R  # noqa: E402
L = R.L
FEE, TAX = R.FEE, R.TAX
tick, limit_up, limit_dn = R.tick, R.limit_up, R.limit_dn

STRICT = dict(cmult=1.0, extra=0.003)
X2 = dict(cmult=2.0, extra=0.006)


def make_lists(P, S, mi0, shift=0, excl=None, min_n=50, ma_len=60, ret_len=20):
    """每個訊號日的 L_MA（依 SURGE 由大到小、同分依代號）與前 20 日漲幅。
    excl：函式 s -> bool[N]，True 的股票整個排除出股池（審查 A 用）。
    「股池內 SURGE 有值不足 50 檔不產生訊號」用未位移的訊號日判斷；其餘（股池、均線、漲幅）在位移後那天判斷。
    回傳 {s: dict(lma=代碼索引陣列, r20=長度 N 陣列, m=營收月列號)}。"""
    out = {}
    codes = np.array(P.codes)
    mav = R.ma(P, ma_len)
    for m in range(S.shape[0]):
        d10, j, s0 = L.rev_avail(P, mi0 + m)
        s = s0 + shift
        if s >= P.T - 1 or s0 < P.t0 or s0 >= P.T - 1:
            continue
        sv = S[m]
        u0 = R.universe(P, s0)
        if excl is not None:
            u0 = u0 & ~excl(s0)
        if int((u0 & np.isfinite(sv)).sum()) < min_n:
            continue
        u = R.universe(P, s)
        if excl is not None:
            u = u & ~excl(s)
        with np.errstate(invalid="ignore", divide="ignore"):
            ok = u & np.isfinite(sv) & (P.G[s] > mav[s])
            r20 = P.G[s] / P.G[s - ret_len] - 1.0
        idx = np.where(ok)[0]
        order = idx[np.lexsort((codes[idx], -sv[idx]))]
        out[s] = dict(lma=order, r20=r20, m=m, n_surge=int((u & np.isfinite(sv)).sum()))
    return out


def lmom(lst, x):
    r = lst["r20"]
    with np.errstate(invalid="ignore"):
        return np.array([i for i in lst["lma"] if r[i] > x], dtype=int)


def cell_plans(lists, cell, x=0.10, nmax=10):
    """回傳 (buy, keep)：buy[s]＝買進名單（最多 nmax 檔、依 SURGE 排）；keep[s]＝續抱名單（集合），持股不在裡面就賣。"""
    buy = {}; keep = {}
    for s, q in lists.items():
        a = q["lma"]
        if cell == "D1":
            b = a[:nmax]; k = a[:nmax]
        elif cell == "E1":
            b = a[:nmax]; k = a[:20]
        elif cell == "E2":
            b = lmom(q, x)[:nmax]; k = b
        elif cell == "E3":
            b = lmom(q, x)[:nmax]; k = a[:20]
        elif cell == "C10":                       # 事後診斷：只買 L_MA 前 10 名中已在漲的、不遞補；出場同 E3
            top = a[:nmax]
            with np.errstate(invalid="ignore"):
                b = np.array([i for i in top if q["r20"][i] > x], dtype=int)
            k = a[:20]
        else:
            raise ValueError(cell)
        buy[s] = np.asarray(b, dtype=int); keep[s] = set(int(i) for i in k)
    return buy, keep


def simulate(P, buy, keep, nmax=10, capital=1_000_000.0, cmult=1.0, extra=0.0, block_span=None, liq_cap=0.05):
    """訊號日收盤後決定、隔日開盤成交；先賣後買；當天賣的當天不買回；開盤 ≥ 漲停價不買、不遞補；
    開盤 ≤ 跌停價或無開盤價順延賣；自日 K 消失以最後收盤結清；每筆目標＝當日淨值 ÷ nmax 且不超過現金；
    單筆 ≤ 當日成交值 5%；2020-10-26 前只能買整張。
    成本：手續費 FEE×cmult（整張、零股各一筆，最低 20×cmult）、賣方稅 TAX×cmult、每邊半跳動點×cmult＋extra。
    buy[s]：買進名單；keep[s]：續抱集合。block_span：{代碼: [(起列號, 迄列號)]} 期間內任何一天不得買進。"""
    T, t0, codes = P.T, P.t0, P.codes
    fee = FEE * cmult; minfee = 20.0 * cmult; tax = TAX * cmult
    cash = capital; pos = {}
    nav = np.full(T, np.nan); nav[t0] = capital
    npos = np.zeros(T); cashw = np.ones(T)
    trades = []
    cnt = dict(cand=0, buy=0, limit_up=0, no_open=0, lot_skip=0, no_cash=0, no_slot=0, blocked=0, liq_capped=0,
               sell_defer=0, sell_split=0, delist=0, cost_total=0.0, buy_amt=0.0, sell_amt=0.0,
               sold_in_buylist=0)

    def fees(val, sh):
        lot = (sh // 1000) * 1000
        f = 0.0
        if lot > 0:
            f += max(minfee, fee * val * lot / sh)
        if sh - lot > 0:
            f += max(minfee, fee * val * (sh - lot) / sh)
        return f

    for t in range(t0 + 1, T):
        closed = []
        for i in sorted(pos, key=lambda j: codes[j]):
            p = pos[i]
            if t > P.last[i]:
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
            cnt["cost_total"] += (gross - net) + fe + tx; cnt["sell_amt"] += gross
            if frac >= 1.0:
                closed.append(i)
            else:
                p["val"] *= (1.0 - frac)
        for i in closed:
            p = pos.pop(i)
            trades.append(dict(code=codes[i], i=i, tb=p["tb"], ts=t, buy_date=P.dstr[p["tb"]], sell_date=P.dstr[t],
                               cost=p["cost"], proceeds=p["recv"], pnl=p["recv"] - p["cost"],
                               ret=p["recv"] / p["cost"] - 1.0, hold=t - p["tb"], delist=p.get("delist", 0),
                               sh=p["sh0"], rank=p["rank"]))
        if (t - 1) in buy:
            eq = cash
            for i, p in pos.items():
                o = P.O[t, i]; rf = P.refe[t, i]
                eq += p["val"] * (o / rf) if (o == o and rf == rf) else p["val"]
            sold_today = set(closed)
            for rk, i in enumerate(buy[t - 1][:nmax]):
                i = int(i)
                if i in sold_today:
                    cnt["sold_in_buylist"] += 1
                if i in pos or i in sold_today:
                    continue
                cnt["cand"] += 1
                if len(pos) >= nmax:
                    cnt["no_slot"] += 1; continue
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
        v = 0.0
        for i, p in pos.items():
            if p.pop("new", False):
                p["val"] = p["sh"] * P.Cf[t, i]
            else:
                r = P.ret[t, i]
                if r == r:
                    p["val"] *= (1.0 + r)
            v += p["val"]
        nav[t] = cash + v; npos[t] = len(pos); cashw[t] = cash / nav[t]
        if t in keep:
            kp = keep[t]
            for i, p in pos.items():
                p["pend"] = i not in kp
    tr = pd.DataFrame(trades)
    return dict(nav=nav, trades=tr, cnt=cnt, npos=npos, cashw=cashw,
                open_pos=[(codes[i], P.dstr[p["tb"]], float(p["cost"]), float(p["val"])) for i, p in pos.items()])


def stats(P, res):
    s = R.stats(P, res)
    tr = res["trades"]
    ih = L.didx(P, "2021-02-19"); e = P.T - 1
    s["ann_h1"] = L.ann(P, res["nav"], P.t0, ih); s["ann_h2"] = L.ann(P, res["nav"], ih, e)
    if len(tr):
        f = int(tr.tb.min())
        s["first_buy"] = P.dstr[f]
        s["avg_npos_from_first_buy"] = float(res["npos"][f:].mean())
        s["avg_cashw_from_first_buy"] = float(res["cashw"][f:].mean())
        s["median_ret"] = float(tr.ret.median())
        tot = float(tr.pnl.sum()); k = tr.pnl.idxmax()
        s["top_trade"] = dict(code=tr.code[k], buy=tr.buy_date[k], sell=tr.sell_date[k], ret=float(tr.ret[k]),
                              pnl=float(tr.pnl[k]), cost=float(tr.cost[k]), share_of_closed_pnl=float(tr.pnl[k] / tot))
        s["max_ret_closed"] = float(tr.ret.max())
    s["max_ret_open"] = max([v / c - 1.0 for _, _, c, v in res["open_pos"]], default=None)
    return s


def brief(s):
    return {k: s.get(k) for k in ("ann_2019", "ann_full", "ann_2024", "mdd_full", "n_trades", "ann_h1", "ann_h2",
                                  "avg_npos_from_first_buy")}


def lma_rank(lists, tr):
    """每筆交易進場當下（訊號日＝進場列號−1）在 L_MA 的名次（1 起算）與前 20 日漲幅。"""
    rk = []; r20 = []
    for i, tb in zip(tr.i, tr.tb):
        q = lists[int(tb) - 1]
        w = np.where(q["lma"] == int(i))[0]
        rk.append(int(w[0]) + 1 if len(w) else -1)
        r20.append(float(q["r20"][int(i)]))
    return np.array(rk), np.array(r20)


def grp(tr, mask):
    x = tr.ret[mask]
    if len(x) == 0:
        return dict(n=0)
    return dict(n=int(len(x)), mean=float(x.mean()), median=float(x.median()), win=float((x > 0).mean()),
                lt_m15=float((x < -0.15).mean()))
