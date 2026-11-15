# -*- coding: utf-8 -*-
"""S5（0050 底倉＋「L_MA 前 10 名已在漲」衛星）獨立複製：自寫函式庫。
只依 s5_core_satellite/FINDINGS.md 第 2 節與 PREREG.json 的文字規格。
未讀、未 import 研究員的 s5lib.py／prep.py／run.py／posthoc.py，也未 import s4lib／s3lib／s2lib／stlib／cslib。
import 的現成程式：s4_e3_purity/replicate/rep_s4_lib.py（S4 獨立複製員：創新板遮罩、最小買進金額、空位 0050）
與它依賴的 rep_s3_lib／rep_s2_lib／rep_s1_lib。
本檔自己寫的部分：simulate_cs ＝ rep_s4_lib.simulate(idle='e50') 的買賣邏輯，加上
  (a) 底倉金額 B 的簿記（跟 0050 一樣漲跌）、衛星資金＝淨值 − B、每檔目標＝max(開盤估淨值 − B, 0) ÷ 10；
  (b) B 被夾住：買完後與收盤後 0 ≤ B ≤ 現金＋0050；
  (c) 換股執行日開盤、賣完之後、買之前的拉回（門檻 band，嚴格大於）；衛星偏高且閒置不足時個股依市值等比例減碼；
  (d) 審查用的兩個變體：mode='fixed30'（每個執行日都把底倉拉回 30%、每檔＝淨值×70%÷10）、
      div_mode='list5'（每檔＝衛星資金 ÷ max(當月名單檔數, 5)）。
"""
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, "/home/user/-/research/tw_stock_selection/agents/s4_e3_purity/replicate")
import rep_s4_lib as S  # noqa: E402
Q, R, L = S.Q, S.R, S.L
FEE, TAX, TAX50 = S.FEE, S.TAX, S.TAX50
tick, limit_up, limit_dn = S.tick, S.limit_up, S.limit_dn


def simulate_cs(P, buy, keep, sat_w=0.7, band=0.10, nmax=10, capital=1_000_000.0, cmult=1.0, extra=0.0,
                block_span=None, liq_cap=0.05, min_frac=0.5, e50_min=20_000.0, mode="fixedB", div_mode="ten"):
    """mode='fixedB'：研究員規格（B 股數固定、偏離 > band 才拉回）。
    mode='fixed30'：審查 A，每個換股執行日都把 B 拉回 (1−sat_w)×淨值（不足時減碼個股），每檔目標＝sat_w×淨值÷nmax。
    div_mode='ten'：每檔目標＝衛星資金÷nmax；'list5'：衛星資金÷max(len(買進名單),5)（審查 B）。"""
    T, t0, codes = P.T, P.t0, P.codes
    fee = FEE * cmult; minfee = 20.0 * cmult; tax = TAX * cmult; tax50 = TAX50 * cmult
    cash = capital; pos = {}; v50 = 0.0
    B = (1.0 - sat_w) * capital
    nav = np.full(T, np.nan); nav[t0] = capital
    npos = np.zeros(T); cashw = np.ones(T); w50 = np.zeros(T); stkw = np.zeros(T); Bs = np.full(T, np.nan)
    Bs[t0] = B
    trades = []; reb = []
    clamp_day = np.zeros(T, bool)
    cnt = dict(cand=0, buy=0, limit_up=0, no_open=0, lot_skip=0, no_cash=0, no_slot=0, blocked=0, liq_capped=0,
               blk_min=0, blk_min_or_zero=0, sell_defer=0, sell_split=0, delist=0, cost_total=0.0, buy_amt=0.0,
               sell_amt=0.0, sold_in_buylist=0, e50_buy=0.0, e50_sell=0.0, e50_cost=0.0,
               reb_up=0, reb_down=0, reb_reduce_events=0, reb_reduce_legs=0, reb_reduce_amt=0.0)

    def fees(val, sh):
        lot = (sh // 1000) * 1000
        f = 0.0
        if lot > 0:
            f += max(minfee, fee * val * lot / sh)
        if sh - lot > 0:
            f += max(minfee, fee * val * (sh - lot) / sh)
        return f

    def e50_hs(t):
        o = P.e50raw_o[t]
        return S.e50_tick(o) / 2.0 / o * cmult

    def e50_liq(t):
        if v50 <= 0:
            return 0.0
        g = v50 * (1.0 - e50_hs(t))
        return max(0.0, g - max(minfee, fee * g) - tax50 * g)

    def e50_sell_for(t, need):
        nonlocal v50
        if v50 <= 0 or need <= 0:
            return 0.0
        hs = e50_hs(t)
        k = 1.0 - hs
        g = need / (k * (1.0 - fee - tax50))
        if fee * g * k < minfee:
            g = (need + minfee) / (k * (1.0 - tax50))
        if g >= v50 * 0.999999:
            got = e50_liq(t); gross = v50; v50 = 0.0
        else:
            gross = g; v50 -= g
            n = g * k; got = n - max(minfee, fee * n) - tax50 * n
        cnt["e50_sell"] += gross; cnt["e50_cost"] += gross - got
        return got

    def clamp(t):
        nonlocal B
        cap = max(cash, 0.0) + v50
        if B > cap + 1e-6:
            B = cap; clamp_day[t] = True
        if B < 0:
            B = 0.0

    for t in range(t0 + 1, T):
        if v50 > 0:
            v50 *= P.e50o[t] / P.e50c[t - 1]
        B *= P.e50o[t] / P.e50c[t - 1]
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
                p["val"] *= (1.0 - frac); p["shc"] = int(p.get("shc", p["sh"]) * (1.0 - frac))
        for i in closed:
            p = pos.pop(i)
            trades.append(dict(code=codes[i], i=i, tb=p["tb"], ts=t, buy_date=P.dstr[p["tb"]], sell_date=P.dstr[t],
                               cost=p["cost"], proceeds=p["recv"], pnl=p["recv"] - p["cost"],
                               ret=p["recv"] / p["cost"] - 1.0, hold=t - p["tb"], delist=p.get("delist", 0),
                               sh=p["sh0"], rank=p["rank"], reduced=p.get("reduced", 0)))
        if (t - 1) in buy:
            def open_eq():
                e_ = cash + v50
                for i_, p_ in pos.items():
                    o_ = P.O[t, i_]; rf_ = P.refe[t, i_]
                    e_ += p_["val"] * (o_ / rf_) if (o_ == o_ and rf_ == rf_) else p_["val"]
                return e_
            eq = open_eq()
            # ---- 拉回（只在衛星比例 < 1 時）
            if sat_w < 1.0:
                sv = eq - B
                dev = sv / eq - sat_w
                do = (mode == "fixed30") or (abs(dev) > band)
                if do:
                    Bt = (1.0 - sat_w) * eq
                    rec = dict(date=P.dstr[t], sat_ratio=sv / eq, B_old=B, B_new=Bt, reduce=0.0)
                    if Bt >= B:
                        cnt["reb_up"] += 1; rec["dir"] = "sat_high"
                    else:
                        cnt["reb_down"] += 1; rec["dir"] = "sat_low"
                    short = Bt - (max(cash, 0.0) + v50)
                    if short > 1.0:
                        # 個股依市值等比例減碼
                        vals = {}
                        for i, p in pos.items():
                            o = P.O[t, i]; rf = P.refe[t, i]
                            if not (o == o) or not (rf == rf) or o <= limit_dn(P.ref[t, i]) + 1e-9:
                                continue
                            vals[i] = p["val"] * (o / rf)
                        V = sum(vals.values())
                        if V > 0:
                            cnt["reb_reduce_events"] += 1
                            f = min(1.0, short / (V * (1.0 - 0.006 * cmult - extra)))
                            lot_only = t < P.t_odd
                            unit = 1000 if lot_only else 1
                            for i in sorted(vals, key=lambda j: codes[j]):
                                p = pos[i]; o = P.O[t, i]
                                cur_sh = p.get("shc", p["sh"])
                                sh = int(cur_sh * f / unit) * unit
                                gross = sh * o
                                if liq_cap is not None and gross > liq_cap * P.AMT[t, i]:
                                    sh = int(liq_cap * P.AMT[t, i] / o / unit) * unit; gross = sh * o
                                if sh <= 0:
                                    continue
                                frac = min(1.0, sh / cur_sh)
                                net = gross * (1.0 - tick(o) / 2.0 / o * cmult - extra)
                                fe = fees(net, sh); tx = tax * net
                                cash += net - fe - tx; p["recv"] += net - fe - tx
                                cnt["cost_total"] += (gross - net) + fe + tx; cnt["sell_amt"] += gross
                                cnt["reb_reduce_legs"] += 1; cnt["reb_reduce_amt"] += gross; rec["reduce"] += gross
                                p["val"] *= (1.0 - frac); p["shc"] = cur_sh - sh; p["sh"] = max(cur_sh - sh, 1); p["reduced"] = 1
                                if cur_sh - sh <= 0:
                                    p["gone"] = True
                            for i in [j for j in list(pos) if pos[j].get("gone")]:
                                p = pos.pop(i)
                                trades.append(dict(code=codes[i], i=i, tb=p["tb"], ts=t, buy_date=P.dstr[p["tb"]],
                                                   sell_date=P.dstr[t], cost=p["cost"], proceeds=p["recv"],
                                                   pnl=p["recv"] - p["cost"], ret=p["recv"] / p["cost"] - 1.0,
                                                   hold=t - p["tb"], delist=0, sh=p["sh0"], rank=p["rank"], reduced=1))
                            eq = open_eq()
                            Bt = (1.0 - sat_w) * eq
                    B = min(Bt, max(cash, 0.0) + v50)
                    rec["B_final"] = B
                    reb.append(rec)
            if mode == "fixed30":
                sat_cap = sat_w * eq
            else:
                sat_cap = max(eq - B, 0.0)
            if div_mode == "list5":
                div = max(len(buy[t - 1]), 5)
            else:
                div = nmax
            target = sat_cap / div
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
                avail = max(cash, 0.0) + e50_liq(t)
                budget = min(target, avail)
                if liq_cap is not None and budget > liq_cap * P.AMT[t, i]:
                    budget = liq_cap * P.AMT[t, i]; cnt["liq_capped"] += 1
                px = o * (1.0 + tick(o) / 2.0 / o * cmult + extra)
                lot_only = t < P.t_odd
                unit = 1000 if lot_only else 1
                sh = int(budget / (px * (1 + fee)) / unit) * unit
                while sh > 0 and sh * px + fees(sh * px, sh) > budget + 1e-9:
                    sh -= unit
                if sh <= 0:
                    cnt["lot_skip" if lot_only else "no_cash"] += 1
                    if min_frac > 0:
                        cnt["blk_min_or_zero"] += 1
                    continue
                if min_frac > 0 and sh * o < min_frac * target:
                    cnt["blk_min"] += 1; cnt["blk_min_or_zero"] += 1; continue
                fe = fees(sh * px, sh)
                c = sh * px + fe
                if c > cash:
                    cash += e50_sell_for(t, c - cash)
                cash -= c
                cnt["buy"] += 1; cnt["cost_total"] += sh * (px - o) + fe; cnt["buy_amt"] += sh * o
                pos[i] = dict(tb=t, sh=sh, sh0=sh, val=sh * o, cost=c, recv=0.0, pend=False, new=True, rank=rk)
            clamp(t)
        if cash < -1e-6:
            cash += e50_sell_for(t, -cash)
        if cash >= e50_min:
            hs = e50_hs(t)
            f = max(minfee, fee * cash / (1.0 + fee))
            g = (cash - f) / (1.0 + hs)
            cnt["e50_buy"] += g; cnt["e50_cost"] += cash - g
            v50 += g; cash = 0.0
        v50 *= P.e50c[t] / P.e50o[t]
        B *= P.e50c[t] / P.e50o[t]
        v = 0.0
        for i, p in pos.items():
            if p.pop("new", False):
                p["val"] = p["sh"] * P.Cf[t, i]
            else:
                r = P.ret[t, i]
                if r == r:
                    p["val"] *= (1.0 + r)
            v += p["val"]
        clamp(t)
        nav[t] = cash + v + v50; npos[t] = len(pos); cashw[t] = cash / nav[t]; w50[t] = v50 / nav[t]
        stkw[t] = v / nav[t]; Bs[t] = B
        if t in keep:
            kp = keep[t]
            for i, p in pos.items():
                p["pend"] = i not in kp
    tr = pd.DataFrame(trades)
    return dict(nav=nav, trades=tr, cnt=cnt, npos=npos, cashw=cashw, w50=w50, stkw=stkw, B=Bs, reb=reb,
                clamp_days=int(clamp_day.sum()), clamp_dates=[P.dstr[k] for k in np.where(clamp_day)[0]],
                open_pos=[(codes[i], P.dstr[p["tb"]], float(p["cost"]), float(p["val"])) for i, p in pos.items()])


def ols(y, x):
    y = np.asarray(y, float); x = np.asarray(x, float)
    X = np.column_stack([np.ones(len(x)), x])
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    e = y - X @ b
    n = len(y)
    s2 = (e @ e) / (n - 2)
    cov = s2 * np.linalg.inv(X.T @ X)
    r2 = 1.0 - (e @ e) / ((y - y.mean()) @ (y - y.mean()))
    return dict(n=n, alpha=float(b[0]), beta=float(b[1]), t_alpha=float(b[0] / np.sqrt(cov[0, 0])),
                t_beta=float(b[1] / np.sqrt(cov[1, 1])), r2=float(r2))


def cell_stats(P, res, e50nav):
    """主要數字：年化、最大回撤、筆數、平均個股部位、拉回、夾住、月 OLS、月超額 t。"""
    s = S.stats(P, res)
    nav = res["nav"]; e = P.T - 1
    i19 = L.didx(P, "2018-12-28")
    s["mdd_2019"] = L.maxdd(P, nav, i19, e)[0]
    tr = res["trades"]
    if len(tr):
        f = int(tr.tb.min())
        s["avg_stock_w_from_first_buy"] = float(res["stkw"][f:].mean())
        s["avg_npos_from_first_buy"] = float(res["npos"][f:].mean())
        s["sat_ratio_close"] = None
    s["reb"] = dict(up=res["cnt"]["reb_up"], down=res["cnt"]["reb_down"], reduce_events=res["cnt"]["reb_reduce_events"],
                    reduce_legs=res["cnt"]["reb_reduce_legs"], reduce_amt=res["cnt"]["reb_reduce_amt"],
                    events=res["reb"])
    s["clamp_days"] = res["clamp_days"]          # 當天 B 被往下夾（降低）的天數
    idle = (res["cashw"] + res["w50"]) * nav
    with np.errstate(invalid="ignore"):
        bind = (res["B"] >= idle - 1.0) & (res["B"] > 0)
    s["bind_days"] = int(np.nansum(bind[P.t0 + 1:]))   # 收盤 B ＝ 現金＋0050（衛星沒有閒置的錢）的天數
    s.pop("sat_ratio_close", None)
    with np.errstate(invalid="ignore", divide="ignore"):
        sr = 1.0 - res["B"] / nav
    if len(tr):
        x = sr[f:]
        x = x[np.isfinite(x)]
        s["sat_ratio_from_first_buy"] = dict(min=float(x.min()), median=float(np.median(x)), max=float(x.max()))
    rm = R.month_ret(P, nav); rb = R.month_ret(P, e50nav)
    o = ols(rm.values, rb.values)
    o["first"] = str(rm.index[0]); o["last"] = str(rm.index[-1])
    s["ols_full"] = o
    m19 = rm.index >= pd.Period("2019-01", "M")
    s["ols_2019"] = ols(rm.values[m19], rb.values[m19])
    s["t_month_excess_full"] = R.paired(P, nav, e50nav, "2015-08")
    s["t_month_excess_2019"] = R.paired(P, nav, e50nav, "2019-01")
    dr = nav[P.t0 + 1:] / nav[P.t0:-1] - 1.0
    s["vol_full"] = float(dr.std(ddof=1) * np.sqrt(252)); s["sharpe_full"] = float(dr.mean() / dr.std(ddof=1) * np.sqrt(252))
    return s
