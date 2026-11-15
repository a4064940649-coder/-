# -*- coding: utf-8 -*-
"""S9（換股時效與訊號日定義）獨立複製：自寫函式庫。
只依 s9_signal_timing/PREREG.json 與 FINDINGS.md 第 3、11 節的文字規格。
未讀、未 import 研究員的 s9lib／s8lib／s7lib／s6lib／s5lib 及其 run／prep／posthoc。
import 的現成程式：s6_satellite_sizing/replicate/rep_s6_lib.py（S6 獨立複製員）及其依賴（rep_s5／s4／s3／s2／s1 lib）。

本檔自己寫的部分：
  1. 訊號日家族 schedule()：
     A_D：d*＝營收公布月 M（營收月 M−1 的次月）第 D 日起（含）第一個交易日，訊號日 k＝d*＋1（交易日列號）。
          D ≥ 10 用 M−1 營收（SURGE 列 m）；A08、A09 整個月用 M−2（列 m−1）。
     B_R：k＝月 M 第一個日期 ≥ R 的交易日。期限日 j＝月 M 第一個日期 ≥ 10 的交易日。
          opt：一律 M−1；pes：k ≤ j 的月份用 M−2。
     月份對應：迴圈變數 m＝營收月 M−1 的列號（與 rep_s1_lib.rev_avail 相同），所以 A10 的 k＝rev_avail(...)[2]。
  2. make_lists_sched()：rep_s3_lib.make_lists 的邏輯，訊號列與營收列改由 schedule 給；
     list_off＝−1 是收盤版（名單放 k−1 列）；shift＝路徑位移（在名單列上再往後移）。
     「股池內 SURGE 有值不足 50 檔不產生訊號」在未位移的名單列判斷（同 S3／S6 複製員）。
  3. simulate_s9(exec_mode)：rep_s6_lib.simulate_s6 的逐行複製（不含審查用的 mode／div_mode／rank_w 以外的改動），
     exec_mode='open'：與 simulate_s6 相同（程式驗收：日淨值、交易逐筆相同）。
     exec_mode='close'：名單列 t−1 → 第 t 日收盤成交。順序：持股依當日報酬估到收盤、0050 與底倉 B 估到收盤
       → 下市結清（在估值之前，用前一收盤市值）→ 賣出待賣股（原始收盤價、無成交不賣、收盤 ≤ 跌停價不賣、單筆 ≤ 成交值 5%）
       → 拉回（收盤估淨值；減碼用收盤價）→ 依名次買進（原始收盤 × (1＋半跳動點＋extra)、無成交不買、收盤 ≥ 漲停價不買）
       → 現金 ≥ 2 萬掃 0050（0050 原始收盤的半跳動點）。成本、5% 上限、最小買進金額、零股規則全部同 open。
"""
import sys, os, json
import numpy as np
import pandas as pd

sys.path.insert(0, "/home/user/-/research/tw_stock_selection/agents/s6_satellite_sizing/replicate")
import rep_s6_lib as K6  # noqa: E402
C, S, Q, R, L = K6.C, K6.S, K6.Q, K6.R, K6.L
FEE, TAX, TAX50 = S.FEE, S.TAX, S.TAX50
tick, limit_up, limit_dn = S.tick, S.limit_up, S.limit_dn
c10_ranks = K6.c10_ranks
ST, X2 = S.STRICT, S.X2

DEFS = {
    "A08": ("A", 8, "M-2"), "A09": ("A", 9, "M-2"), "A10": ("A", 10, "M-1"), "A11": ("A", 11, "M-1"),
    "A12": ("A", 12, "M-1"), "A13": ("A", 13, "M-1"), "A15": ("A", 15, "M-1"),
    "B11opt": ("B", 11, "opt"), "B11pes": ("B", 11, "pes"),
}


def attach_e50_close(P, scratch):
    """0050 原始收盤（收盤版掃 0050 的半跳動點用）。缺值沿用前值。"""
    oc = json.load(open(os.path.join(scratch, "e50_open_close.json")))
    c = np.array([oc.get(d, (np.nan, np.nan))[1] for d in P.dstr], float)
    P.e50raw_c = pd.Series(c).ffill().values


def _first_ge(P, dstr_arr, y, mo, d):
    return int(np.searchsorted(dstr_arr, "%04d-%02d-%02d" % (y, mo, d), side="left"))


def schedule(P, nm, mi0, did):
    """回傳 list of dict(m, k, j, m_rev, ym_pub)：m＝營收月 M−1 列號；k＝訊號日列號；j＝期限日列號；m_rev＝用的 SURGE 列。"""
    conv, D, rv = DEFS[did]
    da = np.array(P.dstr)
    out = []
    for m in range(nm):
        mi = mi0 + m
        y, mo = divmod(mi, 12); mo += 1                      # 營收月 M−1
        py, pmo = (y + 1, 1) if mo == 12 else (y, mo + 1)    # 公布月 M
        j = _first_ge(P, da, py, pmo, 10)
        if conv == "A":
            ds = _first_ge(P, da, py, pmo, D)
            k = ds + 1
            m_rev = m if rv == "M-1" else m - 1
        else:
            k = _first_ge(P, da, py, pmo, D)
            m_rev = m if (rv == "opt" or k > j) else m - 1
        if k >= P.T or m_rev < 0:
            continue
        out.append(dict(m=m, k=k, j=j, m_rev=m_rev, ym_pub="%04d-%02d" % (py, pmo), early=bool(k <= j)))
    return out


def make_lists_sched(P, S_, sched, shift=0, list_off=0, excl=None, min_n=50, ma_len=60, ret_len=20):
    """{名單列 s: dict(lma, r20, m, k)}；名單列 s0＝k＋list_off，s＝s0＋shift（價格面資料在 s 計）。"""
    out = {}
    codes = np.array(P.codes)
    mav = R.ma(P, ma_len)
    for e_ in sched:
        s0 = e_["k"] + list_off
        s = s0 + shift
        if s >= P.T - 1 or s0 < P.t0 or s0 >= P.T - 1:
            continue
        sv = S_[e_["m_rev"]]
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
        assert s not in out
        out[s] = dict(lma=order, r20=r20, m=e_["m"], m_rev=e_["m_rev"], k=e_["k"], ym_pub=e_["ym_pub"])
    return out


def plans_from_lists(lists):
    buy, keep = Q.cell_plans(lists, "C10", x=0.10)
    return buy, keep, c10_ranks(lists, buy)


def simulate_s9(P, buy, keep, sat_w=0.7, band=0.10, nmax=10, capital=1_000_000.0, cmult=1.0, extra=0.0,
                block_span=None, liq_cap=0.05, min_frac=0.5, e50_min=20_000.0, sizing="W10", exec_mode="open",
                rec_pos=False):
    """sizing='W10'：每檔＝衛星資金÷10；'WN'：衛星資金÷max(名單檔數,5)。exec_mode 見檔頭。"""
    assert exec_mode in ("open", "close")
    CL = exec_mode == "close"
    T, t0, codes = P.T, P.t0, P.codes
    fee = FEE * cmult; minfee = 20.0 * cmult; tax = TAX * cmult; tax50 = TAX50 * cmult
    cash = capital; pos = {}; v50 = 0.0
    B = (1.0 - sat_w) * capital
    nav = np.full(T, np.nan); nav[t0] = capital
    npos = np.zeros(T); cashw = np.ones(T); w50 = np.zeros(T); stkw = np.zeros(T); Bs = np.full(T, np.nan)
    Bs[t0] = B
    trades = []; reb = []
    snap = {}; flows = {}
    a3close = []   # 收盤版：成交價 ≠ 收盤 ×(1±成本) 的違規記錄（驗收 A3）
    clamp_day = np.zeros(T, bool)
    cnt = dict(cand=0, buy=0, limit_up=0, no_open=0, lot_skip=0, no_cash=0, no_slot=0, blocked=0, liq_capped=0,
               blk_min=0, blk_min_or_zero=0, sell_defer=0, sell_defer_limdn=0, sell_split=0, delist=0,
               cost_total=0.0, buy_amt=0.0, sell_amt=0.0, sold_in_buylist=0, e50_buy=0.0, e50_sell=0.0,
               e50_cost=0.0, reb_up=0, reb_down=0, reb_reduce_events=0, reb_reduce_legs=0, reb_reduce_amt=0.0)
    limup_list = []
    e50px = P.e50raw_c if CL else P.e50raw_o

    def _fl(t_, i_, x_):
        if rec_pos:
            flows[(t_, i_)] = flows.get((t_, i_), 0.0) + x_

    def fees(val, sh):
        lot = (sh // 1000) * 1000
        f = 0.0
        if lot > 0:
            f += max(minfee, fee * val * lot / sh)
        if sh - lot > 0:
            f += max(minfee, fee * val * (sh - lot) / sh)
        return f

    def e50_hs(t):
        o = e50px[t]
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

    # 價格存取：open 用開盤、close 用收盤；估值倍數 mult（持股 val 是前一收盤 → 成交時點的倍數）
    def px_of(t, i):
        return P.C[t, i] if CL else P.O[t, i]

    def can_trade(t, i):
        if CL:
            return bool(P.traded[t, i])
        o = P.O[t, i]; rf = P.refe[t, i]
        return (o == o) and (rf == rf)

    def mult(t, i):
        if CL:
            return 1.0          # 收盤版：val 已在當天開頭估到收盤
        return P.O[t, i] / P.refe[t, i]

    for t in range(t0 + 1, T):
        if CL:
            v50 *= P.e50c[t] / P.e50c[t - 1]
            B *= P.e50c[t] / P.e50c[t - 1]
        else:
            if v50 > 0:
                v50 *= P.e50o[t] / P.e50c[t - 1]
            B *= P.e50o[t] / P.e50c[t - 1]
        closed = []
        # 下市結清（用前一收盤市值），之後收盤版才把其餘持股估到收盤
        for i in sorted(pos, key=lambda j: codes[j]):
            p = pos[i]
            if t > P.last[i]:
                cash += p["val"]; p["recv"] += p["val"]; _fl(t, i, p["val"]); p["delist"] = 1; cnt["delist"] += 1
                closed.append(i)
        if CL:
            for i, p in pos.items():
                if i in closed:
                    continue
                r = P.ret[t, i]
                if r == r:
                    p["val"] *= (1.0 + r)
        for i in sorted(pos, key=lambda j: codes[j]):
            if i in closed:
                continue
            p = pos[i]
            if not p["pend"]:
                continue
            if not can_trade(t, i):
                cnt["sell_defer"] += 1; continue
            o = px_of(t, i)
            if o <= limit_dn(P.ref[t, i]) + 1e-9:
                cnt["sell_defer"] += 1; cnt["sell_defer_limdn"] += 1; continue
            gross_full = p["val"] * mult(t, i)
            frac = 1.0
            if liq_cap is not None and gross_full > liq_cap * P.AMT[t, i]:
                frac = liq_cap * P.AMT[t, i] / gross_full; cnt["sell_split"] += 1
            gross = gross_full * frac
            net = gross * (1.0 - tick(o) / 2.0 / o * cmult - extra)
            fe = fees(net, p["sh"]); tx = tax * net
            cash += net - fe - tx; p["recv"] += net - fe - tx; _fl(t, i, net - fe - tx)
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
                    if CL:
                        e_ += p_["val"]
                    else:
                        o_ = P.O[t, i_]; rf_ = P.refe[t, i_]
                        e_ += p_["val"] * (o_ / rf_) if (o_ == o_ and rf_ == rf_) else p_["val"]
                return e_
            eq = open_eq()
            if sat_w < 1.0:
                sv = eq - B
                dev = sv / eq - sat_w
                if abs(dev) > band:
                    Bt = (1.0 - sat_w) * eq
                    rec = dict(date=P.dstr[t], sat_ratio=sv / eq, B_old=B, B_new=Bt, reduce=0.0)
                    if Bt >= B:
                        cnt["reb_up"] += 1; rec["dir"] = "sat_high"
                    else:
                        cnt["reb_down"] += 1; rec["dir"] = "sat_low"
                    short = Bt - (max(cash, 0.0) + v50)
                    if short > 1.0:
                        vals = {}
                        for i, p in pos.items():
                            if not can_trade(t, i):
                                continue
                            o = px_of(t, i)
                            if o <= limit_dn(P.ref[t, i]) + 1e-9:
                                continue
                            vals[i] = p["val"] * mult(t, i)
                        V = sum(vals.values())
                        if V > 0:
                            cnt["reb_reduce_events"] += 1
                            f = min(1.0, short / (V * (1.0 - 0.006 * cmult - extra)))
                            lot_only = t < P.t_odd
                            unit = 1000 if lot_only else 1
                            for i in sorted(vals, key=lambda j: codes[j]):
                                p = pos[i]; o = px_of(t, i)
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
                                cash += net - fe - tx; p["recv"] += net - fe - tx; _fl(t, i, net - fe - tx)
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
            sat_cap = max(eq - B, 0.0)
            lst = buy[t - 1]
            n_list = len(lst)
            div = max(n_list, 5) if sizing == "WN" else nmax
            u = sat_cap / div
            tgts = np.full(n_list, u)
            sold_today = set(closed)
            for rk, i in enumerate(buy[t - 1][:nmax]):
                i = int(i)
                target = float(tgts[rk])
                if i in sold_today:
                    cnt["sold_in_buylist"] += 1
                if i in pos or i in sold_today:
                    continue
                cnt["cand"] += 1
                if len(pos) >= nmax:
                    cnt["no_slot"] += 1; continue
                if block_span and any(a <= t <= b for a, b in block_span.get(codes[i], ())):
                    cnt["blocked"] += 1; continue
                if CL:
                    if not P.traded[t, i]:
                        cnt["no_open"] += 1; continue
                    o = P.C[t, i]
                else:
                    o = P.O[t, i]
                    if not (o == o):
                        cnt["no_open"] += 1; continue
                if o >= limit_up(P.ref[t, i]) - 1e-9:
                    cnt["limit_up"] += 1; limup_list.append((P.dstr[t], codes[i], rk + 1)); continue
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
                if CL and abs(px - P.C[t, i] * (1.0 + tick(P.C[t, i]) / 2.0 / P.C[t, i] * cmult + extra)) > 1e-9:
                    a3close.append((P.dstr[t], codes[i]))
                fe = fees(sh * px, sh)
                c = sh * px + fe
                if c > cash:
                    cash += e50_sell_for(t, c - cash)
                cash -= c; _fl(t, i, -c)
                cnt["buy"] += 1; cnt["cost_total"] += sh * (px - o) + fe; cnt["buy_amt"] += sh * o
                pos[i] = dict(tb=t, sh=sh, sh0=sh, val=sh * o, cost=c, recv=0.0, pend=False, new=True, rank=rk,
                              sig=t - 1)
            clamp(t)
        if cash < -1e-6:
            cash += e50_sell_for(t, -cash)
        if cash >= e50_min:
            hs = e50_hs(t)
            f = max(minfee, fee * cash / (1.0 + fee))
            g = (cash - f) / (1.0 + hs)
            cnt["e50_buy"] += g; cnt["e50_cost"] += cash - g
            v50 += g; cash = 0.0
        if not CL:
            v50 *= P.e50c[t] / P.e50o[t]
            B *= P.e50c[t] / P.e50o[t]
        v = 0.0
        for i, p in pos.items():
            if p.pop("new", False):
                if not CL:
                    p["val"] = p["sh"] * P.Cf[t, i]
            elif not CL:
                r = P.ret[t, i]
                if r == r:
                    p["val"] *= (1.0 + r)
            v += p["val"]
        clamp(t)
        nav[t] = cash + v + v50; npos[t] = len(pos); cashw[t] = cash / nav[t]; w50[t] = v50 / nav[t]
        stkw[t] = v / nav[t]; Bs[t] = B
        if rec_pos:
            snap[t] = {i_: p_["val"] for i_, p_ in pos.items()}
        if t in keep:
            kp = keep[t]
            for i, p in pos.items():
                p["pend"] = i not in kp
    tr = pd.DataFrame(trades)
    return dict(nav=nav, trades=tr, cnt=cnt, npos=npos, cashw=cashw, w50=w50, stkw=stkw, B=Bs, reb=reb,
                limup=limup_list, a3close=a3close, snap=snap, flows=flows,
                clamp_days=int(clamp_day.sum()),
                open_pos=[(codes[i], P.dstr[p["tb"]], float(p["cost"]), float(p["val"]), int(i)) for i, p in pos.items()])


def stock_pnl(res):
    """每檔加總損益：已平倉 pnl ＋ 期末未平倉（市值 − 成本）。回傳 {code: pnl}。"""
    out = {}
    tr = res["trades"]
    if len(tr):
        for c, v in tr.groupby("code").pnl.sum().items():
            out[c] = out.get(c, 0.0) + float(v)
    for c, _, cost, val, _i in res["open_pos"]:
        out[c] = out.get(c, 0.0) + val - cost
    return out
