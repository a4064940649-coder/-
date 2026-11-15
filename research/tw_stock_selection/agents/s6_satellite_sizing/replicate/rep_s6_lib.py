# -*- coding: utf-8 -*-
"""S6（衛星資金利用率與名次加權）獨立複製：自寫函式庫。
只依 s6_satellite_sizing/FINDINGS.md 第 2 節與 PREREG.json 的文字規格。
未讀、未 import 研究員的 s6lib.py／prep.py／run.py／posthoc.py，也未 import s5lib／s4lib／s3lib／s2lib／stlib／cslib。
import 的現成程式：s5_core_satellite/replicate/rep_s5_lib.py（S5 獨立複製員；已與 S5 逐筆一致）及其依賴。
simulate_s6 ＝ rep_s5_lib.simulate_cs 的逐行複製（以字串替換從其原始碼產生，diff 只有函式名、參數與目標金額這段），只改「每檔目標金額」：
  sizing='W10'：u＝衛星資金÷10；'W7'：u＝衛星資金÷7（最多仍 10 檔，錢用完為止，沿用引擎 min(目標, 可用資金)）；
  'WN'：u＝衛星資金÷max(當月名單檔數, 5)（名單＝buy[s] 全部，含已持有、含當天要賣的）。
  rank_w=(w_top, w_rest)：L_MA 名次 1～5 權重 w_top、6～10 權重 w_rest；縮放 s＝min(1, 衛星資金÷(u×Σ權重))，
  Σ 對整份名單；每檔目標＝u×權重×s。ranks[s]＝與 buy[s] 對齊的 L_MA 名次（1 起算）。
  最小買進金額用該檔自己的目標（target 換成每檔的目標，其餘不變）。
"""
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, "/home/user/-/research/tw_stock_selection/agents/s5_core_satellite/replicate")
import rep_s5_lib as C  # noqa: E402
S, Q, R, L = C.S, C.Q, C.R, C.L
FEE, TAX, TAX50 = S.FEE, S.TAX, S.TAX50
tick, limit_up, limit_dn = S.tick, S.limit_up, S.limit_dn


def c10_ranks(lists, buy, nmax=10):
    """每個訊號日：buy[s] 裡每檔在 L_MA 的名次（1 起算）。C10 的買進名單一定在 L_MA 前 10 名內。"""
    out = {}
    for s, b in buy.items():
        lma = list(lists[s]["lma"][:nmax])
        out[s] = np.array([lma.index(int(i)) + 1 for i in b], dtype=int)
    return out


def simulate_s6(P, buy, keep, sat_w=0.7, band=0.10, nmax=10, capital=1_000_000.0, cmult=1.0, extra=0.0,
                block_span=None, liq_cap=0.05, min_frac=0.5, e50_min=20_000.0, mode="fixedB", div_mode="ten",
                sizing="W10", rank_w=None, ranks=None):
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
            # ---- S6：每檔目標金額（本檔唯一改動的地方）
            lst = buy[t - 1]
            n_list = len(lst)
            if sizing == "WN" or div_mode == "list5":
                div = max(n_list, 5)
            elif sizing == "W7":
                div = 7
            else:
                div = nmax
            u = sat_cap / div
            if rank_w is not None:
                wts = np.where(np.asarray(ranks[t - 1]) <= 5, rank_w[0], rank_w[1]).astype(float)
                tot = u * float(wts.sum())
                scl = min(1.0, sat_cap / tot) if tot > 0 else 1.0
                tgts = u * wts * scl
            else:
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
