# -*- coding: utf-8 -*-
"""S4 E3'（E3＋排除創新板＋最小買進金額）獨立複製：自寫函式庫。
只依 s4_e3_purity/FINDINGS.md 第 2 節與 PREREG.json 的文字規格。
未讀、未 import 研究員的 s4lib.py／run.py／posthoc.py，也未 import s3lib.py／s2lib.py／stlib.py／cslib.py。
import 的現成程式：s3_winners_momentum/replicate/rep_s3_lib.py（S3 獨立複製員：L_MA／前 20 日漲幅名單、
名單規則 E3／C10）與它依賴的 rep_s2_lib.py、rep_s1_lib.py。
本檔自己寫的部分：
  1. 創新板逐日排除遮罩（依 rep_s3_names.py 掃出的 name 欄 -(KY)?創$ 起訖日）；
  2. 引擎 simulate：rep_s3_lib.simulate 的買賣邏輯，加上
     (a) 最小買進金額 min_frac（股數 × 開盤價 < min_frac × 目標 → 不買、不占名額、看名單下一檔）；
     (b) 空位放 0050（idle='e50'，可用資金＝max(現金,0)＋0050 剩餘可變現額，即 S2 的 e50_fix 口徑）；
  3. 0050 還原開盤、交易對照 0050、分組統計。
"""
import sys, os, json
import numpy as np
import pandas as pd

sys.path.insert(0, "/home/user/-/research/tw_stock_selection/agents/s3_winners_momentum/replicate")
import rep_s3_lib as Q  # noqa: E402
R, L = Q.R, Q.L
FEE, TAX = Q.FEE, Q.TAX
tick, limit_up, limit_dn = Q.tick, Q.limit_up, Q.limit_dn
STRICT, X2 = Q.STRICT, Q.X2
TAX50 = 0.001


def e50_tick(p):
    return 0.01 if p < 50 else 0.05


def attach_e50(P, scratch):
    """0050 還原收盤（etf_0050.csv adj，缺值沿用前值）、還原開盤＝還原收盤 × 開盤 ÷ 收盤（官方日 K，缺值沿用前值）、原始開盤。"""
    oc = json.load(open(os.path.join(scratch, "e50_open_close.json")))
    o = np.array([oc.get(d, (np.nan, np.nan))[0] for d in P.dstr], float)
    c = np.array([oc.get(d, (np.nan, np.nan))[1] for d in P.dstr], float)
    P.e50_missing_oc = int((~np.isfinite(o) | ~np.isfinite(c)).sum())
    o = pd.Series(o).ffill().values; c = pd.Series(c).ffill().values
    P.e50c = P.e50.values.astype(float)
    P.e50o = P.e50c * o / c
    P.e50raw_o = o


def innov_mask_fn(P, scratch):
    """回傳 excl(s) -> bool[N]：訊號日當天 name 欄符合 -(KY)?創$ 的股票。用起訖日判斷，並檢查起訖間每天都帶創字（n_days）。"""
    inj = json.load(open(os.path.join(scratch, "innov_names.json")))["innov"]
    dstr = np.array(P.dstr)
    M = np.zeros((P.T, P.N), bool)
    chk = {}
    for code, e in inj.items():
        if code not in P.cidx:
            continue
        i = P.cidx[code]
        a = int(np.searchsorted(dstr, e["first"])); b = int(np.searchsorted(dstr, e["last"]))
        M[a:b + 1, i] = True
        present = int(P.inbar[a:b + 1, i].sum())
        chk[code] = dict(n_days=e["n_days"], present_in_range=present)
    P.innov_check = chk
    P.innov_M = M
    return lambda s: M[s]


def simulate(P, buy, keep, nmax=10, capital=1_000_000.0, cmult=1.0, extra=0.0, block_span=None, liq_cap=0.05,
             min_frac=0.5, idle="cash", e50_min=20_000.0):
    """訊號日收盤後決定、隔日開盤成交；先賣後買；當天賣的當天不買回；開盤 ≥ 漲停價不買、不遞補；
    開盤 ≤ 跌停價或無開盤價順延賣；自日 K 消失以最後收盤結清；目標＝開盤估淨值 ÷ nmax，不超過可用資金；
    單筆 ≤ 當日成交值 5%；2020-10-26 前只能買整張；股數 × 開盤 < min_frac × 目標 → 不買、不占名額、看下一檔。
    idle='e50'：個股買賣後現金 ≥ 2 萬全數買 0050；買個股現金不夠賣 0050；0050 成本＝手續費同個股、稅 0.1%、
    半個跳動點（< 50 元 0.01、≥ 50 元 0.05），都 ×cmult，不加 extra。"""
    T, t0, codes = P.T, P.t0, P.codes
    fee = FEE * cmult; minfee = 20.0 * cmult; tax = TAX * cmult; tax50 = TAX50 * cmult
    use50 = idle == "e50"
    cash = capital; pos = {}; v50 = 0.0
    nav = np.full(T, np.nan); nav[t0] = capital
    npos = np.zeros(T); cashw = np.ones(T); w50 = np.zeros(T)
    trades = []
    cnt = dict(cand=0, buy=0, limit_up=0, no_open=0, lot_skip=0, no_cash=0, no_slot=0, blocked=0, liq_capped=0,
               blk_min=0, blk_min_or_zero=0, sell_defer=0, sell_split=0, delist=0, cost_total=0.0, buy_amt=0.0,
               sell_amt=0.0, sold_in_buylist=0, e50_buy=0.0, e50_sell=0.0, e50_cost=0.0)

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
        return e50_tick(o) / 2.0 / o * cmult

    def e50_liq(t):
        """0050 全數賣出可拿回的淨額。"""
        if v50 <= 0:
            return 0.0
        g = v50 * (1.0 - e50_hs(t))
        return max(0.0, g - max(minfee, fee * g) - tax50 * g)

    def e50_sell_for(t, need):
        """賣 0050 讓現金增加 need（淨額）；回傳實際增加的現金。"""
        nonlocal v50
        if v50 <= 0 or need <= 0:
            return 0.0
        hs = e50_hs(t)
        k = 1.0 - hs
        g = need / (k * (1.0 - fee - tax50))           # 先假設手續費按比例
        if fee * g * k < minfee:
            g = (need + minfee) / (k * (1.0 - tax50))
        if g >= v50 * 0.999999:
            got = e50_liq(t); gross = v50; v50 = 0.0
        else:
            gross = g; v50 -= g
            n = g * k; got = n - max(minfee, fee * n) - tax50 * n
        cnt["e50_sell"] += gross; cnt["e50_cost"] += gross - got
        return got

    for t in range(t0 + 1, T):
        if use50 and v50 > 0:
            v50 *= P.e50o[t] / P.e50c[t - 1]
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
            eq = cash + (v50 if use50 else 0.0)
            for i, p in pos.items():
                o = P.O[t, i]; rf = P.refe[t, i]
                eq += p["val"] * (o / rf) if (o == o and rf == rf) else p["val"]
            target = eq / nmax
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
                avail = max(cash, 0.0) + (e50_liq(t) if use50 else 0.0)
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
                if use50 and c > cash:
                    cash += e50_sell_for(t, c - cash)
                cash -= c
                cnt["buy"] += 1; cnt["cost_total"] += sh * (px - o) + fe; cnt["buy_amt"] += sh * o
                pos[i] = dict(tb=t, sh=sh, sh0=sh, val=sh * o, cost=c, recv=0.0, pend=False, new=True, rank=rk)
        if use50:
            if cash < -1e-6:
                cash += e50_sell_for(t, -cash)
            if cash >= e50_min:
                hs = e50_hs(t)
                f = max(minfee, fee * cash / (1.0 + fee))
                g = (cash - f) / (1.0 + hs)              # 買進的 0050 市值（開盤價計）
                cnt["e50_buy"] += g; cnt["e50_cost"] += cash - g
                v50 += g; cash = 0.0
            v50 *= P.e50c[t] / P.e50o[t]
        v = 0.0
        for i, p in pos.items():
            if p.pop("new", False):
                p["val"] = p["sh"] * P.Cf[t, i]
            else:
                r = P.ret[t, i]
                if r == r:
                    p["val"] *= (1.0 + r)
            v += p["val"]
        nav[t] = cash + v + v50; npos[t] = len(pos); cashw[t] = cash / nav[t]; w50[t] = v50 / nav[t]
        if t in keep:
            kp = keep[t]
            for i, p in pos.items():
                p["pend"] = i not in kp
    tr = pd.DataFrame(trades)
    return dict(nav=nav, trades=tr, cnt=cnt, npos=npos, cashw=cashw, w50=w50,
                open_pos=[(codes[i], P.dstr[p["tb"]], float(p["cost"]), float(p["val"])) for i, p in pos.items()])


def stats(P, res):
    s = Q.stats(P, res)
    tr = res["trades"]
    if len(tr):
        f = int(tr.tb.min())
        s["avg_idle_from_first_buy"] = float((res["cashw"][f:] + res["w50"][f:]).mean())
        w = tr.ret > 0
        s["avg_win"] = float(tr.ret[w].mean()); s["avg_loss"] = float(-tr.ret[~w].mean())
        s["hold_mean"] = float(tr.hold.mean())
    return s


def brief(s):
    b = Q.brief(s)
    b["blk_min"] = s["cnt"]["blk_min"]; b["blk_min_or_zero"] = s["cnt"]["blk_min_or_zero"]
    return b


def vs_e50(P, tr):
    """每筆交易：同一段持有期（進場日開盤 → 出場日開盤）抱 0050 的報酬（還原開盤、不扣成本）。"""
    tb = tr.tb.values.astype(int); ts = np.minimum(tr.ts.values.astype(int), P.T - 1)
    return P.e50o[ts] / P.e50o[tb] - 1.0


def tstat(x):
    x = np.asarray(x, float)
    n = len(x)
    if n < 2:
        return dict(n=n)
    return dict(n=int(n), mean=float(x.mean()), median=float(np.median(x)), t=float(x.mean() / (x.std(ddof=1) / np.sqrt(n))))


def grp(x, mask):
    x = np.asarray(x, float)[mask]
    if len(x) == 0:
        return dict(n=0)
    return dict(n=int(len(x)), mean=float(x.mean()), median=float(np.median(x)), win=float((x > 0).mean()),
                lt_m15=float((x < -0.15).mean()))
