# -*- coding: utf-8 -*-
"""獨立複製員自寫函式庫（未讀、未 import 研究員的 cslib.py／run.py）。
只依 raw/*.xs、SOURCE.txt 與 FINDINGS.md 第 1、3.1、6、9 節的文字規格。
"""
import os, math, json
import numpy as np
import pandas as pd

PEAD = "/home/user/-/research/tw_stock_selection/agents/o_pead_margin_surprise_low_attention/data"
IDX = "/home/user/jonatw/finlab-txd/data/raw/taiex_twii.csv"
E50 = "/home/user/jonatw/finlab-txd/data/raw/etf_0050.csv"
START = "2017-03-31"
ODD_START = "2020-10-26"
FEE = 0.001425 * 0.28
TAX = 0.003


# ---------------------------------------------------------------- 價格
def tick(p):
    if p < 10: return 0.01
    if p < 50: return 0.05
    if p < 100: return 0.1
    if p < 500: return 0.5
    if p < 1000: return 1.0
    return 5.0


def limit_up(ref):
    raw = ref * 1.1
    tk = tick(raw)
    return math.floor(raw / tk + 1e-7) * tk


def limit_dn(ref):
    raw = ref * 0.9
    tk = tick(raw)
    return math.ceil(raw / tk - 1e-7) * tk


class Panel:
    pass


def load_panel(scratch):
    z = np.load(os.path.join(scratch, "panel.npz"))
    P = Panel()
    P.dates = pd.to_datetime(z["dates"])
    P.dstr = [str(x) for x in z["dates"]]
    P.codes = [str(x) for x in z["codes"]]
    T, N = z["C"].shape
    P.T, P.N = T, N
    O, H, L, C, V, A, G = (z[k] for k in "OHLCVAG")
    P.M = z["M"]; pres = z["P"].astype(bool)
    traded = pres & np.isfinite(C) & (C > 0)
    # 每檔的 K 棒範圍＝第一次出現～最後一次出現
    anyp = pres.any(axis=0)
    first = np.where(anyp, pres.argmax(axis=0), T)
    last = np.where(anyp, T - 1 - pres[::-1].argmax(axis=0), -1)
    tt = np.arange(T)[:, None]
    inbar = (tt >= first[None, :]) & (tt <= last[None, :])
    Cdf = pd.DataFrame(np.where(traded, C, np.nan)).ffill().values
    Cf = np.where(inbar, Cdf, np.nan)                       # 無成交日沿用前一收盤
    Hb = np.where(traded & np.isfinite(H), H, Cf)           # 無成交日 高＝收
    Vb = np.where(traded & np.isfinite(V), V, 0.0)          # 無成交日 量＝0
    Vb = np.where(inbar, Vb, np.nan)
    prevC = np.vstack([np.full((1, N), np.nan), Cf[:-1]])
    ref = np.where(traded & np.isfinite(G), C - G, prevC)   # 參考價＝收盤−漲跌價差
    ref = np.where(ref > 0, ref, prevC)
    ret = np.where(traded & (ref > 0), C / ref - 1.0, 0.0)
    alt = np.where(traded & (prevC > 0), C / prevC - 1.0, 0.0)
    bad = np.abs(ret) > 0.5
    P.n_bad = int(bad.sum())
    P.bad_cells = [(P.dstr[t], P.codes[i], float(ret[t, i]), float(alt[t, i])) for t, i in zip(*np.where(bad))]
    ret = np.where(bad, alt, ret)                           # 規格第 6 節：異常格改用收盤對前收
    ret = np.where(inbar, ret, np.nan)
    P.O = np.where(traded & np.isfinite(O) & (O > 0), O, np.nan)
    P.C, P.Cf, P.Hb, P.Vb, P.ref, P.ret, P.AMT = C, Cf, Hb, Vb, ref, ret, np.where(traded, A, 0.0)
    P.traded, P.inbar, P.first, P.last = traded, inbar, first, last
    P.barno = np.cumsum(inbar, axis=0)
    # 指數與 0050
    ix = pd.read_csv(IDX, parse_dates=["date"]).set_index("date")["close"]
    P.idx_full = ix
    P.idx = ix.reindex(P.dates)
    P.idx_missing = int(P.idx.isna().sum())
    e = pd.read_csv(E50, parse_dates=["date"]).set_index("date")["adj"]
    P.e50_full = e
    P.e50_missing = int(e.reindex(P.dates).isna().sum())
    P.e50 = e.reindex(P.dates).ffill()
    P.t0 = P.dstr.index(START)
    P.t_odd = int(np.searchsorted(np.array(P.dstr), ODD_START))
    return P


# ---------------------------------------------------------------- 基本面
def _first_idx_ge(P, dts):
    return np.searchsorted(P.dates.values, np.asarray(dts, dtype="datetime64[ns]"), side="left")


def fundamentals(P, pit=True, extra_epsy=None):
    """回傳 condC, condA, condR（T×N bool）與輸出欄位（EPS 年增、CAGR、營收均值）。"""
    T, N = P.T, P.N
    cidx = {c: i for i, c in enumerate(P.codes)}
    f = pd.read_parquet(os.path.join(PEAD, "fin_long.parquet"))
    f = f[f.code.isin(cidx)].copy()
    a = pd.read_parquet(os.path.join(PEAD, "avail_dates.parquet"))[["code", "y", "q", "A"]]
    f = f.merge(a, on=["code", "y", "q"], how="left")
    qe = {1: (3, 31), 2: (6, 30), 3: (9, 30), 4: (12, 31)}
    dl = {1: (0, 5, 15), 2: (0, 8, 14), 3: (0, 11, 14), 4: (1, 3, 31)}
    if pit:
        dead = pd.to_datetime([pd.Timestamp(y + dl[q][0], dl[q][1], dl[q][2]) for y, q in zip(f.y, f.q)])
        av = pd.to_datetime(f["A"]).values.copy()
        miss = pd.isna(f["A"]).values
        av[miss] = dead.values[miss]                       # 缺公告日者用法定期限
        f["n_deadline_fallback"] = miss
    else:
        av = pd.to_datetime([pd.Timestamp(y, qe[q][0], qe[q][1]) for y, q in zip(f.y, f.q)]).values
    f["ai"] = _first_idx_ge(P, av)
    f["qi"] = f.y * 4 + (f.q - 1)
    epsq = {}; epsy = {}
    for c, qi, s in zip(f.code, f.qi, f.eps_s):
        epsq[(c, qi)] = s
    for c, y, q, v in zip(f.code, f.y, f.q, f.eps_ytd):
        if q == 4:
            epsy[(c, y)] = v
    if extra_epsy:
        for kk, vv in extra_epsy.items():
            epsy.setdefault(kk, vv)                         # 只補 fin_long 沒有的年度（審查用延伸，不影響凍結數字）
    condC = np.zeros((T, N), bool); condA = np.zeros((T, N), bool)
    vC = np.full((T, N), np.nan, np.float32); vA = np.full((T, N), np.nan, np.float32)
    for c, g in f.groupby("code"):
        i = cidx[c]
        g = g.sort_values(["ai", "qi"])
        ais = g.ai.values; qis = g.qi.values; qs = g.q.values; ys = g.y.values
        # 季
        cur = -1; k = 0; n = len(g)
        while k < n:
            a0 = ais[k]
            if a0 >= T: break
            while k < n and ais[k] == a0:
                if qis[k] > cur: cur = qis[k]
                k += 1
            a1 = ais[k] if k < n else T
            a1 = min(a1, T)
            q0 = epsq.get((c, cur), np.nan); q4 = epsq.get((c, cur - 4), np.nan)
            if q0 == q0 and q4 == q4 and q4 > 0:
                yoy = (q0 / q4 - 1) * 100
                vC[a0:a1, i] = yoy
                if yoy >= 18: condC[a0:a1, i] = True
        # 年
        gy = g[g.q == 4]
        ais = gy.ai.values; ys = gy.y.values
        cur = -1; k = 0; n = len(gy)
        while k < n:
            a0 = ais[k]
            if a0 >= T: break
            while k < n and ais[k] == a0:
                if ys[k] > cur: cur = ys[k]
                k += 1
            a1 = min(ais[k] if k < n else T, T)
            y0 = epsy.get((c, cur), np.nan); y3 = epsy.get((c, cur - 3), np.nan)
            if y0 == y0 and y3 == y3 and y0 > 0 and y3 > 0:
                cg = (math.pow(y0 / y3, 1 / 3) - 1) * 100
                vA[a0:a1, i] = cg
                if cg >= 15: condA[a0:a1, i] = True
    # 月營收
    r = pd.read_parquet(os.path.join(PEAD, "rev_long.parquet"))
    r = r[r.code.isin(cidx)]
    r["mi"] = r.ym.str[:4].astype(int) * 12 + r.ym.str[5:7].astype(int) - 1
    mi0, mi1 = int(r.mi.min()), int(r.mi.max())
    nm = mi1 - mi0 + 1
    Y = np.full((nm, N), np.nan)
    Y[r.mi.values - mi0, r.code.map(cidx).values] = r.yoy_pct.values
    # 每個月的可用日
    m_av = np.zeros(nm, int)
    for m in range(nm):
        mi = mi0 + m
        y, mo = divmod(mi, 12); mo += 1
        if pit:
            ny, nmo = (y + 1, 1) if mo == 12 else (y, mo + 1)
            j = int(np.searchsorted(P.dates.values, np.datetime64(pd.Timestamp(ny, nmo, 10)), side="left"))
            m_av[m] = j + 1                                 # 次月 10 日（順延）之後的第一個交易日
        else:
            me = pd.Timestamp(y, mo, 1) + pd.offsets.MonthEnd(0)
            m_av[m] = int(np.searchsorted(P.dates.values, np.datetime64(me), side="left"))
    condR = np.zeros((T, N), bool); vR = np.full((T, N), np.nan, np.float32)
    for m in range(2, nm):
        a0 = m_av[m]; a1 = m_av[m + 1] if m + 1 < nm else T
        a1 = min(a1, T)
        if a0 >= T or a1 <= a0: continue
        s = np.zeros(N)
        for k in range(3):
            s = s + Y[m - k] / 3                            # 腳本：RevYoY3 += RevM/3；缺任一月→NaN
        vR[a0:a1, :] = s[None, :]
        condR[a0:a1, :] = (s >= 20)[None, :]
    P.rev_month_avail = {mi0 + m: int(m_av[m]) for m in range(nm)}
    return condC, condA, condR, vC, vA, vR


def tech(P, max_from_high=10.0, min_qty=500.0, rs_days=120, market_ma=60, trail=6.0, trail_days=20,
         fast=8, slow=20):
    Cf = pd.DataFrame(P.Cf); Hb = pd.DataFrame(P.Hb); Vb = pd.DataFrame(P.Vb)
    h240 = Hb.rolling(240, min_periods=240).max().values
    fromhigh = (P.Cf / h240 - 1) * 100
    condN = (fromhigh >= -max_from_high) & (P.barno >= 240) & (P.Cf > 0)
    avgq = Vb.rolling(20, min_periods=20).mean().values / 1000.0
    condS = avgq >= min_qty
    old = Cf.shift(rs_days).values
    sret = np.where(old > 0, (P.Cf / old - 1) * 100, np.nan)
    ix = P.idx
    iret = ((ix / ix.shift(rs_days) - 1) * 100).values
    ima = ix.rolling(market_ma, min_periods=market_ma).mean().values
    mok = (ix.values >= ima)
    with np.errstate(invalid="ignore"):
        condL = (sret > iret[:, None]) & (sret > 0)
    excess = sret - iret[:, None]
    hc = Cf.rolling(trail_days, min_periods=trail_days).max().values
    f8 = Cf.rolling(fast, min_periods=fast).mean().values
    s20 = Cf.rolling(slow, min_periods=slow).mean().values
    with np.errstate(invalid="ignore"):
        ex1 = P.Cf <= hc * (1 - trail / 100)
        f8p = np.vstack([np.full((1, P.N), np.nan), f8[:-1]]); s20p = np.vstack([np.full((1, P.N), np.nan), s20[:-1]])
        ex2 = (f8 < s20) & (f8p >= s20p)
    return dict(condN=condN, condS=condS, condL=condL, mok=mok, excess=excess, ex1=ex1, ex2=ex2,
                fromhigh=fromhigh, avgq=avgq, sret=sret, iret=iret)


def signals(P, F, K, universe="twse", drop=None, use_market=True, t_start=None):
    condC, condA, condR = F[0], F[1], F[2]
    sig = np.ones((P.T, P.N), bool)
    parts = {"C": condC, "A": condA, "R": condR, "N": K["condN"], "S": K["condS"], "L": K["condL"]}
    for k, v in parts.items():
        if drop == k: continue
        sig &= v
    if use_market and drop != "M":
        sig &= K["mok"][:, None]
    if universe == "twse":
        sig &= (P.M == 1)
    else:
        sig &= (P.M >= 1)
    sig &= P.inbar & (P.Cf > 0)
    sig[:(P.t0 if t_start is None else t_start)] = False
    return sig


# ---------------------------------------------------------------- 貼文口徑
def post_trades(P, sig, K, t_start=None):
    """每筆等額、不限檔數、下期開盤價進出。訊號日未持有→次日開盤買（無開盤價則該次訊號作廢）；
    出場自進場當日收盤起檢查，成立→下一個有開盤價的交易日開盤賣；賣出當日收盤即可再入選。"""
    ex = K["ex1"] | K["ex2"]
    T = P.T
    out = []
    for i in np.where(sig.any(axis=0))[0]:
        s = sig[:, i]; e = ex[:, i]; o = P.O[:, i]
        held = False; pend = False; want = False; ent = -1
        for t in range(P.t0 if t_start is None else t_start, T):
            ok = o[t] == o[t]
            if held and pend and ok:
                out.append((i, ent, t, 1)); held = False; pend = False
            elif (not held) and want and ok:
                held = True; ent = t
            want = False
            if held:
                if (not pend) and e[t]: pend = True
            elif s[t]:
                want = True
        if held:
            out.append((i, ent, T - 1, 0))
    rows = []
    for (i, a, b, closed) in out:
        bo = P.O[a, i]
        so = P.O[b, i] if closed else P.Cf[b, i]
        raw = so / bo - 1
        # 含息：進場日 C/O，中間日 ret，出場日 O/ref
        g = P.Cf[a, i] / bo
        if closed:
            if b > a + 0:
                g *= np.prod(1 + P.ret[a + 1:b, i]) * (so / P.ref[b, i])
        else:
            g *= np.prod(1 + P.ret[a + 1:b + 1, i])
        rows.append((P.codes[i], int(P.M[a, i]), P.dstr[a - 1], P.dstr[a], P.dstr[b], closed, b - a, bo, so, raw, g - 1,
                     bool(K["ex1"][b - 1, i]) if closed else False, bool(K["ex2"][b - 1, i]) if closed else False, i, a, b))
    df = pd.DataFrame(rows, columns=["code", "mkt", "sig_date", "buy_date", "sell_date", "closed", "hold", "buy_px",
                                     "sell_px", "ret_raw", "ret_div", "ex1", "ex2", "i", "a", "b"])
    return df.sort_values(["buy_date", "code"]).reset_index(drop=True)


def post_tw_curve(P, tr, rt_cost=0.0):
    """時間加權：每日把持有中各筆的含息日報酬等權平均，再連乘。成本記在出場日。"""
    T = P.T
    s = np.zeros(T); n = np.zeros(T)
    for i, a, b, closed in zip(tr.i, tr.a, tr.b, tr.closed):
        s[a] += P.Cf[a, i] / P.O[a, i] - 1; n[a] += 1
        if closed:
            if b - 1 >= a + 1:
                s[a + 1:b] += P.ret[a + 1:b, i]; n[a + 1:b] += 1
            s[b] += (P.O[b, i] / P.ref[b, i]) * (1 - rt_cost) - 1; n[b] += 1
        else:
            s[a + 1:b + 1] += P.ret[a + 1:b + 1, i]; n[a + 1:b + 1] += 1
    r = np.where(n > 0, s / np.maximum(n, 1), 0.0)
    nav = np.cumprod(1 + r[P.t0:])
    return nav, n[P.t0:]


# ---------------------------------------------------------------- 可執行引擎
def run_exec(P, sig, K, capital=1_000_000.0, nmax=10, cost="normal", cash_0050=False, board_lot_rule=True,
             liq_cap=0.05, rank_key=None, t_start=None, same_day_slot=True, e50_ov=None, e50_in=None, e50_cost=(0.000399 + 0.0003, 0.000399 + 0.001 + 0.0003)):
    """3.1 節規則。cost: normal / zero / stress。回傳 dict(nav, trades, counters)。"""
    T = P.T; t0 = P.t0 if t_start is None else t_start
    ex = K["ex1"] | K["ex2"]
    rk = K["excess"] if rank_key is None else rank_key
    fee = FEE if cost != "zero" else 0.0
    tax = TAX if cost != "zero" else 0.0
    slip_on = cost != "zero"
    extra = 0.003 if cost == "stress" else 0.0
    minfee = 20.0 if cost != "zero" else 0.0
    cash = capital
    pos = {}   # i -> dict
    nav = np.full(T, np.nan); nav[t0] = capital
    npos = np.zeros(T); cashw = np.zeros(T)
    trades = []; cnt = dict(cand=0, buy=0, limit_up=0, no_open=0, lot_skip=0, no_slot=0, no_cash=0, sell_defer_limit=0,
                            sell_defer_noopen=0, sell_split=0, delist=0, buy_amt=0.0, fee=0.0, tax=0.0, slip=0.0,
                            orders=0, minfee_orders=0, liq_capped=0)
    e50r = (P.e50 / P.e50.shift(1) - 1).fillna(0).values
    codes = P.codes
    held_at_close = set()

    def order_fee(val, lot_frac):
        f = 0.0
        for fr in (lot_frac, 1 - lot_frac):
            if fr > 1e-12:
                x = fee * val * fr
                cnt["orders"] += 1
                if x < minfee:
                    x = minfee; cnt["minfee_orders"] += 1
                f += x
        return f

    for t in range(t0 + 1, T):
        if cash_0050 == "exact":
            cash *= (1 + e50_ov[t])                         # 空位資金隔夜跟 0050（前收→開盤）
        cash_open = cash
        # ---- 開盤賣出
        for i in sorted(pos, key=lambda j: codes[j]):
            p = pos[i]
            if not p["pend"]:
                continue
            o = P.O[t, i]
            if t > P.last[i]:
                # 下市：以最後市值結清
                cash += p["val"]; cnt["delist"] += 1
                trades.append(dict(code=codes[i], buy_date=P.dstr[p["tb"]], sell_date=P.dstr[t], cost=p["cost"], proceeds=p["recv"] + p["val"], delist=1, hold=t - p["tb"]))
                p["dead"] = True
                continue
            if not (o == o):
                cnt["sell_defer_noopen"] += 1; continue
            ref = P.ref[t, i]
            if ref == ref and o <= limit_dn(ref) + 1e-9:
                cnt["sell_defer_limit"] += 1; continue
            gross_full = p["val"] * (o / ref)
            frac = 1.0
            if liq_cap is not None and gross_full > liq_cap * P.AMT[t, i] > 0:
                frac = liq_cap * P.AMT[t, i] / gross_full; cnt["sell_split"] += 1
            px_adj = (o - (tick(o) / 2 if slip_on else 0.0)) / o
            gross = gross_full * frac
            g2 = gross * px_adj * (1 - extra)
            sh = p["shares"]; lot_frac = ((sh // 1000) * 1000) / sh if sh > 0 else 1.0
            fe = order_fee(g2, lot_frac); tx = tax * g2
            net = g2 - fe - tx
            cnt["fee"] += fe; cnt["tax"] += tx; cnt["slip"] += gross - g2
            cash += net
            p["recv"] += net
            if frac >= 1.0:
                trades.append(dict(code=codes[i], buy_date=P.dstr[p["tb"]], sell_date=P.dstr[t], cost=p["cost"], proceeds=p["recv"], delist=0, hold=t - p["tb"]))
                p["dead"] = True
            else:
                p["val"] *= (1 - frac); p["shares"] = max(1, int(round(sh * (1 - frac))))
        for i in [j for j in pos if pos[j].get("dead")]:
            del pos[i]
        # ---- 開盤估算總權益
        eq = cash
        for i, p in pos.items():
            o = P.O[t, i]; ref = P.ref[t, i]
            eq += p["val"] * (o / ref) if (o == o and ref == ref and ref > 0) else p["val"]
        # ---- 開盤買進（訊號日＝t-1，當時未持有）
        cand = np.where(sig[t - 1])[0]
        held_prev = held_at_close
        cand = [i for i in cand if i not in held_prev]
        cand.sort(key=lambda i: (-rk[t - 1, i] if rk[t - 1, i] == rk[t - 1, i] else 1e18, codes[i]))
        cnt["cand"] += len(cand)
        slots_free_prev = nmax - len(held_prev)
        for i in cand:
            free = nmax - len(pos) if same_day_slot else min(nmax - len(pos), slots_free_prev)
            if free <= 0:
                cnt["no_slot"] += 1; continue
            o = P.O[t, i]
            if not (o == o):
                cnt["no_open"] += 1; continue
            ref = P.ref[t, i]
            if ref == ref and o >= limit_up(ref) - 1e-9:
                cnt["limit_up"] += 1; continue
            budget = min(eq / nmax, cash)
            if liq_cap is not None and budget > liq_cap * P.AMT[t, i]:
                budget = liq_cap * P.AMT[t, i]; cnt["liq_capped"] += 1
            px = (o + (tick(o) / 2 if slip_on else 0.0)) * (1 + extra)
            lot_only = board_lot_rule and t < P.t_odd
            unit = 1000 if lot_only else 1
            sh = int(budget / (px * (1 + fee)) / unit) * unit

            def tot(sh):
                lf = ((sh // 1000) * 1000) / sh
                f = 0.0
                for fr in (lf, 1 - lf):
                    if fr > 1e-12: f += max(minfee, fee * sh * px * fr)
                return sh * px + f
            while sh > 0 and tot(sh) > budget + 1e-9:
                sh -= unit
            if sh <= 0:
                if lot_only: cnt["lot_skip"] += 1
                else: cnt["no_cash"] += 1
                if not same_day_slot: pass
                continue
            lf = ((sh // 1000) * 1000) / sh
            fe = order_fee(sh * px, lf)
            c = sh * px + fe
            cash -= c
            cnt["buy"] += 1; cnt["buy_amt"] += sh * o; cnt["fee"] += fe; cnt["slip"] += sh * (px - o)
            pos[i] = dict(tb=t, shares=sh, val=sh * o, cost=c, recv=0.0, pend=False, new=True)
            if not same_day_slot: slots_free_prev -= 1
        # ---- 收盤：市值滾動、出場檢查
        cash_delta = cash - cash_open
        if cash_0050 == "exact":
            # 開盤同時換手 0050：淨賣出 0050（買股）付賣方費率，淨買進付買方費率；之後賺 0050 盤中報酬（開盤→收盤）
            if cost != "zero":
                c50 = abs(cash_delta) * (e50_cost[1] if cash_delta < 0 else e50_cost[0])
                cash -= c50; cnt["e50_cost"] = cnt.get("e50_cost", 0.0) + c50
            cash *= (1 + e50_in[t])
        elif cash_0050:
            # 近似：現金整日賺 0050 含息日報酬；淨買進 0050 付手續費，淨賣出再付 0.1% 證交稅
            if cost != "zero":
                cash -= abs(cash_delta) * (FEE + (0.001 if cash_delta < 0 else 0.0) + 0.0002)
            cash *= (1 + e50r[t])
        v = 0.0
        for i, p in pos.items():
            if p.pop("new", False):
                p["val"] = p["shares"] * P.O[t, i] * (P.Cf[t, i] / P.O[t, i])
            else:
                r = P.ret[t, i]
                if r == r: p["val"] *= (1 + r)
            v += p["val"]
            if not p["pend"] and t <= P.last[i] and ex[t, i]:
                p["pend"] = True
            if t >= P.last[i] and t < T - 1:
                p["pend"] = True   # 最後一根 K 棒之後視同下市
        nav[t] = cash + v; npos[t] = len(pos); cashw[t] = cash / nav[t]
        held_at_close = set(pos.keys())
    return dict(nav=nav, trades=pd.DataFrame(trades), cnt=cnt, npos=npos, cashw=cashw, open_pos=len(pos))


# ---------------------------------------------------------------- 統計
def ann(v0, v1, d0, d1):
    yrs = (pd.Timestamp(d1) - pd.Timestamp(d0)).days / 365.25
    return (v1 / v0) ** (1 / yrs) - 1, yrs


def month_excess(P, nav, b, i0, i1):
    """日曆月報酬的超額（策略−基準）。i0 為基期（報酬自 i0 的下一日起）。"""
    s = pd.Series(nav[i0:i1 + 1], index=P.dates[i0:i1 + 1])
    bb = pd.Series(np.asarray(b)[i0:i1 + 1], index=P.dates[i0:i1 + 1])
    ms = s.groupby(s.index.to_period("M")).last(); mb = bb.groupby(bb.index.to_period("M")).last()
    # 第一個月的基期＝i0 當日
    rs = ms / ms.shift(1) - 1; rb = mb / mb.shift(1) - 1
    first = ms.index[0]
    if s.index[0].to_period("M") == first and len(s[s.index.to_period("M") == first]) > 1:
        rs.iloc[0] = ms.iloc[0] / s.iloc[0] - 1; rb.iloc[0] = mb.iloc[0] / bb.iloc[0] - 1
    x = (rs - rb).dropna()
    return x, rs.dropna(), rb.dropna()


def tstat(x):
    x = np.asarray(x, float); n = len(x)
    return float(x.mean() / (x.std(ddof=1) / math.sqrt(n))), float(x.mean()), float(x.std(ddof=1)), n


def nw_t(x, lags=6):
    x = np.asarray(x, float); n = len(x); m = x.mean(); d = x - m
    g0 = (d * d).sum() / n; s = g0
    for l in range(1, lags + 1):
        g = (d[l:] * d[:-l]).sum() / n
        s += 2 * (1 - l / (lags + 1)) * g
    return float(m / math.sqrt(max(s, 1e-18) / n))


def maxdd(nav):
    nav = np.asarray(nav, float); pk = np.maximum.accumulate(nav)
    return float((nav / pk - 1).min())
