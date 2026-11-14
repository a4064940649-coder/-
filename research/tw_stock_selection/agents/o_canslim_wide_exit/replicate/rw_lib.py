# -*- coding: utf-8 -*-
"""獨立複製員自寫函式庫（o_canslim_wide_exit）。
依 FINDINGS.md 第 2 節文字規格從頭寫：逐部位出場、組合引擎、0050 精確換手、路徑擾動、安慰劑、統計。
沒有讀、沒有 import 研究員的 cwlib.py / run.py，也沒有 import o_ext_canslim_xs/cslib.py / run.py。
引擎沒有使用 rep_lib.run_exec。
"""
import math, os
import numpy as np

FEE = 0.001425 * 0.28          # 0.0399%
TAX = 0.003
MINFEE = 20.0
E50_TAX = 0.001


# ------------------------------------------------------------------ 價格檔位
def tick(p):
    if p < 10: return 0.01
    if p < 50: return 0.05
    if p < 100: return 0.1
    if p < 500: return 0.5
    if p < 1000: return 1.0
    return 5.0


def lim_up(ref):
    raw = ref * 1.1
    tk = tick(raw)
    return math.floor(round(raw / tk, 6)) * tk


def lim_dn(ref):
    raw = ref * 0.9
    tk = tick(raw)
    return math.ceil(round(raw / tk, 6)) * tk


# ------------------------------------------------------------------ 資料
class D:
    pass


def load(S):
    z = np.load(os.path.join(S, "prep.npz"))
    d = D()
    for k in z.files:
        setattr(d, k, z[k])
    d.dates = [str(x) for x in d.dates]
    d.codes = [str(x) for x in d.codes]
    d.T, d.N = d.Cf.shape
    d.t0 = int(d.t0); d.t_odd = int(d.t_odd); d.t_ext = int(d.t_ext)
    d.dnum = np.array(d.dates, dtype="datetime64[D]")
    return d


def cands_from(sig, key, t_from=0):
    """每個訊號日的候選清單：排序鍵由大到小、同分依代號（欄序）由小到大。"""
    T = sig.shape[0]
    out = [None] * T
    for t in np.flatnonzero(sig.any(axis=1)):
        if t < t_from: continue
        ix = np.flatnonzero(sig[t])
        o = np.argsort(-key[t, ix], kind="stable")
        out[t] = ix[o].tolist()
    return out


class Pos:
    __slots__ = ("i", "tb", "sh", "val", "cost", "recv", "pend", "why", "g", "bo", "ttrig", "new")


# ------------------------------------------------------------------ 組合引擎
def simulate(d, cands, exit_static=None, why_parts=None, fstop=None, hold=None, capital=1_000_000.0, nmax=10,
             cost="normal", idle="cash", lot_rule=True, same_day=True, t_start=None, liq=0.05, e50_lot=False,
             ban=None, record=True, min_frac=None):
    """cands[t]＝訊號日 t 的候選（已排序）；t+1 開盤成交。
    exit_static：T×N bool（移動停損／均線下穿／50 日線的聯集）；why_parts：[(名稱, T×N bool)]。
    fstop：進場價固定停損 %（含息累積報酬）；hold：固定持有交易日數。
    idle：'cash' 或 'e50'（空位放 0050 精確換手）。e50_lot：審查用，0050 部位 2020-10-26 前只能整張。
    ban：審查用，set((i, 進場日 t))＝禁止的買進。"""
    T = d.T
    t0 = d.t0 if t_start is None else t_start
    O, Cf, REF, RET, AMT, LAST = d.O, d.Cf, d.ref, d.ret, d.AMT, d.last
    zero = cost == "zero"
    fee = 0.0 if zero else FEE
    tax = 0.0 if zero else TAX
    minfee = 0.0 if zero else MINFEE
    extra = 0.003 if cost == "stress" else 0.0
    t_odd = d.t_odd
    ov, intr, slip50 = d.ov, d.intr, d.slip50
    e50 = idle == "e50"
    cash = float(capital)          # 閒置資金（cash 版＝現金；e50 版＝0050 部位）
    cash2 = 0.0                    # 只在 e50_lot：買不滿一張 0050 的零頭現金
    n50 = 0
    if e50 and e50_lot:
        cash2 = cash; cash = 0.0   # 審查用：起始全現金，第一個交易日開盤才買 0050（整張）
    pos = []                       # 依建立先後
    nav = np.full(T, np.nan); nav[t0] = capital
    npos = np.zeros(T); idlew = np.zeros(T)
    trades = []
    cnt = dict(cand=0, buy=0, limit_up=0, no_open=0, lot_skip=0, no_cash=0, no_slot=0, sell_defer_limit=0, sell_defer_noopen=0,
               sell_split=0, delist=0, buy_amt=0.0, fee=0.0, tax=0.0, slip=0.0, e50_cost=0.0, liq_capped_buy=0)
    held_prev = set()
    fstop_lv = None if fstop is None else 1 - fstop / 100

    def legs_fee(sh, val):
        b = (sh // 1000) * 1000
        if b == sh or b == 0:
            f = fee * val
            return f if f > minfee else minfee
        vb = val * (b / sh)
        f1 = fee * vb; f2 = fee * (val - vb)
        return (f1 if f1 > minfee else minfee) + (f2 if f2 > minfee else minfee)

    for t in range(t0 + 1, T):
        if e50:
            if e50_lot and t < t_odd:
                e_tr = cash * (1 + ov[t])
                e_px = n50 * d.o50[t]
                cash2 += e_tr - e_px          # 配息（與兩個資料源的捨入差）進現金
                cash = e_px
            else:
                cash *= (1 + ov[t])
            I0 = cash
            if zero:
                c_s = 0.0; reserve = 0.0
            else:
                c_s = FEE + E50_TAX + slip50[t] + extra
                reserve = I0 - max(I0 - 20.0, 0.0) * (1 - c_s)
        # ---------------- 開盤賣出（依部位建立先後）
        if pos:
            keep = []
            for p in pos:
                if not p.pend:
                    keep.append(p); continue
                i = p.i
                if t > LAST[i]:
                    cash += p.val; p.recv += p.val; cnt["delist"] += 1
                    if record: trades.append((i, p.tb, t, p.cost, p.recv, p.why + "|delist", p.sh, p.bo, p.ttrig))
                    continue
                o = O[t, i]
                if not (o == o):
                    cnt["sell_defer_noopen"] += 1; keep.append(p); continue
                ref = REF[t, i]
                if o <= lim_dn(ref) + 1e-9:
                    cnt["sell_defer_limit"] += 1; keep.append(p); continue
                gross_full = p.val * (o / ref)
                frac = 1.0
                if liq is not None:
                    cap = liq * AMT[t, i]
                    if gross_full > cap:
                        frac = cap / gross_full; cnt["sell_split"] += 1
                gross = gross_full * frac
                sl = 0.0 if zero else (0.5 * tick(o) / o + extra)
                g2 = gross * (1 - sl)
                fe = legs_fee(p.sh, g2) if not zero else 0.0
                tx = tax * g2
                net = g2 - fe - tx
                cnt["fee"] += fe; cnt["tax"] += tx; cnt["slip"] += gross - g2
                cash += net; p.recv += net
                if frac >= 1.0:
                    if record: trades.append((i, p.tb, t, p.cost, p.recv, p.why, p.sh, p.bo, p.ttrig))
                else:
                    p.val *= (1 - frac)
                    p.sh = max(1, int(round(p.sh * (1 - frac))))
                    keep.append(p)
            pos = keep
        # ---------------- 開盤估算總權益
        eq = cash + cash2
        for p in pos:
            i = p.i; o = O[t, i]
            eq += p.val * (o / REF[t, i]) if o == o else p.val
        # ---------------- 開盤買進
        cl = cands[t - 1]
        if cl:
            lot_only = lot_rule and t < t_odd
            unit = 1000 if lot_only else 1
            free = nmax - len(pos)
            if not same_day:
                free = min(free, nmax - len(held_prev))
            target = eq / nmax
            for i in cl:
                if i in held_prev: continue
                cnt["cand"] += 1
                if free <= 0:
                    cnt["no_slot"] += 1; continue
                o = O[t, i]
                if not (o == o):
                    cnt["no_open"] += 1; continue
                ref = REF[t, i]
                if o >= lim_up(ref) - 1e-9:
                    cnt["limit_up"] += 1; continue
                if ban is not None and (i, t) in ban:
                    continue
                avail = (cash - reserve) if e50 else cash
                if e50_lot: avail += cash2
                budget = target if target < avail else avail
                if min_frac is not None and budget < min_frac * target:      # 凍結後才加的審查用選項（預設關閉）
                    cnt["dust_skip"] = cnt.get("dust_skip", 0) + 1; continue
                if liq is not None:
                    cap = liq * AMT[t, i]
                    if budget > cap:
                        budget = cap; cnt["liq_capped_buy"] += 1
                sl = 0.0 if zero else (0.5 * tick(o) / o + extra)
                px = o * (1 + sl)
                sh = int(budget / px / unit) * unit
                while sh > 0:
                    v = sh * px
                    fe = legs_fee(sh, v) if not zero else 0.0
                    if v + fe <= budget + 1e-6: break
                    sh -= unit
                if sh <= 0:
                    cnt["lot_skip" if lot_only else "no_cash"] += 1; continue
                c = v + fe
                if e50_lot and cash2 > 0:
                    use = min(cash2, c); cash2 -= use; cash -= (c - use)
                else:
                    cash -= c
                cnt["buy"] += 1; cnt["buy_amt"] += sh * o; cnt["fee"] += fe; cnt["slip"] += sh * (px - o)
                p = Pos(); p.i = i; p.tb = t; p.sh = sh; p.val = sh * o; p.cost = c; p.recv = 0.0; p.pend = False
                p.why = ""; p.g = 1.0; p.bo = o; p.ttrig = -1; p.new = True
                pos.append(p); free -= 1
        # ---------------- 0050 換手與盤中報酬
        if e50:
            if not e50_lot or t >= t_odd:
                dlt = cash + cash2 - I0
                cash = cash + cash2; cash2 = 0.0
                if not zero and dlt != 0.0:
                    s5 = slip50[t] + extra
                    if dlt > 0:
                        c50 = dlt * s5 + max(MINFEE, dlt * FEE)
                    else:
                        g = -dlt / (1 - c_s)
                        c50 = g * (E50_TAX + s5) + max(MINFEE, g * FEE)
                    cash -= c50; cnt["e50_cost"] += c50
            else:
                # 審查用：0050 只能整張。cash＝0050 市值（可為負＝已動用），cash2＝零頭現金
                lv = 1000.0 * d.o50[t]
                tot = cash + cash2            # 買賣後閒置總額（0050 以開盤市值計）
                e_old = I0
                s5 = slip50[t] + extra
                # 目標張數：盡量放 0050，零頭留現金，且現金不得為負
                n = int(math.floor(tot / lv))
                while n >= 0:
                    e_new = n * lv
                    dd = e_new - e_old
                    if dd > 0:
                        c50 = dd * s5 + max(MINFEE, dd * FEE)
                    elif dd < 0:
                        c50 = -dd * (E50_TAX + s5) + max(MINFEE, -dd * FEE)
                    else:
                        c50 = 0.0
                    if zero: c50 = 0.0
                    if tot - e_new - c50 >= -1e-6 or n == 0: break
                    n -= 1
                cash = e_new; cash2 = tot - e_new - c50; cnt["e50_cost"] += c50; n50 = n * 1000
            cash *= (1 + intr[t])
        # ---------------- 收盤：市值滾動、出場檢查
        v = 0.0
        for p in pos:
            i = p.i
            if p.new:
                p.new = False
                p.val = p.sh * Cf[t, i]
                p.g = Cf[t, i] / p.bo
            else:
                r = RET[t, i]
                if r == r:
                    p.val *= (1 + r); p.g *= (1 + r)
            v += p.val
            if not p.pend:
                w = ""
                if t <= LAST[i]:
                    if exit_static is not None and exit_static[t, i]:
                        if why_parts:
                            w = "+".join(nm for nm, a in why_parts if a[t, i])
                        else:
                            w = "static"
                    if fstop_lv is not None and p.g <= fstop_lv:
                        w = (w + "+fstop") if w else "fstop"
                    if hold is not None and t - p.tb >= hold - 1:
                        w = (w + "+hold") if w else "hold"
                if not w and t >= LAST[i] and t < T - 1:
                    w = "lastbar"
                if w:
                    p.pend = True; p.why = w; p.ttrig = t
        nav[t] = cash + cash2 + v; npos[t] = len(pos); idlew[t] = (cash + cash2) / nav[t]
        held_prev = {p.i for p in pos}
    if record:
        for p in pos:
            trades.append((p.i, p.tb, T - 1, p.cost, p.recv + p.val, "open" + ("|" + p.why if p.why else ""), p.sh, p.bo, p.ttrig))
    return dict(nav=nav, trades=trades, cnt=cnt, npos=npos, idlew=idlew, t0=t0, n_open=len(pos))


# ------------------------------------------------------------------ 統計
def ann(d, x, i0, i1):
    days = int((d.dnum[i1] - d.dnum[i0]).astype(int))
    return (x[i1] / x[i0]) ** (365.25 / days) - 1


def month_ends(d, i0, i1):
    """(i0, i1] 內每個日曆月的最後一個交易日位置。"""
    ym = [s[:7] for s in d.dates[i0 + 1:i1 + 1]]
    ends = [i0 + 1 + k for k in range(len(ym)) if k == len(ym) - 1 or ym[k + 1] != ym[k]]
    return ends


def mret(d, x, i0, i1):
    e = month_ends(d, i0, i1)
    pts = np.array([i0] + e)
    v = np.asarray(x)[pts]
    return v[1:] / v[:-1] - 1, [d.dates[k][:7] for k in e]


def tstat(x):
    x = np.asarray(x, float); n = len(x)
    sd = x.std(ddof=1)
    return float(x.mean() / (sd / math.sqrt(n))), float(x.mean()), float(sd), n


def nw_t(x, lags=6):
    x = np.asarray(x, float); n = len(x); m = x.mean(); u = x - m
    s = (u * u).sum() / n
    for l in range(1, lags + 1):
        s += 2 * (1 - l / (lags + 1)) * (u[l:] * u[:-l]).sum() / n
    return float(m / math.sqrt(s / n))


def maxdd(x):
    x = np.asarray(x, float)
    return float((x / np.maximum.accumulate(x) - 1).min())


def last_before(d, date):
    return max(k for k, s in enumerate(d.dates) if s < date)


def window(d, nav, b, i0, i1):
    a_s = ann(d, nav, i0, i1); a_b = ann(d, b, i0, i1)
    rs, _ = mret(d, nav, i0, i1); rb, _ = mret(d, b, i0, i1)
    x = rs - rb
    t, m, sd, n = tstat(x)
    return dict(ann=float(a_s), ann_0050=float(a_b), excess_pp=float((a_s - a_b) * 100), t=t, nw6=nw_t(x), mean_m=m, sd_m=sd, n=n,
                maxdd=maxdd(nav[i0:i1 + 1]), maxdd_0050=maxdd(b[i0:i1 + 1]))


def drop_best3(d, nav, b, i0, i1):
    rs, ym = mret(d, nav, i0, i1); rb, _ = mret(d, b, i0, i1)
    yrs = sorted(set(s[:4] for s in ym))
    yr = np.array([s[:4] for s in ym])
    ex = {}
    for y in yrs:
        m = yr == y
        ex[y] = float(np.prod(1 + rs[m]) - np.prod(1 + rb[m]))
    best = sorted(yrs, key=lambda y: -ex[y])[:3]
    m = ~np.isin(yr, best)
    t, _, _, n = tstat((rs - rb)[m])
    return dict(t=t, n=n, best=best, year_excess_pp={y: ex[y] * 100 for y in yrs},
                year_ret={y: float(np.prod(1 + rs[yr == y]) - 1) for y in yrs},
                year_ret_0050={y: float(np.prod(1 + rb[yr == y]) - 1) for y in yrs})


def ols_beta(rs, rb):
    """rs = a + b*rb；回傳 b、a、a 的 OLS t、NW6 t。"""
    rs = np.asarray(rs); rb = np.asarray(rb); n = len(rs)
    X = np.column_stack([np.ones(n), rb])
    XtXi = np.linalg.inv(X.T @ X)
    co = XtXi @ X.T @ rs
    u = rs - X @ co
    s2 = (u @ u) / (n - 2)
    se = math.sqrt(s2 * XtXi[0, 0])
    # NW6（HAC）
    Z = X * u[:, None]
    Sm = Z.T @ Z
    for l in range(1, 7):
        w = 1 - l / 7
        G = Z[l:].T @ Z[:-l]
        Sm += w * (G + G.T)
    V = XtXi @ Sm @ XtXi
    return dict(beta=float(co[1]), alpha_m=float(co[0]), t_alpha=float(co[0] / se), nw6_alpha=float(co[0] / math.sqrt(V[0, 0])),
                corr=float(np.corrcoef(rs, rb)[0, 1]), resid_sd=float(math.sqrt(s2)))
