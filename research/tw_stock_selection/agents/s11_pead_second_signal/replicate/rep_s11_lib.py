# -*- coding: utf-8 -*-
"""S11（季報驚喜 PEAD 當第二訊號）獨立複製：自寫函式庫。
只依 s11_pead_second_signal/PREREG.json、data_check.json 與 FINDINGS.md 第 1、2 節的文字規格。
未讀、未 import 研究員的 s11lib／s10lib／s9lib／s6lib／s5lib 及其 run／prep／posthoc／s0_datacheck；
未 import o_pead 的程式（只讀它的資料檔 fin_long.parquet、avail_dates.parquet）。
import 的現成程式：s10_exit_details/replicate/rep_s10_lib.py（S10 複製員）→ rep_s9 → rep_s6 → … → rep_s1。

本檔自己寫的部分：
  1. load_fin：單季 EPS（eps_s）、單季營收（rev_s）→ 季索引 qi＝y*4+q−1 × 面板代號的矩陣；
     金融保險＝MOPS 財報格式 industry ≠ 'ci'（bd、basi、ins、fh、mim；逐季用該季的格式，Q4 沿用同年 Q3 的格式）；
     市場可得日 A（avail_dates.parquet 的 A 欄、A_method 欄）。
  2. sue：(X_q − X_{q−4}) ÷ 前 8 季（q−1～q−8 的年增金額，至少 6 個有值）標準差（ddof=1），sd ≤ 0 → 缺值。
     （q−1～q−8 不含當季：這樣第一個可用季才是 data_check 的 2015Q3。）
  3. quarter_sched：每季法定期限（Q1 5/15、Q2 8/14、Q3 11/14、Q4 次年 3/31），k＝第一個 ≥ 期限的交易日，
     訊號列 s0＝k＋1；位移 sh：s＝s0＋sh（股池、A、均線、漲幅都在 s 計）。
  4. eligible／scores：上市股池（rep_s2 universe：60 日均成交值 ≥ 1,000 萬、收盤 ≥ 10、有成交、近 60 根有資料、上市）
     且非創新板、非金融、SUE（與 REVSUE，QR 不要求）有值、A ≤ 訊號列日期（A 缺＝法定期限）。
     pct＝平均名次 ÷ n；SUE⊥＝OLS pct(SUE)＝a＋b·pct(REVSUE) 的殘差。排序：分數由大到小、同分依代號。
  5. q_plans：Q／QR：buy＝keep＝前 10；QF：還原收盤 > 60 日均線者依 SUE⊥ 前 10＝keep，其中 RET20 > 10% 的＝buy。
     可選 < 50 檔的季不換股（不產生 buy／keep）。
"""
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, "/home/user/-/research/tw_stock_selection/agents/s10_exit_details/replicate")
import rep_s10_lib as K10  # noqa: E402
K9 = K10.K9
C, S, Q, R, L = K10.C, K10.S, K10.Q, K10.R, K10.L
ST, X2 = K10.ST, K10.X2
simulate = K9.simulate_s9
adj_open = K10.adj_open

PEAD = "/home/user/-/research/tw_stock_selection/agents/o_pead_margin_surprise_low_attention/data"
DEADLINE = {1: (0, 5, 15), 2: (0, 8, 14), 3: (0, 11, 14), 4: (1, 3, 31)}
Q_FIRST = 2015 * 4 + 0      # 2015Q1
Q_LAST = 2026 * 4 + 1       # 2026Q2


def load_fin(P):
    d = pd.read_parquet(PEAD + "/fin_long.parquet")
    d = d[d.code.isin(P.cidx)]
    qi = (d.y.values * 4 + d.q.values - 1).astype(int)
    q0 = int(qi.min()); nq = Q_LAST - q0 + 1
    keep = qi <= Q_LAST
    d = d[keep]; qi = qi[keep]
    col = d.code.map(P.cidx).values
    EPS = np.full((nq, P.N), np.nan); REV = np.full((nq, P.N), np.nan)
    FIN = np.zeros((nq, P.N), bool); HAS = np.zeros((nq, P.N), bool)
    EPS[qi - q0, col] = d.eps_s.values
    REV[qi - q0, col] = d.rev_s.values
    FIN[qi - q0, col] = (d.industry.values != "ci")
    # Q4 是「全年減前三季」推得的單季，格式沿用同年 Q3 那一列（有的話）。
    # 這條是為了對上 data_check 2016Q4 的非金融家數（2514 的 2016Q3 是 mim、2016Q4 是 ci；只影響這一格）。
    HAS[qi - q0, col] = True
    q4 = np.arange(nq)[(np.arange(nq) + q0) % 4 == 3]
    for jq in q4:
        if jq - 1 >= 0:
            m3 = HAS[jq - 1]
            FIN[jq, m3] = FIN[jq - 1, m3]
    a = pd.read_parquet(PEAD + "/avail_dates.parquet")
    a = a[a.code.isin(P.cidx)]
    aq = (a.y.values * 4 + a.q.values - 1).astype(int)
    ok = (aq >= q0) & (aq <= Q_LAST)
    a = a[ok]; aq = aq[ok]
    Astr = np.full((nq, P.N), "", dtype=object)
    Ameth = np.full((nq, P.N), "", dtype=object)
    Astr[aq - q0, a.code.map(P.cidx).values] = a.A.dt.strftime("%Y-%m-%d").values
    Ameth[aq - q0, a.code.map(P.cidx).values] = a.A_method.values
    # 公司每季的格式缺時（例如該季沒有列），沿用同代號最近一季的格式
    FINf = pd.DataFrame(np.where(HAS, FIN.astype(float), np.nan)).ffill().bfill().values > 0.5
    return dict(q0=q0, nq=nq, EPS=EPS, REV=REV, FIN=FINf, HAS=HAS, A=Astr, Ameth=Ameth)


def sue(X, nwin=8, nmin=6):
    nq, N = X.shape
    D = np.full((nq, N), np.nan)
    D[4:] = X[4:] - X[:-4]
    out = np.full((nq, N), np.nan)
    for q in range(4 + nmin, nq):
        w = D[max(0, q - nwin):q]
        n = np.isfinite(w).sum(axis=0)
        with np.errstate(invalid="ignore", divide="ignore"):
            sd = _nanstd1(w)
            ok = (n >= nmin) & np.isfinite(D[q]) & (sd > 0)
            out[q] = np.where(ok, D[q] / sd, np.nan)
    # 前 nwin 季之前（q < nwin+4）不可能有 nmin 個年增，保持缺值
    return out, D


def _nanstd1(w):
    n = np.isfinite(w).sum(axis=0).astype(float)
    m = np.where(n > 0, np.nansum(w, axis=0) / np.maximum(n, 1), np.nan)
    ss = np.nansum((w - m) ** 2, axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(n >= 2, np.sqrt(ss / (n - 1)), np.nan)


def deadline_str(qi):
    y, q = divmod(qi, 4); q += 1
    dy, mo, dd = DEADLINE[q]
    return "%04d-%02d-%02d" % (y + dy, mo, dd)


def quarter_sched(P, F):
    """list of dict(qi, label, deadline, k, s0)。"""
    da = np.array(P.dstr)
    out = []
    for qi in range(Q_FIRST, Q_LAST + 1):
        dl = deadline_str(qi)
        k = int(np.searchsorted(da, dl, side="left"))
        s0 = k + 1
        if s0 >= P.T - 1:
            continue
        y, q = divmod(qi, 4)
        out.append(dict(qi=qi, label="%dQ%d" % (y, q + 1), deadline=dl, k=k, s0=s0))
    return out


def pct_avg(x):
    return pd.Series(x).rank(method="average").values / len(x)


def resid(y, x):
    X = np.column_stack([np.ones(len(x)), x])
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    return y - X @ b, b


def eligible(P, F, SUE, RSUE, e, s, excl, need_rev=True, detail=False):
    """訊號列 s、季 e 的可選股票布林陣列。"""
    j = e["qi"] - F["q0"]
    u = R.universe(P, s)
    if excl is not None:
        u = u & ~excl(s)
    nonfin = u & ~F["FIN"][j]
    has = nonfin & np.isfinite(SUE[j])
    if need_rev:
        has = has & np.isfinite(RSUE[j])
    A = F["A"][j]
    dl = e["deadline"]; ds = P.dstr[s]
    Aeff = np.where(A == "", dl, A)
    on_time = np.array([a <= ds for a in Aeff])
    el = has & on_time
    if detail:
        return el, dict(univ_twse=int(u.sum()), univ_nonfin=int(nonfin.sum()),
                        with_sue=int((nonfin & np.isfinite(SUE[j])).sum()),
                        late_excluded=int((nonfin & np.isfinite(SUE[j]) & ~on_time).sum()), elig=int(el.sum()))
    return el


def scores(P, F, SUE, RSUE, e, el, kind):
    """kind：'perp'（SUE⊥）或 'raw'（pct(SUE)）。回傳 (idx 依分數排序, 分數陣列對齊 idx, OLS 係數)。"""
    j = e["qi"] - F["q0"]
    idx = np.where(el)[0]
    ps = pct_avg(SUE[j][idx])
    b = None
    if kind == "perp":
        pr = pct_avg(RSUE[j][idx])
        sc, b = resid(ps, pr)
    else:
        sc = ps
    codes = np.array(P.codes)
    o = np.lexsort((codes[idx], -sc))
    return idx[o], sc[o], b


def q_plans(P, F, SUE, RSUE, sched, cell, shift=0, excl=None, min_n=50, ma60=None, x=0.10, nmax=10, info=None):
    """回傳 (buy, keep)。cell ∈ Q、QF、QR。"""
    buy = {}; keep = {}
    for e in sched:
        s = e["s0"] + shift
        if s >= P.T - 1 or e["s0"] < P.t0:
            continue
        need_rev = cell != "QR"
        el = eligible(P, F, SUE, RSUE, e, s, excl, need_rev=need_rev)
        if int(el.sum()) < min_n:
            continue
        order, sc, b = scores(P, F, SUE, RSUE, e, el, "raw" if cell == "QR" else "perp")
        if cell in ("Q", "QR"):
            k = order[:nmax]; bl = k
        elif cell == "QF":
            with np.errstate(invalid="ignore", divide="ignore"):
                above = P.G[s, order] > ma60[s, order]
                r20 = P.G[s] / P.G[s - 20] - 1.0
            k = order[above][:nmax]
            bl = np.array([i for i in k if r20[i] > x], dtype=int)
        else:
            raise ValueError(cell)
        buy[s] = np.asarray(bl, dtype=int); keep[s] = set(int(i) for i in k)
        if info is not None:
            info.append(dict(label=e["label"], s=s, date=P.dstr[s], n_elig=int(el.sum()), ols_b=None if b is None else
                             [float(b[0]), float(b[1])], top=[P.codes[i] for i in k], buy=[P.codes[i] for i in bl]))
    return buy, keep


def adj_open_matrix(P):
    """還原開盤矩陣（同 rep_s10_lib.adj_open 的定義，向量化）：G×開盤÷收盤；沒有開盤用最後一個有值的 G；下市後＝最後 G。
    另回傳 Gf（G 往後補值；下市後＝最後 G）。"""
    Gf = pd.DataFrame(P.G).ffill().values
    with np.errstate(invalid="ignore", divide="ignore"):
        v = P.G * P.O / P.Cf
    ok = np.isfinite(v) & (P.Cf > 0)
    tt = np.arange(P.T)[:, None]
    ok &= tt <= P.last[None, :]
    AO = np.where(ok, v, Gf)
    return AO, Gf


def union_plans(bE, kE, bQ, kQ):
    """審查 C 的名單聯集：每個 E0 或 Q 訊號列都產生 buy／keep；
    keep＝當列（或最近一次）E0 續抱集合 ∪ 當列（或最近一次）Q 續抱集合；buy＝當列 E0 名單＋Q 名單（去重，E0 在前）。"""
    rows = sorted(set(bE) | set(bQ))
    buy = {}; keep = {}
    curE = set(); curQ = set(); both = 0
    for s in rows:
        lst = []
        if s in kE:
            curE = kE[s]; lst += [int(i) for i in bE[s]]
        if s in kQ:
            curQ = kQ[s]; lst += [int(i) for i in bQ[s] if int(i) not in lst]
        if s in kE and s in kQ:
            both += 1
        buy[s] = np.array(lst, dtype=int); keep[s] = set(curE) | set(curQ)
    return buy, keep, both
