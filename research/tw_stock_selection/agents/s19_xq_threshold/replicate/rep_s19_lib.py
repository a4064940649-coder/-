# -*- coding: utf-8 -*-
"""S19（XQ 門檻版回測）獨立複製：自寫函式庫。
只依 s19_xq_threshold/PREREG.json、data_check.json、FINDINGS.md 第 1、2 節，以及自己逐行讀的 xq_scripts/20_*.xs、21_*.xs。
未讀、未 import 研究員的 s19lib／s10lib／s9lib 及其 run／prep／posthoc。
import 的現成程式（都是之前「複製員」寫的）：
  s17_surge_definition_family/replicate/rep_s17_lib.py（STD SURGE、σ≈0 當缺值；→ rep_s14_lib：0050 停牌修正、引擎
  simulate_s14 ＝ rep_s10 引擎＋listmax）→ rep_s10 → rep_s9（B11opt 訊號日 schedule、make_lists_sched）→ … → rep_s1。

本檔自己寫的部分：
  1. simulate_s19：rep_s14_lib._SRC 字串替換（每個替換點斷言命中次數），只多：
       slot_max（持股上限；None＝nmax，與原引擎相同；審查 B 用 9999＝沒有上限）、
       fixed_amt（每檔目標金額固定；None＝衛星資金 ÷ 10，與原引擎相同；審查 B 用起始資金 ÷ 10）、
       skiplog（每個沒買成的候選 (t, i, 原因)：no_slot／limit_up／no_open／blk_min／zero）。
     listmax 沿用 rep_s14（None＝只掃買進名單前 nmax 個，與原引擎相同；門檻版用 9999＝掃完整份候選、買到 10 檔滿）。
  2. script_panel：腳本股池用的還原價 Cadj＝G × 最後收盤 ÷ G[最後]（D4），
     還原成交值 60 日平均＝Cadj × 成交股數（無成交日 0、K 棒範圍內滿 60 列；同研究版 adv60 的算法，只把官方成交金額換成 Cadj×股數）（D5）。
     另做「XQ K 棒版」（審查 C）：只用有成交的日子當 K 棒，60 根平均、60 根均線、20 根前收盤。
  3. make_thr：門檻版訊號（B11opt 訊號列、研究版月份集合、位移在訊號列上往後移）。
       buy[s]＝股池 & SURGE 有值 & SURGE ≥ MinSurge & G > 60 日均線 & G[s]/G[s−20]−1 > 10%，依 SURGE 由大到小（同分依代號）。
       keep[s]（續抱）：腳本版＝SURGE 有值 & SURGE ≥ ExitSurge & 不是（均線有值且 G ≤ 均線）——不看股池（D3）；
                        XU＝研究股池 & SURGE ≥ ExitSurge & G > 均線（離開研究股池也賣）。
  4. make_eod_keep：與引擎「if t in keep: pend = i not in keep」相同，只多記 why＝sig、mk＝標記列（驗收 A3 用）。
"""
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, "/home/user/-/research/tw_stock_selection/agents/s17_surge_definition_family/replicate")
import rep_s17_lib as K17  # noqa: E402
K14 = K17.K14
K10 = K14.K10
K9 = K10.K9
S, Q, R, L = K14.S, K14.Q, K14.R, K14.L
ST, X2 = K14.ST, K14.X2

_SRC = K14._SRC
_REPL = [
    ("def simulate_s14(", "def simulate_s19(", 1),
    ("listmax=None, max_sh=None", "listmax=None, max_sh=None, slot_max=None, fixed_amt=None", 1),
    ("limup_list = []", "limup_list = []\n    skiplog = []", 1),
    ("u = sat_cap / div", "u = (sat_cap / div) if fixed_amt is None else float(fixed_amt)", 1),
    ("if len(pos) >= nmax:", "if len(pos) >= (nmax if slot_max is None else slot_max):", 1),
    ('cnt["no_slot"] += 1; continue', 'cnt["no_slot"] += 1; skiplog.append((t, i, "no_slot")); continue', 1),
    ('cnt["limit_up"] += 1; limup_list', 'skiplog.append((t, i, "limit_up")); cnt["limit_up"] += 1; limup_list', 1),
    ('cnt["no_open"] += 1; continue', 'cnt["no_open"] += 1; skiplog.append((t, i, "no_open")); continue', 2),
    ('cnt["blk_min"] += 1', 'skiplog.append((t, i, "blk_min")); cnt["blk_min"] += 1', 1),
    ('                    if min_frac > 0:\n                        cnt["blk_min_or_zero"] += 1',
     '                    skiplog.append((t, i, "zero"))\n                    if min_frac > 0:\n'
     '                        cnt["blk_min_or_zero"] += 1', 1),
    ("limup=limup_list,", "limup=limup_list, skiplog=skiplog,", 1),
]
for a, b, n in _REPL:
    k = _SRC.count(a)
    assert k == n, (a, k, n)
    _SRC = _SRC.replace(a, b)
_ns = dict(vars(K10))
exec(compile(_SRC, "<rep_s19_lib.simulate_s19>", "exec"), _ns)
simulate_s19 = _ns["simulate_s19"]


# ------------------------------------------------------------------ 腳本股池（D4、D5）
def script_panel(P):
    T, N = P.T, P.N
    last = P.last
    gl = np.array([P.G[last[i], i] if last[i] >= 0 else np.nan for i in range(N)])
    cl = np.array([P.Cf[last[i], i] if last[i] >= 0 else np.nan for i in range(N)])
    with np.errstate(invalid="ignore", divide="ignore"):
        P.Cadj = P.G * (cl / gl)[None, :]
    val = np.where(P.inbar, np.where(P.traded, P.Cadj * np.nan_to_num(P.Vb), 0.0), np.nan)
    a = pd.DataFrame(val).rolling(60, min_periods=60).mean().values
    P.adv_adj = np.where(P.inbar & (P.barno >= 60), a, np.nan)
    del val, a


def script_panel_bars(P):
    """XQ K 棒版（審查 C）：K 棒＝有成交的日子。回傳 (adv_bar, ma_bar, r20_bar)，只在有成交的列有值。"""
    T, N = P.T, P.N
    adv = np.full((T, N), np.nan, np.float32); mab = np.full((T, N), np.nan, np.float32)
    r20 = np.full((T, N), np.nan, np.float32)
    for i in range(N):
        rows = np.where(P.traded[:, i])[0]
        if len(rows) < 21:
            continue
        c = P.Cadj[rows, i]; v = np.nan_to_num(P.Vb[rows, i])
        x = c * v
        cs = np.concatenate([[0.0], np.cumsum(x)]); cc = np.concatenate([[0.0], np.cumsum(c)])
        n = len(rows)
        k = np.arange(n)
        ok60 = k >= 59
        av = np.full(n, np.nan); mv = np.full(n, np.nan)
        av[ok60] = (cs[k[ok60] + 1] - cs[k[ok60] - 59]) / 60.0
        mv[ok60] = (cc[k[ok60] + 1] - cc[k[ok60] - 59]) / 60.0
        rr = np.full(n, np.nan); rr[20:] = c[20:] / c[:-20] - 1.0
        adv[rows, i] = av; mab[rows, i] = mv; r20[rows, i] = rr
    return adv, mab, r20


def script_universe(P, s, excl):
    with np.errstate(invalid="ignore"):
        return P.is_twse[s] & ~excl(s) & P.traded[s] & (P.Cadj[s] >= 10.0) & (P.adv_adj[s] >= 1e7)


def research_universe(P, s, excl):
    return R.universe(P, s) & ~excl(s)


# ------------------------------------------------------------------ 門檻版訊號
def make_thr(P, S_, sched, excl, mav, univ, exit_univ, thr_fn, shift=0, min_n=50, ret_len=20, x=0.10):
    """回傳 (buy, keep, info)。thr_fn(s0, s) -> (MinSurge, ExitSurge)。univ：'res' 或 'scr'。"""
    buy = {}; keep = {}; info = {}
    codes = np.array(P.codes)
    for e_ in sched:
        s0 = e_["k"]; s = s0 + shift
        if s >= P.T - 1 or s0 < P.t0 or s0 >= P.T - 1:
            continue
        sv = S_[e_["m_rev"]]
        u0 = research_universe(P, s0, excl)
        if int((u0 & np.isfinite(sv)).sum()) < min_n:
            continue
        ures = research_universe(P, s, excl)
        u = ures if univ == "res" else script_universe(P, s, excl)
        ms, xs = thr_fn(s0, s)
        G = P.G[s]; ma = mav[s]
        with np.errstate(invalid="ignore", divide="ignore"):
            above = G > ma
            r20 = G / P.G[s - ret_len] - 1.0
            fin = np.isfinite(sv)
            cand = u & fin & above & (sv >= ms) & (r20 > x)
            if exit_univ:
                kp = ures & fin & (sv >= xs) & above
            else:
                kp = fin & (sv >= xs) & ~(np.isfinite(ma) & (G <= ma))
        idx = np.where(cand)[0]
        order = idx[np.lexsort((codes[idx], -sv[idx]))]
        assert s not in buy
        buy[s] = order
        keep[s] = set(int(i) for i in np.where(kp)[0])
        info[s] = dict(s0=s0, m_rev=e_["m_rev"], ms=ms, xs=xs, n_cand=int(len(order)), n_univ=int(u.sum()))
    return buy, keep, info


def make_eod_keep(keep):
    def eod(t, pos):
        if t in keep:
            kp = keep[t]
            for i, p in pos.items():
                p["pend"] = i not in kp
                if p["pend"]:
                    p["why"] = "sig"; p["mk"] = t
    return eod


# ------------------------------------------------------------------ 統計
def jaccard(a, b):
    a = set(int(i) for i in a); b = set(int(i) for i in b)
    u = a | b
    if not u:
        return None
    return len(a & b) / len(u)


def tstat(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) < 2:
        return None
    return float(x.mean() / (x.std(ddof=1) / np.sqrt(len(x))))


def cluster_t(x, g):
    """混合平均的月份聚類 t：SE＝sqrt(G/(G−1) × Σ_g(Σ_i e_gi)²)/n，e＝x−平均。"""
    x = np.asarray(x, float); g = np.asarray(g)
    n = len(x); mu = x.mean(); e = x - mu
    df = pd.DataFrame(dict(g=g, e=e)).groupby("g").e.sum().values
    G = len(df)
    se = np.sqrt(G / (G - 1) * (df ** 2).sum()) / n
    return float(mu / se), G


def month_mean_t(x, g):
    df = pd.DataFrame(dict(g=g, x=np.asarray(x, float))).groupby("g").x.mean()
    return float(df.mean()), tstat(df.values), int(len(df))
