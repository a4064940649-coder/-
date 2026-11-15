# -*- coding: utf-8 -*-
"""S12（月營收驚奇 SURGE 市值分層）獨立複製：自寫函式庫。
只依 s12_surge_size_tiers/PREREG.json、data_check.json 與 FINDINGS.md 第 1、2 節的文字規格。
未讀、未 import 研究員的 s12lib／s10lib／s9lib／s6lib／s5lib／s4lib 及其 run／prep／posthoc／s0_datacheck；
未 import o_lc_sector_rotation_pit 的程式（只讀資料檔 data/shares_full.parquet；該檔 2013 年起段已含 4763、2317 修正，
複製員以 2018-07～10 的 2317、2021 的 4763 數值抽查確認）。
import 的現成程式：s11_pead_second_signal/replicate/rep_s11_lib.py（S11 複製員）→ rep_s10 → rep_s9 → rep_s6 → … → rep_s1。

本檔自己寫的部分：
  1. load_shares：股數表 2013 年起段，依面板日期對齊（向前補值最多 10 天）、依面板代號對齊（沒有的代號＝缺）。
  2. mcap(s)：MC[s]＝股數[s−63] × 收盤[s−63] × G[s] ÷ G[s−63]（列號差 63 個交易日；收盤用沿用前值的 Cf）；
     落後值不能算（s−63 在 K 棒之前、股數缺）才用 股數[s] × 收盤[s]；都缺＝NaN。
  3. tiers(s, by)：在 s 的現行股池（rep_s2 universe 且非創新板）內依 MC（by='mc'）或 60 日均成交值（by='adv'）
     由大到小排名，同值依代號；沒有值的排最後。回傳長度 N 的層別陣列：0＝不在股池、1＝大（1～100）、2＝中（101～300）、3＝小。
  4. tier_plans：由 rep_s3 make_lists 的 L_MA（已在位移後那天算股池、季線、RET20）取出屬於指定層集合的股票（保留 SURGE 順序）；
     buy＝層內 L_MA 前 depth 名中 RET20 > 10%；keep＝{股池內、SURGE 有值、站上季線、SURGE ≥ 層內 L_MA 第 20 名分數（不足 20 不設門檻）}。
     tiers_set=None ＝ 全股池（keep＝L_MA 前 20 名，與 rep_s3 C10 完全相同）。
  5. simulate_s12：rep_s10_lib.SIM10_SRC 字串替換（每個替換點斷言恰好命中 1 次），只多 ntgt（每檔目標的除數；None＝原本）
     與「依名次掃完整份買進名單（最多 10 檔）、名額 nmax」：nmax＝10 時與原引擎逐行相同。
  6. ew_windows：同層等權／名單等權／0050 的訊號時窗報酬（扣成本前）。
"""
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, "/home/user/-/research/tw_stock_selection/agents/s11_pead_second_signal/replicate")
import rep_s11_lib as K11  # noqa: E402
K10 = K11.K10
K9 = K10.K9
C, S, Q, R, L = K10.C, K10.S, K10.Q, K10.R, K10.L
ST, X2 = K10.ST, K10.X2
adj_open_matrix = K11.adj_open_matrix

SHARES = "/home/user/-/research/tw_stock_selection/agents/o_lc_sector_rotation_pit/data/shares_full.parquet"
LAG = 63
LISTMAX = 10

# ------------------------------------------------------------------ 引擎
_SRC = K10.SIM10_SRC
_REPL = [
    ("def simulate_s10(", "def simulate_s12(", 1),
    ("rec_pos=False, eod=None):", "rec_pos=False, eod=None, ntgt=None):", 1),
    ('div = max(n_list, 5) if sizing == "WN" else nmax',
     'div = ntgt if ntgt is not None else (max(n_list, 5) if sizing == "WN" else nmax)', 1),
    ("for rk, i in enumerate(buy[t - 1][:nmax]):", "for rk, i in enumerate(buy[t - 1][:LISTMAX]):", 1),
]
for a, b, n in _REPL:
    k = _SRC.count(a)
    assert k == n, (a, k, n)
    _SRC = _SRC.replace(a, b)
_ns = dict(vars(K10))
_ns["LISTMAX"] = LISTMAX
exec(compile(_SRC, "<rep_s12_lib.simulate_s12>", "exec"), _ns)
simulate_s12 = _ns["simulate_s12"]


# ------------------------------------------------------------------ 市值
def load_shares(P):
    d = pd.read_parquet(SHARES)
    d = d[d.index >= pd.Timestamp("2013-01-01")]
    cols = [c for c in d.columns if c in P.cidx]
    d = d[cols]
    idx = pd.DatetimeIndex(P.dates)
    d = d.reindex(d.index.union(idx)).sort_index().ffill(limit=10).reindex(idx)
    SH = np.full((P.T, P.N), np.nan, dtype=np.float64)
    SH[:, [P.cidx[c] for c in cols]] = d.values.astype(np.float64)
    P.SH = SH
    P.sh_cols = len(cols)
    return SH


def mcap(P, s, detail=False):
    lag = s - LAG
    with np.errstate(invalid="ignore", divide="ignore"):
        if lag >= 0:
            m_lag = P.SH[lag] * P.Cf[lag] * P.G[s] / P.G[lag]
        else:
            m_lag = np.full(P.N, np.nan)
        m_now = P.SH[s] * P.Cf[s]
    okl = np.isfinite(m_lag) & (m_lag > 0)
    mc = np.where(okl, m_lag, np.where(np.isfinite(m_now) & (m_now > 0), m_now, np.nan))
    if detail:
        return mc, okl, m_now
    return mc


def rank_tiers(P, pool, val, nL=100, nM=300):
    codes = np.array(P.codes)
    idx = np.where(pool)[0]
    v = val[idx]
    has = np.isfinite(v)
    key = np.where(has, -v, np.inf)
    o = idx[np.lexsort((codes[idx], key))]
    tier = np.zeros(P.N, np.int8)
    tier[o[:nL]] = 1
    tier[o[nL:nM]] = 2
    tier[o[nM:]] = 3
    return tier, o


def tiers(P, s, excl, by="mc"):
    pool = R.universe(P, s)
    if excl is not None:
        pool = pool & ~excl(s)
    if by == "mc":
        val = mcap(P, s)
    elif by == "adv":
        val = P.adv60[s]
    elif by == "mc_now":
        with np.errstate(invalid="ignore"):
            val = np.where(P.SH[s] * P.Cf[s] > 0, P.SH[s] * P.Cf[s], np.nan)
    else:
        raise ValueError(by)
    t, _ = rank_tiers(P, pool, val)
    return t


def tier_plans(P, S_, lists, tier_of, tiers_set, excl, depth=10, x=0.10, keep_n=20, ma60=None, check=None):
    """tier_of：{s: 層別陣列}；tiers_set：例如 {1}、{2,3}；None＝全股池。
    回傳 (buy, keep, lma_t)；lma_t[s]＝層內 L_MA（SURGE 順序）。check（dict）不為 None 時做驗收 A2。"""
    buy = {}; keep = {}; lmat = {}
    for s, q in lists.items():
        lma = q["lma"]; r20 = q["r20"]
        if tiers_set is None:
            lt = lma
        else:
            tt = tier_of[s]
            lt = np.array([i for i in lma if tt[i] in tiers_set], dtype=int)
        top = lt[:depth]
        with np.errstate(invalid="ignore"):
            b = np.array([i for i in top if r20[i] > x], dtype=int)
        if tiers_set is None:
            k = set(int(i) for i in lma[:keep_n])
        else:
            sv = S_[q["m"]]
            u = R.universe(P, s)
            if excl is not None:
                u = u & ~excl(s)
            thr = sv[lt[keep_n - 1]] if len(lt) >= keep_n else -np.inf
            with np.errstate(invalid="ignore"):
                ok = u & np.isfinite(sv) & (P.G[s] > ma60[s]) & (sv >= thr)
            k = set(int(i) for i in np.where(ok)[0])
            if check is not None:
                tt = tier_of[s]
                inT = set(int(i) for i in np.where(np.isin(tt, list(tiers_set)))[0])
                a = k & inT; bb = set(int(i) for i in lt[:keep_n])
                check["rows"] = check.get("rows", 0) + 1
                check["diff_cells"] = check.get("diff_cells", 0) + len(a ^ bb)
                check["buy_not_in_tier"] = check.get("buy_not_in_tier", 0) + int(sum(int(i) not in inT for i in b))
        buy[s] = b; keep[s] = k; lmat[s] = lt
    return buy, keep, lmat


# ------------------------------------------------------------------ 時窗報酬（扣成本前）
def windows(P, lists):
    rows = sorted(lists)
    out = []
    for n, s in enumerate(rows):
        e = s + 1
        e2 = rows[n + 1] + 1 if n + 1 < len(rows) else None
        out.append((s, e, e2))
    return out


def win_ret(P, AO, Gf, idx, e, e2):
    if len(idx) == 0:
        return np.nan
    a = AO[e, idx]
    b = AO[e2, idx] if e2 is not None else Gf[P.T - 1, idx]
    return float(np.nanmean(b / a - 1.0))


def e50_win(P, e, e2):
    b = P.e50o[e2] if e2 is not None else P.e50c[P.T - 1]
    return float(b / P.e50o[e] - 1.0)
