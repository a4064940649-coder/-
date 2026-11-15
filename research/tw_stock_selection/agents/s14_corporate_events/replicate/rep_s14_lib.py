# -*- coding: utf-8 -*-
"""S14（公司事件：處置股第一次處置當衛星）獨立複製：自寫函式庫。
只依 s14_corporate_events/PREREG.json、data_check.json 與 FINDINGS.md 第 2 節（事件規則）的文字規格。
未讀、未 import 研究員的 s14lib／s10lib／s9lib／s6lib／s5lib 及其 run／prep／posthoc／s0_datacheck／prereg_write。
處置名單：只讀 o_disposal_list_pit_audit_pkl18/data/DISP_OFFICIAL_v2.csv（核對 sha256），不 import 該資料夾任何程式。
import 的現成程式：s10_exit_details/replicate/rep_s10_lib.py（S10 複製員）→ rep_s9 → rep_s6 → … → rep_s1；
rep_s11_lib.adj_open_matrix（S11 複製員；還原開盤矩陣）。

本檔自己寫的部分：
  1. attach_e50_fixed：rep_s4 attach_e50 之後，修正 0050 官方日 K 缺值日（2025-06-11～17 分割停牌）：
     原本沿用 06-10 的開盤÷收盤比（＝拿 06-10 開盤當基準，舊價格）；改為停牌日還原開盤＝前一日還原收盤、
     原始開盤＝前一個有值的原始收盤（S13 複製員指出的問題）。
  2. load_disp：讀 DISP_OFFICIAL_v2（核對 sha256）。
  3. build_chains：事件鏈。起點＝kind＝regular 且 nth ∈ nth_set 的記錄（不在既有鏈中）；
     起日列 a＝第一個 ≥ start 的交易日；最後一天 L＝最後一個 ≤ end 的交易日；
     鏈進行中，同檔任何記錄（不分 kind／nth）只要公告日 ≤ 目前的最後一天（日期字串比較），就併入；
     新記錄的最後一天較晚就延長（可連續）。被併入的記錄不另當事件。公告日 > 目前最後一天的記錄結束這條鏈，
     再看它能不能當新起點。end 超過面板最後一天 → L＝None（持有到資料結束，不賣）。
     鏈在全部記錄上建立，與是否可買無關。
  4. disp_plans：位移 sh：訊號列 s＝a−1＋sh；可買＝s 在股池內（rep_s2 universe 且非創新板）；
     同列候選依 s 的近 20 日平均成交值（列 s−19～s，無成交記 0）由大到小、同值依代號；
     出場標記列 m＝L＋sh（在 m 收盤標記、m＋1 開盤賣）。
  5. simulate_s14：rep_s10_lib.SIM10_SRC 字串替換（每個替換點斷言恰好命中 1 次），只多：
     listmax（買進名單掃到第幾名；None＝nmax，與原引擎相同；處置事件用 9999＝掃完整份候選），
     max_sh（審查 B：單筆股數上限；設上限而被截的那筆不再套用最小買進金額門檻），
     交易記錄 why／mk 照 rep_s10。
  6. make_eod_disp：每日收盤，持股 (訊號列, 代號) 對應的出場標記列 ≤ t 就標記待賣（why＝end）。
"""
import sys, hashlib
import numpy as np
import pandas as pd

sys.path.insert(0, "/home/user/-/research/tw_stock_selection/agents/s10_exit_details/replicate")
sys.path.insert(0, "/home/user/-/research/tw_stock_selection/agents/s11_pead_second_signal/replicate")
import rep_s10_lib as K10  # noqa: E402
import rep_s11_lib as K11  # noqa: E402
K9 = K10.K9
C, S, Q, R, L = K10.C, K10.S, K10.Q, K10.R, K10.L
ST, X2 = K10.ST, K10.X2
adj_open_matrix = K11.adj_open_matrix

DISP = "/home/user/-/research/tw_stock_selection/agents/o_disposal_list_pit_audit_pkl18/data/DISP_OFFICIAL_v2.csv"
DISP_SHA = "8f97b304411c6d013635ccc8fad9976f21f0fb4c9f0704bb0ee1bf678f052326"


# ------------------------------------------------------------------ 0050 停牌日修正
def attach_e50_fixed(P, scratch):
    import json, os
    S.attach_e50(P, scratch)
    oc = json.load(open(os.path.join(scratch, "e50_open_close.json")))
    miss = np.array([d not in oc for d in P.dstr])
    c_raw = np.array([oc.get(d, (np.nan, np.nan))[1] for d in P.dstr], float)
    c_raw = pd.Series(c_raw).ffill().values
    P.e50o = np.array(P.e50o, float).copy(); P.e50raw_o = np.array(P.e50raw_o, float).copy()
    old_o = P.e50o.copy(); old_raw = P.e50raw_o.copy()
    fixed = []
    for t in np.where(miss)[0]:
        if t == 0:
            continue
        P.e50o[t] = P.e50c[t - 1]
        P.e50raw_o[t] = c_raw[t - 1]
        fixed.append(dict(date=P.dstr[t], e50o_old=float(old_o[t]), e50o_new=float(P.e50o[t]),
                          raw_open_old=float(old_raw[t]), raw_open_new=float(P.e50raw_o[t]),
                          e50c=float(P.e50c[t])))
    P.e50_fix = fixed
    return fixed


def attach_e50_orig(P, scratch):
    S.attach_e50(P, scratch)
    P.e50_fix = []


# ------------------------------------------------------------------ 處置名單與事件鏈
def load_disp():
    h = hashlib.sha256(open(DISP, "rb").read()).hexdigest()
    assert h == DISP_SHA, h
    d = pd.read_csv(DISP, dtype={"code": str})
    return d, h


def build_chains(P, d, nth_set=(1,), ext_kinds=None):
    """回傳 DataFrame：code、i、a（起日列）、k0＝a−1、L（最後一天列；None＝超過資料）、ann0、start0、end_final、
    n_merged、n_extend、ext_detail（list：每次併入的 (公告日, 當時最後一天日期, 新最後一天日期, kind, nth)）、pit_ok。
    ext_kinds：可併入的 kind 集合（None＝不限）。"""
    da = np.array(P.dstr)
    last_date = P.dstr[-1]
    dd = d[d.code.isin(P.cidx)].copy()
    dd = dd.sort_values(["code", "announce", "start", "ordr"])
    out = []
    for code, g in dd.groupby("code", sort=True):
        cur = None
        for r in g.itertuples(index=False):
            if cur is not None:
                lstr = cur["L_date"]
                ok_kind = ext_kinds is None or r.kind in ext_kinds
                if lstr is None or (r.announce <= lstr):
                    if ok_kind:
                        new_L = _last_le(da, r.end, last_date)
                        rec = (r.announce, lstr, new_L[1], r.kind, int(r.nth))
                        cur["merged"].append(rec)
                        if cur["L_date"] is not None and (new_L[1] is None or new_L[1] > cur["L_date"]):
                            cur["L"], cur["L_date"] = new_L
                            cur["n_extend"] += 1
                            cur["end_final"] = r.end
                    continue
                out.append(cur); cur = None
            if r.kind == "regular" and int(r.nth) in nth_set:
                a = int(np.searchsorted(da, r.start, side="left"))
                if a >= P.T or a < 1:
                    continue
                Lr = _last_le(da, r.end, last_date)
                cur = dict(code=code, i=P.cidx[code], a=a, k0=a - 1, L=Lr[0], L_date=Lr[1], ann0=r.announce,
                           start0=r.start, end0=r.end, end_final=r.end, nth0=int(r.nth), merged=[], n_extend=0,
                           ann_le_k0=bool(r.announce <= P.dstr[a - 1]))
        if cur is not None:
            out.append(cur)
    ch = pd.DataFrame(out)
    ch["n_merged"] = ch.merged.map(len)
    ch["extended"] = ch.n_extend > 0
    ch["pit_ok"] = ch.merged.map(lambda m: all(x[1] is None or x[0] <= x[1] for x in m))
    return ch


def _last_le(da, end, last_date):
    if end > last_date:
        return None, None
    j = int(np.searchsorted(da, end, side="right")) - 1
    return j, str(da[j])


def adv20(P):
    amt = np.nan_to_num(P.AMT, nan=0.0)
    cs = np.vstack([np.zeros((1, P.N)), np.cumsum(amt, axis=0)])
    return cs


def disp_plans(P, ch, shift, excl, cs, start_row=None, drop=None):
    """回傳 (buy, mark, info)。buy：{訊號列 s: 代號陣列（依近 20 日均成交值由大到小）}；
    mark：{(s, i): 出場標記列（None＝持有到資料結束）}；info：每條鏈的可買判斷。
    start_row：訊號列 < start_row 的不產生（預設 t0）；drop：函式 (s_row)->bool，True 就不產生（審查 C）。"""
    t0 = P.t0 if start_row is None else start_row
    codes = np.array(P.codes)
    cand = {}
    mark = {}
    info = []
    for r in ch.itertuples(index=False):
        s = int(r.k0) + shift
        if s < t0 or s >= P.T - 1:
            continue
        if drop is not None and drop(s):
            continue
        u = R.universe(P, s)
        if excl is not None:
            u = u & ~excl(s)
        el = bool(u[r.i])
        info.append((r.code, s, el))
        if not el:
            continue
        m = None if (r.L is None or pd.isna(r.L)) else int(r.L) + shift
        if m is not None and m >= P.T:
            m = None
        cand.setdefault(s, []).append(int(r.i))
        assert (s, int(r.i)) not in mark
        mark[(s, int(r.i))] = m
    buy = {}
    for s, ii in cand.items():
        ii = np.array(sorted(set(ii)), dtype=int)
        a20 = (cs[s + 1, ii] - cs[s - 19, ii]) / 20.0
        o = ii[np.lexsort((codes[ii], -a20))]
        buy[s] = o
    return buy, mark, info


def make_eod_disp(mark):
    def eod(t, pos):
        for i, p in pos.items():
            if p["pend"]:
                continue
            m = mark.get((p["sig"], i), -1)
            if m is None:
                continue
            if m == -1:
                raise KeyError((p["sig"], i))
            if t >= m:
                p["pend"] = True; p["why"] = "end"; p["mk"] = t
    return eod


# ------------------------------------------------------------------ 引擎
_SRC = K10.SIM10_SRC
_REPL = [
    ("def simulate_s10(", "def simulate_s14(", 1),
    ("rec_pos=False, eod=None):", "rec_pos=False, eod=None, listmax=None, max_sh=None):", 1),
    ("for rk, i in enumerate(buy[t - 1][:nmax]):",
     "for rk, i in enumerate(buy[t - 1][:(nmax if listmax is None else listmax)]):", 1),
    ("                if sh <= 0:\n                    cnt[\"lot_skip\" if lot_only else \"no_cash\"] += 1",
     "                capped = False\n"
     "                if max_sh is not None and sh > max_sh:\n"
     "                    sh = int(max_sh / unit) * unit; capped = True\n"
     "                    cnt[\"capped\"] = cnt.get(\"capped\", 0) + 1\n"
     "                if sh <= 0:\n                    cnt[\"lot_skip\" if lot_only else \"no_cash\"] += 1", 1),
    ("if min_frac > 0 and sh * o < min_frac * target:", "if min_frac > 0 and sh * o < min_frac * target and not capped:", 1),
]
for a, b, n in _REPL:
    k = _SRC.count(a)
    assert k == n, (a, k, n)
    _SRC = _SRC.replace(a, b)
_ns = dict(vars(K10))
exec(compile(_SRC, "<rep_s14_lib.simulate_s14>", "exec"), _ns)
simulate_s14 = _ns["simulate_s14"]


# ------------------------------------------------------------------ 統計
def tstat(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) < 2:
        return None
    return float(x.mean() / (x.std(ddof=1) / np.sqrt(len(x))))


def strict_net(gross_in, gross_out, o_in, o_out, cmult=1.0, extra=0.003):
    """事件層近似淨報酬（嚴成本）：買價×(1＋半跳動點＋0.3%)＋手續費；賣價×(1−半跳動點−0.3%)−手續費−稅。不計最低手續費。"""
    fee = S.FEE * cmult; tax = S.TAX * cmult
    hb = S.tick(o_in) / 2.0 / o_in * cmult + extra
    hs = S.tick(o_out) / 2.0 / o_out * cmult + extra
    buy = (1 + hb) * (1 + fee)
    sell = (1 - hs) * (1 - fee - tax)
    return gross_out / gross_in * sell / buy - 1.0
