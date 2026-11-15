# -*- coding: utf-8 -*-
"""S16（SURGE 股池加入上櫃）獨立複製：自寫函式庫。
只依 s16_surge_otc/PREREG.json、data_check.json 與 FINDINGS.md 第 1 節（資料檢查）的文字規格。
未讀、未 import 研究員的 s16lib／s10lib／s9lib 及其 run／posthoc／s0_datacheck／prereg_write。
import 的現成程式：s14_corporate_events/replicate/rep_s14_lib.py（S14 複製員：0050 停牌日修正 attach_e50_fixed、
引擎 simulate_s14＝rep_s10 引擎＋listmax）→ rep_s10 → rep_s9 → … → rep_s1；rep_s11_lib.adj_open_matrix。
價格面板：s1 複製員 rep_s1_panel.py（本來就逐日記錄 market：twse＝1、tpex＝2，上櫃已在面板內，不需改）。

本檔自己寫的部分：
  1. parse_mops：獨立解析公開資訊觀測站上櫃 107 年 10 月營收彙總表快取頁（核對 sha256），取代號、名稱、當月、上月、去年當月營收。
  2. load_rev_df：rep_s1_lib.load_rev 的同一邏輯（同代碼同月優先上市、營收 ≤ 0 當缺值），可另外併入補洞列；
     patch=None 時與 load_rev 逐格相同（驗收）。
  3. univ_mats：各格股池矩陣。共同＝rep_s2 universe 去掉「當日上市」（K 棒內、當日有成交、收盤 ≥ 10、60 日均成交值 ≥ 1,000 萬
     （無成交記 0、滿 60 根）、近 60 根都有資料列）且非創新板（rep_s4 innov_mask，當日名稱 -(KY)?創$）；
     CUR＝當日上市（M＝1）；OTC＝當日上櫃（M＝2）；ALL＝兩者；ALL_O20＝上市 ∪（上櫃且 60 日均成交值 ≥ 2,000 萬）。
     市場別看當日（M[s]）。代號 91xx 本來就不在面板。
  4. make_lists_u：rep_s3_lib.make_lists 的同一邏輯（L_MA＝股池內 SURGE 有值且 G > 60 日均線，依 SURGE 由大到小、同分依代號；
     RET20＝G[s]/G[s−20]−1；「股池內 SURGE 有值 < 50 檔不產生訊號」在未位移的訊號列判斷），股池改由矩陣 U 給。
  5. prio_plans（審查 B，事後）：上市優先的名額分配。
"""
import sys, re, hashlib
import numpy as np
import pandas as pd

sys.path.insert(0, "/home/user/-/research/tw_stock_selection/agents/s14_corporate_events/replicate")
import rep_s14_lib as K14  # noqa: E402
K10 = K14.K10
K9 = K10.K9
S, Q, R, L = K14.S, K14.Q, K14.R, K14.L
ST, X2 = K14.ST, K14.X2
simulate = K14.simulate_s14
attach_e50_fixed = K14.attach_e50_fixed
adj_open_matrix = K14.adj_open_matrix

MOPS = "/home/user/chijiun/tuc-6274-financial-forecast/data/raw/mops_monthly/t21sc03_107_10_0.html"
MOPS_SHA = "1fb55c7bf25c6d17fd3e249a3733f7746c37f97f333c1e99ed544ba83d38f9ab"
RPATCH = "/home/user/-/research/tw_stock_selection/agents/s16_surge_otc/rev_patch_otc_201810.csv"
RPATCH_SHA = "011f0b0accedf45a26a0f6ceb048f5fb1d80ea671ced544b0bbd326e818d7ab5"
CELLS_U = ("CUR", "ALL", "OTC", "ALL_O20")


# ------------------------------------------------------------------ MOPS 補洞頁
def _num(s):
    s = s.replace("&nbsp;", "").replace(",", "").strip()
    if s in ("", "-", "--"):
        return np.nan
    return float(s)


def parse_mops(path=MOPS):
    b = open(path, "rb").read()
    h = hashlib.sha256(b).hexdigest()
    assert h == MOPS_SHA, h
    s = b.decode("cp950", errors="strict")
    rows = []
    pat = re.compile(r"<tr align=right><td align=center>\s*([0-9A-Z]{4,6})\s*</td><td align=left>(.*?)</td>(.*?)</tr>",
                     re.I | re.S)
    for m in pat.finditer(s):
        code, name, rest = m.group(1), m.group(2).strip(), m.group(3)
        cells = re.findall(r"<td[^>]*>(.*?)</td>", rest, re.I | re.S)
        rows.append(dict(code=code, name=name, cur=_num(cells[0]), prev=_num(cells[1]), ly=_num(cells[2]),
                         n_cells=len(cells)))
    d = pd.DataFrame(rows)
    return d, h, s.count("合計")


# ------------------------------------------------------------------ 月營收
def read_rev_csv():
    return pd.read_csv(L.REV, dtype={"code": str})


def load_rev_df(P, raw, patch=None):
    r = raw if patch is None else pd.concat([raw, patch], ignore_index=True)
    r = r.copy()
    r["pri"] = (r.market != "sii").astype(int)
    r = r.sort_values(["month", "code", "pri"]).drop_duplicates(["month", "code"], keep="first")
    r = r[r.code.isin(P.cidx)]
    mi = r.month.str[:4].astype(int).values * 12 + r.month.str[5:7].astype(int).values - 1
    mi0 = int(mi.min()); nm = int(mi.max()) - mi0 + 1
    Rm = np.full((nm, P.N), np.nan)
    Rm[mi - mi0, r.code.map(P.cidx).values] = r.revenue_k_twd.values
    Rm[~(Rm > 0)] = np.nan
    return Rm, mi0


# ------------------------------------------------------------------ 股池
def base_mat(P, excl_M):
    with np.errstate(invalid="ignore"):
        u = P.inbar & P.traded & (P.C >= 10.0) & (P.adv60 >= 1e7) & P.full60
    return u & ~excl_M


def univ_mats(P, excl_M):
    b = base_mat(P, excl_M)
    tw = P.M == 1
    tp = P.M == 2
    with np.errstate(invalid="ignore"):
        o20 = P.adv60 >= 2e7
    return dict(CUR=b & tw, OTC=b & tp, ALL=b & (tw | tp), ALL_O20=b & (tw | (tp & o20)))


def make_lists_u(P, S_, mi0, U, shift=0, min_n=50, ma_len=60, ret_len=20):
    out = {}
    codes = np.array(P.codes)
    mav = R.ma(P, ma_len)
    for m in range(S_.shape[0]):
        d10, j, s0 = L.rev_avail(P, mi0 + m)
        s = s0 + shift
        if s >= P.T - 1 or s0 < P.t0 or s0 >= P.T - 1:
            continue
        sv = S_[m]
        if int((U[s0] & np.isfinite(sv)).sum()) < min_n:
            continue
        u = U[s]
        with np.errstate(invalid="ignore", divide="ignore"):
            ok = u & np.isfinite(sv) & (P.G[s] > mav[s])
            r20 = P.G[s] / P.G[s - ret_len] - 1.0
        idx = np.where(ok)[0]
        order = idx[np.lexsort((codes[idx], -sv[idx]))]
        out[s] = dict(lma=order, r20=r20, m=m, n_surge=int((u & np.isfinite(sv)).sum()), n_pool=int(u.sum()))
    return out


def prio_plans(lists_tw, lists_x, xfilter=None, nmax=10, x=0.10):
    """審查 B：上市優先。lists_tw＝CUR（只上市）的名單；lists_x＝遞補來源名單（OTC 或 ALL）。
    買進名單＝上市 F1 名單（上市 L_MA 前 10 名中 RET20 > 10%）；不足 nmax 檔時，依序接上遞補來源 F1 名單中
    xfilter(i) 為真的股票（遞補來源 L_MA 前 10 名中 RET20 > 10%），合計最多 nmax 檔。
    續抱＝上市 L_MA 前 20 名 ∪ 遞補來源 L_MA 前 20 名中 xfilter 為真者。"""
    buy = {}; keep = {}; nfill = {}
    bt, kt = Q.cell_plans(lists_tw, "C10", x=x)
    bx, kx = Q.cell_plans(lists_x, "C10", x=x)
    for s in sorted(set(bt) | set(bx)):
        a = [int(i) for i in bt.get(s, [])]
        f = [int(i) for i in bx.get(s, []) if (xfilter is None or xfilter(s, int(i))) and int(i) not in a]
        k = len(a)
        add = f[:max(0, nmax - k)]
        buy[s] = np.array(a + add, dtype=int)
        nfill[s] = len(add)
        kx_s = set(i for i in kx.get(s, set()) if (xfilter is None or xfilter(s, i)))
        keep[s] = set(kt.get(s, set())) | kx_s
    return buy, keep, nfill


# ------------------------------------------------------------------ 時窗報酬（扣成本前）
def win_ret(P, AO, Gf, idx, e, e2):
    if len(idx) == 0:
        return np.nan
    a = AO[e, idx]
    b = AO[e2, idx] if e2 is not None else Gf[P.T - 1, idx]
    return float(np.nanmean(b / a - 1.0))


def e50_win(P, e, e2):
    b = P.e50o[e2] if e2 is not None else P.e50c[P.T - 1]
    return float(b / P.e50o[e] - 1.0)
