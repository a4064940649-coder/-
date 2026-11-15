# -*- coding: utf-8 -*-
"""S20（持續成長 μ÷σ 篩選 F1）獨立複製：自寫函式庫。
只依 s20_persistent_growth/PREREG.json、FROZEN.json、data_check.json 與 FINDINGS.md 第 1 節（事前登記）、第 6 節（規格）。
未讀、未 import 研究員的 s20lib／s17lib／s13lib／s10lib 及其 train／validate／posthoc／prep／prereg_write。
import 的現成程式（都是之前的複製員寫的）：
  s17 replicate/rep_s17_lib.py（yoy、zscore：σ 兩段式、σ≈0 當缺值；simulate＝rep_s14 引擎；0050 停牌修正 attach_e50_fixed）
  → rep_s14 → rep_s10 → rep_s9 → … → rep_s1；
  s13 replicate/rep_s13_lib.py（outcome_e0：E0 持有期對 0050 超額，不扣成本）。
本檔自己寫的部分：
  1. features：MS＝μ÷σ（μ、σ＝D(m−24…m−1) 的平均與樣本標準差，D＝營收(m)−營收(m−12)，與 STD SURGE 同一份 μ、σ），
     MU＝1[μ>0]，NPOS12＝D(m−12…m−1) > 0 的月數。只在 STD SURGE 有值的格子給值（F1 名單必有值）。
  2. filt_plans：F1 買進名單 ∩ 規則；不遞補；續抱集合（L_MA 前 20 名）不變。訊號鍵全部保留（空陣列）以維持引擎的換股日／帶狀拉回時序。
  3. fresh_plans：只留營收月份 ≥ 2021-12 的訊號（更早的訊號鍵保留但買進名單為空）。
  4. backfill_plans（審查 C）：被規則篩掉幾檔，就從 L_MA 第 11～20 名中依序遞補同樣 RET20 > 10% 且 μ > 0 的。
  5. random_keep_plans（審查 A）：F1 名單（位移 0）全部 (營收月份, 代號) 中隨機留下固定檔數（種子固定），其他同 filt_plans。
"""
import sys
import numpy as np

sys.path.insert(0, "/home/user/-/research/tw_stock_selection/agents/s17_surge_definition_family/replicate")
sys.path.insert(0, "/home/user/-/research/tw_stock_selection/agents/s13_big_winner_features/replicate")
import rep_s17_lib as K17  # noqa: E402
import rep_s13_lib as K13  # noqa: E402

S, Q, R, L = K17.S, K17.Q, K17.R, K17.L
ST, X2 = K17.ST, K17.X2
simulate = K17.simulate
attach_e50_fixed = K17.attach_e50_fixed
outcome_e0 = K13.outcome_e0
VALID_MI = 2021 * 12 + 11   # 營收月份 2021-12


def features(Rv, mi0):
    D = K17.yoy(Rv, mi0, "amt")
    Sx, MU, SD = K17.zscore(D, mi0, 24, "z")
    with np.errstate(invalid="ignore", divide="ignore"):
        MS = MU / SD
    MS = np.where(np.isfinite(Sx), MS, np.nan)
    MUb = np.where(np.isfinite(Sx), (MU > 0).astype(float), np.nan)
    nm, N = D.shape
    NP = np.full((nm, N), np.nan)
    for m in range(12, nm):
        w = D[m - 12:m]
        NP[m] = (w > 0).sum(axis=0)
    NP = np.where(np.isfinite(Sx), NP, np.nan)
    return Sx, dict(MS=MS, MU=MUb, NPOS12=NP, mu=MU, sd=SD)


def rule_pass(F, rule, m, i):
    x = F[rule["feat"]][m, i]
    if not np.isfinite(x):
        return False
    op, thr = rule["op"], rule["thr"]
    if op == ">=":
        return bool(x >= thr)
    if op == ">":
        return bool(x > thr)
    if op == "<":
        return bool(x < thr)
    raise ValueError(op)


def filt_plans(lists, plans, F, rule):
    buy, keep = plans
    if rule is None:
        return buy, keep
    b2 = {}
    for s, b in buy.items():
        m = lists[s]["m"]
        b2[s] = np.array([int(i) for i in b if rule_pass(F, rule, m, int(i))], dtype=int)
    return b2, keep


def fresh_plans(lists, plans, mi0, first_mi=VALID_MI):
    buy, keep = plans
    b2 = {}
    for s, b in buy.items():
        b2[s] = b if (mi0 + lists[s]["m"] >= first_mi) else np.array([], dtype=int)
    return b2, keep


def backfill_plans(lists, plans, F, rule, x=0.10):
    buy, keep = plans
    b2 = {}; nfill = 0; ndrop = 0
    for s, b in buy.items():
        q = lists[s]; m = q["m"]
        kept = [int(i) for i in b if rule_pass(F, rule, m, int(i))]
        k = len(b) - len(kept)
        ndrop += k
        cand = []
        with np.errstate(invalid="ignore"):
            for i in q["lma"][10:20]:
                i = int(i)
                if q["r20"][i] > x and np.isfinite(F["mu"][m, i]) and F["mu"][m, i] > 0 and np.isfinite(F["MS"][m, i]):
                    cand.append(i)
        add = cand[:k]
        nfill += len(add)
        b2[s] = np.array(kept + add, dtype=int)
    return (b2, keep), dict(n_dropped=ndrop, n_filled=nfill)


def f1_entries(lists, plans):
    buy, _ = plans
    return [(lists[s]["m"], int(i)) for s in sorted(buy) for i in buy[s]]


def random_keep_plans(lists, plans, entries_keep):
    buy, keep = plans
    b2 = {}
    for s, b in buy.items():
        m = lists[s]["m"]
        b2[s] = np.array([int(i) for i in b if (m, int(i)) in entries_keep], dtype=int)
    return b2, keep
