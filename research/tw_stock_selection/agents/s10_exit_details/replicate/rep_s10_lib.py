# -*- coding: utf-8 -*-
"""S10（出場細節家族）獨立複製：自寫函式庫。
只依 s10_exit_details/PREREG.json 與 FINDINGS.md 第 2、9 節的文字規格。
未讀、未 import 研究員的 s10lib／s9lib／s8lib／s6lib／s5lib／s4lib 及其 run／prep／posthoc。
import 的現成程式：s9_signal_timing/replicate/rep_s9_lib.py（S9 獨立複製員）及其依賴（rep_s6／s5／s4／s3／s2／s1 lib）。

simulate_s10 ＝ rep_s9_lib.simulate_s9 的原始碼（以字串替換產生，每個替換點都斷言恰好命中指定次數），只多：
  eod：每日收盤後的出場標記函式 eod(t, pos)，取代原本的「if t in keep: p['pend'] = i not in kp」。
       eod=None 時與 simulate_s9 完全相同。
  交易記錄多一欄 why（標記原因：sig＝訊號日名次／均線、ts＝每日移動停損、tsm＝換股日移動停損、
       dma＝每日季線、rq＝持有滿 N 個換股月未重新合格、delist、reduce）。

出場變體（make_eod／keep_sets）：
  E0：訊號日不在 L_MA 前 20 名就賣。R10／R15／R30：前 10／15／30 名。
  MA20：股池內（與 L_MA 同一股池：rep_s2 universe 且非創新板）、SURGE 有值、SURGE ≥ 當月 L_MA 第 20 名分數、
        還原收盤 > 20 日均線才續抱。MANO：同上但不看均線。（L_MA 不足 20 名時分數門檻＝−∞。）
        「股池內」是複製員的解讀：E0 離開股池會被賣，驗收 A2 要求拆法在所有股票上等於 L_MA 前 20 名，所以要含股池。
  DMA60：E0＋每日收盤 G ≤ 60 日均線（G＝含息累積指數，均線有值時）就標記。
  TS d（每日）：E0＋每日收盤 g ≤ 持有期最高 g ×(1−d)；g 進場日＝收盤÷開盤，之後 ×(1＋含息日報酬)。
  TS15M：g、最高 g 每日更新，只在訊號列收盤檢查。
  RQ n：E0＋每個訊號列 nsig += 1（進場後經過的訊號列數，含當天）；nsig ≥ n 且不在當月買進名單 → 標記。
  額外觸發（ts／tsm／dma／rq）標記後保持（sticky），不會被下個訊號日的名次重設；名次觸發照原引擎每個訊號日重設。
"""
import sys, inspect
import numpy as np
import pandas as pd

sys.path.insert(0, "/home/user/-/research/tw_stock_selection/agents/s9_signal_timing/replicate")
import rep_s9_lib as K9  # noqa: E402
K6, C, S, Q, R, L = K9.K6, K9.C, K9.S, K9.Q, K9.R, K9.L
FEE, TAX, TAX50 = S.FEE, S.TAX, S.TAX50
tick, limit_up, limit_dn = S.tick, S.limit_up, S.limit_dn
ST, X2 = S.STRICT, S.X2

_SRC = inspect.getsource(K9.simulate_s9)
_REPL = [
    ("def simulate_s9(", "def simulate_s10(", 1),
    ('exec_mode="open",\n                rec_pos=False):', 'exec_mode="open",\n                rec_pos=False, eod=None):', 1),
    ('sh=p["sh0"], rank=p["rank"], reduced=p.get("reduced", 0)))',
     'sh=p["sh0"], rank=p["rank"], reduced=p.get("reduced", 0),\n'
     '                               why=("delist" if p.get("delist", 0) else p.get("why", "")), mk=p.get("mk", -1)))', 1),
    ('hold=t - p["tb"], delist=0, sh=p["sh0"], rank=p["rank"], reduced=1))',
     'hold=t - p["tb"], delist=0, sh=p["sh0"], rank=p["rank"], reduced=1, why="reduce", mk=-1))', 1),
    ("        if t in keep:\n            kp = keep[t]\n",
     "        if eod is not None:\n            eod(t, pos)\n        elif t in keep:\n            kp = keep[t]\n", 1),
]
for a, b, n in _REPL:
    k = _SRC.count(a)
    assert k == n, (a, k, n)
    _SRC = _SRC.replace(a, b)
_ns = dict(globals())
exec(compile(_SRC, "<rep_s10_lib.simulate_s10>", "exec"), _ns)
simulate_s10 = _ns["simulate_s10"]
SIM10_SRC = _SRC

VARS = ["E0", "R10", "R15", "R30", "MA20", "MANO", "DMA60", "TS15", "TS25", "TS15M", "RQ6"]
SPEC = {
    "E0": dict(keep="top20"), "R10": dict(keep="top10"), "R15": dict(keep="top15"), "R30": dict(keep="top30"),
    "MA20": dict(keep="ma20"), "MANO": dict(keep="mano"), "DMA60": dict(keep="top20", dma=True),
    "TS15": dict(keep="top20", ts=0.15, ts_daily=True), "TS25": dict(keep="top20", ts=0.25, ts_daily=True),
    "TS15M": dict(keep="top20", ts=0.15, ts_daily=False), "RQ6": dict(keep="top20", rq=6),
}


def keep_sets(P, S_, lists, kind, excl, mavs):
    """{訊號列 s: 續抱集合}。kind：topN、ma20、mano、ma60split（驗收 A2 用）。"""
    out = {}
    for s, q in lists.items():
        lma = q["lma"]
        if kind.startswith("top"):
            out[s] = set(int(i) for i in lma[:int(kind[3:])])
            continue
        sv = S_[q["m"]]
        u = R.universe(P, s)
        if excl is not None:
            u = u & ~excl(s)
        thr = sv[lma[19]] if len(lma) >= 20 else -np.inf
        with np.errstate(invalid="ignore"):
            ok = u & np.isfinite(sv) & (sv >= thr)
            if kind == "ma20":
                ok &= P.G[s] > mavs[20][s]
            elif kind == "ma60split":
                ok &= P.G[s] > mavs[60][s]
            elif kind != "mano":
                raise ValueError(kind)
        out[s] = set(int(i) for i in np.where(ok)[0])
    return out


def make_eod(P, keep, buy, ma60, ts=None, ts_daily=True, rq=None, dma=False, log=None):
    """回傳 eod(t, pos)。log（list）不為 None 時記錄每個訊號列收盤的持股 (t, i)。"""
    bls = {s: set(int(i) for i in b) for s, b in buy.items()}
    G, Cf, O, ret = P.G, P.Cf, P.O, P.ret
    fac = None if ts is None else (1.0 - ts)

    def eod(t, pos):
        if fac is not None:
            for i, p in pos.items():
                if p["tb"] == t:
                    g = Cf[t, i] / O[t, i]
                    p["g"] = g; p["gm"] = g
                else:
                    r = ret[t, i]
                    if r == r:
                        p["g"] *= (1.0 + r)
                        if p["g"] > p["gm"]:
                            p["gm"] = p["g"]
        if t in keep:
            kp = keep[t]; bl = bls[t]
            for i, p in pos.items():
                if log is not None:
                    log.append((t, i))
                std = i not in kp
                xs = p.get("xs", False)
                p["pend"] = std or xs
                if std and not xs:
                    p["why"] = "sig"; p["mk"] = t
                if rq is not None:
                    p["nsig"] = p.get("nsig", 0) + 1
                    if (not p["pend"]) and p["nsig"] >= rq and i not in bl:
                        p["pend"] = True; p["xs"] = True; p["why"] = "rq"; p["mk"] = t
                if fac is not None and not ts_daily and not p["pend"] and p["g"] <= p["gm"] * fac:
                    p["pend"] = True; p["xs"] = True; p["why"] = "tsm"; p["mk"] = t
        if dma or (fac is not None and ts_daily):
            for i, p in pos.items():
                if p["pend"]:
                    continue
                if dma:
                    m = ma60[t, i]
                    if m == m and G[t, i] == G[t, i] and G[t, i] <= m:
                        p["pend"] = True; p["xs"] = True; p["why"] = "dma"; p["mk"] = t
                        continue
                if fac is not None and ts_daily and p["g"] <= p["gm"] * fac:
                    p["pend"] = True; p["xs"] = True; p["why"] = "ts"; p["mk"] = t
    return eod


def adj_open(P, t, i):
    """還原開盤（含息）：G[t]×開盤÷收盤；沒有開盤或已下市就用最後一個有值的 G（下市＝最後收盤市值結清）。"""
    if t > P.last[i]:
        t = int(P.last[i])
        return float(P.G[t, i])
    o = P.O[t, i]; c = P.Cf[t, i]; g = P.G[t, i]
    if o == o and c == c and g == g and c > 0:
        return float(g * o / c)
    col = P.G[:t + 1, i]
    k = np.where(np.isfinite(col))[0]
    return float(col[k[-1]])
