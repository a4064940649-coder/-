# -*- coding: utf-8 -*-
"""S8（壓回撤：單檔上限與衛星比例）獨立複製：自寫函式庫。
只依 s8_drawdown_cap/PREREG.json 與 FINDINGS.md 第 2、10 節的文字規格。
未讀、未 import 研究員的 s8lib／s7lib／s6lib／s5lib 及其 run／prep／posthoc。
import 的現成程式：s7_locked_base/replicate/rep_s7_lib.py（S7 獨立複製員）及其依賴。
simulate_s8 ＝ rep_s7_lib.simulate_s7 的原始碼，以字串替換產生（每個替換點都斷言恰好命中一次／指定次數），只多：
  cap：單檔上限（占開盤估淨值 eq 的比例；None＝不設）。每檔目標＝min(原目標, cap × eq)，
       eq＝出場賣完、拉回之後、買進之前的開盤估淨值（引擎變數 eq）。最小買進門檻、錢不夠判斷都用套上限後的目標。
       省下的錢不另處理，自然留在衛星現金（≥ 2 萬掃進 0050）。
  記錄：上限觸發（原目標 > cap×eq 且真的買進）、省下金額、實際買到金額；A4（股數×開盤 > cap×eq＋1 元）。
  rec_pos=True：記每日收盤每檔市值 snap[t]={i: val} 與每日每檔現金流 flows[(t,i)]（買進為負、賣出／下市為正），
       供回撤解剖：區間內個股損益＝Σ(市值變動＋現金流)。
cap=None 且 rec_pos=False 時，買賣路徑與 rep_s7_lib.simulate_s7 完全相同。
"""
import sys, inspect
import numpy as np
import pandas as pd

sys.path.insert(0, "/home/user/-/research/tw_stock_selection/agents/s7_locked_base/replicate")
import rep_s7_lib as K7  # noqa: E402
K6, C, S, Q, R, L = K7.K6, K7.C, K7.S, K7.Q, K7.R, K7.L
FEE, TAX, TAX50 = S.FEE, S.TAX, S.TAX50
tick, limit_up, limit_dn = S.tick, S.limit_up, S.limit_dn
c10_ranks, delay_plans, hypo_ret = K7.c10_ranks, K7.delay_plans, K7.hypo_ret

_SRC = inspect.getsource(K7.simulate_s7)
_REPL = [
    ("def simulate_s7(", "def simulate_s8(", 1),
    ("lock=False, lranks=None):", "lock=False, lranks=None, cap=None, rec_pos=False):", 1),
    ("    short_days = set(); skips = []; shrinks = []\n",
     "    short_days = set(); skips = []; shrinks = []\n"
     "    cap_hits = []; a4 = []; snap = {}; flows = {}\n"
     "    def _fl(t_, i_, x_):\n"
     "        if rec_pos:\n"
     "            flows[(t_, i_)] = flows.get((t_, i_), 0.0) + x_\n", 1),
    ('p["recv"] += p["val"]; p["delist"] = 1;', 'p["recv"] += p["val"]; _fl(t, i, p["val"]); p["delist"] = 1;', 1),
    ('p["recv"] += net - fe - tx\n', 'p["recv"] += net - fe - tx; _fl(t, i, net - fe - tx)\n', 2),
    ("            sold_today = set(closed)\n",
     "            tgts_raw = np.array(tgts, dtype=float)\n"
     "            if cap is not None:\n"
     "                tgts = np.minimum(tgts_raw, cap * eq)\n"
     "            sold_today = set(closed)\n", 1),
    ("                target = float(tgts[rk])\n",
     "                target = float(tgts[rk]); traw = float(tgts_raw[rk])\n", 1),
    ("                cash -= c\n",
     "                cash -= c; _fl(t, i, -c)\n"
     "                if cap is not None:\n"
     "                    if traw > target + 1e-9:\n"
     "                        cap_hits.append(dict(t=t, date=P.dstr[t], code=codes[i], pos=rk + 1, n_list=n_list,\n"
     "                                             raw=traw, capped=target, saved=traw - target, bought=sh * o, eq=eq))\n"
     "                    if sh * o > cap * eq + 1.0:\n"
     "                        a4.append(dict(date=P.dstr[t], code=codes[i], bought=sh * o, lim=cap * eq))\n", 1),
    ("        stkw[t] = v / nav[t]; Bs[t] = B\n",
     "        stkw[t] = v / nav[t]; Bs[t] = B\n"
     "        if rec_pos:\n"
     "            snap[t] = {i_: p_['val'] for i_, p_ in pos.items()}\n", 1),
    ("    return dict(a3=a3,", "    return dict(cap_hits=cap_hits, a4=a4, snap=snap, flows=flows, a3=a3,", 1),
]
for a, b, n in _REPL:
    k = _SRC.count(a)
    assert k == n, (a, k, n)
    _SRC = _SRC.replace(a, b)
_ns = dict(globals())
exec(compile(_SRC, "<rep_s8_lib.simulate_s8>", "exec"), _ns)
simulate_s8 = _ns["simulate_s8"]
SIM8_SRC = _SRC


def window_pnl(P, res, ip, it):
    """回撤區間 (ip, it] 每檔損益（市值變動＋現金流），需 rec_pos=True。回傳 {代碼索引: 損益}。"""
    snap, fl = res["snap"], res["flows"]
    out = {}
    for t in range(ip + 1, it + 1):
        a = snap.get(t - 1, {}); b = snap.get(t, {})
        for i in set(a) | set(b):
            out[i] = out.get(i, 0.0) + b.get(i, 0.0) - a.get(i, 0.0) + fl.get((t, i), 0.0)
    return out
