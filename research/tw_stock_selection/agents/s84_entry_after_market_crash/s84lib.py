#!/usr/bin/env python3
"""S84 共用：大盤急跌後那一次換股的歷史（純描述、事件研究；不設計規則、不擇時）。
重用 S75 連續面板與引擎（s75lib → s71lib／r55lib／s47lib／s10lib，全部不改）、S76 的 fwd_matrix。
本檔新增：
- simulate_s84：inspect 取 s10lib.simulate_s10 原始碼，照 S65（s65lib._REPL）的同一組字串替換加上 start_t／odd_from／end_t 後 exec（原檔不動；
  預設參數時與原引擎逐位相同＝run.py 驗收 A2）。只用來跑「某個訊號日 100 萬新帳戶起跑」。
- 0050 急跌指標、日曆月位置、分佈統計小工具。一律 python3 -I -B；不呼叫 git、不連網。"""
import sys, inspect
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
AG = HERE.parent
sys.path.insert(0, str(AG / 's76_early_vs_late_diagnosis'))
import s76lib as D                                            # noqa: E402
from s76lib import K, s10lib, stlib, cslib                    # noqa: E402
COSTS = s10lib.COSTS
CAP = 1e6

_SRC0 = inspect.getsource(s10lib.simulate_s10)
_REPL = [
    ("def simulate_s10(", "def simulate_s84("),
    ("exec_mode='open', trail=None, trail_daily=True, requal=None):",
     "exec_mode='open', trail=None, trail_daily=True, requal=None, start_t=None, odd_from='hist', end_t=None):"),
    ("    T, t0 = P.T, P.t0\n",
     "    T, t0 = P.T, (P.t0 if start_t is None else int(start_t))\n"),
    ("    odd_t = int(np.searchsorted(P.IDX.values, np.datetime64(cslib.ODD_DATE), side='left'))\n",
     "    odd_t = int(np.searchsorted(P.IDX.values, np.datetime64(cslib.ODD_DATE), side='left'))\n"
     "    assert odd_from in ('hist', 'all')\n"
     "    if odd_from == 'all':\n"
     "        odd_t = 0\n"),
    ("    for t in range(t0 + 1, T):\n",
     "    for t in range(t0 + 1, T if end_t is None else min(T, int(end_t) + 1)):\n"),
]
_SRC = _SRC0
for _a, _b in _REPL:
    assert _SRC.count(_a) == 1, f'替換點不唯一或不存在：{_a[:40]}'
    _SRC = _SRC.replace(_a, _b)
_G = dict(s10lib.__dict__)
exec(compile(_SRC, '<s84 simulate_s84>', 'exec'), _G)
simulate_s84 = _G['simulate_s84']


def sim_new(P, S_E0, U, start_t, end_t, odd_from='all'):
    return simulate_s84(P, S_E0, U, cost=COSTS['strict'], capital=CAP, cap_pct=0.05, sat_w=0.7, sizing='W10', lock=False,
                        start_t=start_t, odd_from=odd_from, end_t=end_t)


def e50_metrics(E, k, kprev):
    """E＝0050 含息淨值（日收盤）；k＝訊號日列號。全部只用 ≤ k 的資料。"""
    return dict(r10=E[k] / E[k - 10] - 1, r20=E[k] / E[k - 20] - 1, gap60=E[k] / E[k - 59:k + 1].mean() - 1,
                dd60=E[k] / E[k - 59:k + 1].max() - 1, m2m=(E[k] / E[kprev] - 1) if kprev is not None else np.nan)


def hpos(idx, k, months):
    """日期 ≤ idx[k]＋months 個日曆月的最後一列；超出資料 → None。"""
    d = idx[k] + pd.DateOffset(months=months)
    if d > idx[-1]:
        return None
    return int(np.searchsorted(idx.values, np.datetime64(d), side='right') - 1)


def dist(v):
    v = np.asarray(v, float); v = v[np.isfinite(v)]
    if len(v) == 0:
        return dict(n=0)
    return dict(n=int(len(v)), median=float(np.median(v)), mean=float(v.mean()), worst=float(v.min()), best=float(v.max()),
                p10=float(np.percentile(v, 10)), p25=float(np.percentile(v, 25)), p75=float(np.percentile(v, 75)), p90=float(np.percentile(v, 90)),
                pos=float((v > 0).mean()))


def rand_median_band(pop, n, rng, B=10000):
    """從 pop（一般月份）隨機不放回抽 n 個，取中位數，重複 B 次 → 5／50／95 百分位。只當尺，不是檢定。"""
    pop = np.asarray(pop, float); pop = pop[np.isfinite(pop)]
    if n == 0 or len(pop) < n:
        return None
    m = np.array([np.median(rng.choice(pop, n, replace=False)) for _ in range(B)])
    return dict(n_draw=int(n), B=B, p5=float(np.percentile(m, 5)), p50=float(np.percentile(m, 50)), p95=float(np.percentile(m, 95)))
