#!/usr/bin/env python3
"""S41 事後描述（未登記，不影響判定）：上月 SURGE 能不能代替當月 SURGE？
這群公司（S40 名單＋2884）2016-01～2026-09 每個訊號列：S[m−1] 與 S[m] 的相關、L_MA 前 20 名／前 10 名判斷一致的比例。
輸出 results_posthoc.json。python3 -I -B posthoc.py"""
import sys, json
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
import s41lib as L
from s41lib import s10lib

H = Path(__file__).resolve().parent
P, U, RT = L.prep_panel()
S0, av = RT['S'], RT['av']
sg = s10lib.make_signal_s10(P, U, RT, shift=0)
GJ = [int(np.flatnonzero(P.COLS == c)[0]) for c in L.GROUP if (P.COLS == c).any()]
ri = {m: i for i, m in enumerate(S0.index)}
A0 = S0.to_numpy()
xs, ys, agree20, agree10, n20cur, n20stale = [], [], 0, 0, 0, 0
nall = 0
for x in sg['info']:
    m = pd.Period(x['month'], 'M')
    if m < pd.Period('2016-01', 'M'):
        continue
    k = x['k']; i = ri[m]; s = A0[i]
    ok = U[k] & np.isfinite(s) & P.F_MA60[k]
    for j in GJ:
        v0, v1 = A0[i - 1, j], s[j]
        if not (U[k, j] and P.F_MA60[k, j] and np.isfinite(v0) and np.isfinite(v1)):
            continue
        others = ok.copy(); others[j] = False
        idx = np.flatnonzero(others); sv = s[idx]
        r_cur = int(((sv > v1) | ((sv == v1) & (idx < j))).sum()) + 1
        r_st = int(((sv > v0) | ((sv == v0) & (idx < j))).sum()) + 1
        xs.append(v0); ys.append(v1); nall += 1
        agree20 += int((r_cur <= 20) == (r_st <= 20)); agree10 += int((r_cur <= 10) == (r_st <= 10))
        n20cur += int(r_cur <= 20); n20stale += int(r_st <= 20)
xs, ys = np.array(xs), np.array(ys)
both20 = (n20cur + n20stale - (nall - agree20)) // 2
out = dict(n_cells=nall, pearson=float(np.corrcoef(xs, ys)[0, 1]), spearman=float(pd.Series(xs).corr(pd.Series(ys), method='spearman')),
           top20_by_current=n20cur, top20_by_stale=n20stale, agree_top20=float(agree20 / nall), agree_top10=float(agree10 / nall), top20_both=int(both20),
           note='上月與當月 SURGE 相關只有約 0.36～0.38；當月 SURGE 在前 20 名的格子，用上月 SURGE 也在前 20 名的只占少數（top20_both）。C 在 2026-06 用上月 SURGE 買到 2851、2850，而現行用當月 SURGE 買的是 2885。')
(H / 'results_posthoc.json').write_text(json.dumps(out, ensure_ascii=False, indent=1))
print(out)
