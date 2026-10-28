#!/usr/bin/env python3
"""G2：2013～2021 年 AB_V4 前 10 名股月的首次值覆蓋（不看報酬；只用名單）。
名單：AB_V4（V4 凍結資料、官方換股日、40 seeds 的 mc 公告延遲），每個 seed 的前 10 名股月；營收月＝該股月實際用到的三個月
（未公告者退一個月）。
佐證標準（寫死）：
  嚴格（G2 判定用）：首次公告窗（營收月次月月底前）內，≥ 2 個「獨立來源」（CTBC 兩個資料夾算同一來源；BOF、ELD、PTT 各算一個）
        的水準一致（相對差 ≤ 0.5%；無水準時年增率差 ≤ 0.1pp），或一個來源＋git 提交時間戳鏡像（本題找不到 2013～2021 有 PIT 提交時間的鏡像，此路徑為 0）。
  寬鬆（描述用，比照 o5 F3 的共識）：≥ 2 則不同日期或不同文章的新聞（CTBC 兩資料夾的同一篇視為同一則）一致。
覆蓋：股月三個營收月全部有佐證首次值（主）；只有最新一個月有佐證（並報）。
另報：已佐證格的首次值對最新值（官方水準）不一致率（|差| > 0.5%）。
用法：python3 -I -B firstvalue.py P3
"""
import sys, json
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import numpy as np, pandas as pd
import rglib as G
import proxy as PX
K, V, L = G.K, G.V, G.L
IT = sys.argv[1]
b = G.base()
M, LV, L5, Q = PX.load_inputs(K, HERE / 'data' / f'mentions_{IT}.parquet')
M['ts'] = pd.to_datetime(M.ts)
M['fam'] = M.src.where(M.src != 'CTBC', 'CTBC')
M['day'] = M.ts.dt.normalize()
def first_window(p):
    return (pd.Period(p, 'M') + 1).end_time
cells = {}
for (c, ym), g in M.groupby(['code', 'ym']):
    g = g[g.ts <= first_window(ym)]
    if len(g) == 0: continue
    lv = g.rev_k.to_numpy(); yy = g.yoy.to_numpy()
    def agree(i, j):
        if np.isfinite(lv[i]) and np.isfinite(lv[j]): return abs(lv[i] / lv[j] - 1) <= 0.005
        if np.isfinite(yy[i]) and np.isfinite(yy[j]): return abs(yy[i] - yy[j]) <= 0.1
        return False
    fam = g.fam.to_numpy(); key = list(zip(g.day, g.rev_k.round(-1), g.yoy.round(2)))
    strict = loose = False; val = np.nan
    for i in range(len(g)):
        for j in range(i + 1, len(g)):
            if not agree(i, j): continue
            if fam[i] != fam[j]: strict = True
            if key[i] != key[j] or fam[i] != fam[j]: loose = True
            if np.isnan(val) and np.isfinite(lv[i]): val = lv[i]
    cells[(c, ym)] = (strict, loose, val)
S0 = G.mk()
rows = []
for seed in range(40):
    U = V.make_unavail('mc', seed)
    S = G.mk(unavail=U)
    for e, pk, _, d in S.plan(10, 'official'):
        if not (G.W0 <= d <= G.W1): continue
        u = U(d); m = pd.Period(d, 'M')
        for j in pk:
            c = K.COLS[j]; back = 1 if u[j] else 0
            ms = [str(m - back - k) for k in (1, 2, 3)]
            st = [cells.get((c, x), (False, False, np.nan)) for x in ms]
            rows.append((seed, str(d.date()), c, ms[0], all(s[0] for s in st), st[0][0], all(s[1] for s in st), st[0][1]))
R = pd.DataFrame(rows, columns=['seed', 'sig', 'code', 'ym_latest', 'strict_all3', 'strict_latest', 'loose_all3', 'loose_latest'])
cov = {k: float(R[k].mean()) for k in ('strict_all3', 'strict_latest', 'loose_all3', 'loose_latest')}
# 首次值 vs 最新值（已佐證格）
inc = []
for (c, ym), (st, lo, val) in cells.items():
    if not (lo and np.isfinite(val)): continue
    off = np.nan
    for X in (LV, L5):
        p = pd.Period(ym, 'M')
        if c in X.columns and p in X.index and np.isfinite(X.at[p, c]): off = X.at[p, c]; break
    if np.isfinite(off) and off > 0: inc.append((c, ym, st, val, off, abs(val / off - 1) > 0.005))
I = pd.DataFrame(inc, columns=['code', 'ym', 'strict', 'first', 'latest', 'disagree'])
def wilson(k, n, z=1.96):
    if n == 0: return [None, None]
    p = k / n; d = 1 + z * z / n; c = p + z * z / (2 * n); h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return [float((c - h) / d), float((c + h) / d)]
out = {'iter': IT, 'n_top10_stock_month_seed': int(len(R)), 'n_distinct_stock_months': int(R[['sig', 'code']].drop_duplicates().shape[0]),
       'coverage': cov, 'G2_threshold': 0.20, 'G2_pass': cov['strict_all3'] >= 0.20,
       'coverage_wilson95_strict_all3': wilson(int(R.strict_all3.sum() / 40), int(len(R) / 40)),
       'n_cells_corroborated_strict': int(sum(1 for v in cells.values() if v[0])), 'n_cells_corroborated_loose': int(sum(1 for v in cells.values() if v[1])),
       'first_vs_latest': {'n_loose': int(len(I)), 'disagree_loose': int(I.disagree.sum()), 'rate_loose': float(I.disagree.mean()) if len(I) else None,
                           'n_strict': int(I.strict.sum()), 'disagree_strict': int(I[I.strict].disagree.sum()),
                           'rate_strict_wilson95': wilson(int(I[I.strict].disagree.sum()), int(I.strict.sum()))}}
(HERE / 'data').mkdir(exist_ok=True)
R.to_csv(HERE / 'data' / f'top10_firstvalue_{IT}.csv', index=False)
(HERE / f'firstvalue_{IT}.json').write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding='utf-8')
print(json.dumps(out, ensure_ascii=False, indent=1))
