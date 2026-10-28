#!/usr/bin/env python3
"""獨立複製（統計）：SPA 用 o_main_spa_breaks/replicate_o5 的獨立實作（spa.py／rlib.py）與它自己的族群矩陣 out/X50.parquet、cover.csv；
把族群中 o_v4 的 AB_V4、AB_V4_B10、V4 三條換成「複製員引擎＋複製員面板」的 V7 版（其餘 11 條 o_v4 成員換成研究員 navs_v6 的 V7 版）。
保留期 p_lower：自寫平穩 bootstrap（V7 格點＝複製員 V7 訊號日＋2026-10-12；日曆歸屬：窗終點 ≥ 2024-01-02）。輸出 replicate/rep_stats.json。"""
import sys, json, math
from pathlib import Path
import numpy as np, pandas as pd
H = Path(__file__).resolve().parent; TOP = H.parent; AG = TOP.parent
R5 = AG / 'o_main_spa_breaks' / 'replicate_o5'
sys.path.insert(0, str(R5))
import rlib, spa as SP     # noqa
sys.path.pop(0)
REP = pd.read_parquet(H / 'out_rep.parquet'); NV = pd.read_parquet(TOP / 'navs_v7.parquet')
E = rlib.e0050()
X50 = pd.read_parquet(R5 / 'out' / 'X50.parquet'); Cv = pd.read_csv(R5 / 'out' / 'cover.csv', index_col=0)
FL = list(Cv.index[Cv.W_long])
pref = 'o_v4_revenue_patch_official/navs_v4.parquet::'
r50 = rlib.win_ret(E)
X = X50.copy()
mine_map = {}
for m in [c for c in FL if c.startswith(pref)]:
    k = m[len(pref):]
    if k in ('AB_V4|normal', 'AB_V4_B10|normal', 'V4|normal'):
        nav = REP[{'AB_V4|normal': 'AB_V7', 'AB_V4_B10|normal': 'AB_V7_B10', 'V4|normal': 'V7'}[k]]; mine_map[m] = 'replicator'
    else:
        k6 = k.replace('V4', 'V7'); k6 = 'O3REP_V7' + k6[5:] if k6.startswith('O3REP') else k6
        nav = NV[k6]; mine_map[m] = 'researcher navs_v7:' + k6
    X[m] = rlib.win_ret(nav) - r50
out = {'replaced': mine_map}
D = X.loc[:, FL].fillna(0.0).to_numpy()
for L in (3, 6, 12):
    idx = rlib.sb_indices(D.shape[0], L, 10000, 20261028 + L)
    for foc in ('AB_V4|normal', 'AB_V4_B10|normal'):
        o = SP.spa(D, FL.index(pref + foc), L, idx, stepm=False, unstd=False)
        out[f'{foc.replace("V4", "V7")}|L{L}'] = {k: o[k] for k in ('p_SPA_c_focus', 'rank', 't_main', 'p_SPA_c', 'K')}
        print(foc, L, out[f'{foc.replace("V4", "V7")}|L{L}'], flush=True)
# 保留期
G = pd.DatetimeIndex(list(REP['V7_signal_days'].dropna().index) + [pd.Timestamp('2026-10-12')])
def wr(nav):
    s = nav.dropna(); v = s.reindex(s.index.union(G)).ffill().reindex(G).to_numpy(); return v[1:] / v[:-1] - 1
st = G[:-1]; en = G[1:]
pre = (st >= pd.Timestamp('2006-04-01')) & (en < pd.Timestamp('2024-01-02')); hold = en >= pd.Timestamp('2024-01-02')
for tag in ('AB_V7', 'AB_V7_B10'):
    rm = wr(REP[tag]); rb = wr(E)
    lx = np.log1p(rm) - np.log1p(rb); obs = lx[hold].sum(); src_ = lx[pre]; nh = int(hold.sum())
    o = {'n_hold': nh, 'cumlog': float(obs), 'hold_first_window': str(st[hold][0].date())}
    for L in (3, 6, 12):
        rng = np.random.default_rng(777 + L); p = 1 / L; n0 = len(src_); Bn = 10000
        idx = np.empty((Bn, nh), int); idx[:, 0] = rng.integers(0, n0, Bn)
        nb = rng.random((Bn, nh)) < p; rr = rng.integers(0, n0, (Bn, nh))
        for t in range(1, nh): idx[:, t] = np.where(nb[:, t], rr[:, t], (idx[:, t - 1] + 1) % n0)
        o[f'p_lower_L{L}'] = float((src_[idx].sum(1) <= obs).mean())
        o[f'p_lower_shrink50_L{L}'] = float(((src_ - 0.5 * src_.mean())[idx].sum(1) <= obs).mean()); o[f'p_lower_zero_L{L}'] = float(((src_ - src_.mean())[idx].sum(1) <= obs).mean())
    out[f'hold|{tag}'] = o; print(tag, o, flush=True)
(H / 'rep_stats.json').write_text(json.dumps(out, ensure_ascii=False, indent=1))
