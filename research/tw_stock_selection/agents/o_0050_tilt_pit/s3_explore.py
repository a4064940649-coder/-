#!/usr/bin/env python3
"""探索（只用挑選期：訊號窗起點 2014-01～2018-12）：層結構、2330 固定權重、基準 B 對 0050 的追蹤、分數持續性、
層內資訊係數（IC）、隨機分數安慰劑（設計檢定，1,000 組，毛報酬）、子樣本門檻（凍結供樣本外用）。寫 explore.json。"""
import sys, json, math
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
from scipy.stats import spearmanr
import tcore as T, tlib as L
SEL_A, SEL_B = '2014-01-01', '2018-12-31'
JS = list(range(13))
EX = {'period': f'{SEL_A}～{SEL_B}（訊號窗起點）', 'note': '只用挑選期；2006～2013 不過 PIT 股數閘門，不使用'}
# E1 層結構（官方日）
offd = T.sig_days('off', SEL_A, SEL_B)
assert max(offd[:-1]) <= pd.Timestamp(SEL_B)
rows = []
for kind in ('T1', 'T2'):
    for d in offd[:-1]:
        e = T.IDX[T.IDX.get_loc(d) + 1]
        U, wb, wt, inf = T.target(d, e, kind, 0.0, False, info=True)
        r = {'kind': kind, 'd': d, 'n': len(U)}
        for nm, idx in inf['tiers'].items(): r[f'n_{nm}'] = len(idx); r[f'w_{nm}'] = float(wb[idx].sum())
        r['w2330'] = float(wb[U == T.J2330].sum()); r['elec_w'] = float(wb[T.ELECMASK[U]].sum())
        rows.append(r)
E1 = pd.DataFrame(rows)
EX['E1_tier_structure_official_days'] = {k: E1[E1.kind == k].drop(columns=['kind', 'd']).mean().round(4).to_dict() for k in ('T1', 'T2')}
EX['E1_2330_weight_range'] = {k: [float(E1[E1.kind == k].w2330.min()), float(E1[E1.kind == k].w2330.max())] for k in ('T1', 'T2')}
# E2 基準 B 對 0050（同窗，一般成本）、B 的周轉
bt = {}
for kind in ('T1', 'T2'):
    for j in JS:
        ds = T.sig_days(j, SEL_A, SEL_B)
        pre = T.precompute(ds, kind)
        rB, iB = T.run(ds, {d: (u, w) for d, u, w, tz in pre})
        x = rB - T.bench_win(ds)
        bt[(kind, j)] = {**T.stats(x), 'turnover': float(iB['turnover'][1:].mean())}
EX['E2_base_vs_0050_jmedian'] = {k: {m: float(np.median([bt[(k, j)][m] for j in JS])) for m in ('mean_pp', 't', 'TE', 'turnover')} for k in ('T1', 'T2')}
# E3 分數持續性：層內 z 的月對月 Spearman（官方日）
pers = {'T1': [], 'T2': []}
for kind in ('T1', 'T2'):
    pre = T.precompute(offd, kind); prev = None
    for d, U, wb, tz in pre:
        cur = {}
        for nm, (idx, z) in tz.items():
            for i, zz in zip(U[idx], z): cur[int(i)] = zz
        if prev:
            com = [i for i in cur if i in prev and cur[i] != 0 and prev[i] != 0]
            if len(com) > 10: pers[kind].append(spearmanr([cur[i] for i in com], [prev[i] for i in com]).correlation)
        prev = cur
EX['E3_score_rank_autocorr_monthly'] = {k: float(np.mean(v)) for k, v in pers.items()}
# E4 層內 IC：z 對 e 開盤 → 下一訊號日收盤報酬的 Spearman（各 j，平均 IC、t）
ic = {}
for kind in ('T1', 'T2'):
    for j in JS:
        ds = T.sig_days(j, SEL_A, SEL_B); pre = T.precompute(ds, kind)
        ti = [T.IDX.get_loc(d) for d in ds]
        for k, (d, U, wb, tz) in enumerate(pre):
            f2 = T.factor_open_to_close(ti[k] + 1, ti[k + 1])
            for nm, (idx, z) in tz.items():
                ok = (z != 0) & np.isfinite(f2[U[idx]])
                if ok.sum() >= 4:
                    ic.setdefault((kind, nm, j), []).append(spearmanr(z[ok], f2[U[idx]][ok]).correlation)
icr = {}
for kind in ('T1', 'T2'):
    for nm in ('R2_5', 'R6_50', 'R51p'):
        vals = [(np.mean(ic[(kind, nm, j)]), L.tstat(ic[(kind, nm, j)])) for j in JS if (kind, nm, j) in ic]
        if vals: icr[f'{kind}_{nm}'] = {'mean_IC_jmedian': float(np.median([v[0] for v in vals])), 't_jmedian': float(np.median([v[1] for v in vals])),
                                         'n_months': len(ic[(kind, nm, 0)]) if (kind, nm, 0) in ic else 0}
EX['E4_within_tier_IC'] = icr
# E5 安慰劑（設計檢定）：每版 1,000 組隨機排列 z，毛報酬；對 B 的「各組平均」月超額
VERS = [(0.5, False), (1.0, False), (1.5, False), (0.5, True), (1.0, True), (1.5, True)]
plc = {}
for kind in ('T1', 'T2'):
    for lam, cap in VERS:
        st = []
        for j in JS:
            ds = T.sig_days(j, SEL_A, SEL_B)
            R, Rb = T.gross_matrix(ds, kind, lam, cap, P=1000, seed=1000 + j)
            x = pd.Series(R.mean(0) - Rb.to_numpy(), index=Rb.index)
            s = T.stats(x); s['perm_mean_sd_pp'] = float((R.mean(1) - Rb.mean()).std() * 100)
            st.append(s)
        key = f'{kind}_lam{lam}_{"cap" if cap else "nocap"}'
        plc[key] = {'mean_pp_jmedian': float(np.median([s['mean_pp'] for s in st])), 't_jmedian': float(np.median([s['t'] for s in st])),
                    't_range': [float(min(s['t'] for s in st)), float(max(s['t'] for s in st))], 'perm_mean_sd_pp_j0': st[0]['perm_mean_sd_pp']}
        print(key, plc[key], flush=True)
EX['E5_placebo_gross_vs_B'] = plc
# E6 子樣本門檻（挑選期中位數／三分位，凍結）
disp, elec, rel = [], [], []
for d in offd[:-1]:
    e = T.IDX[T.IDX.get_loc(d) + 1]
    U, wb, wt, inf = T.target(d, e, 'T1', 0.0, False, info=True)
    y = T.y3_ab(d)[U][~T.FINMASK[U]]; y = y[np.isfinite(y)]
    disp.append(float(np.subtract(*np.percentile(y, [75, 25]))))
    elec.append(float(wb[T.ELECMASK[U]].sum()))
ti = [T.IDX.get_loc(d) for d in offd]
r2330 = [T.factor_close_to_close(a, b)[T.J2330] - 1 for a, b in zip(ti[:-1], ti[1:])]
r50 = T.bench_win(offd).to_numpy()
rel = np.array(r2330) - r50
EX['E6_subsample_thresholds'] = {'rev_yoy_IQR_T1_median': float(np.median(disp)), 'elec_weight_T1_median': float(np.median(elec)),
                                 'tsmc_minus_0050_top_tercile': float(np.quantile(rel, 2 / 3))}
(L.HERE / 'explore.json').write_text(L.dumps(EX), encoding='utf-8')
print(L.dumps({k: v for k, v in EX.items() if k != 'E5_placebo_gross_vs_B'}))
