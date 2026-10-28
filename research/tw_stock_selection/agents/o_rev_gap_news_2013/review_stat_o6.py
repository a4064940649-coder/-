#!/usr/bin/env python3
"""反方審查（資料與統計）o6：不改研究員檔案，只讀。
A. 提及表 PIT 健全性：營收月與文章日期的落差（lag 0＝營收月當月就有「該月營收」→ 不可能是實際公告值）。
B. G1 覆蓋的 175 格 N1 代理：用了幾個新聞月、靠哪些來源、是否用到 lag 0 提及。
C. 「B10 入選 29 格代理百分位中位數 0.32」：依股票集中度、以股票為群的 bootstrap、有效樣本數。
D. 驗證（Spearman 0.879）是否可移轉到缺口格：缺口格多半缺基期，驗證卻用官方基期；改用「不用官方基期」重算 N1 驗證。
E. 有覆蓋 vs 無覆蓋的 B10 入選格特徵比較（選樣偏誤）。
"""
import sys, json
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import numpy as np, pandas as pd
from scipy.stats import spearmanr
import rglib as G
import proxy as PX
K, V, L = G.K, G.V, G.L
out = {}
M, LV, L5, Q = PX.load_inputs(K, HERE / 'data' / 'mentions_P3.parquet')
M['ts'] = pd.to_datetime(M.ts)
M['lag'] = (M.ts.dt.year - M.ym.str[:4].astype(int)) * 12 + M.ts.dt.month - M.ym.str[5:7].astype(int)
out['A_lag_dist'] = M.lag.value_counts().sort_index().to_dict()
out['A_lag0_share'] = float((M.lag == 0).mean())
# lag 0 提及與官方水準比對
def offlv(c, ym):
    p = pd.Period(ym, 'M')
    for X in (LV, L5):
        if c in X.columns and p in X.index:
            v = X.at[p, c]
            if np.isfinite(v) and v > 0: return v
    return np.nan
M['off'] = [offlv(c, y) for c, y in zip(M.code, M.ym)]
M['err'] = (M.rev_k / M.off - 1).abs()
out['A_level_err_by_lag'] = {int(k): {'n': int(g.err.notna().sum()), 'share_err_gt_5pct': float((g.err > 0.05).mean()) if g.err.notna().any() else None}
                             for k, g in M[M.err.notna()].groupby('lag')}
# 營收月次月 1 日之前（日期而言不可能已公告）
M['impossible'] = M.ts < (pd.PeriodIndex(M.ym, freq='M') + 1).start_time
out['A_ts_before_next_month'] = int(M.impossible.sum())

b = G.base(); S = G.mk()
P = PX.Proxy(M.drop(columns=['lag', 'off', 'err', 'impossible']), LV, L5, Q, S.YL, S.Y5, V.SWITCH)
X = pd.read_csv(HERE / 'gap_cells_2013_2021.csv', dtype={'code': str})
T = pd.read_csv(HERE / 'data' / 'proxy_table_gap_cells.csv', dtype={'code': str})
assert (T.sig.values == X.sig.values).all() and (T.code.values == X.code.values).all()
# B. N1 覆蓋格的組成
rows = []
for s, c, t, ib, v in zip(T.sig, T.code, T.type, T.in_B10, T.proxy_N1):
    if not np.isfinite(v): continue
    d = pd.Timestamp(s); mv = P.month_vals(c, d)
    nn = sum(1 for _, src in mv if src == 'news'); no = sum(1 for _, src in mv if src == 'off')
    # 用到的提及
    lim = d + PX.CUT; m = pd.Period(d, 'M'); srcs = set(); lag0 = False
    for p in (m - 1, m - 2, m - 3):
        g = M[(M.code == c) & (M.ym == str(p)) & (M.ts <= lim)]
        if len(g): srcs |= set(g.src); lag0 |= bool((g.lag == 0).any())
    rows.append((s, c, t, ib, v, nn, no, '+'.join(sorted(srcs)), lag0))
B = pd.DataFrame(rows, columns=['sig', 'code', 'type', 'in_B10', 'v', 'n_news', 'n_off', 'srcs', 'lag0'])
out['B_n1_cov_cells'] = {'n': int(len(B)), 'n_news_months_dist': B.n_news.value_counts().to_dict(), 'n_off_months_dist': B.n_off.value_counts().to_dict(),
                         'srcs': B.srcs.value_counts().to_dict(), 'uses_lag0': int(B.lag0.sum()), 'n_codes': int(B.code.nunique()),
                         'top_codes': B.code.value_counts().head(10).to_dict()}
# C. 29 格集中度＋股票群 bootstrap
Bb = B[B.in_B10].copy()
pct = []
for s, c, v in zip(Bb.sig, Bb.code, Bb.v):
    d = pd.Timestamp(s); cs = S.cross_section(d); y3 = S.y3(d)
    el = cs['ok_other'] & np.isfinite(y3); tv = y3[el]
    pct.append(float((tv < v).mean() + 0.5 * (tv == v).mean()))
Bb['pct'] = pct
out['C_b10_n1'] = {'n': int(len(Bb)), 'n_codes': int(Bb.code.nunique()), 'n_sig_days': int(Bb.sig.nunique()),
                   'by_code': Bb.groupby('code').pct.agg(['size', 'median']).round(3).reset_index().values.tolist(),
                   'median': float(Bb.pct.median()), 'years': Bb.sig.str[:4].value_counts().sort_index().to_dict(),
                   'n_news_months': Bb.n_news.value_counts().to_dict(), 'n_off_months': Bb.n_off.value_counts().to_dict()}
rng = np.random.default_rng(6)
codes = Bb.code.unique(); gb = {c: Bb[Bb.code == c].pct.to_numpy() for c in codes}
bm, bge = [], []
for _ in range(5000):
    pick = rng.choice(codes, len(codes))
    a = np.concatenate([gb[c] for c in pick]); bm.append(np.median(a)); bge.append((a >= 0.9).mean())
out['C_cluster_boot_by_code'] = {'median_95': [float(np.percentile(bm, 2.5)), float(np.percentile(bm, 97.5))],
                                 'share_ge0.9_95': [float(np.percentile(bge, 2.5)), float(np.percentile(bge, 97.5))]}
# 若真值是均勻分配（隨機股），各股取一格中位數 → 檢驗「≥0.9 為 0」的機率（以股票數為獨立樣本）
nc = len(codes)
out['C_prob_zero_ge0.9_if_uniform_indep_codes'] = float(0.9 ** nc)
# E. B10 入選 133：有覆蓋 vs 無覆蓋
X['cov_N1'] = np.isfinite(T.proxy_N1); X['cov_NQ'] = np.isfinite(T.proxy_NQ)
E = X[X.in_B10]
out['E_b10_covered_vs_not'] = {k: {'n': int(len(g)), 'liq_rank_med': float(g.liq_rank.median()), 's52_pct_med': float(g.s52_pct.median()),
                                   'type': g.type.value_counts().to_dict(), 'n_codes': int(g.code.nunique()), 'top_codes': g.code.value_counts().head(6).to_dict()}
                               for k, g in E.groupby('cov_N1')}
# D. 驗證：不用官方基期（模擬缺口格多半缺基期）
def n1_nobase(c, d):
    m = pd.Period(d, 'M'); vals = []
    for p in (m - 1, m - 2, m - 3):
        v = P.news_yoy(c, p, d, base_official=False)
        if np.isfinite(v): vals.append(v)
    return float(np.mean(vals)) if vals else np.nan
def n1_base(c, d):
    return P.y3(c, d, 'N1', only_news=True)[0]
tp_b, pp_b, tp_n, pp_n, yrs_b = [], [], [], [], []
ncell = 0
for d in S.sig_days:
    if not (G.W0 <= d <= G.W1): continue
    cs = S.cross_section(d); y3 = S.y3(d)
    el = cs['ok_other'] & np.isfinite(y3); J = np.where(el)[0]
    if len(J) < 50: continue
    tv = y3[J]; srt = np.sort(tv); n = len(srt) - 1
    def pc(x, sv):
        r = np.searchsorted(srt, x, 'left') + 0.5 * (np.searchsorted(srt, x, 'right') - np.searchsorted(srt, x, 'left'))
        if sv < x: r -= 1
        elif sv == x: r -= 0.5
        return r / n
    ncell += len(J)
    for k, j in enumerate(J):
        c = K.COLS[j]
        if (c, ) and P.M.get((c, str(pd.Period(d, 'M') - 1))) is None and P.M.get((c, str(pd.Period(d, 'M') - 2))) is None and P.M.get((c, str(pd.Period(d, 'M') - 3))) is None:
            continue
        vb = n1_base(c, d); vn = n1_nobase(c, d)
        if np.isfinite(vb): tp_b.append(pc(tv[k], tv[k])); pp_b.append(pc(vb, tv[k])); yrs_b.append(d.year)
        if np.isfinite(vn): tp_n.append(pc(tv[k], tv[k])); pp_n.append(pc(vn, tv[k]))
tp_b, pp_b, tp_n, pp_n, yrs_b = map(np.array, (tp_b, pp_b, tp_n, pp_n, yrs_b))
out['D_validation'] = {'n_cells': ncell, 'N1_with_off_base': {'n': int(len(tp_b)), 'cov': len(tp_b) / ncell, 'spearman': float(spearmanr(tp_b, pp_b).correlation)},
                       'N1_no_off_base': {'n': int(len(tp_n)), 'cov': len(tp_n) / ncell, 'spearman': float(spearmanr(tp_n, pp_n).correlation) if len(tp_n) > 10 else None},
                       'N1_with_base_by_year': {int(y): [int((yrs_b == y).sum()), float(spearmanr(tp_b[yrs_b == y], pp_b[yrs_b == y]).correlation)]
                                                for y in np.unique(yrs_b) if (yrs_b == y).sum() > 10},
                       'N1_true_pct_of_validation_cells_median': float(np.median(tp_b)), 'N1_proxy_pct_le_true_share': float((pp_b < tp_b).mean()),
                       'N1_proxy_minus_true_median': float(np.median(pp_b - tp_b))}
print(json.dumps(out, ensure_ascii=False, indent=1, default=str))
# F. 驗證依新聞月數分組（缺口覆蓋 175 格中 115 格只有 1 個新聞月）＋2013～2014（缺口格 64%）
tp, pp, nm, yy = [], [], [], []
for d in S.sig_days:
    if not (G.W0 <= d <= G.W1): continue
    cs = S.cross_section(d); y3 = S.y3(d)
    el = cs['ok_other'] & np.isfinite(y3); J = np.where(el)[0]
    if len(J) < 50: continue
    tv = y3[J]; srt = np.sort(tv); n = len(srt) - 1
    def pc(x, sv):
        r = np.searchsorted(srt, x, 'left') + 0.5 * (np.searchsorted(srt, x, 'right') - np.searchsorted(srt, x, 'left'))
        if sv < x: r -= 1
        elif sv == x: r -= 0.5
        return r / n
    for k, j in enumerate(J):
        v, nn = P.y3(K.COLS[j], d, 'N1', only_news=True)
        if np.isfinite(v): tp.append(pc(tv[k], tv[k])); pp.append(pc(v, tv[k])); nm.append(nn); yy.append(d.year)
tp, pp, nm, yy = map(np.array, (tp, pp, nm, yy))
F_ = {}
for lab, m in {'nnews=1': nm == 1, 'nnews>=2': nm >= 2, '2013-2014': yy <= 2014, '2013-2014_nnews=1': (yy <= 2014) & (nm == 1)}.items():
    F_[lab] = {'n': int(m.sum()), 'spearman': float(spearmanr(tp[m], pp[m]).correlation)}
# 以缺口 N1 覆蓋格的（年, 新聞月數）分布加權
w = B.assign(y=B.sig.str[:4].astype(int)).groupby(['y', 'n_news']).size()
acc = 0; tot = 0
for (y_, k_), c_ in w.items():
    m = (yy == y_) & (np.minimum(nm, 3) == k_)
    if m.sum() > 10: acc += c_ * spearmanr(tp[m], pp[m]).correlation; tot += c_
F_['gapcov_weighted_spearman'] = float(acc / tot) if tot else None; F_['gapcov_weighted_n'] = int(tot)
print(json.dumps({'F_validation_by_nnews_year': F_}, ensure_ascii=False, indent=1))
