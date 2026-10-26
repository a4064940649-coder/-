#!/usr/bin/env python3
"""A 機制檢定（不交易）：處置溢酬來自制度，還是來自極端漲幅本身？
樣本：2021-10～2026-09 上市股日（面板），符合 bt_disposal_proxy_history 的價格條件：
  att = 還原價 6 日漲幅 ≥ 32% 且 − 加權指數 6 日漲幅 ≥ 20 個百分點；trig = 連續 3 日 att（參數照原檔）。
  排除 t 當天已在處置期間者。2024-02-15 起 8 個交易日（6 日漲幅跨缺口）不列入。
分組：D＝t 盤後公告處置（t+1 進入處置）；N＝t+1～t+10 都沒有任何處置；L＝t+2～t+10 才進入處置（另列，不進比較）。
每組內同一檔 10 日內重疊的 trig 只留第一筆（另報不去重的全部股日版本）。
配對：t 的 6 日漲幅五分位（全樣本切點）× t 所在月份；只用 D、N 都有的格子；格內差＝D 平均 − N 平均，以格內 D 筆數加權彙總到月，
月份為 cluster：t＝月差平均 / (sd/√月數)。另以「格子固定效果迴歸 + 月份 cluster 穩健標準誤」交叉驗證。
報酬：t+1 收盤 → t+10 收盤的異常報酬（對上市流動性合格等權）；另報 t → t+1（處置第一天）與 t+1 → t+5。
上櫃補充（2025-09-24 起，history.csv.gz）：同條件（仍對加權指數），另列。
門檻：D − N ≥ 2% 且 t ≥ 2 → 「溢酬來自制度」。
"""
import sys, math
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
import adlib as L

att, dp = L.load_events_raw()
TW = L.load_twse(); TP = L.load_tpex()
F = L.calendar(TW['idx'], att, TP['idx'])
codes = sorted(set(att['code']) | set(dp['code']) | set(TW['codes']) | set(TP['codes']))
DPc = L.chains(dp, F)
covF, annF, _ = L.disposal_maps(DPc, F, codes)
tx = pd.read_csv('/home/user/jonatw/finlab-txd/data/raw/taiex_twii.csv', parse_dates=['date']).set_index('date')['close']


def build(M, lo, hi, gap_mask=True):
    idx = M['idx']; adj = M['adj']
    t = tx.reindex(idx).ffill()
    A6 = adj / adj.shift(6) - 1; m6 = t / t.shift(6) - 1
    a = (A6 >= 0.32) & (A6.sub(m6, axis=0) >= 0.20)
    trig = a & a.shift(1, fill_value=False) & a.shift(2, fill_value=False)
    if gap_mask:
        g0 = idx.searchsorted(L.GAP[1], 'right'); trig.iloc[g0:g0 + 8] = False
    trig = trig.to_numpy(); A6n = A6.to_numpy()
    fpos = F.get_indexer(idx); colF = pd.Index(codes).get_indexer(M['codes'])
    Aff = M['A'].to_numpy(); LM = M['LM'].to_numpy(); LQ = M['liq'].to_numpy()
    rows = []
    I, J = np.nonzero(trig)
    for i, j in zip(I, J):
        d = idx[i]
        if d < pd.Timestamp(lo) or d > pd.Timestamp(hi) or i + 10 >= len(idx): continue
        if L.spans_gap(d, idx[i + 10]): continue
        f = fpos[i]; jf = colF[j]
        if f < 0 or jf < 0 or covF[f, jf]: continue
        if F[f + 1] != idx[i + 1]: continue   # 隔日必須是同一交易日（缺口防呆）
        if annF[f, jf]: grp = 'D'
        elif covF[f + 1:f + 11, jf].any(): grp = 'L'
        else: grp = 'N'
        p0, p1, p5, p10 = Aff[i, j], Aff[i + 1, j], Aff[i + 5, j], Aff[i + 10, j]
        if not (np.isfinite(p0) and np.isfinite(p1) and np.isfinite(p10)): continue
        rows.append({'date': d, 'code': M['codes'][j], 'i': i, 'grp': grp, 'run6': A6n[i, j], 'liq': bool(LQ[i, j]),
                     'ab_1_10': (p10 / p1 - 1) - math.expm1(LM[i + 10] - LM[i + 1]),
                     'ab_0_1': (p1 / p0 - 1) - math.expm1(LM[i + 1] - LM[i]),
                     'ab_1_5': (p5 / p1 - 1) - math.expm1(LM[i + 5] - LM[i + 1]),
                     'raw_1_10': p10 / p1 - 1})
    E = pd.DataFrame(rows)
    return E


def dedupe(E):
    keep = []
    for (g, c), x in E.sort_values('i').groupby(['grp', 'code']):
        last = -99
        for k, i in zip(x.index, x['i']):
            if i > last + 10: keep.append(k); last = i
    return E.loc[sorted(keep)]


def matched(E, y='ab_1_10', cuts=None):
    E = E[E['grp'].isin(['D', 'N'])].copy()
    if len(E) < 10 or E['grp'].eq('D').sum() < 3: return {'n_D': int(E['grp'].eq('D').sum()), 'n_N': int(E['grp'].eq('N').sum())}
    if cuts is None: cuts = E['run6'].quantile([0.2, 0.4, 0.6, 0.8]).to_numpy()
    E['q'] = np.searchsorted(cuts, E['run6'].to_numpy(), side='right')
    E['m'] = E['date'].dt.to_period('M')
    cell = E.groupby(['m', 'q', 'grp'])[y].agg(['mean', 'count']).unstack('grp').dropna()
    if not len(cell): return {}
    cell['diff'] = cell[('mean', 'D')] - cell[('mean', 'N')]; cell['w'] = cell[('count', 'D')]
    mon = cell.groupby(level='m').apply(lambda g: np.average(g['diff'], weights=g['w']))
    md = mon.dropna()
    o = {'n_D_total': int(E['grp'].eq('D').sum()), 'n_N_total': int(E['grp'].eq('N').sum()),
         'n_D_matched': int(cell[('count', 'D')].sum()), 'n_N_matched': int(cell[('count', 'N')].sum()),
         'n_cells': int(len(cell)), 'n_months': int(len(md)),
         'D_mean_pct_matched_cells': round(float(np.average(cell[('mean', 'D')], weights=cell['w'])) * 100, 2),
         'N_mean_pct_matched_cells_Dweighted': round(float(np.average(cell[('mean', 'N')], weights=cell['w'])) * 100, 2),
         'diff_D_minus_N_pct': round(float(np.average(cell['diff'], weights=cell['w'])) * 100, 2),
         'diff_month_avg_pct': round(float(md.mean()) * 100, 2), 't_month_cluster': L.tstat(md),
         'median_month_diff_pct': round(float(md.median()) * 100, 2),
         'unmatched_D_mean_pct': round(float(E.loc[E.grp == 'D', y].mean()) * 100, 2), 'unmatched_N_mean_pct': round(float(E.loc[E.grp == 'N', y].mean()) * 100, 2),
         'D_median_pct': round(float(E.loc[E.grp == 'D', y].median()) * 100, 2), 'N_median_pct': round(float(E.loc[E.grp == 'N', y].median()) * 100, 2),
         'D_run6_mean': round(float(E.loc[E.grp == 'D', 'run6'].mean()), 3), 'N_run6_mean': round(float(E.loc[E.grp == 'N', 'run6'].mean()), 3)}
    # 迴歸交叉驗證：y = b·D + 格子固定效果（只保留 D、N 都有的格子），月份 cluster
    keys = set(cell.index)
    S = E[[ (m, q) in keys for m, q in zip(E['m'], E['q'])]].copy()
    S['cid'] = S.groupby(['m', 'q']).ngroup()
    yv = S[y] - S.groupby('cid')[y].transform('mean'); dv = (S['grp'] == 'D').astype(float); dv = dv - dv.groupby(S['cid']).transform('mean')
    b = float((dv * yv).sum() / (dv * dv).sum()); u = yv - b * dv
    sc = (dv * u).groupby(S['m']).sum(); G = len(sc); N = len(S); K = S['cid'].nunique() + 1
    V = (sc ** 2).sum() / (dv * dv).sum() ** 2 * G / (G - 1) * (N - 1) / max(N - K, 1)
    o['fe_reg_coef_pct'] = round(b * 100, 2); o['fe_reg_t_cluster_month'] = round(b / math.sqrt(V), 2) if V > 0 else None
    # winsor / 剔除前 10 的穩健性（以未配對全部樣本的 D 與 N 平均差）
    lo_, hi_ = E[y].quantile([0.05, 0.95]); w = E.assign(yw=E[y].clip(lo_, hi_))
    o['winsor5_diff_unmatched_pct'] = round(float(w.loc[w.grp == 'D', 'yw'].mean() - w.loc[w.grp == 'N', 'yw'].mean()) * 100, 2)
    dsort = E.loc[E.grp == 'D', y].sort_values()
    o['D_ex_top10_mean_pct'] = round(float(dsort.iloc[:-10].mean()) * 100, 2) if len(dsort) > 15 else None
    return o


def pack(E, cuts):
    out = {'group_counts': E['grp'].value_counts().to_dict()}
    E = E.copy()
    q = E['ab_1_10'].quantile([0.05, 0.95]); E['ab_1_10_w5'] = E['ab_1_10'].clip(q.iloc[0], q.iloc[1])
    for y in ['ab_1_10', 'ab_1_10_w5', 'ab_0_1', 'ab_1_5']:
        out[y] = {'all': matched(E, y, cuts)}
        for k, (a, b) in L.SPL.items():
            out[y][k] = matched(E[(E['date'] >= a) & (E['date'] <= b)], y, cuts)
    out['late_group_L_ab_1_10'] = {'n': int((E.grp == 'L').sum()), 'mean_pct': round(float(E.loc[E.grp == 'L', 'ab_1_10'].mean()) * 100, 2) if (E.grp == 'L').any() else None}
    return out


res = {}
E0 = build(TW, '2021-10-01', '2026-09-30')
E = dedupe(E0)
cuts = E[E['grp'].isin(['D', 'N'])]['run6'].quantile([0.2, 0.4, 0.6, 0.8]).to_numpy()
res['quintile_cuts_run6'] = [round(float(c), 4) for c in cuts]
res['TWSE_dedup'] = pack(E, cuts)
res['TWSE_all_stockdays'] = pack(E0, cuts)
res['TWSE_dedup_liquid'] = pack(E[E['liq']], cuts)
pass_main = res['TWSE_dedup']['ab_1_10']['all']
res['verdict'] = 'PASS（溢酬來自制度）' if (pass_main.get('diff_D_minus_N_pct', 0) >= 2 and (pass_main.get('t_month_cluster') or 0) >= 2) else 'FAIL'
# 價格條件的召回：真實處置（上市、2021-10～2026-09）中，公告日當天符合 trig 的比例
ET = build(TP, '2025-09-24', '2026-09-30', gap_mask=False)
res['TPEX_supplement_dedup'] = pack(dedupe(ET), cuts) if len(ET) else {}
L.save(res, 'results_A.json')
import pprint
for k in ['quintile_cuts_run6', 'verdict']: print(k, res[k])
for blk in ['TWSE_dedup', 'TWSE_all_stockdays', 'TWSE_dedup_liquid', 'TPEX_supplement_dedup']:
    print('\n##', blk, res[blk].get('group_counts'), res[blk].get('late_group_L_ab_1_10'))
    for y in ['ab_1_10', 'ab_1_10_w5']:
        for p, x in res[blk].get(y, {}).items():
            print(' ', y, p, {k: x.get(k) for k in ['n_D_matched', 'n_N_matched', 'n_months', 'D_mean_pct_matched_cells', 'N_mean_pct_matched_cells_Dweighted', 'diff_D_minus_N_pct', 'diff_month_avg_pct', 't_month_cluster', 'fe_reg_coef_pct', 'fe_reg_t_cluster_month', 'winsor5_diff_unmatched_pct', 'unmatched_D_mean_pct', 'unmatched_N_mean_pct', 'D_median_pct', 'N_median_pct', 'D_ex_top10_mean_pct']})
