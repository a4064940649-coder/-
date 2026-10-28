#!/usr/bin/env python3
"""反方審查（資料與統計角度，o6）：只讀研究員的檔案，不寫 results.json。
用法：python3 -I -B review_stat_o6.py
檢查：
 R1 G2 設定敏感度（c0 0.01、格點 [−6,6]）與 G3 下的「事後÷原始」是否仍 ≥ 0.80（可證偽條件）
 R2 以 G2 合成驗證的右尾殘留偏誤比做粗略校正
 R3 群集內挑選（228 條變體中選 AB）：以群集內成員做第二層收縮
 R4 j 平移：官方日在 j＝0～12 分布中的排名、平均 vs 中位數
 R5 年份與 HOLD：EXEC_J 逐年超額、HOLD 期 t
 R6 族群前 165 名中有多少是主策略鏈（同群集）——排名 165 的意義
"""
import sys, json, math
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import sc_lib as S

out = {}
X, M = S.load_family()
F = S.load_focus()
fam = json.loads((S.DATA / 'families.json').read_text(encoding='utf-8'))
meta = json.loads((S.DATA / 'focus_meta.json').read_text(encoding='utf-8'))
sdc = meta['cost_uncertainty']['sd_cost_pp']
fx = {c: (F[c] - F['r0050']).values * 100 for c in ['AB', 'B10', 'EXEC', 'EXEC_J']}
COST = {'EXEC', 'EXEC_J'}
dec = S.decompose(X[fam['MAIN']], fx, mode='decomp')

def post_g2(c0, lo, hi, model='G2'):
    S.GRID_LO, S.GRID_HI = lo, hi
    if model == 'G2':
        pr = S.fit_g2(dec['xhat'], dec['s2'], c0=c0, th=S.theta_grid())
    else:
        pr = S.fit_g3(dec['xhat'], dec['s2'], th=S.theta_grid())
    r = {}
    for c in fx:
        s2f = dec['focus_s2'][c] + (sdc ** 2 if c in COST else 0)
        p = S.posterior(pr, model, dec['focus_xhat'][c], s2f, dec['vP'])
        r[c] = {'mean': round(p['mean'], 3), 'ratio': round(p['mean'] / dec['focus_xhat'][c], 3), 'p_pos': round(p['p_pos'], 3)}
    S.GRID_LO, S.GRID_HI = -8.0, 4.0
    return r

R1 = {}
for nm, args in [('G2_c0.1_[-8,4]_main', (0.1, -8, 4)), ('G2_c0.01_[-8,4]', (0.01, -8, 4)), ('G2_c0.1_[-6,6]', (0.1, -6, 6)),
                 ('G2_c0.01_[-6,6]', (0.01, -6, 6)), ('G2_c1_[-8,4]', (1.0, -8, 4))]:
    R1[nm] = post_g2(*args)
R1['G3_[-8,4]'] = post_g2(None, -8, 4, model='G3')
out['R1_model_sens'] = R1

# R2：用 v3 驗證的「最大者偏誤比」b 粗校正：post = truth + b(raw − truth) → truth = (post − b·raw)/(1−b)
R2 = {}
for b in (0.22, 0.26, 0.34):
    R2[str(b)] = {c: round((R1['G2_c0.1_[-8,4]_main'][c]['mean'] - b * dec['focus_xhat'][c]) / (1 - b) / dec['focus_xhat'][c], 3) for c in fx}
out['R2_biascorr_post_over_raw'] = R2

# R3：群集內挑選
cl = pd.read_csv(S.DATA / 'clusters_k161.csv')
c114 = cl[cl.cluster == fam['meta']['main_cluster']].member_id.tolist()
xc = X[c114].mean(axis=0) if False else X[c114].values.mean(axis=0)
ab = dec['focus_xhat']['AB']
rank = int((xc > ab - 1e-9).sum())
dc = S.decompose(X[c114], fx, mode='decomp')
g1 = S.fit_g1(dc['xhat'], dc['s2'])
g3 = S.fit_g3(dc['xhat'], dc['s2'])
R3 = {'n': len(c114), 'xhat_mean': float(xc.mean()), 'xhat_sd': float(xc.std()), 'xhat_max': float(xc.max()), 'AB_rank_in_cluster': rank,
      'AB_xhat': ab, 'cluster_idio_se_median': float(np.median(np.sqrt(dc['s2']))), 'G1': {k: round(v, 4) for k, v in g1.items()}}
for c in fx:
    s2f = dc['focus_s2'][c] + (sdc ** 2 if c in COST else 0)
    p1 = S.posterior(g1, 'G1', dc['focus_xhat'][c], s2f, 0.0)
    p3 = S.posterior(g3, 'G3', dc['focus_xhat'][c], s2f, 0.0)
    R3[c] = {'raw': round(dc['focus_xhat'][c], 3), 'idio_se': round(math.sqrt(s2f), 3), 'G1_within_mean': round(p1['mean'], 3), 'G3_within_mean': round(p3['mean'], 3)}
out['R3_within_cluster'] = R3

# R4：j 平移
js = meta['j_shift']; jm = np.array(list(js['j_means_pp'].values())); off = js['AB_V5_official_mean_pp']
out['R4_j'] = {'official': off, 'n_j': len(jm), 'n_j_ge_official': int((jm >= off).sum()), 'j_mean': float(jm.mean()), 'j_median': float(np.median(jm)),
               'j_sd': float(jm.std(ddof=1)), 'j_min': float(jm.min()), 'delta_mean': float(jm.mean() - off), 'delta_q25': float(np.quantile(jm, 0.25) - off),
               'EXEC_J_raw_if_mean': float(fx['EXEC'].mean() + jm.mean() - off), 'EXEC_J_raw_if_q25': float(fx['EXEC'].mean() + np.quantile(jm, 0.25) - off),
               'corr_j_index_vs_mean': float(np.corrcoef(np.arange(len(jm)), jm)[0, 1])}

# R5：年份、HOLD、最好 10 窗
e = pd.Series(fx['EXEC_J'], index=F.index)
yr = e.groupby(e.index.year).sum()
out['R5_year_sum_EXEC_J_pp'] = {int(k): round(float(v), 2) for k, v in yr.items()}
tot = e.sum()
out['R5_total_sum_pp'] = float(tot)
out['R5_top3_years_share'] = float(yr.sort_values(ascending=False).iloc[:3].sum() / tot)
out['R5_best10_windows_share'] = float(e.sort_values(ascending=False).iloc[:10].sum() / tot)
for nm, (a, b) in (('SEL', S.SEL), ('VAL', S.VAL), ('HOLD', S.HOLD)):
    m = S.wmask(e.index, a, b); s = e[m]
    out[f'R5_{nm}_EXEC_J'] = {'n': int(m.sum()), 'mean': round(float(s.mean()), 3), 't': round(S.tstat(s.values), 2),
                              't_hac6': round(float(s.mean() / math.sqrt(S.hac_var_mean(s.values, 6))), 2)}
    s2 = pd.Series(fx['AB'], index=F.index)[m]
    out[f'R5_{nm}_AB'] = {'mean': round(float(s2.mean()), 3), 't_hac6': round(float(s2.mean() / math.sqrt(S.hac_var_mean(s2.values, 6))), 2)}
# 去掉 2023-12-11 缺口窗
gw = pd.Timestamp('2023-12-11')
out['R5_gap_window'] = {c: round(float(pd.Series(fx[c], index=F.index).get(gw, np.nan)), 2) for c in fx}
out['R5_EXEC_J_mean_excl_gapwin'] = float(e.drop(gw).mean()) if gw in e.index else None

# R6：族群排名
allx = X.values.mean(axis=0)
order = np.argsort(-allx)
top = [X.columns[i] for i in order[:200]]
lab = dict(zip(cl.member_id, cl.cluster))
out['R6_AB_rank_ALL'] = int((allx > ab - 1e-9).sum())
out['R6_top165_cluster114_share'] = float(np.mean([lab.get(t) == fam['meta']['main_cluster'] for t in top[:165]]))
out['R6_top20_clusters'] = [int(lab.get(t, -1)) for t in top[:20]]

print(json.dumps(out, ensure_ascii=False, indent=1, default=float))
