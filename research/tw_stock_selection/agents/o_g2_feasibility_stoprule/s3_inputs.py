#!/usr/bin/env python3
"""模擬輸入（第二次登記前產生）：data/sim_inputs.npz（只含彙總序列，< 1MB）與 explore.json['sim_inputs'] 描述統計。
內容：AB_V4 月超額（ε 與 σ 來源）、0050 窗報酬、V7X 13 j 偏離量 d_j、壓力拖累、300 萬／1,000 萬拖累、6 個鄰居差 η、
族群 SPA_c 最大值 bootstrap 分布 Tb（全族群、F_main 家族）、族群月 SR（DSR 用）、V7X javg 序列（變體 V6）、日曆年與分段標籤。"""
import json, sys
import numpy as np, pandas as pd
sys.path.insert(0, __import__('os').path.dirname(__file__))
import g2lib as G

G.check_prereg()
X, meta = G.family()
n = X.shape[0]
AB = 'o_v4_revenue_patch_official/navs_v4.parquet::AB_V4|normal'
ab = X[AB].fillna(0).to_numpy()
# V7X
def jwin(cell, cost):
    P = G.v7x_paths(cell, cost)
    return np.column_stack([G.win_ex(P[j]).to_numpy() for j in range(13)])
XJ = jwin('100萬', 'normal'); XS = jwin('100萬', 'stress'); X3 = jwin('300萬', 'normal'); X10 = jwin('1000萬', 'normal')
for A in (XJ, XS, X3, X10): A[~np.isfinite(A)] = 0.0
d = XJ - XJ.mean(1, keepdims=True)
c_stress = (XJ - XS).mean(1); c300 = (XJ - X3).mean(1); c1000 = (XJ - X10).mean(1)
xv = G.win_ex(G.javg_nav(G.v7x_paths('100萬', 'normal'))).fillna(0).to_numpy()
# 鄰居
cols = list(X.columns)
def pick(pat):
    c = [k for k in cols if k.endswith(pat)]; assert c, pat; return c[0]
NB = {'AB_V4_B05': pick('::AB_V4_B05|normal'), 'AB_V4_B10': pick('::AB_V4_B10|normal'), 'ABW_V4': pick('::ABW_V4|normal'),
      'main_v2_EXFIN_n30': pick('::EXFIN|n30|normal'), 'conc_EW5': pick('::EW5'), 'conc_EW15': pick('::EW15')}
eta_raw = np.column_stack([X[v].fillna(0).to_numpy() - ab for v in NB.values()])
# 族群 SPA_c 最大值分布（全族群 2,334；F_main 家族）
Xf = X.fillna(0.0).to_numpy()
W = G.SP.sb_counts(n, 6, 10000, 20261102)
def tb_of(M):
    Mb = G.SP.boot_means(W, M)
    om2 = G.SP.lrv(M, 'pr', 6); om = np.sqrt(om2); sn = np.sqrt(n); dbar = M.mean(0)
    thr = -np.sqrt(om2 / n * 2 * np.log(np.log(n)))
    gc = np.where(dbar >= thr, dbar, 0.0)
    Z = (Mb - gc[None, :].astype(np.float32)) * (sn / om).astype(np.float32)[None, :]
    return np.maximum(Z.max(axis=1), 0.0)
ok = np.std(Xf, 0) > 0
Tb_all = tb_of(Xf[:, ok])
fm = meta.F_main.to_numpy().astype(bool) & ok
Tb_main = tb_of(Xf[:, fm])
sr_all = (Xf[:, ok].mean(0) / Xf[:, ok].std(0, ddof=1))
years = G.WIN.year.to_numpy()
seg = np.select([G.pmask('SEG1'), G.pmask('SEG2'), G.pmask('SEG3')], [1, 2, 3], 0)
y19 = G.pmask('Y19')
np.savez_compressed(G.DATA / 'sim_inputs.npz', ab=ab, r50=G.R50.to_numpy(), d=d, c_stress=c_stress, c300=c300, c1000=c1000, eta_raw=eta_raw,
                    Tb_all=Tb_all, Tb_main=Tb_main, sr_all=sr_all, xv=xv, years=years, seg=seg, y19=y19)
info = {'n_windows': n, 'AB_V4': {'mean_pp_m': float(ab.mean() * 100), 'sd_pp_m': float(ab.std(ddof=1) * 100), 'TE_ann': float(ab.std(ddof=1) * np.sqrt(12)),
                                   'TE_ann_Y19': float(ab[y19].std(ddof=1) * np.sqrt(12)), 'IR': G.ir(ab), 'skew': float(pd.Series(ab).skew()), 'kurt': float(pd.Series(ab).kurt()),
                                   'ac1': float(pd.Series(ab).autocorr(1))},
        'V7X_javg': {'IR': G.ir(xv), 'TE_ann': float(xv.std(ddof=1) * np.sqrt(12)), 't': G.tstat(xv)},
        'd_j_mean_pp_m': (d.mean(0) * 100).round(3).tolist(), 'd_j_sd_pp_m': (d.std(0) * 100).round(3).tolist(),
        'd_j_mean_Y19_pp_m': (d[y19].mean(0) * 100).round(3).tolist(),
        'c_stress_pp_m': {'all': float(c_stress.mean() * 100), 'Y19': float(c_stress[y19].mean() * 100)},
        'c300_pp_m': {'all': float(c300.mean() * 100), 'Y19': float(c300[y19].mean() * 100)}, 'c1000_pp_m': {'all': float(c1000.mean() * 100), 'Y19': float(c1000[y19].mean() * 100)},
        'neighbors': {k: {'member': v, 'corr_with_AB_V4': float(np.corrcoef(X[v].fillna(0), ab)[0, 1]), 'mean_diff_pp_m': float(eta_raw[:, i].mean() * 100),
                          'sd_diff_pp_m': float(eta_raw[:, i].std() * 100)} for i, (k, v) in enumerate(NB.items())},
        'Tb_all': {'N': int(ok.sum()), 'q50': float(np.quantile(Tb_all, .5)), 'q90': float(np.quantile(Tb_all, .9)), 'q95': float(np.quantile(Tb_all, .95))},
        'Tb_main': {'N': int(fm.sum()), 'q90': float(np.quantile(Tb_main, .9)), 'q95': float(np.quantile(Tb_main, .95))},
        'var_SR_family': float(np.var(sr_all, ddof=1))}
ex = json.loads((G.HERE / 'explore.json').read_text())
ex['sim_inputs'] = info
(G.HERE / 'explore.json').write_text(json.dumps(ex, ensure_ascii=False, indent=1, default=float))
print(json.dumps(info, ensure_ascii=False, indent=1, default=float))
