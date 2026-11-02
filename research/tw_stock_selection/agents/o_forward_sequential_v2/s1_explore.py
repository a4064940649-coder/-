#!/usr/bin/env python3
"""步驟 1：探索（只用挑選期 SEL：錨定月 ≤ 2019-04）。描述統計，寫 explore.json。
- 8 個可執行格（B10 K5T 一般成本 j 中位數）在 SEL 的月超額：均值、iid sd、HAC sd、AR1、峰態、12 個月滾動 sd 的範圍（波動狀態）。
- 官方日 B10（零衝擊）與 j 中位數格的差。
- 可行性：σ 下 α＝−1pp／月時 36 個月累積 z 的期望值。
- 族群（2,319 條、V4 格點）在 SEL 的窗數與第 114 群大小。
用法：python3 -I -B s1_explore.py
"""
import sys, json, math
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
from scipy import stats
import fslib as F


def desc(x):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    r = pd.Series(x)
    roll = r.rolling(12).std().dropna()
    return dict(n=len(x), mean=float(x.mean()), sd=float(x.std(ddof=1)), hac_sd=F.hac_sd(x), ar1=float(r.autocorr(1)),
                ar2=float(r.autocorr(2)), exkurt=float(stats.kurtosis(x)), skew=float(stats.skew(x)),
                t=float(x.mean() / x.std(ddof=1) * math.sqrt(len(x))),
                roll12_sd_min=float(roll.min()), roll12_sd_max=float(roll.max()), roll12_sd_p10=float(roll.quantile(.1)), roll12_sd_p90=float(roll.quantile(.9)),
                share_abs_gt_2sd=float(np.mean(np.abs(x - x.mean()) > 2 * x.std(ddof=1))))


def main():
    out = {'scope': 'SEL only（錨定月 2006-04～2019-04）', 'utc': F.now_utc()}
    cells = {}
    for tier in F.TIERS:
        for mode in F.MODES:
            M, _, _ = F.cell_jmatrix(tier, mode)
            s = F.jmed(M)
            sm = s[F.per_mask(s.index, None, F.SEL_END)]
            Mm = M[F.per_mask(M.index, None, F.SEL_END)]
            d = desc(sm)
            d['j_mean_by_j'] = {int(j): float(Mm[j].mean()) for j in F.JS}
            d['j_sd_of_means'] = float(Mm.mean().std())
            cells[f'{tier}|{mode}'] = d
    out['cells_jmed_K5T_normal'] = cells
    # 壓力成本主格
    M, _, _ = F.cell_jmatrix(*F.MAIN_CELL, cost='stress')
    s = F.jmed(M); out['main_cell_stress'] = desc(s[F.per_mask(s.index, None, F.SEL_END)])
    # 官方日（零衝擊 B10）
    O = F.official_series()
    o = O.ex[F.per_mask(O.index, None, F.SEL_END)]
    out['official_B10_zero_impact'] = desc(o)
    Mmain, _, _ = F.cell_jmatrix(*F.MAIN_CELL)
    jm = F.jmed(Mmain); jm = jm[F.per_mask(jm.index, None, F.SEL_END)]
    common = o.index.intersection(jm.index)
    out['official_minus_jmed_main_cell'] = dict(mean=float((o.reindex(common) - jm.reindex(common)).mean()),
                                                corr=float(np.corrcoef(o.reindex(common), jm.reindex(common))[0, 1]))
    # 可行性（事前預測 1 的計算）
    sig = cells['|'.join(F.MAIN_CELL)]['hac_sd']; sig_iid = cells['|'.join(F.MAIN_CELL)]['sd']
    out['feasibility'] = {
        'sigma_main_hac': sig, 'sigma_main_iid': sig_iid,
        'z36_alpha_m1': -36 / (sig * 6), 'z48_alpha_m1': -48 / (sig * math.sqrt(48)), 'z60_alpha_m1': -60 / (sig * math.sqrt(60)),
        'single_look_false_down_at_z36': float(stats.norm.cdf(-36 / (sig * 6))),
        'single_look_up_power_alpha_p1_at_z1.96': float(1 - stats.norm.cdf(1.96 - 36 / (sig * 6))),
    }
    # 族群
    X, lab, med, anc = F.family()
    sel = F.per_mask(anc, None, F.SEL_END)
    labv = np.asarray(lab, int)
    out['family'] = dict(n_windows=int(len(anc)), n_sel_windows=int(sel.sum()), n_members=int(X.shape[1]), n_clusters=int(len(med)),
                         n_clusters_m_ge5=int(sum((labv == c).sum() >= 5 for c in med)), main_cluster_size=int((labv == F.MAIN_CLUSTER).sum()),
                         sel_member_mean_sd=float(X.values[sel].mean(axis=0).std()), sel_medoid_mean_sd=float(X.values[sel][:, [med[c] for c in med]].mean(axis=0).std()))
    # 已知（o7 已看過的全期 CV，僅記錄，不當本題證據）
    ot = json.loads((F.VEN / 'o_tier_results.json').read_text())
    out['known_o7_fullsample_cv'] = {f"{r['tag']}|{r['ver']}": dict(sum_logpd=round(r['sum_logpd'], 2), cov80=round(r['cov80'], 3), slope=round(r['slope_y_on_post'], 3))
                                     for r in ot['STEP1_CV']['summary'] if r['tag'] == 'sel'}
    (F.HERE / 'explore.json').write_text(json.dumps(out, ensure_ascii=False, indent=1, default=F._js), encoding='utf-8')
    for k, v in cells.items(): print(k, {a: round(v[a], 3) for a in ('n', 'mean', 'sd', 'hac_sd', 'ar1', 'exkurt', 'roll12_sd_min', 'roll12_sd_max')})
    print('official', {a: round(out['official_B10_zero_impact'][a], 3) for a in ('mean', 'sd', 'hac_sd')}, out['official_minus_jmed_main_cell'])
    print(json.dumps(out['feasibility'], indent=0)); print(out['family'])


if __name__ == '__main__':
    main()
