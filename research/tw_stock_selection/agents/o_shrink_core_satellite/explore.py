#!/usr/bin/env python3
"""步驟 1：探索（α̂ 相關的描述只用挑選期 SEL 2006-04～2019-04 的 157 窗；相關結構與群集用 W_long 相關係數，
這是事前登記的族群定義，不含 α 資訊）。不做任何收縮計算。輸出 explore.json。"""
import sys, json
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import sc_lib as S

THEORY_FOLDERS = ['momentum_family', 'revenue_longhistory', 'rev_announce', 'w12_residual_momentum', 'w3_frog_in_pan_momentum',
                  'w3_idio_vol', 'w3_amihud_illiquidity', 'w3_size_rotation_28', 'w3_revenue_sue', 'w3_52w_low_nearness',
                  'w3_price_delay', 'w3_high_volume_premium', 'w3_weekly_reversal', 'w12_limit_hit_lottery',
                  'o_msci_inclusion_avoid', 'o_issuance_avoid', 'o_cb_board_runup', 'o_largecap_leg', 'o_disposal_rejudge']

def q(x, ps=(5, 10, 25, 50, 75, 90, 95)):
    return {f'p{p}': float(np.percentile(x, p)) for p in ps}

def main():
    X, M = S.load_family()
    F = S.load_focus()
    meta = json.loads((S.DATA / 'focus_meta.json').read_text(encoding='utf-8'))
    idx = X.index
    sel = S.wmask(idx, *S.SEL)
    out = {'note': 'α̂ 描述只用 SEL 157 窗；群集用 W_long 相關係數（族群定義，事前登記）',
           'data_checks': {'non_holiday_nav_gaps': [g for g in meta['nav_gaps_gt5_weekdays'] if g['missing_weekdays'] > 8],
                           'long_windows': meta['long_windows_gt27_weekdays'],
                           'family_windows': int(len(idx)), 'family_first': str(idx[0].date()), 'family_last_start': str(idx[-1].date()),
                           'family_nan_cells_before_fill': None}}
    D0 = pd.read_parquet(S.SPA / 'data' / 'family_excess.parquet')[M.member_id.tolist()]
    na = D0.isna()
    out['data_checks']['family_nan_cells_before_fill'] = int(na.values.sum())
    out['data_checks']['members_with_filled_windows'] = int((na.sum() > 0).sum())
    out['data_checks']['nan_cells_by_window_top'] = {str(k.date()): int(v) for k, v in na.sum(axis=1).sort_values(ascending=False).head(8).items()}
    # 2024-01：窗 2023-12-11 跨缺口
    w2312 = pd.Timestamp('2023-12-11')
    out['data_checks']['window_2023_12_11'] = {'focus_excess_pp': {c: float((F.loc[w2312, c] - F.loc[w2312, 'r0050']) * 100) for c in ['AB', 'B10', 'EXEC']},
                                              'r0050_pct': float(F.loc[w2312, 'r0050'] * 100),
                                              'family_median_excess_pp': float(X.loc[w2312].median())}
    # 成員的歷史長度與成本口徑
    out['members'] = {'n_F_long': int(len(M)), 'n_F_main': int(M.F_main.sum()),
                      'source_counts': M.source.value_counts().to_dict(),
                      'miss_W_long_counts': M.miss_W_long.value_counts().sort_index().to_dict(),
                      'cost_note': '每條成員只留該研究自己的一般成本版（SPA 題規則）；主策略鏈＝滑價 0.1%＋手續費 6 折＋稅 0.3%（來回約 0.47%），'
                                   '網路規則與組合多為 BRIEF 一般成本 0.5% 來回；strategy_ensemble 重跑沿用 elib 一般成本。不一致者：miss_W_long>0 的成員（缺值以 0 補）'}
    # 有效試驗數
    out['eff'] = {'LiJi_W_long': S.li_ji(X.values), 'LiJi_SEL': S.li_ji(X[sel].values)}
    k = 161
    lab, med, C = S.cluster_medoids(X, k)
    cols = list(X.columns)
    lab_s = pd.Series(lab, index=cols)
    sizes = lab_s.value_counts()
    main_c = int(lab_s[S.MAIN_ID])
    chain = [c for c in cols if 'o_v4_revenue_patch_official' in c or 'main_v2' in c or 'o_infra_v3_rebuild' in c]
    out['clusters'] = {'k': k, 'size_quantiles': q(sizes.values), 'n_singletons': int((sizes == 1).sum()), 'largest': int(sizes.max()),
                       'main_cluster': main_c, 'main_cluster_size': int(sizes[main_c]),
                       'main_cluster_medoid': med[sorted(set(lab)).index(main_c)],
                       'main_chain_clusters': lab_s[chain].value_counts().to_dict(),
                       'B10_cluster': int(lab_s[S.B10_ID]),
                       'main_cluster_folders': pd.Series([M.set_index('member_id').loc[c, 'folder'] for c in lab_s[lab_s == main_c].index]).value_counts().head(15).to_dict(),
                       'medoid_folders': pd.Series([M.set_index('member_id').loc[c, 'folder'] for c in med]).value_counts().head(25).to_dict()}
    iu = np.triu_indices(len(med), 1)
    Cm = np.corrcoef(X[med].values, rowvar=False)
    out['clusters']['medoid_mean_pair_corr'] = float(Cm[iu].mean())
    out['clusters']['all_mean_pair_corr'] = float(C[np.triu_indices(len(cols), 1)].mean())
    # 挑選期 α̂ 描述
    Xs = X[sel]
    a = Xs.mean().values
    se = np.sqrt(S.hac_var_mean(Xs.values, 6))
    am = Xs[med].mean().values
    dn = S.decompose(Xs[med], {'AB': (F['AB'] - F['r0050'])[sel].values * 100}, mode='naive')
    dd = S.decompose(Xs[med], {'AB': (F['AB'] - F['r0050'])[sel].values * 100}, mode='decomp')
    out['SEL'] = {'n_windows': int(sel.sum()),
                  'all_alpha_pp': q(a), 'all_se_pp': q(se), 'all_t': q(a / se), 'all_frac_alpha_pos': float((a > 0).mean()),
                  'medoid_alpha_pp': q(am), 'medoid_mean_alpha_pp': float(am.mean()), 'medoid_sd_alpha_pp': float(am.std(ddof=1)),
                  'medoid_frac_pos': float((am > 0).mean()),
                  'medoid_naive_se_pp': q(np.sqrt(dn['s2'])), 'medoid_idio_se_pp': q(np.sqrt(dd['s2'])),
                  'common_sd_pp': float(np.sqrt(dd['vP'])), 'Keff_equiv': dd['Keff'],
                  'naive_tau2_moment': float(am.var(ddof=1) - dn['s2'].mean()),
                  'decomp_tau2_moment': float(am.var(ddof=1) - dd['s2'].mean()),
                  'focus_AB_SEL_alpha_pp': dn['focus_xhat']['AB'], 'focus_AB_SEL_se_naive': float(np.sqrt(dn['focus_s2']['AB'])),
                  'focus_AB_SEL_se_idio': float(np.sqrt(dd['focus_s2']['AB'])),
                  'range_all_alpha_pp': [float(a.min()), float(a.max())]}
    # 挑選期內的贏家詛咒（前半選、後半看）
    h1 = S.wmask(idx, '2006-04-01', '2012-10-31'); h2 = S.wmask(idx, '2012-11-01', '2019-04-30')
    a1 = X[h1][med].mean(); a2 = X[h2][med].mean()
    b = np.polyfit(a1.values, a2.values, 1)
    top = a1.sort_values(ascending=False).index[:16]
    a1a = X[h1].mean(); a2a = X[h2].mean()
    topa = a1a.sort_values(ascending=False).index[:100]
    out['winner_curse_in_SEL'] = {'medoid_slope_h2_on_h1': float(b[0]), 'medoid_corr': float(np.corrcoef(a1, a2)[0, 1]),
                                  'top10pct_medoids_h1_mean': float(a1[top].mean()), 'top10pct_medoids_h2_mean': float(a2[top].mean()),
                                  'all_top100_h1_mean': float(a1a[topa].mean()), 'all_top100_h2_mean': float(a2a[topa].mean()),
                                  'all_slope': float(np.polyfit(a1a.values, a2a.values, 1)[0]),
                                  'AB_h1': float(((F['AB'] - F['r0050'])[h1] * 100).mean()), 'AB_h2': float(((F['AB'] - F['r0050'])[h2] * 100).mean())}
    # 理論家族
    th_mask = M.F_main | M.folder.isin(THEORY_FOLDERS)
    tcols = M.member_id[th_mask].tolist()
    out['theory_family'] = {'n_members': int(len(tcols)), 'LiJi_W_long': S.li_ji(X[tcols].values),
                            'folders': M[th_mask].folder.value_counts().to_dict(), 'SEL_alpha_pp': q(Xs[tcols].mean().values)}
    # 焦點序列（只報挑選期）
    out['focus_SEL'] = {c: {'mean_pp': float(((F[c] - F['r0050'])[sel] * 100).mean()), 't': S.tstat((F[c] - F['r0050'])[sel])}
                        for c in ['AB', 'B10', 'EXEC', 'EXEC_J', 'AB_stress', 'B10_stress', 'EW']}
    out['focus_SEL']['note'] = 'EXEC_J 的 δ_j 由全期 j-scan 計算（−0.347pp），挑選期均值只是描述'
    # 主策略殘差的描述（配置模擬用）
    e = (F['AB'] - F['r0050'])[sel] * 100
    out['focus_resid_SEL'] = {'sd_pp': float(e.std()), 'acf1': float(e.autocorr(1)), 'corr_with_r0050': float(np.corrcoef(e, F['r0050'][sel])[0, 1])}
    (HERE / 'explore.json').write_text(json.dumps(out, ensure_ascii=False, indent=1, default=S._js), encoding='utf-8')
    pd.DataFrame({'member_id': cols, 'cluster': lab, 'is_medoid': [c in set(med) for c in cols]}).to_csv(S.DATA / 'clusters_k161.csv', index=False)
    print(json.dumps(out, ensure_ascii=False, indent=1, default=S._js)[:9000])

if __name__ == '__main__':
    main()
