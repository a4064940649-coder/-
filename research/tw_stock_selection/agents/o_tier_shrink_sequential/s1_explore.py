#!/usr/bin/env python3
"""步驟 1（探索，只用挑選期 SEL＝2006-04～2019-04 的描述統計；不算任何焦點事後分布）。
問題：群內挑選造成的偏誤看得到嗎？有多大？跟群大小 m_c、群內相關有什麼關係？
 E1 族群結構：第 114 群（主策略所在）大小、Li-Ji 有效數、主策略與 B10 在群內的 α̂ 名次（SEL）。
 E2 群內離散：群內偏離 d_k＝α̂_k − α̂_medoid 的變異 vs 雜訊變異（差序列 HAC），推 σ_w。
 E3 群內最大值偏誤（bootstrap）：各群以 SEL 殘差重抽，取群內最大，偏誤 vs m_c 與 Li-Ji 有效數。
 E4 群內贏家詛咒（SEL 前半挑、後半看）：每群在前半挑 α̂ 最大的成員，看後半「相對群 medoid 的偏離」退回多少。
 E5 主策略超額序列（SEL）：追蹤誤差、自相關、峰態（序貫規則模擬要用）。
 E6 可執行拖累與 j 效果（SEL 描述）。
輸出：explore.json（含 data/gaps.json）。
用法：python3 -I -B s1_explore.py
"""
import sys, json, math
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
from scipy import stats
import tlib as T

def main():
    X, M = T.load_family()
    cl = T.load_clusters().set_index('member_id')
    sel = T.wmask(X.index, *T.SEL)
    XS = X[sel]
    n = len(XS)
    a = XS.mean()
    lab = cl.loc[X.columns, 'cluster']
    med = cl[cl.is_medoid].reset_index().set_index('cluster')['member_id']
    out = {'period': T.SEL, 'n_windows_SEL': int(n)}
    # ---------------- E1
    c_main = int(lab[T.MAIN_ID]); c_b10 = int(lab[T.B10_ID])
    mem = lab[lab == c_main].index.tolist()
    rk = a[mem].rank(ascending=False)
    allrk = a.rank(ascending=False)
    top165 = allrk[allrk <= 165].index
    sizes = lab.value_counts()
    out['E1'] = {'main_cluster': c_main, 'b10_cluster': c_b10, 'm_main_cluster': len(mem),
                 'lij_main_cluster_SEL': T.li_ji(XS[mem].values),
                 'main_rank_in_cluster_SEL': float(rk[T.MAIN_ID]), 'b10_rank_in_cluster_SEL': float(rk.get(T.B10_ID, np.nan)),
                 'main_rank_overall_SEL': float(allrk[T.MAIN_ID]),
                 'share_top165_in_main_cluster_SEL': float(np.mean([lab[m] == c_main for m in top165])),
                 'main_alpha_SEL': float(a[T.MAIN_ID]), 'cluster_mean_alpha_SEL': float(a[mem].mean()),
                 'cluster_medoid': med[c_main], 'medoid_alpha_SEL': float(a[med[c_main]]),
                 'cluster_alpha_q': {q: float(a[mem].quantile(q)) for q in (0.1, 0.5, 0.9, 1.0)},
                 'cluster_folders_top': M.set_index('member_id').loc[mem, 'folder'].value_counts().head(12).to_dict(),
                 'cluster_size_dist': {'n_clusters': int(len(sizes)), 'singletons': int((sizes == 1).sum()), 'max': int(sizes.max()),
                                       'q50': float(sizes.median()), 'q90': float(sizes.quantile(0.9))}}
    # ---------------- E2 群內離散（多成員群）
    rows = []
    for c, g in lab.groupby(lab):
        ms = g.index.tolist()
        if len(ms) < 3: continue
        ref = XS[med[c]].values
        D = XS[ms].values - ref[:, None]
        dk = D.mean(axis=0)
        nv = T.hac_var_mean(D, 6)
        keep = np.array([m != med[c] for m in ms])
        rows.append({'cluster': int(c), 'm': len(ms), 'var_d': float(np.var(dk[keep], ddof=1)) if keep.sum() > 1 else np.nan,
                     'noise_var_d': float(np.mean(nv[keep])), 'sd_d': float(np.std(dk[keep], ddof=1)) if keep.sum() > 1 else np.nan})
    E2 = pd.DataFrame(rows)
    w = E2.m - 1
    pooled_var = float((E2.var_d * w).sum() / w.sum()); pooled_noise = float((E2.noise_var_d * w).sum() / w.sum())
    out['E2'] = {'n_clusters_m_ge3': int(len(E2)), 'pooled_var_d': pooled_var, 'pooled_noise_var_d': pooled_noise,
                 'sigma_w_mom_pp': float(math.sqrt(max(pooled_var - pooled_noise, 0) / 2)),
                 'note': 'Var(d)＝2σ_w²（成員與 medoid 各有偏離）＋雜訊；σ_w 以動差法估。',
                 'main_cluster': E2[E2.cluster == c_main].to_dict('records'),
                 'by_size': E2.assign(sz=pd.cut(E2.m, [2, 5, 10, 30, 100, 2000])).groupby('sz', observed=True)
                     .apply(lambda g: pd.Series({'n': len(g), 'sd_d_med': g.sd_d.median(), 'noise_sd_med': np.sqrt(g.noise_var_d).median()}))
                     .reset_index().astype({'sz': str}).to_dict('records')}
    # ---------------- E3 群內最大值偏誤
    rows = []
    for c, g in lab.groupby(lab):
        ms = g.index.tolist()
        if len(ms) < 2: continue
        bt = T.max_noise_boot(XS[ms].values, B=400, L=6, seed=int(c))
        btd = T.max_noise_boot_diff(XS[ms].values, XS[med[c]].values, B=400, L=6, seed=int(c) + 7)
        rows.append({'cluster': int(c), 'm': len(ms), 'lij': T.li_ji(XS[ms].values) if len(ms) > 1 else 1.0,
                     'bias_max_mean': float(bt.mean()), 'bias_max_sd': float(bt.std()),
                     'bias_maxdiff_mean': float(btd.mean()), 'bias_maxdiff_sd': float(btd.std()),
                     'se_member_med': float(np.median(np.sqrt(T.hac_var_mean(XS[ms].values, 6))))})
    E3 = pd.DataFrame(rows)
    out['E3'] = {'main_cluster': E3[E3.cluster == c_main].to_dict('records'),
                 'corr_bias_logm': float(np.corrcoef(np.log(E3.m), E3.bias_max_mean)[0, 1]),
                 'corr_bias_loglij': float(np.corrcoef(np.log(E3.lij), E3.bias_max_mean)[0, 1]),
                 'bias_over_se_med': float((E3.bias_max_mean / E3.se_member_med).median()),
                 'table_by_m': E3.assign(sz=pd.cut(E3.m, [1, 3, 10, 30, 100, 2000])).groupby('sz', observed=True)
                     [['bias_max_mean', 'bias_maxdiff_mean', 'lij', 'se_member_med']].median().reset_index().astype({'sz': str}).to_dict('records')}
    # ---------------- E4 群內贏家詛咒（SEL 前半挑、後半看）
    h = n // 2
    A1, A2 = XS.iloc[:h].mean(), XS.iloc[h:].mean()
    rows = []
    for c, g in lab.groupby(lab):
        ms = g.index.tolist()
        if len(ms) < 5: continue
        dev1 = A1[ms] - A1[med[c]]; dev2 = A2[ms] - A2[med[c]]
        best = dev1.idxmax()
        rows.append({'cluster': int(c), 'm': len(ms), 'dev1_best': float(dev1[best]), 'dev2_best': float(dev2[best]),
                     'best_is_medoid': best == med[c]})
    E4 = pd.DataFrame(rows)
    sl = float(np.polyfit(E4.dev1_best, E4.dev2_best, 1)[0]) if len(E4) > 2 else np.nan
    out['E4'] = {'n_clusters_m_ge5': int(len(E4)), 'mean_dev1_best': float(E4.dev1_best.mean()), 'mean_dev2_best': float(E4.dev2_best.mean()),
                 'retention': float(E4.dev2_best.mean() / E4.dev1_best.mean()), 'slope': sl,
                 'main_cluster': E4[E4.cluster == c_main].to_dict('records'),
                 'large_clusters_m_ge30': E4[E4.m >= 30].to_dict('records')}
    # 群間（medoid）同法，對照
    medcols = med.tolist()
    b1, b2 = A1[medcols], A2[medcols]
    top = b1.sort_values(ascending=False).index[:16]
    out['E4']['between_top10pct_medoid'] = {'first': float(b1[top].mean()), 'second': float(b2[top].mean())}
    # ---------------- E5 主策略超額（SEL）
    F = pd.read_parquet(T.RAW / 'o6shrink' / 'focus_windows.parquet')
    ex = ((F['B10'] - F['r0050']) * 100)[sel]
    acf = [float(pd.Series(ex.values).autocorr(k)) for k in (1, 2, 3, 6, 12)]
    out['E5'] = {'B10_excess_SEL': {'mean': float(ex.mean()), 'sd': float(ex.std(ddof=1)), 'acf_1_2_3_6_12': acf,
                                    'kurtosis_excess': float(stats.kurtosis(ex)), 'skew': float(stats.skew(ex)),
                                    'hac_sd_ratio_L6': float(math.sqrt(T.hac_var_mean(ex.values, 6) * len(ex)) / ex.std(ddof=1))}}
    # ---------------- E6 拖累與 j（SEL 描述）
    EX = pd.read_parquet(T.RAW / 'navs' / 'navs_abavg.parquet')
    EB = pd.read_parquet(T.RAW / 'exec' / 'bench.parquet')
    zi = T.win_ret(EB['AB_V4|normal'])
    dr = {}
    for tier in ('30萬', '100萬', '300萬', '1000萬'):
        for mode in ('FIX', 'GROW'):
            d = (T.win_ret(EX[f'main|{tier}|{mode}|A']) - zi) * 100
            imp = (T.win_ret(EX[f'd5_impact|{tier}|{mode}|A']) - T.win_ret(EX[f'd4_fees|{tier}|{mode}|A'])) * 100
            dr[f'{tier}|{mode}'] = {'drag_SEL': float(d[sel].mean()), 'impact_part_SEL': float(imp[sel].mean())}
    V6 = pd.read_parquet(T.RAW / 'navs' / 'navs_v6.parquet'); JS = pd.read_parquet(T.RAW / 'navs' / 'navs_jscan_v6.parquet')
    off = T.win_ret(V6['AB_V6_B10|normal'])
    dj = {}
    for j in range(13):
        dd = ((T.win_ret(JS[f'AB_V6_B10|normal|J{j}']) - off) * 100)[sel].dropna()
        dj[f'J{j}'] = float(dd.mean())
    out['E6'] = {'exec_drag_pp_per_window_SEL': dr, 'delta_j_B10_SEL': dj, 'delta_j_sd_SEL': float(np.std(list(dj.values()), ddof=1))}
    gaps = json.loads((T.DATA / 'gaps.json').read_text(encoding='utf-8'))
    out['gaps'] = gaps
    out['conclusion'] = {
        'mechanism_visible': None, 'note': '見 FINDINGS 探索節；本檔只放數字。'}
    (T.HERE / 'explore.json').write_text(json.dumps(out, ensure_ascii=False, indent=1, default=T._js), encoding='utf-8')
    pr = {k: v for k, v in out.items() if k not in ('gaps',)}
    print(json.dumps(pr, ensure_ascii=False, indent=1, default=T._js)[:9000])

if __name__ == '__main__':
    main()
