#!/usr/bin/env python3
"""反方審查（資料與統計角度）：o_tier_shrink_sequential。不修改研究員任何檔案，只讀取並輸出到 stdout。
R1 研究員自己的時間切半 CV：事後均值的校準斜率（實現 vs 事後），叢集 bootstrap 信賴區間；把校準套到 B10 與可執行格。
R2 以校準後（或 V3）的平移重跑配置模擬（研究員的 Alloc，5,000 路徑），看 x* 是否還存在。
R3 B10 超額的窗集中度：前 10 窗日期／年份、逐年留一。
R4 第 114 群：medoid、群平均、群中位數與主策略／B10 的位置（W_long）；medoid 是否為群內高分位。
R5 序貫規則：先驗寬度對偵測力的影響（iid 常態雜訊，σ 照登記）。
用法：python3 -I -B review_stat_o7.py
"""
import sys, json, math
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import numpy as np, pandas as pd
import tlib as T, hmethods as H, seqlib as SQ
import s4_formal as S4

def p(*a): print(*a, flush=True)

def main():
    res = json.loads((HERE / 'results.json').read_text(encoding='utf-8'))
    R = res['RESULTS']
    # ---------------- R1
    D = pd.read_parquet(HERE / 'data' / 'cv_runs.parquet')
    p('== R1 CV 校準（y_val 對 post_mean 的斜率；叢集 bootstrap 2000 次）')
    rng = np.random.default_rng(7)
    cal = {}
    for tag in ('sel', 'medoid'):
        for ver in ('V1_o6', 'V3_G2MAX', 'V5_HT'):
            g = D[(D.tag == tag) & (D.ver == ver)]
            b, a = np.polyfit(g.post_mean, g.y_val, 1)
            cls = g.cluster.unique(); bs = []
            for _ in range(2000):
                pick = rng.choice(cls, len(cls))
                gg = pd.concat([g[g.cluster == c] for c in pick])
                bs.append(np.polyfit(gg.post_mean, gg.y_val, 1)[0])
            lo, hi = np.percentile(bs, [5, 95])
            pred = a + b * R['cells_posterior']['B10']['mean']
            cal[(tag, ver)] = (a, b)
            p(f'  {tag:6s} {ver:9s} slope {b:.3f} (90% {lo:.2f}~{hi:.2f}) intercept {a:+.3f}  '
              f'B10 事後 {R["cells_posterior"]["B10"]["mean"]:.3f} → 校準後 {pred:+.3f}   cov80 {g.cov80.mean():.2f}  PIT {g.pit.mean():.2f}')
            for dirn in ('AB', 'BA'):
                gd = g[g.direction == dirn]
                p(f'      方向 {dirn}: slope {np.polyfit(gd.post_mean, gd.y_val, 1)[0]:.3f}, mean post {gd.post_mean.mean():+.3f}, mean y {gd.y_val.mean():+.3f}')
    # 高事後成員（post ≥ 0.4）在 CV 的實現
    g = D[(D.tag == 'sel') & (D.ver == 'V5_HT')]
    hi_ = g[g.post_mean >= 0.4]
    p(f'  V5 sel：事後均值 ≥ 0.4 的 {len(hi_)} 次，事後平均 {hi_.post_mean.mean():.3f}，實現平均 {hi_.y_val.mean():.3f}，'
      f'實現 < 0 的比例 {(hi_.y_val < 0).mean():.2f}')
    g1 = D[(D.tag == 'medoid') & (D.ver == 'V5_HT')]
    hi1 = g1[g1.post_mean >= 0.4]
    p(f'  V5 medoid：事後 ≥ 0.4 的 {len(hi1)} 次，事後平均 {hi1.post_mean.mean():.3f}，實現平均 {hi1.y_val.mean():.3f}')
    # ---------------- R2
    p('\n== R2 平移後重跑配置（5,000 路徑，研究員 Alloc，一般成本格）')
    z = np.load(HERE / 'data' / 'post_cells.npz')
    th = z['theta']; step = float(np.round(th[1] - th[0], 10))
    CS = pd.read_parquet(HERE / 'data' / 'cell_series.parquet')
    r50 = T.win_ret(T.e50_nav()).values
    A = S4.Alloc(r50, 5000)
    b10 = R['cells_posterior']['B10']['mean']
    shifts = {'V3（題目指定扣除）': R['focus_all_versions']['V3_G2MAX']['B10']['mean'] - b10}
    for key in (('medoid', 'V5_HT'), ('sel', 'V5_HT')):
        a, b = cal[key]; shifts[f'CV 校準 {key[0]}'] = a + b * b10 - b10
    cellnames = [c for c in CS.columns if c.startswith('EXEC') and '|normal|' in c] + ['B10_J']
    rows = []
    for nm in cellnames:
        d0 = z[nm.replace('|', '__')]
        m0 = float((d0 * th).sum())
        row = {'cell': nm, 'mean_base': round(m0, 3), 'xstar_base_paper': R['allocation'][nm]['20y']['xstar']}
        for lab_, sh in shifts.items():
            d = T.shift_dens(d0, th, sh)
            out = A.run(d, th, step, CS[nm].values)
            row[f'mean[{lab_}]'] = round(m0 + sh, 3)
            row[f'x*20y[{lab_}]'] = out['20y']['xstar']
            row[f'Pwin10@100[{lab_}]'] = round(out['10y']['grid'][100]['P_win'], 3)
        rows.append(row)
        p('  ', json.dumps(row, ensure_ascii=False))
    p('  平移量：', {k: round(v, 3) for k, v in shifts.items()})
    # ---------------- R3
    p('\n== R3 B10 超額集中度')
    F = pd.read_parquet(T.RAW / 'o6shrink' / 'focus_windows.parquet')
    ex = (F['B10'] - F['r0050']) * 100
    top = ex.sort_values(ascending=False).head(10)
    p('  前 10 窗：', [(str(i.date()), round(v, 1)) for i, v in top.items()])
    p(f'  全期平均 {ex.mean():.3f}，去前 10 窗 {ex.drop(top.index).mean():.3f}，去前 5 窗 {ex.drop(top.index[:5]).mean():.3f}，'
      f'中位數 {ex.median():.3f}，正窗比例 {(ex > 0).mean():.3f}')
    yr = ex.groupby(ex.index.year).agg(['sum', 'count'])
    loo = {int(y): round(ex[ex.index.year != y].mean(), 3) for y in yr.index}
    p('  逐年加總（pp）：', {int(y): round(v, 1) for y, v in yr['sum'].items()})
    p('  逐年留一平均：', loo)
    tot = ex.sum(); srt = yr['sum'].sort_values(ascending=False)
    p(f'  前 3 年貢獻占全期加總 {srt.head(3).sum() / tot:.2f}（{list(srt.head(3).index)}）')
    for per, (a_, b_) in {'SEL': T.SEL, 'VAL': T.VAL, 'HOLD': T.HOLD, '2017+': ('2017-01-01', '2026-10-12')}.items():
        e = ex[T.wmask(ex.index, a_, b_)]
        p(f'  {per}: n {len(e)} 平均 {e.mean():.3f} t_HAC {e.mean() / math.sqrt(T.hac_var_mean(e.values, 6)):.2f}')
    # 執行格 100萬 FIX KAPPA 的窗序列
    for nm in ('EXEC|100萬|FIX|normal|KAPPA', 'EXEC|100萬|FIX|normal|TICK'):
        s = CS[nm]
        if s.abs().max() > 0:
            pass
    # ---------------- R4
    p('\n== R4 第 114 群位置（W_long）')
    X, lab, med = H.load_real()
    a = X.mean()
    mem = lab[lab == 114].index
    medid = X.columns[med[114]]
    am = a[mem]
    p(f'  m {len(mem)}；群平均 {am.mean():.3f}、中位數 {am.median():.3f}、q90 {am.quantile(.9):.3f}、最大 {am.max():.3f}')
    p(f'  medoid α̂ {a[medid]:.3f}（群內百分位 {(am < a[medid]).mean():.2f}）；AB {a[T.MAIN_ID]:.3f}（{(am < a[T.MAIN_ID]).mean():.2f}）；'
      f'B10 {a[T.B10_ID]:.3f}（{(am < a[T.B10_ID]).mean():.2f}）')
    p(f'  medoid 與 AB 的相關 {np.corrcoef(X[medid], X[T.MAIN_ID])[0, 1]:.3f}；medoid 名稱 {medid}')
    allr = a.rank(ascending=False)
    p(f'  medoid 全族群名次 {allr[medid]:.0f} / {len(a)}；群間 161 個 medoid 中名次 '
      f'{a[[X.columns[i] for i in med.values()]].rank(ascending=False)[medid]:.0f}')
    # 若群水準改用群平均（而非 medoid）：B10 相對群平均的偏離
    p(f'  B10 − 群平均 {a[T.B10_ID] - am.mean():+.3f}；B10 − medoid {a[T.B10_ID] - a[medid]:+.3f}；'
      f'medoid − 群平均 {a[medid] - am.mean():+.3f}')
    # ---------------- R5
    p('\n== R5 序貫規則偵測力 vs 先驗寬度（L1 100萬 FIX，iid 常態雜訊 σ 照登記，4,000 路徑）')
    fp = np.load(HERE / 'data' / 'forward_priors.npz')
    thf = fp['theta']; prior = fp['L1_MAIN_B10_EXEC_AT_100萬__FIX']
    pf = json.loads((HERE / 'prereg_forward.json').read_text(encoding='utf-8'))['registration']
    thr = dict(pf['thresholds'])
    meta = pf['lines_L1_variants']['L1_MAIN_B10_EXEC@100萬|FIX']
    sig, amin = meta['sigma_pp'], meta['alpha_min']
    m0 = float((prior * thf).sum() / prior.sum()); sd0 = math.sqrt(float((prior * (thf - m0) ** 2).sum() / prior.sum()))
    p(f'  登記先驗：mean {m0:.3f} sd {sd0:.3f}；先驗相當於 (σ/sd)² = {(sig / sd0) ** 2:.0f} 個月的資料')
    rr = np.random.default_rng(11)
    Z = rr.standard_normal((4000, 120))
    for infl in (1.0, 2.0, 3.0):
        pr = np.interp(m0 + (thf - m0) / infl, thf, prior, left=0, right=0)
        pr = pr / pr.sum()
        for alpha in (-1.0, -0.5, 0.0):
            xs = alpha + sig * Z
            st = SQ.path_stats(xs, pr, thf, sig, amin)
            tg = SQ.triggers(st, thr)
            ft = SQ.first_time(tg['down'])
            p(f'  先驗 sd×{infl:.0f}  α {alpha:+.1f}: 下修 ≤24 月 {((ft > 0) & (ft <= 24)).mean():.4f}，≤36 {((ft > 0) & (ft <= 36)).mean():.4f}，'
              f'≤60 {((ft > 0) & (ft <= 60)).mean():.4f}；24 月時 P(α>0) 中位數 {np.median(st["post_P"][:, 23]):.2f}')
    # α=−1 時 24 個月累積落後
    p(f'  α −1pp／月 持續 24 個月，主策略相對 0050 期望落後約 {100 * (1 - 0.99 ** 24):.1f}%；60 個月 {100 * (1 - 0.99 ** 60):.1f}%')

if __name__ == '__main__':
    main()
