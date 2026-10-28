#!/usr/bin/env python3
"""正式計算 3：樣本外檢查（PREREG 之後，驗證期與保留期各跑一次）。
(a) 族群：以 SEL 的 α̂ 估 MAIN 先驗（G1／G2／G3），算 F_long 全部 2,319 條的事後平均，和 VAL、HOLD 各自實現的平均窗超額比較。
(b) 焦點：SEL 事後分布＋VAL／HOLD 抽樣誤差 → 預測分布，報實現值百分位。
用法：python3 -I -B run_oos.py [--focus ...] [--tag V4]"""
import sys, json, argparse, math
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import sc_lib as S

def member_s2(Xs, mser, L=6):
    return S.hac_var_mean(Xs - mser[:, None], L)

def run(focus_path, tag, check=True):
    if check: S.check_prereg()
    X, M = S.load_family()
    F = S.load_focus(focus_path)
    fam = json.loads((S.DATA / 'families.json').read_text(encoding='utf-8'))
    med = fam['MAIN']
    idx = X.index
    sel = S.wmask(idx, *S.SEL)
    per = {'VAL': S.wmask(idx, *S.VAL), 'HOLD': S.wmask(idx, *S.HOLD)}
    Xs = X[sel]
    dec = S.decompose(Xs[med], None, mode='decomp')
    mser = Xs[med].values.mean(axis=1)
    a_all = Xs.values.mean(axis=0)
    s2_all = member_s2(Xs.values, mser)
    out = {'tag': tag, 'n_members': int(X.shape[1]), 'SEL_windows': int(sel.sum()), 'models': {}}
    is_med = np.array([c in set(med) for c in X.columns])
    for model in ('G1', 'G2', 'G3'):
        pr = S.fit_prior(model, dec['xhat'], dec['s2'])
        pm, lo, hi = S.post_all(pr, model, a_all, s2_all, dec['vP'])
        mo = {}
        for pn, pmask in per.items():
            real = X[pmask].values.mean(axis=0)
            rs2 = S.hac_var_mean(X[pmask].values, 6)
            def stats_for(mask):
                r, a, p = real[mask], a_all[mask], pm[mask]
                top = np.argsort(-a)[:max(10, int(0.05 * mask.sum()))]
                return {'n': int(mask.sum()),
                        'MSE_raw': float(np.mean((r - a) ** 2)), 'MSE_post': float(np.mean((r - p) ** 2)),
                        'MSE_grandmean': float(np.mean((r - a.mean()) ** 2)),
                        'slope_real_on_raw': float(np.polyfit(a, r, 1)[0]), 'slope_real_on_post': float(np.polyfit(p, r, 1)[0]),
                        'corr_raw_real': float(np.corrcoef(a, r)[0, 1]),
                        'top5pct_raw_mean': float(a[top].mean()), 'top5pct_post_mean': float(p[top].mean()), 'top5pct_real_mean': float(r[top].mean()),
                        'cover80_post': float(np.mean((r >= lo[mask] - 1.2816 * np.sqrt(rs2[mask])) & (r <= hi[mask] + 1.2816 * np.sqrt(rs2[mask])))),
                        'mean_real': float(r.mean())}
            mo[pn] = {'all': stats_for(np.ones(len(a_all), bool)), 'medoids': stats_for(is_med)}
        out['models'][model] = mo
    # 焦點預測分布（G2×MAIN，SEL 先驗）
    pr = S.fit_prior('G2', dec['xhat'], dec['s2'])
    meta = json.loads((S.DATA / 'focus_meta.json').read_text(encoding='utf-8'))
    sdc = float(meta['cost_uncertainty']['sd_cost_pp'])
    foc = {}
    th = S.theta_grid()
    for c in ['AB', 'B10', 'EXEC', 'EXEC_J']:
        e = (F[c] - F['r0050']).values * 100
        es = e[sel]
        s2f = S.hac_var_mean(es - mser, 6) + (sdc ** 2 if c.startswith('EXEC') else 0)
        p = S.posterior(pr, 'G2', float(es.mean()), s2f, dec['vP'])
        d = p.pop('dens')
        fo = {'SEL_raw': float(es.mean()), 'SEL_post_mean': p['mean'], 'SEL_post_p_pos': p['p_pos']}
        for pn, pmask in per.items():
            ep = e[pmask]; v = S.hac_var_mean(ep, 6)
            # 預測分布：α ~ 事後、實現均值 ~ N(α, v)
            cdf = float((d * stats.norm.cdf((ep.mean() - th) / math.sqrt(v))).sum())
            cdf_raw = float(stats.norm.cdf((ep.mean() - es.mean()) / math.sqrt(v)))
            fo[pn] = {'realized_mean_pp': float(ep.mean()), 'n': int(pmask.sum()), 'pred_percentile_post': cdf, 'pred_percentile_noshrink': cdf_raw}
        foc[c] = fo
    out['focus'] = foc
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--focus', default=str(S.DATA / 'focus_windows.parquet'))
    ap.add_argument('--tag', default='V4')
    a = ap.parse_args()
    out = run(a.focus, a.tag)
    S.save_key('RESULTS_OOS' + ('' if a.tag == 'V4' else '_' + a.tag), out)
    print(json.dumps(out, ensure_ascii=False, indent=1)[:6000])

if __name__ == '__main__':
    main()
