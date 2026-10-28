#!/usr/bin/env python3
"""步驟 1c：真實資料的留一群、時間切半交叉驗證（PREREG_DESIGN step1_validation.cv_predictive_loglik）。
A＝前 122 窗、B＝後 123 窗。對每個 m_c ≥ 5 的群 c、兩個方向（A→B、B→A）：
  以拿掉 c 的族群在估計期估 g_B、σ_w；在估計期挑群內 α̂ 最大的成員 k*；各版本求 α_k* 事後；
  預測分布＝事後 ⊛ N(0, s²)（s²＝k* 在驗證期的 HAC L6 變異）；記驗證期實現 α̂ 的對數預測密度、80% 涵蓋、PIT。
另以同法預測各群 medoid（一般校準）。
輸出：data/cv_runs.parquet、results.json 的 STEP1_CV。
用法：python3 -I -B s2c_cv.py
"""
import sys, json, math, time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
import tlib as T, hmethods as H

_G = {}
def _init():
    X, lab, med = H.load_real()
    _G.update(X=X.values, lab=lab, med=med)

def one(args):
    c, direction = args
    X, lab, med = _G['X'], _G['lab'], _G['med']
    labv = np.asarray(lab, int)
    n = X.shape[0]; h = 122
    est, val = (slice(0, h), slice(h, n)) if direction == 'AB' else (slice(h, n), slice(0, h))
    XA, XB = X[est], X[val]
    fam = H.Fam(XA, lab, med, need_g3=True, exclude_clusters=(c,))
    ii = np.where(labv == c)[0]
    aA = XA.mean(axis=0)
    ks = int(ii[np.argmax(aA[ii])])
    rows = []
    for tag, k in (('sel', ks), ('medoid', med[c])):
        yB = float(XB[:, k].mean()); sB2 = float(T.hac_var_mean(XB[:, k], 6))
        for ver in H.VERSIONS:
            d = fam.posterior(ver, XA[:, k], c, is_medoid=(k == med[c]), seed=int(c) * 7 + (direction == 'BA'))
            s = T.dens_summary(d, fam.th)
            pred = T.convolve_dens(d, math.sqrt(sB2), fam.G.step)
            dens_at = float(np.interp(yB, fam.th, pred)) / fam.G.step
            cdf = np.cumsum(pred); pit = float(np.interp(yB, fam.th, cdf))
            rows.append(dict(cluster=int(c), m=int(len(ii)), direction=direction, tag=tag, ver=ver, ahat_est=float(aA[k]), y_val=yB,
                             post_mean=s['mean'], post_sd=s['sd'], logpd=math.log(max(dens_at, 1e-300)), pit=pit,
                             cov80=bool(0.1 <= pit <= 0.9), sigma_w=fam.sigma_w))
    return rows

def main():
    T.check_block('PREREG_DESIGN', files=False); T.check_block('PREREG_DESIGN_AMEND1', files=False)
    X, lab, med = H.load_real()
    labv = np.asarray(lab, int)
    cl = [c for c in sorted(med) if (labv == c).sum() >= 5]
    jobs = [(c, d) for c in cl for d in ('AB', 'BA')]
    t0 = time.time(); rows = []
    with ProcessPoolExecutor(max_workers=4, initializer=_init) as ex:
        for rr in ex.map(one, jobs):
            rows += rr
    D = pd.DataFrame(rows)
    D.to_parquet(T.DATA / 'cv_runs.parquet')
    S = D.groupby(['tag', 'ver']).agg(sum_logpd=('logpd', 'sum'), mean_logpd=('logpd', 'mean'), cov80=('cov80', 'mean'),
                                      pit_mean=('pit', 'mean'), err=('post_mean', 'mean'), y=('y_val', 'mean'), n=('logpd', 'size')).reset_index()
    sl = {}
    for (tag, ver), g in D.groupby(['tag', 'ver']):
        sl[f'{tag}|{ver}'] = float(np.polyfit(g.post_mean, g.y_val, 1)[0])
    S['slope_y_on_post'] = [sl[f'{t}|{v}'] for t, v in zip(S.tag, S.ver)]
    byd = D[D.tag == 'sel'].groupby(['direction', 'ver']).logpd.sum().unstack(0)
    out = {'n_clusters': len(cl), 'summary': S.to_dict('records'), 'by_direction_sel': byd.to_dict(), 'runtime_s': time.time() - t0, 'utc': T.now_utc()}
    T.save_key('STEP1_CV', out, need=['PREREG_DESIGN'])
    pd.set_option('display.width', 200)
    print(S.round(3).to_string()); print(byd.round(2))

if __name__ == '__main__':
    main()
