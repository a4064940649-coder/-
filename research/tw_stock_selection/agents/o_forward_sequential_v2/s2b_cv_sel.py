#!/usr/bin/env python3
"""步驟 2b：挑選期 CV（只用 SEL 窗 0～156）。時間切半兩方向＋滾動起點 {60,84,108}×48 窗。
依 PREREG_DESIGN 的規則選版、算 s、k。輸出 data/cv_sel.parquet、data/cv_sel_dens.npy、results.json 的 CV_SEL。
用法：python3 -I -B s2b_cv_sel.py"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
import fslib as F, cvlib as CV


def main_means(est_mask):
    """各版本下主策略 B10（族群成員）的估計期事後均值（同分規則用）。"""
    X, lab, med, anc = F.family()
    fam = F.H.Fam(X.values[est_mask], lab, med, need_g3=True)
    j = list(X.columns).index(F.T.B10_ID)
    out = {}
    for v in F.H.VERSIONS:
        d = fam.posterior(v, X.values[est_mask][:, j], F.MAIN_CLUSTER, is_medoid=(j == med[F.MAIN_CLUSTER]), seed=11)
        out[v] = F.T.dens_summary(d, fam.th)['mean']
    return out, fam.th


def run(n_win, half=True, origins=(), horizon=48, tag='SEL'):
    idx = np.arange(n_win)
    sp = (CV.splits_half(idx) if half else []) + CV.splits_roll(idx, origins, horizon)
    D, dens, rt = CV.run_cv(sp)
    return D, dens, rt, [(n, int(len(a)), int(len(b))) for n, a, b in sp]


def evaluate(D, dens, est_mask, key):
    mm, th = main_means(est_mask)
    S, by, sl, Dt = CV.summarize(D, dens, th, 'sel')
    Sm, _, _, _ = CV.summarize(D, dens, th, 'medoid')
    ver, cand = CV.select_version(S, mm)
    s, svals = CV.calib_s(Dt, ver, {'half': 'half', 'roll': 'roll'})
    m = ((D.tag == 'sel') & (D.ver == ver)).to_numpy()
    k, curve = CV.calib_k(D[m].reset_index(drop=True), dens[m], th, s)
    _, pit1 = CV.transform_eval(D[m].reset_index(drop=True), dens[m], th, s, 1.0)
    _, pitk = CV.transform_eval(D[m].reset_index(drop=True), dens[m], th, s, k)
    return dict(summary_sel=S.round(4).reset_index().to_dict('records'), summary_medoid=Sm.round(4).reset_index().to_dict('records'),
                logpd_by_split=by.round(3).to_dict(), slope_by_split=sl.round(3).to_dict(),
                main_B10_post_mean=mm, selected=ver, tie_candidates=cand, s=s, s_by_scheme=svals, k=k, k_curve=curve,
                cov80_at_s_k1=float(np.mean((pit1 >= .1) & (pit1 <= .9))), cov80_at_s_k=float(np.mean((pitk >= .1) & (pitk <= .9))))


def main():
    F.check_prereg('PREREG_DESIGN', files=False)
    X, lab, med, anc = F.family()
    sel = F.per_mask(anc, None, F.SEL_END)
    n = int(sel.sum()); assert n == 157 and sel[:n].all()
    D, dens, rt, sp = run(n, True, (60, 84, 108))
    D.to_parquet(F.DATA / 'cv_sel.parquet'); np.save(F.DATA / 'cv_sel_dens.npy', dens)
    out = evaluate(D, dens, sel, 'CV_SEL')
    out.update(splits=sp, runtime_s=rt, n_runs=len(D), utc=F.now_utc())
    F.save_key('CV_SEL', out, need=['PREREG_DESIGN'])
    print(pd.DataFrame(out['summary_sel']).to_string())
    print({k: out[k] for k in ('selected', 'tie_candidates', 's', 's_by_scheme', 'k', 'cov80_at_s_k1', 'cov80_at_s_k', 'main_B10_post_mean')})


if __name__ == '__main__':
    main()
