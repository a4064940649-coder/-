#!/usr/bin/env python3
"""前瞻先驗：選定版本 V* 的階層事後（族群在估計窗上擬合；焦點＝決策線序列對到族群窗、群＝114）→ 校準變換（均值×s、離散×k）。
變換：θ' ＝ s·μ ＋ k·(θ − μ)，μ＝事後均值；再以線性內插放回同一格點並正規化。"""
import sys, math, zlib
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
import fslib as F

T, H = F.T, F.H


def transform_density(d, th, s, k):
    d = np.asarray(d, float); d = d / d.sum()
    mu = float((th * d).sum())
    thp = s * mu + k * (th - mu)
    # 變換後的密度：p'(θ') ＝ p(θ)/k；在原格點上內插
    step = th[1] - th[0]
    dens = np.interp(th, thp, d / k, left=0.0, right=0.0)
    if dens.sum() <= 0:
        dens = np.zeros_like(th); dens[np.argmin(np.abs(th - s * mu))] = 1.0
    return dens / dens.sum()


def build(est_mask, version, s, k, focus, focus_cluster=F.MAIN_CLUSTER, seed=0):
    """est_mask：族群窗布林（估計期）；focus：決策線序列（pd.Series，index＝錨定月）。"""
    X, lab, med, anc = F.family()
    Xv = X.values[est_mask]
    fam = H.Fam(Xv, lab, med, need_g3=(version == 'V6_HNG3'))
    x, nmiss = F.align_to_family(focus, anc[est_mask])
    d = fam.posterior(version, x, focus_cluster, is_medoid=False, seed=seed)
    raw = T.dens_summary(d, fam.th)
    pr = transform_density(d, fam.th, s, k)
    cal = T.dens_summary(pr, fam.th)
    return dict(th=fam.th, raw_dens=d, prior=pr, raw=raw, cal=cal, n_missing_windows=nmiss, ahat=float(np.mean(x)),
                sigma_w=fam.sigma_w, n_est=int(est_mask.sum()))
