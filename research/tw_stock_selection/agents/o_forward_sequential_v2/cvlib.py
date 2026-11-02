#!/usr/bin/env python3
"""真實資料交叉驗證（CV）：把 o_tier s2c_cv.one 一般化成任意「估計窗集合 E／驗證窗集合 V」。
對每個 m_c ≥ 5 的群 c：以拿掉 c 的族群在 E 上估 g_B、σ_w（hmethods.Fam）；在 E 上挑群內 α̂ 最大的成員 k*（tag=sel），
另預測該群 medoid（tag=medoid）；7 個版本各給 α_k 的格點事後密度 d；
驗證：y＝k 在 V 的平均窗超額，s_V²＝k 在 V 的 HAC L6 變異；預測分布＝d ⊛ N(0, s_V²)。
記錄 logpd、PIT、事後均值、事後 sd，並保存 d（之後做校準變換：均值×s、離散×k）。
"""
import sys, math, time, zlib
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
from scipy import stats
import fslib as F

T, H = F.T, F.H
_G = {}


def _init():
    X, lab, med = H.load_real()
    _G.update(X=X.values, lab=np.asarray(lab, int), labS=lab, med=med)


def one(args):
    c, split, est, val = args
    X, labv, med = _G['X'], _G['lab'], _G['med']
    XA, XB = X[est], X[val]
    fam = H.Fam(XA, _G['labS'], med, need_g3=True, exclude_clusters=(c,))
    ii = np.where(labv == c)[0]
    aA = XA.mean(axis=0)
    ks = int(ii[np.argmax(aA[ii])])
    rows, dens = [], []
    for tag, k in (('sel', ks), ('medoid', med[c])):
        yB = float(XB[:, k].mean()); sB2 = float(T.hac_var_mean(XB[:, k], 6))
        for ver in H.VERSIONS:
            d = fam.posterior(ver, XA[:, k], c, is_medoid=(k == med[c]), seed=int(c) * 7 + (zlib.crc32(split.encode()) % 97))
            s = T.dens_summary(d, fam.th)
            rows.append(dict(cluster=int(c), m=int(len(ii)), split=split, tag=tag, ver=ver, k=k, ahat_est=float(aA[k]), y_val=yB,
                             sV=math.sqrt(sB2), post_mean=s['mean'], post_sd=s['sd'], n_est=len(est), n_val=len(val), sigma_w=fam.sigma_w))
            dens.append(d.astype(np.float32))
    return rows, dens


def splits_half(n_idx):
    """時間切半（兩個方向）。n_idx：可用窗的位置陣列。"""
    h = len(n_idx) // 2
    A, B = n_idx[:h], n_idx[h:]
    return [('half_AB', A, B), ('half_BA', B, A)]


def splits_roll(n_idx, origins, horizon):
    out = []
    for T0 in origins:
        if T0 + 12 > len(n_idx): continue
        out.append((f'roll_{T0}', n_idx[:T0], n_idx[T0:T0 + horizon]))
    return out


def run_cv(splits, workers=4):
    X, lab, med = H.load_real()
    labv = np.asarray(lab, int)
    cl = [c for c in sorted(med) if (labv == c).sum() >= 5]
    jobs = [(c, nm, np.asarray(E), np.asarray(V)) for (nm, E, V) in splits for c in cl]
    t0 = time.time(); rows, dens = [], []
    with ProcessPoolExecutor(max_workers=workers, initializer=_init) as ex:
        for rr, dd in ex.map(one, jobs):
            rows += rr; dens += dd
    D = pd.DataFrame(rows)
    return D, np.stack(dens), time.time() - t0


def transform_eval(D, dens, th, s=1.0, k=1.0):
    """校準變換：θ' ＝ s·μ ＋ k·(θ − μ)（μ＝事後均值），再 ⊛ N(0, s_V²)。回傳每列 logpd、PIT。"""
    mu = D.post_mean.to_numpy()[:, None]
    thp = s * mu + k * (th[None, :] - mu)
    p = dens / dens.sum(axis=1, keepdims=True)
    y = D.y_val.to_numpy()[:, None]; sv = D.sV.to_numpy()[:, None]
    z = (y - thp) / sv
    pit = (p * stats.norm.cdf(z)).sum(axis=1)
    pdv = (p * stats.norm.pdf(z) / sv).sum(axis=1)
    return np.log(np.maximum(pdv, 1e-300)), pit


def summarize(D, dens, th, tag='sel'):
    m = (D.tag == tag).to_numpy()
    Dt = D[m].copy(); dt = dens[m]
    lp, pit = transform_eval(Dt, dt, th)
    Dt['logpd'] = lp; Dt['pit'] = pit; Dt['cov80'] = (pit >= 0.1) & (pit <= 0.9)
    S = []
    for ver, g in Dt.groupby('ver'):
        b = np.polyfit(g.post_mean, g.y_val, 1)[0] if g.post_mean.std() > 0 else np.nan
        S.append(dict(ver=ver, n=len(g), sum_logpd=float(g.logpd.sum()), cov80=float(g.cov80.mean()), slope=float(b),
                      pit_mean=float(g.pit.mean()), post_mean_avg=float(g.post_mean.mean()), y_avg=float(g.y_val.mean())))
    S = pd.DataFrame(S).set_index('ver')
    by = Dt.groupby(['split', 'ver']).logpd.sum().unstack(0)
    sl = {}
    for (sp, ver), g in Dt.groupby(['split', 'ver']):
        sl[(sp, ver)] = float(np.polyfit(g.post_mean, g.y_val, 1)[0]) if g.post_mean.std() > 0 else np.nan
    return S, by, pd.Series(sl).unstack(0), Dt


def select_version(S, conservative_mean, tie=1.0):
    """事前規則：sum_logpd 最大者；與最佳差 < tie（nats）視為同分，同分時取主策略事後均值較低（較保守）者。"""
    best = S.sum_logpd.max()
    cand = [v for v in S.index if best - S.loc[v, 'sum_logpd'] < tie]
    return min(cand, key=lambda v: conservative_mean.get(v, np.inf)), cand


def calib_s(Dt, ver, schemes):
    """校準斜率 s：各 CV 方案（half／roll）分別合併斜率，取最小（不利端），截在 [0,1]。"""
    g = Dt[Dt.ver == ver]
    vals = {}
    for nm, pref in schemes.items():
        h = g[g.split.str.startswith(pref)]
        if len(h) >= 10 and h.post_mean.std() > 0:
            vals[nm] = float(np.polyfit(h.post_mean, h.y_val, 1)[0])
    s = float(np.clip(min(vals.values()), 0.0, 1.0)) if vals else 1.0
    return s, vals


def calib_k(Dt, dens_t, th, s, grid=None, target=0.80):
    """離散放大 k：k 從 1.0 起以 0.1 遞增，取第一個讓合併 80% 涵蓋率 ≥ target 的值（只放大不縮小）；回傳 k 與涵蓋率曲線。"""
    grid = np.round(np.arange(1.0, 6.01, 0.1), 2) if grid is None else grid
    curve = {}
    for k in grid:
        _, pit = transform_eval(Dt, dens_t, th, s, k)
        curve[float(k)] = float(np.mean((pit >= 0.1) & (pit <= 0.9)))
        if curve[float(k)] >= target: return float(k), curve
    return float(grid[-1]), curve
