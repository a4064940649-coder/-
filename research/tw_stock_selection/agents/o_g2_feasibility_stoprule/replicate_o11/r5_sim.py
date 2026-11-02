"""Gate feasibility simulation (independent rewrite from frozen spec).
usage: python3 -I -B r5_sim.py main|variants|forward|clean
"""
import sys, json, math, time
import numpy as np, pandas as pd
from scipy import stats
sys.path.insert(0, '/home/user/-/research/tw_stock_selection/agents/o_g2_feasibility_stoprule/replicate_o11')
from rlib import sb_indices, counts_from_idx, spa_boot_max, pr_kernel_lrv, sr0_of

OUT = '/tmp/claude-0/-home-user--/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/rep_o11'
Z = np.load(OUT + '/inputs.npz', allow_pickle=True)
F = np.load(OUT + '/family_stats.npz')
FAM = json.load(open(OUT + '/family.json'))
X = np.nan_to_num(Z['X']); r0050 = Z['r0050']; dates = pd.DatetimeIndex(Z['dates'])
n0 = len(r0050)
ab = Z['ab']; NB = Z['NB']; d_j = Z['d_j']
c_st = Z['c_stress']; c3 = Z['c_300']; c10 = Z['c_1000']
fmain = Z['fmain'].astype(bool)
IRS = [0.0, 0.2, 0.3, 0.4, 0.5, 0.66, 0.8, 1.0]
Y19 = np.asarray(dates >= '2019-01-01')
SEGS = [np.asarray(dates <= '2012-12-31'), np.asarray((dates >= '2013-01-01') & (dates <= '2018-12-31')), Y19]
YEARS = dates.year.values
NTR = X.shape[1] + 1  # 2,335
SRm_fam = X.mean(0) / X.std(0, ddof=1)
# SPA family bootstrap (B=10,000, L=6, seed 20261102)
Wc = counts_from_idx(sb_indices(n0, 6, 10000, 20261102), n0)
_, MX_FAM, _ = spa_boot_max(X, Wc)
fm_ids = np.where(fmain[:X.shape[1]])[0]
_, MX_FM, _ = spa_boot_max(X[:, fm_ids], Wc)
print('family 410 size', len(fm_ids))


def geo(rs, rb):
    """rs: (..., n) strategy returns, rb: (..., n) bench returns -> geometric annualized excess."""
    n = rs.shape[-1]
    return np.exp(np.log1p(rs).sum(-1) * 12 / n) - np.exp(np.log1p(rb).sum(-1) * 12 / n)


def tst(x):
    n = x.shape[-1]
    return x.mean(-1) / x.std(-1, ddof=1) * np.sqrt(n)


def lrv_rows(x, L=6):
    """PR kernel lrv per row of x (S x n)."""
    S, n = x.shape
    z = x - x.mean(1, keepdims=True)
    q = 1.0 / L
    v = (z * z).sum(1) / n
    for i in range(1, n):
        k = (1 - i / n) * (1 - q) ** i + (i / n) * (1 - q) ** (n - i)
        if k < 1e-12:
            continue
        v += 2 * k * (z[:, i:] * z[:, :-i]).sum(1) / n
    return v


def spa_p(x, W, MX, full=True):
    """focus p for candidates rows x (S x n) appended to family whose recentred max distribution is MX."""
    S, n = x.shape
    m = x.mean(1); om = np.sqrt(lrv_rows(x))
    t = math.sqrt(n) * m / om
    if not full:
        mx0 = np.maximum(MX, 0)
        srt = np.sort(mx0)
        return 1 - np.searchsorted(srt, t, side='left') / len(srt), t
    gc = np.where(t >= -math.sqrt(2 * math.log(math.log(n))), m, 0.0)
    bm = (W @ x.T) / n  # B x S
    zc = math.sqrt(n) * (bm - gc[None, :]) / om[None, :]
    big = np.maximum(np.maximum(MX[:, None], zc), 0)
    return (big >= t[None, :]).mean(0), t


def dsr_rows(x, var_sr, N):
    T = x.shape[1]
    sr = x.mean(1) / x.std(1, ddof=1)
    g3 = stats.skew(x, axis=1); g4 = stats.kurtosis(x, axis=1, fisher=False)
    sr0 = sr0_of(var_sr, N) if np.isscalar(var_sr) else np.array([sr0_of(v, N) for v in var_sr])
    z = (sr - sr0) * math.sqrt(T - 1) / np.sqrt(np.maximum(1 - g3 * sr + (g4 - 1) / 4 * sr ** 2, 1e-12))
    return stats.norm.cdf(z)


def drop3_t(xj, rs, rb, years):
    """xj, rs: S x J x n ; rb: S x n ; t after removing each path's 3 best calendar years (geometric yearly excess)."""
    S, J, n = xj.shape
    uy = np.unique(years)
    ly = np.log1p(rs); lb = np.log1p(rb)
    ye = np.stack([np.exp(ly[..., years == y].sum(-1)) - np.exp(lb[:, None, years == y].sum(-1)) for y in uy], -1)  # S x J x Y
    order = np.argsort(-ye, axis=-1)[..., :3]
    best = uy[order]  # S x J x 3
    drop = (years[None, None, :, None] == best[:, :, None, :]).any(-1)  # S x J x n
    xm = np.where(drop, np.nan, xj)
    cnt = (~drop).sum(-1)
    mu = np.nanmean(xm, -1); sd = np.nanstd(xm, -1, ddof=1)
    return mu / sd * np.sqrt(cnt)


def gates(x, r0, D, eta, cst, ct3, ct10, y19, segs, years, t2a=3.0, t2b=2.0, t7a=3.0, t7b=2.0):
    """x: S x n javg excess; r0: S x n; D: S x n x J deviations; eta: S x n x 6.
    cst/ct3/ct10: drags (n,) aligned. returns dict of bool arrays."""
    xj = x[:, None, :] + np.transpose(D, (0, 2, 1))  # S x J x n
    rs = r0[:, None, :] + xj
    g = {}
    ge = geo(rs, np.broadcast_to(r0[:, None, :], rs.shape))
    ge_st = geo(rs - cst, np.broadcast_to(r0[:, None, :], rs.shape))
    g['G1'] = (np.median(ge, 1) > 0) & (np.percentile(ge, 10, axis=1) > 0) & (np.median(ge_st, 1) > 0)
    tf = tst(xj); t19 = tst(xj[..., y19])
    g['t_med'] = np.median(tf, 1); g['t19_med'] = np.median(t19, 1)
    g['G2a'] = g['t_med'] >= t2a
    g['G2b'] = g['t19_med'] >= t2b
    ok = np.ones(len(x), bool)
    for s in segs:
        ok &= np.median(geo(rs[..., s], np.broadcast_to(r0[:, None, s], rs[..., s].shape)), 1) > 0
    g['G4a'] = ok
    g['G4b'] = np.median(drop3_t(xj, rs, r0, years), 1) >= 2.0
    nb = x[:, :, None] + eta  # S x n x 6
    g['G4c'] = (geo(np.transpose(r0[:, :, None] + nb, (0, 2, 1)), np.broadcast_to(r0[:, None, :], (len(x), eta.shape[2], x.shape[1]))) > 0).all(1)
    g7 = np.ones(len(x), bool)
    g7r = np.ones(len(x), bool)
    for c in (ct3, ct10):
        xc = xj - c
        a = np.median(tst(xc), 1); b = np.median(tst(xc[..., y19]), 1)
        g7 &= (a >= t7a) & (b >= t7b)
        g7r &= (a >= 2.0) & (b >= 2.0)
    g['G7'] = g7; g['G7_t2'] = g7r
    g7a = np.ones(len(x), bool); g7f2 = np.ones(len(x), bool); g7f3 = np.ones(len(x), bool)
    for c in (ct3, ct10):
        xc = xj - c
        a = np.median(tst(xc), 1)
        e19 = np.median(geo((rs - c)[..., y19], np.broadcast_to(r0[:, None, y19], rs[..., y19].shape)), 1)
        g7a &= (a >= 2.0) & (e19 > 0); g7f2 &= a >= 2.0; g7f3 &= a >= t7a
    g['G7_A'] = g7a; g['G7_full2'] = g7f2; g['G7_full3'] = g7f3
    # alternative A pieces
    g['A_t2'] = g['t_med'] >= 2.0
    g['A_y19ex'] = np.median(geo(rs[..., y19], np.broadcast_to(r0[:, None, y19], rs[..., y19].shape)), 1) > 0
    return g, xj


def draw(rng, S, n, L, eps_src, iid=False):
    if iid:
        return None, rng.standard_normal((S, n))
    idx = np.empty((S, n), dtype=np.int64)
    m = len(eps_src)
    idx[:, 0] = rng.integers(0, m, S)
    newb = rng.random((S, n)) < 1.0 / L
    rnd = rng.integers(0, m, (S, n))
    for t in range(1, n):
        idx[:, t] = np.where(newb[:, t], rnd[:, t], (idx[:, t - 1] + 1) % m)
    return idx, eps_src[idx]


VARIANTS = {'V1': {}, 'V2': {'L': 3}, 'V3': {'L': 12}, 'V4': {'sig19': True}, 'V5': {'iid': True}, 'V6': {'v7x': True}, 'V7': {'nb_mean': True}}


def setup(var):
    o = VARIANTS[var]
    L = o.get('L', 6)
    base = Z['v7x_javg'] if o.get('v7x') else ab
    eps = (base - base.mean()) / base.std(ddof=1)
    sig = base.std(ddof=1)
    if o.get('sig19'):
        sig = ab[Y19].std(ddof=1)
    eta = NB - ab[:, None]
    if not o.get('nb_mean'):
        eta = eta - eta.mean(0)
    return L, eps, sig, eta, o.get('iid', False)


def run_main(var='V1', nsim=4000, irs=IRS, batch=500, alt=True):
    vi = list(VARIANTS).index(var)
    L, eps, sig, eta_src, iid = setup(var)
    out = {}
    for ii, IR in enumerate(IRS):
        if IR not in irs:
            continue
        rng = np.random.default_rng(20261102 + 1000 * ii + 100 * vi)
        mu = IR / math.sqrt(12) * sig
        acc = {}
        for b0 in range(0, nsim, batch):
            S = min(batch, nsim - b0)
            idx, e = draw(rng, S, n0, L, eps, iid)
            if idx is None:  # iid: joint sampling of the rest via iid uniform positions
                idx = rng.integers(0, n0, (S, n0))
            x = mu + sig * e
            r0 = r0050[idx]; D = d_j[idx]; eta = eta_src[idx]
            g, xj = gates(x, r0, D, eta, c_st, c3, c10, Y19, SEGS, YEARS)
            p, tc = spa_p(x, Wc, MX_FAM, full=True)
            g['G3a'] = p < 0.05
            # DSR var_SR over family + candidate
            src = x.mean(1) / x.std(1, ddof=1)
            sall = np.concatenate([np.broadcast_to(SRm_fam, (S, len(SRm_fam))), src[:, None]], 1)
            vs = sall.var(1, ddof=1)
            g['G3b'] = dsr_rows(x, vs, NTR) >= 0.95
            if alt:
                pA, _ = spa_p(x, Wc, MX_FM, full=True)
                g['A_spa10'] = pA < 0.10
                g['G2a_25'] = g['t_med'] >= 2.5
                gB, _ = gates(x, r0, D, eta, c_st, c3, c10, Y19, SEGS, YEARS, t7a=2.5)
                g['G7_25'] = gB['G7']
                # C: forward 24m sequential test on javg (net) excess
                idf, ef = draw(rng, S, 24, L, eps, iid)
                xf = mu + sig * ef
                zk = np.cumsum(xf, 1) / (sig * np.sqrt(np.arange(1, 25)))
                g['C_seq'] = (zk >= 2.43).any(1)
                g['C_fix'] = zk[:, -1] >= 1.645
            for k, v in g.items():
                if v.dtype == bool:
                    acc.setdefault(k, []).append(v)
        R = {k: np.concatenate(v) for k, v in acc.items()}
        base_no = R['G1'] & R['G2a'] & R['G2b'] & R['G3a'] & R['G4a'] & R['G4b'] & R['G4c'] & R['G7']
        res = {k: float(R[k].mean()) for k in ['G1', 'G2a', 'G2b', 'G3a', 'G3b', 'G4a', 'G4b', 'G4c', 'G7']}
        res['joint'] = float((base_no & R['G3b']).mean()); res['joint_noDSR'] = float(base_no.mean())
        if alt:
            A = R['A_t2'] & R['A_y19ex'] & R['A_spa10']
            res['A'] = float(A.mean()); res['A_G1_G7'] = float((A & R['G1'] & R['G7']).mean())
            res['A_G1_G7t2'] = float((A & R['G1'] & R['G7_t2']).mean())
            res['A_G1_G7A'] = float((A & R['G1'] & R['G7_A']).mean())
            res['A_G1_G7full2'] = float((A & R['G1'] & R['G7_full2']).mean())
            Cn2 = R['G1'] & R['G2a'] & R['C_seq'] & R['G3a'] & R['G4a'] & R['G4b'] & R['G4c'] & R['G7_full3']
            res['C_noDSR_G7fullonly'] = float(Cn2.mean())
            Bn = R['G1'] & R['G2a_25'] & R['G2b'] & R['G3a'] & R['G4a'] & R['G4b'] & R['G4c'] & R['G7_25']
            res['B_noDSR'] = float(Bn.mean()); res['B'] = float((Bn & R['G3b']).mean())
            Cn = R['G1'] & R['G2a'] & R['C_seq'] & R['G3a'] & R['G4a'] & R['G4b'] & R['G4c'] & R['G7']
            res['C_noDSR'] = float(Cn.mean()); res['C'] = float((Cn & R['G3b']).mean())
            res['C_seq_power'] = float(R['C_seq'].mean()); res['C_fix_power'] = float(R['C_fix'].mean())
        # marginal binding: drop each gate
        gl = ['G1', 'G2a', 'G2b', 'G3a', 'G3b', 'G4a', 'G4b', 'G4c', 'G7']
        res['drop_one'] = {k: float(np.logical_and.reduce([R[q] for q in gl if q != k]).mean()) for k in gl}
        out[str(IR)] = res
        print(var, IR, {k: round(v, 4) for k, v in res.items() if not isinstance(v, dict)}, flush=True)
    return out


def run_forward(nsim=2000, Ns=(0, 2, 5, 10, 15, 20, 30, 50), batch=250):
    L, eps, sig, eta_src, iid = setup('V1')
    dj19 = d_j[Y19].mean(0)
    cst19, c319, c1019 = c_st[Y19].mean(), c3[Y19].mean(), c10[Y19].mean()
    v_true_m = FAM['npmle']['sd'] ** 2 / 12
    out = {}
    for ii, IR in enumerate(IRS):
        mu = IR / math.sqrt(12) * sig
        for N in Ns:
            rng = np.random.default_rng(20261102 + 1000 * ii + 7 + N)
            m = 12 * N; n = n0 + m
            y19 = np.concatenate([Y19, np.ones(m, bool)])
            segs = [np.concatenate([s, np.full(m, k == 2)]) for k, s in enumerate(SEGS)]
            years = np.concatenate([YEARS, 2026 + (9 + np.arange(m)) // 12])
            cst = np.concatenate([c_st, np.full(m, cst19)]); ct3 = np.concatenate([c3, np.full(m, c319)]); ct10 = np.concatenate([c10, np.full(m, c1019)])
            jn = jd = 0
            for b0 in range(0, nsim, batch):
                S = min(batch, nsim - b0)
                idx, e = draw(rng, S, n, L, eps)
                x = mu + sig * e
                r0 = r0050[idx]; eta = eta_src[idx]
                D = d_j[idx].copy()
                if m:
                    D[:, n0:, :] = dj19[None, None, :]  # new windows: 2019+ mean deviation per j (constant)
                g, _ = gates(x, r0, D, eta, cst, ct3, ct10, y19, segs, years)
                p, _ = spa_p(x, None, MX_FAM, full=False)
                dsr_ok = dsr_rows(x, v_true_m + 1.0 / n, NTR) >= 0.95
                base = g['G1'] & g['G2a'] & g['G2b'] & (p < 0.05) & g['G4a'] & g['G4b'] & g['G4c'] & g['G7']
                jn += base.sum(); jd += (base & dsr_ok).sum()
            out[f'{IR}|{N}'] = {'joint': jd / nsim, 'joint_noDSR': jn / nsim}
            print('fwd', IR, N, out[f'{IR}|{N}'], flush=True)
    return out


def run_clean(nsim=4000, Ts=(5, 10, 15, 20, 30, 50)):
    L, eps, sig, _, _ = setup('V1')
    out = {}
    for ii, IR in enumerate(IRS):
        mu = IR / math.sqrt(12) * sig
        for T in Ts:
            rng = np.random.default_rng(20261102 + 1000 * ii + 900 + T)
            _, e = draw(rng, nsim, 12 * T, L, eps)
            t = tst(mu + sig * e)
            out[f'{IR}|{T}'] = {'t3': float((t >= 3).mean()), 't2': float((t >= 2).mean())}
    return out


if __name__ == '__main__':
    mode = sys.argv[1]
    t0 = time.time()
    if mode == 'main':
        res = run_main('V1', int(sys.argv[2]) if len(sys.argv) > 2 else 4000)
    elif mode == 'variants':
        res = {v: run_main(v, int(sys.argv[2]) if len(sys.argv) > 2 else 2000, irs=[0.5, 0.66], alt=False) for v in VARIANTS}
    elif mode == 'forward':
        res = run_forward(int(sys.argv[2]) if len(sys.argv) > 2 else 2000)
    elif mode == 'clean':
        res = run_clean()
    elif mode == 'typeI':
        L, eps, sig, _, _ = setup('V1')
        rng = np.random.default_rng(5)
        _, e = draw(rng, 200000, 24, L, eps)
        zk = np.cumsum(sig * e, 1) / (sig * np.sqrt(np.arange(1, 25)))
        ziid = np.cumsum(rng.standard_normal((200000, 24)), 1) / np.sqrt(np.arange(1, 25))
        res = {'boot_typeI': float((zk >= 2.43).any(1).mean()), 'iid_typeI': float((ziid >= 2.43).any(1).mean())}
    json.dump(res, open(OUT + f'/sim_{mode}.json', 'w'), indent=1, default=float)
    print(json.dumps(res, default=float)[:3000])
    print('sec', time.time() - t0)
