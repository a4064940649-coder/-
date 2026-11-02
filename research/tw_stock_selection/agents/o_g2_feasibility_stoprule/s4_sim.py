#!/usr/bin/env python3
"""（3）（4）模擬：真實 IR 已知的方法在 G1～G7（及替代門檻 A／B／C、前瞻延長）下的通過機率。
用法：python3 -I -B s4_sim.py <variant>   variant∈{V1,V2,V3,V4,V5,V6,V7}；V1 另跑前瞻延長與純長度版本。
斷言 PREREG 與 PREREG2 存在且雜湊相符。中間檔放 scratchpad；彙總寫 data/sim_<variant>.json。"""
import json, sys, time, math
import numpy as np
from scipy import stats
sys.path.insert(0, __import__('os').path.dirname(__file__))
import g2lib as G

G.check_prereg(need2=True)
VAR = sys.argv[1] if len(sys.argv) > 1 else 'V1'
VIDX = int(VAR[1:])
NSIM = 4000
IRS = [0.0, 0.2, 0.3, 0.4, 0.5, 0.66, 0.8, 1.0]
I = np.load(G.DATA / 'sim_inputs.npz')
n = len(I['ab'])
L = {'V2': 3, 'V3': 12}.get(VAR, 6)
ab = I['ab']; xv = I['xv']
src = xv if VAR == 'V6' else ab
eps0 = (src - src.mean()) / src.std(ddof=1)
sig = src.std(ddof=1)
if VAR == 'V4': sig = ab[I['y19'].astype(bool)].std(ddof=1)
r50 = I['r50']; D = I['d']; cS = I['c_stress']; c3 = I['c300']; c10 = I['c1000']
eta = I['eta_raw'] if VAR == 'V7' else I['eta_raw'] - I['eta_raw'].mean(0)
Tb_all = I['Tb_all']; Tb_main = I['Tb_main']; sr_fam = I['sr_all']
years = I['years']; seg = I['seg']; y19 = I['y19'].astype(bool)
Wspa = G.SP.sb_counts(n, 6, 10000, 20261102).astype(np.float32)
N_FAM = len(sr_fam) + 1

def sb_idx(rng, ns, T, Lb, n0=n):
    idx = np.empty((ns, T), np.int64); idx[:, 0] = rng.integers(0, n0, ns)
    p = 1.0 / Lb
    for t in range(1, T):
        new = rng.random(ns) < p
        idx[:, t] = np.where(new, rng.integers(0, n0, ns), (idx[:, t - 1] + 1) % n0)
    return idx

def tst(x, axis=1, mask=None):
    if mask is not None: x = np.compress(mask, x, axis=axis)
    m = x.mean(axis=axis); s = x.std(axis=axis, ddof=1)
    return m / s * np.sqrt(x.shape[axis])

def geo(xs, rb, mask=None):
    """xs: (S,T,J) 超額；rb: (S,T) 0050。回傳 (S,J) 幾何年化差 %。"""
    if mask is not None: xs = xs[:, mask]; rb = rb[:, mask]
    T = xs.shape[1]
    lb = np.log1p(rb).sum(1)
    ls = np.log(np.clip(1 + rb[:, :, None] + xs, 1e-6, None)).sum(1)
    return (np.exp(ls * 12 / T) - np.exp(lb * 12 / T)[:, None]) * 100

def lrv_pr(x, Lk=6):
    """x: (S,T)。Politis-Romano 核的 ω²（與 spa_lib.lrv 'pr' 相同）。"""
    S, T = x.shape; e = x - x.mean(1, keepdims=True)
    s = (e * e).sum(1) / T; p = 1.0 / Lk
    for i in range(1, T):
        k = (T - i) / T * (1 - p) ** i + i / T * (1 - p) ** (T - i)
        if k < 1e-8 and i > 5 * Lk: break
        s = s + 2 * k * (e[:, i:] * e[:, :-i]).sum(1) / T
    return np.maximum(s, 1e-14)

def spa_p(xa, Tb, full=True):
    """xa: (S,T) javg 超額（T＝n）。回傳焦點 SPA_c p（族群最大值＋候選自己的重新置中統計量）。"""
    S, T = xa.shape
    om2 = lrv_pr(xa); om = np.sqrt(om2); sn = math.sqrt(T); dbar = xa.mean(1)
    t = sn * dbar / om
    if not full:
        return np.array([(Tb >= ti).mean() if ti > 0 else 1.0 for ti in t]), t
    thr = -np.sqrt(om2 / T * 2 * math.log(math.log(T)))
    g = np.where(dbar >= thr, dbar, 0.0)
    Mb = (Wspa @ xa.T.astype(np.float32)) / T          # (B,S)
    Z = (Mb - g[None, :]) * (sn / om)[None, :]
    Tmax = np.maximum(np.maximum(Tb[:, None], Z), 0.0)
    p = (Tmax >= t[None, :]).mean(0)
    p[t <= 0] = 1.0
    return p, t

def dsr_vec(xa, var_sr=None, N=N_FAM):
    S, T = xa.shape
    sr = xa.mean(1) / xa.std(1, ddof=1)
    g3 = stats.skew(xa, axis=1); g4 = stats.kurtosis(xa, axis=1, fisher=False)
    emc = 0.5772156649
    if var_sr is None:
        v = (np.var(sr_fam, ddof=1) * (N - 2) + 0.0) / (N - 1)  # 近似：加入候選後的變異（候選逐一加入差異 < 1e-4）
        v = np.array([np.var(np.append(sr_fam, s), ddof=1) for s in sr]) if S <= 4000 else np.full(S, v)
    else:
        v = np.full(S, var_sr)
    sr0 = np.sqrt(v) * ((1 - emc) * stats.norm.ppf(1 - 1 / N) + emc * stats.norm.ppf(1 - 1 / (N * math.e)))
    z = (sr - sr0) * np.sqrt(T - 1) / np.sqrt(np.maximum(1 - g3 * sr + (g4 - 1) / 4 * sr ** 2, 1e-12))
    return stats.norm.cdf(z), sr0

def drop3_t(xj, rb, yrs):
    """xj: (S,T,J)。每個 j 拿掉超額最好的 3 個日曆年後的 t（回傳 (S,J)）。"""
    uy = np.unique(yrs); S, T, J = xj.shape
    lr = np.log(np.clip(1 + rb[:, :, None] + xj, 1e-6, None)); lb = np.log1p(rb)
    Y = np.stack([np.exp(lr[:, yrs == y].sum(1)) - np.exp(lb[:, yrs == y].sum(1))[:, None] for y in uy], axis=1)  # (S,Y,J)
    rank = np.argsort(-Y, axis=1)[:, :3, :]     # (S,3,J)
    out = np.empty((S, J))
    for j in range(J):
        drop = np.zeros((S, len(uy)), bool)
        np.put_along_axis(drop, rank[:, :, j], True, axis=1)
        keepw = ~drop[:, np.searchsorted(uy, yrs)]   # (S,T)
        x = xj[:, :, j]
        cnt = keepw.sum(1); m = (x * keepw).sum(1) / cnt
        v = (((x - m[:, None]) ** 2) * keepw).sum(1) / (cnt - 1)
        out[:, j] = m / np.sqrt(v) * np.sqrt(cnt)
    return out

def evaluate(x, rb, Dj, cs, cc3, cc10, et, yrs, sg, y19m, spa=True, var_sr=None, Tb=Tb_all, full_spa=True):
    """x: (S,T) javg 真實超額；Dj: (S,T,13)；cs/cc3/cc10: (T,) 或 (S,T)；et: (S,T,6)。回傳各門檻布林 dict 與輔助量。"""
    xj = x[:, :, None] + Dj
    med = lambda a: np.median(a, axis=1)
    gx = geo(xj, rb)
    G1 = (med(gx) > 0) & (np.percentile(gx, 10, axis=1) > 0) & (med(geo(xj - np.atleast_2d(cs)[..., None] if np.ndim(cs) == 2 else xj - cs[None, :, None], rb)) > 0)
    tA = med(tst(xj, axis=1)); tY = med(tst(xj, axis=1, mask=y19m))
    G2a = tA >= 3.0; G2b = tY >= 2.0
    def lvl(c):
        xl = xj - (c[..., None] if np.ndim(c) == 2 else c[None, :, None])
        return med(tst(xl, axis=1)), med(tst(xl, axis=1, mask=y19m))
    t3A, t3Y = lvl(cc3); t10A, t10Y = lvl(cc10)
    G7 = (t3A >= 3) & (t3Y >= 2) & (t10A >= 3) & (t10Y >= 2)
    G4a = np.ones(len(x), bool)
    for s in (1, 2, 3): G4a &= med(geo(xj, rb, mask=(sg == s))) > 0
    G4b = med(drop3_t(xj, rb, yrs)) >= 2.0
    nb = x[:, :, None] + et
    G4c = (geo(nb, rb) > 0).all(1)
    out = dict(G1=G1, G2a=G2a, G2b=G2b, G4a=G4a, G4b=G4b, G4c=G4c, G7=G7, tA=tA, tY=tY, t3A=t3A, t3Y=t3Y, t10A=t10A, t10Y=t10Y)
    if spa:
        if x.shape[1] == n:
            p, tj = spa_p(x, Tb, full=full_spa)
        else:
            p, tj = spa_p(x, Tb, full=False)
        out['p_spa'] = p; out['G3a'] = p < 0.05
        dsr, sr0 = dsr_vec(x, var_sr); out['dsr'] = dsr; out['G3b'] = dsr >= 0.95; out['SR0'] = sr0
    return out

GATES = ['G1', 'G2a', 'G2b', 'G3a', 'G3b', 'G4a', 'G4b', 'G4c', 'G7']
def joint(o, gates=GATES): return np.logical_and.reduce([o[g] for g in gates])

def run_hist(irv, seed, ns=NSIM, with_alt=True):
    rng = np.random.default_rng(seed)
    idx = sb_idx(rng, ns, n, L)
    eps = rng.standard_normal((ns, n)) if VAR == 'V5' else eps0[idx]
    mu = irv / math.sqrt(12) * sig
    x = mu + sig * eps
    rb = r50[idx]; Dj = D[idx]; et = eta[idx]
    o = evaluate(x, rb, Dj, cS, c3, c10, et, years, seg, y19)
    res = {g: float(o[g].mean()) for g in GATES}
    res['joint_G1_G7'] = float(joint(o).mean())
    res['G2'] = float((o['G2a'] & o['G2b']).mean()); res['G3'] = float((o['G3a'] & o['G3b']).mean()); res['G4'] = float((o['G4a'] & o['G4b'] & o['G4c']).mean())
    res['joint_without'] = {g: float(joint(o, [h for h in GATES if h != g]).mean()) for g in GATES}
    res['joint_without_G3b'] = res['joint_without']['G3b']
    res['median_tA'] = float(np.median(o['tA'])); res['median_tY'] = float(np.median(o['tY'])); res['median_p_spa'] = float(np.median(o['p_spa']))
    res['median_dsr'] = float(np.median(o['dsr'])); res['SR0_monthly'] = float(np.median(o['SR0']))
    if with_alt:
        # A：全期 t≥2＋Y19 超額>0＋家族限定 SPA p<0.10
        pM, _ = spa_p(x, Tb_main, full=True)
        gY = np.median(geo(x[:, :, None] + Dj, rb, mask=y19), axis=1)
        A = (o['tA'] >= 2.0) & (gY > 0) & (pM < 0.10)
        res['A'] = float(A.mean()); res['A_plus_G1_G7'] = float((A & o['G1'] & o['G7']).mean())
        res['A_plus_G1_G7_relaxed'] = float((A & o['G1'] & (o['t3A'] >= 2) & (o['t10A'] >= 2)).mean())
        # B：G2a 改 t≥2.5（G7 各級同樣改 2.5）
        G2aB = o['tA'] >= 2.5; G7B = (o['t3A'] >= 2.5) & (o['t3Y'] >= 2) & (o['t10A'] >= 2.5) & (o['t10Y'] >= 2)
        res['B'] = float((o['G1'] & G2aB & o['G2b'] & o['G3a'] & o['G3b'] & o['G4a'] & o['G4b'] & o['G4c'] & G7B).mean())
        res['B_without_G3b'] = float((o['G1'] & G2aB & o['G2b'] & o['G3a'] & o['G4a'] & o['G4b'] & o['G4c'] & G7B).mean())
        # C：G2b 改前瞻 24 個月序貫檢定（G7 各級只要求 G2a）
        fidx = sb_idx(rng, ns, 24, L)
        ef = rng.standard_normal((ns, 24)) if VAR == 'V5' else eps0[fidx]
        xf = mu + sig * ef
        z = np.cumsum(xf, 1) / (sig * np.sqrt(np.arange(1, 25)))[None, :]
        seqpass = (z >= C_POCOCK).any(1); fixpass = z[:, -1] >= 1.645
        G7C = (o['t3A'] >= 3) & (o['t10A'] >= 3)
        baseC = o['G1'] & o['G2a'] & o['G3a'] & o['G3b'] & o['G4a'] & o['G4b'] & o['G4c'] & G7C
        res['C'] = float((baseC & seqpass).mean()); res['C_without_G3b'] = float((o['G1'] & o['G2a'] & o['G3a'] & o['G4a'] & o['G4b'] & o['G4c'] & G7C & seqpass).mean())
        res['power_seq24'] = float(seqpass.mean()); res['power_fixed24'] = float(fixpass.mean())
    return res, o

# Pocock 常數邊界（iid 常態、IR=0、24 次）
rng0 = np.random.default_rng(20261102 + 999)
Z0 = np.cumsum(rng0.standard_normal((200000, 24)), 1) / np.sqrt(np.arange(1, 25))[None, :]
C_POCOCK = float(np.quantile(Z0.max(1), 0.95))

t0 = time.time()
OUT = {'variant': VAR, 'L': L, 'sigma_monthly': float(sig), 'nsim': NSIM, 'C_pocock': C_POCOCK, 'by_IR': {}}
# 預檢：SPA 近似 vs 完整（200 次）
chk = {}
for k, irv in enumerate(IRS):
    seed = 20261102 + 1000 * k + 100 * VIDX
    res, o = run_hist(irv, seed)
    OUT['by_IR'][str(irv)] = res
    if irv in (0.5, 0.66):
        rng = np.random.default_rng(seed + 7); sel = rng.choice(NSIM, min(200, NSIM), replace=False)
        idx = sb_idx(np.random.default_rng(seed), NSIM, n, L)
        xx = (irv / math.sqrt(12) * sig + sig * (np.random.default_rng(seed).standard_normal((NSIM, n)) if VAR == 'V5' else eps0[idx]))[sel]
        pa, _ = spa_p(xx, Tb_all, full=False); pf, _ = spa_p(xx, Tb_all, full=True)
        chk[str(irv)] = {'max_abs_diff_p': float(np.abs(pa - pf).max()), 'mean_abs_diff_p': float(np.abs(pa - pf).mean()), 'pass_diff': float(((pa < .05) != (pf < .05)).mean())}
    print(VAR, irv, {kk: (round(v, 4) if isinstance(v, float) else '') for kk, v in res.items() if not isinstance(v, dict)}, round(time.time() - t0, 1), flush=True)
OUT['spa_approx_vs_full_200'] = chk
# 主策略後驗點（描述用，非事前登記格點）
post = G.load_res()['FAMILY']['posterior_main']
OUT['main_posterior_points'] = {}
for lab, irv in [('AB_V4_post_mean', post['AB_V4_ideal']['post_mean']), ('V7X_post_mean', post['V7X_javg_100w_GROW_normal']['post_mean']),
                 ('AB_V4_measured', post['AB_V4_ideal']['x_obs'])]:
    res, _ = run_hist(irv, 20261102 + 50000 + 100 * VIDX + int(irv * 100))
    OUT['main_posterior_points'][lab] = {'IR': irv, **res}

if VAR == 'V1':
    # 前瞻延長：歷史 245 窗＋N 年新窗
    Ns = [0, 2, 5, 10, 15, 20, 30, 50]; NS2 = 2000
    var_true_m = (G.load_res()['FAMILY']['deconvolution']['main_folder_dedup_iid']['prior_sd'] / math.sqrt(12)) ** 2
    d19 = D[y19].mean(0); dm = D.mean(0)
    fwd = {}
    for k, irv in enumerate(IRS):
        fwd[str(irv)] = {}
        for N in Ns:
            rng = np.random.default_rng(20261102 + 7000 + 1000 * k + N)
            T = n + 12 * N
            idx = sb_idx(rng, NS2, T, L)
            x = irv / math.sqrt(12) * sig + sig * eps0[idx]
            rb = r50[idx]
            Dj = D[idx]
            if N > 0: Dj[:, n:] = Dj[:, n:] - dm + d19
            def ext(c): return np.concatenate([c, np.full(12 * N, c[y19].mean())])
            yrs = np.concatenate([years, np.repeat(np.arange(2027, 2027 + N), 12)]) if N > 0 else years
            sg = np.concatenate([seg, np.full(12 * N, 3)]); y19m = np.concatenate([y19, np.ones(12 * N, bool)])
            vs = var_true_m + 1.0 / T
            o = evaluate(x, rb, Dj, ext(cS), ext(c3), ext(c10), eta[idx], yrs, sg, y19m, var_sr=vs)
            fwd[str(irv)][str(N)] = {'joint': float(joint(o).mean()), 'joint_without_G3b': float(joint(o, [g for g in GATES if g != 'G3b']).mean()),
                                     **{g: float(o[g].mean()) for g in GATES}}
        print('fwd', irv, {N: round(v['joint'], 3) for N, v in fwd[str(irv)].items()}, {N: round(v['joint_without_G3b'], 3) for N, v in fwd[str(irv)].items()}, flush=True)
    OUT['forward_extension'] = fwd
    def need(key, thr):
        o = {}
        for irv in fwd:
            ok = [int(N) for N in Ns if fwd[irv][str(N)][key] >= thr]
            o[irv] = min(ok) if ok else '>50'
        return o
    OUT['years_needed'] = {'joint_50': need('joint', .5), 'joint_80': need('joint', .8), 'joint_without_G3b_50': need('joint_without_G3b', .5), 'joint_without_G3b_80': need('joint_without_G3b', .8)}
    # 純長度：全新乾淨樣本 T 年，只看 t≥3、t≥2（javg）
    clean = {}
    for k, irv in enumerate(IRS):
        clean[str(irv)] = {}
        for Ty in [5, 8, 10, 15, 20, 30, 40, 50]:
            rng = np.random.default_rng(20261102 + 9000 + 1000 * k + Ty)
            idx = sb_idx(rng, NS2, 12 * Ty, L)
            x = irv / math.sqrt(12) * sig + sig * eps0[idx]
            t = tst(x)
            clean[str(irv)][str(Ty)] = {'P_t_ge3': float((t >= 3).mean()), 'P_t_ge2': float((t >= 2).mean())}
    OUT['clean_length'] = clean
OUT['runtime_s'] = time.time() - t0
(G.DATA / f'sim_{VAR}.json').write_text(json.dumps(OUT, ensure_ascii=False, indent=1, default=float))
print('done', VAR, round(time.time() - t0, 1))
