"""Family stats: IR distribution, gate pass counts, SPA_c focus, DSR, Li-Ji, dedup, NPMLE."""
import json, sys, math
import numpy as np, pandas as pd
sys.path.insert(0, '/home/user/-/research/tw_stock_selection/agents/o_g2_feasibility_stoprule/replicate_o11')
from rlib import *
OUT = '/tmp/claude-0/-home-user--/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/rep_o11'
Z = np.load(OUT + '/inputs.npz', allow_pickle=True)
X = Z['X']; names = list(Z['names']); folders = list(Z['folders']); fmain = Z['fmain'].astype(bool)
r0 = Z['r0050']; dates = pd.DatetimeIndex(Z['dates'])
n, K = X.shape
y19 = dates >= '2019-01-01'
seg = [dates <= '2012-12-31', (dates >= '2013-01-01') & (dates <= '2018-12-31'), y19]
res = {}
q = [1, 5, 10, 25, 50, 75, 90, 95, 99, 100]
IR = ir(X); IR19 = ir(X[y19])
res['IR_all_q'] = dict(zip(map(str, q), np.round(np.nanpercentile(IR, q), 3).tolist()))
res['IR_y19_q'] = dict(zip(map(str, q), np.round(np.nanpercentile(IR19, q), 3).tolist()))
res['IR_max_member'] = names[int(np.nanargmax(IR))]
res['IR19_max_member'] = names[int(np.nanargmax(IR19))]
excl = np.array([bool(EXCL_RE.search(s)) for s in names])

T = tstat(X); T19 = tstat(X[y19])
G2a = T >= 3; G2b = T19 >= 2
# G4a segments geometric excess
Xf = np.nan_to_num(X)  # NaN windows -> excess 0 (treated as tracking 0050)
G4a = np.ones(K, bool)
for s in seg:
    G4a &= geo_ex(Xf[s] + r0[s, None], r0[s]) > 0
# G4b drop best 3 calendar years (by member's calendar-year geometric excess)
yrs = dates.year.values
uy = np.unique(yrs)
ysum = np.column_stack([np.prod(1 + Xf[yrs == y] + r0[yrs == y, None], 0) - np.prod(1 + r0[yrs == y]) for y in uy])  # K x Y, geometric yearly excess
G4b = np.zeros(K, bool); t_drop = np.zeros(K)
for k in range(K):
    best = uy[np.argsort(-ysum[k])[:3]]
    keep = ~np.isin(yrs, best)
    t_drop[k] = tstat(X[keep, k]); G4b[k] = t_drop[k] >= 2
# SPA_c focus
idx = sb_indices(n, 6, 10000, 20261102)
W = counts_from_idx(idx, n)
tt, mx, _ = spa_boot_max(Xf, W)
mx0 = np.maximum(mx, 0)
p_focus = np.array([(mx0 >= tt[k]).mean() for k in range(K)])
G3a = p_focus < 0.05
# DSR
SRm = np.nanmean(X, 0) / np.nanstd(X, 0, ddof=1)
v_sr = float(np.var(SRm, ddof=1))
dsr_v = np.array([dsr(X[~np.isnan(X[:, k]), k], v_sr, K)[0] for k in range(K)])
G3b = dsr_v >= 0.95
_, sr0 = dsr(X[:, 0][~np.isnan(X[:, 0])], v_sr, K)
cnt = lambda m: int(m.sum())
res['counts'] = {'G2a_t3': cnt(G2a), 'G2b_y19t2': cnt(G2b), 'both': cnt(G2a & G2b), 'both_excl_lev_tsmc': cnt(G2a & G2b & ~excl),
                 'G3a_spa': cnt(G3a), 'G3b_dsr': cnt(G3b), 'G4a': cnt(G4a), 'G4b': cnt(G4b),
                 'joint': cnt(G2a & G2b & G3a & G3b & G4a & G4b), 'joint_noDSR': cnt(G2a & G2b & G3a & G4a & G4b)}
res['both_names'] = [names[k] for k in np.where(G2a & G2b)[0]]
res['spa_names'] = [names[k] for k in np.where(G3a)[0]]


def req_ir(sr0_m, T_=n):
    # IR needed for DSR=0.95 with skew 0, kurt 3
    from scipy.optimize import brentq
    f = lambda s: (s - sr0_m) * math.sqrt(T_ - 1) / math.sqrt(1 + 0.5 * s * s) - stats.norm.ppf(0.95)
    return brentq(f, 0, 3) * math.sqrt(12)


# Li-Ji, dedup
lj_all = liji(Xf)
keep = []
for fo in dict.fromkeys(folders):
    ids = [k for k in range(K) if folders[k] == fo]
    kept = []
    for k in ids:
        ok = True
        for j in kept:
            m = ~np.isnan(X[:, k]) & ~np.isnan(X[:, j])
            if np.corrcoef(X[m, k], X[m, j])[0, 1] > 0.95:
                ok = False; break
        if ok: kept.append(k)
    keep += kept
keep = np.array(sorted(keep))
lj_dd = liji(Xf[:, keep])
sr0_lj = sr0_of(v_sr, lj_all)
fm = np.where(fmain)[0]
v_fm = float(np.var(SRm[fm], ddof=1))
sr0_fm = sr0_of(v_fm, len(fm) + 1)
res['DSR'] = {'var_SR': v_sr, 'SR0_m': sr0, 'SR0_ann': sr0 * math.sqrt(12), 'req_IR': req_ir(sr0),
              'LiJi': lj_all, 'req_IR_LiJi': req_ir(sr0_lj), 'N_Fmain': int(len(fm)), 'req_IR_Fmain': req_ir(sr0_fm)}
res['dedup'] = {'n_keep': int(len(keep)), 'LiJi_dedup': lj_dd}

# NPMLE on dedup sample
def se_of(x):
    x = x[~np.isnan(x)]; sr = x.mean() / x.std(ddof=1)
    return math.sqrt(12) * math.sqrt((1 + sr * sr / 2) / len(x))
y = IR[keep]; se = np.array([se_of(X[:, k]) for k in keep])
th, w, _ = npmle(y, se)
res['npmle'] = {'P>0.3': float(w[th > 0.3 + 1e-9].sum()), 'P>=0.5': float(w[th >= 0.5 - 1e-9].sum()),
                'P>=0.66': float(w[th >= 0.66 - 1e-9].sum()), 'mean': float((w * th).sum()), 'sd': float(math.sqrt((w * th * th).sum() - (w * th).sum() ** 2))}
kmax = int(np.nanargmax(IR))
res['post_family_max'] = posterior(th, w, IR[kmax], se_of(X[:, kmax])) | {'IR': float(IR[kmax])}
# normal EB
mu = y.mean(); tau2 = max(y.var(ddof=1) - np.mean(se ** 2), 1e-9)
res['normalEB'] = {'mu': float(mu), 'tau': math.sqrt(tau2), 'P>=0.5': float(1 - stats.norm.cdf((0.5 - mu) / math.sqrt(tau2))),
                   'P>0.3': float(1 - stats.norm.cdf((0.3 - mu) / math.sqrt(tau2)))}
def neb(yy, ss):
    b = tau2 / (tau2 + ss * ss); return float(mu + b * (yy - mu))
ab = Z['ab']; vx = Z['v7x_javg']
res['post_ABV4'] = posterior(th, w, ir(ab), se_of(ab)) | {'IR': float(ir(ab)), 'normalEB': neb(ir(ab), se_of(ab))}
res['post_V7X'] = posterior(th, w, ir(vx), se_of(vx)) | {'IR': float(ir(vx)), 'normalEB': neb(ir(vx), se_of(vx))}
# infinite-sample SR0 : var = deconvolved true IR variance (monthly)
v_true_m = res['npmle']['sd'] ** 2 / 12
res['DSR']['SR0_ann_inf'] = sr0_of(v_true_m, K + 1) * math.sqrt(12)
# with full-family NPMLE variance as sensitivity
yA = IR; seA = np.array([se_of(X[:, k]) for k in range(K)])
thA, wA, _ = npmle(yA, seA)
vA = (wA * thA * thA).sum() - (wA * thA).sum() ** 2
res['DSR']['SR0_ann_inf_fullfam'] = sr0_of(vA / 12, K + 1) * math.sqrt(12)
res['npmle_full'] = {'P>0.3': float(wA[thA > 0.3 + 1e-9].sum()), 'P>=0.5': float(wA[thA >= 0.5 - 1e-9].sum())}
np.savez(OUT + '/family_stats.npz', keep=keep, th=th, w=w, mx=mx, v_sr=v_sr, IR=IR)
json.dump(res, open(OUT + '/family.json', 'w'), ensure_ascii=False, indent=1, default=float)
print(json.dumps(res, ensure_ascii=False, indent=1, default=float))
