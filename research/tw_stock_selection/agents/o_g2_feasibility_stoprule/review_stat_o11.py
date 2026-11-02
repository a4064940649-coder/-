#!/usr/bin/env python3
"""反方審查（資料與統計）o11：只讀研究員的 data/、results.json；不 import 研究員程式。
用法：python3 -I -B review_stat_o11.py
檢查：
 1. DSR SR0 重算與「需要 IR」在不同族群定義下的敏感度（是否只是定義造成）。
 2. 拿掉 DSR 後 2.1% 有多少來自 V7X 載體特有的拖累（壓力成本、300萬／1000萬、j 離散）。
 3. NPMLE 尾端機率與後驗區間是否為離散支撐的人工產物（主策略 0.29 與族群最大值 0.30 是否同一件事）。
 4. 基金資料：-100% 月、與 0050 相關性逐年、追蹤誤差。
"""
import json, math
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats

H = Path(__file__).resolve().parent
I = np.load(H / 'data' / 'sim_inputs.npz')
R = json.loads((H / 'results.json').read_text())
out = {}

# ---------- 1. DSR ----------
emc = 0.5772156649
def sr0(var, N):
    return math.sqrt(var) * ((1 - emc) * stats.norm.ppf(1 - 1 / N) + emc * stats.norm.ppf(1 - 1 / (N * math.e)))
def ir_needed(s0, T=245):
    return (s0 + 1.645 / math.sqrt(T - 1)) * math.sqrt(12)
sr = I['sr_all']; T = len(I['ab'])
v_all = np.var(sr, ddof=1)
d = {}
d['raw_N2335'] = (v_all, 2335)
d['raw_var_LiJi161'] = (v_all, 161)
v_pos = np.var(sr[sr > 0], ddof=1); d['positive_members_only_N'] = (v_pos, int((sr > 0).sum()))
v_true = max(v_all - 1 / T, 1e-6); d['var_minus_sampling_LiJi161'] = (v_true, 161)
v_trim = np.var(sr[sr > np.quantile(sr, 0.25)], ddof=1); d['drop_bottom25pct_N1750'] = (v_trim, 1750)
out['DSR'] = {k: {'var_SR_m': round(float(v), 5), 'N': N, 'SR0_ann': round(sr0(v, N) * math.sqrt(12), 3), 'IR_needed': round(ir_needed(sr0(v, N)), 3)} for k, (v, N) in d.items()}
out['DSR']['share_members_SR_negative'] = float((sr < 0).mean())
out['DSR']['researcher_F_main'] = R['FINAL']['dsr_reachability'].get('F_main_rawN')

# ---------- 2. 載體特有拖累 ----------
n = T; L = 6
ab = I['ab']; eps0 = (ab - ab.mean()) / ab.std(ddof=1); sig = ab.std(ddof=1)
D = I['d']; cS = I['c_stress']; c3 = I['c300']; c10 = I['c1000']; y19 = I['y19'].astype(bool); r50 = I['r50']; years = I['years']
Tb = I['Tb_all']
def sb_idx(rng, ns, T, Lb):
    idx = np.empty((ns, T), np.int64); idx[:, 0] = rng.integers(0, T, ns); p = 1 / Lb
    for t in range(1, T):
        new = rng.random(ns) < p
        idx[:, t] = np.where(new, rng.integers(0, T, ns), (idx[:, t - 1] + 1) % T)
    return idx
def tt(x, m=None):
    if m is not None: x = x[:, m]
    return x.mean(1) / x.std(1, ddof=1) * math.sqrt(x.shape[1])
def tmed(xj, m=None):
    return np.median(np.stack([tt(xj[:, :, j], m) for j in range(xj.shape[2])], 1), 1)
def drop3(x, rb):
    uy = np.unique(years); S = x.shape[0]
    Y = np.stack([np.prod(1 + rb[:, years == y] + x[:, years == y], 1) - np.prod(1 + rb[:, years == y], 1) for y in uy], 1)
    top = np.argsort(-Y, 1)[:, :3]; o = np.empty(S)
    for s in range(S):
        keep = ~np.isin(years, uy[top[s]]); xx = x[s, keep]; o[s] = xx.mean() / xx.std(ddof=1) * math.sqrt(keep.sum())
    return o
NS = 3000; q95 = float(np.quantile(Tb, 0.95))
sim = {}
for k, irv in enumerate([0.3, 0.5, 0.66, 0.8]):
    rng = np.random.default_rng(11000 + k)
    idx = sb_idx(rng, NS, n, L)
    x = irv / math.sqrt(12) * sig + sig * eps0[idx]; rb = r50[idx]; Dj = D[idx]
    xj = x[:, :, None] + Dj
    g2_j = (tmed(xj) >= 3) & (tmed(xj, y19) >= 2)
    g2_avg = (tt(x) >= 3) & (tt(x, y19) >= 2)
    g7 = (tmed(xj - c3[None, :, None]) >= 3) & (tmed(xj - c3[None, :, None], y19) >= 2) & (tmed(xj - c10[None, :, None]) >= 3) & (tmed(xj - c10[None, :, None], y19) >= 2)
    spa_approx = tt(x) > q95  # iid t 對族群最大值 95% 分位，粗略
    g4b = drop3(x, rb) >= 2
    sim[str(irv)] = {'G2_javg_no_jdisp': float(g2_avg.mean()), 'G2_median_j': float(g2_j.mean()),
                     'G2_and_G7': float((g2_j & g7).mean()), 'G2_SPAapprox_G4b_javg': float((g2_avg & spa_approx & g4b).mean()),
                     'G2_SPAapprox_G4b_medj_G7': float((g2_j & g7 & spa_approx & g4b).mean())}
out['vehicle_drag_decomp'] = sim
out['drag_pp_m'] = {'stress': float(cS.mean() * 100), 'c300': float(c3.mean() * 100), 'c1000': float(c10.mean() * 100), 'sigma_pp_m': float(sig * 100),
                    'TE_ann': float(sig * math.sqrt(12)), 'stress_in_IR_units': float(cS.mean() / sig * math.sqrt(12)), 'c1000_in_IR_units': float(c10.mean() / sig * math.sqrt(12))}

# ---------- 3. NPMLE 尾端 ----------
dec = R['FAMILY']['deconvolution']['main_folder_dedup_iid']
sup = {float(k): v for k, v in dec['support'].items()}
g = np.array(sorted(sup)); w = np.array([sup[k] for k in g]); w = w / w.sum()
def post(x0, se):
    p = w * np.exp(-0.5 * ((x0 - g) / se) ** 2); p /= p.sum(); return float((p * g).sum()), float(p[g >= 0.5].sum())
out['NPMLE'] = {'support_max': float(g.max()), 'mass_above_0.33': float(w[g > 0.331].sum()),
                'posterior_mean_for_obs': {str(x0): round(post(x0, 0.22)[0], 3) for x0 in (0.4, 0.556, 0.8, 0.94, 1.5, 3.0)},
                'note': '任何觀察值的後驗平均都被支撐上限 0.33 封頂；主策略 0.29 與族群最大值 0.30 是同一個先驗上限，不是獨立來源'}
# 族群觀察 IR 的離散程度 vs 支撐：用 Li-Ji 有效 N 137 粗估支撐上限的不確定性（觀察 IR 前 1% 的 bootstrap）
fam = pd.read_parquet(H / 'data' / 'family_ir.parquet')
out['family_cols'] = list(fam.columns)[:20]

# ---------- 4. 基金 ----------
fm = pd.read_parquet(H / 'data' / 'fund_monthly.parquet'); fi = pd.read_parquet(H / 'data' / 'fund_ir.parquet')
out['fund'] = {'n_abs_r_ge_0.4': int((fm.r.abs() >= 0.4).sum()), 'n_r_eq_minus1': int((fm.r <= -0.99).sum()),
               'r_minus1_rows': fm[fm.r <= -0.99][['date', 'fund_id']].astype(str).values.tolist()[:10]}
s15 = fi[fi.alive & (fi.n >= 180)]
out['fund']['TE_median_surv15'] = float(s15.TE.median())
out['fund']['t_needed_IR_for_t3_at_n'] = float(3 / math.sqrt(s15.n.median() / 12))
dead = fi[(fi['first'] <= '2011-10-31') & ~fi.alive]
out['fund']['dead_with_minus1'] = int(dead.fund_id.isin(fm[fm.r <= -0.99].fund_id).sum())
print(json.dumps(out, ensure_ascii=False, indent=1, default=float))

# ---------- 2b. 拿掉 DSR 與 G7（載體特有容量拖累）後，(a) 是否仍 < 5% ----------
seg = I['seg']; eta = I['eta_raw'] - I['eta_raw'].mean(0)
def geo(xs, rb, m=None):
    if m is not None: xs = xs[:, m]; rb = rb[:, m]
    Tn = xs.shape[1]
    return np.exp(np.log(np.clip(1 + rb[:, :, None] + xs, 1e-6, None)).sum(1) * 12 / Tn) - np.exp(np.log1p(rb).sum(1) * 12 / Tn)[:, None]
res2 = {}
for k, irv in enumerate([0.5, 0.66]):
    rng = np.random.default_rng(12000 + k)
    idx = sb_idx(rng, NS, n, L)
    x = irv / math.sqrt(12) * sig + sig * eps0[idx]; rb = r50[idx]; xj = x[:, :, None] + D[idx]
    gx = geo(xj, rb)
    G1 = (np.median(gx, 1) > 0) & (np.percentile(gx, 10, 1) > 0) & (np.median(geo(xj - cS[None, :, None], rb), 1) > 0)
    G2 = (tmed(xj) >= 3) & (tmed(xj, y19) >= 2)
    G3a = tt(x) > q95
    G4a = np.ones(NS, bool)
    for s_ in (1, 2, 3): G4a &= np.median(geo(xj, rb, seg == s_), 1) > 0
    G4b = np.median(np.stack([drop3(xj[:, :, j], rb) for j in range(13)], 1), 1) >= 2
    G4c = (geo(x[:, :, None] + eta[idx], rb) > 0).all(1)
    res2[str(irv)] = {'G1': float(G1.mean()), 'joint_noDSR_noG7': float((G1 & G2 & G3a & G4a & G4b & G4c).mean()),
                      'joint_noDSR_noG7_noG1stress': float(((np.median(gx, 1) > 0) & (np.percentile(gx, 10, 1) > 0) & G2 & G3a & G4a & G4b & G4c).mean())}
print(json.dumps({'no_DSR_no_G7': res2}, ensure_ascii=False, indent=1))
