#!/usr/bin/env python3
"""反方審查（成本與可成交）o11：只讀研究員的 data/sim_inputs.npz 與 results.json；不 import 研究員程式。
用法：python3 -I -B review_exec_o11.py
獨立輕量模擬（G3a 用族群最大值近似，研究員自己量過與完整版判定差 <= 0.5%；不含 DSR）：
 S0 基準（應接近研究員 V1 拿掉 DSR 的數字）
 S1 容量拖累歸零（大型股／低周轉載體的情境：G7 是否只是 V7X 中小型載體的特性）
 S2 壓力成本拖累歸零
 S3 所有載體拖累（壓力、300萬、1000萬、j 離散）歸零
 S4 G7 嚴格讀法：300萬／1000萬也要壓力成本下 G1 成立（使用者「都成立」）
 S5 j 離散 D 歸零
另：理想→可執行尺度換算後，替代門檻 A、A+G1+G7 放寬在 IR 0.05～0.15 的機率。
"""
import json, math
from pathlib import Path
import numpy as np

H = Path(__file__).resolve().parent
I = np.load(H / 'data' / 'sim_inputs.npz')
R = json.loads((H / 'results.json').read_text())
ab = I['ab']; n = len(ab); r50 = I['r50']; D0 = I['d']; cS0 = I['c_stress']; c30 = I['c300']; c100 = I['c1000']
eta = I['eta_raw'] - I['eta_raw'].mean(0)
Tb_all = np.sort(I['Tb_all']); Tb_main = np.sort(I['Tb_main'])
years = I['years']; seg = I['seg']; y19 = I['y19'].astype(bool)
sig = ab.std(ddof=1); eps0 = (ab - ab.mean()) / sig
L = 6; NS = 3000

def sb_idx(rng, ns, T):
    idx = np.empty((ns, T), np.int64); idx[:, 0] = rng.integers(0, n, ns)
    for t in range(1, T):
        new = rng.random(ns) < 1 / L
        idx[:, t] = np.where(new, rng.integers(0, n, ns), (idx[:, t - 1] + 1) % n)
    return idx

def tst(x, mask=None):
    if mask is not None: x = x[:, mask]
    return x.mean(1) / x.std(1, ddof=1) * np.sqrt(x.shape[1])

def geo(xs, rb, mask=None):
    if mask is not None: xs = xs[:, mask]; rb = rb[:, mask]
    T = xs.shape[1]
    lb = np.log1p(rb).sum(1); ls = np.log(np.clip(1 + rb[:, :, None] + xs, 1e-6, None)).sum(1)
    return (np.exp(ls * 12 / T) - np.exp(lb * 12 / T)[:, None]) * 100

def drop3(xj, rb):
    uy = np.unique(years); S, T, J = xj.shape
    lr = np.log(np.clip(1 + rb[:, :, None] + xj, 1e-6, None)); lb = np.log1p(rb)
    Y = np.stack([np.exp(lr[:, years == y].sum(1)) - np.exp(lb[:, years == y].sum(1))[:, None] for y in uy], 1)
    rk = np.argsort(-Y, 1)[:, :3, :]; out = np.empty((S, J)); pos = np.searchsorted(uy, years)
    for j in range(J):
        dr = np.zeros((S, len(uy)), bool); np.put_along_axis(dr, rk[:, :, j], True, 1)
        kw = ~dr[:, pos]; x = xj[:, :, j]; c = kw.sum(1); m = (x * kw).sum(1) / c
        v = (((x - m[:, None]) ** 2) * kw).sum(1) / (c - 1); out[:, j] = m / np.sqrt(v) * np.sqrt(c)
    return out

def lrv(x, Lk=6):
    S, T = x.shape; e = x - x.mean(1, keepdims=True); s = (e * e).sum(1) / T; p = 1 / Lk
    for i in range(1, T):
        k = (T - i) / T * (1 - p) ** i + i / T * (1 - p) ** (T - i)
        if k < 1e-8 and i > 5 * Lk: break
        s = s + 2 * k * (e[:, i:] * e[:, :-i]).sum(1) / T
    return s

def spa_approx(x, Tb):
    t = math.sqrt(n) * x.mean(1) / np.sqrt(lrv(x))
    p = 1 - np.searchsorted(Tb, t, side='left') / len(Tb)
    return np.where(t > 0, p, 1.0)

med = lambda a: np.median(a, 1)

def run(irv, seed, D=D0, cS=cS0, c3=c30, c10=c100, strictG7=False):
    rng = np.random.default_rng(seed); idx = sb_idx(rng, NS, n)
    x = irv / math.sqrt(12) * sig + sig * eps0[idx]; rb = r50[idx]; xj = x[:, :, None] + D[idx]
    gx = geo(xj, rb)
    G1 = (med(gx) > 0) & (np.percentile(gx, 10, 1) > 0) & (med(geo(xj - cS[None, :, None], rb)) > 0)
    tA = med(np.stack([tst(xj[:, :, j]) for j in range(13)], 1)); tY = med(np.stack([tst(xj[:, :, j], y19) for j in range(13)], 1))
    def lv(c):
        xl = xj - c[None, :, None]
        return med(np.stack([tst(xl[:, :, j]) for j in range(13)], 1)), med(np.stack([tst(xl[:, :, j], y19) for j in range(13)], 1)), xl
    a3, y3, x3 = lv(c3); a10, y10, x10 = lv(c10)
    G7 = (a3 >= 3) & (y3 >= 2) & (a10 >= 3) & (y10 >= 2)
    if strictG7:
        for xl in (x3, x10):
            g = geo(xl, rb)
            G7 &= (med(g) > 0) & (np.percentile(g, 10, 1) > 0) & (med(geo(xl - cS[None, :, None], rb)) > 0)
    G4a = np.ones(NS, bool)
    for s in (1, 2, 3): G4a &= med(geo(xj, rb, seg == s)) > 0
    G4b = med(drop3(xj, rb)) >= 2
    G4c = (geo(x[:, :, None] + eta[idx], rb) > 0).all(1)
    G3a = spa_approx(x, Tb_all) < 0.05
    gates = dict(G1=G1, G2a=tA >= 3, G2b=tY >= 2, G3a=G3a, G4a=G4a, G4b=G4b, G4c=G4c, G7=G7)
    j = np.logical_and.reduce(list(gates.values()))
    A = (tA >= 2) & (med(geo(xj, rb, y19)) > 0) & (spa_approx(x, Tb_main) < 0.10)
    Arel = A & G1 & (a3 >= 2) & (a10 >= 2)
    return {'joint_noDSR': float(j.mean()), **{k: float(v.mean()) for k, v in gates.items()}, 'A': float(A.mean()), 'A_G1_G7rel': float(Arel.mean())}

out = {'note': 'NS=%d per cell, MC se ~ sqrt(p(1-p)/NS)' % NS, 'drags_pp_m': {'stress': float(cS0.mean() * 100), 'c300': float(c30.mean() * 100), 'c1000': float(c100.mean() * 100),
       'd_j_sd_mean': float(D0.std(0).mean() * 100)}}
Z = np.zeros(n)
SC = {'S0_base': {}, 'S1_no_capacity_drag': dict(c3=Z, c10=Z), 'S2_no_stress_drag': dict(cS=Z), 'S3_no_carrier_drag': dict(cS=Z, c3=Z, c10=Z, D=np.zeros_like(D0)),
      'S4_strict_G7': dict(strictG7=True), 'S5_no_j_dispersion': dict(D=np.zeros_like(D0))}
for name, kw in SC.items():
    out[name] = {}
    for k, irv in enumerate([0.3, 0.5, 0.66, 0.8]):
        out[name][str(irv)] = run(irv, 777 + 31 * k, **kw)
    print(name, {i: (round(v['joint_noDSR'], 4), round(v['G7'], 3), round(v['G1'], 3)) for i, v in out[name].items()}, flush=True)
# 尺度換算：理想 IR 0.30 －P4 差 0.22 ≈ 可執行 0.08
P4 = R['FAMILY']['P4_ideal_vs_exec']['100萬']['ALL']['median_diff']
out['scale'] = {'P4_gap_100w': P4, 'family_max_post_ideal': 0.30, 'implied_exec': 0.30 - P4}
out['exec_scale_alt'] = {}
for k, irv in enumerate([0.05, 0.08, 0.10, 0.15]):
    r = run(irv, 999 + 17 * k); out['exec_scale_alt'][str(irv)] = {kk: r[kk] for kk in ('A', 'A_G1_G7rel', 'joint_noDSR', 'G1')}
print('exec_scale', out['exec_scale_alt'])
# 研究員對照
v1 = json.loads((H / 'data' / 'sim_V1.json').read_text())['by_IR']
out['researcher_V1_noDSR'] = {i: v1[i]['joint_without_G3b'] for i in ('0.3', '0.5', '0.66', '0.8')}
print('researcher', out['researcher_V1_noDSR'])
(Path('/tmp/claude-0/-home-user--/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad') / 'review_exec_o11.json').write_text(json.dumps(out, indent=1))
print(json.dumps(out, indent=1))
