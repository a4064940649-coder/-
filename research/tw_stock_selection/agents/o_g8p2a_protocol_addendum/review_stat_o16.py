#!/usr/bin/env python3
"""反方審查（資料與統計，o16 G8P-2a）：只讀研究員檔案，不寫任何檔案。
用法：python3 -I -B review_stat_o16.py
1. 重算 A1（v2a 合併規則）在 OC 網格外、但與挑選期證據相容的相依 DGP 下的誤升級（主口徑 μ＝s−Δ_w）。
2. D5 區塊 bootstrap（平均區塊 6）對挑選期相依的保留程度：生成路徑的 acf／24 個月變異比 vs 原始序列。
3. 挑選期變異比的抽樣不確定性（區塊 bootstrap CI）。
4. 變異數跳動（前 12 月 σ 低、後 12 月 σ 高）的誤升級。
"""
import sys, json, math
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import numpy as np
import g8alib as L

BA = json.loads((HERE / 'G8_PROTOCOL_v2a.json').read_text(encoding='utf-8'))['params']['b_a']
P = 200_000
H = 24


def false_up(X):
    Z = L.all_stats(X)
    o = L.combined(Z, 'A1', BA)
    o2 = L.outcomes_rule(Z['lit'], Z['v212'], Z['v218'], Z['v224'], L.B_V2, L.CD_V2)
    p = float(o['up'].mean())
    return dict(A1=round(p, 5), se=round(math.sqrt(p * (1 - p) / X.shape[0]), 5), v2=round(float(o2['up'].mean()), 5))


def ma1(rng, th, P, H):
    u = rng.standard_normal((P, H + 1))
    return (u[:, 1:] + th * u[:, :-1]) / math.sqrt(1 + th * th)


def ar1(rng, rho, P, H, t5=True):
    u = L._tstd(rng, (P, H)) if t5 else rng.standard_normal((P, H))
    e = np.empty((P, H)); e[:, 0] = u[:, 0]; a = math.sqrt(1 - rho * rho)
    for t in range(1, H): e[:, t] = rho * e[:, t - 1] + a * u[:, t]
    return e


def mix_longmem(rng, w, phi, P, H):
    """√(1−w)·iid ＋ √w·AR(1)(phi)：acf_k＝w·phi^k（低權重持續成分，對應挑選期 acf6 > 0）。"""
    return math.sqrt(1 - w) * rng.standard_normal((P, H)) + math.sqrt(w) * ar1(rng, phi, P, H, t5=False)


def boot(e, P, H, rng, block):
    n = len(e); idx = np.empty((P, H), np.int64); idx[:, 0] = rng.integers(0, n, P)
    for t in range(1, H):
        j = rng.random(P) < 1.0 / block
        idx[:, t] = np.where(j, rng.integers(0, n, P), (idx[:, t - 1] + 1) % n)
    return e[idx]


def acf(x, k):
    x = x - x.mean(); return float((x[k:] * x[:-k]).sum() / (x * x).sum())


def vr_sum(X, n):
    """路徑前 n 期和的變異 ÷ (n·邊際變異)。"""
    S = X[:, :n].sum(axis=1); return float(S.var() / (n * X[:, :n].var()))


def lrv_ratio_mix(w, phi, n):
    s = 1.0
    for k in range(1, n): s += 2 * (1 - k / n) * w * phi ** k
    return s


out = {}
rng0 = np.random.default_rng(20261107)
worst = [(9.5, 0.78), (10.4, 0.78), (8.4, 0.78)]

# 1. 網格外相依
ext = {}
for lab, f in [('MA0.30_check', lambda r: ma1(r, 0.30, P, H)), ('MA0.35', lambda r: ma1(r, 0.35, P, H)), ('MA0.40', lambda r: ma1(r, 0.40, P, H)),
               ('MA0.50', lambda r: ma1(r, 0.50, P, H)), ('AR0.15', lambda r: ar1(r, 0.15, P, H)), ('AR0.20', lambda r: ar1(r, 0.20, P, H)),
               ('AR0.30', lambda r: ar1(r, 0.30, P, H)),
               ('MIX_w0.11_phi0.9', lambda r: mix_longmem(r, 0.11, 0.9, P, H)), ('MIX_w0.15_phi0.85', lambda r: mix_longmem(r, 0.15, 0.85, P, H)),
               ('MIX_w0.20_phi0.8', lambda r: mix_longmem(r, 0.20, 0.8, P, H))]:
    ext[lab] = {}
    for sg, s in worst:
        e = f(np.random.default_rng([20261107, sum(map(ord, lab)), int(sg * 10)]))
        ext[lab][f'{sg}|{s}'] = false_up(s - L.DW + sg * e)
    print(lab, ext[lab], flush=True)
out['ext_dependence'] = ext
out['mix_theory'] = {f'w{w}_phi{phi}': dict(acf1=w * phi, acf6=round(w * phi ** 6, 4), vr12=round(lrv_ratio_mix(w, phi, 12), 3), vr24=round(lrv_ratio_mix(w, phi, 24), 3))
                     for w, phi in ((0.11, 0.9), (0.15, 0.85), (0.20, 0.8))}

# 2. D5 bootstrap 相依保留
z = L.resid_sel()
orig = dict(acf1=acf(z, 1), acf6=acf(z, 6), acf12=acf(z, 12))
def vr_series(x, n):
    x = x - x.mean(); s = 1.0
    for k in range(1, n): s += 2 * (1 - k / n) * acf(x, k)
    return s
orig.update(vr12_bartlett=vr_series(z, 12), vr24_bartlett=vr_series(z, 24))
d5 = dict(orig={k: round(v, 4) for k, v in orig.items()})
for blk in (6, 12, 24):
    Xb = boot(z, 100_000, H, np.random.default_rng([7, blk]), blk)
    d5[f'block{blk}'] = dict(vr12=round(vr_sum(Xb, 12), 3), vr24=round(vr_sum(Xb, 24), 3))
    fu = {}
    for sg, s in worst:
        Xs = s - L.DW + sg * boot(z, P, H, np.random.default_rng([8, blk, int(sg * 10)]), blk)
        fu[f'{sg}|{s}'] = false_up(Xs)
    d5[f'block{blk}']['false_up'] = fu
    print('D5 block', blk, d5[f'block{blk}'], flush=True)
out['D5_block'] = d5

# 3. 挑選期 VR 的不確定性（以 block 12 bootstrap 重抽 157 個月，算 Bartlett VR24）
n = len(z); vrs = []
r = np.random.default_rng(11)
for _ in range(4000):
    idx = boot(z, 1, n, r, 12)[0]
    vrs.append(vr_series(idx, 24))
vrs = np.array(vrs)
out['sel_vr24_boot'] = dict(point=round(orig['vr24_bartlett'], 3), p05=round(float(np.quantile(vrs, .05)), 3), p50=round(float(np.quantile(vrs, .5)), 3),
                            p95=round(float(np.quantile(vrs, .95)), 3), note='VR 估計向下偏誤，CI 只作量級參考')
print('VR boot', out['sel_vr24_boot'], flush=True)

# 4. 變異跳動（iid 常態）
sh = {}
for lo, hi in ((6.14, 10.4), (10.4, 6.14), (6.14, 12.0), (8.0, 14.0)):
    rr = np.random.default_rng([99, int(lo * 10), int(hi * 10)])
    sg = np.r_[np.full(12, lo), np.full(12, hi)]
    e = ma1(rr, 0.3, P, H)
    for s in (0.78, 0.527):
        sh[f'MA3_{lo}->{hi}|s{s}'] = false_up(s - L.DW + sg * e)
    e = rr.standard_normal((P, H))
    sh[f'D1_{lo}->{hi}|s0.78'] = false_up(0.78 - L.DW + sg * e)
print('shift', sh, flush=True)
out['sigma_shift'] = sh

# 5. σ 網格外（A1 不擋升級）
og = {}
for sg in (11.5, 13.0, 15.0):
    og[f'MA3|{sg}|0.78'] = false_up(0.78 - L.DW + sg * ma1(np.random.default_rng([5, int(sg * 10)]), 0.3, P, H))
print('out-of-grid', og, flush=True)
out['out_of_grid_sigma'] = og
print(json.dumps(out, ensure_ascii=False, indent=1))

# 6. 持續成分校準到挑選期變異比（研究員 E1：VR24 1.222）
print('mix_theory', out['mix_theory'], flush=True)
cal = {}
for phi in (0.8, 0.9):
    for target in (1.222, 1.40):
        lo_, hi_ = 0.0, 0.5
        for _ in range(60):
            mid = (lo_ + hi_) / 2
            if lrv_ratio_mix(mid, phi, 24) < target: lo_ = mid
            else: hi_ = mid
        w = (lo_ + hi_) / 2
        key = f'phi{phi}_VR24_{target}'
        cal[key] = dict(w=round(w, 4), acf1=round(w * phi, 4), acf6=round(w * phi ** 6, 4), vr12=round(lrv_ratio_mix(w, phi, 12), 3))
        for sg, s in worst[:2]:
            cal[key][f'{sg}|{s}'] = false_up(s - L.DW + sg * mix_longmem(np.random.default_rng([31, int(phi * 10), int(target * 1000), int(sg * 10)]), w, phi, P, H))
        print(key, cal[key], flush=True)
out['mix_calibrated'] = cal

# 7. K14：V3 選擇性作廢對字面 G8（S_12>0）的膨脹（iid 常態、μ＝0；前 12 個有效月內最多 2 個作廢，貪婪丟掉前 2 個負月）
r7 = np.random.default_rng(77); Pq = 400_000
X = r7.standard_normal((Pq, 14))
neg = X < 0
keep = np.ones_like(X, bool); used = np.zeros(Pq, int)
valid_cnt = np.zeros(Pq, int); S12 = np.zeros(Pq)
done = np.zeros(Pq, bool)
for t in range(14):
    drop = neg[:, t] & (used < 2) & ~done
    used += drop
    take = ~drop & ~done
    S12 += np.where(take, X[:, t], 0.0); valid_cnt += take
    done |= valid_cnt >= 12
ok = valid_cnt >= 12
out['K14_literal'] = dict(honest=round(float((X[:, :12].sum(1) > 0).mean()), 4), greedy_drop2=round(float((S12[ok] > 0).mean()), 4))
print('K14', out['K14_literal'], flush=True)
