#!/usr/bin/env python3
"""研究員自我複製（不是 G6 獨立複製）：只讀 G8_PROTOCOL_v2.json，不 import 研究程式；新 seed（990000 起）。
重算 c_d、b（主口徑 135 格 q_0.9525(B*) 的最大值，下限 1.6787）與數格 OC，和協定比較（2 倍 MC 標準誤）。
用法：python3 -I -B replicate_self/rep_from_protocol.py"""
import json, math, hashlib
from pathlib import Path
import numpy as np
from scipy import stats as st
HERE = Path(__file__).resolve().parent.parent
pp = HERE / 'G8_PROTOCOL_v2.json'
P = json.loads(pp.read_text(encoding='utf-8'))
pr = P['params']; sd, c, dw = pr['sigma_design'], pr['c'], pr['delta_w']
e = np.array(P['data']['resid_post2019'], float); e = e - e.mean(); e = e / e.std(ddof=1)
grid, lab, sg = P['oc']['sigma_grid'], P['oc']['sigma_label'], P['oc']['s_grid']

def draw(dg, mu, s, n, rng):
    if dg == 'D1': return mu + s * rng.normal(size=(n, 24))
    if dg == 'D2': return mu + s * rng.standard_t(4, size=(n, 24)) / math.sqrt(2)
    pos = rng.integers(0, len(e), n); out = np.empty((n, 24)); out[:, 0] = e[pos]
    for t in range(1, 24):
        nw = rng.integers(0, len(e), n); pos = np.where(rng.random(n) < 1 / 6, nw, (pos + 1) % len(e)); out[:, t] = e[pos]
    return mu + s * out

def z(X, n):
    y = X[:, :n]; s_n = y.std(axis=1, ddof=1)
    return y.sum(axis=1) / (np.maximum(c * sd, s_n) * math.sqrt(n))

rng = np.random.default_rng(990001)
X = -1 - dw + sd * rng.normal(size=(400_000, 24))
zm = np.minimum(np.minimum(z(X, 12), z(X, 18)), z(X, 24))
cd = float(-np.quantile(zm, 0.55))
se_cd = float(np.sqrt(0.55 * 0.45 / 400_000) / st.gaussian_kde(zm[:20000])(np.quantile(zm, 0.55))[0])
def bstar(X, cd):
    lit = X[:, :12].sum(1) > 0; z12, z18, z24 = z(X, 12), z(X, 18), z(X, 24)
    B = np.maximum(z12 / math.sqrt(2), np.where(z18 <= -cd, -1e9, z24))
    return np.where(lit & (z12 > -cd), B, -1e9)
qs = {}; fu = {}
k = 0
for dg in ('D1', 'D2', 'D3'):
    for i, s in enumerate(grid):
        for sv in sg:
            k += 1
            Xc = draw(dg, sv - dw, s, 100_000, np.random.default_rng(990100 + k))
            B = bstar(Xc, pr['c_d'])
            qs[f'{dg}|{lab[i]}|{sv}'] = float(np.quantile(B, 0.9525)); fu[f'{dg}|{lab[i]}|{sv}'] = float(np.mean(B >= pr['b']))
b_rep = max(1.6787025976148637, max(qs.values()))
zc = {c_: (fu[c_] - P['oc']['false_up_main'][c_]) / math.sqrt(2 * max(fu[c_] * (1 - fu[c_]), 1e-9) / 100_000) for c_ in fu}
out = dict(protocol_sha256=hashlib.sha256(pp.read_bytes()).hexdigest(), c_d_rep=cd, c_d_protocol=pr['c_d'], c_d_se_approx=se_cd,
           b_rep=b_rep, b_protocol=pr['b'], b_rep_worst=max(qs, key=qs.get), false_up_rep_max=max(fu.values()), false_up_rep_argmax=max(fu, key=fu.get),
           n_cells_rep_gt_05=int(sum(v > 0.05 for v in fu.values())), max_abs_z_vs_protocol=float(max(abs(v) for v in zc.values())),
           n_abs_z_gt_2=int(sum(abs(v) > 2 for v in zc.values())), n_cells=len(fu))
print(json.dumps(out, ensure_ascii=False, indent=1))
(HERE / 'replicate_self' / 'rep_out.json').write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding='utf-8')
