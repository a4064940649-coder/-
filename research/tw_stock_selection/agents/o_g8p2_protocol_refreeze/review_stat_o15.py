#!/usr/bin/env python3
"""反方審查（資料與統計）o15：獨立驗證 G8P-2 V1 的幾個未涵蓋情境。只讀研究員檔案，不寫入。
用法：python3 -I -B review_stat_o15.py"""
import sys, json, math
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
P = json.loads((HERE / 'G8_PROTOCOL_v2.json').read_text(encoding='utf-8'))
OC = json.loads((HERE / 'oc_final.json').read_text(encoding='utf-8'))
SD = 6.141373712658076; C = 1.0596378779547428
pr = P['params']; B, CD = pr['b'], pr['c_d']
print('protocol params', {k: pr[k] for k in ('b', 'c_d', 'sigma_design', 'c')})
# A. 重算摘要
T = OC['tables']
v1 = {k: v['V1']['up'] for k, v in T['main'].items()}
g1 = {k: v['up'] for k, v in T['g8p1'].items()}
st = {k: v['V1']['up'] for k, v in T['strict'].items()}
print('A V1 main max', max(v1.values()), max(v1, key=v1.get), 'n>0.05', sum(x > .05 for x in v1.values()))
print('A G8P1 max', max(g1.values()), max(g1, key=g1.get), 'n>0.05', sum(x > .05 for x in g1.values()))
print('A strict max', max(st.values()), 'n>0.05', sum(x > .05 for x in st.values()))
print('A V1 main n within 2SE of .05 (P=1e5):', sum(abs(x - .05) < 2 * math.sqrt(x * (1 - x) / 1e5) for x in v1.values()), ' n>0.045:', sum(x > .045 for x in v1.values()))

def V1_up(X):
    n12, n18, n24 = 12, 18, 24
    S = np.cumsum(X, 1); S2 = np.cumsum(X * X, 1)
    def z(n):
        m = S[:, n - 1] / n; s = np.sqrt(np.maximum((S2[:, n - 1] - n * m * m) / (n - 1), 0))
        return S[:, n - 1] / (np.maximum(C * SD, s) * math.sqrt(n))
    z12, z18, z24 = z(12), z(18), z(24)
    lit = S[:, 11] > 0
    d12 = z12 <= -CD; d18 = z18 <= -CD
    u12 = (z12 / math.sqrt(2) >= B) & ~d12
    u24 = ~u12 & ~d12 & ~d18 & (z24 >= B)
    return float(np.mean(lit & (u12 | u24)))

rng = np.random.default_rng(20261105)
NP = 200_000
DW = 0.527
def ar1(mu, sig, rho, P_, H=24):
    e = np.empty((P_, H)); e[:, 0] = rng.standard_normal(P_)
    for t in range(1, H): e[:, t] = rho * e[:, t - 1] + math.sqrt(1 - rho * rho) * rng.standard_normal(P_)
    return mu + sig * e
def ma(mu, sig, theta, P_, H=24):  # MA(1)：重疊窗型態
    u = rng.standard_normal((P_, H + 1)); e = (u[:, 1:] + theta * u[:, :-1]) / math.sqrt(1 + theta * theta)
    return mu + sig * e
res = {}
print("B AR(1) / MA(1) 主口徑 s=0.78（μ=+0.253），V1：")
for rho in (0.0, 0.1, 0.2, 0.3):
    row = {}
    for sg in (6.14, 7.0, 8.4, 9.5):
        p = V1_up(ar1(0.78 - DW, sg, rho, NP)); row[sg] = p
    res[f'AR{rho}'] = row; print(' rho', rho, {k: round(v, 4) for k, v in row.items()})
for th in (0.3, 0.5):
    row = {sg: V1_up(ma(0.78 - DW, sg, th, NP)) for sg in (6.14, 8.4, 9.5)}
    res[f'MA{th}'] = row; print(' MA theta', th, {k: round(v, 4) for k, v in row.items()})
print('C σ 超出格（主口徑 s=0.78，D1）：')
for sg in (10.4, 11.5, 13.0, 15.0):
    p = V1_up(ar1(0.78 - DW, sg, 0.0, NP)); print(' σ', sg, round(p, 4))
print('D 漂移 μ 掃描（D1, σ 9.5）：哪個真實漂移開始超 5%')
for mu in (0.253, 0.35, 0.45, 0.6, 0.8, 1.0):
    print(' μ', mu, round(V1_up(ar1(mu, 9.5, 0.0, NP)), 4))

# E. 歷史較差成本 JMED 線的自相關（只描述；用研究員的 hist_windows，唯讀）
if '--hist' in sys.argv:
    sys.path.insert(0, str(HERE)); import g8lib2 as L; import pandas as pd
    Mn, Ms, F = L.hist_windows()
    w = L.jmed_line(Mn, Ms, worse=True)
    def acf(x, k): x = x - x.mean(); return float((x[k:] * x[:-k]).sum() / (x * x).sum())
    for nm, a, b in (('sel', '2006-04', '2019-04'), ('post', '2019-05', '2026-10'), ('all', '2006-04', '2026-10')):
        s = w[(w.index >= pd.Period(a, 'M')) & (w.index <= pd.Period(b, 'M'))].values
        n = len(s)
        ac = [acf(s, k) for k in range(1, 7)]
        # 24 個月和的變異比（非重疊 24 月塊 vs iid）
        vr = 1 + 2 * sum((1 - k / 24) * acf(s, k) for k in range(1, 12))
        print('E', nm, 'n', n, 'acf1-6', [round(x, 3) for x in ac], 'VR24(trunc11)', round(vr, 3), '1/sqrt(n) se', round(1 / math.sqrt(n), 3))
    # 逐 j 窗期間是否跨到下一錨定月（重疊）
    print('E window overlap check: J12 end vs next anchor J0 start')
    for j in (0, 6, 12):
        print('  j', j, F.j_signals(j, 'AB_V7_B10')[:2] if hasattr(F.j_signals(j, 'AB_V7_B10'), '__getitem__') else '')
if '--rho' in sys.argv:
    for rho in (0.057, 0.08):
        print('B2 rho', rho, {sg: round(V1_up(ar1(0.78 - DW, sg, rho, NP)), 4) for sg in (6.14, 8.4, 9.5)},
              's=0 (μ=-0.527):', {sg: round(V1_up(ar1(-DW, sg, rho, NP)), 4) for sg in (8.4, 9.5)})

# F. D3 改用挑選期（2006-04～2019-04）較差成本線殘差做區塊 bootstrap（研究員 D3 只用 2019 年起殘差）
if '--d3sel' in sys.argv:
    sys.path.insert(0, str(HERE)); import g8lib2 as L; import pandas as pd
    Mn, Ms, F = L.hist_windows()
    w = L.jmed_line(Mn, Ms, worse=True)
    for nm, a, b_ in (('sel', '2006-04', '2019-04'), ('all', '2006-04', '2026-09'), ('post', '2019-05', '2026-09')):
        e = w[(w.index >= pd.Period(a, 'M')) & (w.index <= pd.Period(b_, 'M'))].values.astype(float)
        e = e - e.mean(); e = e / e.std(ddof=1); n = len(e)
        out = {}
        for sg in (6.14, 8.4, 9.5):
            idx = np.empty((NP, 24), np.int64); idx[:, 0] = rng.integers(0, n, NP)
            for t in range(1, 24):
                jump = rng.random(NP) < 1 / 6
                idx[:, t] = np.where(jump, rng.integers(0, n, NP), (idx[:, t - 1] + 1) % n)
            out[sg] = round(V1_up(0.78 - DW + sg * e[idx]), 4)
        print('F D3-like', nm, 'n', n, out)
