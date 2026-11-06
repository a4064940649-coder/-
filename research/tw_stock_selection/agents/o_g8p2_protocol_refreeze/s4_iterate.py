#!/usr/bin/env python3
"""步驟 4：迭代（seed 集 A；只用 G8P-1 殘差形狀與設計 σ）：
 每版 V1～V4：c_d 校準（D1、σ_d、s＝0、α_net＝−1、目標 0.55、P＝400,000、seed 2026110602）→ b 依最差格校準（主口徑 135 格、P＝200,000/格、
 1−0.0475 分位數、下限 G8P-1 b）→ 集 A 初步 OC（135 格誤升級、σ_d 下修、σ×α 檢定力 P＝100,000）。G8P1 原樣同時在集 A 上對照。
輸出 iterations.json、results.json ITERATIONS。用法：python3 -I -B s4_iterate.py"""
import sys, json, math, time
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import numpy as np
import g8lib2 as L

L.check_reg('PREREG_DESIGN', files=False)
EX = L.load_res()['EXPLORE']
DW = EX['delta_w']
T0 = time.time()
def log(*a): print(f'[{time.time() - T0:6.0f}s]', *a, flush=True)

VERS = {'V1': dict(stat='MAXSD', delta=0.0, upgrade=True), 'V2': dict(stat='HAC_T', delta=0.0, upgrade=True),
        'V3': dict(stat='MAXSD', delta=L.DELTA_V3, upgrade=True), 'V4': dict(stat='LIT_ONLY', delta=0.0, upgrade=False)}
G8P1 = dict(stat='FIX_REFIT', upgrade=True)
ALPHAS = [-1.0, -0.5, 0.0, 0.5, 1.0, 2.0]
A_BASE, A_P = 2026110601, 200_000
TARGET = 0.0475

# ---------------- c_d
CD = {}
for v, spec in VERS.items():
    if v == 'V4': continue
    CD[v] = L.calib_cd(spec, 400_000, 2026110602, target=0.55, s=0.0, alpha=-1.0, dw=DW)  # V3 的 δ 由 zstats 內扣
CD['V4'] = CD['V1']
log('c_d', CD)

# ---------------- b：135 格 B*
qcell = {v: {} for v in VERS if v != 'V4'}
tails = {v: {} for v in VERS if v != 'V4'}
g8p1_a = {}
for dg in L.DGP_PRIMARY:
    for i, sg in enumerate(L.SIG_GRID):
        for k, s in enumerate(L.S_GRID):
            X = L.gen(dg, 0.0 + s - DW, sg, A_P, 24, L.cell_rng(A_BASE, dg, i, k, ALPHAS.index(0.0), 0))
            for v in qcell:
                Z = L.zstats(X, VERS[v])
                B = L.bstar(Z, CD[v])
                qcell[v][f'{dg}|{L.SIG_LABEL[i]}|{s}'] = float(np.quantile(B, 1 - TARGET))
                tails[v][f'{dg}|{L.SIG_LABEL[i]}|{s}'] = np.sort(B)[-int(0.12 * A_P):]
            # G8P1 原樣（一般成本線：μ＝s）
            X1 = L.gen(dg, 0.0 + s, sg, A_P, 24, L.cell_rng(A_BASE, dg, i, k, ALPHAS.index(0.0), 1))
            o1 = L.outcomes(L.zstats(X1, G8P1), L.B_G8P1, L.CD_G8P1)
            g8p1_a[f'{dg}|{L.SIG_LABEL[i]}|{s}'] = dict(up=float(o1['up'].mean()), up12=float(o1['up12'].mean()))
    log('b 格', dg)
BV = {}
for v in qcell:
    worst = max(qcell[v], key=qcell[v].get)
    BV[v] = dict(b=max(L.B_G8P1, qcell[v][worst]), b_raw=qcell[v][worst], worst_cell=worst, floor_binding=bool(qcell[v][worst] < L.B_G8P1),
                 top5=sorted(qcell[v].items(), key=lambda kv: -kv[1])[:5])
BV['V4'] = dict(b=None, note='無升級路徑')
log('b', {v: BV[v].get('b') for v in BV})

# ---------------- 集 A 初步 OC：135 格誤升級（用 tails）
def up_rate(tail, b): return float(np.sum(tail >= b) / A_P)
pre = {}
for v in qcell:
    fu = {c: up_rate(t, BV[v]['b']) for c, t in tails[v].items()}
    pre[v] = dict(false_up_max=max(fu.values()), false_up_argmax=max(fu, key=fu.get), n_cells_gt_05=int(sum(x > 0.05 for x in fu.values())),
                  false_up_by_dgp_max={dg: max(x for c, x in fu.items() if c.startswith(dg + '|')) for dg in L.DGP_PRIMARY},
                  false_up=fu)
pre['V4'] = dict(false_up_max=0.0)

# ---------------- 集 A 初步：下修（σ_d、s＝0）與檢定力（σ 6.51、s＝Δ_w）
def run_cell(dg, i, a, mu, P, mode=0, s_i=99):
    X = L.gen(dg, mu, L.SIG_GRID[i], P, 24, L.cell_rng(A_BASE, dg, i, s_i, ALPHAS.index(a), mode))
    r = {}
    for v, spec in VERS.items():
        o = L.outcomes(L.zstats(X, spec), BV[v]['b'] if spec['upgrade'] else 9.9, CD[v], upgrade=spec['upgrade'])
        r[v] = dict(up=float(o['up'].mean()), dn=float(o['dn_seq'].mean()), fail=float(o['fail'].mean()), lit=float(o['lit'].mean()))
    return r
down = {}; power = {}
for dg in L.DGP_PRIMARY:
    for a in (-1.0, 0.0):
        down[f'{dg}|6.14|a{a}|s0'] = run_cell(dg, 0, a, a + 0.0 - DW, 100_000, s_i=0)
    for a in (0.5, 1.0, 2.0):
        power[f'{dg}|6.51|a{a}|sDW'] = run_cell(dg, 1, a, a, 100_000, s_i=5)
log('下修／檢定力完成')
out = dict(utc=L.now_utc(), delta_w=DW, versions=VERS, c_d=CD, b=BV, target=TARGET, seed_A=A_BASE, P_A=A_P,
           prelim_false_up={v: {k: x for k, x in pre[v].items() if k != 'false_up'} for v in pre},
           prelim_false_up_cells={v: pre[v].get('false_up') for v in pre}, b_quantile_by_cell=qcell,
           prelim_down=down, prelim_power=power, g8p1_on_setA=g8p1_a,
           n_versions=4, note='集 A 只用於校準與初步檢視；正式 OC 用集 B／C（PREREG 之後）。版本數 4（V1～V4），G8P1 只是對照。')
(HERE / 'iterations.json').write_text(json.dumps(out, ensure_ascii=False, indent=1, default=L._js), encoding='utf-8')
L.save_key('ITERATIONS', out, need=('PREREG_DESIGN', 'EXPLORE'))
print(json.dumps(dict(c_d=CD, b=BV, prelim={v: {k: x for k, x in pre[v].items() if k != 'false_up'} for v in pre}, down=down, power=power),
                 ensure_ascii=False, indent=1, default=L._js))
