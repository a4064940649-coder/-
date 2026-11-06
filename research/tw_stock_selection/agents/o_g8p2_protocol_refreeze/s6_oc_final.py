#!/usr/bin/env python3
"""步驟 6：正式 OC（集 B，P＝100,000/格；接近 0.05 的格用集 C，P＝400,000）＋ 依 PREREG 選版。
斷言 PREREG 存在且未被改。輸出 oc_final.json（摘要表）與 results.json OC_FINAL。用法：python3 -I -B s6_oc_final.py"""
import sys, json, math, time
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import numpy as np
import g8lib2 as L

PR = L.check_reg('PREREG')
VERS = PR['versions']; DW = PR['delta_w']; AL = PR['alphas']
OC = PR['oc_final']; BB, PB, CB, PC = OC['seed_B'], OC['P_B'], OC['seed_C'], OC['P_C']
G8P1 = dict(stat='FIX_REFIT', upgrade=True, c_d=L.CD_G8P1, b=L.B_G8P1, delta=0.0)
T0 = time.time()
def log(*a): print(f'[{time.time() - T0:6.0f}s]', *a, flush=True)


def evalX(X, vers):
    r = {}
    for v, sp in vers.items():
        o = L.outcomes(L.zstats(X, sp), sp['b'] if sp['upgrade'] else 9.9, sp['c_d'], upgrade=sp['upgrade'])
        r[v] = {k: float(np.mean(x)) for k, x in o.items()}
    return r


def cell(base, P, dg, i, s_i, a, mu, mode, vers):
    X = L.gen(dg, mu, L.SIG_GRID[i], P, 24, L.cell_rng(base, dg, i, s_i, AL.index(a), mode))
    return evalX(X, vers)


T = dict(main={}, strict={}, g8p1={}, down={}, power={}, sens={}, sens_g8p1={})
# 主口徑誤升級（α＝0）＋ 嚴格口徑＋G8P1
for dg in L.DGP_PRIMARY:
    for i, lab in enumerate(L.SIG_LABEL):
        for k, s in enumerate(L.S_GRID):
            T['main'][f'{dg}|{lab}|{s}'] = cell(BB, PB, dg, i, k, 0.0, s - DW, 0, VERS)
            r = cell(BB, PB, dg, i, k, 0.0, s, 1, dict(VERS, G8P1=G8P1))
            T['g8p1'][f'{dg}|{lab}|{s}'] = r.pop('G8P1'); T['strict'][f'{dg}|{lab}|{s}'] = r
    log('main/strict', dg)
# 下修與檢定力
for dg in L.DGP_PRIMARY:
    for i, lab in enumerate(L.SIG_LABEL):
        for a in (-1.0, -0.5, 0.0, 0.5, 1.0, 2.0):
            for s_i, sval, nm in ((0, 0.0, 's0'), (5, DW, 'sDW')):
                key = f'{dg}|{lab}|a{a}|{nm}'
                r = cell(BB, PB, dg, i, s_i, a, a + sval - DW, 0, VERS)
                (T['down'] if a <= 0 else T['power'])[key] = r
    log('down/power', dg)
# 敏感度 DGP
for dg in ('D3x', 'D4'):
    for i, lab in enumerate(L.SIG_LABEL):
        for k, s in enumerate(L.S_GRID):
            T['sens'][f'{dg}|{lab}|{s}'] = cell(BB, PB, dg, i, k, 0.0, s - DW, 0, VERS)
            T['sens_g8p1'][f'{dg}|{lab}|{s}'] = cell(BB, PB, dg, i, k, 0.0, s, 1, dict(G8P1=G8P1))['G8P1']
        T['sens'][f'{dg}|{lab}|a1.0|sDW'] = cell(BB, PB, dg, i, 5, 1.0, 1.0, 0, VERS)
    log('sens', dg)

# 集 C 重跑
SE = lambda p, P: math.sqrt(max(p * (1 - p), 1e-12) / P)
rerun = {}
for v in ('V1', 'V2', 'V3'):
    for c, r in T['main'].items():
        p = r[v]['up']
        if abs(p - 0.05) < 2 * SE(p, PB):
            dg, lab, s = c.split('|'); i = L.SIG_LABEL.index(lab); k = L.S_GRID.index(float(s))
            rc = cell(CB, PC, dg, i, k, 0.0, float(s) - DW, 0, {v: VERS[v]})[v]
            rerun[f'{v}|{c}'] = dict(B=p, C=rc['up'], se_C=SE(rc['up'], PC))
log('rerun C', len(rerun))

# 合格判定
def final_up(v, c):
    return rerun[f'{v}|{c}']['C'] if f'{v}|{c}' in rerun else T['main'][c][v]['up']
Q = {}
for v in VERS:
    fu = {c: final_up(v, c) for c in T['main']}
    worst = max(fu, key=fu.get)
    dn_ok = {dg: dict(dn_m1=T['down'][f'{dg}|6.14|a-1.0|s0'][v]['dn_seq'], fdn_0=T['down'][f'{dg}|6.14|a0.0|s0'][v]['dn_seq']) for dg in L.DGP_PRIMARY}
    pass_up = all(x <= 0.05 for x in fu.values())
    pass_dn = all(d['dn_m1'] >= 0.50 and d['fdn_0'] <= 0.30 for d in dn_ok.values())
    pw = {dg: T['power'][f'{dg}|6.51|a1.0|sDW'][v]['up'] for dg in L.DGP_PRIMARY}
    Q[v] = dict(pass_false_up=pass_up, pass_down=pass_dn, qualified=bool(pass_up and pass_dn), worst_cell=worst, worst_false_up=fu[worst],
                n_cells_gt_05=int(sum(x > 0.05 for x in fu.values())), max_by_dgp={dg: max(x for c, x in fu.items() if c.startswith(dg + '|')) for dg in L.DGP_PRIMARY},
                down=dn_ok, power_a1_s651_sDW=pw, power_sel=min(pw.values()), power_sel_dgp=min(pw, key=pw.get))
# 選版
cand = [v for v in ('V1', 'V2', 'V3') if Q[v]['qualified']]
if cand:
    best = max(cand, key=lambda v: Q[v]['power_sel'])
    tie = [v for v in cand if Q[best]['power_sel'] - Q[v]['power_sel'] <= 2 * math.sqrt(2) * SE(Q[best]['power_sel'], PB)]
    sel = min(tie, key=PR['simplicity_order'].index)
    oc_all_pass = True
else:
    sel = 'V4' if Q['V4']['qualified'] else min(VERS, key=lambda v: Q[v]['worst_false_up']); oc_all_pass = False; tie = []
log('選版', sel, Q[sel])

out = dict(utc=L.now_utc(), prereg_sha256=L.load_res()['PREREG_sha256'], delta_w=DW, qualification=Q, selected=sel, tie_set=tie,
           oc_all_cells_pass=oc_all_pass, rerun_C=rerun, tables=T)
(HERE / 'oc_final.json').write_text(json.dumps(out, ensure_ascii=False, indent=0, default=L._js), encoding='utf-8')
L.save_key('OC_FINAL', dict(utc=out['utc'], qualification=Q, selected=sel, tie_set=tie, oc_all_cells_pass=oc_all_pass, rerun_C=rerun,
                            oc_final_json_sha256=L.sha(HERE / 'oc_final.json')), need=('PREREG',))
print(json.dumps(dict(selected=sel, tie=tie, Q=Q, rerun=rerun), ensure_ascii=False, indent=1, default=L._js))
