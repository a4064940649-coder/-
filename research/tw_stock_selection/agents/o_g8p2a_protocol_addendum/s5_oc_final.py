#!/usr/bin/env python3
"""步驟 5：正式 OC（PREREG 之後）：集 B（P＝100,000）main／strict／down／power／sens；|p−0.05|<2SE 的主口徑 (版本, 格) 用集 C（P＝400,000）重跑。
每格同一路徑算 v2（G8P-2 V1 原樣）、A1～A4 與診斷 A1b0；逐路徑子集檢查 viol_up／viol_fail。合格判定、選版 → oc_final.json、results.json OC_FINAL。
用法：python3 -I -B s5_oc_final.py"""
import sys, json, math, time
from pathlib import Path
from multiprocessing import Pool
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import numpy as np
import g8alib as L

PR = L.check_reg('PREREG')
BA = PR['params']['b_a']
SEED_B, P_B, SEED_C, P_C = 2026110702, 100_000, 2026110703, 400_000
VERS = ['A1', 'A2', 'A3', 'A4', 'A1b0']
SENS = ['ARN057', 'ARN08', 'ARN10', 'MAT3']


def run_cell(args):
    fam, dgp, si, ki, ai, mode, seed, P = args
    s = L.DW if ki == 5 else L.S_GRID[ki]
    mu = L.ALPHAS[ai] + s - (L.DW if mode == 0 else 0.0)
    X = L.gen(dgp, mu, L.SIG_GRID[si], P, 24, L.cell_rng(seed, dgp, si, ki, ai, mode))
    Z = L.all_stats(X)
    sl = 'DW' if ki == 5 else str(L.S_GRID[ki])
    key = f'{dgp}|{L.SIG_LABEL[si]}|{sl}' + ('' if fam in ('main', 'strict', 'sens') else f'|a{L.ALPHAS[ai]}')
    out = dict(fam=fam, key=key, P=P)
    o2 = L.outcomes_rule(Z['lit'], Z['v212'], Z['v218'], Z['v224'], L.B_V2, L.CD_V2)
    out['v2'] = dict(up=float(o2['up'].mean()), fail=float(o2['fail'].mean()), dn=float(o2['dn_seq'].mean()))
    for v in VERS:
        if v == 'A1b0': o = L.combined(Z, 'A1', L.B_V2)
        elif v == 'A4': o = L.combined(Z, 'A4', 99.0)
        else: o = L.combined(Z, v, BA[v])
        out[v] = dict(up=float(o['up'].mean()), up12=float(o['up12'].mean()), fail=float(o['fail'].mean()), dn=float(o['dn_seq'].mean()),
                      lit=float(o['lit'].mean()), viol_up=o['viol_up'], viol_fail=o['viol_fail'])
    return out


def jobs_B():
    J = []
    for d in L.DGPS:
        for si in range(9):
            for ki in range(5):
                J.append(('main', d, si, ki, 2, 0, SEED_B, P_B)); J.append(('strict', d, si, ki, 2, 1, SEED_B, P_B))
            for ai in (0, 1, 2):
                for ki in (0, 5): J.append(('down', d, si, ki, ai, 0, SEED_B, P_B))
            for ai in (3, 4, 5):
                for ki in (5, 0): J.append(('power', d, si, ki, ai, 0, SEED_B, P_B))
    for d in SENS:
        for si in range(9):
            for ki in range(5): J.append(('sens', d, si, ki, 2, 0, SEED_B, P_B))
    return J


if __name__ == '__main__':
    t0 = time.time()
    with Pool(4) as pool:
        rowsB = pool.map(run_cell, jobs_B(), chunksize=6)
    T = {}
    for r in rowsB: T.setdefault(r['fam'], {})[r['key']] = r
    # 集 C
    need = []
    for k, r in T['main'].items():
        for v in ('A1', 'A2', 'A3'):
            p = r[v]['up']; se = math.sqrt(max(p * (1 - p), 1e-12) / P_B)
            if abs(p - 0.05) < 2 * se: need.append((k, v))
    cells_C = sorted({k for k, _ in need})
    rowsC = {}
    if cells_C:
        jc = []
        for k in cells_C:
            d, sl, s = k.split('|'); jc.append(('main', d, L.SIG_LABEL.index(sl), L.S_GRID.index(float(s)), 2, 0, SEED_C, P_C))
        with Pool(4) as pool:
            for r in pool.map(run_cell, jc, chunksize=1): rowsC[r['key']] = r
    final_main = {}
    for k, r in T['main'].items():
        final_main[k] = {v: (rowsC[k][v] if (k, v) in need else r[v]) for v in VERS}
        final_main[k]['v2'] = r['v2']
    # 合格
    qual = {}
    for v in ('A1', 'A2', 'A3', 'A4'):
        ups = {k: final_main[k][v]['up'] for k in final_main}
        viol = sum(r[v]['viol_up'] + r[v]['viol_fail'] for fam in T.values() for r in fam.values()) + \
               sum(r[v]['viol_up'] + r[v]['viol_fail'] for r in rowsC.values())
        worst = max(ups, key=ups.get)
        by_dgp = {d: round(max(u for k, u in ups.items() if k.startswith(d + '|')), 5) for d in L.DGPS}
        by_sig = {sl: round(max(u for k, u in ups.items() if k.split('|')[1] == sl), 5) for sl in L.SIG_LABEL}
        pw = {d: T['power'][f'{d}|6.51|DW|a1.0'][v]['up'] for d in L.DGPS}
        qual[v] = dict(max_false_up=round(ups[worst], 5), worst_cell=worst, n_gt_05=int(sum(u > 0.05 for u in ups.values())),
                       viol_total=int(viol), qualified=bool(max(ups.values()) <= 0.05 and viol == 0), max_by_dgp=by_dgp, max_by_sigma=by_sig,
                       power_a1_s651_sDW=pw, power_min=min(pw.values()), power_min_dgp=min(pw, key=pw.get))
    # 選版
    order = ['A1', 'A2', 'A3']
    cand = [v for v in order if qual[v]['qualified']]
    sel = None; trail = []
    if cand:
        best = max(cand, key=lambda v: qual[v]['power_min'])
        sel = best
        for v in order:
            if v not in cand or order.index(v) >= order.index(best): continue
            pa, pb = qual[v]['power_min'], qual[best]['power_min']
            se = max(math.sqrt(pa * (1 - pa) / P_B), math.sqrt(pb * (1 - pb) / P_B))
            trail.append(dict(simpler=v, best=best, gap=pb - pa, thr=2 * math.sqrt(2) * se))
            if pb - pa < 2 * math.sqrt(2) * se: sel = v; break
    else:
        sel = 'A4'
    # 摘要
    def fam_max(fam, v, field='up', filt=lambda k: True):
        vals = {k: r[v][field] for k, r in T[fam].items() if filt(k)}
        k = max(vals, key=vals.get); return round(vals[k], 5), k
    S = dict(selected=sel, selection_trail=trail, qualification=qual, P_B=P_B, P_C=P_C, seed_B=SEED_B, seed_C=SEED_C, b_a=BA,
             n_cells=dict({f: len(t) for f, t in T.items()}, C=len(rowsC)), cells_rerun_C=need, runtime_s=round(time.time() - t0, 1))
    v2m = {k: final_main[k]['v2']['up'] for k in final_main}
    S['v2_main'] = dict(max=round(max(v2m.values()), 5), worst=max(v2m, key=v2m.get), n_gt_05=int(sum(u > 0.05 for u in v2m.values())),
                        max_by_dgp={d: round(max(u for k, u in v2m.items() if k.startswith(d + '|')), 5) for d in L.DGPS},
                        primary135_max=round(max(u for k, u in v2m.items() if k.split('|')[0] in ('D1', 'D2', 'D3')), 5))
    a0 = {k: final_main[k]['A1b0']['up'] for k in final_main}
    S['A1b0_main'] = dict(primary135_max=round(max(u for k, u in a0.items() if k.split('|')[0] in ('D1', 'D2', 'D3')), 5),
                          max=round(max(a0.values()), 5), worst=max(a0, key=a0.get), n_gt_05=int(sum(u > 0.05 for u in a0.values())),
                          max_by_dgp={d: round(max(u for k, u in a0.items() if k.startswith(d + '|')), 5) for d in L.DGPS},
                          AR10_s078_hi={sl: round(a0[f'AR10|{sl}|0.78'], 5) for sl in ('8.4', '9.0', '9.5', '10.4')})
    for v in ('A1', 'A2', 'A3', 'A4'):
        S.setdefault('strict', {})[v] = dict(zip(('max', 'cell'), fam_max('strict', v))) | dict(n_gt_05=int(sum(r[v]['up'] > 0.05 for r in T['strict'].values())))
        S.setdefault('sens', {})[v] = {d: fam_max('sens', v, filt=lambda k, d=d: k.startswith(d + '|'))[0] for d in SENS}
        dn = T['down']
        S.setdefault('down', {})[v] = dict(
            dn_m1_s0_sigd={d: round(dn[f'{d}|6.14|0.0|a-1.0'][v]['dn'], 4) for d in L.DGPS},
            fdn_0_s0_sigd={d: round(dn[f'{d}|6.14|0.0|a0.0'][v]['dn'], 4) for d in L.DGPS},
            dn_m1_sDW_range=[round(min(r[v]['dn'] for k, r in dn.items() if k.endswith('|DW|a-1.0')), 4), round(max(r[v]['dn'] for k, r in dn.items() if k.endswith('|DW|a-1.0')), 4)],
            fdn_0_max=dict(zip(('max', 'cell'), fam_max('down', v, 'dn', lambda k: k.endswith('|a0.0')))),
            n_fdn0_gt_030=int(sum(r[v]['dn'] > 0.30 for k, r in dn.items() if k.endswith('|a0.0'))),
            n_fdn0_gt_030_D5=int(sum(r[v]['dn'] > 0.30 for k, r in dn.items() if k.endswith('|a0.0') and k.startswith('D5|'))))
        pw = T['power']
        S.setdefault('power', {})[v] = {f'a{a}|{sl}': [round(min(r[v]['up'] for k, r in pw.items() if k.endswith(f'|{sl}|a{a}')), 4),
                                                       round(max(r[v]['up'] for k, r in pw.items() if k.endswith(f'|{sl}|a{a}')), 4)]
                                        for a in (0.5, 1.0, 2.0) for sl in ('DW', '0.0')}
    S['power_v2_same_paths'] = {f'a{a}|{sl}': [round(min(r['v2']['up'] for k, r in T['power'].items() if k.endswith(f'|{sl}|a{a}')), 4),
                                               round(max(r['v2']['up'] for k, r in T['power'].items() if k.endswith(f'|{sl}|a{a}')), 4)]
                                for a in (0.5, 1.0, 2.0) for sl in ('DW', '0.0')}
    S['power_D1_651_DW_a1'] = dict(v2=T['power']['D1|6.51|DW|a1.0']['v2']['up'], **{v: T['power']['D1|6.51|DW|a1.0'][v]['up'] for v in ('A1', 'A2', 'A3')})
    S['power_gt_v2_any'] = {v: int(sum(r[v]['up'] > r['v2']['up'] + 1e-12 for r in T['power'].values())) for v in ('A1', 'A2', 'A3', 'A4')}
    S['false_up_main_le_v2_all'] = {v: bool(all(final_main[k][v]['up'] <= final_main[k]['v2']['up'] + 1e-12 for k in final_main if (k, v) not in need))
                                    for v in ('A1', 'A2', 'A3', 'A4')}
    sv = sel
    fm = {k: final_main[k][sv]['up'] for k in final_main}
    S['selected_detail'] = dict(
        max_by_sigma={sl: round(max(u for k, u in fm.items() if k.split('|')[1] == sl), 5) for sl in L.SIG_LABEL},
        max_by_dgp={d: round(max(u for k, u in fm.items() if k.startswith(d + '|')), 5) for d in L.DGPS},
        s0_range=[round(min(u for k, u in fm.items() if k.endswith('|0.0')), 5), round(max(u for k, u in fm.items() if k.endswith('|0.0')), 5)],
        sig614_s0={d: round(fm[f'{d}|6.14|0.0'], 5) for d in L.DGPS},
        top8=sorted(((round(u, 5), k) for k, u in fm.items()), reverse=True)[:8],
        n_gt_045=int(sum(u > 0.045 for u in fm.values())))
    out = dict(summary=S, main=final_main,
               strict={k: {v: round(r[v]['up'], 5) for v in ('A1', 'A2', 'A3')} | dict(v2=round(r['v2']['up'], 5)) for k, r in T['strict'].items()},
               down={k: {v: [round(r[v]['dn'], 4), round(r[v]['fail'], 4)] for v in ('A1', 'A2', 'A3', 'A4')} | dict(v2=[round(r['v2']['dn'], 4), round(r['v2']['fail'], 4)]) for k, r in T['down'].items()},
               power={k: {v: round(r[v]['up'], 5) for v in ('A1', 'A2', 'A3')} | dict(v2=round(r['v2']['up'], 5)) for k, r in T['power'].items()},
               sens={k: {v: round(r[v]['up'], 5) for v in ('A1', 'A2', 'A3')} | dict(v2=round(r['v2']['up'], 5)) for k, r in T['sens'].items()},
               setC={k: {v: r[v]['up'] for v in VERS} for k, r in rowsC.items()})
    for k in out['main']:
        out['main'][k] = {v: dict(up=round(x['up'], 5), up12=round(x.get('up12', 0), 5), viol=x.get('viol_up', 0) + x.get('viol_fail', 0)) if v != 'v2' else dict(up=round(x['up'], 5))
                          for v, x in out['main'][k].items()}
    (HERE / 'oc_final.json').write_text(json.dumps(out, ensure_ascii=False, separators=(',', ':')), encoding='utf-8')
    L.save_key('OC_FINAL', dict(file='oc_final.json', sha256=L.sha(HERE / 'oc_final.json'), summary=S, utc=L.now_utc()), need=('PREREG',))
    print(json.dumps(S, ensure_ascii=False, indent=1)[:12000])
