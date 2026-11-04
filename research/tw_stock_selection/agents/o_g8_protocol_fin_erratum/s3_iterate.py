#!/usr/bin/env python3
"""步驟 3（迭代，最多 4 版）：依 PREREG_DESIGN 計算各版界限，seed 集 A、每格 P＝20,000，主 DGP（D1～D3）OC；依事前選版規則選版。
輸出 iterations.json、results.json 的 ITERATIONS。用法：python3 -I -B s3_iterate.py"""
import sys, json, math, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import g8lib as G

D = G.check_reg('PREREG_DESIGN')
C = D['inflation_c']['value']
RES = np.load(G.SCR / 'resid_post2019.npy')
TH, W = np.array(D['mixture_prior']['theta']), np.array(D['mixture_prior']['w'])
DGP_ID = {'D1': 1, 'D2': 2, 'D3': 3, 'D3_2024': 4, 'D3_L3': 5, 'D3_L12': 6, 'D4_o8': 7}


def params():
    cd = G.dn_cd(C, 0.55, P=400_000, seed=20261106)
    b2 = G.obf_b([12, 24], P=2_000_000, seed=20261105)
    b4 = G.obf_b([6, 12, 18, 24], P=2_000_000, seed=20261105)
    base = dict(sigma=G.SIGMA_D, c=C, c_d=cd)
    return {'v1': ('R-FIX', dict(base, up_looks=[12, 24], b=b2)), 'v2': ('R-OBF', dict(base, up_looks=[6, 12, 18, 24], b=b4)),
            'v3': ('R-E', dict(base, theta=TH, w=W, e_thr=20.0)), 'v4': ('R-E+', dict(base, theta=TH, w=W, e_thr=20.0))}


def run_grid(vers, dgps, alphas, P, seed_base, H=24, resid_map=None, block_map=None, shift=0.0):
    out = {v: {} for v in vers}
    for dgp in dgps:
        for a in alphas:
            rng = np.random.default_rng([seed_base, int(round((a + 3) * 100)), DGP_ID.get(dgp, 9)])
            res = (resid_map or {}).get(dgp, RES); blk = (block_map or {}).get(dgp, 6)
            g = 'D3' if dgp.startswith('D3') else dgp
            X = G.gen(g, a + shift, P, H, rng, resid=res, block=blk)
            for v, (rule, par) in vers.items():
                u, d = G.run_rule(rule, X, par)
                out[v][f'{dgp}|{a:+.1f}'] = G.oc(u, d, X, horizons=tuple(h for h in (12, 18, 24, 30, 36, 42, 48, 60) if h <= H))
    return out


def check(o, crit, dgps=('D1', 'D2', 'D3')):
    det, ok = {}, True
    for g in dgps:
        fu = o[f'{g}|+0.0'][24]['up']; pw = o[f'{g}|-1.0'][24]['down']; fd = o[f'{g}|+0.0'][24]['down']; p1 = o[f'{g}|+1.0'][24]['up']
        det[g] = dict(false_up=fu, down_power=pw, false_down=fd, up_power_p1=p1, up_power_p2=o[f'{g}|+2.0'][24]['up'])
        ok &= (fu <= crit['false_up_24_max']) and (pw >= crit['down_power_24_alpha_m1_min']) and (fd <= crit['false_down_24_max'])
    return bool(ok), det


def main():
    t0 = time.time()
    V = params()
    S = D['sims']['iteration']
    grid = run_grid(V, D['pass_criteria']['primary_dgps'], D['alphas'], S['P'], S['seed_base'])
    it = {'smoke_v0': dict(note='登記前程式煙霧測試（P＝20,000、seed 1，b/c_d 用 P 較小的近似值），只檢查程式能跑，沒有用於選版；結果：R-FIX／R-OBF 誤升級 0.02～0.04、R-E／R-E+ 升級機率 0、DN 誤下修 0.25～0.28。設計未因此修改。'),
          'params': {v: dict(rule=r, **{k: (vv if not isinstance(vv, np.ndarray) else f'array[{len(vv)}]') for k, vv in p.items()}) for v, (r, p) in V.items()},
          'versions': {}}
    P = S['P']
    for v, (rule, par) in V.items():
        ok, det = check(grid[v], D['pass_criteria'])
        it['versions'][v] = dict(rule=rule, pass_=ok, by_dgp=det, worst_p1=min(x['up_power_p1'] for x in det.values()), oc=grid[v])
        print(v, rule, ok, {g: {k: round(x, 4) for k, x in dd.items()} for g, dd in det.items()}, flush=True)
    passing = [v for v in ('v1', 'v2', 'v3', 'v4') if it['versions'][v]['pass_']]
    sel = None
    if passing:
        best = max(it['versions'][v]['worst_p1'] for v in passing)
        se2 = 2 * math.sqrt(2) * G.mc_se(best, P)
        tied = [v for v in passing if best - it['versions'][v]['worst_p1'] <= se2]
        sel = tied[0]                                    # 已依簡單程度排序
    it['passing'] = passing; it['selected'] = sel
    it['selection_note'] = '合格者中 min_DGP P(升級｜α＝+1) 最高；差距 ≤ 2×√2×MC 標準誤視為同分取較簡單'
    it['n_versions'] = 4; it['utc'] = G.now_utc(); it['runtime_s'] = time.time() - t0
    (G.HERE / 'iterations.json').write_text(json.dumps(it, ensure_ascii=False, indent=1, default=G._js), encoding='utf-8')
    G.save_key('ITERATIONS', it, need=['PREREG_DESIGN'])
    print('passing', passing, 'selected', sel, 'runtime', round(time.time() - t0))


if __name__ == '__main__':
    main()
