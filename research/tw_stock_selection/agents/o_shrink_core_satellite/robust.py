#!/usr/bin/env python3
"""正式計算 4：穩健性（PREREG 之後）。判定欄 EXEC_J 與 AB、B10、EXEC 的 G2×MAIN 事後分布與 x*（10 年、L6、10pp 格點、5,000 路徑）：
拿掉每一年、起點 2008、拿掉焦點最好 10 窗、HAC L 3／12、G2 的 c0 0.05／0.2／1（deconvolveR 預設）與 df 4／7。
用法：python3 -I -B robust.py [--focus ...] [--tag V4]"""
import sys, json, argparse, math
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import sc_lib as S

KOU = ['AB', 'B10', 'EXEC', 'EXEC_J']

def xstar(F, c, dens, B=5000, H=120):
    r50 = F['r0050'].values; e = F[c].values - r50; resid = e - e.mean()
    idx = S.sb_index(len(r50), H, 6, B, 99)
    a = S.sample_post(dens, B, np.random.default_rng(7)) / 100
    r50p = r50[idx]; ep = resid[idx]
    el = []
    for x in [i / 10 for i in range(11)]:
        o = S.simulate(r50p, ep, a, x, 'M'); el.append(o['logR'].mean())
    return int(np.argmax(el) * 10)

def post_case(X, F, cols, mask, sdc, L=6, c0=None, df=5, drop_focus=None):
    Xm = X.loc[mask, cols]
    out = {}
    fx = {}
    for c in KOU:
        e = ((F[c] - F['r0050'])[mask].values * 100)
        if drop_focus is not None:
            keep = np.ones(len(e), bool); keep[np.argsort(-e)[:drop_focus]] = False
            fx[c] = (e, keep)
        else:
            fx[c] = (e, None)
    A = Xm.values
    m = A.mean(axis=1)
    s2 = S.hac_var_mean(A - m[:, None], L); vP = float(S.hac_var_mean(m, L))
    pr = S.fit_g2(A.mean(axis=0), s2, df=df, c0=c0)
    for c in KOU:
        e, keep = fx[c]
        if keep is not None:
            ee, mm = e[keep], m[keep]
        else:
            ee, mm = e, m
        s2f = float(S.hac_var_mean(ee - mm, L)) + (sdc ** 2 if c.startswith('EXEC') else 0)
        p = S.posterior(pr, 'G2', float(ee.mean()), s2f, vP)
        out[c] = {'raw': float(ee.mean()), 'mean': p['mean'], 'p_pos': p['p_pos'], 'dens': p['dens']}
    return out

def run(focus_path, tag, check=True):
    if check: S.check_prereg()
    X, M = S.load_family()
    F = S.load_focus(focus_path)
    meta = json.loads((S.DATA / 'focus_meta.json').read_text(encoding='utf-8'))
    sdc = float(meta['cost_uncertainty']['sd_cost_pp'])
    cols = json.loads((S.DATA / 'families.json').read_text(encoding='utf-8'))['MAIN']
    idx = X.index
    allm = np.ones(len(idx), bool)
    cases = {'base': dict(mask=allm)}
    for y in sorted(set(idx.year)):
        cases[f'drop_{y}'] = dict(mask=(idx.year != y))
    cases['start_2008'] = dict(mask=(idx >= '2008-01-01'))
    cases['drop_best10_focus'] = dict(mask=allm, drop_focus=10)
    cases['HAC_L3'] = dict(mask=allm, L=3); cases['HAC_L12'] = dict(mask=allm, L=12)
    cases['G2_c0_0.05'] = dict(mask=allm, c0=0.05); cases['G2_c0_0.2'] = dict(mask=allm, c0=0.2); cases['G2_c0_1_deconvolveR'] = dict(mask=allm, c0=1.0)
    cases['G2_df4'] = dict(mask=allm, df=4); cases['G2_df7'] = dict(mask=allm, df=7)
    out = {'tag': tag, 'cases': {}}
    for k, kw in cases.items():
        r = post_case(X, F, cols, sdc=sdc, **kw)
        Fm = F[kw['mask']] if 'drop_focus' not in kw else F
        row = {}
        for c in KOU:
            row[c] = {'raw': r[c]['raw'], 'mean': r[c]['mean'], 'p_pos': r[c]['p_pos']}
            if c in ('EXEC_J', 'AB') and (k in ('base', 'start_2008', 'drop_best10_focus', 'HAC_L12', 'G2_c0_1_deconvolveR', 'G2_df7') or k.startswith('drop_20')):
                row[c]['x_star_10y'] = xstar(Fm, c, r[c]['dens'])
        out['cases'][k] = row
        print(k, {c: (round(row[c]['mean'], 3), round(row[c]['p_pos'], 3), row[c].get('x_star_10y')) for c in KOU}, flush=True)
    yr = [k for k in out['cases'] if k.startswith('drop_20')]
    out['drop_year_range'] = {c: {'mean_min': min(out['cases'][k][c]['mean'] for k in yr), 'mean_max': max(out['cases'][k][c]['mean'] for k in yr),
                                  'p_pos_min': min(out['cases'][k][c]['p_pos'] for k in yr), 'p_pos_max': max(out['cases'][k][c]['p_pos'] for k in yr),
                                  'xstar_min': min(out['cases'][k][c].get('x_star_10y', 999) for k in yr) if c in ('EXEC_J', 'AB') else None,
                                  'xstar_max': max(out['cases'][k][c].get('x_star_10y', -1) for k in yr) if c in ('EXEC_J', 'AB') else None}
                              for c in KOU}
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--focus', default=str(S.DATA / 'focus_windows.parquet'))
    ap.add_argument('--tag', default='V4')
    a = ap.parse_args()
    out = run(a.focus, a.tag)
    S.save_key('RESULTS_ROBUST' + ('' if a.tag == 'V4' else '_' + a.tag), out)
    print(json.dumps(out['drop_year_range'], ensure_ascii=False, indent=1))

if __name__ == '__main__':
    main()
