#!/usr/bin/env python3
"""步驟 1a：程式一致性——以 V1（o6 G2、c0 0.1、格點 [−8,4]）重現 o6 的 G2×MAIN 事後均值與 P(α>0)（差 ≤ 0.01）。
另外重現 G1、G3 主欄。只讀 raw/。輸出 results.json 的 STEP1_O6_REPRO。
用法：python3 -I -B s2a_repro_o6.py
"""
import sys, json, math
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
import tlib as T, hmethods as H

def main():
    T.check_block('PREREG_DESIGN', files=False)
    X, lab, med = H.load_real()
    fam = H.Fam(X.values, lab, med)
    F = pd.read_parquet(T.RAW / 'o6shrink' / 'focus_windows.parquet')
    meta = json.loads((T.RAW / 'o6shrink' / 'focus_meta.json').read_text(encoding='utf-8'))
    sdc = float(meta['cost_uncertainty']['sd_cost_pp'])
    o6 = json.loads((T.RAW / 'o6shrink' / 'o6_results.json').read_text(encoding='utf-8'))['RESULTS_POSTERIOR']['families']['MAIN']['post']
    out = {}
    worst = 0.0
    for c in ['AB', 'B10', 'EXEC', 'EXEC_J']:
        x = ((F[c] - F['r0050']) * 100).values
        xh, s2 = fam._xs(x)
        if c.startswith('EXEC'): s2 += sdc ** 2
        r = {}
        for g, key in (('g2', 'G2'), ('g3', 'G3')):
            pr = fam.g2 if g == 'g2' else fam.g3
            d = T.posterior_g(pr['g'], fam.th, xh, s2, fam.vP, fam.G.step)
            s = T.dens_summary(d, fam.th)
            ref = o6[key][c]
            dm, dp = abs(s['mean'] - ref['mean']), abs(s['p_pos'] - ref['p_pos'])
            worst = max(worst, dm, dp)
            r[key] = {'mean': s['mean'], 'p_pos': s['p_pos'], 'o6_mean': ref['mean'], 'o6_p_pos': ref['p_pos'], 'diff_mean': dm, 'diff_p': dp}
        out[c] = r
    res = {'by_focus': out, 'max_abs_diff': worst, 'pass_le_0.01': bool(worst <= 0.01), 'sigma_w_real_Wlong': fam.sigma_w,
           'sigma_w_cluster114': fam.sw_c.get(114), 'vP': fam.vP, 'utc': T.now_utc()}
    T.save_key('STEP1_O6_REPRO', res, need=['PREREG_DESIGN'])
    print(json.dumps(res, ensure_ascii=False, indent=1))

if __name__ == '__main__':
    main()
