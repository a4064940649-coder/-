#!/usr/bin/env python3
"""事後描述（不在 PREREG 內，不做判定）：歷史上「主策略 x%＋0050」每窗再平衡（含再平衡成本）的各期間年化、對 0050 t、對主策略（同口徑 x＝100%）t。
用法：python3 -I -B hist_mix.py → results.json 的 POSTHOC_hist_mix。"""
import sys, json, math
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import sc_lib as S

def mix(r50, rm, x):
    out = []; wm, we = x, 1 - x
    for a, b in zip(r50, rm):
        wm *= 1 + b; we *= 1 + a; W = wm + we
        if 0 < x < 1:
            d = wm - x * W
            W -= d * (S.C_M_SELL + S.C_E_BUY) if d > 0 else -d * (S.C_E_SELL + S.C_M_BUY)
        out.append(W); wm, we = x * W, (1 - x) * W
    W = np.array(out); return np.r_[W[0] - 1, W[1:] / W[:-1] - 1]

def main():
    S.check_prereg()
    F = S.load_focus()
    G = S.FC.GRID
    per = {'FULL': S.W_LONG, 'SEL': S.SEL, 'VAL': S.VAL, 'HOLD': S.HOLD}
    out = {}
    for c in ['AB', 'B10', 'EXEC', 'EXEC_J']:
        for x in (0.56, 0.8, 0.9, 1.0):
            r = mix(F['r0050'].values, F[c].values, x)
            rr = pd.Series(r, index=F.index)
            full = pd.Series(mix(F['r0050'].values, F[c].values, 1.0), index=F.index)
            row = {}
            for pn, (a, b) in per.items():
                m = S.wmask(F.index, a, b)
                idx = np.where(m)[0]
                days = (G[idx[-1] + 1] - G[idx[0]]).days
                cagr = (np.prod(1 + rr[m]) ** (365.25 / days) - 1) * 100
                e = (rr - F['r0050'])[m]; d = (rr - full)[m]
                row[pn] = {'cagr_pp': float(cagr), 't_vs_0050': S.tstat(e), 't_vs_main_x100': (S.tstat(d) if x < 1 else None),
                           'cagr_0050_pp': float((np.prod(1 + F['r0050'][m]) ** (365.25 / days) - 1) * 100)}
            W = np.cumprod(1 + rr.values)
            row['mdd_window'] = float((W / np.maximum.accumulate(W) - 1).min())
            out[f'{c}|x{int(x * 100)}'] = row
    S.save_key('POSTHOC_hist_mix', out)
    for k, v in out.items():
        print(k, {p: (round(v[p]['cagr_pp'], 1), round(v[p]['t_vs_0050'], 2), None if v[p]['t_vs_main_x100'] is None else round(v[p]['t_vs_main_x100'], 2)) for p in per}, round(v['mdd_window'], 3))
    print('0050', {p: round(out['AB|x100'][p]['cagr_0050_pp'], 1) for p in per})

if __name__ == '__main__':
    main()
