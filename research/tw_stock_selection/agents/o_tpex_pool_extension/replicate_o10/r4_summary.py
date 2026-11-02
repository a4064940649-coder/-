#!/usr/bin/env python3
"""replicate_o10 r4：彙整所有格，輸出 out/summary.json。"""
import json, sys, math
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import numpy as np, pandas as pd
import r3_stats as R

K = ['ann_ALL', 'b_ALL', 'exc_ALL', 't_ALL', 'nwt_ALL', 't_drop3', 'daily_cagr', 'daily_cagr_0050', 'maxdd', 'n_ALL',
     'exc_SEL', 't_SEL', 'exc_VAL', 't_VAL', 'ann_HOLD', 'b_HOLD', 'exc_HOLD', 't_HOLD', 'exc_OOS', 't_OOS', 'exc_Y19', 't_Y19',
     'exc_S1', 'exc_S2', 'exc_S3', 'mpp_ALL', 'mpp_Y19']
out = {}
for var, cell in [('M', '100G'), ('BASE', '100G'), ('M', '100F'), ('M', '300G'), ('M', '300F'), ('M', '1000G'), ('M', '1000F'),
                  ('M', '100GS'), ('BASE', '300G'), ('BASE', '1000G'), ('R1', '100G'), ('R2', '100G'), ('R2', '30G'),
                  ('N5', '100G'), ('N15', '100G'), ('P300', '100G'), ('P500', '100G'), ('L0', '100G'), ('L20', '100G'),
                  ('M', '100G_d05'), ('M', '100G_d00'), ('M', '100G_oa05')]:
    s = R.summ(R.load(var, cell))
    out[f'{var}|{cell}'] = {k: s[k] for k in K}
s = R.summ(R.load('M', '100G'), holes=True)
out['M|100G|holes_incl'] = {k: s[k] for k in ('exc_ALL', 't_ALL', 'n_ALL')}
out['PAIR_M_BASE_100G'] = R.pair(R.load('M', '100G'), R.load('BASE', '100G'))
out['PAIR_M_BASE_300G'] = R.pair(R.load('M', '300G'), R.load('BASE', '300G'))
out['PAIR_M_BASE_1000G'] = R.pair(R.load('M', '1000G'), R.load('BASE', '1000G'))
# BASE j0 逐年報酬
n = R.load('BASE', '100G')[0][0]
out['BASE_j0_yearly_ret'] = {str(k.year): round(float(v), 2) for k, v in (n.resample('YE').last().pct_change() * 100).dropna().items()}


# 0050 核心＋衛星
def sat(sv, sc, w):
    res = []
    for j in range(13):
        a = R.win(R.load(sv, sc)[j][0], j); a = a[~a.hole]
        ws = 1 - w
        # 窗末權重漂移 → 再平衡換手
        post_s = ws * (1 + a.rs) / (ws * (1 + a.rs) + w * (1 + a.rb))
        turn = 2 * np.abs(post_s - ws)   # 兩邊合計
        rp = w * a.rb + ws * a.rs - turn * 0.0015
        x = rp - a.rb
        o = dict(exc=R.ann(rp, a.days) - R.ann(a.rb, a.days), t=R.tstat(x), te=float(x.std(ddof=1) * math.sqrt(12) * 100))
        res.append(o)
    return {k: round(float(np.median([r[k] for r in res])), 3) for k in res[0]}
out['SAT_70_R2'] = sat('R2', '30G', 0.7)
out['SAT_50_R2'] = sat('R2', '50G', 0.5)
out['SAT_70_BASE'] = sat('BASE', '30G', 0.7)
te = []
for j in range(13):
    a = R.win(R.load('M', '100G')[j][0], j); a = a[~a.hole]; te.append(a.x.std(ddof=1) * math.sqrt(12) * 100)
out['TE_M_100'] = round(float(np.median(te)), 2)
cr = []
for j in range(13):
    a = R.win(R.load('R2', '100G')[j][0], j); b = R.win(R.load('BASE', '100G')[j][0], j)
    m = ~a.hole; cr.append(np.corrcoef(a.x[m], b.x[m])[0, 1])
out['CORR_R2x_BASEx'] = round(float(np.median(cr)), 3)
json.dump(out, open(HERE / 'out' / 'summary.json', 'w'), ensure_ascii=False, indent=1)
for k, v in out.items():
    if isinstance(v, dict) and 'exc_ALL' in v:
        print(k, 'exc', v['exc_ALL'], 't', v['t_ALL'], 'SEL', v.get('exc_SEL'), v.get('t_SEL'), 'OOS', v.get('exc_OOS'), v.get('t_OOS'), 'HOLD', v.get('exc_HOLD'), 'Y19', v.get('exc_Y19'), v.get('t_Y19'))
    else:
        print(k, v)
