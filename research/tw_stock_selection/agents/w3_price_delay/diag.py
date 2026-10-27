#!/usr/bin/env python3
"""事後機制診斷（看完主結果後才加，不改判）：D1 高是否只是「R² 低／特質波動高」。"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run as R  # noqa: E402
from btlib import tstat  # noqa: E402
rows = []
for k, d in enumerate(R.ME[:-1]):
    ex0, pool = R.pool_at(d); ex1 = R.LPOS[R.NEXT[R.ME[k + 1]]]
    kk = R.WPOS[d]; lo = max(kk - 51, 4)
    d1, r2r, r2u = R.delay_D1(R.RWn[lo:kk + 1], R.RM_EW[lo - 4:kk + 1])
    vol = np.nanstd(R.RWn[lo:kk + 1], axis=0)
    fr = R.L.AO[ex1, R.LCOL] / R.L.AO[ex0, R.LCOL] - 1
    ok = pool & np.isfinite(d1) & np.isfinite(fr) & np.isfinite(r2r) & np.isfinite(vol)
    if ok.sum() < 50: continue
    g = R.qcut(d1, ok, 10)
    y = np.clip(fr[ok], -0.5, 1.0); rd = R.rank_pct(d1[ok]); rr = R.rank_pct(r2r[ok]); rv = R.rank_pct(vol[ok]); one = np.ones(len(y))
    b = np.linalg.lstsq(np.column_stack([one, rd, rr, rv]), y, rcond=None)[0]
    br = np.linalg.lstsq(np.column_stack([one, rr]), y, rcond=None)[0]
    s = pd.Series(d1[ok])
    rows.append((d, s.corr(pd.Series(r2r[ok]), method='spearman'), s.corr(pd.Series(vol[ok]), method='spearman'),
                 np.median(r2r[g == 9]), np.median(r2r[ok]), np.median(r2u[g == 9]), np.median(vol[g == 9]), np.median(vol[ok]),
                 b[1], b[2], b[3], br[1]))
df = pd.DataFrame(rows, columns=['d', 'sp_d1_r2r', 'sp_d1_vol52w', 'r2r_med_D10', 'r2r_med_pool', 'r2u_med_D10', 'vol_med_D10', 'vol_med_pool',
                                 'b_d1_ctrl', 'b_r2r_ctrl', 'b_vol_ctrl', 'b_r2r_only']).set_index('d')
out = {}
for w, (a, b) in R.WIN.items():
    x = df[(df.index >= pd.Timestamp(a)) & (df.index <= pd.Timestamp(b))]
    o = {c: round(float(x[c].mean()), 3) for c in ['sp_d1_r2r', 'sp_d1_vol52w', 'r2r_med_D10', 'r2r_med_pool', 'r2u_med_D10', 'vol_med_D10', 'vol_med_pool']}
    for c in ['b_d1_ctrl', 'b_r2r_ctrl', 'b_vol_ctrl', 'b_r2r_only']:
        o[c + '_pct_m'] = round(float(x[c].mean() * 100), 3); o['t_' + c] = tstat(x[c])
    out[w] = o; print(w, o)
f = HERE / 'results.json'; dd = json.loads(f.read_text(encoding='utf-8'))
dd['diag_mechanism_posthoc'] = {'note': '看完主結果後才加的事後診斷，只解釋機制、不改判。FM：下月報酬對 D1、R²_受限、52 週週報酬波動 的百分位回歸。', **out}
f.write_text(json.dumps(dd, ensure_ascii=False, indent=1), encoding='utf-8')
