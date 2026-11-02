#!/usr/bin/env python3
"""複製者股數 c 與研究員 PIT (a) 的市值名次比較（只用股數×收盤，無報酬）。python3 -I -B r_compare.py"""
import sys, json
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent)); import pcore as P
L = P.L
Sa = P.build_shares('pit_a').to_numpy(); Sc = pd.read_parquet(HERE / 'shares_c.parquet').reindex(index=L.IDX, columns=L.COLS).to_numpy()
Se = P.build_shares('est').to_numpy()
rows = []
for d in L.official_days('2019-05-01', '2026-10-12'):
    t = L.IDX.get_loc(d); live = L.TRADED[max(t - 5, 0):t + 1].any(axis=0) & ~L.DRMASK
    def rk(S):
        mc = np.where(live & np.isfinite(S[t]), S[t] * L.CFF[t], np.nan); return pd.Series(mc).rank(ascending=False).to_numpy()
    ra, rc, re = rk(Sa), rk(Sc), rk(Se)
    def ov(x, y, k): return len(set(np.where(x <= k)[0]) & set(np.where(y <= k)[0]))
    both = np.isfinite(Sa[t]) & np.isfinite(Sc[t]) & live
    rel = np.abs(np.log(Sc[t][both] / Sa[t][both]))
    rows.append(dict(d=str(d.date()), top100_c_a=ov(rc, ra, 100), top300_c_a=ov(rc, ra, 300), top100_c_est=ov(rc, re, 100), top300_c_est=ov(rc, re, 300),
                     med_abs_log_c_a=float(np.median(rel)), p90_abs_log_c_a=float(np.percentile(rel, 90)), share_gt5pct=float(np.mean(rel > np.log(1.05)))))
D = pd.DataFrame(rows)
out = dict(mean=D.drop(columns='d').mean().to_dict(), min=D.drop(columns='d').min().to_dict(), by_year=D.assign(y=D.d.str[:4]).groupby('y').mean(numeric_only=True).round(4).to_dict('index'))
(HERE / 'r_compare.json').write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding='utf-8')
print(json.dumps(out['mean'], indent=1)); print(json.dumps(out['by_year'], indent=1))
