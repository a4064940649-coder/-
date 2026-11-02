#!/usr/bin/env python3
"""事後診斷（非事前登記，結論只當下一輪線索）：
 (1) P7 區辨：同一個 T2／T1 λ1.5 層內傾斜，市值改用 o7 回推股數（非 PIT：2025 錨點往回推，下市股缺）時，POST2019 對隨機分數的 t 是否明顯變高。
 (2) 隨機傾斜的結構性偏移：乘數 exp(λz) 在層內正規化後，平均把權重從層內大股移到小股（Jensen），量化各層平均主動權重對 log 市值的斜率。
寫 posthoc.json。"""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
import tcore as T, tlib as L
PRE = json.loads((L.DATA / 'PREREG.json').read_text(encoding='utf-8'))
OUT = {'note': '事後、未事前登記'}
def vs_random(kind, lam, cap, a, b, P=300):
    st = []
    for j in range(13):
        ds = T.sig_days(j, a, b)
        R, Rb = T.gross_matrix(ds, kind, lam, cap, P=P, seed=9000 + j)
        Rr, _ = T.gross_matrix(ds, kind, lam, cap, real=True)
        y = pd.Series(Rr[0] - R.mean(0), index=Rb.index); st.append(T.stats(y))
    return {'mean_pp_jmed': float(np.median([s['mean_pp'] for s in st])), 't_jmed': float(np.median([s['t'] for s in st]))}
res = {}
for kind in ('T1', 'T2'):
    res[f'{kind}_PIT'] = vs_random(kind, 1.5, False, '2019-01-01', '2026-10-31')
SH_PIT = T.SH.copy()
SE = pd.read_parquet(L.RAW / 'agents' / 'capmatched_shares_est.parquet'); SE.index = pd.to_datetime(SE.index)
T.SH = SE.reindex(index=T.IDX, columns=T.COLS).to_numpy(); T._PRE.clear()
for kind in ('T1', 'T2'):
    res[f'{kind}_o7shares'] = vs_random(kind, 1.5, False, '2019-01-01', '2026-10-31')
T.SH = SH_PIT; T._PRE.clear()
OUT['P7_vs_random_POST2019_PIT_vs_o7shares'] = res
print(res, flush=True)
# (2) 隨機傾斜的平均主動權重 vs log 市值（官方日、POST2019）
rng = np.random.default_rng(1)
slopes = {}
for kind in ('T1', 'T2'):
    ds = T.sig_days('off', '2019-01-01', '2026-10-31'); pre = T.precompute(ds, kind)
    for d, U, wb, tz in pre:
        t = T.IDX.get_loc(d); mc = T.mcap_row(t)[U]
        for nm, (idx, z) in tz.items():
            okz = z != 0
            if okz.sum() < 3: continue
            Z = np.tile(z, (500, 1)); sub = rng.permuted(Z[:, okz], axis=1); Z[:, okz] = sub
            Wp = T.tilt_mat(wb[idx], Z, 1.5, False).mean(0)
            act = (Wp - wb[idx]) / wb[idx].sum()
            x = np.log(mc[idx])
            slopes.setdefault(f'{kind}_{nm}', []).append(float(np.polyfit(x - x.mean(), act, 1)[0]))
OUT['random_tilt_active_weight_slope_on_logmcap'] = {k: {'mean': float(np.mean(v)), 'share_negative': float(np.mean(np.array(v) < 0)), 'n': len(v)} for k, v in slopes.items()}
(L.HERE / 'posthoc.json').write_text(L.dumps(OUT), encoding='utf-8')
print(L.dumps(OUT))
