#!/usr/bin/env python3
"""S104 驗證：FROZEN 各題凍結格 × 20 條路徑，驗證窗 2021-12-30 → 2026-10-22，只跑一次（results.json 已存在就拒絕）。
另報：全期連續（訓練起點→2026-10-22）、位移 0 逐年、借券費 1%／3% 敏感度、K3 隱波倍數 × 價差敏感度、判定。用法：python3 -I -B validate.py"""
import sys, json, time, resource, hashlib
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert sha(HERE / 'PREREG.json') == (HERE / 'PREREG.sha256').read_text().split()[0]
assert sha(HERE / 'FROZEN.json') == (HERE / 'FROZEN.sha256').read_text().split()[0]
for ln in (HERE / 'code.sha256').read_text().splitlines():
    h, f = ln.split()
    assert sha(HERE / f) == h, f'{f} 與登記時不同'
assert not (HERE / 'results.json').exists(), 'results.json 已存在（驗證只跑一次）'
sys.path.insert(0, str(HERE))
import s104lib as L, s104cells as C, s104eval as E
t0 = time.time(); rss = lambda: round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
FZ = json.loads((HERE / 'FROZEN.json').read_text())
d = L.load_cache(); L.set_idx(d)
a, b, A0 = d.TB, d.TE, d.TA
offi = E.off_array(d)
res = dict(FROZEN_sha256=sha(HERE / 'FROZEN.json'), window_valid=[str(d.IDX[a].date()), str(d.IDX[b].date())], window_full=[str(d.IDX[A0].date()), str(d.IDX[b].date())])
res['official_E0'] = dict(valid=L.metrics(offi, d, a, b), full=L.metrics(offi, d, A0, b), yearly=L.yearly(offi, d, A0, b))
res['0050'] = dict(valid=L.metrics(d.E50, d, a, b), full=L.metrics(d.E50, d, A0, b), yearly=L.yearly(d.E50, d, A0, b))
NV = {}
cnt = lambda v, thr=0: int((v > thr).sum())
# ---------------- K1
c1 = FZ['selection']['K1']['frozen']
P, nv = E.k1_paths(d, c1, a, b, keep_nav=True)
Pf, _ = E.k1_paths(d, c1, A0, b)
nav0, tr0 = nv[0]
_, vS = E.med(P, 'short', 'cagr'); _, vD = E.med(P, 'combo_w0.3', 'd_sharpe'); _, vM = E.med(P, 'combo_w0.3', 'd_mdd')
P3, _ = E.k1_paths(d, c1, a, b, borrow=0.03); P1, _ = E.k1_paths(d, c1, a, b, borrow=0.01)
_, vS3 = E.med(P3, 'short', 'cagr')
V = dict(V1_short_cagr_pos_ge15=cnt(vS) >= 15, V2_dSharpe_w03_pos_ge15=cnt(vD) >= 15, V3_dMDD_w03_median_ge_m2pp=float(np.median(vM)) >= -0.02,
         V4_borrow3pct_short_cagr_median_pos=float(np.median(vS3)) > 0)
res['K1'] = dict(frozen=c1, T0=FZ['selection']['K1']['T0'], valid=E.summarize(P), full=E.summarize(Pf), borrow1=E.summarize(P1), borrow3=E.summarize(P3),
                 counts=dict(n_short_cagr_pos=cnt(vS), n_dSharpe_pos=cnt(vD), dMDD_median=float(np.median(vM))), verdict=V,
                 usable=bool(FZ['selection']['K1']['T0'] and all(V.values())),
                 yearly_shift0=dict(short=L.yearly(nav0, d, A0, b), **{f'combo_w{w}': L.yearly(L.overlay(offi, nav0, w), d, A0, b) for w in C.K1_W}),
                 trades_shift0_valid=dict(n=int(((tr0['ent_t'] > a)).sum()), cover_by=tr0[tr0['ent_t'] > a]['why'].value_counts().to_dict()))
NV['K1_short'] = nav0; NV['K1_combo_w0.3'] = L.overlay(offi, nav0, 0.3); NV['K1_combo_w0.5'] = L.overlay(offi, nav0, 0.5)
print('K1', c1, json.dumps(V), flush=True)
# ---------------- K2
c2 = FZ['selection']['K2']['frozen']
pre = C.K2Pre(d)
P, nv = E.k2_paths(d, pre, c2, a, b, keep_nav=True)
Pf, _ = E.k2_paths(d, pre, c2, A0, b)
P3, _ = E.k2_paths(d, pre, c2, a, b, borrow=0.03); P1, _ = E.k2_paths(d, pre, c2, a, b, borrow=0.01)
nL, tL, nS, tS, dual = nv[0]
_, vD = E.med(P, 'dual', 'cagr'); _, vC = E.med(P, 'combo_70off_30dual', 'd_sharpe'); _, vL = E.med(P, 'long', 'cagr'); _, vSS = E.med(P, 'short', 'cagr')
_, vD3 = E.med(P3, 'dual', 'cagr')
V = dict(V1_dual_cagr_pos_ge15=cnt(vD) >= 15, V2_combo_dSharpe_pos_ge15=cnt(vC) >= 15, V3_long_and_short_median_pos=bool(np.median(vL) > 0 and np.median(vSS) > 0),
         V4_borrow3pct_dual_pos_ge15=cnt(vD3) >= 15)
res['K2'] = dict(frozen=c2, T0=FZ['selection']['K2']['T0'], valid=E.summarize(P), full=E.summarize(Pf), borrow1=E.summarize(P1), borrow3=E.summarize(P3),
                 counts=dict(n_dual_pos=cnt(vD), n_combo_dSharpe_pos=cnt(vC)), verdict=V, usable=bool(FZ['selection']['K2']['T0'] and all(V.values())),
                 yearly_shift0=dict(long=L.yearly(nL, d, A0, b), short=L.yearly(nS, d, A0, b), dual=L.yearly(dual, d, A0, b), combo=L.yearly(L.blend(offi, dual, 0.3), d, A0, b)))
NV['K2_long'] = nL; NV['K2_short'] = nS; NV['K2_dual'] = dual; NV['K2_combo'] = L.blend(offi, dual, 0.3)
del pre
print('K2', c2, json.dumps(V), flush=True)
# ---------------- K3
c3 = FZ['selection']['K3']['frozen']
ON = pd.read_parquet(L.TMP / 'off_navs.parquet'); ON.columns = [str(x) for x in ON.columns]
TR = pd.read_parquet(L.TMP / 'off_trades.parquet')
P, nv = E.k3_paths(d, c3, a, b, C.K3_BASE['ivm'], C.K3_BASE['spread'], ON, TR, keep_nav=True)
Pf, _ = E.k3_paths(d, c3, A0, b, C.K3_BASE['ivm'], C.K3_BASE['spread'], ON, TR)
nW, tW, cb, offs = nv[0]
_, vC = E.med(P, 'combo', 'd_cagr'); _, vSh = E.med(P, 'combo', 'd_sharpe'); _, vM = E.med(P, 'combo', 'd_mdd')
sens = {}
for k, s in C.K3_SENS.items():
    Ps, _ = E.k3_paths(d, c3, a, b, s['ivm'], s['spread'], ON, TR)
    sens[k] = dict(d_cagr_median=E.med(Ps, 'combo', 'd_cagr')[0], n_pos=cnt(E.med(Ps, 'combo', 'd_cagr')[1]), d_sharpe_median=E.med(Ps, 'combo', 'd_sharpe')[0],
                   d_mdd_median=E.med(Ps, 'combo', 'd_mdd')[0], sleeve_cagr_median=E.med(Ps, 'warrant_sleeve', 'cagr')[0])
V = dict(V1_dCAGR_pos_ge15=cnt(vC) >= 15, V2_dCAGR_median_ge2pp=float(np.median(vC)) >= 0.02, V3_dSharpe_median_ge0=float(np.median(vSh)) >= 0,
         V4_dMDD_median_ge_m5pp=float(np.median(vM)) >= -0.05, V5_pessimistic_iv15_spr4_dCAGR_median_pos=sens['iv1.5|spr4']['d_cagr_median'] > 0)
res['K3'] = dict(frozen=c3, T0=FZ['selection']['K3']['T0'], valid=E.summarize(P), full=E.summarize(Pf), sensitivity_valid=sens,
                 counts=dict(n_dCAGR_pos=cnt(vC)), verdict=V, usable=bool(FZ['selection']['K3']['T0'] and all(V.values())),
                 yearly_shift0=dict(warrant_sleeve=L.yearly(nW, d, A0, b), combo=L.yearly(cb, d, A0, b), official_shift0=L.yearly(offs, d, A0, b)),
                 warrant_trades_shift0=dict(n=int(len(tW)), by_kind=tW['why'].value_counts().to_dict(), win_rate=float((tW['ret'] > 0).mean()),
                                            ret_median=float(tW['ret'].median()), ret_mean=float(tW['ret'].mean()),
                                            share_pnl_top5pct=float(tW['ret'].sort_values(ascending=False).head(max(1, len(tW) // 20)).sum() / tW['ret'].clip(lower=0).sum())))
NV['K3_warrant_sleeve'] = nW; NV['K3_combo'] = cb
print('K3', c3, json.dumps(V), flush=True)
NV['official_E0'] = offi; NV['0050'] = d.E50 / d.E50[A0]
pd.DataFrame(NV, index=d.IDX).iloc[A0:b + 1].round(6).to_csv(HERE / 'navs_shift0.csv.gz')
res['meta'] = dict(runtime_s=round(time.time() - t0), maxrss_mb=rss())
(HERE / 'results.json').write_text(json.dumps(res, ensure_ascii=False, indent=1))
print(res['meta'])
