#!/usr/bin/env python3
"""S104 訓練期：三題各格 × 20 條路徑，只算訓練窗（2016-01-30 → 2021-12-30）→ 依 PREREG 選格 → FROZEN.json＋sha256。
PREREG.json 與 code.sha256 必須相符；FROZEN.json 已存在就拒絕。用法：python3 -I -B train.py"""
import sys, json, time, resource, hashlib, datetime
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
PSHA = sha(HERE / 'PREREG.json')
assert PSHA == (HERE / 'PREREG.sha256').read_text().split()[0], 'PREREG 不符'
for ln in (HERE / 'code.sha256').read_text().splitlines():
    h, f = ln.split()
    assert sha(HERE / f) == h, f'{f} 與登記時不同'
assert not (HERE / 'FROZEN.json').exists(), 'FROZEN.json 已存在'
sys.path.insert(0, str(HERE))
import s104lib as L, s104cells as C, s104eval as E
t0 = time.time(); rss = lambda: round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
d = L.load_cache(); L.set_idx(d)
a, b = d.TA, d.TB
assert (str(d.IDX[a].date()), str(d.IDX[b].date())) == ('2016-01-30', '2021-12-30')
res = dict(PREREG_sha256=PSHA, window=[str(d.IDX[a].date()), str(d.IDX[b].date())], K1={}, K2={}, K3={})
# K1
for c in C.K1_CELLS:
    P, _ = E.k1_paths(d, c, a, b)
    sc, v = E.med(P, 'combo_w0.3', 'd_sharpe')
    sm, vs = E.med(P, 'short', 'cagr')
    res['K1'][c] = dict(score_median_dSharpe_w03=sc, n_dSharpe_pos=int((v > 0).sum()), short_cagr_median=sm, n_short_pos=int((vs > 0).sum()), summary=E.summarize(P))
    print(c, round(sc, 3), int((v > 0).sum()), round(sm, 4), int((vs > 0).sum()), flush=True)
# K2
pre = C.K2Pre(d)
for c in C.K2_CELLS:
    P, _ = E.k2_paths(d, pre, c, a, b)
    sc, v = E.med(P, 'dual', 'cagr')
    res['K2'][c] = dict(score_median_dual_cagr=sc, n_dual_pos=int((v > 0).sum()), summary=E.summarize(P))
    s = res['K2'][c]['summary']
    print(c, round(sc, 4), int((v > 0).sum()), 'L', round(s['long']['cagr']['median'], 4), 'S', round(s['short']['cagr']['median'], 4), flush=True)
del pre
# K3
ON = pd.read_parquet(L.TMP / 'off_navs.parquet'); ON.columns = [str(x) for x in ON.columns]
TR = pd.read_parquet(L.TMP / 'off_trades.parquet')
for c in C.K3_CELLS:
    P, _ = E.k3_paths(d, c, a, b, C.K3_BASE['ivm'], C.K3_BASE['spread'], ON, TR)
    sc, v = E.med(P, 'combo', 'd_cagr')
    ds, vs = E.med(P, 'combo', 'd_sharpe')
    res['K3'][c] = dict(score_median_dCAGR=sc, n_dCAGR_pos=int((v > 0).sum()), dSharpe_median=ds, summary=E.summarize(P))
    print(c, round(sc, 4), int((v > 0).sum()), round(ds, 3), flush=True)
# 選格
def pick(K, key, cells):
    return sorted(cells, key=lambda c: (res[K][c][key], -cells.index(c)), reverse=True)
sel = {}
o1 = pick('K1', 'score_median_dSharpe_w03', list(C.K1_CELLS)); t1 = res['K1'][o1[0]]
sel['K1'] = dict(order=o1, frozen=o1[0], T0=bool(t1['short_cagr_median'] > 0 and t1['n_dSharpe_pos'] >= 14))
o2 = pick('K2', 'score_median_dual_cagr', list(C.K2_CELLS)); t2 = res['K2'][o2[0]]
sel['K2'] = dict(order=o2, frozen=o2[0], T0=bool(t2['score_median_dual_cagr'] > 0 and t2['n_dual_pos'] >= 14))
o3 = pick('K3', 'score_median_dCAGR', list(C.K3_CELLS)); t3 = res['K3'][o3[0]]
sel['K3'] = dict(order=o3, frozen=o3[0], T0=bool(t3['score_median_dCAGR'] >= 0.01 and t3['n_dCAGR_pos'] >= 14 and t3['dSharpe_median'] >= 0))
res['selection'] = sel
res['meta'] = dict(runtime_s=round(time.time() - t0), maxrss_mb=rss())
(HERE / 'results_train.json').write_text(json.dumps(res, ensure_ascii=False, indent=1))
FZ = dict(written_at=datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S'), PREREG_sha256=PSHA, selection=sel,
          frozen_params={K: (C.K1_CELLS, C.K2_CELLS, C.K3_CELLS)[i][sel[K]['frozen']] for i, K in enumerate(('K1', 'K2', 'K3'))},
          note='凍結後 validate.py 只跑一次；不換格、不換題。T0 不過者驗證期只描述，判定必為不通過。')
(HERE / 'FROZEN.json').write_text(json.dumps(FZ, ensure_ascii=False, indent=1))
(HERE / 'FROZEN.sha256').write_text(f"{sha(HERE / 'FROZEN.json')}  FROZEN.json\n")
print(json.dumps(sel, ensure_ascii=False)); print(res['meta'])
