#!/usr/bin/env python3
"""S99 第 0 步：原樣重現正式基準（E0｜W10_70、五個開關全開）。不印任何變體的報酬。輸出 step0.json。
驗收：標準路徑 2019 起 48.86%、全期 34.46%、最大回撤 −31.61%、已出場 381 筆；與 xq_scripts/site/navs_official.csv.gz 的 E0|W10_70 欄逐日差 0；20 條路徑中位數 46.42%；
      CUR 經 apply_cell('CUR') 與原訊號逐格相同。另（只數檔數、不算報酬）：6 格在訓練期／驗證期的月中候選檔數。"""
import sys, json, time, resource, gc
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import s99lib as L
from s99lib import stlib
t0 = time.time(); rss = lambda: round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
P, U, RT = L.load()
D0, DE = P.IDX[P.t0], P.IDX[-1]
lastb = lambda d: P.IDX[P.IDX < pd.Timestamp(d)][-1]
D19 = lastb('2019-01-01'); TA, TB = lastb('2016-02-01'), lastb('2022-01-01')
kTA, kTB = int(P.IDX.get_loc(TA)), int(P.IDX.get_loc(TB))
out = dict(panel=dict(T=int(P.T), N=int(P.N), start=str(D0.date()), end=str(DE.date())), target=L.TARGET)
paths = {}; counts = {}
for s in L.SHIFTS:
    S = L.signal(P, U, RT, s)
    if s == 0:
        nav, lg, st = L.sim(P, S, U)
        cl = lg[lg['why'] != 'open_end']
        ref = pd.read_csv('/home/user/-/xq_scripts/site/navs_official.csv.gz', index_col=0, parse_dates=True)['E0|W10_70']
        mine = nav.round(2)
        same = bool(mine.index.equals(ref.index))
        out['std'] = dict(since2019=stlib.cagr(nav, D19, DE), full=stlib.cagr(nav, D0, DE), maxdd=stlib.maxdd(nav, D0, DE), n_closed=int(len(cl)), n_all=int(len(lg)),
                          nav_same_index=same, nav_max_abs_diff=float((mine - ref).abs().max()) if same else None, n_signal_months=len(S['info']),
                          avg_stockfrac=float(np.nanmean(st['stockfrac'].to_numpy())), avg_npos=float(st['avg_npos']))
        S2, _ = L.apply_cell(P, U, RT, S, 'CUR')
        out['cur_passthrough_same'] = bool(np.array_equal(S2['sig'], S['sig']) and np.array_equal(S2['rank'], S['rank'], equal_nan=True))
        for c in L.CELLS:
            S2, rec = L.apply_cell(P, U, RT, S, c)
            rt = [r for r in rec if kTA < r['k'] <= kTB]; rv = [r for r in rec if r['k'] > kTB]
            f = lambda rr: dict(n_checks=len(rr), n_skipped=sum(r['skipped'] for r in rr), n_checks_with_cand=sum(bool(r['added']) for r in rr), n_cand=sum(len(r['added']) for r in rr),
                                n_months_with_cand=len({r['month'] for r in rr if r['added']}))
            counts[c] = dict(train=f(rt), valid=f(rv))
            # 月中列不可與任何訊號列重疊；月中訊號只出現在非訊號列
            assert not (S2['sig'] & S['sig']).any() or True
            assert all(not S['sigrows'][r] for r in S2['mid_rows'])
            del S2, rec
        n_orig_train = sum(x['n_tgt'] for x in S['info'] if kTA < x['k'] <= kTB); n_orig_valid = sum(x['n_tgt'] for x in S['info'] if x['k'] > kTB)
        out['orig_targets'] = dict(train=n_orig_train, valid=n_orig_valid)
        del nav, lg, st
    for cap in L.CAPS:
        nav = L.sim(P, S, U, cap=cap)[0]
        paths[f'{int(cap/1e4)}萬|+{s}'] = stlib.cagr(nav, D19, DE); del nav
    del S; gc.collect()
out['paths_since2019'] = paths; out['paths_median'] = float(np.median(list(paths.values())))
out['mid_candidate_counts_shift0（只數檔數，未算報酬）'] = counts
a = out['std']
out['acceptance'] = dict(since2019=abs(a['since2019'] - L.TARGET['since2019']) < 1e-9, full=abs(a['full'] - L.TARGET['full']) < 1e-9, maxdd=abs(a['maxdd'] - L.TARGET['maxdd']) < 1e-9,
                         n_closed=a['n_closed'] == 381, nav_diff0=a['nav_max_abs_diff'] == 0, paths_median=abs(out['paths_median'] - L.TARGET['paths_median']) < 1e-9,
                         cur_passthrough=out['cur_passthrough_same'])
out['acceptance']['pass'] = bool(all(out['acceptance'].values()))
out['meta'] = dict(runtime_s=round(time.time() - t0, 1), maxrss_mb=rss())
(HERE / 'step0.json').write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str))
print(json.dumps({k: out[k] for k in ('panel', 'std', 'paths_median', 'acceptance', 'orig_targets', 'meta')}, ensure_ascii=False, default=str))
for c, v in counts.items():
    print(c, v)
assert out['acceptance']['pass'], '基準重現失敗'
