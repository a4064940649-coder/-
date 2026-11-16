#!/usr/bin/env python3
"""S30 訓練期：4 條規則＋現行 × 20 條路徑，只看訓練窗（2016-01-29 → 2021-12-30，全期連續模擬切出來）；依 PREREG 凍結一條。
輸出 results_train.json、FROZEN.json、FROZEN.sha256。PREREG.json 的 sha256 必須相符；FROZEN.json 已存在就拒絕重跑。python3 -I -B train.py"""
import sys, json, hashlib, time, resource, gc, datetime
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
import s30lib
from s30lib import s17lib, s10lib, s4lib, stlib

H = Path(__file__).resolve().parent
pre = (H / 'PREREG.json').read_bytes()
sha = hashlib.sha256(pre).hexdigest()
assert sha == (H / 'PREREG.sha256').read_text().split()[0], 'PREREG.json 與 PREREG.sha256 不符'
assert not (H / 'FROZEN.json').exists(), 'FROZEN.json 已存在，拒絕重跑'
PRE = json.loads(pre)
RULES = {'R_LOWB': 'LOWB', 'R_LBQ': 'LBQ', 'R_HDLV': 'HDLV', 'R_MIX': 'MIX'}
assert set(RULES) == {k for k in PRE['rules（4 條，全部計入 N；參與訓練期選擇）'] if k != 'common'}
t_all = time.time()
rss = lambda: round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
P = s30lib.load()
assert (P.T, P.N, str(P.IDX[-1].date())) == (2876, 2142, '2026-10-22'), 'P0 失敗'
INNOV, _ = s4lib.innov_panel(P)
U = stlib.universe(P, 'twse') & ~INNOV
RT = s30lib.surge(P)
s17lib.e50_fix_suspension(P)
Hd, info, FR = s30lib.holdings(P, INNOV)
IDXS = {b: s30lib.basket_index(P, Hd[b], FR) for b in s30lib.BASKETS}
IDXS['MIX'] = s30lib.mix_index(P, IDXS['LBQ'])
for b, D in IDXS.items():
    assert np.isfinite(D['Dc'][P.t0:]).all() and np.isfinite(D['Do'][P.t0:]).all(), b
BX = {r: s30lib.base_arrays(P, b, IDXS) for r, b in RULES.items()}
BX['CUR'] = None
del INNOV
for a in ('H', 'L', 'VOL', 'CHG', 'HF', 'VOLF', 'adjH', 'HI240', 'MA60', 'nbar', 'F_HI', 'IDXC', 'AO', 'listed', 'NEARHI', 'V60', 'liq'):
    if hasattr(P, a):
        delattr(P, a)
gc.collect()
print('載入完成', f'{time.time()-t_all:.0f}s', rss(), 'MB', flush=True)

lastb = lambda d: P.IDX[P.IDX < pd.Timestamp(d)][-1]
TA, TB = lastb('2016-02-01'), lastb('2022-01-01')
CAPS = (6e5, 8e5, 1e6, 1.5e6, 2e6)


def tr_metrics(nav):
    sg = s17lib.seg_dd(nav)
    return dict(train=stlib.cagr(nav, TA, TB), train_dd=stlib.maxdd(nav, TA, TB), seg2018=sg['2018'], seg2020=sg['2020-03'])


# 驗收：CUR 主跑＝A0
S0 = s10lib.make_signal_s10(P, U, RT, shift=0)
nav, lg, _ = s30lib.sim(P, S0, U)
D19 = lastb('2019-01-01')
a0 = dict(since2019=stlib.cagr(nav, D19, P.IDX[-1]), full=stlib.cagr(nav, P.IDX[P.t0], P.IDX[-1]), maxdd=stlib.maxdd(nav, P.IDX[P.t0], P.IDX[-1]), n_closed=int((lg['why'] != 'open_end').sum()))
assert abs(a0['since2019'] - 0.4899) < 5e-5 and abs(a0['full'] - 0.3454) < 5e-5 and abs(a0['maxdd'] + 0.3211) < 5e-5 and a0['n_closed'] == 381, a0
del nav, lg

M = {c: {} for c in BX}
for s in range(4):
    S = S0 if s == 0 else s10lib.make_signal_s10(P, U, RT, shift=s)
    for cap in CAPS:
        key = f'{int(cap/1e4)}萬|+{s}'
        for c, B in BX.items():
            nav = s30lib.sim(P, S, U, B=B, cap=cap)[0]
            M[c][key] = tr_metrics(nav)
            del nav
    if s:
        del S
    gc.collect()
    print('位移', s, f'{time.time()-t_all:.0f}s', rss(), 'MB', flush=True)

keys = list(M['CUR'])
res = {'PREREG_sha256': sha, 'acceptance': dict(A0=a0), 'window': dict(train=[str(TA.date()), str(TB.date())]), 'paths': M, 'summary': {}}
bench = pd.Series(P.E50, index=P.IDX)
res['bench_0050_train'] = dict(cagr=stlib.cagr(bench, TA, TB), maxdd=stlib.maxdd(bench, TA, TB))
for c in BX:
    v = np.array([M[c][k]['train'] for k in keys]); dd = np.array([M[c][k]['train_dd'] for k in keys])
    o = dict(train_median=float(np.median(v)), train_dd_median=float(np.median(dd)))
    if c != 'CUR':
        d = (v - np.array([M['CUR'][k]['train'] for k in keys])) * 100
        ddd = (dd - np.array([M['CUR'][k]['train_dd'] for k in keys])) * 100          # > 0＝較淺
        s18 = np.array([M[c][k]['seg2018'] - M['CUR'][k]['seg2018'] for k in keys]) * 100
        s20 = np.array([M[c][k]['seg2020'] - M['CUR'][k]['seg2020'] for k in keys]) * 100
        o.update(cagr_diff_median_pp=float(np.median(d)), cagr_npos=int((d > 0).sum()), cagr_diff_min_pp=float(d.min()), cagr_diff_max_pp=float(d.max()),
                 dd_improve_median_pp=float(np.median(ddd)), n_dd_shallower_05=int((ddd >= 0.5).sum()), dd_improve_min_pp=float(ddd.min()), dd_improve_max_pp=float(ddd.max()),
                 seg2018_improve_median_pp=float(np.median(s18)), seg2020_improve_median_pp=float(np.median(s20)))
        o['T0'] = bool(o['n_dd_shallower_05'] >= 14 and o['cagr_diff_median_pp'] >= -1.0)
    res['summary'][c] = o
cand = list(RULES)
passing = [c for c in cand if res['summary'][c]['T0']]
if passing:
    best = sorted(passing, key=lambda c: (res['summary'][c]['dd_improve_median_pp'], res['summary'][c]['cagr_diff_median_pp']), reverse=True)[0]
else:
    best = sorted(cand, key=lambda c: (res['summary'][c]['n_dd_shallower_05'], res['summary'][c]['cagr_diff_median_pp']), reverse=True)[0]
t0 = bool(res['summary'][best]['T0'])
res['selection'] = dict(T0_passing=passing, frozen=best, T0_pass=t0,
                        T0_text='訓練期找到「回撤較淺且報酬不明顯變差」的防禦底倉' if t0 else '訓練期沒有規則通過 T0（仍凍結回撤改善路徑數最多的一條，驗證期只描述，判定必為不通過）')
res['basket_turns'] = {b: dict(avg_turnover=float(np.mean([x['turnover'] for x in IDXS[b]['turns'][1:]])), avg_cost=float(np.mean([x['cost'] for x in IDXS[b]['turns'][1:]])),
                               n=len(IDXS[b]['turns']) - 1) for b in s30lib.BASKETS}
res['meta'] = dict(runtime_s=round(time.time() - t_all, 1), maxrss_mb=rss())
(H / 'results_train.json').write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str))
FZ = dict(written_at=datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S'), PREREG_sha256=sha, frozen_rule=best, frozen_basket=RULES[best], T0_pass=t0,
          T0_text=res['selection']['T0_text'], train_score=res['summary'][best], note='凍結後 validate.py 只跑一次；不得換規則、組合或門檻。')
txt = json.dumps(FZ, ensure_ascii=False, indent=1, default=str)
(H / 'FROZEN.json').write_text(txt)
fsha = hashlib.sha256(txt.encode()).hexdigest()
(H / 'FROZEN.sha256').write_text(f'{fsha}  FROZEN.json\n')
for c in BX:
    print(c, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in res['summary'][c].items()}, flush=True)
print('凍結', best, 'T0', t0, fsha, res['meta'])
