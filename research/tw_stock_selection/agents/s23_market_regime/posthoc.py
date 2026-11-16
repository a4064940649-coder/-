#!/usr/bin/env python3
"""S23 事後描述（未登記，驗證期判定之後才跑；不影響判定、不得拿來換規則）。輸出 results_posthoc.json。python3 -I -B posthoc.py
(1) 全部 7 條候選在驗證窗（重新起跑，20 條路徑）與全期連續（20 條路徑）的年化差、回撤差、四段空頭差。
(2) 現行衛星實際持股比例（位移 0、100 萬全期連續）在四段空頭期間的平均，說明「錢進 0050」的回撤下限。"""
import sys, json, time, resource, gc
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
import s23lib
from s23lib import s17lib, s10lib, s4lib, stlib, COSTS

H = Path(__file__).resolve().parent
assert (H / 'results.json').exists(), '先跑 validate.py'
PRE = json.loads((H / 'PREREG.json').read_text())
RULES = PRE['rule_form']['candidates（7 條＝5 種規則，其中 BRD／LMA 各 2 個分位數門檻；全部計入 N，參與訓練期選擇）']
t0 = time.time()
rss = lambda: round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
P = s23lib.load()
INNOV, _ = s4lib.innov_panel(P)
U = stlib.universe(P, 'twse') & ~INNOV
del INNOV
RT = s23lib.surge(P)
s17lib.e50_fix_suspension(P)
for a in ('H', 'L', 'VOL', 'CHG', 'HF', 'VOLF', 'adjH', 'HI240', 'MA60', 'nbar', 'F_HI', 'IDXC', 'AO', 'listed', 'NEARHI', 'V60', 'liq'):
    if hasattr(P, a):
        delattr(P, a)
gc.collect()
lastb = lambda d: P.IDX[P.IDX < pd.Timestamp(d)][-1]
D0, DE = P.IDX[P.t0], P.IDX[-1]
D19, TB = lastb('2019-01-01'), lastb('2022-01-01')
CAPS = (6e5, 8e5, 1e6, 1.5e6, 2e6)
CELL = dict(sat_w=0.7, sizing='W10', lock=False)
ratio = json.loads((H / 'results.json').read_text())['meta']['fresh_capital_scale']


def sim(S1, cap=1e6, fresh=False):
    if fresh:
        S1 = s23lib.fresh_from(S1, s23lib.VALID_FIRST_MONTH)
    S2, ekw = s10lib.variant_S(S1, 'E0')
    return s10lib.simulate_s10(P, S2, U, cost=COSTS['strict'], capital=cap, **CELL, **ekw)


CFG = ['CUR'] + [r['id'] for r in RULES]
RD = {r['id']: r for r in RULES}
M = {c: {} for c in CFG}
for s in range(4):
    S = s10lib.make_signal_s10(P, U, RT, shift=s)
    F = s23lib.regime_features(P, U, RT, S)
    VV = {'CUR': S}
    for c in CFG[1:]:
        r = RD[c]
        VV[c] = s23lib.apply_regime(S, F, s23lib.triggered(F, r, r.get('thr')).to_numpy(), r['action'])[0]
    for cap in CAPS:
        key = f'{int(cap/1e4)}萬|+{s}'
        for c in CFG:
            nf = sim(VV[c], cap / ratio, fresh=True)[0]
            nc, _, st = sim(VV[c], cap)
            sgf = s17lib.seg_dd(nf); sgc = s17lib.seg_dd(nc)
            M[c][key] = dict(valid=stlib.cagr(nf, TB, DE), valid_dd=stlib.maxdd(nf, TB, DE), seg_fresh={k: sgf[k] for k in ('2022', '2024-07~2025-04')},
                             since2019=stlib.cagr(nc, D19, DE), full_dd=stlib.maxdd(nc, D0, DE), seg_cont=sgc)
            if c == 'CUR' and cap == 1e6 and s == 0:
                sf = st['stockfrac']
                stockfrac_seg = {k: float(sf.loc[a:b].mean()) for k, (a, b) in s17lib.SEGMENTS.items()}
                stockfrac_all = float(sf.loc[D19:].mean())
            del nf, nc, st
    del VV, S, F
    gc.collect()
    print('位移', s, f'{time.time()-t0:.0f}s', rss(), 'MB', flush=True)
keys = list(M['CUR'])
med = lambda v: float(np.median(v))
out = {'note': '事後描述，未登記；不影響判定、不得拿來換規則', 'candidates_valid_and_cont': {}}
for c in CFG[1:]:
    d = lambda f, sub=None: np.array([((M[c][k][f] if sub is None else M[c][k][f][sub]) - (M['CUR'][k][f] if sub is None else M['CUR'][k][f][sub])) for k in keys]) * 100
    o = {}
    for lab, f, sub in (('valid_cagr', 'valid', None), ('valid_dd', 'valid_dd', None), ('fresh_seg2022', 'seg_fresh', '2022'), ('fresh_seg2407', 'seg_fresh', '2024-07~2025-04'),
                        ('cont_since2019', 'since2019', None), ('cont_full_dd', 'full_dd', None), ('cont_seg2018', 'seg_cont', '2018'), ('cont_seg2020', 'seg_cont', '2020-03'),
                        ('cont_seg2022', 'seg_cont', '2022'), ('cont_seg2407', 'seg_cont', '2024-07~2025-04')):
        x = d(f, sub)
        o[lab] = dict(median_pp=med(x), npos=int((x > 0).sum()))
    out['candidates_valid_and_cont'][c] = o
out['CUR_paths_median'] = dict(valid=med([M['CUR'][k]['valid'] for k in keys]), valid_dd=med([M['CUR'][k]['valid_dd'] for k in keys]),
                               since2019=med([M['CUR'][k]['since2019'] for k in keys]), full_dd=med([M['CUR'][k]['full_dd'] for k in keys]),
                               seg_cont={sg: med([M['CUR'][k]['seg_cont'][sg] for k in keys]) for sg in s17lib.SEGMENTS})
out['CUR_stockfrac_mean_seg'] = stockfrac_seg; out['CUR_stockfrac_mean_since2019'] = stockfrac_all
out['meta'] = dict(runtime_s=round(time.time() - t0, 1), maxrss_mb=rss())
(H / 'results_posthoc.json').write_text(json.dumps(out, ensure_ascii=False, indent=1))
for c, o in out['candidates_valid_and_cont'].items():
    print(c, {k: (round(v['median_pp'], 2), v['npos']) for k, v in o.items()})
print(json.dumps({k: v for k, v in out.items() if k not in ('candidates_valid_and_cont',)}, ensure_ascii=False))
