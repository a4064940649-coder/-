#!/usr/bin/env python3
"""S45 第 0 步（只做描述，不算任何報酬）：python3 -I -B prep.py → data_check.json
- 驗收 A1：s45lib.universe(1e7, 10) 與 stlib.universe(P,'twse') & ~INNOV 逐位相同。
- 15 格股池：訊號日（位移 0）股池檔數、SURGE 有值檔數、L_MA（季線上）檔數；F1 目標數與對現行的重疊率。"""
import sys, json, time, resource
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
import s45lib as L
from s45lib import s17lib, s10lib, s4lib, stlib, cslib

H = Path(__file__).resolve().parent
t0 = time.time()
rss = lambda: round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
P = L.load()
assert (P.T, P.N, str(P.IDX[-1].date())) == (2876, 2142, '2026-10-22'), 'P0 失敗'
INNOV, _ = s4lib.innov_panel(P)
U_ref = stlib.universe(P, 'twse') & ~INNOV
U = {c['id']: L.universe(P, INNOV, c['adv'], c['px']) for c in L.CELLS}
a1 = bool(np.array_equal(U[L.CUR], U_ref))
F = s17lib.rev_family(P, defs=['STD'])
RT = dict(S=F['S']['STD'], av=F['av'])
out = dict(flags=dict(EXRIGHT_FIX=cslib.EXRIGHT_FIX, DEADLINE_FIX=cslib.DEADLINE_FIX, INS15_MASK=cslib.INS15_MASK),
           A1_universe_bitwise_equal=a1, P0=dict(T=P.T, N=P.N, last=str(P.IDX[-1].date())), cells={})
assert a1, 'A1 失敗：現行門檻沒有逐位重現'
sg = {}
for c in L.CELLS:
    sg[c['id']] = s10lib.make_signal_s10(P, U[c['id']], RT, shift=0)
cur_t = {x['month']: set(x['targets']) for x in sg[L.CUR]['info']}
for c in L.CELLS:
    S = sg[c['id']]; u = U[c['id']]
    ks = [x['k'] for x in S['info']]
    months = [x['month'] for x in S['info']]
    nu = np.array([int(u[k].sum()) for k in ks])
    sv = {m: F['S']['STD'].loc[pd.Period(m, 'M')].to_numpy() for m in months}
    nsv = np.array([int((u[k] & np.isfinite(sv[m])).sum()) for k, m in zip(ks, months)])
    nma = np.array([x['n_ma'] for x in S['info']]); ntg = np.array([x['n_tgt'] for x in S['info']])
    m19 = np.array([m >= '2019-01' for m in months])
    ov = [len(set(x['targets']) & cur_t.get(x['month'], set())) for x in S['info']]
    tot = sum(len(x['targets']) for x in S['info'])
    out['cells'][c['id']] = dict(
        adv=c['adv'], px=c['px'], n_months=len(ks),
        univ_mean=float(nu.mean()), univ_mean_2019=float(nu[m19].mean()), univ_min=int(nu.min()),
        surge_valid_mean=float(nsv.mean()), lma_mean=float(nma.mean()), f1_targets_mean=float(ntg.mean()), f1_targets_total=int(tot),
        f1_overlap_with_cur=float(sum(ov) / max(tot, 1)),
        univ_days_mean_since2015_07=float(u[P.t0:].sum(axis=1).mean()))
    print(c['id'], json.dumps(out['cells'][c['id']], ensure_ascii=False), flush=True)
out['runtime_s'] = round(time.time() - t0, 1); out['maxrss_mb'] = rss()
(H / 'data_check.json').write_text(json.dumps(out, ensure_ascii=False, indent=1))
print('完成', out['runtime_s'], 's', out['maxrss_mb'], 'MB')
