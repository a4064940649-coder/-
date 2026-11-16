#!/usr/bin/env python3
"""S32 第 0 步（不算任何規則的報酬）：面板（EXRIGHT_FIX＝True）、引擎驗收 A0（現行 48.99／34.54／−32.11／381）、
包裝引擎驗收 A1（simulate_w 在權重＝等額時與 s10lib.simulate_s10 逐位相同：位移 0／100 萬、位移 2／150 萬、位移 3／60 萬）、
F1 名單上 σ60、σ250、β60 的涵蓋率與分布、各規則倍數 m 的分布（只看權重，不算報酬）。輸出 data_check.json。python3 -I -B prep.py"""
import sys, json, time, resource, gc
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
import s32lib
from s32lib import s17lib, s10lib, s4lib, stlib, COSTS

H = Path(__file__).resolve().parent
t_all = time.time()
rss = lambda: round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
P = s32lib.load()
assert (P.T, P.N, str(P.IDX[-1].date())) == (2876, 2142, '2026-10-22'), 'P0 失敗'
INNOV, _ = s4lib.innov_panel(P)
U = stlib.universe(P, 'twse') & ~INNOV
del INNOV
RT = s32lib.surge(P)
fix = s17lib.e50_fix_suspension(P)
for a in ('H', 'L', 'VOL', 'CHG', 'HF', 'VOLF', 'adjH', 'HI240', 'MA60', 'nbar', 'F_HI', 'IDXC', 'AO', 'listed', 'NEARHI', 'V60', 'liq'):
    if hasattr(P, a):
        delattr(P, a)
gc.collect()
print('載入完成', f'{time.time()-t_all:.0f}s', rss(), 'MB', flush=True)
lastb = lambda d: P.IDX[P.IDX < pd.Timestamp(d)][-1]
D19 = lastb('2019-01-01')
CELL = dict(sat_w=0.7, sizing='W10', lock=False)
cols = ['code', 'ent_t', 'exit_t', 'why', 'shares', 'cost', 'proceeds']
acc = {'exright_fix': bool(P.exright_fix), 'patch': s32lib.PATCH}
S0 = s10lib.make_signal_s10(P, U, RT, shift=0)
for s, cap in ((0, 1e6), (2, 1.5e6), (3, 6e5)):
    S = S0 if s == 0 else s10lib.make_signal_s10(P, U, RT, shift=s)
    S2, ekw = s10lib.variant_S(S, 'E0')
    nav0, lg0, st0 = s10lib.simulate_s10(P, S2, U, cost=COSTS['strict'], capital=cap, **CELL, **ekw)
    Sw, _ = s32lib.build_wmul(P, S, None)
    S2w, ekw = s10lib.variant_S(Sw, 'E0')
    nav1, lg1, st1 = s32lib.simulate_w(P, S2w, U, cost=COSTS['strict'], capital=cap, **CELL, **ekw)
    same = dict(nav_bitwise=bool(np.array_equal(nav0.to_numpy(), nav1.to_numpy())),
                trades_bitwise=bool(len(lg0) == len(lg1) and (lg0[cols].reset_index(drop=True) == lg1[cols].reset_index(drop=True)).all().all()))
    # 另驗：wmul 全部明確給 1.0 也相同
    Sw1 = dict(S); Sw1['wmul'] = {(x['k'], int(np.flatnonzero(P.COLS == c)[0])): 1.0 for x in S['info'] for c in x['targets']}
    S2w1, ekw = s10lib.variant_S(Sw1, 'E0')
    nav2 = s32lib.simulate_w(P, S2w1, U, cost=COSTS['strict'], capital=cap, **CELL, **ekw)[0]
    same['nav_bitwise_explicit_ones'] = bool(np.array_equal(nav0.to_numpy(), nav2.to_numpy()))
    if s == 0:
        a0 = dict(since2019=stlib.cagr(nav0, D19, P.IDX[-1]), full=stlib.cagr(nav0, P.IDX[P.t0], P.IDX[-1]), maxdd=stlib.maxdd(nav0, P.IDX[P.t0], P.IDX[-1]),
                  n_closed=int((lg0['why'] != 'open_end').sum()))
        a0['ok'] = bool(abs(a0['since2019'] - 0.4899) < 0.00025 and abs(a0['full'] - 0.3454) < 0.00025 and abs(a0['maxdd'] + 0.3211) < 0.00025 and a0['n_closed'] == 381)
        acc['A0'] = a0
        acc['CUR_maxw'] = dict(max=float(st1['maxw'].max()), mean=float(st1['maxw'][st1['nposd'] > 0].mean()))
    acc[f'A1|+{s}|{int(cap/1e4)}萬'] = same
    assert all(same.values()), ('A1 失敗', s, same)
assert acc['A0']['ok'], acc['A0']
print('驗收', json.dumps({k: v for k, v in acc.items() if k != 'patch'}, ensure_ascii=False), flush=True)

# ---------------------------------------------------------------- 特徵涵蓋率與權重分布（不算報酬）
chk = dict(panel=dict(T=P.T, N=P.N, last=str(P.IDX[-1].date())), acceptance=acc)
per = lambda m: 'train' if pd.Period(m, 'M') <= s32lib.TRAIN_LAST_MONTH else ('valid' if pd.Period(m, 'M') >= s32lib.VALID_FIRST_MONTH else 'pre')
feat = []
for r in s32lib.RULES + [s32lib.CONTROL]:
    _, lg = s32lib.build_wmul(P, S0, r)
    o = {}
    for pp in ('train', 'valid'):
        L = [x for x in lg if per(x['month']) == pp]
        m = np.concatenate([x['m'] for x in L]); n = np.array([len(x['m']) for x in L])
        o[pp] = dict(n_months=len(L), n_names=int(len(m)), m_q=[float(np.quantile(m, q)) for q in (0, .1, .25, .5, .75, .9, 1)],
                     share_at_lo=float((m <= s32lib.LO + 1e-9).mean()), share_at_hi=float((m >= s32lib.HI - 1e-9).mean()),
                     share_m_eq_1=float((np.abs(m - 1) < 1e-9).mean()), months_sum_dev_max=float(max(abs(sum(x['m']) - len(x['m'])) for x in L)),
                     months_single=int((n == 1).sum()))
    chk[r['id']] = o
    if r['id'] == 'IV60':
        for x in lg:
            for c, s6, b6 in zip(x['codes'], x['sig60'], x['beta60']):
                feat.append((x['month'], per(x['month']), c, s6, b6))
fd = pd.DataFrame(feat, columns=['month', 'period', 'code', 'sig60', 'beta60'])
for pp in ('train', 'valid'):
    d = fd[fd['period'] == pp]
    chk[f'features_{pp}'] = dict(n=int(len(d)), sig60_missing=float(d['sig60'].isna().mean()), beta60_missing=float(d['beta60'].isna().mean()),
                                 sig60_ann_q=[float(np.nanquantile(d['sig60'] * np.sqrt(252), q)) for q in (.1, .25, .5, .75, .9)],
                                 beta60_q=[float(np.nanquantile(d['beta60'], q)) for q in (.1, .25, .5, .75, .9)],
                                 beta_below_floor=float((d['beta60'] < s32lib.BETA_FLOOR).mean()),
                                 corr_sig_beta=float(d[['sig60', 'beta60']].corr().iloc[0, 1]),
                                 within_month_sig_ratio_max_min_median=float(d.groupby('month')['sig60'].agg(lambda v: v.max() / v.min() if len(v) > 1 else np.nan).median()))
chk['runtime_s'] = round(time.time() - t_all, 1); chk['maxrss_mb'] = rss()
(H / 'data_check.json').write_text(json.dumps(chk, ensure_ascii=False, indent=1))
print(json.dumps({k: v for k, v in chk.items() if k != 'acceptance'}, ensure_ascii=False, indent=1))
