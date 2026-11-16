#!/usr/bin/env python3
"""S42 主程式：資料版本（CUR／FPM／ALL／COMB）× 訊號日（A10、A11、A12、A13、A15）× 位移 0～3 × 本金 5 種 × 滑價（0.3／0.4／0.5%）全部跑。
ON＝只用 A10（20 條）；LATE＝A10～A15（100 條）。每條另算 S27 HALF／ZERO 歷史類比。輸出 results.json。python3 -I -B run.py"""
import sys, json, time, resource, gc, hashlib
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
import s42lib as L
from s42lib import s17lib, s10lib, s4lib, stlib, s39lib, s40lib

H = Path(__file__).resolve().parent
t0 = time.time()
rss = lambda: round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
res = dict(SPEC_sha256=hashlib.sha256((H / 'SPEC.json').read_bytes()).hexdigest())

P = L.load()
assert (P.T, P.N, str(P.IDX[-1].date())) == (2876, 2142, '2026-10-22'), 'P0 失敗'
INNOV, _ = s4lib.innov_panel(P)
U = stlib.universe(P, 'twse') & ~INNOV
del INNOV
F = s17lib.rev_family(P, defs=['STD'])
s17lib.e50_fix_suspension(P)
fp = s39lib.load_fp()
for a in ('H', 'L', 'VOL', 'CHG', 'HF', 'VOLF', 'adjH', 'HI240', 'MA60', 'nbar', 'F_HI', 'IDXC', 'AO', 'listed', 'NEARHI', 'V60', 'liq'):
    if hasattr(P, a):
        delattr(P, a)
gc.collect()
print('載入', f'{time.time()-t0:.0f}s', rss(), 'MB', flush=True)

S_CUR = F['S']['STD']
S_FPM = s39lib.surge_variant(P, F, fp, 'FPM')
av0 = F['av']
months = list(S_CUR.index)
AVD = {D: L.av_family(P, months, D) for D in L.FAM_D}
AVDL = s40lib.av_fix(P, av0, s40lib.EXT_ALL)
LATE = L.Late(P)
acc = dict(A3_A10_av_equal=bool(AVD[10] == av0),
           FPM_rows_before_2024_06_identical=bool(np.array_equal(S_FPM.loc[:'2024-05'].to_numpy(), S_CUR.loc[:'2024-05'].to_numpy(), equal_nan=True)))
VARS = ('CUR', 'FPM', 'ALL', 'COMB')
ncell = {}


def RT(v, D, shift):
    S = S_FPM if v in ('FPM', 'COMB') else S_CUR
    av = AVD[D]
    if v in ('ALL', 'COMB'):
        av = L.av_combine(av, AVDL)
        cells = LATE.cells(s40lib.LATE_PM, av, shift)
        S, n = s40lib.mask_S(S, cells, 'nan')
        ncell[f'{v}|A{D}|+{shift}'] = n
    return dict(S=S, av=av)


lastb = lambda d: P.IDX[P.IDX < pd.Timestamp(d)][-1]
D0, DE, D19, TA = P.IDX[P.t0], P.IDX[-1], lastb('2019-01-01'), lastb('2024-07-01')
bench = pd.Series(P.E50[P.t0:], index=P.IDX[P.t0:])
CELL = dict(sat_w=0.7, sizing='W10', lock=False)
CAPS = (6e5, 8e5, 1e6, 1.5e6, 2e6)
SLIPS = (0.003, 0.004, 0.005)


def met(nav):
    o = dict(since2019=stlib.cagr(nav, D19, DE), full=stlib.cagr(nav, D0, DE), maxdd_full=stlib.maxdd(nav, D0, DE),
             since2024_07=stlib.cagr(nav, TA, DE), maxdd_2024_07=stlib.maxdd(nav, TA, DE))
    for nm, h in (('HALF', 0.5), ('ZERO', 1.0)):
        n2, dbar = L.haircut(nav.loc[D0:], bench, h)
        o[f'{nm}_since2019'] = stlib.cagr(n2, D19, DE); o[f'{nm}_full'] = stlib.cagr(n2, D0, DE); o[f'{nm}_maxdd_full'] = stlib.maxdd(n2, D0, DE)
        o[f'{nm}_since2024_07'] = stlib.cagr(n2, TA, DE)
    o['dbar_monthly'] = dbar
    return o


res['bench_0050'] = dict(since2019=stlib.cagr(bench, D19, DE), full=stlib.cagr(bench, D0, DE), maxdd_full=stlib.maxdd(bench, D0, DE),
                         since2024_07=stlib.cagr(bench, TA, DE), maxdd_2024_07=stlib.maxdd(bench, TA, DE))
res['meta'] = dict(start=str(D0.date()), since2019_from=str(D19.date()), since2024_07_from=str(TA.date()), end=str(DE.date()))

PATH = {}            # key：v|slip|A{D}|cap|+s
NAVSTD = {}
for v in VARS:
    for D in L.FAM_D:
        for s in range(4):
            sg = s10lib.make_signal_s10(P, U, RT(v, D, s), shift=s)
            S2, ekw = s10lib.variant_S(sg, 'E0')
            for x in SLIPS:
                for cap in CAPS:
                    nav, lg, _ = s10lib.simulate_s10(P, S2, U, cost=L.cost(x), capital=cap, **CELL, **ekw)
                    m = met(nav); m['n_closed'] = int((lg['why'] != 'open_end').sum())
                    PATH[f'{v}|{x}|A{D}|{int(cap/1e4)}萬|+{s}'] = m
                    if D == 10 and s == 0 and cap == 1e6 and x == 0.003:
                        NAVSTD[v] = nav
            del sg, S2
        gc.collect()
    print('variant', v, f'{time.time()-t0:.0f}s', rss(), 'MB', flush=True)

# ---------------------------------------------------------------- 驗收
std = lambda v, x=0.003: PATH[f'{v}|{x}|A10|100萬|+0']
on_keys = lambda v, x: [k for k in PATH if k.startswith(f'{v}|{x}|A10|')]
pdiff = lambda v, f: np.array([PATH[k][f] - PATH[k.replace(v, 'CUR', 1)][f] for k in on_keys(v, 0.003)]) * 100
c = std('CUR')
acc['A0'] = dict(since2019=c['since2019'], full=c['full'], maxdd=c['maxdd_full'], n_closed=c['n_closed'],
                 ok=bool(abs(c['since2019'] - 0.4899) < 0.00025 and abs(c['full'] - 0.3454) < 0.00025 and abs(c['maxdd_full'] + 0.3211) < 0.00025 and c['n_closed'] == 381))
f = std('FPM')
acc['A1_FPM'] = dict(window_2024_07=f['since2024_07'], d_window_pp=(f['since2024_07'] - c['since2024_07']) * 100, since2019=f['since2019'],
                     paths20_window_diff_median_pp=float(np.median(pdiff('FPM', 'since2024_07'))),
                     s39_reported=dict(window=0.61171, since2019=0.48779, paths_median_pp=-0.9298))
acc['A1_FPM']['ok'] = bool(abs(f['since2024_07'] - 0.61171) < 0.0001 and abs(f['since2019'] - 0.48779) < 0.0001 and abs(acc['A1_FPM']['paths20_window_diff_median_pp'] + 0.9298) < 0.01)
a = std('ALL')
acc['A2_ALL'] = dict(since2019=a['since2019'], d_since2019_pp=(a['since2019'] - c['since2019']) * 100,
                     paths20_since2019_diff_median_pp=float(np.median(pdiff('ALL', 'since2019'))), s40_reported=dict(since2019=0.48756, paths_median_pp=-0.3459))
acc['A2_ALL']['ok'] = bool(abs(a['since2019'] - 0.48756) < 0.0001 and abs(acc['A2_ALL']['paths20_since2019_diff_median_pp'] + 0.3459) < 0.01)
dd = np.flatnonzero(NAVSTD['COMB'].to_numpy() != NAVSTD['CUR'].to_numpy())
acc['A3_COMB_first_nav_diff_date'] = str(NAVSTD['COMB'].index[dd[0]].date()) if len(dd) else None
acc['A3_COMB_identical_before_2019_01_signal'] = bool(len(dd) == 0 or NAVSTD['COMB'].index[dd[0]] >= pd.Timestamp('2019-02-01'))
res['acceptance'] = acc
print(json.dumps(acc, ensure_ascii=False, default=str), flush=True)
assert acc['A0']['ok'] and acc['A1_FPM']['ok'] and acc['A2_ALL']['ok'] and acc['A3_A10_av_equal'], '驗收失敗'

# ---------------------------------------------------------------- 彙總
FIELDS = ('since2019', 'full', 'since2024_07', 'maxdd_full', 'maxdd_2024_07', 'HALF_since2019', 'HALF_full', 'HALF_since2024_07', 'HALF_maxdd_full',
          'ZERO_since2019', 'ZERO_full', 'ZERO_maxdd_full', 'dbar_monthly')


def summ(keys):
    o = dict(n_paths=len(keys))
    for fld in FIELDS:
        a_ = np.array([PATH[k][fld] for k in keys])
        o[fld] = dict(median=float(np.median(a_)), p10=float(np.percentile(a_, 10)), p90=float(np.percentile(a_, 90)), min=float(a_.min()), max=float(a_.max()))
    return o


grid = {}
for v in VARS:
    for x in SLIPS:
        for tm in ('ON', 'LATE'):
            keys = [k for k in PATH if k.startswith(f'{v}|{x}|') and (tm == 'LATE' or '|A10|' in k)]
            g = summ(keys)
            g['standard'] = {k2: std(v, x)[k2] for k2 in FIELDS}
            grid[f'{v}|{x}|{tm}'] = g
res['grid'] = grid


def med(v, x, tm, fld='since2019'):
    return grid[f'{v}|{x}|{tm}'][fld]['median']


steps = [('現行回測（除權息修正後，準時、每邊 0.3%）', 'CUR', 0.003, 'ON'), ('＋S39 首次公布營收（FPM）', 'FPM', 0.003, 'ON'),
         ('＋S40 期限前視（ALL）＝COMB', 'COMB', 0.003, 'ON'), ('＋每邊滑價 0.4%', 'COMB', 0.004, 'ON'), ('＋每邊滑價 0.5%', 'COMB', 0.005, 'ON'),
         ('COMB＋有時晚 0～4 天＋0.3%', 'COMB', 0.003, 'LATE'), ('COMB＋有時晚 0～4 天＋0.4%', 'COMB', 0.004, 'LATE'), ('COMB＋有時晚 0～4 天＋0.5%', 'COMB', 0.005, 'LATE')]
stack = []
for nm, v, x, tm in steps:
    g = grid[f'{v}|{x}|{tm}']
    stack.append(dict(step=nm, key=f'{v}|{x}|{tm}', since2019_median=g['since2019']['median'], full_median=g['full']['median'], since2024_07_median=g['since2024_07']['median'],
                      maxdd_full_median=g['maxdd_full']['median'], since2019_standard=g['standard']['since2019'], maxdd_standard=g['standard']['maxdd_full'],
                      d_vs_CUR_since2019_median_pp=(g['since2019']['median'] - med('CUR', 0.003, 'ON')) * 100,
                      HALF_since2019_median=g['HALF_since2019']['median']))
res['stack'] = stack
# 可加性：FPM、ALL 各自對 CUR 的配對差和 COMB 的配對差（20 條、2019 起）
add = {}
for fld in ('since2019', 'since2024_07', 'full'):
    dF, dA, dC = pdiff('FPM', fld), pdiff('ALL', fld), pdiff('COMB', fld)
    add[fld] = dict(FPM_median_pp=float(np.median(dF)), ALL_median_pp=float(np.median(dA)), COMB_median_pp=float(np.median(dC)),
                    sum_of_medians_pp=float(np.median(dF) + np.median(dA)), COMB_minus_sum_paired_median_pp=float(np.median(dC - dF - dA)),
                    COMB_min_pp=float(dC.min()), COMB_max_pp=float(dC.max()), COMB_n_lower=int((dC < -1e-9).sum()))
res['additivity_paths20'] = add
# 配對：滑價、晚下單效果（COMB）
pe = {}
for x in (0.004, 0.005):
    d = np.array([PATH[k.replace('|0.003|', f'|{x}|')]['since2019'] - PATH[k]['since2019'] for k in PATH if k.startswith('COMB|0.003|')]) * 100
    pe[f'slip_{x}_vs_0.003_since2019_pp'] = dict(median=float(np.median(d)), min=float(d.min()), max=float(d.max()))
for D in (11, 12, 13, 15):
    d = np.array([PATH[k.replace('|A10|', f'|A{D}|')]['since2019'] - PATH[k]['since2019'] for k in PATH if k.startswith('COMB|0.003|A10|')]) * 100
    pe[f'A{D}_vs_A10_since2019_pp'] = dict(median=float(np.median(d)), min=float(d.min()), max=float(d.max()))
for v in ('CUR', 'COMB'):
    pe[f'{v}_LATE_minus_ON_median_pp'] = (med(v, 0.003, 'LATE') - med(v, 0.003, 'ON')) * 100
res['paired_effects'] = pe
res['n_cells_masked'] = {k: n for k, n in ncell.items() if k.endswith('+0')}
res['paths'] = {k: {f2: round(v2, 6) if isinstance(v2, float) else v2 for f2, v2 in m.items()} for k, m in PATH.items()}
res['run_runtime_s'] = round(time.time() - t0, 1); res['run_maxrss_mb'] = rss()
(H / 'results.json').write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str))
for r in stack:
    print(r['step'], f"{r['since2019_median']*100:.2f}", f"{r['maxdd_full_median']*100:.2f}", f"{r['d_vs_CUR_since2019_median_pp']:+.2f}", f"HALF {r['HALF_since2019_median']*100:.2f}")
print(json.dumps(dict(add=add, pe=pe, bench=res['bench_0050']), ensure_ascii=False, indent=1))
print('done', res['run_runtime_s'], 's', res['run_maxrss_mb'], 'MB')
