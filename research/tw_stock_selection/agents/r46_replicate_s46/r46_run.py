#!/usr/bin/env python3
"""R46：獨立重現 S46「進場動能門檻家族」（未讀 S46 的 run.py／posthoc.py，只依 SPEC.json／PREREG.json 與共用引擎原始碼自寫）。
引擎鏈照 agents/deadline_engine_fix/run_engine.py 的 ALL：s20lib→s17lib→s10lib；EXRIGHT_FIX 在 load 前開；
SURGE（s17lib.rev_family STD）在 DEADLINE_FIX 關閉下建；之後 DEADLINE_FIX＝True 用 cslib.rev_avail_days(P, months, True) 算可得日；
INS15_MASK＝True（make_signal_s10 讀）；s17lib.e50_fix_suspension；股池＝stlib.universe(P,'twse') & ~創新板。
家族：g∈{0,5,10,15,20}% × L∈{10,20,40} × D∈{8,10,12}；make_signal_s10(depth=D, mom_lo=g, nsel=10)；
L≠20 時暫把 P.RET20 換成 G/shift(G,L)−1（同 s3lib 寫法），算完換回。E0、W10_70、嚴成本；20 條路徑＝{60,80,100,150,200} 萬 × 位移 0～3。
用法：python3 -I -B r46_run.py   輸出：results.json（同資料夾）。暫存：scratchpad/r46/cache。"""
import sys, json, time, resource, gc
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
AG = HERE.parent
CACHE = Path('/tmp/claude-0/-home-user/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/r46/cache')
CACHE.mkdir(parents=True, exist_ok=True)
t0 = time.time()
rss = lambda: round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
sys.path.insert(0, str(AG / 's20_persistent_growth'))
import s20lib                                              # noqa: E402
from s20lib import s17lib, s10lib, s9lib, s8lib, s7lib, s6lib, s5lib, s4lib, s3lib, s2lib, stlib, cslib, COSTS   # noqa: E402
for mod, attr in ((s20lib, 'S20'), (s17lib, 'S17'), (s10lib, 'S10'), (s9lib, 'S9'), (s8lib, 'S8'), (s7lib, 'S7'), (s6lib, 'S6'), (s5lib, 'S5'),
                  (s4lib, 'S4'), (s3lib, 'S3'), (s2lib, 'S2'), (stlib, 'S1'), (cslib, 'TMP')):
    setattr(mod, attr, CACHE)
assert cslib.EXRIGHT_FIX is False and cslib.DEADLINE_FIX is False and cslib.INS15_MASK is False
cslib.EXRIGHT_FIX = True

P = s20lib.load()
assert P.exright_fix
assert (P.T, P.N, str(P.IDX[-1].date())) == (2876, 2142, '2026-10-22')
INNOV, _ = s4lib.innov_panel(P)
U = stlib.universe(P, 'twse') & ~INNOV
del INNOV
F = s17lib.rev_family(P, defs=['STD'])                     # 開關關閉下建
s17lib.e50_fix_suspension(P)
S0 = F['S']['STD']; months = list(S0.index); del F
cslib.DEADLINE_FIX = True
av = cslib.rev_avail_days(P, months, True)
cslib.INS15_MASK = True
RT = dict(S=S0, av=av)
# 回看期報酬
RET20_ORIG = P.RET20
RETL = {}
with np.errstate(invalid='ignore', divide='ignore'):
    for L in (10, 20, 40):
        RETL[L] = P.G / cslib.shift(P.G, L) - 1
ret20_bitwise = bool(np.array_equal(RETL[20], RET20_ORIG, equal_nan=True))
RETL[20] = RET20_ORIG
for a in ('H', 'L', 'VOL', 'CHG', 'HF', 'VOLF', 'adjH', 'HI240', 'MA60', 'nbar', 'F_HI', 'IDXC', 'AO', 'listed', 'NEARHI', 'V60', 'liq'):
    if hasattr(P, a):
        delattr(P, a)
gc.collect()
print('載入', f'{time.time()-t0:.0f}s', rss(), 'MB', 'RET20 逐位相同', ret20_bitwise, flush=True)

lastb = lambda d: P.IDX[P.IDX < pd.Timestamp(d)][-1]
D19, D24, TB, D0 = lastb('2019-01-01'), lastb('2024-01-01'), P.IDX[-1], P.IDX[P.t0]
CAPS = (6e5, 8e5, 1e6, 1.5e6, 2e6)
GS, LS, DS = (0.0, 0.05, 0.10, 0.15, 0.20), (10, 20, 40), (8, 10, 12)
CUR = (0.10, 20, 10)
cid = lambda g, L, D: f'g{int(round(g*100))}_L{L}_D{D}'


def signal(g, L, D, s):
    P.RET20 = RETL[L]
    try:
        return s10lib.make_signal_s10(P, U, RT, shift=s, depth=D, mom_lo=g, nsel=10)
    finally:
        P.RET20 = RET20_ORIG


def sim(sg, cap):
    S2, ekw = s10lib.variant_S(sg, 'E0')
    return s10lib.simulate_s10(P, S2, U, cost=COSTS['strict'], capital=cap, sat_w=0.7, sizing='W10', lock=False, **ekw)


cells = {}
for g in GS:
    for L in LS:
        for D in DS:
            c = cid(g, L, D); R = dict(g=g, L=L, D=D, paths={})
            for s in range(4):
                sg = signal(g, L, D, s)
                for cap in CAPS:
                    nav, lg, st = sim(sg, cap)
                    m = dict(since2019=stlib.cagr(nav, D19, TB), maxdd_full=stlib.maxdd(nav, D0, TB), maxdd19=stlib.maxdd(nav, D19, TB),
                             seg1=stlib.cagr(nav, D19, D24), seg2=stlib.cagr(nav, D24, TB))
                    R['paths'][f'{int(cap/1e4)}萬|+{s}'] = m
                    if s == 0 and cap == 1e6:
                        nt = [x['n_tgt'] for x in sg['info']]
                        R['main'] = dict(m, full=stlib.cagr(nav, D0, TB), n_closed=int((lg['why'] != 'open_end').sum()), n_buys=int(len(lg)),
                                         avg_npos=st.get('avg_npos') if isinstance(st, dict) else None,
                                         mean_n_tgt=float(np.mean(nt)), n_zero_months=int(sum(1 for x in nt if x == 0)))
                    del nav, lg, st
                del sg
            v = np.array([x['since2019'] for x in R['paths'].values()])
            R['med20'] = float(np.median(v)); R['p10_20'] = float(np.percentile(v, 10))
            R['med20_maxdd'] = float(np.median([x['maxdd_full'] for x in R['paths'].values()]))
            cells[c] = R
            gc.collect()
            print(c, round(R['main']['since2019']*100, 2), R['main']['n_closed'], round(R['med20']*100, 2), f'{time.time()-t0:.0f}s', rss(), 'MB', flush=True)

# ---- 彙總
cur = cid(*CUR); keys = list(cells[cur]['paths'])
med = {c: cells[c]['med20'] for c in cells}
fam_med = float(np.median(list(med.values())))
order = sorted(med, key=lambda c: -med[c]); rank = order.index(cur) + 1
allp = np.array([cells[c]['paths'][k]['since2019'] for c in cells for k in keys])
lb = [cid(0.10, L, 10) for L in LS]
pair = {}
for c in cells:
    if c == cur:
        continue
    d = np.array([cells[c]['paths'][k]['since2019'] - cells[cur]['paths'][k]['since2019'] for k in keys]) * 100
    d1 = np.array([cells[c]['paths'][k]['seg1'] - cells[cur]['paths'][k]['seg1'] for k in keys]) * 100
    d2 = np.array([cells[c]['paths'][k]['seg2'] - cells[cur]['paths'][k]['seg2'] for k in keys]) * 100
    npos, md = int((d > 0).sum()), float(np.median(d))
    lab = '比現行好' if (npos >= 15 and md >= 2) else ('比現行差' if (npos <= 5 and md <= -2) else '差不多')
    pair[c] = dict(n_pos=npos, med_pp=md, seg1_med_pp=float(np.median(d1)), seg2_med_pp=float(np.median(d2)), label=lab)
summary = dict(
    current_main_since2019=cells[cur]['main']['since2019'], current_n_closed=cells[cur]['main']['n_closed'], current_med20=med[cur],
    family_median_of_med20=fam_med, family_min=min(med.values()), family_max=max(med.values()), best_cell=order[0],
    current_rank=rank, P3_pp=(med[cur] - fam_med) * 100,
    all900_median=float(np.median(allp)), all900_p10=float(np.percentile(allp, 10)),
    all900_p10_methods={m: float(np.percentile(allp, 10, method=m)) for m in ('linear', 'lower', 'higher', 'nearest', 'midpoint')},
    lookback_line_cells=lb, lookback_line_median=float(np.median([med[c] for c in lb])),
    n_better=sum(1 for x in pair.values() if x['label'] == '比現行好'), n_worse=sum(1 for x in pair.values() if x['label'] == '比現行差'),
    n_cells_positive_majority=sum(1 for x in pair.values() if x['n_pos'] > 10),
    ret20_rebuild_bitwise=ret20_bitwise)
acc_ref = json.loads((AG / 'deadline_engine_fix' / 'out' / 'results.json').read_text())
ra = acc_ref['standard']['ALL']; rp = acc_ref['paths']['ALL']
summary['acceptance_vs_deadline_engine_fix_ALL'] = dict(
    main_bitwise=cells[cur]['main']['since2019'] == ra['since2019'], n_closed_eq=cells[cur]['main']['n_closed'] == ra['n_closed'],
    paths_bitwise=all(cells[cur]['paths'][k]['since2019'] == rp[k]['since2019'] for k in keys))
res = dict(note='R46 獨立重現 S46（不看 S46 程式碼）', summary=summary, pairs=pair,
           cells={c: dict(g=v['g'], L=v['L'], D=v['D'], med20=v['med20'], p10_20=v['p10_20'], med20_maxdd=v['med20_maxdd'], main=v['main'],
                          paths={k: x['since2019'] for k, x in v['paths'].items()}) for c, v in cells.items()},
           runtime_s=round(time.time() - t0), maxrss_mb=rss())
(HERE / 'results.json').write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str))
print(json.dumps(summary, ensure_ascii=False, indent=1, default=str))
print('done', res['runtime_s'], 's', res['maxrss_mb'], 'MB')
