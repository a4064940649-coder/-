#!/usr/bin/env python3
"""S46 進場動能門檻 × 回看期 × 深度 家族檢查（規格 SPEC.json、判定 PREREG.json，寫於任何報酬計算之前）。
用法：python3 -I -B run.py [--quick]（--quick 只跑現行格驗收）
鏈＝agents/deadline_engine_fix/run_engine.py 的 ALL：s20lib→s17lib→s10lib、EXRIGHT_FIX／DEADLINE_FIX／INS15_MASK 全開、e50_fix_suspension、
股池 twse & ~創新板、STD SURGE、make_signal_s10(depth, mom_lo, nsel=10)、E0、simulate_s10(strict, sat_w=0.7, W10, lock=False)。
回看期 L ≠ 20：產生訊號時把 P.RET20 暫時換成 RET_L，產生完換回。暫存 scratchpad/s46/（結束刪除）。不改任何既有檔案。"""
import sys, json, time, resource, gc, itertools, hashlib
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
AG = HERE.parent
TMP = Path('/tmp/claude-0/-home-user--/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/s46')
CACHE = TMP / 'cache'
CACHE.mkdir(parents=True, exist_ok=True)
QUICK = '--quick' in sys.argv
t0 = time.time()
rss = lambda: round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()

sys.path.insert(0, str(AG / 's20_persistent_growth'))
import s20lib                                              # noqa: E402
from s20lib import s17lib, s10lib, s9lib, s8lib, s7lib, s6lib, s5lib, s4lib, s3lib, s2lib, stlib, cslib, COSTS   # noqa: E402
for mod, attr in ((s20lib, 'S20'), (s17lib, 'S17'), (s10lib, 'S10'), (s9lib, 'S9'), (s8lib, 'S8'), (s7lib, 'S7'), (s6lib, 'S6'), (s5lib, 'S5'),
                  (s4lib, 'S4'), (s3lib, 'S3'), (s2lib, 'S2'), (stlib, 'S1'), (cslib, 'TMP')):
    setattr(mod, attr, CACHE)
assert cslib.DEADLINE_FIX is False and cslib.INS15_MASK is False
cslib.EXRIGHT_FIX = True

P = s20lib.load()
assert P.exright_fix
assert (P.T, P.N, str(P.IDX[-1].date())) == (2876, 2142, '2026-10-22'), '面板不符'
INNOV, _ = s4lib.innov_panel(P)
U = stlib.universe(P, 'twse') & ~INNOV
del INNOV
F = s17lib.rev_family(P, defs=['STD'])                     # 開關關閉下建（同 run_engine）
s17lib.e50_fix_suspension(P)
for a in ('H', 'L', 'VOL', 'CHG', 'HF', 'VOLF', 'adjH', 'HI240', 'MA60', 'nbar', 'F_HI', 'IDXC', 'AO', 'listed', 'NEARHI', 'V60', 'liq'):
    if hasattr(P, a):
        delattr(P, a)
S0 = F['S']['STD']; months = list(S0.index)
del F; gc.collect()
cslib.DEADLINE_FIX = True; cslib.INS15_MASK = True         # 正式設定：三個全開
AV = cslib.rev_avail_days(P, months, True)
RT = dict(S=S0, av=AV)
RET = {20: P.RET20}
for L in (10, 40):
    RET[L] = P.G / cslib.shift(P.G, L) - 1
print('載入', f'{time.time()-t0:.0f}s', rss(), 'MB', flush=True)

lastb = lambda d: P.IDX[P.IDX < pd.Timestamp(d)][-1]
TB = P.IDX[-1]; D19 = lastb('2019-01-01'); D24 = lastb('2024-01-01')
FULL0 = P.IDX[P.t0]
CELL = dict(sat_w=0.7, sizing='W10', lock=False)
CAPS = (6e5, 8e5, 1e6, 1.5e6, 2e6)
GATES = (0.0, 0.05, 0.10, 0.15, 0.20); LBS = (10, 20, 40); DEPTHS = (8, 10, 12)
CUR = (0.10, 20, 10)
cells = [CUR] if QUICK else list(itertools.product(GATES, LBS, DEPTHS))
cid = lambda c: f'g{int(round(c[0]*100))}_L{c[1]}_D{c[2]}'


def signal(c, shift):
    g, L, D = c
    orig = P.RET20
    P.RET20 = RET[L]
    try:
        sg = s10lib.make_signal_s10(P, U, RT, shift=shift, depth=D, mom_lo=g, nsel=10)
    finally:
        P.RET20 = orig
    return sg


def metrics(nav):
    return dict(since2019=stlib.cagr(nav, D19, TB), full=stlib.cagr(nav, FULL0, TB), maxdd_full=stlib.maxdd(nav, FULL0, TB),
                maxdd_2019=stlib.maxdd(nav, D19, TB), seg19_23=stlib.cagr(nav, D19, D24), seg24=stlib.cagr(nav, D24, TB))


R = {}
for c in cells:
    k = cid(c); out = dict(gate=c[0], lookback=c[1], depth=c[2], paths={})
    for s in range(4):
        sg = signal(c, s)
        S2, ekw = s10lib.variant_S(sg, 'E0')
        if s == 0:
            ntg = np.array([x['n_tgt'] for x in sg['info']])
            out['list_mean'] = float(ntg.mean()); out['list_zero_months'] = int((ntg == 0).sum()); out['n_signal_months'] = int(len(ntg))
        for cap in CAPS:
            nav, lg, st = s10lib.simulate_s10(P, S2, U, cost=COSTS['strict'], capital=cap, **CELL, **ekw)
            m = metrics(nav); m['avg_npos'] = float(st['avg_npos']); m['n_closed'] = int((lg['why'] != 'open_end').sum()); m['n_buy'] = int(st['n_buy'])
            out['paths'][f'{int(cap/1e4)}萬|+{s}'] = m
            if s == 0 and cap == 1e6:
                out['main'] = dict(m, avg_idle_frac=float(st['avg_idle_frac']))
        del sg, S2
    gc.collect()
    R[k] = out
    print(k, f"{out['main']['since2019']:.4%}", f"med {np.median([v['since2019'] for v in out['paths'].values()]):.4%}", f'{time.time()-t0:.0f}s', rss(), 'MB', flush=True)

# ---- 驗收：現行格 vs deadline_engine_fix ALL
ref = json.loads((AG / 'deadline_engine_fix' / 'out' / 'results.json').read_text())
cur = R[cid(CUR)]
acc = dict(main_since2019=cur['main']['since2019'], ref_since2019=ref['standard']['ALL']['since2019'],
           main_n_closed=cur['main']['n_closed'], ref_n_closed=ref['standard']['ALL']['n_closed'])
acc['main_bitwise'] = bool(acc['main_since2019'] == acc['ref_since2019'] and acc['main_n_closed'] == acc['ref_n_closed']
                           and cur['main']['full'] == ref['standard']['ALL']['full'] and cur['main']['maxdd_full'] == ref['standard']['ALL']['maxdd_full'])
acc['paths_bitwise'] = bool(all(cur['paths'][p]['since2019'] == ref['paths']['ALL'][p]['since2019'] for p in ref['paths']['ALL']))
acc['ok'] = acc['main_bitwise'] and acc['paths_bitwise']
print('驗收', json.dumps(acc, ensure_ascii=False), flush=True)
assert acc['ok'], '驗收失敗：現行格與 deadline_engine_fix ALL 不同'
if QUICK:
    print('quick ok', f'{time.time()-t0:.0f}s', rss(), 'MB'); sys.exit(0)

# ---- 彙總
keys = list(cur['paths'])
for k, o in R.items():
    a = np.array([o['paths'][p]['since2019'] for p in keys])
    o['path'] = dict(median=float(np.median(a)), min=float(a.min()), max=float(a.max()), p10=float(np.percentile(a, 10)),
                     maxdd_full_median=float(np.median([o['paths'][p]['maxdd_full'] for p in keys])),
                     maxdd_2019_median=float(np.median([o['paths'][p]['maxdd_2019'] for p in keys])),
                     avg_npos_median=float(np.median([o['paths'][p]['avg_npos'] for p in keys])),
                     n_closed_median=float(np.median([o['paths'][p]['n_closed'] for p in keys])))
    d = np.array([o['paths'][p]['since2019'] - cur['paths'][p]['since2019'] for p in keys]) * 100
    d1 = np.array([o['paths'][p]['seg19_23'] - cur['paths'][p]['seg19_23'] for p in keys]) * 100
    d2 = np.array([o['paths'][p]['seg24'] - cur['paths'][p]['seg24'] for p in keys]) * 100
    dd = np.array([o['paths'][p]['maxdd_full'] - cur['paths'][p]['maxdd_full'] for p in keys]) * 100
    npos, med = int((d > 1e-9).sum()), float(np.median(d))
    lab = '比現行好' if (npos >= 15 and med >= 2) else ('比現行差' if (npos <= 5 and med <= -2) else '差不多')
    if lab == '比現行好' and np.median(d1) > 0 and np.median(d2) > 0:
        lab = '比現行好且兩段都正'
    o['vs_current'] = dict(n_pos=npos, median_pp=med, seg19_23_median_pp=float(np.median(d1)), seg24_median_pp=float(np.median(d2)),
                           maxdd_full_median_pp=float(np.median(dd)), n_dd_shallower=int((dd > 1e-9).sum()), label=lab if k != cid(CUR) else '現行')


def family(members, name):
    meds = {k: R[k]['path']['median'] for k in members}
    dds = {k: R[k]['path']['maxdd_full_median'] for k in members}
    fm = float(np.median(list(meds.values())))
    rank = 1 + sum(v > meds[cid(CUR)] for v in meds.values())
    rank_dd = 1 + sum(v > dds[cid(CUR)] for v in dds.values())
    allp = np.array([R[k]['paths'][p]['since2019'] for k in members for p in keys])
    p3 = (meds[cid(CUR)] - fm) * 100
    n = len(members)
    o = dict(name=name, n=n, current_rank=int(rank), current_rank_dd=int(rank_dd), family_median=fm, family_min=float(min(meds.values())),
             family_max=float(max(meds.values())), best=max(meds, key=meds.get), worst=min(meds, key=meds.get),
             all_paths_median=float(np.median(allp)), all_paths_p10=float(np.percentile(allp, 10)), n_all_paths=int(len(allp)),
             current_minus_median_pp=p3, verdict='偏樂觀（被挑過）' if p3 >= 3 else '沒被挑過（不必因門檻／深度打折）')
    if n == 45:
        o['is_best'] = '是' if rank == 1 else ('前段' if rank <= 11 else '不是')
    return o


ALLK = [cid(c) for c in itertools.product(GATES, LBS, DEPTHS)]
fams = dict(
    main45=family(ALLK, '主：門檻 × 回看期 × 深度 45 格'),
    A15=family([cid((g, L, 10)) for g in GATES for L in LBS], '次要 A：門檻 × 回看期 15 格（D＝10）'),
    B3=family([cid((0.10, 20, D)) for D in DEPTHS], '次要 B：深度 3 格（g＝10%、L＝20）'),
    gate5=family([cid((g, 20, 10)) for g in GATES], '次要：門檻線 5 格（L＝20、D＝10）'),
    lb3=family([cid((0.10, L, 10)) for L in LBS], '次要：回看期線 3 格（g＝10%、D＝10）'),
)
nb = [cid(c) for c in ((0.05, 20, 10), (0.15, 20, 10), (0.10, 10, 10), (0.10, 40, 10), (0.10, 20, 8), (0.10, 20, 12))]
nbd = {k: (R[cid(CUR)]['path']['median'] - R[k]['path']['median']) * 100 for k in nb}
fams['neighbours'] = dict(current_minus_neighbour_pp=nbd, isolated_peak=bool(all(v >= 2 for v in nbd.values())))
labels = pd.Series({k: R[k]['vs_current']['label'] for k in ALLK}).value_counts().to_dict()

res = dict(spec_sha256=sha(HERE / 'SPEC.json'), prereg_sha256=sha(HERE / 'PREREG.json'), run_sha256=sha(Path(__file__)),
           chain='s20lib→s17lib→s10lib；EXRIGHT_FIX/DEADLINE_FIX/INS15_MASK=True；e50_fix_suspension；twse & ~創新板；STD；make_signal_s10(depth,mom_lo,nsel=10)；E0；simulate_s10(strict,0.7,W10,lock=False)',
           acceptance=acc, families=fams, label_counts=labels, cells=R, path_keys=keys,
           runtime_s=round(time.time() - t0, 1), maxrss_mb=rss())
(HERE / 'results.json').write_text(json.dumps(res, ensure_ascii=False, indent=1))
print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk not in ('best', 'worst')} for k, v in fams.items()}, ensure_ascii=False, indent=1))
print('labels', labels)
print('done', res['runtime_s'], 's', res['maxrss_mb'], 'MB')
