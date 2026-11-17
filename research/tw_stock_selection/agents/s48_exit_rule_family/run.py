#!/usr/bin/env python3
"""S48 出場規則家族：排名門檻 N × 出場均線 M（24 格；規格 SPEC.json、判定 PREREG.json，寫於任何非現行格報酬計算之前）。
用法：python3 -I -B run.py [--quick]（--quick 只跑驗收 A0／A1／A2：現行格）
鏈＝agents/s46_momentum_gate_family/run.py（＝deadline_engine_fix ALL）：s20lib→s17lib→s10lib、EXRIGHT_FIX／DEADLINE_FIX／INS15_MASK 全開、
e50_fix_suspension、股池 twse & ~創新板、STD SURGE、make_signal_s10(depth=10, mom_lo=0.10, nsel=10)（進場 F1 不動）、
出場矩陣 X 由本檔自建（S10 MANO／MA20 的拆法），simulate_s10(strict, sat_w=0.7, W10, lock=False)。
暫存 scratchpad/s48/（結束刪除）。不改任何既有檔案。"""
import sys, json, time, resource, gc, hashlib
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
AG = HERE.parent
TMP = Path('/tmp/claude-0/-home-user/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/s48')
CACHE = TMP / 'cache'
CACHE.mkdir(parents=True, exist_ok=True)
QUICK = '--quick' in sys.argv
t0 = time.time()
rss = lambda: round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
for f in ('SPEC', 'PREREG'):
    assert sha(HERE / f'{f}.json') == (HERE / f'{f}.sha256').read_text().split()[0], f'{f}.json 與登記的 sha256 不符'

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
MAS = (20, 40, 60, 120)
FMA = {}
with np.errstate(invalid='ignore'):
    for M in MAS:
        FMA[M] = P.G > stlib.rmean(P.G, M)
A0 = bool(np.array_equal(FMA[60], P.F_MA60))
print('A0 F_MA60 重建逐格相同', A0, flush=True)
assert A0, '驗收 A0 失敗'
INNOV, _ = s4lib.innov_panel(P)
U = stlib.universe(P, 'twse') & ~INNOV
del INNOV
F = s17lib.rev_family(P, defs=['STD'])                     # 開關關閉下建（同 run_engine／S46）
s17lib.e50_fix_suspension(P)
for a in ('H', 'L', 'VOL', 'CHG', 'HF', 'VOLF', 'adjH', 'HI240', 'MA60', 'nbar', 'F_HI', 'IDXC', 'AO', 'listed', 'NEARHI', 'V60', 'liq'):
    if hasattr(P, a):
        delattr(P, a)
S0 = F['S']['STD']; months = list(S0.index)
del F; gc.collect()
cslib.DEADLINE_FIX = True; cslib.INS15_MASK = True         # 正式設定：三個全開
AV = cslib.rev_avail_days(P, months, True)
RT = dict(S=S0, av=AV)
print('載入', f'{time.time()-t0:.0f}s', rss(), 'MB', flush=True)

lastb = lambda d: P.IDX[P.IDX < pd.Timestamp(d)][-1]
TB = P.IDX[-1]; D19 = lastb('2019-01-01'); D24 = lastb('2024-01-01'); D16 = lastb('2016-03-01')
FULL0 = P.IDX[P.t0]
CELL = dict(sat_w=0.7, sizing='W10', lock=False)
CAPS = (6e5, 8e5, 1e6, 1.5e6, 2e6)
NS = (15, 20, 25, 30, None); MS = (20, 40, 60, 120, None)  # None＝不用
CUR = (20, 60)
ALLC = [(n, m) for n in NS for m in MS if not (n is None and m is None)]
assert len(ALLC) == 24
cells = [CUR] if QUICK else ALLC
cid = lambda c: f"N{c[0] if c[0] else 'none'}_M{c[1] if c[1] else 'none'}"


def build_exits(sg, shift, cl):
    """依 make_signal_s10 的同一個迴圈（同條件、同 INS15 遮罩）重建每個訊號列的 o_ma，回傳 {cell: X}。"""
    X = {c: np.zeros((P.T, P.N), bool) for c in cl}
    ks = []
    for m in RT['S'].index:
        k0 = RT['av'][m]
        if k0 >= P.T or k0 < 61:
            continue
        s = RT['S'].loc[m].to_numpy()
        if int((U[k0] & np.isfinite(s)).sum()) < 50:
            continue
        k = k0 + shift
        if k >= P.T:
            continue
        s = cslib.ins15_mask_row(P, m, k, s)
        fin = U[k] & np.isfinite(s)
        ok_ma = fin & P.F_MA60[k]                          # 進場 L_MA（60MA，不動）
        o_ma = np.flatnonzero(ok_ma)[np.argsort(-s[ok_ma], kind='stable')]
        sn = np.nan_to_num(s, nan=-np.inf)
        ks.append(int(k))
        for (n, mm) in cl:
            keep = fin.copy()
            if n is not None:
                thr = s[o_ma[n - 1]] if len(o_ma) >= n else -np.inf
                keep &= sn >= thr
            if mm is not None:
                keep &= FMA[mm][k]
            X[(n, mm)][k, :] = ~keep
    assert ks == [int(x['k']) for x in sg['info']], '訊號列不符'
    return X


def metrics(nav, lg):
    cl = lg[lg['why'] != 'open_end']
    return dict(since2019=stlib.cagr(nav, D19, TB), full=stlib.cagr(nav, FULL0, TB), maxdd_full=stlib.maxdd(nav, FULL0, TB),
                maxdd_2019=stlib.maxdd(nav, D19, TB), seg19_23=stlib.cagr(nav, D19, D24), seg24=stlib.cagr(nav, D24, TB),
                seg16_18_posthoc=stlib.cagr(nav, D16, D19), n_closed=int(len(cl)),
                hold_mean=float(cl['hold_days'].mean()), hold_median=float(cl['hold_days'].median()))


R = {cid(c): dict(rank_N=c[0], ma_M=c[1], paths={}) for c in cells}
A1 = {}
for sft in range(4):
    sg = s10lib.make_signal_s10(P, U, RT, shift=sft, depth=10, mom_lo=0.10, nsel=10)
    XS = build_exits(sg, sft, cells)
    A1[sft] = bool(np.array_equal(XS[CUR], sg['XV']['E0']))
    assert A1[sft], f'驗收 A1 失敗（位移 {sft}）'
    base = dict(sg); del base['XV']
    for c in cells:
        S2 = dict(base); S2['X'] = XS[c]
        for cap in CAPS:
            nav, lg, st = s10lib.simulate_s10(P, S2, U, cost=COSTS['strict'], capital=cap, **CELL)
            m = metrics(nav, lg); m['avg_npos'] = float(st['avg_npos']); m['n_buy'] = int(st['n_buy'])
            m['why_counts'] = {str(k): int(v) for k, v in lg['why'].value_counts().items()}
            R[cid(c)]['paths'][f'{int(cap/1e4)}萬|+{sft}'] = m
            if sft == 0 and cap == 1e6:
                R[cid(c)]['main'] = m
    del sg, XS, base; gc.collect()
    print('位移', sft, 'A1', A1[sft], f'{time.time()-t0:.0f}s', rss(), 'MB', flush=True)

# ---- 驗收 A2：現行格 vs deadline_engine_fix ALL
ref = json.loads((AG / 'deadline_engine_fix' / 'out' / 'results.json').read_text())
cur = R[cid(CUR)]
acc = dict(A0_fma60=A0, A1_exit_matrix=A1, main_since2019=cur['main']['since2019'], ref_since2019=ref['standard']['ALL']['since2019'],
           main_n_closed=cur['main']['n_closed'], ref_n_closed=ref['standard']['ALL']['n_closed'])
acc['main_bitwise'] = bool(acc['main_since2019'] == acc['ref_since2019'] and acc['main_n_closed'] == acc['ref_n_closed']
                           and cur['main']['full'] == ref['standard']['ALL']['full'] and cur['main']['maxdd_full'] == ref['standard']['ALL']['maxdd_full'])
acc['paths_bitwise'] = bool(all(cur['paths'][p]['since2019'] == ref['paths']['ALL'][p]['since2019'] for p in ref['paths']['ALL']))
acc['paths_median'] = float(np.median([cur['paths'][p]['since2019'] for p in cur['paths']]))
acc['ok'] = acc['main_bitwise'] and acc['paths_bitwise'] and all(A1.values()) and A0
print('驗收', json.dumps(acc, ensure_ascii=False), flush=True)
assert acc['ok'], '驗收失敗：現行格與 deadline_engine_fix ALL 不同'
if QUICK:
    print('quick ok', f'{time.time()-t0:.0f}s', rss(), 'MB'); sys.exit(0)

# ---- 彙總
keys = list(cur['paths'])
med = lambda o, f: float(np.median([o['paths'][p][f] for p in keys]))
for k, o in R.items():
    a = np.array([o['paths'][p]['since2019'] for p in keys])
    o['path'] = dict(median=float(np.median(a)), min=float(a.min()), max=float(a.max()), p10=float(np.percentile(a, 10)),
                     maxdd_full_median=med(o, 'maxdd_full'), maxdd_2019_median=med(o, 'maxdd_2019'), avg_npos_median=med(o, 'avg_npos'),
                     n_closed_median=med(o, 'n_closed'), hold_mean_median=med(o, 'hold_mean'),
                     seg19_23_median=med(o, 'seg19_23'), seg24_median=med(o, 'seg24'), seg16_18_posthoc_median=med(o, 'seg16_18_posthoc'))
    d = np.array([o['paths'][p]['since2019'] - cur['paths'][p]['since2019'] for p in keys]) * 100
    d1 = np.array([o['paths'][p]['seg19_23'] - cur['paths'][p]['seg19_23'] for p in keys]) * 100
    d2 = np.array([o['paths'][p]['seg24'] - cur['paths'][p]['seg24'] for p in keys]) * 100
    dd = np.array([o['paths'][p]['maxdd_full'] - cur['paths'][p]['maxdd_full'] for p in keys]) * 100
    npos, mdn = int((d > 1e-9).sum()), float(np.median(d))
    lab = '比現行好' if (npos >= 15 and mdn >= 2) else ('比現行差' if (npos <= 5 and mdn <= -2) else '差不多')
    if lab == '比現行好' and np.median(d1) > 0 and np.median(d2) > 0:
        lab = '比現行好且兩段都正'
    o['vs_current'] = dict(n_pos=npos, median_pp=mdn, seg19_23_median_pp=float(np.median(d1)), seg24_median_pp=float(np.median(d2)),
                           maxdd_full_median_pp=float(np.median(dd)), n_dd_shallower=int((dd > 1e-9).sum()), label=lab if k != cid(CUR) else '現行')


def family(members, name, top_q=None):
    meds = {k: R[k]['path']['median'] for k in members}
    dds = {k: R[k]['path']['maxdd_full_median'] for k in members}
    fm = float(np.median(list(meds.values())))
    rank = 1 + sum(v > meds[cid(CUR)] for v in meds.values())
    rank_dd = 1 + sum(v > dds[cid(CUR)] for v in dds.values())
    allp = np.array([R[k]['paths'][p]['since2019'] for k in members for p in keys])
    p3 = (meds[cid(CUR)] - fm) * 100
    o = dict(name=name, n=len(members), current_rank=int(rank), current_rank_dd=int(rank_dd), family_median=fm, family_min=float(min(meds.values())),
             family_max=float(max(meds.values())), best=max(meds, key=meds.get), worst=min(meds, key=meds.get),
             all_paths_median=float(np.median(allp)), all_paths_p10=float(np.percentile(allp, 10)), n_all_paths=int(len(allp)),
             current_minus_median_pp=p3, verdict='偏樂觀（被挑過）' if p3 >= 3 else '沒被挑過（不必因出場打折）')
    if top_q:
        o['is_best'] = '是' if rank == 1 else ('前段' if rank <= top_q else '不是')
    return o


ALLK = [cid(c) for c in ALLC]
fams = dict(
    main24=family(ALLK, '主：排名門檻 × 出場均線 24 格', top_q=6),
    A16=family([cid((n, m)) for n in NS[:4] for m in MS[:4]], '次要 A：兩者都用 4 × 4＝16 格'),
    rank5=family([cid((n, 60)) for n in NS], '次要：排名線 5 格（M＝60）'),
    ma5=family([cid((20, m)) for m in MS], '次要：均線線 5 格（N＝20）'),
)
nb = [cid(c) for c in ((15, 60), (25, 60), (20, 40), (20, 120))]
nbd = {k: (R[cid(CUR)]['path']['median'] - R[k]['path']['median']) * 100 for k in nb}
fams['neighbours'] = dict(current_minus_neighbour_pp=nbd, isolated_peak=bool(all(v >= 2 for v in nbd.values())))
labels = pd.Series({k: R[k]['vs_current']['label'] for k in ALLK}).value_counts().to_dict()
# 事後描述：2016-03～2018-12 各格路徑中位數與現行名次（不影響判定）
ph = {k: R[k]['path']['seg16_18_posthoc_median'] for k in ALLK}
s19 = {k: R[k]['path']['seg19_23_median'] for k in ALLK}; s24 = {k: R[k]['path']['seg24_median'] for k in ALLK}
rk = lambda d: int(1 + sum(v > d[cid(CUR)] for v in d.values()))
posthoc = dict(note='事後描述，不影響判定', seg16_18=dict(current_rank=rk(ph), family_median=float(np.median(list(ph.values()))),
                                                 current_minus_median_pp=(ph[cid(CUR)] - float(np.median(list(ph.values())))) * 100),
               seg19_23=dict(current_rank=rk(s19), current_minus_median_pp=(s19[cid(CUR)] - float(np.median(list(s19.values())))) * 100),
               seg24=dict(current_rank=rk(s24), current_minus_median_pp=(s24[cid(CUR)] - float(np.median(list(s24.values())))) * 100))

res = dict(spec_sha256=sha(HERE / 'SPEC.json'), prereg_sha256=sha(HERE / 'PREREG.json'), run_sha256=sha(Path(__file__)),
           chain='s20lib→s17lib→s10lib；EXRIGHT_FIX/DEADLINE_FIX/INS15_MASK=True；e50_fix_suspension；twse & ~創新板；STD；make_signal_s10(10,0.10,10)；自建出場 X(N,M)；simulate_s10(strict,0.7,W10,lock=False)',
           acceptance=acc, families=fams, posthoc=posthoc, label_counts=labels, cells=R, path_keys=keys,
           runtime_s=round(time.time() - t0, 1), maxrss_mb=rss())
(HERE / 'results.json').write_text(json.dumps(res, ensure_ascii=False, indent=1))
print(json.dumps(fams, ensure_ascii=False, indent=1))
print(json.dumps(posthoc, ensure_ascii=False))
print('labels', labels)
print('done', res['runtime_s'], 's', res['maxrss_mb'], 'MB')
