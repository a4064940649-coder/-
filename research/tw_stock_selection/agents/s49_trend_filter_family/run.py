#!/usr/bin/env python3
"""S49 進場均線家族檢查（規格 SPEC.json、判定 PREREG.json，寫於任何報酬計算之前）。
用法：python3 -I -B run.py [--quick]（--quick 只跑驗收：F_MA60、訊號逐格、現行格 20 條）
鏈＝S46 run.py／deadline_engine_fix ALL：s20lib→s17lib→s10lib、EXRIGHT_FIX／DEADLINE_FIX／INS15_MASK 全開、e50_fix_suspension、
股池 twse & ~創新板、STD SURGE、F1（前 10 名、RET20 > 10%）、E0、simulate_s10(strict, sat_w=0.7, W10, lock=False)。
唯一改動：買進池用 F_MA_n（n∈{20,40,60,120,240}，或不設均線）排名；E0 出場仍用 L_MA60 前 20 名。
暫存 scratchpad/s49/（結束刪除）。不改任何既有檔案。"""
import sys, json, time, resource, gc, hashlib
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
AG = HERE.parent
TMP = Path('/tmp/claude-0/-home-user/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/s49')
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

MAS = (20, 40, 60, 120, 240, 'NONE')
FMA = {}
for n in MAS:
    if n == 'NONE':
        FMA[n] = np.ones((P.T, P.N), bool)
    else:
        FMA[n] = P.G > stlib.rmean(P.G, n)                 # 同 s2lib：P.F_MA60 = P.G > rmean(P.G, 60)
acc_ma60 = bool(np.array_equal(FMA[60], P.F_MA60))
print('載入', f'{time.time()-t0:.0f}s', rss(), 'MB', 'F_MA60 逐格相同', acc_ma60, flush=True)
assert acc_ma60, '驗收失敗：自算 F_MA60 與 P.F_MA60 不同'


def signal_s49(ent, shift, depth=10, mom_lo=0.10, nsel=10):
    """make_signal_s10 的 E0 部分逐行複製；買進池用 F_MA_ent 排名，E0 的 X 仍用 P.F_MA60 排名（L_MA60 前 20 名）。"""
    S, av = RT['S'], RT['av']
    FE = FMA[ent]
    sig = np.zeros((P.T, P.N), bool); rank = np.full((P.T, P.N), np.nan)
    X = np.zeros((P.T, P.N), bool)
    info = []; RQ = {}; below60 = 0; ntot = 0; n_en = []
    for m in S.index:
        k0 = av[m]
        if k0 >= P.T or k0 < 61:
            continue
        s = S.loc[m].to_numpy()
        univ_ = U
        if int((univ_[k0] & np.isfinite(s)).sum()) < 50:
            continue
        k = k0 + shift
        if k >= P.T:
            continue
        s = cslib.ins15_mask_row(P, m, k, s)
        fin = univ_[k] & np.isfinite(s)
        ok_ex = fin & P.F_MA60[k]
        o_ex = np.flatnonzero(ok_ex)[np.argsort(-s[ok_ex], kind='stable')]
        ok_en = fin & FE[k]
        o_en = np.flatnonzero(ok_en)[np.argsort(-s[ok_en], kind='stable')]
        r20 = P.RET20[k]
        pool = o_en[:depth]
        okr = r20[pool] > mom_lo
        tgt = pool[okr][:nsel]
        sig[k, tgt] = True; rank[k, tgt] = s[tgt]
        RQ[k] = set(int(j) for j in tgt)
        X[k, :] = True; X[k, o_ex[:20]] = False
        below60 += int((~P.F_MA60[k, tgt]).sum()); ntot += len(tgt); n_en.append(int(ok_en.sum()))
        info.append(dict(month=str(m), k=int(k), sig_date=str(P.IDX[k].date()), n_ma=int(ok_ex.sum()), n_tgt=int(len(tgt)),
                         targets=[str(P.COLS[j]) for j in tgt]))
    sigrows = np.zeros(P.T, bool); sigrows[[x['k'] for x in info]] = True
    return dict(sig=sig, rank=rank, X=X, SD=None, max_hold=None, info=info, RQ=RQ, sigrows=sigrows,
                _below60=below60, _ntot=ntot, _n_en_mean=float(np.mean(n_en)))


# ---- 驗收 (b)：ent＝60 與 s10lib.make_signal_s10 逐格相同
acc_sig = {}
for s in range(4):
    a = signal_s49(60, s); b = s10lib.make_signal_s10(P, U, RT, shift=s, depth=10, mom_lo=0.10, nsel=10)
    ok = (np.array_equal(a['sig'], b['sig']) and np.array_equal(a['rank'], b['rank'], equal_nan=True) and np.array_equal(a['X'], b['XV']['E0'])
          and a['info'] == b['info'] and np.array_equal(a['sigrows'], b['sigrows']) and a['RQ'] == b['RQ'])
    acc_sig[s] = bool(ok)
    del a, b
gc.collect()
print('訊號逐格', acc_sig, flush=True)
assert all(acc_sig.values()), '驗收失敗：ent=60 訊號與 make_signal_s10 不同'

lastb = lambda d: P.IDX[P.IDX < pd.Timestamp(d)][-1]
TB = P.IDX[-1]; D19 = lastb('2019-01-01'); D24 = lastb('2024-01-01'); D16 = lastb('2016-03-01')
FULL0 = P.IDX[P.t0]
CELL = dict(sat_w=0.7, sizing='W10', lock=False)
CAPS = (6e5, 8e5, 1e6, 1.5e6, 2e6)
CUR = 60
cells = [CUR] if QUICK else list(MAS)
cid = lambda n: f'MA{n}' if n != 'NONE' else 'NONE'
YEARS = list(range(2016, 2027))


def metrics(nav):
    o = dict(since2019=stlib.cagr(nav, D19, TB), full=stlib.cagr(nav, FULL0, TB), maxdd_full=stlib.maxdd(nav, FULL0, TB),
             maxdd_2019=stlib.maxdd(nav, D19, TB), seg19_23=stlib.cagr(nav, D19, D24), seg24=stlib.cagr(nav, D24, TB),
             seg16_18=stlib.cagr(nav, D16, D19), maxdd_16_18=stlib.maxdd(nav, D16, D19))
    for y in YEARS:
        a = lastb(f'{y}-01-01'); b = lastb(f'{y+1}-01-01') if y < 2026 else TB
        o[f'y{y}'] = float(nav.loc[b] / nav.loc[a] - 1)
    return o


R = {}
for n in cells:
    k = cid(n); out = dict(entry_ma=n, paths={})
    for s in range(4):
        sg = signal_s49(n, s)
        if s == 0:
            ntg = np.array([x['n_tgt'] for x in sg['info']])
            first = next((x['month'] for x in sg['info'] if x['n_tgt'] > 0), None)
            out.update(list_mean=float(ntg.mean()), list_zero_months=int((ntg == 0).sum()), n_signal_months=int(len(ntg)),
                       first_nonempty_month=first, lma_entry_mean=sg['_n_en_mean'],
                       buys_below_ma60_frac=(sg['_below60'] / sg['_ntot']) if sg['_ntot'] else None, list_total=int(sg['_ntot']))
        for cap in CAPS:
            nav, lg, st = s10lib.simulate_s10(P, sg, U, cost=COSTS['strict'], capital=cap, **CELL)
            m = metrics(nav); m['avg_npos'] = float(st['avg_npos']); m['n_closed'] = int((lg['why'] != 'open_end').sum()); m['n_buy'] = int(st['n_buy'])
            out['paths'][f'{int(cap/1e4)}萬|+{s}'] = m
            if s == 0 and cap == 1e6:
                out['main'] = dict(m, avg_idle_frac=float(st['avg_idle_frac']))
                out['main_exit_reasons'] = {str(a): int(b) for a, b in lg['why'].value_counts().items()}
        del sg
    gc.collect()
    R[k] = out
    print(k, f"{out['main']['since2019']:.4%}", f"med {np.median([v['since2019'] for v in out['paths'].values()]):.4%}", f'{time.time()-t0:.0f}s', rss(), 'MB', flush=True)

# ---- 驗收 (c)：現行格 vs deadline_engine_fix ALL
ref = json.loads((AG / 'deadline_engine_fix' / 'out' / 'results.json').read_text())
cur = R[cid(CUR)]
acc = dict(main_since2019=cur['main']['since2019'], ref_since2019=ref['standard']['ALL']['since2019'],
           main_n_closed=cur['main']['n_closed'], ref_n_closed=ref['standard']['ALL']['n_closed'])
acc['main_bitwise'] = bool(acc['main_since2019'] == acc['ref_since2019'] and acc['main_n_closed'] == acc['ref_n_closed']
                           and cur['main']['full'] == ref['standard']['ALL']['full'] and cur['main']['maxdd_full'] == ref['standard']['ALL']['maxdd_full'])
acc['paths_bitwise'] = bool(all(cur['paths'][p]['since2019'] == ref['paths']['ALL'][p]['since2019'] for p in ref['paths']['ALL']))
acc['paths_median'] = float(np.median([cur['paths'][p]['since2019'] for p in cur['paths']]))
acc['f_ma60_bitwise'] = acc_ma60; acc['signal_bitwise_shift0_3'] = acc_sig
acc['ok'] = acc['main_bitwise'] and acc['paths_bitwise'] and acc_ma60 and all(acc_sig.values())
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
                     n_closed_median=med(o, 'n_closed'), seg19_23_median=med(o, 'seg19_23'), seg24_median=med(o, 'seg24'),
                     seg16_18_median=med(o, 'seg16_18'), maxdd_16_18_median=med(o, 'maxdd_16_18'),
                     years_median={str(y): med(o, f'y{y}') for y in YEARS})
    d = np.array([o['paths'][p]['since2019'] - cur['paths'][p]['since2019'] for p in keys]) * 100
    d1 = np.array([o['paths'][p]['seg19_23'] - cur['paths'][p]['seg19_23'] for p in keys]) * 100
    d2 = np.array([o['paths'][p]['seg24'] - cur['paths'][p]['seg24'] for p in keys]) * 100
    d0 = np.array([o['paths'][p]['seg16_18'] - cur['paths'][p]['seg16_18'] for p in keys]) * 100
    dd = np.array([o['paths'][p]['maxdd_full'] - cur['paths'][p]['maxdd_full'] for p in keys]) * 100
    npos, mpp = int((d > 1e-9).sum()), float(np.median(d))
    lab = '比現行好' if (npos >= 15 and mpp >= 2) else ('比現行差' if (npos <= 5 and mpp <= -2) else '差不多')
    if lab == '比現行好' and np.median(d1) > 0 and np.median(d2) > 0:
        lab = '比現行好且兩段都正'
    o['vs_current'] = dict(n_pos=npos, median_pp=mpp, seg19_23_median_pp=float(np.median(d1)), seg24_median_pp=float(np.median(d2)),
                           seg16_18_median_pp_posthoc=float(np.median(d0)), seg16_18_n_pos_posthoc=int((d0 > 1e-9).sum()),
                           maxdd_full_median_pp=float(np.median(dd)), n_dd_shallower=int((dd > 1e-9).sum()), label=lab if k != cid(CUR) else '現行')


def family(members, name, field='median'):
    meds = {k: R[k]['path'][field] for k in members}
    dds = {k: R[k]['path']['maxdd_full_median'] for k in members}
    fm = float(np.median(list(meds.values())))
    rank = 1 + sum(v > meds[cid(CUR)] for v in meds.values())
    rank_dd = 1 + sum(v > dds[cid(CUR)] for v in dds.values())
    p3 = (meds[cid(CUR)] - fm) * 100
    o = dict(name=name, field=field, n=len(members), current_rank=int(rank), current_rank_dd=int(rank_dd), family_median=fm,
             family_min=float(min(meds.values())), family_max=float(max(meds.values())), best=max(meds, key=meds.get), worst=min(meds, key=meds.get),
             current_minus_median_pp=p3)
    if field == 'median':
        allp = np.array([R[k]['paths'][p]['since2019'] for k in members for p in keys])
        o.update(all_paths_median=float(np.median(allp)), all_paths_p10=float(np.percentile(allp, 10)), n_all_paths=int(len(allp)),
                 verdict='偏樂觀（被挑過）' if p3 >= 3 else '沒被挑過（不必因進場均線打折）',
                 is_best='是' if rank == 1 else ('前段' if rank == 2 else '不是'))
    return o


ALLK = [cid(n) for n in MAS]
fams = dict(main6=family(ALLK, '主：進場均線 6 格'),
            ma5=family([cid(n) for n in MAS if n != 'NONE'], '次要：有均線 5 格'),
            posthoc_16_18=family(ALLK, '事後描述：2016-03～2018-12', 'seg16_18_median'),
            posthoc_19_23=family(ALLK, '事後描述：2019～2023', 'seg19_23_median'),
            posthoc_24=family(ALLK, '事後描述：2024 起', 'seg24_median'))
yr = {}
for y in YEARS:
    v = {k: R[k]['path']['years_median'][str(y)] for k in ALLK}
    yr[str(y)] = dict(current_rank=int(1 + sum(x > v[cid(CUR)] for x in v.values())), current_minus_median_pp=(v[cid(CUR)] - float(np.median(list(v.values())))) * 100)
fams['posthoc_years'] = yr
nbd = {k: (R[cid(CUR)]['path']['median'] - R[k]['path']['median']) * 100 for k in ('MA40', 'MA120')}
fams['neighbours'] = dict(current_minus_neighbour_pp=nbd, isolated_peak=bool(all(v >= 2 for v in nbd.values())))
labels = pd.Series({k: R[k]['vs_current']['label'] for k in ALLK}).value_counts().to_dict()

res = dict(spec_sha256=sha(HERE / 'SPEC.json'), prereg_sha256=sha(HERE / 'PREREG.json'), run_sha256=sha(Path(__file__)),
           chain='s20lib→s17lib→s10lib；EXRIGHT_FIX/DEADLINE_FIX/INS15_MASK=True；e50_fix_suspension；twse & ~創新板；STD；signal_s49(進場 F_MA_n、出場 E0=L_MA60 前 20)；simulate_s10(strict,0.7,W10,lock=False)',
           acceptance=acc, families=fams, label_counts=labels, cells=R, path_keys=keys,
           runtime_s=round(time.time() - t0, 1), maxrss_mb=rss())
(HERE / 'results.json').write_text(json.dumps(res, ensure_ascii=False, indent=1))
print(json.dumps({k: v for k, v in fams.items() if k != 'posthoc_years'}, ensure_ascii=False, indent=1))
print('labels', labels)
print('done', res['runtime_s'], 's', res['maxrss_mb'], 'MB')
