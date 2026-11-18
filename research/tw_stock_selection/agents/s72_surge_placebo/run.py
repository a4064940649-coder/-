#!/usr/bin/env python3
"""S72 安慰劑檢定：現行策略的超額有多少來自「月營收驚奇 SURGE」排名、有多少來自「站上季線＋近 20 日漲 >10%＋集中持股」這個殼。
規格 SPEC.json、判定 PREREG.json（寫於任何安慰劑報酬計算之前；本檔啟動時核對 sha256）。
用法：python3 -I -B run.py --quick      只跑驗收（現行格：A1 訊號矩陣逐格相同、A2 逐位重現 48.76%／381 筆）＋計時，不算任何安慰劑
      python3 -I -B run.py [--workers N] 全部（預設單行程）（可中斷續跑：逐工作結果寫在 scratchpad/s72/ck.jsonl）
鏈＝agents/s48_exit_rule_family/run.py（＝deadline_engine_fix ALL＝official_nav E0）：s20lib→s17lib→s10lib，EXRIGHT_FIX／DEADLINE_FIX／INS15_MASK 全開，
e50_fix_suspension，股池 twse & ~創新板，STD SURGE。訊號由本檔 build()（s10lib.make_signal_s10 的逐行複製，只把排序分數換掉）產生，
交給 s10lib.simulate_s10(strict, sat_w=0.7, W10, lock=False)。不改任何既有檔案。暫存 scratchpad/s72/（結束刪除）。"""
import sys, json, time, resource, gc, hashlib, os
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
AG = HERE.parent
TMP = Path('/tmp/claude-0/-home-user/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/s72')
CACHE = TMP / 'cache'
CACHE.mkdir(parents=True, exist_ok=True)
QUICK = '--quick' in sys.argv
NW = int(sys.argv[sys.argv.index('--workers') + 1]) if '--workers' in sys.argv else 1
t0 = time.time()
rss = lambda: round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
if not QUICK:
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
INNOV, _ = s4lib.innov_panel(P)
U = stlib.universe(P, 'twse') & ~INNOV
del INNOV
F = s17lib.rev_family(P, defs=['STD'])                     # 開關關閉下建（同 run_engine／S46／S48）
s17lib.e50_fix_suspension(P)
for a in ('H', 'L', 'VOL', 'CHG', 'HF', 'VOLF', 'adjH', 'HI240', 'MA60', 'nbar', 'F_HI', 'IDXC', 'AO', 'listed', 'NEARHI', 'V60', 'liq'):
    if hasattr(P, a):
        delattr(P, a)
S0 = F['S']['STD']; months = list(S0.index)
Rv = F['Rv']
with np.errstate(divide='ignore', invalid='ignore'):
    YOY = (Rv / Rv.shift(12) - 1).replace([np.inf, -np.inf], np.nan)
YOY[YOY.index.year == 2013] = np.nan                       # IFRS 斷點，同 s17lib
YOY = YOY.reindex(S0.index).to_numpy()
SARR = S0.to_numpy()
del F, Rv; gc.collect()
cslib.DEADLINE_FIX = True; cslib.INS15_MASK = True         # 正式設定：三個全開
AV = cslib.rev_avail_days(P, months, True)
NM = len(months)
print('載入', f'{time.time()-t0:.0f}s', rss(), 'MB', 'T0', str(P.IDX[P.t0].date()), 'IDX0', str(P.IDX[0].date()), 'NM', NM, flush=True)

lastb = lambda d: P.IDX[P.IDX < pd.Timestamp(d)][-1]
TB = P.IDX[-1]; D16 = lastb('2016-01-01'); D19 = lastb('2019-01-01'); D24 = lastb('2024-01-01')
FULL0 = P.IDX[P.t0]
E50 = pd.Series(P.E50, index=P.IDX)
YEARS_FULL = (TB - FULL0).days / 365.25
CELL = dict(sat_w=0.7, sizing='W10', lock=False)
CAPS = (6e5, 8e5, 1e6, 1.5e6, 2e6)
G = P.G


def mom(k, a, b):
    """還原收盤 G[k−a] ÷ G[k−b] − 1（列數不足＝缺值）。"""
    if k - b < 0:
        return np.full(P.N, np.nan)
    with np.errstate(divide='ignore', invalid='ignore'):
        r = G[k - a] / G[k - b] - 1
    r[~np.isfinite(r)] = np.nan
    return r


def rand_scores(kind, draw):
    """P1：每月獨立均勻亂數；P2：每檔股票的分數持續 3 個月才重抽（各股票錯開相位）。回傳 (NM, N)。"""
    if kind == 'P1':
        return np.random.default_rng([72, 1, draw]).random((NM, P.N))
    rng = np.random.default_rng([72, 2, draw])
    ph = rng.integers(0, 3, P.N)
    Q = rng.random((NM, P.N))
    for mi in range(1, NM):
        keep = (mi + ph) % 3 != 0
        Q[mi, keep] = Q[mi - 1, keep]
    return Q


def scorefn(kind, draw):
    if kind in ('P1', 'P2'):
        Q = rand_scores(kind, draw)
        return lambda mi, k, s: Q[mi]
    return {'CUR': lambda mi, k, s: s,
            'P3_RET20': lambda mi, k, s: P.RET20[k],
            'P3_RET60': lambda mi, k, s: mom(k, 0, 60),
            'P3_MOM12_1': lambda mi, k, s: mom(k, 21, 252),
            'P4_REVERSE': lambda mi, k, s: -s,
            'P5_YOY': lambda mi, k, s: YOY[mi]}[kind]


def build(shift, fn, depth=10, mom_lo=0.10, nsel=10):
    """s10lib.make_signal_s10 的逐行複製（只保留 E0），排序分數 s 換成 q＝fn(月序, 訊號列, 遮罩後的 SURGE)。
    合格名單仍要求「股池內、SURGE 有值（INS15 遮罩後）」，另要求 q 有值；進場前 10 名、出場前 20 名、買進順序（rank）都用 q。"""
    sig = np.zeros((P.T, P.N), bool); rank = np.full((P.T, P.N), np.nan); X = np.zeros((P.T, P.N), bool)
    info = []
    for mi, m in enumerate(S0.index):
        k0 = AV[m]
        if k0 >= P.T or k0 < 61:
            continue
        s = SARR[mi]
        if int((U[k0] & np.isfinite(s)).sum()) < 50:
            continue
        k = k0 + shift
        if k >= P.T:
            continue
        s = cslib.ins15_mask_row(P, m, k, s)
        q = fn(mi, k, s)
        fin = U[k] & np.isfinite(s) & np.isfinite(q)
        ok_ma = fin & P.F_MA60[k]
        o_ma = np.flatnonzero(ok_ma)[np.argsort(-q[ok_ma], kind='stable')]
        r20 = P.RET20[k]
        pool = o_ma[:depth]
        okr = r20[pool] > mom_lo
        tgt = pool[okr][:nsel]
        sig[k, tgt] = True; rank[k, tgt] = q[tgt]
        X[k, :] = True; X[k, o_ma[:20]] = False
        info.append(dict(month=str(m), k=int(k), sig_date=str(P.IDX[k].date()), n_ma=int(ok_ma.sum()), n_tgt=int(len(tgt)),
                         targets=[str(P.COLS[j]) for j in tgt]))
    sigrows = np.zeros(P.T, bool); sigrows[[x['k'] for x in info]] = True
    return dict(sig=sig, rank=rank, X=X, SD=None, max_hold=None, info=info, RQ={}, sigrows=sigrows)


def metrics(nav, lg, st, sg):
    cl = lg[lg['why'] != 'open_end']
    c = lambda a, b: float(stlib.cagr(nav, a, b)); e = lambda a, b: float(stlib.cagr(E50, a, b))
    o = dict(since2019=c(D19, TB), full=c(FULL0, TB), seg16_18=c(D16, D19), seg19_23=c(D19, D24), seg24=c(D24, TB),
             maxdd_full=float(stlib.maxdd(nav, FULL0, TB)), maxdd_2019=float(stlib.maxdd(nav, D19, TB)),
             n_closed=int(len(cl)), n_buy=int(st['n_buy']), hold_mean=float(cl['hold_days'].mean()) if len(cl) else float('nan'),
             avg_npos=float(st['avg_npos']),
             turnover=float(st['buy_val'] / nav.loc[FULL0:TB].mean() / YEARS_FULL),
             n_tgt_mean=float(np.mean([x['n_tgt'] for x in sg['info']])), n_sigmonths=len(sg['info']))
    o['ex_since2019'] = o['since2019'] - e(D19, TB); o['ex_full'] = o['full'] - e(FULL0, TB)
    o['ex_seg16_18'] = o['seg16_18'] - e(D16, D19); o['ex_seg19_23'] = o['seg19_23'] - e(D19, D24); o['ex_seg24'] = o['seg24'] - e(D24, TB)
    return o


def run_task(task):
    """task＝(kind, draw, all_paths)。回傳 {路徑: 指標}。"""
    kind, draw, allp = task
    fn = scorefn(kind, draw)
    out = {}
    for sft in (range(4) if allp else (0,)):
        sg = build(sft, fn)
        for cap in (CAPS if allp else (1e6,)):
            nav, lg, st = s10lib.simulate_s10(P, sg, U, cost=COSTS['strict'], capital=cap, **CELL)
            out[f'{int(cap/1e4)}萬|+{sft}'] = metrics(nav, lg, st, sg)
            del nav, lg, st
        del sg
    gc.collect()
    return dict(kind=kind, draw=draw, allp=allp, paths=out, rss=rss())


# ---------------------------------------------------------------- 驗收（現行格）
A1 = {}
for sft in range(4):
    ref = s10lib.make_signal_s10(P, U, dict(S=S0, av=AV), shift=sft, depth=10, mom_lo=0.10, nsel=10)
    mine = build(sft, scorefn('CUR', 0))
    A1[sft] = bool(np.array_equal(ref['sig'], mine['sig']) and np.array_equal(ref['rank'], mine['rank'], equal_nan=True)
                   and np.array_equal(ref['XV']['E0'], mine['X']) and [x['targets'] for x in ref['info']] == [x['targets'] for x in mine['info']])
    assert A1[sft], f'驗收 A1 失敗（位移 {sft}）'
    del ref, mine
gc.collect()
ts = time.time()
CUR = run_task(('CUR', 0, False))
t_one = time.time() - ts
refj = json.loads((AG / 'deadline_engine_fix' / 'out' / 'results.json').read_text())
cm = CUR['paths']['100萬|+0']
acc = dict(A1_signal_bitwise=A1, main_since2019=cm['since2019'], ref_since2019=refj['standard']['ALL']['since2019'],
           main_n_closed=cm['n_closed'], ref_n_closed=refj['standard']['ALL']['n_closed'],
           e50_since2019=float(stlib.cagr(E50, D19, TB)), e50_full=float(stlib.cagr(E50, FULL0, TB)),
           e50_seg16_18=float(stlib.cagr(E50, D16, D19)), e50_seg19_23=float(stlib.cagr(E50, D19, D24)), e50_seg24=float(stlib.cagr(E50, D24, TB)))
acc['main_bitwise'] = bool(acc['main_since2019'] == acc['ref_since2019'] and acc['main_n_closed'] == acc['ref_n_closed'] == 381
                           and cm['full'] == refj['standard']['ALL']['full'] and cm['maxdd_full'] == refj['standard']['ALL']['maxdd_full'])
print('驗收', json.dumps(acc, ensure_ascii=False), f'單次（訊號＋模擬）{t_one:.1f}s', rss(), 'MB', flush=True)
assert acc['main_bitwise'] and abs(acc['main_since2019'] - 0.4876) < 2e-4, '驗收 A2 失敗'
if QUICK:
    print('quick ok', f'{time.time()-t0:.0f}s', rss(), 'MB'); sys.exit(0)

# ---------------------------------------------------------------- 工作清單
N1, N1P, N2 = 300, 30, 300
DET = ['CUR', 'P3_RET20', 'P3_RET60', 'P3_MOM12_1', 'P4_REVERSE', 'P5_YOY']
tasks = [(k, 0, True) for k in DET]
tasks += [('P1', i, True) for i in range(N1P)] + [('P1', i, False) for i in range(N1P, N1)]
tasks += [('P2', i, False) for i in range(N2)]
CK = TMP / 'ck.jsonl'
done = {}
if CK.exists():
    for ln in CK.read_text().splitlines():
        try:
            r = json.loads(ln); done[(r['kind'], r['draw'])] = r
        except Exception:
            pass
todo = [t for t in tasks if (t[0], t[1]) not in done]
print('工作', len(tasks), '已完成', len(done), '待跑', len(todo), flush=True)
if todo:
    import multiprocessing as mp
    pool = mp.get_context('fork').Pool(NW) if NW > 1 else None
    it = pool.imap_unordered(run_task, todo, chunksize=1) if pool else map(run_task, todo)
    with open(CK, 'a') as fh:
        for i, r in enumerate(it):
            done[(r['kind'], r['draw'])] = r
            fh.write(json.dumps(r, ensure_ascii=False) + '\n'); fh.flush()
            if i % 25 == 0:
                print(i, r['kind'], r['draw'], f'{time.time()-t0:.0f}s', '子行程峰值', r['rss'], 'MB', flush=True)
    if pool:
        pool.close(); pool.join()
peak_child = max(r['rss'] for r in done.values())

# ---------------------------------------------------------------- 彙總
STD = '100萬|+0'
FLD = ['since2019', 'full', 'maxdd_full', 'maxdd_2019', 'ex_since2019', 'ex_full', 'seg16_18', 'seg19_23', 'seg24', 'ex_seg16_18', 'ex_seg19_23', 'ex_seg24',
       'n_closed', 'hold_mean', 'avg_npos', 'turnover', 'n_tgt_mean']
cur = done[('CUR', 0)]
assert cur['paths'][STD]['since2019'] == cm['since2019']
keys = list(cur['paths'])
assert len(keys) == 20 and all(cur['paths'][p]['since2019'] == refj['paths']['ALL'][p]['since2019'] for p in refj['paths']['ALL']), '現行 20 條與 deadline_engine_fix 不同'
pmed = lambda r, f: float(np.median([r['paths'][p][f] for p in keys]))


def dist(vals, curv):
    a = np.asarray(vals, float); n = len(a)
    return dict(n=n, median=float(np.median(a)), mean=float(a.mean()), p5=float(np.percentile(a, 5)), p95=float(np.percentile(a, 95)),
                min=float(a.min()), max=float(a.max()), current=float(curv), current_minus_median_pp=float((curv - np.median(a)) * 100),
                current_percentile=float((a < curv).mean() * 100), n_ge_current=int((a >= curv).sum()), p_one_sided=float((1 + (a >= curv).sum()) / (n + 1)))


def verdict(d):
    if d['p_one_sided'] < 0.05 and d['current_minus_median_pp'] >= 5:
        return 'SURGE 有明確貢獻'
    if d['p_one_sided'] < 0.05:
        return '有貢獻但小'
    return '看不出 SURGE 的貢獻'


R = dict(acceptance=acc, n_draws=dict(P1_std=N1, P1_paths20=N1P, P2_std=N2), std_path=STD)
R['current'] = dict(std={f: cur['paths'][STD][f] for f in FLD}, paths20_median={f: pmed(cur, f) for f in FLD},
                    paths20_since2019={p: cur['paths'][p]['since2019'] for p in keys})
for kind, n in (('P1', N1), ('P2', N2)):
    rs = [done[(kind, i)] for i in range(n)]
    R[kind] = dict(std={f: dist([r['paths'][STD][f] for r in rs], cur['paths'][STD][f]) for f in FLD},
                   draws_std_since2019=[r['paths'][STD]['since2019'] for r in rs],
                   draws_std_full=[r['paths'][STD]['full'] for r in rs])
    R[kind]['verdict_primary'] = verdict(R[kind]['std']['since2019'])
    R[kind]['verdict_full_descriptive'] = verdict(R[kind]['std']['full'])
rs = [done[('P1', i)] for i in range(N1P)]
R['P1']['paths20'] = {f: dist([pmed(r, f) for r in rs], pmed(cur, f)) for f in FLD}
R['P1']['paths20_all600_since2019'] = dist([r['paths'][p]['since2019'] for r in rs for p in keys], pmed(cur, 'since2019'))
R['P1']['paths20_paired_diff_pp'] = dist([np.median([cur['paths'][p]['since2019'] - r['paths'][p]['since2019'] for p in keys]) * 100 for r in rs], 0.0)
R['deterministic'] = {}
p1s = {f: np.array([done[('P1', i)]['paths'][STD][f] for i in range(N1)]) for f in FLD}
p1p = {f: np.array([pmed(done[('P1', i)], f) for i in range(N1P)]) for f in FLD}
for k in DET[1:]:
    r = done[(k, 0)]
    d = np.array([r['paths'][p]['since2019'] - cur['paths'][p]['since2019'] for p in keys]) * 100
    R['deterministic'][k] = dict(std={f: r['paths'][STD][f] for f in FLD}, paths20_median={f: pmed(r, f) for f in FLD},
                                 vs_current=dict(n_pos=int((d > 1e-9).sum()), median_pp=float(np.median(d))),
                                 pct_in_P1_std={f: float((p1s[f] < r['paths'][STD][f]).mean() * 100) for f in ('since2019', 'full', 'seg16_18', 'seg19_23', 'seg24')},
                                 pct_in_P1_paths20={f: float((p1p[f] < pmed(r, f)).mean() * 100) for f in ('since2019', 'full')})
e = acc
R['decomposition_pp'] = dict(
    note='2019 起、標準路徑；殼＝P1 隨機排名中位數；0050＝含息（引擎口徑）',
    e50_since2019=e['e50_since2019'] * 100, shell_P1_median=R['P1']['std']['since2019']['median'] * 100, current=cm['since2019'] * 100,
    shell_minus_0050=(R['P1']['std']['since2019']['median'] - e['e50_since2019']) * 100,
    surge_minus_shell=R['P1']['std']['since2019']['current_minus_median_pp'],
    current_minus_0050=(cm['since2019'] - e['e50_since2019']) * 100,
    full=dict(e50=e['e50_full'] * 100, shell_P1_median=R['P1']['std']['full']['median'] * 100, current=cm['full'] * 100,
              shell_minus_0050=(R['P1']['std']['full']['median'] - e['e50_full']) * 100, surge_minus_shell=R['P1']['std']['full']['current_minus_median_pp']))
R['resources'] = dict(peak_rss_main_mb=rss(), peak_rss_child_mb=peak_child, workers=NW, seconds=round(time.time() - t0))
R['code_sha256'] = dict(run=sha(HERE / 'run.py'), SPEC=sha(HERE / 'SPEC.json'), PREREG=sha(HERE / 'PREREG.json'))
(HERE / 'results.json').write_text(json.dumps(R, ensure_ascii=False, indent=1))
print('完成', f'{time.time()-t0:.0f}s', '主行程峰值', rss(), 'MB 子行程峰值', peak_child, 'MB', flush=True)
print(json.dumps(dict(P1=R['P1']['std']['since2019'], verdict=R['P1']['verdict_primary'], decomposition=R['decomposition_pp']), ensure_ascii=False, indent=1))
