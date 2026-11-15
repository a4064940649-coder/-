#!/usr/bin/env python3
"""S9：換股時效與訊號日定義。9 個訊號日定義 × 2 種成交時點（次日開盤／訊號日收盤）× 2 格（W10_70、WN_70，不鎖定）＝36 個組合，
每個組合跑 100 萬主跑、20 條路徑（5 種資金 × 位移 0～3）、成本加倍、禁買最賺 5 筆。
python3 -I -B run.py → results.json、trades.csv.gz、navs.csv.gz。PREREG.json 的 sha256 必須相符。單行程。"""
import sys, json, hashlib, time, resource, gc
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
import s9lib
from s9lib import s8lib, s4lib, s2lib, stlib, COSTS

H = Path(__file__).resolve().parent
pre = (H / 'PREREG.json').read_bytes()
sha = hashlib.sha256(pre).hexdigest()
assert sha == (H / 'PREREG.sha256').read_text().split()[0], 'PREREG.json 與 PREREG.sha256 不符'
PRE = json.loads(pre)
t_all = time.time()
rss = lambda: round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
P = s9lib.load()
assert (P.T, P.N, str(P.IDX[-1].date())) == (2876, 2142, '2026-10-22'), 'P0 失敗'
U0 = stlib.universe(P, 'twse')
RT = s2lib.rev_tables(P)
INNOV, INNOV_NAMES = s4lib.innov_panel(P)
U = U0 & ~INNOV
del U0, INNOV
for a in ('H', 'L', 'VOL', 'CHG', 'HF', 'VOLF', 'adjH', 'HI240', 'MA60', 'nbar', 'F_HI', 'IDXC', 'AO', 'listed', 'NEARHI', 'V60', 'liq'):
    if hasattr(P, a):
        delattr(P, a)
gc.collect()
print('載入完成', f'{time.time()-t_all:.0f}s', rss(), 'MB', flush=True)
bench = pd.Series(P.E50[P.t0:], index=P.IDX[P.t0:])
i_mid = (P.t0 + P.T - 1) // 2
D0, DM, DE = P.IDX[P.t0], P.IDX[i_mid], P.IDX[-1]
D19 = P.IDX[P.IDX < pd.Timestamp('2019-01-01')][-1]
D24 = P.IDX[P.IDX < pd.Timestamp('2024-01-01')][-1]
WIN = {'since2019': (D19, DE), 'full': (D0, DE), 'since2024': (D24, DE), 'half1': (D0, DM), 'half2': (DM, DE)}
W = lambda nav: {k: stlib.window(nav, bench, a, b) for k, (a, b) in WIN.items()}
cg = lambda nav, k: stlib.cagr(nav, *WIN[k])
mdd = lambda nav: stlib.maxdd(nav, D0, DE)
CAPS = (6e5, 8e5, 1e6, 1.5e6, 2e6)
KW = dict(depth=10, mom_lo=0.10, exit='top20', keep_info=True)
CELLS = {c['id']: c for c in PRE['grid']['cells']}
DEFS = PRE['signal_day_definitions']['列表']
FAM = ['A08', 'A09', 'A10', 'A11', 'A12', 'A13', 'A15']
FAM2 = ['A10', 'A11', 'A12', 'A13', 'A15']
MODES = ('open', 'close')
bw = {k: dict(cagr=stlib.cagr(bench, a, b), maxdd=stlib.maxdd(bench, a, b)) for k, (a, b) in WIN.items()}


def same_trades(l1, l2):
    a = l1[['code', 'ent_t', 'exit_t', 'why']].reset_index(drop=True); b = l2[['code', 'ent_t', 'exit_t', 'why']].reset_index(drop=True)
    return bool(len(a) == len(b) and (a == b).all().all())


def sim(c, S, mode, cost='strict', **kw):
    return s9lib.simulate_s9(P, S, U, sat_w=c['sat_w'], sizing=c['sizing'], lock=c['lock'], cost=COSTS[cost], exec_mode=mode, **kw)


def slim(w, keys=('since2019', 'full', 'since2024', 'half1', 'half2')):
    return {k: dict(cagr=w[k]['cagr'], diff=w[k]['diff'], maxdd=w[k]['maxdd']) for k in keys}


# ---------------------------------------------------------------- 驗收
acc = {'P0': True}
S0 = s4lib.make_signal(P, U, RT, **KW)
DD = {d['id']: d for d in DEFS}
rtA, infA = s9lib.rt_for(P, RT, DD['A10'])
acc['A1_av_identical'] = bool(all(rtA['av'][m] == RT['av'][m] for m in RT['S'].index))
SA = s4lib.make_signal(P, U, rtA, **KW)
acc['A1_signal_identical'] = bool(np.array_equal(SA['sig'], S0['sig']) and np.array_equal(SA['X'], S0['X'])
                                  and np.array_equal(np.nan_to_num(SA['rank'], nan=-9e9), np.nan_to_num(S0['rank'], nan=-9e9))
                                  and [x['k'] for x in SA['info']] == [x['k'] for x in S0['info']])
del SA; gc.collect()
pub = {'W10_70': (0.4813, 0.3408, -0.3340), 'WN_70': (0.6224, 0.4455, -0.3950)}
TOL = 0.00025
for cid, (a19, afull, add) in pub.items():
    c = CELLS[cid]
    n8, l8, _ = s8lib.simulate_s8(P, S0, U, sat_w=c['sat_w'], sizing=c['sizing'], lock=c['lock'], cap=None, cost=COSTS['strict'])
    acc[f'A0_{cid}'] = dict(since2019=cg(n8, 'since2019'), full=cg(n8, 'full'), maxdd=mdd(n8),
                            ok=bool(abs(cg(n8, 'since2019') - a19) < TOL and abs(cg(n8, 'full') - afull) < TOL and abs(mdd(n8) - add) < TOL))
    n9, l9, _ = sim(c, S0, 'open')
    acc[f'A2_{cid}'] = dict(nav_identical=bool(np.allclose(n8.to_numpy(), n9.to_numpy(), rtol=0, atol=1e-6)), trades_identical=same_trades(l8, l9))
    del n8, l8, n9, l9
ok = acc['A1_av_identical'] and acc['A1_signal_identical'] and all(acc[f'A0_{c}']['ok'] for c in pub) \
    and all(acc[f'A2_{c}']['nav_identical'] and acc[f'A2_{c}']['trades_identical'] for c in pub)
assert ok, ('驗收失敗', acc)
del S0; gc.collect()
print('驗收（A0～A2）', acc, flush=True)

# ---------------------------------------------------------------- 36 個組合
res = {'PREREG_sha256': sha, 'acceptance': acc,
       'meta': dict(panel_T=P.T, panel_N=P.N, start=str(D0.date()), mid=str(DM.date()), since2019_from=str(D19.date()), since2024_from=str(D24.date()), end=str(DE.date())),
       'bench_0050': bw, 'defs': {}, 'combos': {}}
navs = {'0050': bench}; trades = []
a3 = dict(n_fills=0, n_bad_price=0, n_bad_sigrow=0, n_bad_execrow=0)
for d in DEFS:
    did = d['id']
    for mode in MODES:
        rt, inf = s9lib.rt_for(P, RT, d, close=(mode == 'close'))
        if mode == 'open':
            res['defs'][did] = dict(spec=d, n_stale_months=len(inf['stale_months']), early_months=inf['early_months'],
                                    exec_dates={str(m): (str(P.IDX[k].date()) if k < P.T else None) for m, k in inf['av_exec'].items() if m >= pd.Period('2016-01', 'M')})
        for cid, c in CELLS.items():
            res['combos'][f'{did}|{mode}|{cid}'] = dict(defn=did, mode=mode, cell=cid, paths={}, paths_dd={}, paths_dd_dates={})
        for s in range(4):
            tc = time.time()
            S = s4lib.make_signal(P, U, rt, shift=s, **KW)
            if s == 0:
                # 訊號列 → 成交列：open 版成交在訊號列＋1 開盤；close 版訊號列＝成交列−1（成交列＝該定義的訊號日）
                for x in S['info']:
                    if mode == 'close':
                        m = pd.Period(x['month'], 'M')
                        if x['k'] + 1 != inf['av_exec'][m]:
                            a3['n_bad_execrow'] += 1
                sigdates = [x['sig_date'] for x in S['info']]
            for cid, c in CELLS.items():
                R = res['combos'][f'{did}|{mode}|{cid}']
                for cap in CAPS:
                    nav, lg, st = sim(c, S, mode, capital=cap)
                    key = f'{int(cap/1e4)}萬|+{s}'
                    R['paths'][key] = cg(nav, 'since2019'); R['paths_dd'][key] = mdd(nav)
                    ddd = s9lib.dd_dates(nav); R['paths_dd_dates'][key] = [ddd['peak'], ddd['trough']]
                    if s == 0 and cap == 1e6:
                        name = f'{did}|{mode}|{cid}'
                        navs[name] = nav
                        w = W(nav)
                        R['signal_dates'] = sigdates
                        R['windows'] = slim(w)
                        R['sharpe'] = {k: w[k]['sharpe'] for k in ('full', 'since2019')}
                        R['dd_dates'] = ddd
                        R['trades_full'] = stlib.trade_stats(lg); R['trades_since2019'] = stlib.trade_stats(lg, '2019-01-01')
                        R['n_buy'] = int(st['n_buy']); R['skip_cash'] = int(st['skip_cash']); R['avg_npos'] = float(st['avg_npos'])
                        R['base_sold_n'] = len(st['base_sold'])
                        R['blk_limup'] = int(st['blk_limup']); R['sell_defer_limdn'] = int(st['sell_defer_limdn']); R['blk_noopen'] = int(st['blk_noopen'])
                        x = lg.copy(); x['run'] = name
                        trades.append(x[['run', 'code', 'sig_date', 'ent_date', 'exit_date', 'hold_days', 'shares', 'cost', 'proceeds', 'pnl', 'ret', 'why', 'score']])
                        if mode == 'close':                    # 驗收 A3
                            for (t, j, kind, px) in st['fills']:
                                a3['n_fills'] += 1
                                if not (np.isfinite(P.C[t, j]) and abs(px - P.C[t, j]) < 1e-9):
                                    a3['n_bad_price'] += 1
                            a3['n_bad_sigrow'] += int((lg['sig_t'] != lg['ent_t'] - 1).sum())
                        nx2 = sim(c, S, mode, cost='strict_x2')[0]
                        R['costx2'] = slim(W(nx2), ('since2019', 'full'))
                        blk, top = s2lib.block_from(lg, 5)
                        nb, _, sb = sim(c, S, mode, block=blk)
                        R['block5'] = slim(W(nb), ('since2019', 'full'))
                        R['block5_top'] = [dict(code=y.code, ent=str(y.ent_date.date()), exit=str(y.exit_date.date()), ret=float(y.ret), pnl=float(y.pnl)) for y in top.itertuples()]
                        del nx2, nb
                    del nav, lg, st
            del S; gc.collect()
            print(did, mode, '位移', s, f'{time.time()-tc:.0f}s 累計 {time.time()-t_all:.0f}s {rss()}MB', flush=True)
        for cid in CELLS:
            R = res['combos'][f'{did}|{mode}|{cid}']; w = R['windows']
            print(f"  {did}|{mode}|{cid} 2019起 {w['since2019']['cagr']*100:.2f}% 全期 {w['full']['cagr']*100:.2f}% dd {w['full']['maxdd']*100:.2f}% "
                  f"路徑中位數 {np.median(list(R['paths'].values()))*100:.2f}%", flush=True)
acc['A3_close'] = a3
assert a3['n_bad_price'] == 0 and a3['n_bad_sigrow'] == 0 and a3['n_bad_execrow'] == 0 and a3['n_fills'] > 0, ('A3 失敗', a3)

# ---------------------------------------------------------------- 彙整、八項、分類
for key, R in res['combos'].items():
    pv = np.array(list(R['paths'].values())); pdd = np.array([R['paths_dd'][k] for k in R['paths']])
    assert len(pv) == 20
    R['paths20'] = dict(min=float(pv.min()), median=float(np.median(pv)), max=float(pv.max()), n_above_0050=int((pv > bw['since2019']['cagr']).sum()),
                        maxdd_min=float(pdd.min()), maxdd_median=float(np.median(pdd)), maxdd_max=float(pdd.max()))
    w = R['windows']; x2 = R['costx2']; b5 = R['block5']
    chk = dict(net_since2019=w['since2019']['diff'] > 0, net_full=w['full']['diff'] > 0, costx2_since2019=x2['since2019']['diff'] > 0, costx2_full=x2['full']['diff'] > 0,
               block5_since2019=b5['since2019']['diff'] > 0, block5_full=b5['full']['diff'] > 0, half1=w['half1']['diff'] > 0, half2=w['half2']['diff'] > 0)
    R['checks'] = {k: bool(v) for k, v in chk.items()}
    R['class'] = '值得上XQ測' if all(chk.values()) else ('不值得' if not (chk['net_since2019'] or chk['net_full']) else '有條件')
    R['fails'] = [k for k, v in chk.items() if not v]


def rank_hi(vals, tid):
    """名次：比它大的個數＋1（1＝最高）。回撤是負數，數值越大＝越淺，所以同一函式也給回撤用（1＝最淺）。"""
    return 1 + sum(1 for k, x in vals.items() if k != tid and x > vals[tid])


fam = {}
for cid in CELLS:
    for mode in MODES:
        g = lambda did: res['combos'][f'{did}|{mode}|{cid}']
        r19 = {did: g(did)['windows']['since2019']['cagr'] for did in FAM}
        rfu = {did: g(did)['windows']['full']['cagr'] for did in FAM}
        ddm = {did: g(did)['windows']['full']['maxdd'] for did in FAM}
        pooled = np.array([v for did in FAM for v in g(did)['paths'].values()])
        pooled_dd = np.array([v for did in FAM for v in g(did)['paths_dd'].values()])
        pooled2 = np.array([v for did in FAM2 for v in g(did)['paths'].values()])
        pooled2_dd = np.array([v for did in FAM2 for v in g(did)['paths_dd'].values()])
        f = dict(r19=r19, rfull=rfu, maxdd=ddm,
                 S1=dict(r19_median=float(np.median(list(r19.values()))), r19_min=float(min(r19.values())), r19_min_def=min(r19, key=r19.get),
                         r19_max=float(max(r19.values())), r19_max_def=max(r19, key=r19.get),
                         A10_minus_median_pp=(r19['A10'] - np.median(list(r19.values()))) * 100, A10_minus_min_pp=(r19['A10'] - min(r19.values())) * 100,
                         full_median=float(np.median(list(rfu.values()))), full_min=float(min(rfu.values())),
                         maxdd_median=float(np.median(list(ddm.values()))), maxdd_worst=float(min(ddm.values())), maxdd_worst_def=min(ddm, key=ddm.get),
                         fam2_r19_median=float(np.median([r19[x] for x in FAM2])), fam2_r19_min=float(min(r19[x] for x in FAM2)),
                         fam2_maxdd_median=float(np.median([ddm[x] for x in FAM2]))),
                 S2=dict(A10_rank_r19=rank_hi(r19, 'A10'), A10_rank_full=rank_hi(rfu, 'A10'), A10_rank_maxdd=rank_hi(ddm, 'A10'),
                         A10_rank_r19_fam2=rank_hi({x: r19[x] for x in FAM2}, 'A10'), A10_rank_maxdd_fam2=rank_hi({x: ddm[x] for x in FAM2}, 'A10'), n=len(FAM)),
                 S3=dict(n_paths=int(len(pooled)), r19_median=float(np.median(pooled)), r19_p10=float(np.percentile(pooled, 10)), r19_min=float(pooled.min()),
                         maxdd_median=float(np.median(pooled_dd)), maxdd_p10=float(np.percentile(pooled_dd, 10)), maxdd_min=float(pooled_dd.min()),
                         n_above_0050=int((pooled > bw['since2019']['cagr']).sum()),
                         fam2_r19_median=float(np.median(pooled2)), fam2_r19_p10=float(np.percentile(pooled2, 10)),
                         fam2_maxdd_median=float(np.median(pooled2_dd)), fam2_maxdd_p10=float(np.percentile(pooled2_dd, 10)),
                         A10_paths_r19_median=float(np.median(list(g('A10')['paths'].values()))), A10_paths_maxdd_median=float(np.median(list(g('A10')['paths_dd'].values())))))
        fam[f'{cid}|{mode}'] = f
# S4 成交時點
s4 = {}
for cid in CELLS:
    dif = {did: (res['combos'][f'{did}|close|{cid}']['windows']['since2019']['cagr'] - res['combos'][f'{did}|open|{cid}']['windows']['since2019']['cagr']) * 100 for did in FAM}
    difp = {did: (res['combos'][f'{did}|close|{cid}']['paths20']['median'] - res['combos'][f'{did}|open|{cid}']['paths20']['median']) * 100 for did in FAM}
    difd = {did: (res['combos'][f'{did}|close|{cid}']['windows']['full']['maxdd'] - res['combos'][f'{did}|open|{cid}']['windows']['full']['maxdd']) * 100 for did in FAM}
    npos = sum(v > 0 for v in dif.values()); med = float(np.median(list(dif.values())))
    rec = '收盤' if (npos >= 5 and med > 0) else ('開盤' if (npos <= 2 and med < 0) else '差不多，選做得到的')
    s4[cid] = dict(close_minus_open_pp=dif, median_pp=med, n_positive=int(npos), paths_median_diff_pp=difp, paths_median_diff_median_pp=float(np.median(list(difp.values()))),
                   paths_median_diff_npos=int(sum(v > 0 for v in difp.values())), maxdd_diff_pp=difd, recommendation=rec)
# S5 XQ 口徑
s5 = {}
for cid in CELLS:
    base = res['combos'][f'A10|open|{cid}']
    s5[cid] = {f'{did}|{mode}': dict(r19=res['combos'][f'{did}|{mode}|{cid}']['windows']['since2019']['cagr'],
                                     maxdd=res['combos'][f'{did}|{mode}|{cid}']['windows']['full']['maxdd'],
                                     paths_median=res['combos'][f'{did}|{mode}|{cid}']['paths20']['median'],
                                     paths_dd_median=res['combos'][f'{did}|{mode}|{cid}']['paths20']['maxdd_median'],
                                     r19_vs_A10open_pp=(res['combos'][f'{did}|{mode}|{cid}']['windows']['since2019']['cagr'] - base['windows']['since2019']['cagr']) * 100,
                                     paths_median_vs_A10open_pp=(res['combos'][f'{did}|{mode}|{cid}']['paths20']['median'] - base['paths20']['median']) * 100)
               for did in ('A10', 'B11opt', 'B11pes') for mode in MODES}
# 判定
r1 = [fam[f'{cid}|open']['S2']['A10_rank_r19'] == 1 for cid in CELLS]
v1 = '是，剛好最好' if all(r1) else ('部分' if any(r1) else '不是')
gap = {cid: fam[f'{cid}|open']['S1']['A10_minus_median_pp'] for cid in CELLS}
v2 = '標準定義偏樂觀，實際預期改用家族中位數' if any(g >= 3 for g in gap.values()) else '挑選偏誤不大'
famopen = [res['combos'][f'{did}|open|{cid}'] for did in FAM for cid in CELLS]
v3_fails = {f'{R["defn"]}|open|{R["cell"]}': R['fails'] for R in famopen if R['fails']}
v3 = '成立' if not v3_fails else '部分成立'
res['family'] = fam; res['S4_exec'] = s4; res['S5_xq'] = s5
res['verdict'] = dict(standard_is_best=v1, standard_rank_open={cid: fam[f'{cid}|open']['S2']['A10_rank_r19'] for cid in CELLS},
                      selection_bias=v2, A10_minus_median_pp=gap, robust=v3, robust_fails=v3_fails,
                      exec_recommendation={cid: s4[cid]['recommendation'] for cid in CELLS},
                      all_combos_class={k: R['class'] for k, R in res['combos'].items()})
print('判定', json.dumps(res['verdict'], ensure_ascii=False), flush=True)
for k, f in fam.items():
    print(k, 'S1', {a: (round(b, 4) if isinstance(b, float) else b) for a, b in f['S1'].items()}, 'S2', f['S2'], 'S3', {a: (round(b, 4) if isinstance(b, float) else b) for a, b in f['S3'].items()}, flush=True)
print('S4', json.dumps(s4, ensure_ascii=False, default=float), flush=True)

res['meta']['runtime_s'] = round(time.time() - t_all, 1); res['meta']['maxrss_mb'] = rss()
res['meta']['N'] = PRE['data_snooping']
(H / 'results.json').write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str))
pd.DataFrame(navs).round(1).to_csv(H / 'navs.csv.gz', index_label='date')
tr = pd.concat(trades, ignore_index=True)
num = tr.select_dtypes('number').columns
tr[num] = tr[num].round(4)
for c in ('sig_date', 'ent_date', 'exit_date'):
    tr[c] = pd.to_datetime(tr[c]).dt.strftime('%Y-%m-%d')
tr.to_csv(H / 'trades.csv.gz', index=False)
print(res['meta']['runtime_s'], 's', res['meta']['maxrss_mb'], 'MB')
