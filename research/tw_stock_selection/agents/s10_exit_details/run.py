#!/usr/bin/env python3
"""S10：出場細節家族。11 個出場（E0＋10 個新變體）× 2 格（W10_70、WN_70，不鎖定，A10 開盤）＝22 個組合；
每個組合跑 100 萬主跑、20 條路徑（5 種資金 × 位移 0～3）、成本加倍、禁買最賺 5 筆；另跑 min_frac 0.45／0.55 的 20 條路徑。
python3 -I -B run.py → results.json、trades.csv.gz、navs.csv.gz。PREREG.json 的 sha256 必須相符。單行程。"""
import sys, json, hashlib, time, resource, gc
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
import s10lib
from s10lib import s9lib, s4lib, s2lib, stlib, COSTS, EXITS

H = Path(__file__).resolve().parent
pre = (H / 'PREREG.json').read_bytes()
sha = hashlib.sha256(pre).hexdigest()
assert sha == (H / 'PREREG.sha256').read_text().split()[0], 'PREREG.json 與 PREREG.sha256 不符'
PRE = json.loads(pre)
t_all = time.time()
rss = lambda: round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
P = s10lib.load()
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
KW4 = dict(depth=10, mom_lo=0.10, exit='top20', keep_info=True)
CELLS = {c['id']: c for c in PRE['grid']['cells']}
MF = (0.5, 0.45, 0.55)
bw = {k: dict(cagr=stlib.cagr(bench, a, b), maxdd=stlib.maxdd(bench, a, b)) for k, (a, b) in WIN.items()}
YRS = (P.IDX[-1] - P.IDX[P.t0]).days / 365.25


def same_trades(l1, l2):
    a = l1[['code', 'ent_t', 'exit_t', 'why']].reset_index(drop=True); b = l2[['code', 'ent_t', 'exit_t', 'why']].reset_index(drop=True)
    return bool(len(a) == len(b) and (a == b).all().all())


def sim(c, S, eid, cost='strict', **kw):
    S2, ekw = s10lib.variant_S(S, eid)
    ekw.update(kw)
    return s10lib.simulate_s10(P, S2, U, sat_w=c['sat_w'], sizing=c['sizing'], lock=c['lock'], cost=COSTS[cost], **ekw)


def slim(w, keys=('since2019', 'full', 'since2024', 'half1', 'half2')):
    return {k: dict(cagr=w[k]['cagr'], diff=w[k]['diff'], maxdd=w[k]['maxdd']) for k in keys}


# ---------------------------------------------------------------- 驗收
acc = {'P0': True}


def build_signal(s):
    """位移 s 的訊號＋出場家族；同時做驗收 A1、A2（記進 acc，失敗就停）。"""
    S = s10lib.make_signal_s10(P, U, RT, shift=s)
    S4 = s4lib.make_signal(P, U, RT, shift=s, **KW4)
    acc[f'A1_shift{s}'] = bool(np.array_equal(S['sig'], S4['sig']) and np.array_equal(S['X'], S4['X'])
                               and np.array_equal(np.nan_to_num(S['rank'], nan=-9e9), np.nan_to_num(S4['rank'], nan=-9e9))
                               and [x['k'] for x in S['info']] == [x['k'] for x in S4['info']])
    acc[f'A2_shift{s}_mismatch_cells'] = int(S['A2_mismatch'])
    del S4; gc.collect()
    assert acc[f'A1_shift{s}'] and acc[f'A2_shift{s}_mismatch_cells'] == 0, ('驗收 A1／A2 失敗', s, acc)
    return S


SIG = {0: build_signal(0)}
pub = {'W10_70': (0.4813, 0.3408, -0.3340), 'WN_70': (0.6224, 0.4455, -0.3950)}
TOL = 0.00025
S0 = SIG[0]
for cid, (a19, afull, add) in pub.items():
    c = CELLS[cid]
    n9, l9, _ = s9lib.simulate_s9(P, S0, U, sat_w=c['sat_w'], sizing=c['sizing'], lock=c['lock'], cost=COSTS['strict'], exec_mode='open')
    n10, l10, _ = sim(c, S0, 'E0')
    acc[f'A0_{cid}'] = dict(since2019=cg(n10, 'since2019'), full=cg(n10, 'full'), maxdd=mdd(n10),
                            nav_identical=bool(np.allclose(n9.to_numpy(), n10.to_numpy(), rtol=0, atol=1e-6)), trades_identical=same_trades(l9, l10),
                            ok=bool(abs(cg(n10, 'since2019') - a19) < TOL and abs(cg(n10, 'full') - afull) < TOL and abs(mdd(n10) - add) < TOL))
    nt, lt, _ = sim(c, S0, 'E0', trail=1.0)
    nr, lr, _ = sim(c, S0, 'E0', requal=10000)
    acc[f'A3_{cid}'] = dict(trail1_identical=bool(np.allclose(n10.to_numpy(), nt.to_numpy(), rtol=0, atol=1e-6) and same_trades(l10, lt)),
                            requal_inf_identical=bool(np.allclose(n10.to_numpy(), nr.to_numpy(), rtol=0, atol=1e-6) and same_trades(l10, lr)))
    del n9, l9, n10, l10, nt, lt, nr, lr
ok = all(acc[f'A0_{c}']['ok'] and acc[f'A0_{c}']['nav_identical'] and acc[f'A0_{c}']['trades_identical'] for c in pub) \
    and all(acc[f'A3_{c}']['trail1_identical'] and acc[f'A3_{c}']['requal_inf_identical'] for c in pub)
assert ok, ('驗收失敗', acc)
print('驗收（A0～A3）', acc, flush=True)


def a4_check(lg, thr):
    """移動停損出場：用還原收盤 G 獨立重算。標記日＝出場列 − 1（或順延：找第一個成立日，要求 ≤ 出場列 − 1 且之前沒有成立）。"""
    bad = 0; n = 0
    for r in lg[lg['why'] == 'trail'].itertuples():
        n += 1
        e, x, j = int(r.ent_t), int(r.exit_t), int(r.j)
        g = P.G[e:x, j] / P.G[e, j] * (P.CF[e, j] / P.O[e, j])
        pk = np.maximum.accumulate(g)
        hit = np.flatnonzero(g <= pk * (1 - thr) * (1 + 1e-6))
        if len(hit) == 0:
            bad += 1; continue
        first = e + int(hit[0])
        # 標記日之後要等到可賣（開盤 > 跌停且有開盤價）的第一天才出場
        tt = first + 1
        while tt < P.T and not (np.isfinite(P.O[tt, j]) and P.O[tt, j] > 0 and P.O[tt, j] > P.LIMDN[tt, j] + 1e-9):
            tt += 1
        if tt != x:
            bad += 1
    return dict(n=n, mismatch=bad)


# ---------------------------------------------------------------- 22 個組合
res = {'PREREG_sha256': sha, 'acceptance': acc,
       'meta': dict(panel_T=P.T, panel_N=P.N, start=str(D0.date()), mid=str(DM.date()), since2019_from=str(D19.date()), since2024_from=str(D24.date()), end=str(DE.date())),
       'bench_0050': bw, 'combos': {}}
navs = {'0050': bench}; trades = []
a4 = {}
for eid in EXITS:
    for cid in CELLS:
        res['combos'][f'{eid}|{cid}'] = dict(exit=eid, cell=cid, paths={}, paths_dd={}, paths_dd_dates={}, paths_mf={str(m): {} for m in MF[1:]}, paths_dd_mf={str(m): {} for m in MF[1:]})
for s in range(4):
    tc = time.time()
    S = SIG.pop(s) if s in SIG else build_signal(s)
    for eid in EXITS:
        for cid, c in CELLS.items():
            R = res['combos'][f'{eid}|{cid}']
            for cap in CAPS:
                key = f'{int(cap/1e4)}萬|+{s}'
                for mf in MF:
                    nav, lg, st = sim(c, S, eid, capital=cap, min_frac=mf)
                    if mf != 0.5:
                        R['paths_mf'][str(mf)][key] = cg(nav, 'since2019'); R['paths_dd_mf'][str(mf)][key] = mdd(nav)
                        del nav, lg, st
                        continue
                    R['paths'][key] = cg(nav, 'since2019'); R['paths_dd'][key] = mdd(nav)
                    ddd = s10lib.dd_dates(nav); R['paths_dd_dates'][key] = [ddd['peak'], ddd['trough']]
                    if s == 0 and cap == 1e6:
                        name = f'{eid}|{cid}'
                        navs[name] = nav
                        w = W(nav)
                        R['windows'] = slim(w)
                        R['sharpe'] = {k: w[k]['sharpe'] for k in ('full', 'since2019')}
                        R['t_month_excess'] = {k: w[k]['t_month_excess'] for k in ('full', 'since2019')}
                        R['dd_dates'] = ddd
                        R['yearly'] = stlib.yearly(nav, bench)
                        R['trades_full'] = stlib.trade_stats(lg); R['trades_since2019'] = stlib.trade_stats(lg, '2019-01-01')
                        R['n_buy'] = int(st['n_buy']); R['skip_cash'] = int(st['skip_cash']); R['avg_npos'] = float(st['avg_npos'])
                        R['blk_limup'] = int(st['blk_limup']); R['sell_defer_limdn'] = int(st['sell_defer_limdn']); R['blk_min'] = int(st['blk_min'])
                        mnav = float(nav.mean())
                        R['turnover_oneway_ann'] = float((st['buy_val'] + st['sell_val']) / 2 / mnav / YRS)
                        R['cost_ann_pct_of_nav'] = float((st['fee'] + st['tax'] + st['slip'] + st['e50_cost']) / mnav / YRS)
                        R['stock_cost_ann_pct_of_nav'] = float((st['fee'] + st['tax'] + st['slip']) / mnav / YRS)
                        R['avg_stockfrac'] = float(np.nanmean(st['stockfrac'].to_numpy()))
                        ed = st['minfrac_edge']
                        R['minfrac_edge'] = dict(bought_50_55=int(sum(1 for x in ed if 0.5 <= x[2] < 0.55 and x[3])), skipped_45_50=int(sum(1 for x in ed if 0.45 <= x[2] < 0.5 and not x[3])),
                                                 list=[(str(P.IDX[x[0]].date()), str(P.COLS[x[1]]), round(x[2], 4), x[3]) for x in ed if 0.45 <= x[2] < 0.55])
                        if eid in ('TS15', 'TS25'):
                            a4[name] = a4_check(lg, {'TS15': 0.15, 'TS25': 0.25}[eid])
                        x = lg.copy(); x['run'] = name
                        trades.append(x[['run', 'code', 'sig_date', 'ent_date', 'exit_date', 'hold_days', 'shares', 'cost', 'proceeds', 'pnl', 'ret', 'why', 'score']])
                        nx2 = sim(c, S, eid, cost='strict_x2')[0]
                        R['costx2'] = slim(W(nx2), ('since2019', 'full'))
                        blk, top = s2lib.block_from(lg, 5)
                        nb = sim(c, S, eid, block=blk)[0]
                        R['block5'] = slim(W(nb), ('since2019', 'full'))
                        R['block5_top'] = [dict(code=y.code, ent=str(y.ent_date.date()), exit=str(y.exit_date.date()), ret=float(y.ret), pnl=float(y.pnl)) for y in top.itertuples()]
                        del nx2, nb
                    del nav, lg, st
    print('位移', s, f'{time.time()-tc:.0f}s 累計 {time.time()-t_all:.0f}s {rss()}MB', flush=True)
    del S; gc.collect()
acc['A4_trail'] = a4
assert all(v['n'] > 0 for v in a4.values()), ('A4：移動停損沒有觸發', a4)
print('驗收 A4（不符筆數，目標 0）', a4, flush=True)

# ---------------------------------------------------------------- 彙整、八項、分類
for key, R in res['combos'].items():
    pv = np.array(list(R['paths'].values())); pdd = np.array([R['paths_dd'][k] for k in R['paths']])
    assert len(pv) == 20
    R['paths20'] = dict(min=float(pv.min()), median=float(np.median(pv)), max=float(pv.max()), n_above_0050=int((pv > bw['since2019']['cagr']).sum()),
                        maxdd_min=float(pdd.min()), maxdd_median=float(np.median(pdd)), maxdd_max=float(pdd.max()),
                        n_dd_2024_07_to_2025_04=int(sum(1 for v in R['paths_dd_dates'].values() if v[0][:7] >= '2024-06' and v[0][:7] <= '2024-08' and v[1][:7] == '2025-04')))
    w = R['windows']; x2 = R['costx2']; b5 = R['block5']
    chk = dict(net_since2019=w['since2019']['diff'] > 0, net_full=w['full']['diff'] > 0, costx2_since2019=x2['since2019']['diff'] > 0, costx2_full=x2['full']['diff'] > 0,
               block5_since2019=b5['since2019']['diff'] > 0, block5_full=b5['full']['diff'] > 0, half1=w['half1']['diff'] > 0, half2=w['half2']['diff'] > 0)
    R['checks'] = {k: bool(v) for k, v in chk.items()}
    R['class'] = '值得上XQ測' if all(chk.values()) else ('不值得' if not (chk['net_since2019'] or chk['net_full']) else '有條件')
    R['fails'] = [k for k, v in chk.items() if not v]


def rank_hi(vals, tid):
    """名次：比它大的個數＋1（1＝最高）。回撤是負數，數值越大＝越淺，同一函式給回撤用（1＝最淺）。"""
    return 1 + sum(1 for k, x in vals.items() if k != tid and x > vals[tid])


def judge_ret(d):
    npos = int((d > 0).sum()); med = float(np.median(d)) * 100
    if npos >= 15 and med >= 2:
        return '比標準好', npos, med
    if npos <= 5 and med <= -2:
        return '比標準差', npos, med
    return '差不多', npos, med


def judge_dd(d, dret):
    nsh = int((d > 0).sum()); med = float(np.median(d)) * 100; mret = float(np.median(dret)) * 100
    if nsh >= 15 and med >= 2:
        return ('有效率地壓回撤' if mret >= -2 else '能壓回撤'), nsh, med
    return '不能', nsh, med


fam = {}; pair = {}
for cid in CELLS:
    g = lambda eid: res['combos'][f'{eid}|{cid}']
    pm = {e: g(e)['paths20']['median'] for e in EXITS}
    pdm = {e: g(e)['paths20']['maxdd_median'] for e in EXITS}
    m19 = {e: g(e)['windows']['since2019']['cagr'] for e in EXITS}
    mdm = {e: g(e)['windows']['full']['maxdd'] for e in EXITS}
    fam[cid] = dict(path_median=pm, path_maxdd_median=pdm, main_r19=m19, main_maxdd=mdm,
                    P1=dict(path_median_median=float(np.median(list(pm.values()))), path_median_min=float(min(pm.values())), path_median_min_exit=min(pm, key=pm.get),
                            path_median_max=float(max(pm.values())), path_median_max_exit=max(pm, key=pm.get),
                            path_dd_median=float(np.median(list(pdm.values()))), path_dd_deepest=float(min(pdm.values())), path_dd_deepest_exit=min(pdm, key=pdm.get),
                            path_dd_shallowest=float(max(pdm.values())), path_dd_shallowest_exit=max(pdm, key=pdm.get),
                            main_r19_median=float(np.median(list(m19.values()))), main_r19_min=float(min(m19.values())), main_r19_max=float(max(m19.values())),
                            main_dd_median=float(np.median(list(mdm.values()))), main_dd_deepest=float(min(mdm.values())), main_dd_shallowest=float(max(mdm.values())),
                            n_exits=len(EXITS)),
                    P2=dict(E0_rank_path_median=rank_hi(pm, 'E0'), E0_rank_path_dd=rank_hi(pdm, 'E0'), E0_rank_main_r19=rank_hi(m19, 'E0'), E0_rank_main_dd=rank_hi(mdm, 'E0'),
                            order_path_median=sorted(pm, key=pm.get, reverse=True), order_path_dd=sorted(pdm, key=pdm.get, reverse=True)))
    keys = list(g('E0')['paths'].keys())
    for eid in EXITS[1:]:
        d = np.array([g(eid)['paths'][k] - g('E0')['paths'][k] for k in keys])
        dd = np.array([g(eid)['paths_dd'][k] - g('E0')['paths_dd'][k] for k in keys])
        vr, npos, med = judge_ret(d); vd, nsh, medd = judge_dd(dd, d)
        mf = {}
        for m in MF[1:]:
            dm = np.array([g(eid)['paths_mf'][str(m)][k] - g('E0')['paths_mf'][str(m)][k] for k in keys])
            ddm = np.array([g(eid)['paths_dd_mf'][str(m)][k] - g('E0')['paths_dd_mf'][str(m)][k] for k in keys])
            a, b, cc = judge_ret(dm); a2, b2, c2 = judge_dd(ddm, dm)
            mf[str(m)] = dict(verdict_ret=a, npos=b, median_pp=cc, verdict_dd=a2, n_shallower=b2, dd_median_pp=c2)
        pair[f'{eid}|{cid}'] = dict(verdict_ret=vr, npos=npos, median_pp=med, min_pp=float(d.min()) * 100, max_pp=float(d.max()) * 100,
                                    verdict_dd=vd, n_shallower=nsh, dd_median_pp=medd, minfrac=mf,
                                    main_diff_r19_pp=(g(eid)['windows']['since2019']['cagr'] - g('E0')['windows']['since2019']['cagr']) * 100,
                                    main_diff_dd_pp=(g(eid)['windows']['full']['maxdd'] - g('E0')['windows']['full']['maxdd']) * 100)
# 判定
stable = {}
for eid in EXITS[1:]:
    both = all(pair[f'{eid}|{c}']['verdict_ret'] == '比標準好' for c in CELLS)
    one = any(pair[f'{eid}|{c}']['verdict_ret'] == '比標準好' for c in CELLS)
    mf_ok = all(pair[f'{eid}|{c}']['minfrac'][str(m)]['verdict_ret'] == '比標準好' for c in CELLS for m in MF[1:])
    stable[eid] = '穩定地比較好' if (both and mf_ok) else ('比標準好但對最小買進門檻敏感' if both else ('只有一格較好' if one else '否'))
ts = {e: [pair[f'{e}|{c}']['verdict_dd'] != '不能' for c in CELLS] for e in ('TS15', 'TS25')}
v_ts = '能' if any(all(v) for v in ts.values()) else ('部分' if any(any(v) for v in ts.values()) else '不能')
r = [fam[c]['P2']['E0_rank_path_median'] for c in CELLS]
v_e0 = '是' if all(x == 1 for x in r) else ('前段' if all(x <= 3 for x in r) else '不是')
xq = [e for e in EXITS[1:] if stable[e] == '穩定地比較好' or all(pair[f'{e}|{c}']['verdict_dd'] == '有效率地壓回撤' for c in CELLS)]
res['family'] = fam; res['pairs'] = pair
res['verdict'] = dict(E0_is_best=v_e0, E0_rank_path_median={c: fam[c]['P2']['E0_rank_path_median'] for c in CELLS}, E0_rank_path_dd={c: fam[c]['P2']['E0_rank_path_dd'] for c in CELLS},
                      stable_better=stable, any_stable_better=[e for e, v in stable.items() if v == '穩定地比較好'],
                      trailing_stop_cuts_dd=v_ts, trailing_detail={e: {c: pair[f'{e}|{c}']['verdict_dd'] for c in CELLS} for e in ('TS15', 'TS25', 'TS15M')},
                      per_cell_ret={k: v['verdict_ret'] for k, v in pair.items()}, per_cell_dd={k: v['verdict_dd'] for k, v in pair.items()},
                      xq_change=xq if xq else '維持現行參數',
                      all_combos_class={k: R['class'] for k, R in res['combos'].items()},
                      class_fails={k: R['fails'] for k, R in res['combos'].items() if R['fails']})
print('判定', json.dumps(res['verdict'], ensure_ascii=False), flush=True)
for c in CELLS:
    print(c, 'P1', {a: (round(b, 4) if isinstance(b, float) else b) for a, b in fam[c]['P1'].items()}, 'P2', fam[c]['P2'], flush=True)
for k, v in pair.items():
    print(k, v['verdict_ret'], v['npos'], round(v['median_pp'], 2), '|', v['verdict_dd'], v['n_shallower'], round(v['dd_median_pp'], 2),
          '| mf', {m: (x['verdict_ret'], x['npos'], round(x['median_pp'], 2)) for m, x in v['minfrac'].items()}, flush=True)

res['meta']['runtime_s'] = round(time.time() - t_all, 1); res['meta']['maxrss_mb'] = rss()
res['meta']['N'] = PRE['data_snooping']
(H / 'results.json').write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str))
pd.DataFrame(navs).round(0).to_csv(H / 'navs.csv.gz', index_label='date')
tr = pd.concat(trades, ignore_index=True)
num = tr.select_dtypes('number').columns
tr[num] = tr[num].round(4)
for c in ('sig_date', 'ent_date', 'exit_date'):
    tr[c] = pd.to_datetime(tr[c]).dt.strftime('%Y-%m-%d')
tr.to_csv(H / 'trades.csv.gz', index=False)
print(res['meta']['runtime_s'], 's', res['meta']['maxrss_mb'], 'MB')
