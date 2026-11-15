#!/usr/bin/env python3
"""S17：SURGE 定義家族。7 個定義（STD＋6 個新定義）× 2 格（W10_70、WN_70，不鎖定，A10 開盤，F1 進場＋E0 出場）＝14 個組合；
每個組合跑 100 萬主跑、20 條路徑（5 種資金 × 位移 0～3）、成本加倍、禁買最賺 5 筆；另跑 min_frac 0.45／0.55 的 20 條路徑。
python3 -I -B run.py → results.json、trades.csv.gz、navs.csv.gz。PREREG.json 的 sha256 必須相符。單行程。"""
import sys, json, hashlib, time, resource, gc, random, statistics, math
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
import s17lib
from s17lib import s10lib, s9lib, s4lib, s2lib, stlib, COSTS, DEFS

H = Path(__file__).resolve().parent
pre = (H / 'PREREG.json').read_bytes()
sha = hashlib.sha256(pre).hexdigest()
assert sha == (H / 'PREREG.sha256').read_text().split()[0], 'PREREG.json 與 PREREG.sha256 不符'
PRE = json.loads(pre)
t_all = time.time()
rss = lambda: round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
P = s17lib.load()
assert (P.T, P.N, str(P.IDX[-1].date())) == (2876, 2142, '2026-10-22'), 'P0 失敗'
U0 = stlib.universe(P, 'twse')
INNOV, _ = s4lib.innov_panel(P)
U = U0 & ~INNOV
del U0, INNOV
RT0 = s2lib.rev_tables(P)
F = s17lib.rev_family(P)
RTD = {d: dict(S=F['S'][d], av=F['av']) for d in DEFS}
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
WIN = {'since2019': (D19, DE), 'full': (D0, DE), 'since2024': (D24, DE), 'half1': (D0, DM), 'half2': (DM, DE), 'y2019_2023': (D19, D24)}
W = lambda nav: {k: stlib.window(nav, bench, a, b) for k, (a, b) in WIN.items()}
cg = lambda nav, k: stlib.cagr(nav, *WIN[k])
mdd = lambda nav: stlib.maxdd(nav, D0, DE)
CAPS = (6e5, 8e5, 1e6, 1.5e6, 2e6)
CELLS = {c['id']: c for c in PRE['grid']['cells']}
MF = (0.5, 0.45, 0.55)
bw = {k: dict(cagr=stlib.cagr(bench, a, b), maxdd=stlib.maxdd(bench, a, b)) for k, (a, b) in WIN.items()}
bw['seg'] = s17lib.seg_dd(bench)
YRS = (P.IDX[-1] - P.IDX[P.t0]).days / 365.25
SEGS = list(s17lib.SEGMENTS)


def same_trades(l1, l2):
    a = l1[['code', 'ent_t', 'exit_t', 'why']].reset_index(drop=True); b = l2[['code', 'ent_t', 'exit_t', 'why']].reset_index(drop=True)
    return bool(len(a) == len(b) and (a == b).all().all())


def sim(c, S, cost='strict', **kw):
    S2, ekw = s10lib.variant_S(S, 'E0')
    ekw.update(kw)
    return s10lib.simulate_s10(P, S2, U, sat_w=c['sat_w'], sizing=c['sizing'], lock=c['lock'], cost=COSTS[cost], **ekw)


def slim(w, keys=('since2019', 'full', 'since2024', 'half1', 'half2', 'y2019_2023')):
    return {k: dict(cagr=w[k]['cagr'], diff=w[k]['diff'], maxdd=w[k]['maxdd']) for k in keys}


# ---------------------------------------------------------------- 驗收
acc = {'P0': True}
# A2：獨立重算（純 Python）
Rv = F['Rv']
rng = random.Random(20261115)


def indep(d, mi, code):
    sp = s17lib.DEF_SPEC[d]
    rv = Rv[code]
    months = rv.index

    def rev(i):
        if i < 0:
            return None
        x = rv.iat[i]
        return float(x) if (x == x and x > 0) else None

    def dval(i):
        m = months[i]
        if sp['smooth'] == 3:
            if m <= pd.Period('2014-02', 'M'):
                return None
            a = [rev(i - k) for k in range(3)]; b = [rev(i - 12 - k) for k in range(3)]
            if any(v is None for v in a + b):
                return None
            return sum(a) - sum(b)
        if m.year == 2013:
            return None
        a, b = rev(i), rev(i - 12)
        if a is None or b is None:
            return None
        return a / b - 1 if sp['scale'] == 'rate' else a - b
    if months[mi] < s17lib.FIRST_MONTH:
        return float('nan')
    cur = dval(mi)
    w = sp['win']
    hist = [dval(mi - k) for k in range(1, w + 1)]
    if cur is None or any(v is None for v in hist):
        return float('nan')
    if sp['center'] == 'median':
        c = statistics.median(hist); s = 1.4826 * statistics.median([abs(v - c) for v in hist])
    else:
        mu = statistics.fmean(hist); s = statistics.stdev(hist); c = mu if sp['center'] == 'mean' else 0.0
    if not s > 1e-9 * (statistics.fmean([abs(v) for v in hist]) + 1e-12):
        return float('nan')
    return (cur - c) / s


a2 = {}
col_has = Rv.notna().any().to_numpy()
for d in DEFS:
    Z = F['S'][d]
    zv = Z.to_numpy()
    m16 = int(np.searchsorted(np.array([p.ordinal for p in Z.index]), s17lib.FIRST_MONTH.ordinal))
    fin = np.argwhere(np.isfinite(zv)); nan = np.argwhere(~np.isfinite(zv[m16:])) + [m16, 0]
    nan = nan[col_has[nan[:, 1]]]
    picks = [tuple(fin[rng.randrange(len(fin))]) for _ in range(30)] + [tuple(nan[rng.randrange(len(nan))]) for _ in range(10)]
    bad = 0; maxrel = 0.0
    for i, j in picks:
        x = indep(d, i, Z.columns[j]); y = zv[i, j]
        if np.isfinite(x) != np.isfinite(y):
            bad += 1
        elif np.isfinite(x):
            rel = abs(x - y) / max(1.0, abs(x)); maxrel = max(maxrel, rel)
            if rel > 1e-8:
                bad += 1
    a2[d] = dict(n=len(picks), mismatch=bad, max_rel=maxrel)
acc['A2'] = a2
assert all(v['mismatch'] == 0 for v in a2.values()), ('驗收 A2 失敗', a2)
del Rv, F['Rv']

# A1：STD 名單和 s2lib.rev_tables 名單相同（位移 0～3）
for s in range(4):
    Sa = s10lib.make_signal_s10(P, U, RT0, shift=s); Sb = s10lib.make_signal_s10(P, U, RTD['STD'], shift=s)
    ra, rb = np.nan_to_num(Sa['rank'], nan=0.0), np.nan_to_num(Sb['rank'], nan=0.0)
    acc[f'A1_shift{s}'] = dict(sig=bool(np.array_equal(Sa['sig'], Sb['sig'])), X=bool(np.array_equal(Sa['X'], Sb['X'])), sigrows=bool(np.array_equal(Sa['sigrows'], Sb['sigrows'])),
                               rank_maxdiff=float(np.abs(ra - rb).max()))
    del Sa, Sb; gc.collect()
assert all(v['sig'] and v['X'] and v['sigrows'] and v['rank_maxdiff'] < 1e-5 for k, v in acc.items() if k.startswith('A1_')), ('驗收 A1 失敗', acc)

# A0：先不修正 0050 停牌日（＝S10 E0），再修正
pub = {'W10_70': (0.4813, 0.3408, -0.3340), 'WN_70': (0.6224, 0.4455, -0.3950)}
TOL = 0.00025
S_old = s10lib.make_signal_s10(P, U, RT0, shift=0)
S_std = s10lib.make_signal_s10(P, U, RTD['STD'], shift=0)
nofix = {}
for cid in pub:
    c = CELLS[cid]
    n1, l1, _ = sim(c, S_old); n2, l2, _ = sim(c, S_std)
    nofix[cid] = (n2, l2)
    acc[f'A0_nofix_{cid}'] = dict(since2019=cg(n2, 'since2019'), full=cg(n2, 'full'), maxdd=mdd(n2),
                                  nav_identical_vs_S10E0=bool(np.allclose(n1.to_numpy(), n2.to_numpy(), rtol=0, atol=1e-6)), trades_identical=same_trades(l1, l2))
    del n1, l1
acc['e50_suspension_fix'] = s17lib.e50_fix_suspension(P)
for cid, (a19, afull, add) in pub.items():
    c = CELLS[cid]
    n3, l3, _ = sim(c, S_std)
    n2, l2 = nofix[cid]
    acc[f'A0_{cid}'] = dict(since2019=cg(n3, 'since2019'), full=cg(n3, 'full'), maxdd=mdd(n3), since2024=cg(n3, 'since2024'),
                            trades_identical_vs_nofix=same_trades(l2, l3), max_nav_diff_vs_nofix=float(np.abs(n3.to_numpy() - n2.to_numpy()).max()),
                            end_nav_diff_pct=float(n3.iloc[-1] / n2.iloc[-1] - 1) * 100,
                            ok=bool(abs(cg(n3, 'since2019') - a19) < TOL and abs(cg(n3, 'full') - afull) < TOL and abs(mdd(n3) - add) < TOL))
    del n3, l3
del nofix, S_old, S_std; gc.collect()
ok = all(acc[f'A0_nofix_{c}']['nav_identical_vs_S10E0'] and acc[f'A0_nofix_{c}']['trades_identical'] and acc[f'A0_{c}']['ok'] for c in pub)
assert ok, ('驗收 A0 失敗', acc)
print('驗收（A0～A2）', json.dumps(acc, ensure_ascii=False, default=str), flush=True)

# ---------------------------------------------------------------- 14 個組合
res = {'PREREG_sha256': sha, 'acceptance': acc, 'data_check_sigma0': F['n_sigma0'],
       'meta': dict(panel_T=P.T, panel_N=P.N, start=str(D0.date()), mid=str(DM.date()), since2019_from=str(D19.date()), since2024_from=str(D24.date()), end=str(DE.date())),
       'bench_0050': bw, 'combos': {}, 'lists': {}}
navs = {'0050': bench}; trades = []
for d in DEFS:
    for cid in CELLS:
        res['combos'][f'{d}|{cid}'] = dict(defn=d, cell=cid, paths={}, paths_dd={}, paths_dd_dates={}, paths_1923={}, paths_24={}, paths_seg={}, paths_top5={},
                                           paths_mf={str(m): {} for m in MF[1:]})
for s in range(4):
    tc = time.time()
    for d in DEFS:
        S = s10lib.make_signal_s10(P, U, RTD[d], shift=s)
        if s == 0:
            top20 = {}
            for x in S['info']:
                k = x['k']; top20[str(x['sig_date'])] = [str(P.COLS[j]) for j in np.flatnonzero(~S['X'][k])]
            res['lists'][d] = dict(targets={x['sig_date']: x['targets'] for x in S['info']}, top20=top20)
        for cid, c in CELLS.items():
            R = res['combos'][f'{d}|{cid}']
            for cap in CAPS:
                key = f'{int(cap/1e4)}萬|+{s}'
                for mf in MF:
                    nav, lg, st = sim(c, S, capital=cap, min_frac=mf)
                    if mf != 0.5:
                        R['paths_mf'][str(mf)][key] = cg(nav, 'since2019')
                        del nav, lg, st
                        continue
                    R['paths'][key] = cg(nav, 'since2019'); R['paths_dd'][key] = mdd(nav)
                    R['paths_1923'][key] = cg(nav, 'y2019_2023'); R['paths_24'][key] = cg(nav, 'since2024')
                    R['paths_seg'][key] = s17lib.seg_dd(nav); R['paths_top5'][key] = s17lib.top5_share(lg, nav, cap)
                    ddd = s10lib.dd_dates(nav); R['paths_dd_dates'][key] = [ddd['peak'], ddd['trough']]
                    if s == 0 and cap == 1e6:
                        name = f'{d}|{cid}'
                        navs[name] = nav
                        w = W(nav)
                        R['windows'] = slim(w)
                        R['sharpe'] = {k: w[k]['sharpe'] for k in ('full', 'since2019')}
                        R['t_month_excess'] = {k: w[k]['t_month_excess'] for k in ('full', 'since2019')}
                        R['dd_dates'] = ddd; R['seg'] = s17lib.seg_dd(nav); R['top5_share'] = s17lib.top5_share(lg, nav, cap)
                        R['yearly'] = stlib.yearly(nav, bench)
                        R['trades_full'] = stlib.trade_stats(lg); R['trades_since2019'] = stlib.trade_stats(lg, '2019-01-01')
                        R['n_buy'] = int(st['n_buy']); R['avg_npos'] = float(st['avg_npos']); R['blk_min'] = int(st['blk_min']); R['blk_limup'] = int(st['blk_limup'])
                        mnav = float(nav.mean())
                        R['turnover_oneway_ann'] = float((st['buy_val'] + st['sell_val']) / 2 / mnav / YRS)
                        R['cost_ann_pct_of_nav'] = float((st['fee'] + st['tax'] + st['slip'] + st['e50_cost']) / mnav / YRS)
                        R['avg_stockfrac'] = float(np.nanmean(st['stockfrac'].to_numpy()))
                        x = lg.copy(); x['run'] = name
                        trades.append(x[['run', 'code', 'sig_date', 'ent_date', 'exit_date', 'hold_days', 'shares', 'cost', 'proceeds', 'pnl', 'ret', 'why', 'score']])
                        nx2 = sim(c, S, cost='strict_x2')[0]
                        R['costx2'] = slim(W(nx2), ('since2019', 'full'))
                        blk, top = s2lib.block_from(lg, 5)
                        nb = sim(c, S, block=blk)[0]
                        R['block5'] = slim(W(nb), ('since2019', 'full'))
                        R['block5_top'] = [dict(code=y.code, ent=str(y.ent_date.date()), exit=str(y.exit_date.date()), ret=float(y.ret), pnl=float(y.pnl)) for y in top.itertuples()]
                        del nx2, nb
                    del nav, lg, st
        del S; gc.collect()
    print('位移', s, f'{time.time()-tc:.0f}s 累計 {time.time()-t_all:.0f}s {rss()}MB', flush=True)

# ---------------------------------------------------------------- 彙整、八項、分類
for key, R in res['combos'].items():
    ks = list(R['paths'])
    pv = np.array([R['paths'][k] for k in ks]); pdd = np.array([R['paths_dd'][k] for k in ks])
    assert len(pv) == 20
    R['paths20'] = dict(min=float(pv.min()), median=float(np.median(pv)), max=float(pv.max()), n_above_0050=int((pv > bw['since2019']['cagr']).sum()),
                        maxdd_min=float(pdd.min()), maxdd_median=float(np.median(pdd)), maxdd_max=float(pdd.max()),
                        median_1923=float(np.median([R['paths_1923'][k] for k in ks])), median_24=float(np.median([R['paths_24'][k] for k in ks])),
                        seg_median={sg: float(np.median([R['paths_seg'][k][sg] for k in ks])) for sg in SEGS},
                        top5_median=float(np.median([R['paths_top5'][k] for k in ks])),
                        n_dd_2024_07_to_2025_04=int(sum(1 for v in R['paths_dd_dates'].values() if '2024-06' <= v[0][:7] <= '2024-08' and v[1][:7] == '2025-04')))
    w = R['windows']; x2 = R['costx2']; b5 = R['block5']
    chk = dict(net_since2019=w['since2019']['diff'] > 0, net_full=w['full']['diff'] > 0, costx2_since2019=x2['since2019']['diff'] > 0, costx2_full=x2['full']['diff'] > 0,
               block5_since2019=b5['since2019']['diff'] > 0, block5_full=b5['full']['diff'] > 0, half1=w['half1']['diff'] > 0, half2=w['half2']['diff'] > 0)
    R['checks'] = {k: bool(v) for k, v in chk.items()}
    R['class'] = '值得上XQ測' if all(chk.values()) else ('不值得' if not (chk['net_since2019'] or chk['net_full']) else '有條件')
    R['fails'] = [k for k, v in chk.items() if not v]


def rank_hi(vals, tid):
    """名次：比它大的個數＋1（1＝最高）。回撤是負數，數值越大＝越淺（1＝最淺）。"""
    return 1 + sum(1 for k, x in vals.items() if k != tid and x > vals[tid])


def judge_ret(dv):
    npos = int((dv > 0).sum()); med = float(np.median(dv)) * 100
    if npos >= 15 and med >= 2:
        return '比標準好', npos, med
    if npos <= 5 and med <= -2:
        return '比標準差', npos, med
    return '差不多', npos, med


def judge_seg(dv, thr):
    n = int((dv > 0).sum()); med = float(np.median(dv)) * 100
    if n >= 15 and med >= thr:
        return '較淺', n, med
    if n <= 5 and med <= -thr:
        return '較深', n, med
    return '差不多', n, med


fam = {}; pair = {}
for cid in CELLS:
    g = lambda d: res['combos'][f'{d}|{cid}']
    pm = {d: g(d)['paths20']['median'] for d in DEFS}
    pdm = {d: g(d)['paths20']['maxdd_median'] for d in DEFS}
    m19 = {d: g(d)['windows']['since2019']['cagr'] for d in DEFS}
    allp = np.array([v for d in DEFS for v in g(d)['paths'].values()])
    allp_dd = np.array([v for d in DEFS for v in g(d)['paths_dd'].values()])
    fmed = float(np.median(list(pm.values())))
    fam[cid] = dict(path_median=pm, path_maxdd_median=pdm, main_r19=m19,
                    P1=dict(family_median_of_path_medians=fmed, min=float(min(pm.values())), min_def=min(pm, key=pm.get), max=float(max(pm.values())), max_def=max(pm, key=pm.get),
                            paths140_median=float(np.median(allp)), paths140_p10=float(np.percentile(allp, 10)), paths140_n_above_0050=int((allp > bw['since2019']['cagr']).sum()),
                            paths140_dd_median=float(np.median(allp_dd)), paths140_dd_p10=float(np.percentile(allp_dd, 10)),
                            main_r19_median=float(np.median(list(m19.values())))),
                    P2=dict(STD_rank_path_median=rank_hi(pm, 'STD'), STD_rank_path_dd=rank_hi(pdm, 'STD'), STD_rank_main_r19=rank_hi(m19, 'STD'),
                            order_path_median=sorted(pm, key=pm.get, reverse=True), order_path_dd=sorted(pdm, key=pdm.get, reverse=True)),
                    P3_STD_minus_family_median_pp=(pm['STD'] - fmed) * 100,
                    seg_median={d: g(d)['paths20']['seg_median'] for d in DEFS})
    keys = list(g('STD')['paths'].keys())
    for d in DEFS[1:]:
        dv = np.array([g(d)['paths'][k] - g('STD')['paths'][k] for k in keys])
        d1 = np.array([g(d)['paths_1923'][k] - g('STD')['paths_1923'][k] for k in keys])
        d2 = np.array([g(d)['paths_24'][k] - g('STD')['paths_24'][k] for k in keys])
        ddv = np.array([g(d)['paths_dd'][k] - g('STD')['paths_dd'][k] for k in keys])
        vr, npos, med = judge_ret(dv)
        mf = {}
        for m in MF[1:]:
            dm = np.array([g(d)['paths_mf'][str(m)][k] - g('STD')['paths_mf'][str(m)][k] for k in keys])
            a, b, cc = judge_ret(dm)
            mf[str(m)] = dict(verdict_ret=a, npos=b, median_pp=cc)
        vdd, nsh, mdd_pp = judge_seg(ddv, 2.0)
        seg = {}
        for sg in SEGS:
            x = np.array([g(d)['paths_seg'][k][sg] - g('STD')['paths_seg'][k][sg] for k in keys])
            v, n, mm = judge_seg(x, 1.0)
            seg[sg] = dict(verdict=v, n_shallower=n, median_pp=mm)
        pair[f'{d}|{cid}'] = dict(verdict_ret=vr, npos=npos, median_pp=med, min_pp=float(dv.min()) * 100, max_pp=float(dv.max()) * 100,
                                  y2019_2023=dict(npos=int((d1 > 0).sum()), median_pp=float(np.median(d1)) * 100),
                                  since2024=dict(npos=int((d2 > 0).sum()), median_pp=float(np.median(d2)) * 100),
                                  minfrac=mf, dd_verdict=('能壓回撤' if vdd == '較淺' else '不能'), dd_n_shallower=nsh, dd_median_pp=mdd_pp, seg=seg,
                                  main_diff_r19_pp=(g(d)['windows']['since2019']['cagr'] - g('STD')['windows']['since2019']['cagr']) * 100,
                                  main_diff_dd_pp=(g(d)['windows']['full']['maxdd'] - g('STD')['windows']['full']['maxdd']) * 100)
stable = {}
for d in DEFS[1:]:
    pr = [pair[f'{d}|{c}'] for c in CELLS]
    both = all(p['verdict_ret'] == '比標準好' for p in pr)
    one = any(p['verdict_ret'] == '比標準好' for p in pr)
    halves = all(p['y2019_2023']['median_pp'] > 0 and p['since2024']['median_pp'] > 0 for p in pr)
    mf_ok = all(p['minfrac'][str(m)]['verdict_ret'] == '比標準好' for p in pr for m in MF[1:])
    if both and halves and mf_ok:
        stable[d] = '穩定地比較好'
    elif both and not halves:
        stable[d] = '比標準好但只靠一段'
    elif both:
        stable[d] = '比標準好但對最小買進門檻敏感'
    elif one:
        stable[d] = '只有一格較好'
    else:
        stable[d] = '否'
r = [fam[c]['P2']['STD_rank_path_median'] for c in CELLS]
v_best = '是' if all(x == 1 for x in r) else ('前段' if all(x <= 3 for x in r) else '不是')
optim = {c: ('標準定義偏樂觀' if fam[c]['P3_STD_minus_family_median_pp'] >= 3 else '不必因定義打折') for c in CELLS}
res['family'] = fam; res['pairs'] = pair
res['verdict'] = dict(STD_is_best=v_best, STD_rank_path_median={c: fam[c]['P2']['STD_rank_path_median'] for c in CELLS},
                      STD_rank_path_dd={c: fam[c]['P2']['STD_rank_path_dd'] for c in CELLS},
                      STD_minus_family_median_pp={c: fam[c]['P3_STD_minus_family_median_pp'] for c in CELLS}, optimism=optim,
                      reestimate={c: (dict(paths140_median=fam[c]['P1']['paths140_median'], paths140_p10=fam[c]['P1']['paths140_p10']) if optim[c] == '標準定義偏樂觀' else '沿用 STD') for c in CELLS},
                      stable_better=stable, any_stable_better=[d for d, v in stable.items() if v == '穩定地比較好'],
                      per_cell_ret={k: v['verdict_ret'] for k, v in pair.items()},
                      per_cell_seg={k: {sg: v['seg'][sg]['verdict'] for sg in SEGS} for k, v in pair.items()},
                      xq_change=[d for d, v in stable.items() if v == '穩定地比較好'] or '維持現行 SURGE 算法',
                      all_combos_class={k: R['class'] for k, R in res['combos'].items()},
                      class_fails={k: R['fails'] for k, R in res['combos'].items() if R['fails']})
print('判定', json.dumps(res['verdict'], ensure_ascii=False), flush=True)
for c in CELLS:
    print(c, 'P1', {a: (round(b, 4) if isinstance(b, float) else b) for a, b in fam[c]['P1'].items()}, 'P2', fam[c]['P2'], 'P3', round(fam[c]['P3_STD_minus_family_median_pp'], 2), flush=True)
for k, v in pair.items():
    print(k, v['verdict_ret'], v['npos'], round(v['median_pp'], 2), '| 19-23', v['y2019_2023'], '| 24+', v['since2024'], '| dd', v['dd_n_shallower'], round(v['dd_median_pp'], 2),
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
