#!/usr/bin/env python3
"""S39 第 3 步：首次值版 SURGE（FPM＝只有當月 D 用首次值；FPA＝能用首次值的月份都用）vs 現行（CUR）。
(a) 驗收：本題包裝的 CUR 與 s17lib.rev_family 逐位相同；引擎 48.99／34.54／−32.11／381；FP 版 2024-07-01 前淨值與 CUR 逐位相同。
(b) 名單：每個訊號月份（2024-07～2026-10）L_MA 前 20 名、F1、E0 賣出名單的差異；SURGE 數值差。
(c) 引擎：標準路徑（100 萬、位移 0；另成本加倍）與 20 條路徑（5 本金 × 4 位移），窗 2024-07-01 → 2026-10-22。
輸出 results.json 的 lists／engine／acceptance 鍵、data/list_diff.csv、data/trades_diff.csv。python3 -I -B run.py"""
import sys, json, time, resource, gc
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
import s39lib
from s39lib import s17lib, s10lib, s4lib, stlib, COSTS

H = Path(__file__).resolve().parent
RES = H / 'results.json'
res = json.loads(RES.read_text()) if RES.exists() else {}
t0 = time.time()
rss = lambda: round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)

P = s39lib.load()
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

# ---------------------------------------------------------------- (a) SURGE 三版
SV = {}
for mode in ('CUR', 'FPM', 'FPA'):
    SV[mode] = s39lib.surge_variant(P, F, fp, mode)
    print('surge', mode, f'{time.time()-t0:.0f}s', rss(), 'MB', flush=True)
ref = F['S']['STD']
acc = dict(cur_wrapper_bitwise=bool(np.array_equal(SV['CUR'].to_numpy(), ref.to_numpy(), equal_nan=True)
                                    and SV['CUR'].index.equals(ref.index) and SV['CUR'].columns.equals(ref.columns)))
first = pd.Period('2024-06', 'M')
for mode in ('FPM', 'FPA'):
    A, B = SV[mode].to_numpy(), ref.to_numpy()
    pre = SV[mode].index < first
    diff = ~((A == B) | (np.isnan(A) & np.isnan(B)))
    acc[f'{mode}_rows_before_2024-06_identical'] = bool(not diff[pre].any())
    acc[f'{mode}_cells_changed'] = int(diff.sum())
    acc[f'{mode}_months_changed'] = sorted(set(str(m) for m, r in zip(SV[mode].index, diff.any(axis=1)) if r))
assert acc['cur_wrapper_bitwise'], 'CUR 包裝未逐位重現'
av = F['av']
RT = {k: dict(S=v, av=av) for k, v in SV.items()}
del SV
gc.collect()

lastb = lambda d: P.IDX[P.IDX < pd.Timestamp(d)][-1]
TA, TB = lastb('2024-07-01'), P.IDX[-1]
D19 = lastb('2019-01-01')
CELL = dict(sat_w=0.7, sizing='W10', lock=False)
CAPS = (6e5, 8e5, 1e6, 1.5e6, 2e6)


def sim(S1, cap=1e6, cost='strict'):
    S2, ekw = s10lib.variant_S(S1, 'E0')
    return s10lib.simulate_s10(P, S2, U, cost=COSTS[cost], capital=cap, **CELL, **ekw)


SIG = {k: s10lib.make_signal_s10(P, U, RT[k], shift=0) for k in RT}
main = {}
for k in RT:
    nav, lg, st = sim(SIG[k])
    navx = sim(SIG[k], cost='strict_x2')[0]
    lg = lg.copy(); lg['code'] = [P.COLS[j] for j in lg['j']]
    lg['ent_date'] = P.IDX[lg['ent_t'].to_numpy()]; lg['exit_date'] = P.IDX[lg['exit_t'].to_numpy()]
    main[k] = dict(nav=nav, lg=lg, navx=navx, st=st)
    print('main', k, f'{time.time()-t0:.0f}s', rss(), 'MB', flush=True)
nc, lc = main['CUR']['nav'], main['CUR']['lg']
a0 = dict(since2019=stlib.cagr(nc, D19, TB), full=stlib.cagr(nc, P.IDX[P.t0], TB), maxdd=stlib.maxdd(nc, P.IDX[P.t0], TB), n_trades=int((lc['why'] != 'open_end').sum()))
a0['ok'] = bool(abs(a0['since2019'] - 0.4899) < 0.00025 and abs(a0['full'] - 0.3454) < 0.00025 and abs(a0['maxdd'] + 0.3211) < 0.00025 and a0['n_trades'] == 381)
acc['A0_engine_current'] = a0
for k in ('FPM', 'FPA'):
    acc[f'{k}_nav_identical_before_window'] = bool(np.array_equal(main[k]['nav'].loc[:TA].to_numpy(), nc.loc[:TA].to_numpy()))
    # 第一個不同的淨值日
    dd = np.flatnonzero(main[k]['nav'].to_numpy() != nc.to_numpy())
    acc[f'{k}_first_nav_diff_date'] = str(main[k]['nav'].index[dd[0]].date()) if len(dd) else None
res['acceptance'] = acc
assert a0['ok'], f'A0 失敗 {a0}'
print(json.dumps(acc, ensure_ascii=False, default=str)[:3000], flush=True)

# ---------------------------------------------------------------- (b) 名單差異
months = [m for m in s39lib.MONTHS if av[m] < P.T]
hold_rows = []
lists = []
for m in months:
    k = av[m]
    rec = dict(rev_month=str(m), sig_date=str(P.IDX[k].date()))
    sets = {}
    for v in RT:
        s = RT[v]['S'].loc[m].to_numpy()
        fin = U[k] & np.isfinite(s); ok_ma = fin & P.F_MA60[k]
        o_ma = np.flatnonzero(ok_ma)[np.argsort(-s[ok_ma], kind='stable')]
        t20 = [P.COLS[j] for j in o_ma[:20]]
        f1 = [x for x in SIG[v]['info'] if x['k'] == k][0]['targets']
        sets[v] = dict(t20=t20, f1=f1, n_lma=int(ok_ma.sum()), n_univ=int(fin.sum()))
    # E0 賣出（a）：現行版在訊號列的持股，套各版的前 20 名
    held = lc[(lc['ent_t'] <= k) & (lc['exit_t'] > k)]['code'].tolist()
    for v in RT:
        sets[v]['sell_a'] = sorted(c for c in held if c not in set(sets[v]['t20']))
        lg = main[v]['lg']
        sets[v]['sell_b'] = sorted(lg[(lg['exit_t'] == k + 1) & (lg['why'] == 'rule')]['code'].tolist())
        sets[v]['buy_b'] = sorted(lg[lg['ent_t'] == k + 1]['code'].tolist())
    sc = RT['CUR']['S'].loc[m].to_numpy()
    for v in ('FPM', 'FPA'):
        for key in ('t20', 'f1', 'sell_a', 'sell_b', 'buy_b'):
            a, b = set(sets[v][key]), set(sets['CUR'][key])
            rec[f'{v}_{key}_added'] = sorted(a - b); rec[f'{v}_{key}_removed'] = sorted(b - a)
            rec[f'{v}_{key}_ndiff'] = len(a - b) + len(b - a)
        rec[f'{v}_t20_order_changed'] = sets[v]['t20'] != sets['CUR']['t20']
        sv = RT[v]['S'].loc[m].to_numpy()
        inu = U[k] & (np.isfinite(sc) | np.isfinite(sv))
        ch = inu & ~((sv == sc) | (np.isnan(sv) & np.isnan(sc)))
        rec[f'{v}_surge_changed_in_univ'] = int(ch.sum())
        rec[f'{v}_surge_absdiff_max'] = float(np.nanmax(np.abs(sv[ch] - sc[ch]))) if ch.any() else 0.0
    rec['CUR_f1'] = sets['CUR']['f1']; rec['CUR_t20'] = sets['CUR']['t20']; rec['CUR_held'] = sorted(held)
    lists.append(rec)
L = pd.DataFrame(lists)
summ = {}
for v in ('FPM', 'FPA'):
    o = {}
    for key in ('t20', 'f1', 'sell_a', 'sell_b', 'buy_b'):
        nd = L[f'{v}_{key}_ndiff']
        na = L[f'{v}_{key}_added'].map(len)
        o[key] = dict(months_any_diff=int((nd > 0).sum()), n_months=int(len(L)), total_added=int(na.sum()), total_ndiff=int(nd.sum()),
                      per_month_added={r: int(x) for r, x in zip(L['rev_month'], na) if x})
    o['t20_order_changed_months'] = int(L[f'{v}_t20_order_changed'].sum())
    o['surge_changed_cells_in_univ_total'] = int(L[f'{v}_surge_changed_in_univ'].sum())
    summ[v] = o
res['lists'] = dict(summary=summ, note='added＝該版有、現行沒有；ndiff＝多出＋少了。sell_a＝以現行版持股套各版前 20 名；sell_b／buy_b＝各版自己引擎在 k＋1 的實際 rule 出場／進場。')
L.to_csv(H / 'data' / 'list_diff.csv', index=False)

# ---------------------------------------------------------------- (c) 引擎
bench = pd.Series(P.E50[P.t0:], index=P.IDX[P.t0:])
eng = dict(window=[str(TA.date()), str(TB.date())], bench_0050=dict(cagr=stlib.cagr(bench, TA, TB), maxdd=stlib.maxdd(bench, TA, TB)))
std = {}
for v in RT:
    nav, lg, navx = main[v]['nav'], main[v]['lg'], main[v]['navx']
    w = lg[lg['ent_date'] > TA]
    std[v] = dict(cagr_window=stlib.cagr(nav, TA, TB), maxdd_window=stlib.maxdd(nav, TA, TB), cagr_window_costx2=stlib.cagr(navx, TA, TB),
                  since2019=stlib.cagr(nav, D19, TB), full=stlib.cagr(nav, P.IDX[P.t0], TB), maxdd_full=stlib.maxdd(nav, P.IDX[P.t0], TB),
                  n_trades_entered_window=int(len(w)), n_closed_window=int((w['why'] != 'open_end').sum()), n_closed_all=int((lg['why'] != 'open_end').sum()),
                  end_nav=float(nav.iloc[-1]))
key = lambda d: set(zip(d['code'], d['ent_t'], d['exit_t']))
tdiff = []
for v in ('FPM', 'FPA'):
    a, b = main[v]['lg'], lc
    a, b = a[a['ent_date'] > TA], b[b['ent_date'] > TA]
    ka, kb = key(a), key(b)
    std[v]['trades_only_in_this'] = len(ka - kb); std[v]['trades_only_in_CUR'] = len(kb - ka)
    std[v]['d_cagr_window_pp'] = (std[v]['cagr_window'] - std['CUR']['cagr_window']) * 100
    std[v]['d_maxdd_window_pp'] = (std[v]['maxdd_window'] - std['CUR']['maxdd_window']) * 100
    std[v]['d_cagr_window_costx2_pp'] = (std[v]['cagr_window_costx2'] - std['CUR']['cagr_window_costx2']) * 100
    std[v]['d_since2019_pp'] = (std[v]['since2019'] - std['CUR']['since2019']) * 100
    for nm, d, ks in ((v, a, ka - kb), ('CUR', b, kb - ka)):
        for y in d.itertuples():
            if (y.code, y.ent_t, y.exit_t) in ks:
                tdiff.append(dict(variant=v, side=nm, code=y.code, ent=str(y.ent_date.date()), exit=str(y.exit_date.date()), why=y.why,
                                  ret=float(y.proceeds / y.cost - 1) if y.cost else None))
eng['standard'] = std
pd.DataFrame(tdiff).to_csv(H / 'data' / 'trades_diff.csv', index=False)
# 年度
eng['yearly'] = {v: stlib.yearly(main[v]['nav'].loc[TA:], bench) for v in RT}
del main
gc.collect()
print('standard', json.dumps({v: {k2: x for k2, x in d.items() if k2.startswith('d_') or k2.startswith('trades') or k2 == 'cagr_window'} for v, d in std.items()}, ensure_ascii=False), flush=True)

PATH = {v: {} for v in RT}
for s in range(4):
    Ss = SIG if s == 0 else {v: s10lib.make_signal_s10(P, U, RT[v], shift=s) for v in RT}
    for cap in CAPS:
        kk = f'{int(cap/1e4)}萬|+{s}'
        for v in RT:
            nav, lg, _ = sim(Ss[v], cap)
            PATH[v][kk] = dict(cagr=stlib.cagr(nav, TA, TB), maxdd=stlib.maxdd(nav, TA, TB), n_tr=int((lg['ent_t'] > P.IDX.get_loc(TA)).sum()))
    if s:
        del Ss
    gc.collect()
    print('shift', s, f'{time.time()-t0:.0f}s', rss(), 'MB', flush=True)
keys = list(PATH['CUR'])
pairs = {}
for v in ('FPM', 'FPA'):
    o = {}
    for f in ('cagr', 'maxdd'):
        d = np.array([PATH[v][k][f] - PATH['CUR'][k][f] for k in keys]) * 100
        o[f] = dict(median_pp=float(np.median(d)), mean_pp=float(d.mean()), min_pp=float(d.min()), max_pp=float(d.max()),
                    n_lower=int((d < -1e-9).sum()), n_higher=int((d > 1e-9).sum()), n_equal=int((np.abs(d) <= 1e-9).sum()))
    dn = np.array([PATH[v][k]['n_tr'] - PATH['CUR'][k]['n_tr'] for k in keys])
    o['n_trades_diff'] = dict(median=float(np.median(dn)), min=int(dn.min()), max=int(dn.max()))
    pairs[v] = o
eng['paths'] = PATH
eng['paths_pairs'] = pairs
eng['paths_median_cagr'] = {v: float(np.median([PATH[v][k]['cagr'] for k in keys])) for v in PATH}
res['engine'] = eng
res['run_runtime_s'] = round(time.time() - t0, 1); res['run_maxrss_mb'] = rss()
RES.write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str))
print(json.dumps(dict(pairs=pairs, med=eng['paths_median_cagr'], lists=summ), ensure_ascii=False, indent=1, default=str)[:8000])
print('done', res['run_runtime_s'], 's', res['run_maxrss_mb'], 'MB')
