#!/usr/bin/env python3
"""S40 第 3 步：可得日修正版 vs 現行（CUR）。
變體：DL（2019-01、2023-03 訊號列改到延後期限後）、DL20（再加 2020-01 最壞）、PM／PMS／PMW（2024-06～2026-09 看不到的股月 SURGE＝NaN；主／確定／最壞）、
PST（看不到的改用上月 SURGE，敏感度）、ALL＝DL＋PM。
位移 s＞0 的路徑：看不到的判斷改用位移後的訊號日（已知 T → T 晚於該日當晚才算；從未見到 → INS15 以法定 15 日期限判斷，其他維持）。
輸出 results.json 的 acceptance／engine、data/trades_diff.csv。python3 -I -B run.py"""
import sys, json, time, resource, gc
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
import s40lib as L
from s40lib import s17lib, s10lib, s4lib, stlib, cslib, COSTS

H = Path(__file__).resolve().parent
RES = H / 'results.json'
res = json.loads(RES.read_text()) if RES.exists() else {}
t0 = time.time()
rss = lambda: round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)

P = L.load()
assert (P.T, P.N, str(P.IDX[-1].date())) == (2876, 2142, '2026-10-22'), 'P0 失敗'
INNOV, _ = s4lib.innov_panel(P)
U = stlib.universe(P, 'twse') & ~INNOV
del INNOV
F = s17lib.rev_family(P, defs=['STD'])
s17lib.e50_fix_suspension(P)
for a in ('H', 'L', 'VOL', 'CHG', 'HF', 'VOLF', 'adjH', 'HI240', 'MA60', 'nbar', 'F_HI', 'IDXC', 'AO', 'listed', 'NEARHI', 'V60', 'liq'):
    if hasattr(P, a):
        delattr(P, a)
gc.collect()
S0, av0 = F['S']['STD'], F['av']
# 包裝驗收：rev_avail_days 重算與 F['av'] 相同
acc = dict(av_rebuilt_equal=cslib.rev_avail_days(P, list(S0.index), True) == av0)

C = pd.read_csv(H / 'data' / 'visibility_cells.csv', dtype={'code': str})
cal = pd.read_csv(H / 'data' / 'deadline_calendar.csv').set_index('rev_month')


def late_cells(cats, shift):
    d = C[C['cat'].isin(cats)]
    out = []
    for r in d.itertuples():
        m = pd.Period(r.rev_month, 'M')
        k = av0[m] + shift
        if k >= P.T:
            continue
        sd = P.IDX[k]
        send = pd.Timestamp(sd.date()).tz_localize(L.TZ) + pd.Timedelta(hours=23, minutes=59, seconds=59)
        if isinstance(r.T, str):
            if pd.Timestamp(r.T) <= send:
                continue
        elif r.ins15 and isinstance(cal.loc[r.rev_month].get('deadline_ins15'), str):
            if pd.Timestamp(cal.loc[r.rev_month, 'deadline_ins15'][:10]) <= sd:
                continue
        out.append((m, r.code))
    return out


EXT_DL = dict(L.EXT_ALL)
EXT_DL20 = {**L.EXT_ALL, **L.EXT_2020}
VAR = {'CUR': dict(), 'DL': dict(ext=EXT_DL), 'DL20': dict(ext=EXT_DL20), 'PM': dict(cats=L.LATE_PM, mode='nan'),
       'PMS': dict(cats=L.LATE_PMS, mode='nan'), 'PMW': dict(cats=L.LATE_PMW, mode='nan'), 'PST': dict(cats=L.LATE_PM, mode='stale'),
       'ALL': dict(ext=EXT_DL, cats=L.LATE_PM, mode='nan')}
ncell = {}


def RT(v, shift):
    cfg = VAR[v]
    av = L.av_fix(P, av0, cfg['ext']) if 'ext' in cfg else av0
    S = S0
    if 'cats' in cfg:
        cells = late_cells(cfg['cats'], shift)
        S, n = L.mask_S(S0, cells, cfg['mode'])
        ncell[(v, shift)] = n
    return dict(S=S, av=av)


lastb = lambda d: P.IDX[P.IDX < pd.Timestamp(d)][-1]
TA, TB = lastb('2024-07-01'), P.IDX[-1]
D19 = lastb('2019-01-01')
CELL = dict(sat_w=0.7, sizing='W10', lock=False)
CAPS = (6e5, 8e5, 1e6, 1.5e6, 2e6)


def sim(S1, cap=1e6, cost='strict'):
    S2, ekw = s10lib.variant_S(S1, 'E0')
    return s10lib.simulate_s10(P, S2, U, cost=COSTS[cost], capital=cap, **CELL, **ekw)


def metrics(nav):
    return dict(since2019=stlib.cagr(nav, D19, TB), full=stlib.cagr(nav, P.IDX[P.t0], TB), maxdd_full=stlib.maxdd(nav, P.IDX[P.t0], TB),
                cagr_window=stlib.cagr(nav, TA, TB), maxdd_window=stlib.maxdd(nav, TA, TB))


# 驗收：CUR 包裝 S 與 av 與原版相同
r0 = RT('CUR', 0)
acc['cur_S_identical'] = bool(r0['S'] is S0)
main = {}
for v in VAR:
    rt = RT(v, 0)
    sg = s10lib.make_signal_s10(P, U, rt, shift=0)
    nav, lg, st = sim(sg)
    navx = sim(sg, cost='strict_x2')[0]
    lg = lg.copy(); lg['code'] = [P.COLS[j] for j in lg['j']]
    lg['ent_date'] = P.IDX[lg['ent_t'].to_numpy()]; lg['exit_date'] = P.IDX[lg['exit_t'].to_numpy()]
    main[v] = dict(nav=nav, lg=lg, navx=navx, info={x['month']: x for x in sg['info']})
    print('main', v, f'{time.time()-t0:.0f}s', rss(), 'MB', flush=True)
nc, lc = main['CUR']['nav'], main['CUR']['lg']
a0 = dict(since2019=stlib.cagr(nc, D19, TB), full=stlib.cagr(nc, P.IDX[P.t0], TB), maxdd=stlib.maxdd(nc, P.IDX[P.t0], TB), n_trades=int((lc['why'] != 'open_end').sum()))
a0['ok'] = bool(abs(a0['since2019'] - 0.4899) < 0.00025 and abs(a0['full'] - 0.3454) < 0.00025 and abs(a0['maxdd'] + 0.3211) < 0.00025 and a0['n_trades'] == 381)
acc['A0_engine_current'] = a0
for v in VAR:
    if v == 'CUR':
        continue
    dd = np.flatnonzero(main[v]['nav'].to_numpy() != nc.to_numpy())
    acc[f'{v}_first_nav_diff_date'] = str(main[v]['nav'].index[dd[0]].date()) if len(dd) else None
acc['n_cells_masked_shift0'] = {v: ncell.get((v, 0)) for v in VAR if (v, 0) in ncell}
res['acceptance'] = acc
assert a0['ok'] and acc['av_rebuilt_equal'], f'A0 失敗 {a0}'
print(json.dumps(acc, ensure_ascii=False, default=str), flush=True)

# 標準路徑
key = lambda d: set(zip(d['code'], d['ent_t'], d['exit_t']))
std, tdiff, sigdiff = {}, [], {}
for v in VAR:
    m = metrics(main[v]['nav']); m['cagr_window_costx2'] = stlib.cagr(main[v]['navx'], TA, TB)
    m['since2019_costx2'] = stlib.cagr(main[v]['navx'], D19, TB)
    lg = main[v]['lg']; m['n_closed_all'] = int((lg['why'] != 'open_end').sum()); m['end_nav'] = float(main[v]['nav'].iloc[-1])
    std[v] = m
for v in VAR:
    if v == 'CUR':
        continue
    for f in ('since2019', 'full', 'maxdd_full', 'cagr_window', 'maxdd_window', 'cagr_window_costx2', 'since2019_costx2'):
        std[v][f'd_{f}_pp'] = (std[v][f] - std['CUR'][f]) * 100
    a, b = main[v]['lg'], lc
    ka, kb = key(a), key(b)
    std[v]['trades_only_in_this'] = len(ka - kb); std[v]['trades_only_in_CUR'] = len(kb - ka)
    for nm, d, ks in ((v, a, ka - kb), ('CUR', b, kb - ka)):
        for y in d.itertuples():
            if (y.code, y.ent_t, y.exit_t) in ks:
                tdiff.append(dict(variant=v, side=nm, code=y.code, ent=str(y.ent_date.date()), exit=str(y.exit_date.date()), why=y.why,
                                  ret=float(y.proceeds / y.cost - 1) if y.cost else None))
    # 訊號（F1）不同的月份
    sd = {}
    for mo, x in main[v]['info'].items():
        y = main['CUR']['info'].get(mo)
        if y is None or x['targets'] != y['targets'] or x['sig_date'] != y['sig_date']:
            sd[mo] = dict(sig_date=x['sig_date'], cur_sig_date=y['sig_date'] if y else None, f1=x['targets'], cur_f1=y['targets'] if y else None)
    sigdiff[v] = sd
eng = dict(window=[str(TA.date()), str(TB.date())], since2019_start=str(D19.date()), standard=std, f1_diff_months=sigdiff)
pd.DataFrame(tdiff).to_csv(H / 'data' / 'trades_diff.csv', index=False)
eng['yearly'] = {v: {str(k): x for k, x in (main[v]['nav'].resample('YE').last().pct_change().dropna() * 100).round(2).items()} for v in VAR}
del main
gc.collect()
print('standard', json.dumps({v: {k: round(x, 3) for k, x in d.items() if k.startswith('d_')} for v, d in std.items()}, ensure_ascii=False), flush=True)

PATH = {v: {} for v in VAR}
for s in range(4):
    for v in VAR:
        sg = s10lib.make_signal_s10(P, U, RT(v, s), shift=s)
        for cap in CAPS:
            nav, lg, _ = sim(sg, cap)
            PATH[v][f'{int(cap/1e4)}萬|+{s}'] = metrics(nav)
        del sg
    gc.collect()
    print('shift', s, f'{time.time()-t0:.0f}s', rss(), 'MB', flush=True)
keys = list(PATH['CUR'])
pairs = {}
for v in VAR:
    if v == 'CUR':
        continue
    o = {}
    for f in ('since2019', 'full', 'maxdd_full', 'cagr_window', 'maxdd_window'):
        d = np.array([PATH[v][k][f] - PATH['CUR'][k][f] for k in keys]) * 100
        o[f] = dict(median_pp=float(np.median(d)), mean_pp=float(d.mean()), min_pp=float(d.min()), max_pp=float(d.max()),
                    n_lower=int((d < -1e-9).sum()), n_higher=int((d > 1e-9).sum()), n_equal=int((np.abs(d) <= 1e-9).sum()))
    pairs[v] = o
eng['paths'] = PATH
eng['paths_pairs'] = pairs
eng['paths_median'] = {v: {f: float(np.median([PATH[v][k][f] for k in keys])) for f in ('since2019', 'maxdd_full', 'cagr_window')} for v in PATH}
eng['n_cells_masked'] = {f'{v}|+{s}': n for (v, s), n in ncell.items()}
res['engine'] = eng
res['run_runtime_s'] = round(time.time() - t0, 1); res['run_maxrss_mb'] = rss()
RES.write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str))
print(json.dumps(dict(pairs=pairs, med=eng['paths_median']), ensure_ascii=False, indent=1, default=str)[:9000])
print('done', res['run_runtime_s'], 's', res['run_maxrss_mb'], 'MB')
