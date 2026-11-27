#!/usr/bin/env python3
"""S102 第 0 步：原樣重現正式基準（2015-07 起全期 34.46%、2019 起 48.86%、381 筆、與 navs_official 逐日差 0、20 條路徑中位數 46.42%）；
驗收 A1：s102sim.simulate_u(rebuy=None) 與 s10lib.simulate_s10 逐日逐筆相同；A2：make_signal_u(fill=None) 與 make_signal_s10 的 sig／rank／X／sigrows 相同；
A3：U1 遞補不改 X、U2 不改訊號。只數「遞補幾檔、實際買幾檔、破線賣出幾次、觸發／買回幾次、平均持股比例」——不印、不存任何變體的報酬。
用法：python3 -I -B step0.py  → step0.json"""
import sys, json, time, resource, gc
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import s102lib as L
from s102lib import stlib, s10lib, COSTS
import s102sim
t0 = time.time(); rss = lambda: round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
P, U, RT = L.load()
D0, DE = P.IDX[P.t0], P.IDX[-1]
lastb = lambda d: P.IDX[P.IDX < pd.Timestamp(d)][-1]
D19 = lastb('2019-01-01'); TA, TB = lastb('2016-02-01'), lastb('2022-01-01')
out = dict(panel=dict(T=int(P.T), N=int(P.N), start=str(D0.date()), end=str(DE.date())), target=L.TARGET)
paths = {}; counts = {}
inwin = lambda d, a, b: (d > a) & (d <= b)
for s in L.SHIFTS:
    S = L.signal(P, U, RT, s)
    if s == 0:
        S10 = s10lib.make_signal_s10(P, U, RT, shift=0)
        a2 = bool(np.array_equal(S['sig'], S10['sig']) and np.array_equal(S['rank'], S10['rank'], equal_nan=True) and np.array_equal(S['X'], S10['X'])
                  and np.array_equal(S['sigrows'], S10['sigrows']) and [x['k'] for x in S['info']] == [x['k'] for x in S10['info']])
        nav, lg, st = L.sim(P, S10, U)
        cl = lg[lg['why'] != 'open_end']
        ref = pd.read_csv('/home/user/-/xq_scripts/site/navs_official.csv.gz', index_col=0, parse_dates=True)['E0|W10_70']
        mine = nav.round(2); same = bool(mine.index.equals(ref.index))
        sf = st['stockfrac']
        out['std'] = dict(since2019=stlib.cagr(nav, D19, DE), full=stlib.cagr(nav, D0, DE), maxdd=stlib.maxdd(nav, D0, DE), n_closed=int(len(cl)), n_all=int(len(lg)),
                          nav_same_index=same, nav_max_abs_diff=float((mine - ref).abs().max()) if same else None, n_signal_months=len(S['info']),
                          avg_stockfrac=float(np.nanmean(sf.to_numpy())), avg_stockfrac_train=float(np.nanmean(sf.loc[TA:TB].to_numpy())), avg_stockfrac_valid=float(np.nanmean(sf.loc[TB:DE].to_numpy())))
        kw = dict(sat_w=0.7, sizing='W10', lock=False, cost=COSTS['strict'], capital=1e6)
        n2, l2, s2 = s102sim.simulate_u(P, S, U, **kw)
        cols = ['j', 'sig_t', 'ent_t', 'exit_t', 'cost', 'proceeds', 'why', 'shares']
        a1 = bool(np.array_equal(nav.to_numpy(), n2.to_numpy(), equal_nan=True) and lg[cols].equals(l2[cols]))
        out['A1_sim_copy_identical'] = a1; out['A2_signal_copy_identical'] = a2
        n_lots_rule = int((lg['why'] == 'rule').sum())
        del n2, l2, s2, S10
        for c, cfg in L.CELLS.items():
            Sc = L.signal(P, U, RT, 0, c)
            nv, lgc, stc = L.sim(P, Sc, U, cell=c)
            f = stc['stockfrac']
            o = dict(n_positions=int(len(lgc)), avg_stockfrac=float(np.nanmean(f.to_numpy())), avg_stockfrac_train=float(np.nanmean(f.loc[TA:TB].to_numpy())), avg_stockfrac_valid=float(np.nanmean(f.loc[TB:DE].to_numpy())),
                     n_positions_train=int(inwin(lgc['sig_date'], TA, TB).sum()), n_positions_valid=int((lgc['sig_date'] > TB).sum()), noslot=int(stc['noslot']))
            if cfg['topic'] == 'U1':
                o['A3_X_unchanged'] = bool(np.array_equal(Sc['X'], S['X']) and np.array_equal(Sc['sigrows'], S['sigrows']))
                fk = Sc['FILL']
                fd = pd.Series({P.IDX[k]: len(v) for k, v in fk.items()}, dtype=float)
                bought = lgc[[int(j) in set(fk.get(int(k), [])) for j, k in zip(lgc['j'], lgc['sig_t'])]]
                o.update(n_months_short=int(sum(1 for x in Sc['info'] if x['n_orig'] < 10)), n_months_with_fill=int(len(fk)), n_fill_names=int(fd.sum()),
                         n_fill_names_train=int(fd[(fd.index > TA) & (fd.index <= TB)].sum()), n_fill_names_valid=int(fd[fd.index > TB].sum()),
                         n_fill_bought=int(len(bought)), n_fill_bought_train=int(inwin(bought['sig_date'], TA, TB).sum()), n_fill_bought_valid=int((bought['sig_date'] > TB).sum()))
            else:
                o['A3_signal_unchanged'] = bool(np.array_equal(Sc['sig'], S['sig']))
                rb = stc['rebuys']; d = P.IDX[rb['t'].to_numpy()] if len(rb) else pd.DatetimeIndex([])
                o.update({k: int(stc[k]) for k in ('rb_flag', 'rb_win', 'rb_trig', 'rb_buy', 'rb_fail', 'rb_noslot')}, n_rule_exits_current=n_lots_rule,
                         rb_buy_train=int(((d > TA) & (d <= TB)).sum()), rb_buy_valid=int((d > TB).sum()))
            counts[c] = o
            del Sc, nv, lgc, stc; gc.collect()
        del nav, lg, st
    for cap in L.CAPS:
        nav = L.sim(P, S, U, cap=cap)[0]
        paths[f'{int(cap/1e4)}萬|+{s}'] = stlib.cagr(nav, D19, DE); del nav
    del S; gc.collect()
out['paths_since2019'] = paths; out['paths_median'] = float(np.median(list(paths.values())))
out['variant_counts_shift0_std（只數筆數與持股比例，未算報酬）'] = counts
a = out['std']
out['acceptance'] = dict(since2019=abs(a['since2019'] - L.TARGET['since2019']) < 1e-9, full=abs(a['full'] - L.TARGET['full']) < 1e-9, maxdd=abs(a['maxdd'] - L.TARGET['maxdd']) < 1e-9,
                         n_closed=a['n_closed'] == 381, nav_diff0=a['nav_max_abs_diff'] == 0, paths_median=abs(out['paths_median'] - L.TARGET['paths_median']) < 1e-9,
                         A1=out['A1_sim_copy_identical'], A2=out['A2_signal_copy_identical'],
                         A3=all(v.get('A3_X_unchanged', v.get('A3_signal_unchanged')) for v in counts.values()))
out['acceptance']['pass'] = bool(all(out['acceptance'].values()))
out['meta'] = dict(runtime_s=round(time.time() - t0, 1), maxrss_mb=rss())
(HERE / 'step0.json').write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str))
print(json.dumps({k: out[k] for k in ('panel', 'std', 'paths_median', 'acceptance', 'meta')}, ensure_ascii=False, default=str))
for c, v in counts.items():
    print(c, v)
assert out['acceptance']['pass'], '基準重現或驗收失敗'
