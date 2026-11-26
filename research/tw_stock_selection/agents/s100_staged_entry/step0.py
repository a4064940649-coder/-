#!/usr/bin/env python3
"""S100 第 0 步：原樣重現正式基準；驗收複製引擎（stage=None 與原引擎逐日逐筆相同）；只數「會影響幾筆、補買幾次、平均持股比例」——不印、不存任何變體的報酬。
用法：python3 -I -B step0.py  → step0.json"""
import sys, json, time, resource, gc
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import s100lib as L
from s100lib import stlib, s10lib, s100sim, COSTS
t0 = time.time(); rss = lambda: round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
P, U, RT = L.load()
D0, DE = P.IDX[P.t0], P.IDX[-1]
lastb = lambda d: P.IDX[P.IDX < pd.Timestamp(d)][-1]
D19 = lastb('2019-01-01'); TA, TB = lastb('2016-02-01'), lastb('2022-01-01')
out = dict(panel=dict(T=int(P.T), N=int(P.N), start=str(D0.date()), end=str(DE.date())), target=L.TARGET)
EXTRA = {'X|2/3→4/3@1': dict(init=2 / 3, final=4 / 3, check=1), 'X|1/2→4/3@1': dict(init=1 / 2, final=4 / 3, check=1), 'X|1/2→2@1': dict(init=1 / 2, final=2.0, check=1),
         'X|1/2→1.25@1': dict(init=1 / 2, final=1.25, check=1), 'X|1/2→1.75@2': dict(init=1 / 2, final=1.75, check=2), 'X|1/2→2@2': dict(init=1 / 2, final=2.0, check=2)}
paths = {}; counts = {}
for s in L.SHIFTS:
    S = L.signal(P, U, RT, s)
    if s == 0:
        nav, lg, st = L.sim(P, S, U)
        cl = lg[lg['why'] != 'open_end']
        ref = pd.read_csv('/home/user/-/xq_scripts/site/navs_official.csv.gz', index_col=0, parse_dates=True)['E0|W10_70']
        mine = nav.round(2); same = bool(mine.index.equals(ref.index))
        sf = st['stockfrac']
        out['std'] = dict(since2019=stlib.cagr(nav, D19, DE), full=stlib.cagr(nav, D0, DE), maxdd=stlib.maxdd(nav, D0, DE), n_closed=int(len(cl)), n_all=int(len(lg)),
                          nav_same_index=same, nav_max_abs_diff=float((mine - ref).abs().max()) if same else None, n_signal_months=len(S['info']),
                          avg_stockfrac=float(np.nanmean(sf.to_numpy())), avg_stockfrac_train=float(np.nanmean(sf.loc[TA:TB].to_numpy())), avg_stockfrac_valid=float(np.nanmean(sf.loc[TB:DE].to_numpy())), avg_npos=float(st['avg_npos']))
        # A1：複製引擎不開分批＝原引擎
        kw = dict(sat_w=0.7, sizing='W10', lock=False, cost=COSTS['strict'], capital=1e6)
        n2, l2, s2 = s100sim.simulate_staged(P, S, U, **kw)
        cols = ['j', 'sig_t', 'ent_t', 'exit_t', 'cost', 'proceeds', 'why', 'shares']
        a1 = bool(np.array_equal(nav.to_numpy(), n2.to_numpy(), equal_nan=True) and lg[cols].equals(l2[cols]))
        lots = s2['lots']
        a1b = bool(len(lots) == len(l2) and abs(lots['proceeds'].sum() - l2['proceeds'].sum()) < 1e-6 * l2['proceeds'].sum() and (lots['n_lots'] == 1).all())
        # A2：init＝final＝1（分批掛鉤開著但不補）＝原引擎
        n3, l3, s3 = L.sim(P, S, U, cell=dict(init=1.0, final=1.0, check=1))
        a2 = bool(np.array_equal(nav.to_numpy(), n3.to_numpy(), equal_nan=True) and lg[cols].equals(l3[cols]))
        out['A1_copy_engine_identical'] = a1; out['A1b_lots_consistent'] = a1b; out['A2_stage_1_1_identical'] = a2
        del n2, l2, s2, n3, l3, s3
        for c, cfg in list(L.CELLS.items()) + list(EXTRA.items()):
            nv, lgc, stc = L.sim(P, S, U, cell=cfg)
            f = stc['stockfrac']; tp = stc['topups']; lt = stc['lots']
            # 會計恆等：各批成本與所得加總＝部位
            chk = bool(abs(lt['cost'].sum() - lgc['cost'].sum()) < 1e-6 * lgc['cost'].sum() and abs(lt['proceeds'].sum() - lgc['proceeds'].sum()) < 1e-6 * lgc['proceeds'].sum())
            K = L.topup_K(stc)
            rng = np.random.default_rng(100)
            _, lgr, str_ = L.sim(P, S, U, cell=cfg, rand=dict(K=K, rng=rng))
            counts[c] = dict(registered=c in L.CELLS, n_positions=int(len(lgc)), n_positions_train=int(((lgc['sig_date'] > TA) & (lgc['sig_date'] <= TB)).sum()), n_positions_valid=int((lgc['sig_date'] > TB).sum()),
                             n_topups=int(len(tp)), n_topups_train=int(((P.IDX[tp['t']] > TA) & (P.IDX[tp['t']] <= TB)).sum()) if len(tp) else 0, n_topups_valid=int((P.IDX[tp['t']] > TB).sum()) if len(tp) else 0,
                             top_pend=int(stc['top_pend']), top_blk_limup=int(stc['top_blk_limup']), top_blk_min=int(stc['top_blk_min']), top_blk_noopen=int(stc['top_blk_noopen']), top_room_bind=int(stc['top_room_bind']),
                             blk_min=int(stc['blk_min']), blk_lot=int(stc['blk_lot']), blk_nocash=int(stc['blk_nocash']), noslot=int(stc['noslot']),
                             avg_stockfrac=float(np.nanmean(f.to_numpy())), avg_stockfrac_train=float(np.nanmean(f.loc[TA:TB].to_numpy())), avg_stockfrac_valid=float(np.nanmean(f.loc[TB:DE].to_numpy())),
                             max_stockfrac=float(np.nanmax(f.to_numpy())), lots_accounting_ok=chk,
                             random_ctrl_check=dict(n_positions=int(len(lgr)), n_topups=int(len(str_['topups'])), new_full=int(str_['new_full']), avg_stockfrac=float(np.nanmean(str_['stockfrac'].to_numpy()))))
            del nv, lgc, stc, lgr, str_
        out['cur_counts'] = dict(blk_min=int(st['blk_min']), blk_lot=int(st['blk_lot']), blk_nocash=int(st['blk_nocash']), noslot=int(st['noslot']), max_stockfrac=float(np.nanmax(sf.to_numpy())))
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
                         A1=out['A1_copy_engine_identical'], A1b=out['A1b_lots_consistent'], A2=out['A2_stage_1_1_identical'], lots_ok=all(v['lots_accounting_ok'] for v in counts.values()))
out['acceptance']['pass'] = bool(all(out['acceptance'].values()))
out['meta'] = dict(runtime_s=round(time.time() - t0, 1), maxrss_mb=rss())
(HERE / 'step0.json').write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str))
print(json.dumps({k: out[k] for k in ('panel', 'std', 'paths_median', 'acceptance', 'cur_counts', 'meta')}, ensure_ascii=False, default=str))
for c, v in counts.items():
    print(c, v)
assert out['acceptance']['pass'], '基準重現或驗收失敗'
