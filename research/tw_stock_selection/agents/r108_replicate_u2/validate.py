#!/usr/bin/env python3
"""R108 驗證期重算：凍結格 U2|any vs 現行，重新起跑（營收月份 ≥ 2021-12 的進場訊號；起始資金＝本金 ÷ 現行 2021-12-30 淨值倍數）× 20 條路徑，
驗證窗 2021-12-30 → 2026-10-22；V1～V4（S102 PREREG 定義）、R2 觸發檢查、買回逐筆、2022–23／2024 起分段。輸出 valid.json、valid_rebuys.csv。
用法：python3 -I -B validate.py"""
import sys, json, time, resource, gc
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import r108lib as L
t_all = time.time(); rss = lambda: round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
P, U, RT = L.load()
from s19lib import s10lib, s2lib, stlib
lastb = lambda d: P.IDX[P.IDX < pd.Timestamp(d)][-1]
D0, DE, TB, T23 = P.IDX[P.t0], P.IDX[-1], lastb('2022-01-01'), lastb('2024-01-01')
WIN = dict(valid=(TB, DE), y2022_2023=(TB, T23), since2024=(T23, DE))
cg = lambda nav, k: stlib.cagr(nav, *WIN[k])
bench = pd.Series(P.E50, index=P.IDX)
sfm = lambda st, a=TB, b=DE: float(np.nanmean(st['stockfrac'].loc[a:b].to_numpy()))
FZ = dict(r5=False)
sim = lambda S, ROWS, cap, rb, block=None: L.simulate_u2(P, S, U, ROWS, rebuy=rb, capital=cap, block=block)

S0 = s10lib.make_signal_s10(P, U, RT, shift=0); R0 = L.row_scores(P, U, RT, S0, 0)
nc, lc, sc = sim(S0, R0, 1e6, None)
ratio = float(nc.loc[TB] / 1e6); FCAP = {c: c / ratio for c in L.CAPS}
nf, lf, sf = sim(S0, R0, 1e6, FZ)
expo = dict(current=sfm(sc, D0, DE), frozen=sfm(sf, D0, DE)); expo['diff_pp'] = (expo['frozen'] - expo['current']) * 100
R2_TRIGGER = bool(abs(expo['diff_pp']) > 3.0)
res = dict(ratio_nav_TB=ratio, exposure_full_std=expo, R2_triggered=R2_TRIGGER,
           cont_std=dict(CUR_full=stlib.cagr(nc, D0, DE), FZ_full=stlib.cagr(nf, D0, DE), CUR_rb=None, FZ_rb=dict(rb_buy=sf['rb_buy'], rb_trig=sf['rb_trig'], rb_fail=sf['rb_fail'])))
del nc, lc, sc, nf, lf, sf

F0 = L.fresh_from(S0)
nC, lC, stC = sim(F0, R0, FCAP[1e6], None)
nF, lF, stF = sim(F0, R0, FCAP[1e6], FZ)
assert (lC['ent_date'] >= pd.Timestamp('2022-01-01')).all() and (lF['ent_date'] >= pd.Timestamp('2022-01-01')).all()
main = {}
for nm, nav, lg, st in (('CUR', nC, lC, stC), ('FZ', nF, lF, stF)):
    cl = lg[lg['why'] != 'open_end']
    main[nm] = dict(cagr={k: cg(nav, k) for k in WIN}, maxdd=stlib.maxdd(nav, TB, DE), seg=L.seg_dd(nav), n_closed=int(len(cl)), n_all=int(len(lg)), avg_stockfrac=sfm(st),
                    top5_by_pnl=[str(c) for c in cl.sort_values('pnl', ascending=False).head(5)['code']])
main['FZ'].update(rb_trig=stF['rb_trig'], rb_buy=stF['rb_buy'], rb_fail=stF['rb_fail'], rb_flag=stF['rb_flag'], rb_win=stF['rb_win'])
rb = lF[lF['tag'] == 'rebuy'].copy()
b = bench
rb['e0050'] = b.reindex(pd.DatetimeIndex(rb['exit_date'])).to_numpy() / b.reindex(pd.DatetimeIndex(rb['ent_date'])).to_numpy() - 1
rb['excess'] = rb['ret'] - rb['e0050']
main['FZ']['rebuys'] = dict(n=int(len(rb)), n_closed=int((rb['why'] != 'open_end').sum()), mean_ret=float(rb['ret'].mean()), median_ret=float(rb['ret'].median()), win=float((rb['ret'] > 0).mean()),
                            pnl_sum=float(rb['pnl'].sum()), mean_excess=float(rb['excess'].mean()),
                            list=[dict(code=str(r.code), ent=str(r.ent_date.date()), exit=str(r.exit_date.date()), why=r.why, ret=float(r.ret), pnl=float(r.pnl), excess=float(r.excess)) for r in rb.itertuples()])
rb[['code', 'ent_date', 'exit_date', 'why', 'cost', 'proceeds', 'ret', 'pnl', 'e0050', 'excess']].round(6).to_csv(HERE / 'valid_rebuys.csv', index=False)
res['main'] = main
BLK = {j: list(v) for j, v in s2lib.block_from(lC, 5)[0].items()}
for j, v in s2lib.block_from(lF, 5)[0].items():
    BLK.setdefault(j, []).extend(v)
res['V3_block_union'] = {str(P.COLS[j]): [[str(P.IDX[a].date()), str(P.IDX[b_].date())] for a, b_ in v] for j, v in BLK.items()}
pd.DataFrame({'CUR': nC.loc[TB:], 'FZ': nF.loc[TB:]}).round(2).to_csv(HERE / 'my_nav_valid.csv.gz')
del S0, F0, R0, nC, lC, stC, nF, lF, stF; gc.collect()

PATH = {k: {} for k in ('CUR', 'FZ', 'SCALED', 'CUR_blk', 'FZ_blk')}
for s in L.SHIFTS:
    S = s10lib.make_signal_s10(P, U, RT, shift=s); ROWS = L.row_scores(P, U, RT, S, s); Fx = L.fresh_from(S)
    for cap in L.CAPS:
        key = f'{int(cap/1e4)}萬|+{s}'; navs = {}
        for nm, rbk in (('CUR', None), ('FZ', FZ)):
            nav, lg, st = sim(Fx, ROWS, FCAP[cap], rbk)
            PATH[nm][key] = dict(valid=cg(nav, 'valid'), y2022_2023=cg(nav, 'y2022_2023'), since2024=cg(nav, 'since2024'), valid_dd=stlib.maxdd(nav, TB, DE), seg=L.seg_dd(nav),
                                 avg_stockfrac=sfm(st), n_pos=int(len(lg)), rb_buy=int(st['rb_buy']))
            navs[nm] = nav; del lg, st
            PATH[nm + '_blk'][key] = dict(valid=cg(sim(Fx, ROWS, FCAP[cap], rbk, block=BLK)[0], 'valid'))
        lam = PATH['FZ'][key]['avg_stockfrac'] / PATH['CUR'][key]['avg_stockfrac']
        rc = navs['CUR'].loc[TB:DE].pct_change().iloc[1:]; rbm = bench.reindex(navs['CUR'].index).loc[TB:DE].pct_change().iloc[1:]
        scn = pd.concat([pd.Series([1.0], index=[navs['CUR'].loc[TB:DE].index[0]]), (1 + lam * rc + (1 - lam) * rbm).cumprod()])
        PATH['SCALED'][key] = dict(valid=stlib.cagr(scn, TB, DE), lam=float(lam)); del navs, scn
    del S, Fx, ROWS; gc.collect()
    print('shift', s, f'{time.time()-t_all:.0f}s', rss(), 'MB', flush=True)
res['paths'] = PATH
keys = list(PATH['CUR'])


def pair(a, b, f, sub=None):
    ga = np.array([PATH[a][k][f][sub] if sub else PATH[a][k][f] for k in keys]); gb = np.array([PATH[b][k][f][sub] if sub else PATH[b][k][f] for k in keys])
    d = (ga - gb) * 100
    return dict(median=float(np.median(d)), npos=int((d > 0).sum()), min=float(d.min()), max=float(d.max()), a_median=float(np.median(ga)), b_median=float(np.median(gb)), per_path=dict(zip(keys, d.round(4).tolist())))


pv = dict(valid=pair('FZ', 'CUR', 'valid'), y2022_2023=pair('FZ', 'CUR', 'y2022_2023'), since2024=pair('FZ', 'CUR', 'since2024'), valid_dd=pair('FZ', 'CUR', 'valid_dd'),
          seg={g: pair('FZ', 'CUR', 'seg', g) for g in L.SEGMENTS}, blk=pair('FZ_blk', 'CUR_blk', 'valid'), R2_vs_scaled=pair('FZ', 'SCALED', 'valid'),
          stockfrac=pair('FZ', 'CUR', 'avg_stockfrac'))
res['pairs'] = pv
res['rb_buy_per_path'] = {k: PATH['FZ'][k]['rb_buy'] for k in keys}
V = dict(V1=pv['valid']['npos'] >= 15, V2=pv['valid']['median'] >= 2.0, V3=pv['blk']['median'] > 0, V4=all(pv['seg'][g]['median'] >= -2.0 for g in L.SEGMENTS))
V = {k: bool(v) for k, v in V.items()}
R2raw = bool(pv['R2_vs_scaled']['median'] > 0 and pv['R2_vs_scaled']['npos'] >= 15)
res['verdict'] = dict(**V, n_pass_V=int(sum(V.values())), R2_triggered=R2_TRIGGER, R2_beats_scaled=R2raw, R2=bool(R2raw or not R2_TRIGGER),
                      selected='不改（T0 未過，FROZEN.json）', overall='不通過')
res['meta'] = dict(runtime_s=round(time.time() - t_all, 1), maxrss_mb=rss())
(HERE / 'valid.json').write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str))
print(json.dumps(res['verdict'], ensure_ascii=False))
print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != 'per_path'} if 'median' in v else v for k, v in pv.items()}, ensure_ascii=False, default=str))
print(json.dumps({k: v for k, v in main['FZ'].items() if k != 'rebuys'}, ensure_ascii=False, default=str)); print(main['CUR']); print(res['meta'])
