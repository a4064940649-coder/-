#!/usr/bin/env python3
"""R75 第 3 步：連續帳戶（100 萬、2008-01-02 起、W10_70、嚴成本、正式開關）兩版：V_BRIDGE（s71lib.rev_rule('B2')）、V_MASK（rev_rule('MASK')）。
python3 -I -B step2_run.py [cache 目錄名，預設 cache_full] → scratchpad/r75/run_<cache>.json、nav_<cache>.csv、trades_<cache>_<版>.csv、signals_<cache>_<版>.csv、daily_<cache>_<版>.csv"""
import sys, json, time, resource, gc
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import r75lib as Q
from r75lib import R, L, stlib, s17lib, s10lib, cslib
import numpy as np, pandas as pd

t0 = time.time()
CN = sys.argv[1] if len(sys.argv) > 1 else 'cache_full'
Q.setup(CN, '2008-01-02', '2008-01')
P, U, fix = Q.load_all()
for a_ in ('H', 'L', 'VOL', 'HF', 'VOLF'):
    if hasattr(P, a_):
        delattr(P, a_)
gc.collect()
rss = lambda: round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
print('載入', P.T, P.N, str(P.IDX[P.t0].date()), str(P.IDX[-1].date()), P.exright_info, 'bad_ret', P.bad_ret, 'e50_open_missing', P.e50_open_missing, rss(), 'MB', flush=True)
D0, DE = P.IDX[P.t0], P.IDX[-1]
bench = pd.Series(P.E50, index=P.IDX).loc[D0:DE]
b1 = bench / bench.iloc[0] * 1e6
lastb = lambda d: P.IDX[P.IDX < pd.Timestamp(d)][-1]
PERIODS = {'2008~2012': ('2008-01', '2012-12'), '2013~2015': ('2013-01', '2015-12'), '2016~2018': ('2016-01', '2018-12'), '2019~2023': ('2019-01', '2023-12'), '2024~2026': ('2024-01', '2026-10'),
           'OOS_2008-01~2015-06': ('2008-01', '2015-06'), 'IS_2015-07~': ('2015-07', '2026-10'), 'ALL': ('2008-01', '2026-10')}


def summarize(nav, lg, st, sg):
    o = dict(final=float(nav.iloc[-1]), cagr=stlib.cagr(nav, D0, DE), **Q.dd_info(nav), vol=float(nav.pct_change().std(ddof=1) * np.sqrt(252)), yearly=Q.yearly(nav))
    rs, rb = R.monthly_returns(nav, bench)
    o['periods'] = {}
    for k, (a, b) in PERIODS.items():
        m = (rs.index >= pd.Period(a, 'M')) & (rs.index <= pd.Period(b, 'M'))
        sb = R.stats_block(rs[m], rb[m])
        base = D0 if a == '2008-01' else lastb(a + '-01'); end = P.IDX[P.IDX <= (pd.Period(b, 'M').to_timestamp(how='end'))][-1]
        sb['cagr_cal'] = stlib.cagr(nav, base, end); sb['cagr_cal_0050'] = stlib.cagr(bench, base, end)
        sb['maxdd'] = R.maxdd(nav.loc[base:end]); sb['maxdd_0050'] = R.maxdd(bench.loc[base:end])
        o['periods'][k] = sb
    x = lg[lg['why'] != 'open_end']
    o['trades'] = dict(n_closed=int(len(x)), n_open_end=int((lg['why'] == 'open_end').sum()), win_rate=float((x['ret'] > 0).mean()), avg_ret=float(x['ret'].mean()), avg_hold=float(x['hold_days'].mean()),
                       n_buy=int(st['n_buy']), n_delist=int(st['n_delist']), avg_npos=float(st['avg_npos']), avg_idle_frac=float(st['avg_idle_frac']), why=lg['why'].value_counts().to_dict(),
                       blk={k: int(st[k]) for k in ('blk_limup', 'blk_noopen', 'blk_lot', 'blk_cap', 'blk_nocash', 'blk_min', 'noslot', 'sell_defer_limdn', 'sell_defer_noopen', 'sell_partial_cap')},
                       by_entry={'2008~2015': dict(n=int((x['ent_date'] < '2016-01-01').sum()), win=float((x.loc[x['ent_date'] < '2016-01-01', 'ret'] > 0).mean())),
                                 '2016~2026': dict(n=int((x['ent_date'] >= '2016-01-01').sum()), win=float((x.loc[x['ent_date'] >= '2016-01-01', 'ret'] > 0).mean()))})
    info = sg['info']
    o['signal'] = dict(n_months=len(info), first=info[0]['sig_date'], last=info[-1]['sig_date'], n_targets=int(sum(i['n_tgt'] for i in info)),
                       n_months_2013_02_to_2016_01=int(sum('2013-02-01' <= i['sig_date'] <= '2016-01-31' for i in info)))
    # 2015-07-01 重新基準化
    d15 = pd.Timestamp('2015-07-01'); d19 = lastb('2019-01-01')
    o['rebased'] = dict(since_2015_07=stlib.cagr(nav, d15, DE), since2019=stlib.cagr(nav, d19, DE))
    return o


out = dict(cache=CN, P=dict(T=P.T, N=P.N, t0=str(D0.date()), last=str(DE.date()), exright_info=P.exright_info, bad_ret=P.bad_ret, e50_open_missing=P.e50_open_missing, e50_missing=P.e50_missing, e50_fix=fix),
           bench=dict(final=float(b1.iloc[-1]), cagr=stlib.cagr(bench, D0, DE), **Q.dd_info(bench), yearly=Q.yearly(bench)))
navs = {'0050': b1}
for ver, rule in (('V_BRIDGE', 'B2'), ('V_MASK', 'MASK')):
    F, sg, nav, lg, st = Q.run_rule(P, U, rule, '2008-01')
    out[ver] = summarize(nav, lg, st, sg)
    navs[ver] = nav
    lg.drop(columns=['j']).to_csv(Q.TMP / f'trades_{CN}_{ver}.csv', index=False)
    pd.DataFrame(sg['info']).assign(targets=[' '.join(i['targets']) for i in sg['info']]).to_csv(Q.TMP / f'signals_{CN}_{ver}.csv', index=False)
    pd.DataFrame({'nav': nav, 'stockfrac': st['stockfrac'], 'B': st['B']}).to_csv(Q.TMP / f'daily_{CN}_{ver}.csv')
    F['S']['STD'].to_pickle(Q.TMP / f'surge_{CN}_{ver}.pkl')
    print(ver, {k: v for k, v in out[ver].items() if k in ('final', 'cagr', 'maxdd', 'peak', 'trough', 'recover')}, out[ver]['trades']['n_closed'], out[ver]['trades']['win_rate'], f'{time.time()-t0:.0f}s', rss(), 'MB', flush=True)
    del F, sg, lg, st; gc.collect()
a, b = navs['V_BRIDGE'], navs['V_MASK']
ra, rb_ = a.pct_change(), b.pct_change()
m16 = ra.index >= '2016-02-01'
out['bridge_vs_mask'] = dict(corr_daily_2016_02_on=float(np.corrcoef(ra[m16].dropna(), rb_[m16].dropna())[0, 1]),
                             cagr_2016_on_bridge=stlib.cagr(a, lastb('2016-01-01'), DE), cagr_2016_on_mask=stlib.cagr(b, lastb('2016-01-01'), DE),
                             nav_2015_12_31=dict(bridge=float(a.loc[lastb('2016-01-01')]), mask=float(b.loc[lastb('2016-01-01')])))
out['runtime_s'] = round(time.time() - t0, 1); out['maxrss_mb'] = rss()
pd.DataFrame(navs).to_csv(Q.TMP / f'nav_{CN}.csv')
(Q.TMP / f'run_{CN}.json').write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str))
print('done', out['runtime_s'], 's', out['maxrss_mb'], 'MB')
