#!/usr/bin/env python3
"""S75 正式執行（只跑一次；results.json 已存在就拒絕）：連續面板上兩個版本（V_MASK、V_BRIDGE）× 20 條路徑，2008-01-02～2026-10-22 連續帳戶。
先核對 SPEC／PREREG／程式的 sha256。python3 -I -B run.py → results.json、nav_2008_2026.csv.gz、yearly.csv、trades.csv.gz、signals.csv、paths.csv"""
import sys, json, time, resource, hashlib, gc
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
HERE = Path(__file__).resolve().parent
assert not (HERE / 'results.json').exists(), 'results.json 已存在：本題只跑一次'
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
for f in ('SPEC', 'PREREG'):
    assert sha(HERE / f'{f}.json') == (HERE / f'{f}.sha256').read_text().split()[0], f'{f}.json 與登記的 sha256 不符'
for line in (HERE / 'code.sha256').read_text().strip().split('\n'):
    h, f = line.split(); assert sha(HERE / f) == h, f'{f} 與登記的 sha256 不符'
import numpy as np, pandas as pd
import s75lib as K
from s75lib import R

t0 = time.time()
rss = lambda: round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
P, U = K.load_cont()
D0, DE = P.IDX[P.t0], P.IDX[-1]
assert (str(D0.date()), str(DE.date())) == ('2008-01-02', '2026-10-22')
D19 = K.last_before(P.IDX, '2019-01-01'); D157 = pd.Timestamp('2015-07-01'); D24 = K.last_before(P.IDX, '2024-01-01')
bench = pd.Series(P.E50, index=P.IDX).loc[D0:]; bench = bench / bench.iloc[0] * 1e6
off = pd.read_csv(K.TMP / 'is_nav.csv', index_col=0, parse_dates=True)['nav']
s71nav = pd.read_csv(K.AG / 's71_ifrs_bridge_oos' / 'oos_nav.csv.gz', index_col=0, parse_dates=True)
s71m = pd.read_csv(K.AG / 's71_ifrs_bridge_oos' / 'oos_monthly.csv', index_col=0); s71m.index = pd.PeriodIndex(s71m.index, freq='M')
out = dict(task='S75 現行策略 2008-01～2026-10 連續帳戶長期成績單', sha=dict(SPEC=sha(HERE / 'SPEC.json'), PREREG=sha(HERE / 'PREREG.json')),
           start=str(D0.date()), end=str(DE.date()), capital=1e6, panel=dict(T=P.T, N=P.N), versions={})
NAV = {}; LG = {}; trades = []; sigs = []; prow = []
for v, rule in K.VERSIONS.items():
    F, RT = K.surge(P, rule)
    paths = {}
    for s in K.SHIFTS:
        sg = K.signal(P, U, RT, s)
        for cap in K.CAPS:
            nav, lg, st = K.sim(P, sg, U, cap)
            key = f'{int(cap / 1e4)}萬|+{s}'
            paths[key] = dict(cagr_full=K.cagr(nav, D0, DE), maxdd=R.maxdd(nav), cagr_since2019=K.cagr(nav, D19, DE), cagr_since2015_07=K.cagr(nav, D157, DE),
                              cagr_2008_2015H1=K.cagr(nav, D0, K.last_before(P.IDX, '2015-07-01')), final_multiple=float(nav.iloc[-1] / cap),
                              n_closed=int((lg['why'] != 'open_end').sum()))
            prow.append(dict(version=v, capital=int(cap), shift=s, **paths[key]))
            if s == 0 and cap == 1e6:
                std = (sg, nav, lg, st)
            del nav, lg, st
        del sg; gc.collect()
    sg, nav, lg, st = std
    NAV[v] = nav; LG[v] = lg
    b = bench.reindex(nav.index)
    rs, rb = R.monthly_returns(nav, b)
    blk = R.stats_block(rs, rb)
    yt = K.yearly_table(nav, b)
    dd, ddb = K.dd_info(nav), K.dd_info(b)
    full = dict(final_nav=float(nav.iloc[-1]), final_nav_0050=float(b.iloc[-1]), multiple=float(nav.iloc[-1] / 1e6), multiple_0050=float(b.iloc[-1] / 1e6),
                cagr=K.cagr(nav, D0, DE), cagr_0050=K.cagr(b, D0, DE), cagr_diff_pp=(K.cagr(nav, D0, DE) - K.cagr(b, D0, DE)) * 100,
                maxdd=dd, maxdd_0050=ddb, vol=K.ann_vol(nav), vol_0050=K.ann_vol(b), months=blk['months'], ex_ann_pp=blk['ex_ann_pp'], te_ann_pp=blk['te_ann_pp'],
                IR=blk['IR'], t_nw6=blk['t_nw6'], t_iid=blk['t_iid'], beta=blk['beta'],
                beat_years=int(sum(y['strat'] > y['e0050'] for y in yt)), n_years=len(yt),
                beat_years_ex2026=int(sum(y['strat'] > y['e0050'] for y in yt if y['year'] != 2026)), n_years_ex2026=len(yt) - 1,
                sharpe_rf0=float(nav.pct_change().dropna().mean() / nav.pct_change().dropna().std(ddof=1) * np.sqrt(252)),
                sharpe_rf0_0050=float(b.pct_change().dropna().mean() / b.pct_change().dropna().std(ddof=1) * np.sqrt(252)))
    periods = {k: K.window_block(nav, b, a, e) for k, (a, e) in K.PERIODS.items()}
    extra_windows = {k: K.window_block(nav, b, a, e) for k, (a, e) in {'2008-01～2015-06（規則樣本外）': ('2008-01', '2015-06'), '2015-07～2026-10（規則樣本內）': ('2015-07', '2026-10'),
                                                                     '2008-01～2013-01': ('2008-01', '2013-01'), '2013-02～2015-06': ('2013-02', '2015-06'),
                                                                     '2016-02～2026-10': ('2016-02', '2026-10'), '2019-01～2026-10': ('2019-01', '2026-10')}.items()}
    v_ = np.array([p['cagr_full'] for p in paths.values()]); d_ = np.array([p['maxdd'] for p in paths.values()]); c19 = np.array([p['cagr_since2019'] for p in paths.values()])
    psum = dict(n=len(paths), cagr_full_median=float(np.median(v_)), cagr_full_min=float(v_.min()), cagr_full_max=float(v_.max()),
                maxdd_median=float(np.median(d_)), maxdd_worst=float(d_.min()), maxdd_best=float(d_.max()),
                cagr19_median=float(np.median(c19)), cagr19_min=float(c19.min()), cagr19_max=float(c19.max()),
                by_shift={f'+{s}': float(np.median([p['cagr_full'] for k, p in paths.items() if k.endswith(f'+{s}')])) for s in K.SHIFTS},
                by_capital={f'{int(c / 1e4)}萬': float(np.median([p['cagr_full'] for k, p in paths.items() if k.startswith(f'{int(c / 1e4)}萬|')])) for c in K.CAPS})
    tr = dict(all=K.trade_block(P, lg, st), p2008_2015=K.trade_block(P, lg, st, None, '2015-12-31'), p2016_2026=K.trade_block(P, lg, st, '2016-01-01', None),
              n_buy=int(st['n_buy']), engine_avg_npos=float(st['avg_npos']), engine_avg_idle_frac=float(st['avg_idle_frac']),
              engine=dict({k: int(st[k]) for k in ('n_sig', 'blk_limup', 'blk_noopen', 'blk_lot', 'blk_cap', 'blk_nocash', 'blk_min', 'noslot', 'n_delist', 'odd_buys', 'sell_defer_limdn')}),
              why=lg['why'].value_counts().to_dict(), limit7=K.limit7_diag(P, lg))
    x = lg[lg['why'] != 'open_end'].sort_values('pnl', ascending=False)
    tr['top5'] = [dict(code=str(c), ent=str(e.date()), exit=str(q.date()), ret=float(r), pnl=float(p)) for c, e, q, r, p in zip(x['code'].head(5), x['ent_date'].head(5), x['exit_date'].head(5), x['ret'].head(5), x['pnl'].head(5))]
    tr['top5_pnl_share_of_gain'] = float(x['pnl'].head(5).sum() / (nav.iloc[-1] - 1e6))
    tr['open_end'] = sorted(lg.loc[lg['why'] == 'open_end', 'code'].astype(str).tolist())
    # ---- A5：2015-07-01 重新基準化 vs 正式引擎
    o = off.reindex(nav.loc[D157:].index); nb = nav.loc[D157:] / nav.loc[D157] * 1e6
    yo = K.yearly_table(o, o); yn = K.yearly_table(nb, nb)
    r1, r2 = nb.loc['2016-03-01':].pct_change().dropna(), o.loc['2016-03-01':].pct_change().dropna()
    a5 = dict(since2019=K.cagr(nav, D19, DE), official_since2019=K.ACC['since2019'], gap_since2019_pp=(K.cagr(nav, D19, DE) - K.ACC['since2019']) * 100,
              since2015_07=K.cagr(nav, D157, DE), official_since2015_07=K.ACC['full'], gap_since2015_07_pp=(K.cagr(nav, D157, DE) - K.ACC['full']) * 100,
              since2016_02_16=K.cagr(nav, pd.Timestamp('2016-02-16'), DE), official_since2016_02_16=K.cagr(off, pd.Timestamp('2016-02-16'), DE),
              since2024=K.cagr(nav, D24, DE), official_since2024=K.cagr(off, D24, DE),
              maxdd_since2015_07=R.maxdd(nav.loc[D157:]), official_maxdd=K.ACC['maxdd'],
              nav_on_2015_07_01=float(nav.loc[D157]), stock_frac_on_2015_07_01=float(st['stockfrac'].loc[D157]), npos_on_2015_07_01=int(K.npos_series(P, lg).loc[D157]),
              nav_on_2016_02_16=float(nav.loc['2016-02-16']), npos_on_2016_02_16=int(K.npos_series(P, lg).loc['2016-02-16']),
              rebased_final=float(nb.iloc[-1]), official_final=float(o.iloc[-1]), daily_ret_corr_from_2016_03=float(np.corrcoef(r1, r2)[0, 1]),
              yearly=[dict(year=a['year'], cont=a['strat'], official=b_['strat'], diff_pp=(a['strat'] - b_['strat']) * 100) for a, b_ in zip(yn, yo)],
              n_closed_entered_from_2016_02=int(((lg['why'] != 'open_end') & (lg['ent_date'] >= pd.Timestamp('2016-02-16'))).sum()), official_n_closed=K.ACC['n_closed'])
    # ---- A4：與 S71 存檔比對
    col = 'B2_nav' if rule == 'B2' else None
    a4 = {}
    if col:
        z = s71nav[col]; m = nav.reindex(z.index)
        rel = (m / z - 1).abs()
        a4 = dict(vs='S71 oos_nav B2_nav', max_rel_to_2014_12_31=float(rel.loc[:'2014-12-31'].max()), max_rel_to_2015_06_30=float(rel.max()),
                  nav_2015_06_30=float(m.iloc[-1]), s71_nav_2015_06_30=float(z.iloc[-1]))
    mm = s71m['MASK_strat' if rule == 'MASK' else 'B2_strat']
    dm = (rs.reindex(mm.index) - mm).abs()
    a4.update(monthly_vs_S71_max_abs_to_2014_12=float(dm.loc[:pd.Period('2014-12', 'M')].max()), monthly_vs_S71_max_abs_to_2015_06=float(dm.max()))
    out['versions'][v] = dict(rule=rule, full=full, yearly=yt, periods=periods, extra_windows=extra_windows, bears=K.bear_block(nav, b), paths_summary=psum, paths=paths, trades=tr,
                              signal=dict(n_signal_months=len(sg['info']), first=sg['info'][0]['sig_date'], last=sg['info'][-1]['sig_date'],
                                          mean_targets=float(np.mean([i['n_tgt'] for i in sg['info']])),
                                          mean_targets_2008_2015=float(np.mean([i['n_tgt'] for i in sg['info'] if i['sig_date'] < '2016-01-01'])),
                                          mean_targets_2016_2026=float(np.mean([i['n_tgt'] for i in sg['info'] if i['sig_date'] >= '2016-01-01'])),
                                          months_zero_target=int(sum(1 for i in sg['info'] if i['n_tgt'] == 0))),
                              A4_vs_S71=a4, A5_rebase_2015_07=a5)
    trades.append(lg.drop(columns=['j']).assign(version=v))
    sigs.append(pd.DataFrame(sg['info']).drop(columns=['targets']).assign(targets=[' '.join(i['targets']) for i in sg['info']], version=v))
    print(v, {k: (round(x, 4) if isinstance(x, float) else x) for k, x in full.items() if not isinstance(x, dict)}, psum['cagr_full_median'], f'{time.time() - t0:.0f}s', rss(), 'MB', flush=True)
    del std, F, RT; gc.collect()
# ---- 兩版互比
a, b_ = NAV['V_MASK'], NAV['V_BRIDGE']
ra, rb_ = a.loc['2016-03-01':].pct_change().dropna(), b_.loc['2016-03-01':].pct_change().dropna()
out['mask_vs_bridge'] = dict(identical_to=str(a.index[(np.abs(a / b_ - 1) > 1e-12).to_numpy().argmax() - 1].date()), cagr_diff_full_pp=(K.cagr(b_) - K.cagr(a)) * 100,
                             cagr_diff_since2016_02_16_pp=(K.cagr(b_, pd.Timestamp('2016-02-16'), DE) - K.cagr(a, pd.Timestamp('2016-02-16'), DE)) * 100,
                             daily_ret_corr_from_2016_03=float(np.corrcoef(ra, rb_)[0, 1]))
out['official_alone'] = json.load(open(HERE / 'accept_is.json'))
out['accept'] = dict(A1_bitwise=out['official_alone']['bitwise_vs_ACC'], A2_A3_pass=json.load(open(HERE / 'accept_sig.json'))['pass'],
                     seam=json.load(open(HERE / 'panel_check.json'))['overlap_2015H1'])
out['runtime_s'] = round(time.time() - t0, 1); out['maxrss_mb'] = rss()
(HERE / 'results.json').write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str))
df = pd.DataFrame({'V_MASK': NAV['V_MASK'], 'V_BRIDGE': NAV['V_BRIDGE'], '0050': bench.reindex(NAV['V_MASK'].index)}).round(2); df.index.name = 'date'
df.to_csv(HERE / 'nav_2008_2026.csv.gz')
ym, yb = out['versions']['V_MASK']['yearly'], out['versions']['V_BRIDGE']['yearly']
pd.DataFrame([dict(year=m['year'], V_MASK=m['strat'], V_BRIDGE=q['strat'], e0050=m['e0050'], V_MASK_diff_pp=m['diff_pp'], V_BRIDGE_diff_pp=q['diff_pp']) for m, q in zip(ym, yb)]).to_csv(HERE / 'yearly.csv', index=False)
pd.concat(trades).to_csv(HERE / 'trades.csv.gz', index=False)
pd.concat(sigs).to_csv(HERE / 'signals.csv', index=False)
pd.DataFrame(prow).to_csv(HERE / 'paths.csv', index=False)
print('done', out['runtime_s'], 's', out['maxrss_mb'], 'MB')
