#!/usr/bin/env python3
"""S97 執行：python3 -I -B run_s97.py official|cont
official：正式引擎 E0|W10_70（2015-07 起；同 official_nav/run_official.py 的設定，但不寫任何正式檔）→ out_official.json
cont    ：連續引擎 V_BRIDGE（2008 起；同 s75 v2_2026-11-21/run_v2.py v2 的設定，直接讀既有的 e50_spliced.csv）→ out_cont.json
先驗收基準（不過就 assert 停下），再跑 PREREG 的情境格，一格一格跑、跑完釋放。"""
import sys, json, time, resource, gc, math
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
AG = HERE.parent
MODE = sys.argv[1]; assert MODE in ('official', 'cont')
t00 = time.time(); rss = lambda: round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
sys.path.insert(0, str(HERE))
import s97lib

if MODE == 'official':
    TMP = Path('/tmp/claude-0/-home-user--/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/offfix/cache')   # 既有面板快取（只讀）
    for f in ('panel_cs.npz', 'e50_ohlc.csv', 'innov.npz'):
        assert (TMP / f).exists(), f'快取缺 {f}'
    sys.path.insert(0, str(AG / 's19_xq_threshold'))
    import s19lib
    from s19lib import s10lib, s9lib, s8lib, s7lib, s6lib, s5lib, s4lib, s3lib, s2lib, stlib, cslib, COSTS
    for mod, attr in ((s19lib, 'S19'), (s10lib, 'S10'), (s9lib, 'S9'), (s8lib, 'S8'), (s7lib, 'S7'), (s6lib, 'S6'), (s5lib, 'S5'),
                      (s4lib, 'S4'), (s3lib, 'S3'), (s2lib, 'S2'), (stlib, 'S1'), (cslib, 'TMP')):
        setattr(mod, attr, TMP)
    assert cslib.EXRIGHT_FIX is False and cslib.DEADLINE_FIX is False and cslib.INS15_MASK is False and cslib.REDUCE_FIX is False and cslib.E50_SOURCE is None
    cslib.EXRIGHT_FIX = True; cslib.DEADLINE_FIX = True; cslib.INS15_MASK = True; cslib.REDUCE_FIX = True; cslib.E50_SOURCE = cslib.E50_OFFICIAL_CSV
    P = s19lib.load()
    assert P.exright_fix and P.reduce_fix and P.e50_source == str(cslib.E50_OFFICIAL_CSV)
    s19lib.e50_fix_suspension(P)
    INNOV, _ = s4lib.innov_panel(P)
    U = stlib.universe(P, 'twse') & ~INNOV
    S, _ = s19lib.rev_std(P)
    RT = dict(S=S, av=s19lib.a10_rows(P, list(S.index)))
    CAPS = (6e5, 8e5, 1e6, 1.5e6, 2e6); SHIFTS = (0, 1, 2, 3)
    HIST7 = False                                            # 2015-07 起全是 10%（面板 2015-01～05 不在回測期）
else:
    S75 = AG / 's75_continuous_2008_2026'
    sys.path.insert(0, str(S75))
    import s75lib as K
    from s75lib import R as R55, L, s20lib, s17lib, s10lib, s4lib, stlib, cslib
    from s10lib import COSTS
    assert cslib.REDUCE_FIX is False and cslib.E50_SOURCE is None
    cslib.E50_SOURCE = S75 / 'v2_2026-11-21' / 'e50_spliced.csv'; cslib.REDUCE_FIX = True
    # 以下同 s75lib.load_cont()，只差不刪 H、L（口徑 B 要用）
    R55.point_cache(K.TMP / 'cache_cont')
    stlib.START = pd.Timestamp(K.START)
    assert cslib.EXRIGHT_FIX and cslib.DEADLINE_FIX and cslib.INS15_MASK
    P = s20lib.load()
    INNOV, _ = s4lib.innov_panel(P)
    U = stlib.universe(P, 'twse') & ~INNOV
    s17lib.e50_fix_suspension(P)
    F, RT = K.surge(P, 'B2'); del F
    CAPS = (1e6,); SHIFTS = (0,)
    HIST7 = True
PH, PL = P.H, P.L
for a in ('VOL', 'HF', 'VOLF', 'adjH', 'HI240', 'IDXC', 'NEARHI', 'F_HI', 'V60', 'liq', 'nbar'):
    if hasattr(P, a):
        delattr(P, a)
del INNOV; gc.collect()
print('載入', MODE, P.T, P.N, f'{time.time()-t00:.0f}s', rss(), 'MB', flush=True)

sim97, SRC = s97lib.build(s10lib)
LU, LD = s97lib.limit_prices(P, cslib.tick, hist7=HIST7)
LU10, LD10 = s97lib.limit_prices(P, cslib.tick, hist7=False)
assert np.array_equal(LU10, P.LIMUP, equal_nan=True) and np.array_equal(LD10, P.LIMDN, equal_nan=True), '10% 漲跌停價必須與 cslib 逐位相同'
if not HIST7:
    del LU10, LD10
D0, DE = P.IDX[P.t0], P.IDX[-1]
D19 = P.IDX[P.IDX < pd.Timestamp('2019-01-01')][-1]
D157 = P.IDX[P.IDX < pd.Timestamp('2015-07-01')][-1] if D0 < pd.Timestamp('2015-07-01') else D0
bench = pd.Series(P.E50, index=P.IDX)
KW = dict(sat_w=0.7, sizing='W10', lock=False, cost=COSTS['strict'], cap_pct=0.05)
SCEN = {'NONE': ('NONE', 'i'), 'A_i': ('A', 'i'), 'A_ii': ('A', 'ii'), 'A_iii': ('A', 'iii'), 'B_i': ('B', 'i'), 'B_ii': ('B', 'ii'), 'B_iii': ('B', 'iii')}


def run(Sg, cap, scen):
    if scen == 'BASE':
        return s10lib.simulate_s10(P, Sg, U, capital=cap, **KW)
    if scen == 'PLUG_OFF':
        return sim97(P, Sg, U, capital=cap, lim=None, **KW)
    c, b = SCEN[scen]
    return sim97(P, Sg, U, capital=cap, lim=dict(crit=c, buy=b, LU=LU, LD=LD, LOW=PL, HIGH=PH), **KW)


def metrics(nav, lg, st):
    x = lg[lg['why'] != 'open_end']
    sd = nav.loc[D0:DE]; dd = sd / sd.cummax() - 1; tr = dd.idxmin(); pk = sd.loc[:tr].idxmax()
    m = dict(final_nav=float(nav.iloc[-1]), cagr_full=stlib.cagr(nav, D0, DE), cagr_since2019=stlib.cagr(nav, D19, DE),
             maxdd=float(dd.min()), dd_peak=str(pk.date()), dd_trough=str(tr.date()), n_closed=int(len(x)), n_all=int(len(lg)),
             win_rate=float((x['ret'] > 0).mean()), e50_cagr_full=stlib.cagr(bench, D0, DE), e50_cagr_since2019=stlib.cagr(bench, D19, DE),
             n_buy=int(st['n_buy']), n_sig=int(st['n_sig']), blk_limup=int(st['blk_limup']), blk_noopen=int(st['blk_noopen']),
             sell_defer_limdn=int(st['sell_defer_limdn']), sell_defer_noopen=int(st['sell_defer_noopen']))
    m['ex_full_pp'] = (m['cagr_full'] - m['e50_cagr_full']) * 100; m['ex_since2019_pp'] = (m['cagr_since2019'] - m['e50_cagr_since2019']) * 100
    if MODE == 'cont':
        m['cagr_2008_2015H1'] = stlib.cagr(nav, D0, D157); m['cagr_since2015_07'] = stlib.cagr(nav, D157, DE)
        m['e50_2008_2015H1'] = stlib.cagr(bench, D0, D157); m['e50_since2015_07'] = stlib.cagr(bench, D157, DE)
    ev = st.get('lim_ev')
    if ev is not None:
        kinds = {}
        for e in ev:
            kinds[e[3]] = kinds.get(e[3], 0) + 1
        m['events'] = kinds
        m['event_list'] = [dict(date=str(P.IDX[t].date()), code=str(P.COLS[j]), sig_date=str(P.IDX[k].date()), kind=kd) for t, j, k, kd in ev]
    return m


out = dict(mode=MODE, start=str(D0.date()), end=str(DE.date()), hist7=HIST7, scen={}, paths={})
Sg0 = s10lib.make_signal_s10(P, U, RT, shift=0)

# ---------------------------------------------------------------- 驗收：基準
nav0, lg0, st0 = run(Sg0, 1e6, 'BASE')
mb = metrics(nav0, lg0, st0); out['scen']['BASE'] = mb
if MODE == 'official':
    ref = pd.read_csv('/home/user/-/xq_scripts/site/navs_official.csv.gz', index_col=0, parse_dates=True)['E0|W10_70']
    tgt = dict(cagr_since2019=0.4886, cagr_full=0.3446, n_closed=381)
else:
    ref = pd.read_csv(S75 / 'v2_2026-11-21' / 'nav_2008_2026_v2.csv.gz', index_col=0, parse_dates=True)['V_BRIDGE']
    tgt = dict(cagr_full=0.2356, n_all=632, final_nav=53392744)
same = nav0.index.equals(ref.index)
acc = dict(target=tgt, got={k: mb[k] for k in tgt}, nav_same_index=bool(same), nav_max_abs_diff=float((nav0.round(2) - ref).abs().max()) if same else None)
acc['pass'] = bool(same and acc['nav_max_abs_diff'] == 0 and all((abs(mb[k] - v) <= 5e-5) if isinstance(v, float) else (round(mb[k]) == v) for k, v in tgt.items()))
out['acceptance_base'] = acc
print('基準驗收', acc, flush=True)
assert acc['pass'], '基準重現失敗，停止'
nav1, lg1, st1 = run(Sg0, 1e6, 'PLUG_OFF')
off_same = bool(np.array_equal(nav0.to_numpy(), nav1.to_numpy()) and len(lg0) == len(lg1) and np.array_equal(lg0['proceeds'].to_numpy(), lg1['proceeds'].to_numpy()))
out['acceptance_plug_off'] = dict(nav_bitwise_same=off_same)
print('外掛關閉＝基準', off_same, flush=True)
assert off_same
del nav1, lg1, st1

# ---------------------------------------------------------------- 情境（標準路徑）
NAVS = {'BASE': nav0}; LOGS = {}
for sc in SCEN:
    nav, lg, st = run(Sg0, 1e6, sc)
    m = metrics(nav, lg, st)
    m['nav_max_abs_diff_vs_BASE'] = float((nav - nav0).abs().max()); m['d_cagr_full_vs_BASE_pp'] = (m['cagr_full'] - mb['cagr_full']) * 100
    m['d_cagr_since2019_vs_BASE_pp'] = (m['cagr_since2019'] - mb['cagr_since2019']) * 100; m['d_maxdd_vs_BASE_pp'] = (m['maxdd'] - mb['maxdd']) * 100
    out['scen'][sc] = m; NAVS[sc] = nav
    if sc in ('NONE', 'A_i', 'B_i'):
        LOGS[sc] = (lg, st['lim_ev'])
    print(sc, {k: (round(v, 5) if isinstance(v, float) else v) for k, v in m.items() if k not in ('event_list',)}, rss(), 'MB', flush=True)
    del nav, lg, st; gc.collect()
if MODE == 'official':
    assert out['scen']['A_i']['nav_max_abs_diff_vs_BASE'] == 0, '正式引擎 A_i 必須等於基準'
out['check_B_iii_equals_B_i'] = float((NAVS['B_iii'] - NAVS['B_i']).abs().max())
df = pd.DataFrame(NAVS).round(2); df.index.name = 'date'
df.to_csv(HERE / f'navs_{MODE}.csv.gz')

# ---------------------------------------------------------------- 描述統計
O, C = P.O, P.C


def d1(Sg, lo=None, hi=None):
    r = dict(n_pairs=0, n_noopen=0, n_A=0, n_B=0, n_A_opened_intraday=0, n_A_close_at_limit=0, n_A_next_open_locked=0, rows=[])
    for x in Sg['info']:
        t = x['k'] + 1
        if t >= P.T:
            continue
        d = P.IDX[t]
        if (lo is not None and d < pd.Timestamp(lo)) or (hi is not None and d > pd.Timestamp(hi)):
            continue
        for j in np.flatnonzero(Sg['sig'][x['k']] & U[x['k']]):
            r['n_pairs'] += 1; o = O[t, j]
            if not (np.isfinite(o) and o > 0):
                r['n_noopen'] += 1; continue
            a = bool(s97lib.lock_buy(o, LU[t, j], PL[t, j], 'A'))
            if not a:
                continue
            b = bool(s97lib.lock_buy(o, LU[t, j], PL[t, j], 'B'))
            r['n_A'] += 1; r['n_B'] += int(b); r['n_A_opened_intraday'] += int(not b)
            cl = bool(np.isfinite(C[t, j]) and C[t, j] >= LU[t, j] - 1e-9); r['n_A_close_at_limit'] += int(cl)
            nx = None
            if t + 1 < P.T and np.isfinite(O[t + 1, j]):
                nx = bool(s97lib.lock_buy(O[t + 1, j], LU[t + 1, j], PL[t + 1, j], 'A')); r['n_A_next_open_locked'] += int(nx)
            r['rows'].append(dict(t=int(t), j=int(j), date=str(d.date()), code=str(P.COLS[j]), month=x['month'], B=b, close_at_limit=cl, next_open_locked=nx,
                                  close_vs_open=float(C[t, j] / o - 1) if np.isfinite(C[t, j]) else None,
                                  next_open_vs_open=float(O[t + 1, j] / o - 1) if (t + 1 < P.T and np.isfinite(O[t + 1, j])) else None))
    for k in ('A', 'B', 'noopen'):
        r['pct_' + k] = r['n_' + k] / r['n_pairs'] * 100 if r['n_pairs'] else None
    return r


desc = {}
if MODE == 'cont':
    desc['D1'] = {'2008-01～2026-10': d1(Sg0), '2008-01～2015-06': d1(Sg0, hi='2015-06-30'), '2015-07～2026-10': d1(Sg0, lo='2015-07-01')}
else:
    desc['D1'] = {'2015-07～2026-10': d1(Sg0)}


def split_counts(dates):
    dates = pd.to_datetime(pd.Series(dates, dtype='datetime64[ns]'))
    return {'all': int(len(dates)), 'before_2015_07': int((dates < pd.Timestamp('2015-07-01')).sum()), 'from_2015_07': int((dates >= pd.Timestamp('2015-07-01')).sum())}


# D2、D3：A_i（＝現行口徑；連續引擎用 7%）與 B_i 的事件
for sc in ('A_i', 'B_i'):
    lg, ev = LOGS[sc]
    bl = [(t, j) for t, j, k, kd in ev if kd == 'buy_locked']
    sl = [(t, j, k) for t, j, k, kd in ev if kd == 'sell_locked']
    x = lg[lg['why'] != 'open_end']
    desc[f'D2_{sc}'] = dict(buy_attempts=out['scen'][sc]['n_sig'], buys_done=out['scen'][sc]['n_buy'], blocked_limup=len(bl),
                            pct_blocked=len(bl) / out['scen'][sc]['n_sig'] * 100, blocked_by_period=split_counts([P.IDX[t] for t, j in bl]),
                            blocked=[dict(date=str(P.IDX[t].date()), code=str(P.COLS[j])) for t, j in bl])
    tr = sorted(set((j, k) for t, j, k in sl))
    desc[f'D3_{sc}'] = dict(sell_locked_stock_days=len(sl), sell_locked_trades=len(tr), n_closed=int(len(x)), pct_trades=len(tr) / len(x) * 100,
                            stock_days_by_period=split_counts([P.IDX[t] for t, j, k in sl]),
                            days=[dict(date=str(P.IDX[t].date()), code=str(P.COLS[j])) for t, j, k in sl])
# D4：NONE 引擎的交易，標出進場日開盤鎖漲停、出場日開盤鎖跌停
lg, _ = LOGS['NONE']; lg = lg.copy()
e, xj, xt = lg['ent_t'].to_numpy(), lg['j'].to_numpy(), lg['exit_t'].to_numpy()
lg['ent_lock_A'] = s97lib.lock_buy(O[e, xj], LU[e, xj], PL[e, xj], 'A'); lg['ent_lock_B'] = s97lib.lock_buy(O[e, xj], LU[e, xj], PL[e, xj], 'B')
sold = lg['why'].isin(['rule', 'time', 'trail', 'requal', 'ma_daily']).to_numpy()
lg['exit_lock_A'] = sold & s97lib.lock_sell(O[xt, xj], LD[xt, xj], PH[xt, xj], 'A'); lg['exit_lock_B'] = sold & s97lib.lock_sell(O[xt, xj], LD[xt, xj], PH[xt, xj], 'B')
x = lg[lg['why'] != 'open_end']
ntop = int(math.ceil(0.05 * len(x)))
top_ret = x.sort_values('ret', ascending=False).head(ntop); top_pnl = x.sort_values('pnl', ascending=False).head(ntop)


def grp(g, allx):
    if not len(g):
        return dict(n=0)
    return dict(n=int(len(g)), mean_ret=float(g['ret'].mean()), median_ret=float(g['ret'].median()), win_rate=float((g['ret'] > 0).mean()), pnl_sum=float(g['pnl'].sum()),
                pnl_share_of_closed=float(g['pnl'].sum() / allx['pnl'].sum()), in_top5pct_by_ret=int(g.index.isin(top_ret.index).sum()), in_top5pct_by_pnl=int(g.index.isin(top_pnl.index).sum()),
                by_period=split_counts(g['ent_date']))


d4 = dict(n_closed=int(len(x)), n_top5pct=ntop, top5pct_ret_threshold=float(top_ret['ret'].min()), all_closed=grp(x, x),
          entry_locked_A=grp(x[x['ent_lock_A']], x), entry_locked_B=grp(x[x['ent_lock_B']], x), entry_locked_A_opened=grp(x[x['ent_lock_A'] & ~x['ent_lock_B']], x),
          exit_locked_A=grp(x[x['exit_lock_A']], x), exit_locked_B=grp(x[x['exit_lock_B']], x),
          open_end_entry_locked_A=int((lg['why'] == 'open_end').to_numpy()[lg['ent_lock_A'].to_numpy()].sum()))
if MODE == 'cont':
    for nm, m_ in (('before_2015_07', x['ent_date'] < pd.Timestamp('2015-07-01')), ('from_2015_07', x['ent_date'] >= pd.Timestamp('2015-07-01'))):
        xs_ = x[m_]
        d4[nm] = dict(n_closed=int(len(xs_)), all_closed=dict(n=int(len(xs_)), mean_ret=float(xs_['ret'].mean()), median_ret=float(xs_['ret'].median()), pnl_sum=float(xs_['pnl'].sum())),
                      entry_locked_A=dict(n=int(xs_['ent_lock_A'].sum()), mean_ret=float(xs_.loc[xs_['ent_lock_A'], 'ret'].mean()) if xs_['ent_lock_A'].any() else None,
                                          pnl_sum=float(xs_.loc[xs_['ent_lock_A'], 'pnl'].sum())))
desc['D4_NONE'] = d4
fl = lg[lg['ent_lock_A'] | lg['exit_lock_A']].drop(columns=['j'])
fl.to_csv(HERE / f'd4_locked_trades_{MODE}.csv', index=False)
out['desc'] = desc
del LOGS, lg; gc.collect()

# ---------------------------------------------------------------- 20 條路徑（正式引擎）
if len(CAPS) > 1:
    PV = {sc: {} for sc in ['BASE'] + list(SCEN)}
    for s in SHIFTS:
        Sg = Sg0 if s == 0 else s10lib.make_signal_s10(P, U, RT, shift=s)
        for cap in CAPS:
            key = f'{int(cap/1e4)}萬|+{s}'
            for sc in PV:
                nav, lg, st = run(Sg, cap, sc)
                PV[sc][key] = dict(cagr_since2019=stlib.cagr(nav, D19, DE), cagr_full=stlib.cagr(nav, D0, DE), maxdd=stlib.maxdd(nav, D0, DE),
                                   n_closed=int((lg['why'] != 'open_end').sum()), blk_limup=int(st['blk_limup']), sell_defer_limdn=int(st['sell_defer_limdn']))
                del nav, lg, st
        if s:
            del Sg
        gc.collect()
        print('路徑 位移', s, f'{time.time()-t00:.0f}s', rss(), 'MB', flush=True)
    summ = {}
    for sc, d in PV.items():
        a = np.array([v['cagr_since2019'] for v in d.values()]); f_ = np.array([v['cagr_full'] for v in d.values()]); dd = np.array([v['maxdd'] for v in d.values()])
        da = a - np.array([PV['BASE'][k]['cagr_since2019'] for k in d]); df_ = f_ - np.array([PV['BASE'][k]['cagr_full'] for k in d])
        summ[sc] = dict(cagr19_median=float(np.median(a)), cagr19_min=float(a.min()), cagr19_max=float(a.max()), cagr_full_median=float(np.median(f_)),
                        maxdd_median=float(np.median(dd)), maxdd_worst=float(dd.min()),
                        d19_vs_BASE_pp=dict(median=float(np.median(da) * 100), min=float(da.min() * 100), max=float(da.max() * 100), mean=float(da.mean() * 100), n_pos=int((da > 1e-12).sum()), n_neg=int((da < -1e-12).sum())),
                        dfull_vs_BASE_pp=dict(median=float(np.median(df_) * 100), min=float(df_.min() * 100), max=float(df_.max() * 100), mean=float(df_.mean() * 100)),
                        blk_limup_total=int(sum(v['blk_limup'] for v in d.values())), sell_defer_total=int(sum(v['sell_defer_limdn'] for v in d.values())))
    out['paths'] = PV; out['paths_summary'] = summ
    for sc in summ:
        print('路徑彙總', sc, json.dumps(summ[sc], ensure_ascii=False), flush=True)
out['peak_mb'] = rss(); out['secs'] = round(time.time() - t00)
(HERE / f'out_{MODE}.json').write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str))
(HERE / 'simulate_s97_generated.py.txt').write_text(SRC)
print('done', out['secs'], 's', out['peak_mb'], 'MB')
