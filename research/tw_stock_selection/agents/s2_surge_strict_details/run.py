#!/usr/bin/env python3
"""S2：八格 × {輕, 嚴, 零股, 壓力 0.5%, 嚴×2} × {空位現金, 空位 0050} ＋ 禁買最賺 5／10 筆（整段持有期）＋ 換股日位移 3／5／10 ＋ 三個診斷。
python3 -I -B run.py → results.json、trades.csv.gz、navs.csv.gz。必須先有 PREREG.json（sha256 會寫進 results.json）。"""
import sys, json, hashlib, time, resource
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
import s2lib
from s2lib import stlib, COSTS

H = Path(__file__).resolve().parent
pre = (H / 'PREREG.json').read_bytes()
sha = hashlib.sha256(pre).hexdigest()
assert sha == (H / 'PREREG.sha256').read_text().split()[0], 'PREREG.json 與 PREREG.sha256 不符'
t_all = time.time()
P = s2lib.load()
U = stlib.universe(P, 'twse')
RT = s2lib.rev_tables(P)
bench = pd.Series(P.E50[P.t0:], index=P.IDX[P.t0:])
i_mid = (P.t0 + P.T - 1) // 2
D0, DM, DE = P.IDX[P.t0], P.IDX[i_mid], P.IDX[-1]
D19 = P.IDX[P.IDX < pd.Timestamp('2019-01-01')][-1]
D24 = P.IDX[P.IDX < pd.Timestamp('2024-01-01')][-1]
WIN = {'since2019': (D19, DE), 'full': (D0, DE), 'since2024': (D24, DE), 'half1': (D0, DM), 'half2': (DM, DE), 'y2019_2023': (D19, D24)}

# ---- 驗收：輕成本 B0 必須與 stlib.simulate 完全相同
S_ref = stlib.sig_revsurge(P, U)
S_b0 = s2lib.make_signal(P, U, RT)
assert (S_ref['sig'] == S_b0['sig']).all() and (S_ref['X'] == S_b0['X']).all()
accept = {}
for idle in ('cash', 'e50'):
    n0, _, _ = stlib.simulate(P, S_ref, U, idle=idle)
    n1, _, _ = s2lib.simulate(P, S_b0, U, idle=idle, cost=COSTS['light'])
    accept[idle] = float((n0 - n1).abs().max())
    assert accept[idle] < 0.01, ('引擎驗收失敗', idle, accept[idle])
print('驗收通過', accept, flush=True)

GRID = {
    'B0': dict(sigkw=dict(kind='surge', nsel=10), slots=10, buy_at='open'),
    'D1': dict(sigkw=dict(kind='surge', nsel=10, filt=P.F_MA60), slots=10, buy_at='open'),
    'D2': dict(sigkw=dict(kind='surge', nsel=10, filt=P.F_HI), slots=10, buy_at='open'),
    'D3': dict(sigkw=dict(kind='surge', nsel=10), slots=10, buy_at='close'),
    'D4': dict(sigkw=dict(kind='surge', nsel=10, max_hold=20), slots=10, buy_at='open'),
    'D5': dict(sigkw=dict(kind='surge', nsel=20), slots=20, buy_at='open'),
    'D6': dict(sigkw=dict(kind='surge', nsel=10, filt=P.F_MA60), slots=10, buy_at='close'),
    'C4': dict(sigkw=dict(kind='yoy3', nsel=10, filt=P.F_MA60), slots=10, buy_at='open'),
}
k2m = {x['k']: x['month'] for x in S_b0['info']}
MI = {str(m): i for i, m in enumerate(RT['S'].index)}
SA, YA, Y3A = RT['S'].to_numpy(), RT['YOY'].to_numpy(), RT['YOY3'].to_numpy()


def W(nav):
    return {k: stlib.window(nav, bench, a, b) for k, (a, b) in WIN.items()}


def slim(w, keys=('since2019', 'full')):
    return {k: dict(cagr=w[k]['cagr'], cagr_0050=w[k]['cagr_0050'], diff=w[k]['diff'], maxdd=w[k]['maxdd']) for k in keys}


def feats(lg):
    """進場當下（訊號日）的特徵。"""
    k = lg['sig_t'].to_numpy(); j = lg['j'].to_numpy()
    lg = lg.copy()
    mi = np.array([MI.get(k2m.get(a, ''), -1) for a in k])
    for nm, A in (('surge', SA), ('yoy_m', YA), ('yoy3', Y3A)):
        lg[nm] = np.where(mi >= 0, A[np.maximum(mi, 0), j], np.nan)
    lg['near_hi240'] = P.NEARHI[k, j]
    lg['above_ma60'] = P.F_MA60[k, j]
    lg['close_sig'] = P.C[k, j]
    lg['v60'] = P.V60[k, j]
    lg['pre20'] = P.G[k, j] / P.G[np.maximum(k - 20, 0), j] - 1
    lg['ind2'] = lg['code'].str[:2]
    lg['gap_open'] = P.O[lg['ent_t'].to_numpy(), j] / P.REF[lg['ent_t'].to_numpy(), j] - 1
    return lg


res = {'PREREG_sha256': sha, 'acceptance_max_nav_diff_vs_stlib': accept,
       'meta': dict(panel_T=P.T, panel_N=P.N, start=str(D0.date()), mid=str(DM.date()), since2019_from=str(D19.date()), since2024_from=str(D24.date()), end=str(DE.date()),
                    n_signal_days=len(S_b0['info']), first_signal=S_b0['info'][0]['sig_date'], last_signal=S_b0['info'][-1]['sig_date']),
       'bench_0050': {k: dict(cagr=stlib.cagr(bench, a, b), maxdd=stlib.maxdd(bench, a, b)) for k, (a, b) in WIN.items()},
       'costs': COSTS, 'grid': {}, 'targets': {}, 'diagnostics': {}}
navs = {'0050': bench}; trades = []; keep = {}

for gid, g in GRID.items():
    S = s2lib.make_signal(P, U, RT, **g['sigkw'])
    res['targets'][gid] = [dict(sig_date=x['sig_date'], month=x['month'], n_ok=x['n_ok'], targets=x['targets'], scores=x['scores']) for x in S['info']]
    R = {'runs': {}}
    for idle in ('cash', 'e50', 'e50legacy'):                 # e50＝事後修正可用資金的版本（報這個）；e50legacy＝與 stlib 完全相同（留底）
        for cn in ('light', 'strict', 'oddlot', 'stress05', 'strict_x2'):
            nav, lg, st = s2lib.simulate(P, S, U, idle=idle[:3] if idle != 'cash' else idle, cost=COSTS[cn], slots=g['slots'], buy_at=g['buy_at'], e50_fix=(idle == 'e50'))
            key = f'{gid}|{idle}|{cn}'
            navs[key] = nav
            r = dict(windows=W(nav), final_nav=float(nav.iloc[-1]),
                     engine={k: float(v) for k, v in st.items()},
                     roundtrip_cost_pct=float((st['fee'] + st['tax'] + st['slip']) / st['buy_val']) if st['buy_val'] > 0 else None)
            if cn in ('light', 'strict') and idle != 'e50legacy':
                r['yearly'] = stlib.yearly(nav, bench)
                r['trades_full'] = stlib.trade_stats(lg); r['trades_since2019'] = stlib.trade_stats(lg, '2019-01-01')
                trades.append(feats(lg).assign(run=key))
            R['runs'][f'{idle}|{cn}'] = r
            if idle == 'cash' and cn == 'strict':
                keep[gid] = (S, nav, lg, st)
    S, nav, lg, st = keep[gid]
    # ---- 整段持有期禁買最賺 5／10 筆
    for n in (5, 10):
        blk, top = s2lib.block_from(lg, n)
        nb, lb, sb = s2lib.simulate(P, S, U, idle='cash', cost=COSTS['strict'], block=blk, slots=g['slots'], buy_at=g['buy_at'])
        navs[f'{gid}|cash|strict|block{n}'] = nb
        R[f'block{n}'] = dict(windows=slim(W(nb), ('since2019', 'full', 'since2024')), n_dropped=int(sb['blk_dropped']),
                              top=[dict(code=x.code, ent=str(x.ent_date.date()), exit=str(x.exit_date.date()), ret=float(x.ret), pnl=float(x.pnl)) for x in top.itertuples()],
                              top_pnl_share_of_closed=float(top['pnl'].sum() / lg[lg['why'] != 'open_end']['pnl'].sum()))
    # ---- 換股日位移
    R['shift'] = {'0': slim(R['runs']['cash|strict']['windows'], ('since2019', 'full', 'since2024'))}
    for s in (3, 5, 10):
        Ss = s2lib.make_signal(P, U, RT, shift=s, **g['sigkw'])
        ns, _, _ = s2lib.simulate(P, Ss, U, idle='cash', cost=COSTS['strict'], slots=g['slots'], buy_at=g['buy_at'])
        navs[f'{gid}|cash|strict|shift{s}'] = ns
        R['shift'][str(s)] = slim(W(ns), ('since2019', 'full', 'since2024'))
    res['grid'][gid] = R
    w = R['runs']['cash|strict']['windows']; wl = R['runs']['cash|light']['windows']; ts = R['runs']['cash|strict']['trades_full']
    print(f"{gid} 嚴 2019起 {w['since2019']['cagr']*100:.2f}% 全期 {w['full']['cagr']*100:.2f}% 2024起 {w['since2024']['cagr']*100:.2f}% | 輕 {wl['since2019']['cagr']*100:.2f}% | x2 {R['runs']['cash|strict_x2']['windows']['since2019']['cagr']*100:.2f}% | blk5 {R['block5']['windows']['since2019']['cagr']*100:.2f}% blk10 {R['block10']['windows']['since2019']['cagr']*100:.2f}% | n {ts['n']} win {ts['win_rate']*100:.1f}% med {ts['median_ret']*100:.2f}% | {time.time()-t_all:.0f}s", flush=True)

# ---- 分類與相對 B0
b0nav = keep['B0'][1]; b0nav_e = navs['B0|e50|strict']
for gid in GRID:
    R = res['grid'][gid]; w = R['runs']['cash|strict']['windows']; x2 = R['runs']['cash|strict_x2']['windows']; b5 = R['block5']['windows']
    chk = dict(net_since2019=w['since2019']['diff'] > 0, net_full=w['full']['diff'] > 0, costx2_since2019=x2['since2019']['diff'] > 0, costx2_full=x2['full']['diff'] > 0,
               block5_since2019=b5['since2019']['diff'] > 0, block5_full=b5['full']['diff'] > 0, half1=w['half1']['diff'] > 0, half2=w['half2']['diff'] > 0)
    cls = '值得上XQ測' if all(chk.values()) else ('不值得' if not (chk['net_since2019'] or chk['net_full']) else '有條件')
    R['checks'] = {k: bool(v) for k, v in chk.items()}; R['class'] = cls; R['fails'] = [k for k, v in chk.items() if not v]
    if gid != 'B0':
        wb = res['grid']['B0']['runs']['cash|strict']['windows']
        pr = {'since2016_03': s2lib.paired(keep[gid][1], b0nav, '2016-03'), 'since2019_01': s2lib.paired(keep[gid][1], b0nav, '2019-01'),
              'idle0050_since2016_03': s2lib.paired(navs[f'{gid}|e50|strict'], b0nav_e, '2016-03')}
        five = dict(paired_mean=pr['since2016_03']['mean'], **{k: w[k]['cagr'] - wb[k]['cagr'] for k in ('since2019', 'full', 'half1', 'half2')})
        lab = '改善' if all(v > 0 for v in five.values()) else ('變差' if all(v < 0 for v in five.values()) else '沒差')
        R['paired_vs_B0'] = pr; R['vs_B0_five'] = five; R['vs_B0'] = lab
        R['xs'] = bool(cls in ('值得上XQ測', '有條件') and lab == '改善')
    print(gid, cls, R['fails'], R.get('vs_B0'), R.get('vs_B0_five'), flush=True)

# ---- 診斷 1：獲利集中在哪（B0、嚴、空位現金）
S, nav, lg, st = keep['B0']
F = feats(lg); cl = F[F['why'] != 'open_end'].copy()
top = cl.sort_values('ret', ascending=False).head(20); bot = cl.sort_values('ret').head(20)
rest = cl.drop(top.index).drop(bot.index)
topamt = cl.sort_values('pnl', ascending=False).head(20)
fcols = ['surge', 'yoy_m', 'near_hi240', 'close_sig', 'v60', 'pre20', 'gap_open', 'hold_days', 'ret']


def desc(x):
    d = {c: float(x[c].median()) for c in fcols}
    d.update({c + '_mean': float(x[c].mean()) for c in fcols})
    d['n'] = int(len(x)); d['above_ma60'] = float(x['above_ma60'].mean()); d['near_hi_ge_090'] = float((x['near_hi240'] >= 0.90).mean())
    d['near_hi_missing'] = float(x['near_hi240'].isna().mean())
    d['ind2'] = x['ind2'].value_counts().head(8).to_dict(); d['ent_year'] = x['ent_date'].dt.year.value_counts().sort_index().to_dict()
    d['price_lt20'] = float((x['close_sig'] < 20).mean()); d['price_gt100'] = float((x['close_sig'] > 100).mean())
    return d


def rows(x):
    return [dict(code=r.code, sig=str(r.sig_date.date()), exit=str(r.exit_date.date()), hold=int(r.hold_days), ret=float(r.ret), pnl=float(r.pnl), surge=float(r.surge), yoy_m=float(r.yoy_m),
                 near_hi240=float(r.near_hi240), close=float(r.close_sig), v60=float(r.v60), pre20=float(r.pre20), ind2=r.ind2, above_ma60=bool(r.above_ma60)) for r in x.itertuples()]


res['diagnostics']['1_concentration'] = dict(top20=desc(top), bottom20=desc(bot), rest=desc(rest), all_closed=desc(cl), top20_rows=rows(top), bottom20_rows=rows(bot),
                                              overlap_top20_ret_vs_pnl=int(len(set(top.index) & set(topamt.index))),
                                              top20_ret_sum_share=float(top['ret'].sum() / cl['ret'].sum()), top20_pnl_share=float(topamt['pnl'].sum() / cl['pnl'].sum()),
                                              top10_pnl_share=float(cl.sort_values('pnl', ascending=False).head(10)['pnl'].sum() / cl['pnl'].sum()),
                                              pnl_by_exit_year=(cl.groupby(cl['exit_date'].dt.year)['pnl'].sum() / cl['pnl'].sum()).to_dict())

# ---- 診斷 2：持有期間的報酬路徑（B0、嚴、空位現金；不含成本）
paths = s2lib.trade_paths(P, lg)
res['diagnostics']['2_path'] = dict(all=s2lib.path_segments(paths),
                                     since2019=s2lib.path_segments([p for p, e in zip(paths, lg['ent_date']) if e >= pd.Timestamp('2019-01-01')]),
                                     n_trades=len(paths), mean_len=float(np.mean([len(p) for p in paths])),
                                     daily_mean_by_daynum={str(d): float(np.mean([p[d - 1] for p in paths if len(p) >= d])) for d in range(1, 26)},
                                     daily_n_by_daynum={str(d): int(sum(len(p) >= d for p in paths)) for d in range(1, 26)})

# ---- 診斷 3：2024 起輸 0050 的原因
nz, _, _ = s2lib.simulate(P, S, U, idle='cash', cost=COSTS['zero'])
ew = s2lib.univ_ew_nav(P, U, [x['k'] for x in S['info']])
j2330 = P.ci['2330']; tsmc = pd.Series(P.G[P.t0:, j2330], index=P.IDX[P.t0:])
navs['B0|cash|zero'] = nz; navs['UNIV_EW_twse'] = ew * 1e6; navs['2330'] = tsmc / tsmc.iloc[0] * 1e6
d3 = {}
for nm, (a, b) in {'y2019_2023': (D19, D24), 'since2024': (D24, DE), 'since2019': (D19, DE), 'full': (D0, DE)}.items():
    c = dict(strat_strict=stlib.cagr(nav, a, b), strat_light=stlib.cagr(navs['B0|cash|light'], a, b), strat_zero=stlib.cagr(nz, a, b), univ_ew=stlib.cagr(ew, a, b), e0050=stlib.cagr(bench, a, b), tsmc_2330=stlib.cagr(tsmc, a, b))
    c['strat_minus_ew'] = c['strat_strict'] - c['univ_ew']; c['ew_minus_0050'] = c['univ_ew'] - c['e0050']; c['strat_minus_0050'] = c['strat_strict'] - c['e0050']
    c['zero_minus_ew'] = c['strat_zero'] - c['univ_ew']; c['cost_drag'] = c['strat_zero'] - c['strat_strict']
    d3[nm] = c
yr = {}
for y in range(2016, 2027):
    a = P.IDX[P.IDX < pd.Timestamp(f'{y}-01-01')][-1]; b = P.IDX[P.IDX < pd.Timestamp(f'{y + 1}-01-01')][-1]
    f = lambda s: float(s.loc[b] / s.loc[a] - 1)
    yr[y] = dict(strat_strict=f(nav), strat_zero=f(nz), univ_ew=f(ew), e0050=f(bench), tsmc_2330=f(tsmc))
res['diagnostics']['3_since2024'] = dict(periods=d3, yearly=yr)

res['meta']['runtime_s'] = round(time.time() - t_all, 1); res['meta']['maxrss_mb'] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
(H / 'results.json').write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str))
pd.DataFrame(navs).round(2).to_csv(H / 'navs.csv.gz', index_label='date')
tr = pd.concat(trades, ignore_index=True)
tr = tr[['run', 'code', 'market', 'sig_date', 'ent_date', 'exit_date', 'hold_days', 'shares', 'cost', 'proceeds', 'pnl', 'ret', 'why', 'score', 'surge', 'yoy_m', 'yoy3', 'near_hi240', 'above_ma60',
    'close_sig', 'v60', 'pre20', 'gap_open', 'ind2']]
num = tr.select_dtypes('number').columns
tr[num] = tr[num].round(6)
tr.to_csv(H / 'trades.csv.gz', index=False)
print(res['meta'])
