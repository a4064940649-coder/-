#!/usr/bin/env python3
"""S4：E3'（E3＋排除創新板＋最小買進金額）與 F1～F5、三個鄰域。
python3 -I -B run.py → results.json、trades.csv.gz、navs.csv.gz。必須先有 PREREG.json（sha256 寫進 results.json）。"""
import sys, json, hashlib, time, resource, gc
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
import s4lib
from s4lib import s3lib, s2lib, stlib, COSTS

H = Path(__file__).resolve().parent
pre = (H / 'PREREG.json').read_bytes()
sha = hashlib.sha256(pre).hexdigest()
assert sha == (H / 'PREREG.sha256').read_text().split()[0], 'PREREG.json 與 PREREG.sha256 不符'
SEED = 20261115
t_all = time.time()
rss = lambda: round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
P = s4lib.load()
U0 = stlib.universe(P, 'twse')
RT = s2lib.rev_tables(P)
INNOV, INNOV_NAMES = s4lib.innov_panel(P)
U = U0 & ~INNOV
IND = s4lib.industry_map(P)
for a in ('H', 'L', 'VOL', 'CHG', 'HF', 'VOLF', 'adjH', 'HI240', 'MA60', 'nbar', 'F_HI', 'IDXC', 'AO', 'listed'):
    if hasattr(P, a):
        delattr(P, a)
gc.collect()
print('載入完成', f'{time.time()-t_all:.0f}s', rss(), 'MB；創新板代號', len(INNOV_NAMES), flush=True)
bench = pd.Series(P.E50[P.t0:], index=P.IDX[P.t0:])
i_mid = (P.t0 + P.T - 1) // 2
D0, DM, DE = P.IDX[P.t0], P.IDX[i_mid], P.IDX[-1]
D19 = P.IDX[P.IDX < pd.Timestamp('2019-01-01')][-1]
D24 = P.IDX[P.IDX < pd.Timestamp('2024-01-01')][-1]
WIN = {'since2019': (D19, DE), 'full': (D0, DE), 'since2024': (D24, DE), 'half1': (D0, DM), 'half2': (DM, DE)}
W = lambda nav: {k: stlib.window(nav, bench, a, b) for k, (a, b) in WIN.items()}
cg = lambda nav, k: stlib.cagr(nav, *WIN[k])


def slim(w, keys=('since2019', 'full', 'since2024', 'half1', 'half2')):
    return {k: dict(cagr=w[k]['cagr'], cagr_0050=w[k]['cagr_0050'], diff=w[k]['diff'], maxdd=w[k]['maxdd']) for k in keys}


# ---------------------------------------------------------------- 驗收
S_ref = s3lib.make_signal(P, U0, RT, mom=0.10, hold_n=20)
S_e3 = s4lib.make_signal(P, U0, RT, depth=None, mom_lo=0.10, exit='top20')
A1 = bool((S_ref['sig'] == S_e3['sig']).all() and (S_ref['X'] == S_e3['X']).all() and np.array_equal(S_ref['rank'], S_e3['rank'], equal_nan=True))
assert A1, 'A1 失敗'
del S_ref; gc.collect()
n_a, l_a, _ = s4lib.simulate(P, S_e3, U0, idle='cash', cost=COSTS['strict'], min_frac=0)
n_b, _, _ = s2lib.simulate(P, S_e3, U0, idle='cash', cost=COSTS['strict'])
acc = dict(A1=A1, E3_since2019=cg(n_a, 'since2019'), E3_full=cg(n_a, 'full'), E3_since2024=cg(n_a, 'since2024'), E3_n_closed=int((l_a['why'] != 'open_end').sum()),
           nav_identical_to_s2lib=bool(np.allclose(n_a.to_numpy(), n_b.to_numpy(), rtol=0, atol=1e-6)))
assert abs(acc['E3_since2019'] - 0.5920) < 0.0002 and abs(acc['E3_full'] - 0.4428) < 0.0002 and acc['E3_n_closed'] == 819 and acc['nav_identical_to_s2lib'], ('A2 失敗', acc)
S_e3i = s4lib.make_signal(P, U, RT, depth=None, mom_lo=0.10, exit='top20')
n_c, l_c, _ = s4lib.simulate(P, S_e3i, U, idle='cash', cost=COSTS['strict'], min_frac=0)
acc.update(A3_innov_only_since2019=cg(n_c, 'since2019'), A3_innov_only_full=cg(n_c, 'full'), A3_innov_only_since2024=cg(n_c, 'since2024'),
           A3_n_closed=int((l_c['why'] != 'open_end').sum()), A3_ok=bool(abs(cg(n_c, 'since2019') - 0.4894) < 0.0005 and abs(cg(n_c, 'full') - 0.3779) < 0.0005))
# 最小買進金額單獨的影響（含創新板、min_frac=0.5）
n_d, l_d, s_d = s4lib.simulate(P, S_e3, U0, idle='cash', cost=COSTS['strict'], min_frac=0.5)
acc.update(minbuy_only_since2019=cg(n_d, 'since2019'), minbuy_only_full=cg(n_d, 'full'), minbuy_only_since2024=cg(n_d, 'since2024'), minbuy_only_n_closed=int((l_d['why'] != 'open_end').sum()),
           minbuy_only_blk_min=int(s_d['blk_min']), E3_tiny_positions_lt1000=int((l_a['cost'] < 1000).sum()))
del S_e3, S_e3i, n_b; gc.collect()
print('驗收', {k: (round(v * 100, 3) if isinstance(v, float) else v) for k, v in acc.items()}, flush=True)

# ---------------------------------------------------------------- 格子
CELLS = {
    "E3'": dict(kw=dict(depth=None, mom_lo=0.10, exit='top20'), idle='cash'),
    'F1':  dict(kw=dict(depth=10, mom_lo=0.10, exit='top20'), idle='e50'),
    'F2':  dict(kw=dict(depth=20, mom_lo=0.10, exit='top20'), idle='cash'),
    'F3':  dict(kw=dict(depth=None, mom_lo=0.10, mom_hi=0.30, exit='top20'), idle='cash'),
    'F4':  dict(kw=dict(depth=None, mom_lo=0.10, exit='f4'), idle='cash'),
    'F5':  dict(kw=dict(depth=10, mom_lo=0.10, mom_hi=0.30, exit='f4'), idle='e50'),
    'N5':  dict(kw=dict(depth=10, mom_lo=0.05, exit='top20'), idle='e50'),
    'N15': dict(kw=dict(depth=10, mom_lo=0.15, exit='top20'), idle='e50'),
    'Nd15': dict(kw=dict(depth=15, mom_lo=0.10, exit='top20'), idle='e50'),
}
OTHER = {'cash': 'e50', 'e50': 'cash'}
CAPS = (6e5, 8e5, 1e6, 1.5e6, 2e6)
YOYa = RT['YOY'].to_numpy(); YOYi = {m: i for i, m in enumerate(RT['YOY'].index)}

res = {'PREREG_sha256': sha, 'acceptance': acc,
       'meta': dict(panel_T=P.T, panel_N=P.N, start=str(D0.date()), mid=str(DM.date()), since2019_from=str(D19.date()), since2024_from=str(D24.date()), end=str(DE.date()),
                    innov_codes=INNOV_NAMES, innov_in_univ_signal_days=None),
       'bench_0050': {k: dict(cagr=stlib.cagr(bench, a, b), maxdd=stlib.maxdd(bench, a, b)) for k, (a, b) in WIN.items()},
       'bench_0050_yearly': {y: v['e0050'] for y, v in stlib.yearly(bench, bench).items()},
       'cells': {}, 'diagnostic': {}}
navs = {'0050': bench}; trades = []; keepnav = {}; keep = {}


def feats(S, lg):
    lg = lg.copy()
    k = lg['sig_t'].to_numpy(); j = lg['j'].to_numpy()
    lg['surge'] = S['rank'][k, j]
    lg['pos_ma'] = [S['pos_ma'].get(int(a), {}).get(int(b), np.nan) for a, b in zip(k, j)]
    lg['pre20'] = P.RET20[k, j]
    lg['yoy'] = [YOYa[YOYi[S['kmonth'][int(a)]], int(b)] if int(a) in S['kmonth'] else np.nan for a, b in zip(k, j)]
    lg['near_hi240'] = P.NEARHI[k, j] - 1
    lg['close_sig'] = P.C[k, j]; lg['v60'] = P.V60[k, j]
    lg['industry'] = [IND[c] for c in lg['code']]; lg['code2'] = lg['code'].str[:2]
    lg['innov_at_sig'] = INNOV[k, j]
    return lg


for cid, c in CELLS.items():
    tc = time.time()
    S = s4lib.make_signal(P, U, RT, keep_info=True, **c['kw'])
    nsd = len(S['info']); ntg = np.array([x['n_tgt'] for x in S['info']])
    mi = c['idle']; R = {'idle_main': mi, 'runs': {}, 'list': dict(n_signal_days=nsd, months_lt10=int((ntg < 10).sum()), months_0=int((ntg == 0).sum()),
                                                                       mean_n=float(ntg.mean()), median_n=float(np.median(ntg)))}
    if cid == "E3'":
        inf = pd.DataFrame(S['info'])
        res['xs_helper_surge_nth'] = {cc: dict(q25=float(inf[cc].quantile(.25)), median=float(inf[cc].median()), q75=float(inf[cc].quantile(.75)),
                                               min=float(inf[cc].min()), max=float(inf[cc].max()), n=int(inf[cc].notna().sum())) for cc in ('surge_10th', 'surge_15th', 'surge_20th')}
        res['targets_E3p'] = [dict(sig_date=x['sig_date'], targets=x['targets']) for x in S['info']]
    for idle in (mi, OTHER[mi]):
        for cn in ('strict', 'light', 'strict_x2'):
            nav, lg, st = s4lib.simulate(P, S, U, idle=idle, cost=COSTS[cn])
            key = f'{cid}|{idle}|{cn}'; navs[key] = nav
            r = dict(windows=W(nav), final_nav=float(nav.iloc[-1]), extra=s3lib.extra_stats(P, nav, lg, st, nsd), blk_min=int(st['blk_min']))
            if cn != 'strict_x2':
                r['yearly'] = stlib.yearly(nav, bench)
                r['trades_full'] = stlib.trade_stats(lg); r['trades_since2019'] = stlib.trade_stats(lg, '2019-01-01')
                F = feats(S, lg)
                trades.append(F.assign(run=key))
                if cn == 'strict':
                    keepnav[(cid, idle)] = nav
                    if idle == mi:
                        keep[cid] = (lg, F)
                        x = F[F['why'] != 'open_end']
                        r['top_trade'] = x.sort_values('pnl', ascending=False).head(3)[['code', 'ent_date', 'exit_date', 'ret', 'pnl']].astype(str).to_dict('records')
                        r['top1_pnl_share'] = float(x['pnl'].max() / x['pnl'].sum())
            R['runs'][f'{idle}|{cn}'] = r
    lg, F = keep[cid]
    for n in (5, 10):
        blk, top = s2lib.block_from(lg, n)
        nb, lb, sb = s4lib.simulate(P, S, U, idle=mi, cost=COSTS['strict'], block=blk)
        navs[f'{cid}|{mi}|strict|block{n}'] = nb
        R[f'block{n}'] = dict(windows=slim(W(nb)), n_dropped=int(sb['blk_dropped']),
                              top=[dict(code=x.code, ent=str(x.ent_date.date()), exit=str(x.exit_date.date()), ret=float(x.ret), pnl=float(x.pnl)) for x in top.itertuples()],
                              top_pnl_share_of_closed=float(top['pnl'].sum() / lg[lg['why'] != 'open_end']['pnl'].sum()))
    # 含創新板並列欄
    Si = s4lib.make_signal(P, U0, RT, **c['kw'])
    ni, li, _ = s4lib.simulate(P, Si, U0, idle=mi, cost=COSTS['strict'])
    navs[f'{cid}|{mi}|strict|with_innov'] = ni
    xi = li[li['why'] != 'open_end']
    R['with_innov'] = dict(windows=slim(W(ni)), n_closed=int(len(xi)), innov_trades=[dict(code=a.code, ent=str(a.ent_date.date()), exit=str(a.exit_date.date()), ret=float(a.ret))
                                                                                      for a in li.itertuples() if INNOV[int(a.sig_t), int(a.j)]])
    del Si; gc.collect()
    # 位移與 20 條路徑
    R['shift'] = {'0': slim(R['runs'][f'{mi}|strict']['windows'])}
    paths = {}
    for s in (1, 2, 3, 5, 10):
        Ss = s4lib.make_signal(P, U, RT, shift=s, **c['kw'])
        if s in (3, 5, 10):
            ns, _, _ = s4lib.simulate(P, Ss, U, idle=mi, cost=COSTS['strict'])
            navs[f'{cid}|{mi}|strict|shift{s}'] = ns
            R['shift'][str(s)] = slim(W(ns))
        if s <= 3:
            for cap in CAPS:
                if cap == 1e6 and s == 3:
                    paths[f'{int(cap/1e4)}萬|+{s}'] = R['shift']['3']['since2019']['cagr']; continue
                npth, _, _ = s4lib.simulate(P, Ss, U, idle=mi, cost=COSTS['strict'], capital=cap)
                paths[f'{int(cap/1e4)}萬|+{s}'] = cg(npth, 'since2019')
        del Ss; gc.collect()
    for cap in CAPS:
        if cap == 1e6:
            paths['100萬|+0'] = R['runs'][f'{mi}|strict']['windows']['since2019']['cagr']; continue
        npth, _, _ = s4lib.simulate(P, S, U, idle=mi, cost=COSTS['strict'], capital=cap)
        paths[f'{int(cap/1e4)}萬|+0'] = cg(npth, 'since2019')
    pv = np.array(list(paths.values()))
    R['paths20'] = dict(values=paths, min=float(pv.min()), median=float(np.median(pv)), max=float(pv.max()), n=int(len(pv)),
                        n_above_0050=int((pv > res['bench_0050']['since2019']['cagr']).sum()))
    if cid == "E3'":
        keep['S_E3p'] = S
    else:
        del S
    gc.collect()
    res['cells'][cid] = R
    w = R['runs'][f'{mi}|strict']['windows']; ts = R['runs'][f'{mi}|strict']['trades_full']
    print(f"{cid}[{mi}] 2019起 {w['since2019']['cagr']*100:.2f}% 全期 {w['full']['cagr']*100:.2f}% 2024起 {w['since2024']['cagr']*100:.2f}% dd {w['full']['maxdd']*100:.1f}% "
          f"| x2 {R['runs'][f'{mi}|strict_x2']['windows']['since2019']['cagr']*100:.2f}% | blk5 {R['block5']['windows']['since2019']['cagr']*100:.2f}% | n {ts['n']} "
          f"npos {R['runs'][f'{mi}|strict']['extra']['avg_npos_from_first']:.2f} | 路徑 {pv.min()*100:.1f}/{np.median(pv)*100:.1f}/{pv.max()*100:.1f} | {time.time()-tc:.0f}s {rss()}MB", flush=True)

# ---------------------------------------------------------------- 分類與相對 E3'
for cid, c in CELLS.items():
    R = res['cells'][cid]; mi = R['idle_main']
    w = R['runs'][f'{mi}|strict']['windows']; x2 = R['runs'][f'{mi}|strict_x2']['windows']; b5 = R['block5']['windows']
    chk = dict(net_since2019=w['since2019']['diff'] > 0, net_full=w['full']['diff'] > 0, costx2_since2019=x2['since2019']['diff'] > 0, costx2_full=x2['full']['diff'] > 0,
               block5_since2019=b5['since2019']['diff'] > 0, block5_full=b5['full']['diff'] > 0, half1=w['half1']['diff'] > 0, half2=w['half2']['diff'] > 0)
    cls = '值得上XQ測' if all(chk.values()) else ('不值得' if not (chk['net_since2019'] or chk['net_full']) else '有條件')
    R['checks'] = {k: bool(v) for k, v in chk.items()}; R['class'] = cls; R['fails'] = [k for k, v in chk.items() if not v]
    if cid == "E3'":
        print(cid, cls, R['fails'], flush=True); continue
    rel = {}
    for idle in ('cash', 'e50'):
        a, b = keepnav[(cid, idle)], keepnav[("E3'", idle)]
        wa = R['runs'][f'{idle}|strict']['windows']; wb = res['cells']["E3'"]['runs'][f'{idle}|strict']['windows']
        pr = {'since2016_03': s2lib.paired(a, b, '2016-03', seed=SEED), 'since2019_01': s2lib.paired(a, b, '2019-01', seed=SEED)}
        five = dict(paired_mean=pr['since2016_03']['mean'], **{k: wa[k]['cagr'] - wb[k]['cagr'] for k in ('since2019', 'full', 'half1', 'half2')})
        lab = '改善' if all(v > 0 for v in five.values()) else ('變差' if all(v < 0 for v in five.values()) else '沒差')
        ya = R['runs'][f'{idle}|strict']['yearly']; yb = res['cells']["E3'"]['runs'][f'{idle}|strict']['yearly']
        rel[idle] = dict(paired=pr, five=five, label=lab, since2024_diff=wa['since2024']['cagr'] - wb['since2024']['cagr'],
                         yearly_diff={y: ya[y]['strat'] - yb[y]['strat'] for y in ya})
    # F1、F5、鄰域另報：主欄（空位 0050）對 E3' 空位現金（並列，不判定）
    if mi == 'e50':
        a, b = keepnav[(cid, 'e50')], keepnav[("E3'", 'cash')]
        wa = R['runs']['e50|strict']['windows']; wb = res['cells']["E3'"]['runs']['cash|strict']['windows']
        rel['e50_vs_E3p_cash'] = dict(paired=s2lib.paired(a, b, '2016-03', seed=SEED), five={k: wa[k]['cagr'] - wb[k]['cagr'] for k in ('since2019', 'full', 'half1', 'half2', 'since2024')})
    R['vs_E3p'] = rel; R['vs_E3p_label'] = rel[mi]['label']
    R['xs'] = bool(cls != '不值得' and R['vs_E3p_label'] == '改善' and cid in ('F1', 'F2', 'F3', 'F4', 'F5'))
    print(cid, cls, R['fails'], R['vs_E3p_label'], {k: round(v * 100, 3) for k, v in rel[mi]['five'].items()}, 't', round(rel[mi]['paired']['since2016_03']['t'], 2), flush=True)

# ---------------------------------------------------------------- 診斷：E3' 最賺 20 筆與最賠 20 筆
lg, F = keep["E3'"]
x = F[F['why'] != 'open_end'].copy()
NUM = ['surge', 'yoy', 'pre20', 'near_hi240', 'close_sig', 'v60', 'pos_ma']
diag = {}
for by in ('ret', 'pnl'):
    srt = x.sort_values(by, ascending=False)
    top, bot = srt.head(20), srt.tail(20)
    d = {'n_all': int(len(x)), 'features': {}}
    for f in NUM:
        d['features'][f] = dict(top20=s4lib.desc(top[f]), bottom20=s4lib.desc(bot[f]), all=s4lib.desc(x[f]), auc_top_vs_bottom=s4lib.auc(top[f], bot[f]),
                                auc_top_vs_rest=s4lib.auc(top[f], srt.iloc[20:][f]), auc_bottom_vs_rest=s4lib.auc(bot[f], srt.iloc[:-20][f]))
    for f in ('industry', 'code2'):
        d['features'][f] = dict(top20=top[f].value_counts().to_dict(), bottom20=bot[f].value_counts().to_dict(), all_top8=x[f].value_counts().head(8).to_dict())
    d['top20_list'] = top[['code', 'ent_date', 'exit_date', 'ret', 'pnl', 'surge', 'yoy', 'pre20', 'near_hi240', 'close_sig', 'v60', 'pos_ma', 'industry']].astype(str).to_dict('records')
    d['bottom20_list'] = bot[['code', 'ent_date', 'exit_date', 'ret', 'pnl', 'surge', 'yoy', 'pre20', 'near_hi240', 'close_sig', 'v60', 'pos_ma', 'industry']].astype(str).to_dict('records')
    d['top20_year'] = top['ent_date'].dt.year.value_counts().sort_index().to_dict(); d['bottom20_year'] = bot['ent_date'].dt.year.value_counts().sort_index().to_dict()
    d['top20_hold_days'] = s4lib.desc(top['hold_days']); d['bottom20_hold_days'] = s4lib.desc(bot['hold_days'])
    diag['by_' + by] = d
diag['by_entry_rank'] = {lab: dict(**s4lib.desc(x.loc[f, 'ret']), mean_hold=float(x.loc[f, 'hold_days'].mean()) if f.any() else None)
                         for lab, f in (('1-10', x['pos_ma'] <= 10), ('11-20', (x['pos_ma'] > 10) & (x['pos_ma'] <= 20)), ('21+', x['pos_ma'] > 20))}
diag['by_pre20'] = {lab: s4lib.desc(x.loc[f, 'ret']) for lab, f in (('10-30', x['pre20'] <= 0.30), ('>30', x['pre20'] > 0.30))}
res['diagnostic'] = diag

res['meta']['runtime_s'] = round(time.time() - t_all, 1); res['meta']['maxrss_mb'] = rss()
(H / 'results.json').write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str))
pd.DataFrame(navs).round(2).to_csv(H / 'navs.csv.gz', index_label='date')
tr = pd.concat(trades, ignore_index=True)
tr = tr[['run', 'code', 'market', 'sig_date', 'ent_date', 'exit_date', 'hold_days', 'shares', 'cost', 'proceeds', 'pnl', 'ret', 'why', 'surge', 'pos_ma', 'pre20', 'yoy',
         'near_hi240', 'close_sig', 'v60', 'industry', 'innov_at_sig']]
num = tr.select_dtypes('number').columns
tr[num] = tr[num].round(6)
tr.to_csv(H / 'trades.csv.gz', index=False)
print(res['meta']['runtime_s'], 's', res['meta']['maxrss_mb'], 'MB')
