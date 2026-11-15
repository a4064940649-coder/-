#!/usr/bin/env python3
"""S8：壓回撤——單檔上限與衛星比例。5 個新格（WN_70L_C12 主格、WN_70L_C10、WN_70L_C08、WN_60L、WN_60L_C10）＋2 個 S7 對照（W10_70L、WN_70L），全部底倉鎖定；
主格與 W10_70L 另跑執行延遲 d＝1／2。python3 -I -B run.py → results.json、trades.csv.gz、navs.csv.gz。PREREG.json 的 sha256 必須相符。單行程。"""
import sys, json, hashlib, time, resource, gc
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
import s8lib
from s8lib import s7lib, s6lib, s5lib, s4lib, s3lib, s2lib, stlib, COSTS

H = Path(__file__).resolve().parent
pre = (H / 'PREREG.json').read_bytes()
sha = hashlib.sha256(pre).hexdigest()
assert sha == (H / 'PREREG.sha256').read_text().split()[0], 'PREREG.json 與 PREREG.sha256 不符'
PRE = json.loads(pre)
t_all = time.time()
rss = lambda: round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
P = s8lib.load()
assert (P.T, P.N, str(P.IDX[-1].date())) == (2876, 2142, '2026-10-22'), 'P0 失敗'
U0 = stlib.universe(P, 'twse')
RT = s2lib.rev_tables(P)
INNOV, INNOV_NAMES = s4lib.innov_panel(P)
U = U0 & ~INNOV
del U0
for a in ('H', 'L', 'VOL', 'CHG', 'HF', 'VOLF', 'adjH', 'HI240', 'MA60', 'nbar', 'F_HI', 'IDXC', 'AO', 'listed'):
    if hasattr(P, a):
        delattr(P, a)
gc.collect()
U_ALL = np.ones((P.T, P.N), bool)
print('載入完成', f'{time.time()-t_all:.0f}s', rss(), 'MB', flush=True)
bench = pd.Series(P.E50[P.t0:], index=P.IDX[P.t0:])
i_mid = (P.t0 + P.T - 1) // 2
D0, DM, DE = P.IDX[P.t0], P.IDX[i_mid], P.IDX[-1]
D19 = P.IDX[P.IDX < pd.Timestamp('2019-01-01')][-1]
D24 = P.IDX[P.IDX < pd.Timestamp('2024-01-01')][-1]
WIN = {'since2019': (D19, DE), 'full': (D0, DE), 'since2024': (D24, DE), 'half1': (D0, DM), 'half2': (DM, DE)}
W = lambda nav: {k: stlib.window(nav, bench, a, b) for k, (a, b) in WIN.items()}
cg = lambda nav, k: stlib.cagr(nav, *WIN[k])
mdd = lambda nav: stlib.maxdd(nav, D0, DE)
CAPS = (6e5, 8e5, 1e6, 1.5e6, 2e6)
KW = dict(depth=10, mom_lo=0.10, exit='top20', keep_info=True)
S7RES = json.load(open(H.parent / 's7_locked_base' / 'results.json'))
S6RES = json.load(open(H.parent / 's6_satellite_sizing' / 'results.json'))


def slim(w, keys=('since2019', 'full', 'since2024', 'half1', 'half2')):
    return {k: dict(cagr=w[k]['cagr'], cagr_0050=w[k]['cagr_0050'], diff=w[k]['diff'], maxdd=w[k]['maxdd']) for k in keys}


def same_trades(l1, l2):
    a = l1[['code', 'ent_t', 'exit_t', 'why']].reset_index(drop=True); b = l2[['code', 'ent_t', 'exit_t', 'why']].reset_index(drop=True)
    return bool(len(a) == len(b) and (a == b).all().all())


CELLS = {c['id']: c for c in PRE['grid']['cells']}
MAIN = PRE['grid']['main']
NEW = [c for c in CELLS if c not in ('W10_70L', 'WN_70L')]


def sim(c, S, cost='strict', univ=None, **kw):
    return s8lib.simulate_s8(P, S, U if univ is None else univ, sat_w=c['sat_w'], sizing=c['sizing'], lock=c['lock'], cap=c['cap'], cost=COSTS[cost], **kw)


def sim7(c, S, cost='strict', univ=None, **kw):
    return s7lib.simulate_s7(P, S, U if univ is None else univ, sat_w=c['sat_w'], sizing=c['sizing'], lock=c['lock'], cost=COSTS[cost], **kw)


# ---------------------------------------------------------------- 驗收
S0 = s4lib.make_signal(P, U, RT, **KW)
for x in S0['info']:
    assert int(S0['sig'][x['k']].sum()) == x['n_tgt'], ('名單檔數不符', x['k'])
acc = {'P0': True}
pub = {'W10_70L': (0.4749, 0.3368, -0.3338), 'WN_70L': (0.6022, 0.4307, -0.3934)}
TOL = 0.0002 + 5e-5
for cid, (a19, afull, add) in pub.items():
    c = CELLS[cid]
    n7, l7, _ = sim7(c, S0)
    acc[f'A0_{cid}'] = dict(since2019=cg(n7, 'since2019'), full=cg(n7, 'full'), maxdd=mdd(n7),
                            ok=bool(abs(cg(n7, 'since2019') - a19) < TOL and abs(cg(n7, 'full') - afull) < TOL and abs(mdd(n7) - add) < TOL))
    n8, l8, _ = sim(c, S0)
    acc[f'A1_{cid}'] = dict(nav_identical=bool(np.allclose(n7.to_numpy(), n8.to_numpy(), rtol=0, atol=1e-6)), trades_identical=same_trades(l7, l8))
    if cid == 'WN_70L':
        n9, l9, s9 = sim(dict(c, cap=1.0), S0)
        acc['A1_WN_70L_cap1.0'] = dict(nav_identical=bool(np.allclose(n7.to_numpy(), n9.to_numpy(), rtol=0, atol=1e-6)), trades_identical=same_trades(l7, l9),
                                       n_cap_events=len(s9['cap_events']))
        del n9, l9, s9
    del n7, l7, n8, l8
S00 = s8lib.delay_signal(P, S0, 0)
na = sim(CELLS[MAIN], S0)[0]; nb = sim(CELLS[MAIN], S00, univ=U_ALL)[0]
acc[f'A2_{MAIN}'] = bool(np.allclose(na.to_numpy(), nb.to_numpy(), rtol=0, atol=1e-6))
del S00, na, nb
ok = all(v['ok'] for k, v in acc.items() if k.startswith('A0')) and all(v['nav_identical'] and v['trades_identical'] for k, v in acc.items() if k.startswith('A1')) \
    and all(v for k, v in acc.items() if k.startswith('A2'))
assert ok, ('驗收失敗', acc)
gc.collect()
print('驗收', acc, flush=True)

# ---------------------------------------------------------------- 格子
bw = {k: dict(cagr=stlib.cagr(bench, a, b), maxdd=stlib.maxdd(bench, a, b), vol=s5lib.ann_vol(bench, a, b), sharpe=stlib.sharpe(bench, a, b)) for k, (a, b) in WIN.items()}
ntg = np.array([x['n_tgt'] for x in S0['info']])
res = {'PREREG_sha256': sha, 'acceptance': acc,
       'meta': dict(panel_T=P.T, panel_N=P.N, start=str(D0.date()), mid=str(DM.date()), since2019_from=str(D19.date()), since2024_from=str(D24.date()), end=str(DE.date())),
       'list': dict(n_signal_days=len(ntg), mean_n=float(ntg.mean()), dist={int(v): int((ntg == v).sum()) for v in np.unique(ntg)}),
       'bench_0050': bw, 'bench_0050_yearly': {y: v['e0050'] for y, v in stlib.yearly(bench, bench).items()},
       'bench_0050_dd': s8lib.dd_dates(bench), 'cells': {}}
navs = {'0050': bench}; trades = []


def feats(S, lg):
    lg = lg.copy()
    k = lg['sig_t'].to_numpy(); j = lg['j'].to_numpy()
    lg['surge'] = S['rank'][k, j]
    lg['pos_ma'] = [S['pos_ma'].get(int(a), {}).get(int(b), np.nan) for a, b in zip(k, j)]
    lg['n_list'] = [int(S['sig'][int(a)].sum()) for a in k]
    lg['pre20'] = P.RET20[k, j]
    lg['close_sig'] = P.C[k, j]
    return lg


def metrics(nav, lg, st, nsd, c, full=True):
    w = W(nav)
    r = dict(windows=w, final_nav=float(nav.iloc[-1]))
    r['vol'] = {k: s5lib.ann_vol(nav, *WIN[k]) for k in ('full', 'since2019')}
    r['sharpe'] = {k: w[k]['sharpe'] for k in ('full', 'since2019')}
    r['lock_short'] = dict(shrunk=int(st['shrunk']), shrunk_amt=float(st['shrunk_amt']), skip_cash=int(st['skip_cash']))
    r['n_cap_events'] = len(st['cap_events'])
    if not full:
        return r
    r['yearly'] = stlib.yearly(nav, bench)
    r['beta_alpha'] = {k: s5lib.beta_alpha(nav, bench, *WIN[k]) for k in ('full', 'since2019')}
    r['trades_full'] = stlib.trade_stats(lg); r['trades_since2019'] = stlib.trade_stats(lg, '2019-01-01')
    r['rolling12'] = s8lib.rolling12(nav, bench)
    r['dd_dates'] = s8lib.dd_dates(nav)
    ex = s3lib.extra_stats(P, nav, lg, st, nsd)
    t1 = int(lg['ent_t'].min())
    sf = st['stockfrac'].loc[P.IDX[t1]:]
    ex['avg_stock_frac_from_first'] = float(sf.mean())
    ex['avg_stock_frac_since2019'] = float(st['stockfrac'].loc[D19:].mean())
    ex['max_stock_frac'] = float(sf.max())
    ex['avg_per_position_frac'] = float(sf.mean() / ex['avg_npos_from_first']) if ex['avg_npos_from_first'] > 0 else None
    nv1 = nav.loc[P.IDX[t1]:]
    bf = st['B'].loc[P.IDX[t1]:] / nv1
    ex['base_frac'] = dict(mean=float(bf.mean()), median=float(bf.median()), min=float(bf.min()), frac_days_lt10=float((bf < 0.10).mean()),
                           frac_days_lt20=float((bf < 0.20).mean()), frac_days_lt1=float((bf < 0.01).mean()))
    si = (1 - sf - bf)                                         # 衛星閒置（衛星現金＋衛星 0050）占總資產
    ex['sat_idle_frac'] = dict(mean=float(si.mean()), median=float(si.median()))
    for k in ('rb_up', 'rb_down', 'rb_trim_days', 'rb_trim_orders', 'rb_trim_val', 'rb_trim_short', 'b_clamped', 'b_drop_nonexec', 'blk_min', 'blk_nocash', 'blk_lot', 'noslot',
              'e50_cost', 'e50_orders', 'shrunk', 'shrunk_amt', 'skip_cash'):
        ex[k] = float(st[k]) if isinstance(st[k], float) else int(st[k])
    bs = st['base_sold']
    rel = [x[1] / float(nav.loc[P.IDX[x[0]]]) for x in bs]
    ex['base_sold'] = dict(n=len(bs), n_ge_0p1pct_nav=int(sum(v >= 0.001 for v in rel)), n_ge_1pct_nav=int(sum(v >= 0.01 for v in rel)),
                           max_pct_nav=float(max(rel, default=0.0)), total_amt=float(sum(x[1] for x in bs)))
    ev = st['short_events']
    ex['short_events'] = [dict(date=str(P.IDX[t].date()), code=str(P.COLS[j]), target=round(tg, 0), bought=round(bv, 0), kind=kd, nav=round(float(nav.loc[P.IDX[t]]), 0))
                          for t, j, tg, bv, kd in ev]
    ce = st['cap_events']
    ex['cap_events'] = [dict(date=str(P.IDX[t].date()), code=str(P.COLS[j]), wn_target=round(u, 0), cap=round(cv, 0), bought=round(bv, 0)) for t, j, u, cv, bv in ce]
    ex['cap_summary'] = dict(n=len(ce), n_days=len(set(x[0] for x in ce)), saved_amt=float(sum(x[2] - x[3] for x in ce)),
                             saved_pct_nav_sum=float(sum((x[2] - x[3]) / x[3] * c['cap'] for x in ce)) if c['cap'] else 0.0,
                             n_buys=len(st['buy_events']))
    if c['cap']:                                               # 驗收 A4
        viol = [x for x in st['buy_events'] if x[2] > c['cap'] * x[3] + 1.0]
        ex['A4_violations'] = len(viol)
    r['extra'] = ex
    return r


R = {cid: dict(spec=c, runs={}, shift={}, delay={}, paths={}, paths_dd={}) for cid, c in CELLS.items()}
nsd = len(S0['info'])
# ---- 位移 0
for cid, c in CELLS.items():
    tc = time.time()
    for cn in ('strict', 'light', 'strict_x2'):
        nav, lg, st = sim(c, S0, cn)
        navs[f'{cid}|{cn}'] = nav
        if cn == 'strict':
            navs[f'{cid}|strict|B'] = st['B']; navs[f'{cid}|strict|stockfrac'] = st['stockfrac']
        R[cid]['runs'][cn] = metrics(nav, lg, st, nsd, c, full=(cn != 'strict_x2'))
        if cn != 'strict_x2':
            F = feats(S0, lg); trades.append(F.assign(run=f'{cid}|{cn}'))
            if cn == 'strict':
                lg_main = lg
                x = F[F['why'] != 'open_end']
                R[cid]['top_trades'] = x.sort_values('pnl', ascending=False).head(5)[['code', 'ent_date', 'exit_date', 'ret', 'pnl']].astype(str).to_dict('records')
                R[cid]['top5_pnl_share_of_closed'] = float(x.sort_values('pnl', ascending=False).head(5)['pnl'].sum() / x['pnl'].sum())
    ex = R[cid]['runs']['strict']['extra']
    assert ex['base_sold']['n_ge_0p1pct_nav'] == 0, ('A3 失敗：鎖定格底倉被賣去買股', cid, ex['base_sold'])
    if c['cap']:
        assert ex['A4_violations'] == 0, ('A4 失敗：買進超過單檔上限', cid)
    blk, top = s2lib.block_from(lg_main, 5)
    nb, lb, sb = sim(c, S0, 'strict', block=blk)
    navs[f'{cid}|strict|block5'] = nb
    R[cid]['block5'] = dict(windows=slim(W(nb)), n_dropped=int(sb['blk_dropped']),
                            top=[dict(code=x.code, ent=str(x.ent_date.date()), exit=str(x.exit_date.date()), ret=float(x.ret), pnl=float(x.pnl)) for x in top.itertuples()])
    R[cid]['shift']['0'] = slim(R[cid]['runs']['strict']['windows'])
    R[cid]['delay']['0'] = slim(R[cid]['runs']['strict']['windows'])
    R[cid]['paths']['100萬|+0'] = R[cid]['runs']['strict']['windows']['since2019']['cagr']
    R[cid]['paths_dd']['100萬|+0'] = R[cid]['runs']['strict']['windows']['full']['maxdd']
    for cap in CAPS:
        if cap != 1e6:
            nx = sim(c, S0, 'strict', capital=cap)[0]
            R[cid]['paths'][f'{int(cap/1e4)}萬|+0'] = cg(nx, 'since2019'); R[cid]['paths_dd'][f'{int(cap/1e4)}萬|+0'] = mdd(nx)
    w = R[cid]['runs']['strict']['windows']; ls_ = R[cid]['runs']['strict']['lock_short']
    print(f"{cid} 2019起 {w['since2019']['cagr']*100:.2f}% 全期 {w['full']['cagr']*100:.2f}% 2024起 {w['since2024']['cagr']*100:.2f}% dd {w['full']['maxdd']*100:.2f}% "
          f"| 縮小 {ls_['shrunk']} 跳過 {ls_['skip_cash']} 上限觸發 {ex['cap_summary']['n']} | {time.time()-tc:.0f}s {rss()}MB", flush=True)
    gc.collect()
# ---- 執行延遲 d＝1、2（只跑主格與 W10_70L）
for d in (1, 2):
    Sd = s8lib.delay_signal(P, S0, d)
    for cid in (MAIN, 'W10_70L'):
        nd_, ld_, sd_ = sim(CELLS[cid], Sd, 'strict', univ=U_ALL)
        navs[f'{cid}|strict|delay{d}'] = nd_
        R[cid]['delay'][str(d)] = slim(W(nd_))
        R[cid]['delay'][str(d)]['n_closed'] = int((ld_['why'] != 'open_end').sum())
        R[cid]['delay'][str(d)]['lock_short'] = dict(shrunk=int(sd_['shrunk']), skip_cash=int(sd_['skip_cash']))
        R[cid]['delay'][str(d)]['n_cap_events'] = len(sd_['cap_events'])
    del Sd; gc.collect()
    print('延遲', d, f'{time.time()-t_all:.0f}s', rss(), 'MB', flush=True)
# ---- 位移 1、2、3、5（S6 定義：訊號日本身往後移）
for s in (1, 2, 3, 5):
    Ss = s4lib.make_signal(P, U, RT, shift=s, **KW)
    for cid, c in CELLS.items():
        if s in (1, 3, 5):
            ns, _, _ = sim(c, Ss, 'strict')
            navs[f'{cid}|strict|shift{s}'] = ns
            R[cid]['shift'][str(s)] = slim(W(ns))
        if s <= 3:
            for cap in CAPS:
                if cap == 1e6 and s in (1, 3):
                    R[cid]['paths'][f'100萬|+{s}'] = R[cid]['shift'][str(s)]['since2019']['cagr']
                    R[cid]['paths_dd'][f'100萬|+{s}'] = mdd(navs[f'{cid}|strict|shift{s}']); continue
                nx = sim(c, Ss, 'strict', capital=cap)[0]
                R[cid]['paths'][f'{int(cap/1e4)}萬|+{s}'] = cg(nx, 'since2019'); R[cid]['paths_dd'][f'{int(cap/1e4)}萬|+{s}'] = mdd(nx)
    del Ss; gc.collect()
    print('位移', s, f'{time.time()-t_all:.0f}s', rss(), 'MB', flush=True)

# ---- 彙整、分類
for cid, Rc in R.items():
    pv = np.array(list(Rc['paths'].values())); pd_ = np.array([Rc['paths_dd'][k] for k in Rc['paths']])
    assert len(pv) == 20 and len(pd_) == 20
    Rc['paths20'] = dict(values=Rc.pop('paths'), maxdd=Rc.pop('paths_dd'), min=float(pv.min()), median=float(np.median(pv)), max=float(pv.max()), n=int(len(pv)),
                         n_above_0050=int((pv > bw['since2019']['cagr']).sum()),
                         maxdd_min=float(pd_.min()), maxdd_median=float(np.median(pd_)), maxdd_max=float(pd_.max()), n_dd_within_36=int((pd_ >= -0.36).sum()))
    w = Rc['runs']['strict']['windows']; x2 = Rc['runs']['strict_x2']['windows']; b5 = Rc['block5']['windows']
    chk = dict(net_since2019=w['since2019']['diff'] > 0, net_full=w['full']['diff'] > 0, costx2_since2019=x2['since2019']['diff'] > 0, costx2_full=x2['full']['diff'] > 0,
               block5_since2019=b5['since2019']['diff'] > 0, block5_full=b5['full']['diff'] > 0, half1=w['half1']['diff'] > 0, half2=w['half2']['diff'] > 0)
    Rc['checks'] = {k: bool(v) for k, v in chk.items()}
    Rc['class'] = '值得上XQ測' if all(chk.values()) else ('不值得' if not (chk['net_since2019'] or chk['net_full']) else '有條件')
    Rc['fails'] = [k for k, v in chk.items() if not v]
    dl = Rc['delay']
    if '1' in dl:
        Rc['delay_cost_pp'] = {d: {k: (dl[d][k]['cagr'] - dl['0'][k]['cagr']) * 100 for k in ('since2019', 'full', 'since2024')} for d in ('1', '2')}
    res['cells'][cid] = Rc
    ts = Rc['runs']['strict']['trades_full']; ba = Rc['runs']['strict']['beta_alpha']['full']; ex = Rc['runs']['strict']['extra']
    print(f"{cid} {w['since2019']['cagr']*100:.2f}/{w['full']['cagr']*100:.2f}/{w['since2024']['cagr']*100:.2f} dd {w['full']['maxdd']*100:.2f} "
          f"| x2 {x2['since2019']['cagr']*100:.2f}/{x2['full']['cagr']*100:.2f} | blk5 {b5['since2019']['cagr']*100:.2f}/{b5['full']['cagr']*100:.2f} "
          f"| 半 {w['half1']['cagr']*100:.2f}/{w['half2']['cagr']*100:.2f} | n {ts['n']} EV {ts['expectancy']*100:.2f} | 個股 {ex['avg_stock_frac_from_first']*100:.1f}% "
          f"| 檔數 {ex['avg_npos_from_first']:.2f} | 衛星閒置 {ex['sat_idle_frac']['mean']*100:.1f}% | beta {ba['beta']:.2f} a {ba['alpha_m']*100:.2f}% t {ba['alpha_t']:.2f} "
          f"| 路徑 {pv.min()*100:.1f}/{np.median(pv)*100:.1f}/{pv.max()*100:.1f} dd {pd_.min()*100:.1f}/{np.median(pd_)*100:.1f}/{pd_.max()*100:.1f} "
          f"| 拉回 {ex['rb_up']}/{ex['rb_down']} 減碼 {ex['rb_trim_days']}天 | 上限 {ex['cap_summary']['n']} | {Rc['class']} {Rc['fails']}", flush=True)

# ---- 保住率、回撤壓回（對 WN_70L）與主格判定
ref = res['cells']['WN_70L']['runs']['strict']['windows']
eff = {}
for cid in CELLS:
    if cid == 'WN_70L':
        continue
    a = res['cells'][cid]['runs']['strict']['windows']
    pa, pb = res['cells'][cid]['paths20'], res['cells']['WN_70L']['paths20']
    ddiff = np.array([(pa['maxdd'][k] - pb['maxdd'][k]) * 100 for k in pa['values']])
    rdiff = np.array([(pa['values'][k] - pb['values'][k]) * 100 for k in pa['values']])
    eff[cid] = dict(keep_ratio={k: (a[k]['cagr'] - bw[k]['cagr']) / (ref[k]['cagr'] - bw[k]['cagr']) for k in ('since2019', 'full', 'since2024')},
                    ret_diff_pp={k: (a[k]['cagr'] - ref[k]['cagr']) * 100 for k in ('since2019', 'full', 'since2024')},
                    dd_shallower_pp=(abs(ref['full']['maxdd']) - abs(a['full']['maxdd'])) * 100,
                    paths_dd_shallower_n=int((ddiff > 0).sum()), paths_dd_diff_median_pp=float(np.median(ddiff)),
                    paths_ret_diff_median_pp=float(np.median(rdiff)), paths_ret_pos_n=int((rdiff > 0).sum()))
res['vs_WN_70L'] = eff
m = eff[MAIN]; mw = res['cells'][MAIN]['runs']['strict']['windows']
c1 = mw['full']['maxdd'] >= -0.36
c2 = m['keep_ratio']['since2019'] >= 0.75 and m['keep_ratio']['full'] >= 0.75
c3 = m['paths_dd_shallower_n'] >= 15
res['main_verdict'] = dict(dd_within_36=bool(c1), keep_ge_75=bool(c2), paths_dd_shallower_ge15=bool(c3),
                           verdict='能' if (c1 and c2 and c3) else ('能壓回但代價大' if c1 and not c2 else ('壓回成立但不穩定' if c1 and c2 else '不能（主格）')))
print('主格判定', res['main_verdict'], json.dumps(eff, ensure_ascii=False), flush=True)

# ---- 取捨表（a）本題 7 格＋0050；（b）加 S7 其他 7 格＋S6 其他 5 格；（c）同（b）拿掉名次加權
rows = {cid: dict(dd=Rc['runs']['strict']['windows']['full']['maxdd'], r19=Rc['runs']['strict']['windows']['since2019']['cagr'],
                  rfull=Rc['runs']['strict']['windows']['full']['cagr'], src='S8') for cid, Rc in res['cells'].items()}
rows['0050'] = dict(dd=bw['full']['maxdd'], r19=bw['since2019']['cagr'], rfull=bw['full']['cagr'], src='0050')
rows_b = dict(rows)
for cid in ('W7_70L', 'W10_80L', 'W7_80L', 'WN_80L', 'W10_70', 'WN_70', 'WN_100'):
    w7 = S7RES['cells'][cid]['runs']['strict']['windows']
    rows_b['S7_' + cid] = dict(dd=w7['full']['maxdd'], r19=w7['since2019']['cagr'], rfull=w7['full']['cagr'], src='S7 results.json')
for cid in ('W7_70', 'W10_100', 'W7_100', 'RW10_70', 'RWN_70'):
    w6 = S6RES['cells'][cid]['runs']['strict']['windows']
    rows_b['S6_' + cid] = dict(dd=w6['full']['maxdd'], r19=w6['since2019']['cagr'], rfull=w6['full']['cagr'], src='S6 results.json')
rows_c = {k: v for k, v in rows_b.items() if k not in ('S6_RW10_70', 'S6_RWN_70')}
fr = s8lib.frontier
res['tradeoff'] = dict(rows_b=rows_b,
                       frontier_a_since2019=fr(rows, 'r19'), frontier_a_full=fr(rows, 'rfull'),
                       frontier_b_since2019=fr(rows_b, 'r19'), frontier_b_full=fr(rows_b, 'rfull'),
                       frontier_c_since2019=fr(rows_c, 'r19'), frontier_c_full=fr(rows_c, 'rfull'), note='最大回撤用全期、嚴成本')
for k in ('a', 'b', 'c'):
    print(f'前緣({k}) 2019起', res['tradeoff'][f'frontier_{k}_since2019'], '全期', res['tradeoff'][f'frontier_{k}_full'], flush=True)

res['meta']['runtime_s'] = round(time.time() - t_all, 1); res['meta']['maxrss_mb'] = rss()
(H / 'results.json').write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str))
pd.DataFrame(navs).round(4).to_csv(H / 'navs.csv.gz', index_label='date')
tr = pd.concat(trades, ignore_index=True)
tr = tr[['run', 'code', 'market', 'sig_date', 'ent_date', 'exit_date', 'hold_days', 'shares', 'cost', 'proceeds', 'pnl', 'ret', 'why', 'surge', 'pos_ma', 'n_list', 'pre20', 'close_sig']]
num = tr.select_dtypes('number').columns
tr[num] = tr[num].round(6)
tr.to_csv(H / 'trades.csv.gz', index=False)
print(res['meta']['runtime_s'], 's', res['meta']['maxrss_mb'], 'MB')
