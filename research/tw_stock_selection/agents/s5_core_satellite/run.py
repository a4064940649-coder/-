#!/usr/bin/env python3
"""S5：0050 底倉＋F1 規則衛星（6 格＋對照）。先跑 prep.py（建面板快取）。
python3 -I -B run.py → results.json、trades.csv.gz、navs.csv.gz。PREREG.json 的 sha256 必須相符。"""
import sys, json, hashlib, time, resource, gc
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
import s5lib
from s5lib import s4lib, s3lib, s2lib, stlib, COSTS

H = Path(__file__).resolve().parent
pre = (H / 'PREREG.json').read_bytes()
sha = hashlib.sha256(pre).hexdigest()
assert sha == (H / 'PREREG.sha256').read_text().split()[0], 'PREREG.json 與 PREREG.sha256 不符'
PRE = json.loads(pre)
t_all = time.time()
rss = lambda: round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
P = s5lib.load()
U0 = stlib.universe(P, 'twse')
RT = s2lib.rev_tables(P)
INNOV, INNOV_NAMES = s4lib.innov_panel(P)
U = U0 & ~INNOV
del U0
for a in ('H', 'L', 'VOL', 'CHG', 'HF', 'VOLF', 'adjH', 'HI240', 'MA60', 'nbar', 'F_HI', 'IDXC', 'AO', 'listed'):
    if hasattr(P, a):
        delattr(P, a)
gc.collect()
print('載入完成', f'{time.time()-t_all:.0f}s', rss(), 'MB', flush=True)
bench = pd.Series(P.E50[P.t0:], index=P.IDX[P.t0:])
i_mid = (P.t0 + P.T - 1) // 2
D0, DM, DE = P.IDX[P.t0], P.IDX[i_mid], P.IDX[-1]
D19 = P.IDX[P.IDX < pd.Timestamp('2019-01-01')][-1]
D24 = P.IDX[P.IDX < pd.Timestamp('2024-01-01')][-1]
WIN = {'since2019': (D19, DE), 'full': (D0, DE), 'since2024': (D24, DE), 'half1': (D0, DM), 'half2': (DM, DE)}
W = lambda nav: {k: stlib.window(nav, bench, a, b) for k, (a, b) in WIN.items()}
cg = lambda nav, k: stlib.cagr(nav, *WIN[k])
CAPS = (6e5, 8e5, 1e6, 1.5e6, 2e6)


def slim(w, keys=('since2019', 'full', 'since2024', 'half1', 'half2')):
    return {k: dict(cagr=w[k]['cagr'], cagr_0050=w[k]['cagr_0050'], diff=w[k]['diff'], maxdd=w[k]['maxdd']) for k in keys}


def same_trades(l1, l2):
    a = l1[['code', 'ent_t', 'exit_t', 'why']].reset_index(drop=True); b = l2[['code', 'ent_t', 'exit_t', 'why']].reset_index(drop=True)
    return bool(len(a) == len(b) and (a == b).all().all())


# ---------------------------------------------------------------- 驗收
tA = time.time()
S_f1 = s4lib.make_signal(P, U, RT, depth=10, mom_lo=0.10, exit='top20', keep_info=True)
n4, l4, s4 = s4lib.simulate(P, S_f1, U, idle='e50', cost=COSTS['strict'])
acc = dict(A1_F1_since2019=cg(n4, 'since2019'), A1_F1_full=cg(n4, 'full'), A1_F1_n_closed=int((l4['why'] != 'open_end').sum()), sim_seconds=round(time.time() - tA, 2))
acc['A1_ok'] = bool(abs(acc['A1_F1_since2019'] - 0.5363) < 0.0002 and abs(acc['A1_F1_full'] - 0.3770) < 0.0002 and acc['A1_F1_n_closed'] == 389)
assert acc['A1_ok'], ('A1 失敗', acc)
n5, l5, _ = s5lib.simulate_cs(P, S_f1, U, sat_w=1.0, cost=COSTS['strict'])
acc['A2_F1_nav_identical'] = bool(np.allclose(n4.to_numpy(), n5.to_numpy(), rtol=0, atol=1e-6)); acc['A2_F1_trades_identical'] = same_trades(l4, l5)
S_e3 = s4lib.make_signal(P, U, RT, depth=None, mom_lo=0.10, exit='top20', keep_info=True)
n4e, l4e, _ = s4lib.simulate(P, S_e3, U, idle='e50', cost=COSTS['strict'])
n5e, l5e, _ = s5lib.simulate_cs(P, S_e3, U, sat_w=1.0, cost=COSTS['strict'])
acc['A2_E3p_nav_identical'] = bool(np.allclose(n4e.to_numpy(), n5e.to_numpy(), rtol=0, atol=1e-6)); acc['A2_E3p_trades_identical'] = same_trades(l4e, l5e)
acc['A2_E3p_since2019'] = cg(n5e, 'since2019'); acc['A2_E3p_full'] = cg(n5e, 'full'); acc['A2_E3p_n_closed'] = int((l5e['why'] != 'open_end').sum())
assert acc['A2_F1_nav_identical'] and acc['A2_F1_trades_identical'] and acc['A2_E3p_nav_identical'] and acc['A2_E3p_trades_identical'], ('A2 失敗', acc)
del n4, l4, s4, n5, l5, n4e, l4e, n5e, l5e, S_e3; gc.collect()
print('驗收', {k: (round(v * 100, 3) if isinstance(v, float) and k != 'sim_seconds' else v) for k, v in acc.items()}, flush=True)

# ---------------------------------------------------------------- 格子
CELLS = {c['id']: dict(sat_w=c['sat_w'], kw=dict(depth=10, mom_lo=c['mom_lo'], exit='top20')) for c in PRE['grid']['cells']}
CELLS["CTRL_E3p_e50"] = dict(sat_w=1.0, kw=dict(depth=None, mom_lo=0.10, exit='top20'), control=True)
MAIN = 'W70_T10'

bw = {k: dict(cagr=stlib.cagr(bench, a, b), maxdd=stlib.maxdd(bench, a, b), vol=s5lib.ann_vol(bench, a, b), sharpe=stlib.sharpe(bench, a, b)) for k, (a, b) in WIN.items()}
res = {'PREREG_sha256': sha, 'acceptance': acc,
       'meta': dict(panel_T=P.T, panel_N=P.N, start=str(D0.date()), mid=str(DM.date()), since2019_from=str(D19.date()), since2024_from=str(D24.date()), end=str(DE.date())),
       'bench_0050': bw, 'bench_0050_yearly': {y: v['e0050'] for y, v in stlib.yearly(bench, bench).items()}, 'cells': {}, 'diagnostic': {}}
navs = {'0050': bench}; trades = []; keep = {}


def feats(S, lg):
    lg = lg.copy()
    k = lg['sig_t'].to_numpy(); j = lg['j'].to_numpy()
    lg['surge'] = S['rank'][k, j]
    lg['pos_ma'] = [S['pos_ma'].get(int(a), {}).get(int(b), np.nan) for a, b in zip(k, j)]
    lg['pre20'] = P.RET20[k, j]
    lg['close_sig'] = P.C[k, j]
    return lg


def metrics(nav, lg, st, nsd, full=True):
    w = W(nav)
    r = dict(windows=w, final_nav=float(nav.iloc[-1]))
    r['vol'] = {k: s5lib.ann_vol(nav, *WIN[k]) for k in ('full', 'since2019')}
    if not full:
        return r
    r['yearly'] = stlib.yearly(nav, bench)
    r['beta_alpha'] = {k: s5lib.beta_alpha(nav, bench, *WIN[k]) for k in ('full', 'since2019')}
    r['trades_full'] = stlib.trade_stats(lg); r['trades_since2019'] = stlib.trade_stats(lg, '2019-01-01')
    ex = s3lib.extra_stats(P, nav, lg, st, nsd)
    t1 = int(lg['ent_t'].min())
    sf = st['stockfrac'].loc[P.IDX[t1]:]
    ex['avg_stock_frac_from_first'] = float(sf.mean())
    ex['avg_stock_frac_since2019'] = float(st['stockfrac'].loc[D19:].mean())
    ex['avg_per_position_frac'] = float(sf.mean() / ex['avg_npos_from_first']) if ex['avg_npos_from_first'] > 0 else None
    ex['avg_base_frac_from_first'] = float((st['B'].loc[P.IDX[t1]:] / nav.loc[P.IDX[t1]:]).mean())
    for k in ('rb_up', 'rb_down', 'rb_trim_days', 'rb_trim_orders', 'rb_trim_val', 'rb_trim_short', 'b_clamped', 'blk_min', 'e50_cost', 'e50_orders'):
        ex[k] = float(st[k]) if isinstance(st[k], float) else int(st[k])
    r['extra'] = ex
    return r


for cid, c in CELLS.items():
    tc = time.time()
    S = s4lib.make_signal(P, U, RT, keep_info=True, **c['kw'])
    nsd = len(S['info']); ntg = np.array([x['n_tgt'] for x in S['info']])
    w_ = c['sat_w']
    R = dict(sat_w=w_, kw=c['kw'], control=bool(c.get('control', False)), runs={},
             list=dict(n_signal_days=nsd, months_lt10=int((ntg < 10).sum()), months_0=int((ntg == 0).sum()), mean_n=float(ntg.mean()), median_n=float(np.median(ntg))))
    for cn in ('strict', 'light', 'strict_x2'):
        nav, lg, st = s5lib.simulate_cs(P, S, U, sat_w=w_, cost=COSTS[cn])
        navs[f'{cid}|{cn}'] = nav
        if cn == 'strict':
            navs[f'{cid}|strict|B'] = st['B']; navs[f'{cid}|strict|stockfrac'] = st['stockfrac']
        R['runs'][cn] = metrics(nav, lg, st, nsd, full=(cn != 'strict_x2'))
        if cn != 'strict_x2':
            F = feats(S, lg); trades.append(F.assign(run=f'{cid}|{cn}'))
            if cn == 'strict':
                keep[cid] = (lg, F, nav)
                x = F[F['why'] != 'open_end']
                R['top_trades'] = x.sort_values('pnl', ascending=False).head(5)[['code', 'ent_date', 'exit_date', 'ret', 'pnl']].astype(str).to_dict('records')
                R['top5_pnl_share_of_closed'] = float(x.sort_values('pnl', ascending=False).head(5)['pnl'].sum() / x['pnl'].sum())
    lg, F, nav_main = keep[cid]
    blk, top = s2lib.block_from(lg, 5)
    nb, lb, sb = s5lib.simulate_cs(P, S, U, sat_w=w_, cost=COSTS['strict'], block=blk)
    navs[f'{cid}|strict|block5'] = nb
    R['block5'] = dict(windows=slim(W(nb)), n_dropped=int(sb['blk_dropped']),
                       top=[dict(code=x.code, ent=str(x.ent_date.date()), exit=str(x.exit_date.date()), ret=float(x.ret), pnl=float(x.pnl)) for x in top.itertuples()])
    R['shift'] = {'0': slim(R['runs']['strict']['windows'])}
    paths = {}
    for s in (1, 2, 3, 5):
        Ss = s4lib.make_signal(P, U, RT, shift=s, **c['kw'])
        if s in (1, 3, 5):
            ns, _, _ = s5lib.simulate_cs(P, Ss, U, sat_w=w_, cost=COSTS['strict'])
            navs[f'{cid}|strict|shift{s}'] = ns
            R['shift'][str(s)] = slim(W(ns))
        if s <= 3:
            for cap in CAPS:
                if cap == 1e6 and s in (1, 3):
                    paths[f'{int(cap/1e4)}萬|+{s}'] = R['shift'][str(s)]['since2019']['cagr']; continue
                npth, _, _ = s5lib.simulate_cs(P, Ss, U, sat_w=w_, cost=COSTS['strict'], capital=cap)
                paths[f'{int(cap/1e4)}萬|+{s}'] = cg(npth, 'since2019')
        del Ss; gc.collect()
    for cap in CAPS:
        if cap == 1e6:
            paths['100萬|+0'] = R['runs']['strict']['windows']['since2019']['cagr']; continue
        npth, _, _ = s5lib.simulate_cs(P, S, U, sat_w=w_, cost=COSTS['strict'], capital=cap)
        paths[f'{int(cap/1e4)}萬|+0'] = cg(npth, 'since2019')
    pv = np.array(list(paths.values()))
    R['paths20'] = dict(values=paths, min=float(pv.min()), median=float(np.median(pv)), max=float(pv.max()), n=int(len(pv)),
                        n_above_0050=int((pv > bw['since2019']['cagr']).sum()))
    if cid == MAIN:
        keep['S_main'] = S
    del S; gc.collect()
    # 分類
    w = R['runs']['strict']['windows']; x2 = R['runs']['strict_x2']['windows']; b5 = R['block5']['windows']
    chk = dict(net_since2019=w['since2019']['diff'] > 0, net_full=w['full']['diff'] > 0, costx2_since2019=x2['since2019']['diff'] > 0, costx2_full=x2['full']['diff'] > 0,
               block5_since2019=b5['since2019']['diff'] > 0, block5_full=b5['full']['diff'] > 0, half1=w['half1']['diff'] > 0, half2=w['half2']['diff'] > 0)
    R['checks'] = {k: bool(v) for k, v in chk.items()}
    R['class'] = '值得上XQ測' if all(chk.values()) else ('不值得' if not (chk['net_since2019'] or chk['net_full']) else '有條件')
    R['fails'] = [k for k, v in chk.items() if not v]
    # 回撤對報酬
    dd = {}
    for k in ('full', 'since2019'):
        ddx = (abs(w[k]['maxdd']) - abs(bw[k]['maxdd'])) * 100; rx = w[k]['diff'] * 100
        dd[k] = dict(maxdd=w[k]['maxdd'], maxdd_0050=bw[k]['maxdd'], deeper_pp=ddx, cagr_diff_pp=rx, dd_per_1pp=(ddx / rx) if rx > 0 else None)
    R['dd_vs_return'] = dd
    res['cells'][cid] = R
    ts = R['runs']['strict']['trades_full']; ba = R['runs']['strict']['beta_alpha']['full']
    print(f"{cid} w={w_} 2019起 {w['since2019']['cagr']*100:.2f}% 全期 {w['full']['cagr']*100:.2f}% 2024起 {w['since2024']['cagr']*100:.2f}% dd {w['full']['maxdd']*100:.1f}% "
          f"| x2 {x2['since2019']['cagr']*100:.2f}/{x2['full']['cagr']*100:.2f} | blk5 {b5['since2019']['cagr']*100:.2f}/{b5['full']['cagr']*100:.2f} | 半 {w['half1']['cagr']*100:.2f}/{w['half2']['cagr']*100:.2f} "
          f"| n {ts['n']} | beta {ba['beta']:.2f} alpha {ba['alpha_m']*100:.2f}% t {ba['alpha_t']:.2f} | 路徑 {pv.min()*100:.1f}/{np.median(pv)*100:.1f}/{pv.max()*100:.1f} "
          f"| {R['class']} {R['fails']} | {time.time()-tc:.0f}s {rss()}MB", flush=True)

# ---------------------------------------------------------------- 診斷（主格）
lg, F, nav_main = keep[MAIN]
d1, xx = s5lib.vs0050_trades(P, F)
res['diagnostic']['D1_vs_0050_same_period'] = d1
x = F[F['why'] != 'open_end']
res['diagnostic']['D2_price'] = {lab: s4lib.desc(x.loc[f, 'ret']) for lab, f in (('le75', x['close_sig'] <= 75), ('gt75', x['close_sig'] > 75))}
for lab, f in (('le75', x['close_sig'] <= 75), ('gt75', x['close_sig'] > 75)):
    res['diagnostic']['D2_price'][lab]['excess_vs_0050_mean'] = float(xx.loc[f.to_numpy(), 'excess'].mean())
print('診斷', json.dumps(res['diagnostic'], ensure_ascii=False)[:1500], flush=True)

res['meta']['runtime_s'] = round(time.time() - t_all, 1); res['meta']['maxrss_mb'] = rss()
(H / 'results.json').write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str))
pd.DataFrame(navs).round(4).to_csv(H / 'navs.csv.gz', index_label='date')
tr = pd.concat(trades, ignore_index=True)
tr = tr[['run', 'code', 'market', 'sig_date', 'ent_date', 'exit_date', 'hold_days', 'shares', 'cost', 'proceeds', 'pnl', 'ret', 'why', 'surge', 'pos_ma', 'pre20', 'close_sig']]
num = tr.select_dtypes('number').columns
tr[num] = tr[num].round(6)
tr.to_csv(H / 'trades.csv.gz', index=False)
print(res['meta']['runtime_s'], 's', res['meta']['maxrss_mb'], 'MB')
