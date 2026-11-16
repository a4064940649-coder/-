#!/usr/bin/env python3
"""S37 主程式：依 SPEC.json 跑（1）進出場日開盤狀況描述、（2）成交價敏感度、（3）容量。輸出 results.json。
python3 -I -B run.py"""
import sys, json, hashlib, time, resource, gc
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
import s37lib as L
from s37lib import s17lib, s10lib, s4lib, stlib, cslib, COSTS

H = Path(__file__).resolve().parent
t_all = time.time()
rss = lambda: round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
spec_sha = hashlib.sha256((H / 'SPEC.json').read_bytes()).hexdigest()

P = L.load()
assert P.exright_fix
INNOV, _ = s4lib.innov_panel(P)
U = stlib.universe(P, 'twse') & ~INNOV
del INNOV
F = s17lib.rev_family(P, defs=['STD']); RT = dict(S=F['S']['STD'], av=F['av']); del F
s17lib.e50_fix_suspension(P)
L.add_avg(P)
VOLF = np.where(np.isfinite(P.VOL), P.VOL, 0.0)
VOL20 = pd.DataFrame(VOLF).rolling(20, min_periods=10).mean().shift(1).to_numpy()
for a in ('HF', 'VOLF', 'adjH', 'HI240', 'MA60', 'nbar', 'F_HI', 'IDXC', 'AO', 'listed', 'NEARHI', 'V60', 'liq', 'CHG'):
    if hasattr(P, a):
        delattr(P, a)
gc.collect()
print('載入完成', f'{time.time()-t_all:.0f}s', rss(), 'MB', flush=True)

lastb = lambda d: P.IDX[P.IDX < pd.Timestamp(d)][-1]
D0, DE, D19 = P.IDX[P.t0], P.IDX[-1], lastb('2019-01-01')
CELL = dict(sat_w=0.7, sizing='W10', lock=False)
CAPS = (6e5, 8e5, 1e6, 1.5e6, 2e6)
SIG = {s: s10lib.make_signal_s10(P, U, RT, shift=s) for s in range(4)}
SV = {s: s10lib.variant_S(SIG[s], 'E0') for s in range(4)}


def run(s, cap=1e6, fill='open', cap_pct=0.05):
    S2, kw = SV[s]
    return L.simulate_x(P, S2, U, cost=COSTS['strict'], capital=cap, cap_pct=cap_pct, fill=fill, **CELL, **kw)


def summ(nav, lg):
    return dict(since2019=stlib.cagr(nav, D19, DE), full=stlib.cagr(nav, D0, DE), maxdd=stlib.maxdd(nav, D0, DE), n_closed=int((lg['why'] != 'open_end').sum()))


res = dict(SPEC_sha256=spec_sha, meta=dict(start=str(D0.date()), since2019_from=str(D19.date()), end=str(DE.date()), avgv_fallback_cells=P.avgv_fallback))

# ---------------------------------------------------------------- 驗收
S2, kw = SV[0]
nav_ref, lg_ref, _ = s10lib.simulate_s10(P, S2, U, cost=COSTS['strict'], capital=1e6, **CELL, **kw)
acc_dev = 0.0
BASE = {}
for s in range(4):
    for cap in CAPS:
        nav, lg, st = run(s, cap)
        if s == 0 and cap == 1e6:
            acc_dev = max(acc_dev, float(np.abs(nav - nav_ref).max()))
        else:
            S2, kw = SV[s]
            nr = s10lib.simulate_s10(P, S2, U, cost=COSTS['strict'], capital=cap, **CELL, **kw)[0]
            acc_dev = max(acc_dev, float(np.abs(nav - nr).max()))
        BASE[(s, cap)] = (nav, lg, st)
std = summ(*BASE[(0, 1e6)][:2])
ok = (round(std['since2019'] * 100, 2), round(std['full'] * 100, 2), round(std['maxdd'] * 100, 2), std['n_closed']) == (48.99, 34.54, -32.11, 381)
res['acceptance'] = dict(standard=std, match_48_99=ok, A1_max_abs_nav_diff_vs_simulate_s10=acc_dev)
assert ok and acc_dev < 1e-6, res['acceptance']
print('驗收通過', std, acc_dev, flush=True)

# ---------------------------------------------------------------- 第 1 部分：進出場日開盤狀況
O, Hh, Ll, C, REF, AMT, LU, LD, AV, A4, G = P.O, P.H, P.L, P.C, P.REF, P.AMT, P.LIMUP, P.LIMDN, P.AVGV, P.AVG4, P.G
ab = pd.concat([st['att_b'] for (_, _, st) in BASE.values()]).drop_duplicates(['t', 'j'])
ab = ab[~ab['res'].isin(['noslot', 'banned'])].reset_index(drop=True)
asl = pd.concat([st['att_s'] for (_, _, st) in BASE.values()]).drop_duplicates(['t', 'j']).reset_index(drop=True)
ab_std = BASE[(0, 1e6)][2]['att_b']; ab_std = ab_std[~ab_std['res'].isin(['noslot', 'banned'])]
qs = (5, 25, 50, 75, 95)


def dist(x):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    if not len(x):
        return dict(n=0)
    return dict(n=int(len(x)), mean=float(x.mean()), **{f'p{q}': float(np.percentile(x, q)) for q in qs}, gt0=float((x > 0).mean()))


def day_stats(df, side):
    t = df['t'].to_numpy(); j = df['j'].to_numpy()
    o, h, l, c, ref, lu, ld = O[t, j], Hh[t, j], Ll[t, j], C[t, j], REF[t, j], LU[t, j], LD[t, j]
    has = np.isfinite(o) & (o > 0)
    o_, h_, l_, c_, ref_, lu_, ld_ = o[has], h[has], l[has], c[has], ref[has], lu[has], ld[has]
    gap = o_ / ref_ - 1
    out = dict(n=int(len(df)), n_no_trade=int((~has).sum()), gap=dist(gap),
               gap_gt3=float((gap > 0.03).mean()), gap_gt5=float((gap > 0.05).mean()), gap_gt7=float((gap > 0.07).mean()), gap_lt_m3=float((gap < -0.03).mean()),
               avgv_vs_open=dist(AV[t, j][has] / o_ - 1), avg4_vs_open=dist(A4[t, j][has] / o_ - 1),
               high_vs_open=dist(h_ / o_ - 1), low_vs_open=dist(l_ / o_ - 1), close_vs_open=dist(c_ / o_ - 1))
    vr = VOLF[t, j][has] / VOL20[t, j][has]
    if side == 'B':
        lim_open = o_ >= lu_ - 1e-9; one = l_ >= lu_ - 1e-9
        out.update(open_limup=int(lim_open.sum()), open_limup_rate=float(lim_open.mean()), one_line_limup=int(one.sum()),
                   open_limup_then_opened=int((lim_open & ~one).sum()),
                   limup_vol_ratio_median=float(np.nanmedian(vr[lim_open])) if lim_open.any() else None,
                   limup_vol_ratio_lt05=float(np.nanmean(vr[lim_open] < 0.5)) if lim_open.any() else None,
                   all_vol_ratio_median=float(np.nanmedian(vr)))
    else:
        lim_open = o_ <= ld_ + 1e-9; one = h_ <= ld_ + 1e-9
        out.update(open_limdn=int(lim_open.sum()), open_limdn_rate=float(lim_open.mean()), one_line_limdn=int(one.sum()),
                   open_limdn_then_opened=int((lim_open & ~one).sum()))
    return out


def yearly_b(df):
    t = df['t'].to_numpy(); j = df['j'].to_numpy(); o = O[t, j]; has = np.isfinite(o) & (o > 0)
    y = P.IDX[t].year
    d = pd.DataFrame(dict(y=y[has], lim=(o >= LU[t, j] - 1e-9)[has], gap=(o / REF[t, j] - 1)[has], av=(AV[t, j] / o - 1)[has]))
    return {int(k): dict(n=int(len(g)), open_limup=int(g['lim'].sum()), gap_median=float(g['gap'].median()), avgv_vs_open_median=float(g['av'].median()))
            for k, g in d.groupby('y')}


def fwd(t, j, start_open=True, h=20):
    """第 t 日開盤（或收盤）進場、持有到 t＋h 收盤的含息報酬（G 指數；t+h 超出面板用最後一日）。"""
    t2 = np.minimum(t + h, P.T - 1)
    with np.errstate(invalid='ignore', divide='ignore'):
        return G[t2, j] / G[t, j] * (C[t, j] / O[t, j] if start_open else 1.0) - 1


def limit_sim(df):
    t = df['t'].to_numpy(); j = df['j'].to_numpy(); o = O[t, j]
    keep = np.isfinite(o) & (o > 0); t, j, o = t[keep], j[keep], o[keep]
    ref, l, lu = REF[t, j], Ll[t, j], LU[t, j]
    r20 = fwd(t, j)
    out = {}
    for x in (0.0, 0.01, 0.02, 0.03, 0.05, 0.07, 'limup'):
        if x == 'limup':
            lim = lu
        else:
            tk = cslib.tick(ref * (1 + x)); lim = np.minimum(np.floor(ref * (1 + x) / tk + 1e-9) * tk, lu)
        at_open = o <= lim + 1e-9
        queue = at_open & (o >= lu - 1e-9)                  # 開盤就在漲停價：排隊，不保證成交
        intra = ~at_open & (l <= lim + 1e-9)
        filled = (at_open & ~queue) | intra
        px = np.where(at_open, o, lim)
        out[str(x)] = dict(fill_rate=float(filled.mean()), queue_at_limup=int(queue.sum()), miss=int((~filled & ~queue).sum()),
                           fill_vs_open_mean=float((px[filled] / o[filled] - 1).mean()), fill_vs_ref_mean=float((px[filled] / ref[filled] - 1).mean()),
                           fwd20_from_open_filled_median=float(np.nanmedian(r20[filled])),
                           fwd20_from_open_missed_median=float(np.nanmedian(r20[~filled & ~queue])) if (~filled & ~queue).any() else None)
    return out


def limup_follow(df):
    t = df['t'].to_numpy(); j = df['j'].to_numpy(); o = O[t, j]
    m = np.isfinite(o) & (o >= LU[t, j] - 1e-9) & (t + 1 < P.T)
    t, j = t[m], j[m]
    rows = []
    for a, b in zip(t, j):
        o1 = O[a + 1, b]
        rows.append(dict(date=str(P.IDX[a].date()), code=str(P.COLS[b]), one_line=bool(Ll[a, b] >= LU[a, b] - 1e-9),
                         next_open_limup=bool(np.isfinite(o1) and o1 >= LU[a + 1, b] - 1e-9),
                         next_open_vs_open=float(o1 / O[a, b] - 1) if np.isfinite(o1) else None,
                         fwd20_from_next_open=float(fwd(np.array([a + 1]), np.array([b]))[0]) if np.isfinite(o1) else None,
                         fwd20_from_open_hypo=float(fwd(np.array([a]), np.array([b]))[0])))
    return rows


def defer_days(BASEd):
    n_ex = 0; n_def = 0; dl = []
    for (_, _, st) in BASEd.values():
        a = st['att_s']
        for j, g in a.groupby('j', sort=False):
            k = 0
            for r in g['res']:
                if r in ('limdn', 'noopen'):
                    k += 1
                else:
                    n_ex += 1
                    if k:
                        n_def += 1; dl.append(k)
                    k = 0
    return dict(n_exit_events=n_ex, n_deferred=n_def, defer_days={str(k): int(v) for k, v in pd.Series(dl).value_counts().sort_index().items()} if dl else {})


asl_ok = asl[asl['res'].isin(['sold', 'partial', 'limdn', 'noopen'])]
res['part1'] = dict(
    entry_all_paths=day_stats(ab, 'B'), entry_standard=day_stats(ab_std, 'B'),
    entry_yearly=yearly_b(ab), entry_limit_order_sim=limit_sim(ab), entry_limup_followup=limup_follow(ab),
    exit_all_paths=day_stats(asl_ok, 'S'), exit_standard=day_stats(BASE[(0, 1e6)][2]['att_s'], 'S'), exit_defer=defer_days(BASE),
    entry_result_counts={k: int(v) for k, v in ab['res'].value_counts().items()}, exit_result_counts={k: int(v) for k, v in asl['res'].value_counts().items()})
# 執行日的全市場對照（同一批執行日、股池內全部股票）：開盤漲停比例
ed = sorted(set(int(x['k']) + 1 for s in range(4) for x in SIG[s]['info'] if int(x['k']) + 1 < P.T))
mk = []
for t in ed:
    u = U[t - 1] & np.isfinite(O[t]) & (O[t] > 0)
    mk.append(((O[t, u] >= LU[t, u] - 1e-9).mean(), np.median(O[t, u] / REF[t, u] - 1)))
res['part1']['universe_exec_days'] = dict(n_days=len(ed), open_limup_rate_mean=float(np.mean([a for a, _ in mk])), gap_median_mean=float(np.mean([b for _, b in mk])))
print('第 1 部分完成', f'{time.time()-t_all:.0f}s', rss(), 'MB', flush=True)

# ---------------------------------------------------------------- 第 2 部分：成交價敏感度
VARS = {'BASE': 'open', 'AVGV': 'avgv', 'AVG4': 'avg4', 'OPEN05': 'open05', 'DEFER': 'defer'}
P2 = {}
for nm, fl in VARS.items():
    D = {}
    for s in range(4):
        for cap in CAPS:
            if nm == 'BASE':
                nav, lg, st = BASE[(s, cap)]
            else:
                nav, lg, st = run(s, cap, fl)
            r = summ(nav, lg)
            r.update(blk_limup=int(st['blk_limup']), sell_defer_limdn=int(st['sell_defer_limdn']), n_buy=int(st['n_buy']),
                     n_defer_try=int(st['n_defer_try']), n_defer_buy=int(st['n_defer_buy']),
                     buy_val=float(st['buy_val']), slip=float(st['slip']))
            D[f'{int(cap/1e4)}萬|+{s}'] = r
    P2[nm] = D
    print('變體', nm, f'{time.time()-t_all:.0f}s', flush=True)
keys = list(P2['BASE'])
STDK = '100萬|+0'
p2s = {}
for nm, D in P2.items():
    o = dict(standard=D[STDK], paths_median={f: float(np.median([D[k][f] for k in keys])) for f in ('since2019', 'full', 'maxdd', 'n_closed', 'blk_limup', 'n_defer_buy')})
    if nm != 'BASE':
        dd = {}
        for f in ('since2019', 'full', 'maxdd'):
            d = np.array([D[k][f] - P2['BASE'][k][f] for k in keys]) * 100
            dd[f] = dict(standard_pp=float((D[STDK][f] - P2['BASE'][STDK][f]) * 100), median_pp=float(np.median(d)), min_pp=float(d.min()), max_pp=float(d.max()),
                         n_worse=int((d < 0).sum()))
        o['vs_BASE'] = dd
    p2s[nm] = o
res['part2'] = dict(summary=p2s, paths=P2)
# 現行（BASE）標準路徑＋20 條路徑的全部交易（進場、出場日、股票、進出場日開盤）
tr = []
for (s, cap), (nav, lg, st) in BASE.items():
    x = lg[['code', 'sig_date', 'ent_date', 'exit_date', 'shares', 'ret', 'why']].copy()
    x.insert(0, 'path', f'{int(cap/1e4)}萬|+{s}')
    x['ent_open'] = O[lg['ent_t'], lg['j']]; x['exit_open'] = np.where(lg['why'] == 'open_end', np.nan, O[lg['exit_t'], lg['j']])
    tr.append(x)
tr = pd.concat(tr, ignore_index=True)
for c in ('sig_date', 'ent_date', 'exit_date'):
    tr[c] = pd.to_datetime(tr[c]).dt.strftime('%Y-%m-%d')
tr['ret'] = tr['ret'].round(4)
tr.to_csv(H / 'trades_base.csv.gz', index=False)
res['meta']['n_trades_base_all_paths'] = int(len(tr))
del BASE
gc.collect()
print('第 2 部分完成', f'{time.time()-t_all:.0f}s', rss(), 'MB', flush=True)

# ---------------------------------------------------------------- 第 3 部分：容量
CAPITAL = (1e6, 5e6, 1e7, 3e7)
CP = (0.0, 0.01, 0.02, 0.05)
P3 = {}
for K in CAPITAL:
    for cp in CP:
        for s in range(4):
            nav, lg, st = run(s, K, 'open', cp)
            a = st['att_b']; sized = a[a['res'].isin(['bought', 'cap0', 'nocash', 'min'])]
            capped = sized['capped'].to_numpy(bool)
            dropped = capped & ~sized['res'].eq('bought').to_numpy()
            ss = st['att_s']; nsold = int(ss['res'].isin(['sold', 'partial']).sum())
            x = lg[lg['why'] != 'open_end']
            part = (lg['shares'] * O[lg['ent_t'], lg['j']] / AMT[lg['ent_t'], lg['j']]).to_numpy()
            r = summ(nav, lg)
            r.update(n_sized=int(len(sized)), n_capped=int(capped.sum()), capped_rate=float(capped.mean()) if len(sized) else 0.0,
                     n_dropped=int(dropped.sum()), dropped_rate=float(dropped.mean()) if len(sized) else 0.0,
                     sell_partial=int(st['sell_partial_cap']), sell_partial_rate=float(st['sell_partial_cap'] / max(nsold, 1)),
                     n_buy=int(st['n_buy']), participation=dist(part))
            P3[f'{int(K/1e4)}萬|{cp:.2f}|+{s}'] = r
    print('容量', K, f'{time.time()-t_all:.0f}s', flush=True)
p3s = {}
for K in CAPITAL:
    for cp in CP:
        ks = [f'{int(K/1e4)}萬|{cp:.2f}|+{s}' for s in range(4)]
        k0 = [f'{int(K/1e4)}萬|0.00|+{s}' for s in range(4)]
        d = np.array([P3[a]['since2019'] - P3[b]['since2019'] for a, b in zip(ks, k0)]) * 100
        p3s[f'{int(K/1e4)}萬|{cp:.2f}'] = dict(
            shift0=dict((f, P3[ks[0]][f]) for f in ('since2019', 'maxdd', 'capped_rate', 'dropped_rate', 'sell_partial_rate', 'n_buy')),
            shift0_vs_nocap_pp=float(d[0]), median4_vs_nocap_pp=float(np.median(d)),
            median4={f: float(np.median([P3[a][f] for a in ks])) for f in ('since2019', 'maxdd', 'capped_rate', 'dropped_rate', 'sell_partial_rate')},
            participation_shift0=P3[ks[0]]['participation'])
res['part3'] = dict(summary=p3s, runs=P3)
res['meta'].update(runtime_s=round(time.time() - t_all, 1), maxrss_mb=rss())
(H / 'results.json').write_text(json.dumps(res, ensure_ascii=False, indent=1, default=lambda o: o.item() if hasattr(o, 'item') else str(o)))
print('完成', res['meta'], flush=True)
