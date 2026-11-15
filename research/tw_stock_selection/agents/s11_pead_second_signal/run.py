#!/usr/bin/env python3
"""S11：季報驚喜 Q（SUE⊥ 前 10 名、季換股）單獨衛星與和 SURGE 各半合併的衛星 C。
格子 E0、Q、QF、QR、C、CF、C0；每格 100 萬主跑、20 條路徑（5 種資金 × 位移 0～3）、成本加倍、禁買最賺 5 筆；事件層級 f。
python3 -I -B run.py → results.json、trades.csv.gz、navs.csv.gz。PREREG.json 的 sha256 必須相符。單行程。"""
import sys, json, hashlib, time, resource, gc
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
import s11lib
from s11lib import s10lib, s9lib, s4lib, s2lib, stlib, COSTS

H = Path(__file__).resolve().parent
pre = (H / 'PREREG.json').read_bytes()
sha = hashlib.sha256(pre).hexdigest()
assert sha == (H / 'PREREG.sha256').read_text().split()[0], 'PREREG.json 與 PREREG.sha256 不符'
PRE = json.loads(pre)
t_all = time.time()
rss = lambda: round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
P = s11lib.load()
assert (P.T, P.N, str(P.IDX[-1].date())) == (2876, 2142, '2026-10-22'), 'P0 失敗'
U0 = stlib.universe(P, 'twse')
RT = s2lib.rev_tables(P)
INNOV, _ = s4lib.innov_panel(P)
U = U0 & ~INNOV
del U0, INNOV
FT = s11lib.fin_tables()
print('載入完成', f'{time.time()-t_all:.0f}s', rss(), 'MB', flush=True)

# ---------------------------------------------------------------- f 事件層級（位移 0；先算，之後刪掉 AO）
def event_level():
    S = s11lib.make_signal_q(P, U, FT, shift=0, kind='perp')
    rows = [x['k'] for x in S['info']]
    out = []
    for i, x in enumerate(S['info']):
        k = x['k']; kq = x['kq']
        e0 = k + 1; e1 = min(rows[i + 1] + 1, P.T - 1) if i + 1 < len(rows) else P.T - 1
        if e0 >= P.T - 1:
            continue
        w, sc, _ = s11lib.q_scores(P, FT, U, kq, k, 'perp')
        r = P.AO[e1, w] / P.AO[e0, w] - 1
        ok = np.isfinite(r)
        w, r = w[ok], r[ok]
        s = sc[w]
        o = np.argsort(-s, kind='stable')
        top10 = r[o[:10]].mean(); q5 = r[o[:max(1, len(o) // 5)]].mean(); allm = r.mean()
        ic = pd.Series(s).rank().corr(pd.Series(r).rank())
        out.append(dict(fq=x['fq'], n=int(len(w)), top10_minus_all=float(top10 - allm), top20pct_minus_all=float(q5 - allm), ic=float(ic), all_mean=float(allm)))
    df = pd.DataFrame(out)
    tt = lambda v: float(v.mean() / (v.std(ddof=1) / np.sqrt(len(v))))
    summ = {c: dict(mean=float(df[c].mean()), t=tt(df[c]), n=int(len(df))) for c in ('top10_minus_all', 'top20pct_minus_all', 'ic')}
    for nm, m in (('2015_2018', df['fq'] < '2019'), ('2019_2023', (df['fq'] >= '2019') & (df['fq'] < '2024')), ('2024_', df['fq'] >= '2024')):
        summ[nm] = {c: dict(mean=float(df.loc[m, c].mean()), t=tt(df.loc[m, c]), n=int(m.sum())) for c in ('top10_minus_all', 'top20pct_minus_all', 'ic')}
    return dict(summary=summ, per_quarter=out)


EVT = event_level()
print('事件層級', json.dumps(EVT['summary'], ensure_ascii=False), flush=True)
for a in ('H', 'L', 'VOL', 'CHG', 'HF', 'VOLF', 'adjH', 'HI240', 'MA60', 'nbar', 'F_HI', 'IDXC', 'AO', 'listed', 'NEARHI', 'V60', 'liq'):
    if hasattr(P, a):
        delattr(P, a)
gc.collect()

bench = pd.Series(P.E50[P.t0:], index=P.IDX[P.t0:])
i_mid = (P.t0 + P.T - 1) // 2
D0, DM, DE = P.IDX[P.t0], P.IDX[i_mid], P.IDX[-1]
D19 = P.IDX[P.IDX < pd.Timestamp('2019-01-01')][-1]
D24 = P.IDX[P.IDX < pd.Timestamp('2024-01-01')][-1]
WIN = {'since2019': (D19, DE), 'full': (D0, DE), 'since2024': (D24, DE), 'y2019_2023': (D19, D24), 'half1': (D0, DM), 'half2': (DM, DE)}
W = lambda nav: {k: stlib.window(nav, bench, a, b) for k, (a, b) in WIN.items()}
cg = lambda nav, k: stlib.cagr(nav, *WIN[k])
mdd = lambda nav: stlib.maxdd(nav, D0, DE)
CAPS = (6e5, 8e5, 1e6, 1.5e6, 2e6)
bw = {k: dict(cagr=stlib.cagr(bench, a, b), maxdd=stlib.maxdd(bench, a, b)) for k, (a, b) in WIN.items()}
bw['seg_dd'] = s11lib.seg_dd(bench)
YRS = (P.IDX[-1] - P.IDX[P.t0]).days / 365.25
CELLS = ['E0', 'Q', 'QF', 'QR', 'C', 'CF', 'C0']


def sim(S, cap, cost='strict', block=None):
    return s10lib.simulate_s10(P, S, U, sat_w=0.7, sizing='W10', lock=False, cost=COSTS[cost], capital=cap, block=block)


def same_trades(l1, l2):
    a = l1[['code', 'ent_t', 'exit_t', 'why']].reset_index(drop=True); b = l2[['code', 'ent_t', 'exit_t', 'why']].reset_index(drop=True)
    return bool(len(a) == len(b) and (a == b).all().all())


def slim(w, keys=('since2019', 'full', 'since2024', 'y2019_2023', 'half1', 'half2')):
    return {k: dict(cagr=w[k]['cagr'], diff=w[k]['diff'], maxdd=w[k]['maxdd']) for k in keys}


def half0050(cap):
    return cap / 2 * bench / bench.iloc[0]


def top5_share(lg, nav, cap):
    x = lg[lg['why'] != 'open_end'].sort_values('pnl', ascending=False).head(5)
    return float(x['pnl'].sum() / (nav.iloc[-1] - cap))


# ---------------------------------------------------------------- 驗收
acc = {'P0': True}


def build(s):
    SE = s11lib.surge_signal(P, U, RT, shift=s)
    SQ = s11lib.make_signal_q(P, U, FT, shift=s, kind='perp')
    SQF = s11lib.make_signal_q(P, U, FT, shift=s, kind='perp', filt='F1')
    SQR = s11lib.make_signal_q(P, U, FT, shift=s, kind='raw')
    # A1
    bad = 0
    for S, kind in ((SQ, 'perp'), (SQF, 'perp'), (SQR, 'raw')):
        for x in S['info']:
            k = x['k']
            if not set(x['targets']) <= set(x['keep']):
                bad += 1
            if S is not SQF and x['targets'] != x['keep']:
                bad += 1
            w, _, _ = s11lib.q_scores(P, FT, U, x['kq'], k, kind)
            el = set(P.COLS[w])
            if not set(x['targets']) <= el:
                bad += 1
            if not P.IDX[k] > s11lib.deadline(x['kq'] // 4, x['kq'] % 4 + 1):
                bad += 1
            if set(P.COLS[np.flatnonzero(S['sig'][k])]) != set(x['targets']):
                bad += 1
    acc[f'A1_shift{s}_violations'] = bad
    assert bad == 0, ('A1 失敗', s, bad)
    return SE, SQ, SQF, SQR


SIG0 = build(0)
SE0 = SIG0[0]
n9, l9, _ = s9lib.simulate_s9(P, SE0, U, sat_w=0.7, sizing='W10', lock=False, cost=COSTS['strict'], exec_mode='open')
n10, l10, _ = sim(SE0, 1e6)
acc['A0'] = dict(since2019=cg(n10, 'since2019'), full=cg(n10, 'full'), maxdd=mdd(n10), trades_identical=same_trades(l9, l10),
                 ok=bool(abs(cg(n10, 'since2019') - 0.4813) < 0.00025 and abs(cg(n10, 'full') - 0.3408) < 0.00025 and abs(mdd(n10) + 0.3340) < 0.00025))
assert acc['A0']['ok'] and acc['A0']['trades_identical'], ('A0 失敗', acc)
del n9, l9, n10, l10
print('驗收 A0/A1', acc, flush=True)

# ---------------------------------------------------------------- 主迴圈
res = {'PREREG_sha256': sha, 'acceptance': acc,
       'meta': dict(panel_T=P.T, panel_N=P.N, start=str(D0.date()), mid=str(DM.date()), since2019_from=str(D19.date()), since2024_from=str(D24.date()), end=str(DE.date())),
       'bench_0050': bw, 'event_level': EVT, 'cells': {c: dict(paths={}) for c in CELLS}, 'q_info': {}}
navs = {'0050': bench}; trades = []
for s in range(4):
    tc = time.time()
    SE, SQ, SQF, SQR = SIG0 if s == 0 else build(s)
    if s == 0:
        res['q_info'] = {nm: [{k: v for k, v in x.items()} for x in S['info']] for nm, S in (('Q', SQ), ('QF', SQF), ('QR', SQR))}
    for cap in CAPS:
        key = f'{int(cap/1e4)}萬|+{s}'
        r = {}
        r['E0'] = sim(SE, cap); r['Q'] = sim(SQ, cap); r['QF'] = sim(SQF, cap); r['QR'] = sim(SQR, cap)
        eh = sim(SE, cap / 2); qh = sim(SQ, cap / 2); qfh = sim(SQF, cap / 2)
        nv = {c: r[c][0] for c in ('E0', 'Q', 'QF', 'QR')}
        nv['C'] = eh[0] + qh[0]; nv['CF'] = eh[0] + qfh[0]; nv['C0'] = eh[0] + half0050(cap)
        for c in CELLS:
            n = nv[c]
            res['cells'][c]['paths'][key] = dict(r19=cg(n, 'since2019'), full=cg(n, 'full'), r1923=cg(n, 'y2019_2023'), r24=cg(n, 'since2024'), maxdd=mdd(n),
                                                 seg=s11lib.seg_dd(n), corr_E0=s11lib.corr_excess(n, nv['E0'], bench) if c != 'E0' else 1.0)
        if s == 0 and cap == 1e6:
            assert np.allclose(nv['C'].to_numpy(), (eh[0] + qh[0]).to_numpy(), atol=1.0)
            acc['A2'] = True
            lgs = {'E0': r['E0'][1], 'Q': r['Q'][1], 'QF': r['QF'][1], 'QR': r['QR'][1],
                   'C': pd.concat([eh[1].assign(sleeve='E0'), qh[1].assign(sleeve='Q')], ignore_index=True),
                   'CF': pd.concat([eh[1].assign(sleeve='E0'), qfh[1].assign(sleeve='QF')], ignore_index=True),
                   'C0': eh[1].assign(sleeve='E0')}
            sts = {'E0': [r['E0'][2]], 'Q': [r['Q'][2]], 'QF': [r['QF'][2]], 'QR': [r['QR'][2]], 'C': [eh[2], qh[2]], 'CF': [eh[2], qfh[2]], 'C0': [eh[2]]}
            sigS = {'E0': [SE], 'Q': [SQ], 'QF': [SQF], 'QR': [SQR], 'C': [SE, SQ], 'CF': [SE, SQF], 'C0': [SE]}
            for c in CELLS:
                R = res['cells'][c]; n = nv[c]; lg = lgs[c]
                navs[c] = n
                w = W(n)
                R['windows'] = slim(w); R['sharpe'] = {k: w[k]['sharpe'] for k in ('full', 'since2019')}
                R['t_month_excess'] = {k: w[k]['t_month_excess'] for k in ('full', 'since2019')}
                R['dd_dates'] = s10lib.dd_dates(n); R['seg_dd'] = s11lib.seg_dd(n); R['yearly'] = stlib.yearly(n, bench)
                R['trades_full'] = stlib.trade_stats(lg); R['trades_since2019'] = stlib.trade_stats(lg, '2019-01-01')
                R['top5_share_of_gain'] = top5_share(lg, n, 1e6)
                mnav = float(n.mean())
                bv = sum(x['buy_val'] for x in sts[c]); sv = sum(x['sell_val'] for x in sts[c])
                R['turnover_oneway_ann'] = float((bv + sv) / 2 / mnav / YRS)
                R['cost_ann_pct_of_nav'] = float(sum(x['fee'] + x['tax'] + x['slip'] + x['e50_cost'] for x in sts[c]) / mnav / YRS)
                R['n_buy'] = int(sum(x['n_buy'] for x in sts[c])); R['blk_min'] = int(sum(x['blk_min'] for x in sts[c]))
                R['blk_lot'] = int(sum(x['blk_lot'] for x in sts[c])); R['blk_limup'] = int(sum(x['blk_limup'] for x in sts[c]))
                R['avg_npos'] = float(sum(x['avg_npos'] for x in sts[c]))
                sleeve_navs = {'C': [eh[0], qh[0]], 'CF': [eh[0], qfh[0]], 'C0': [eh[0]]}.get(c, [n])
                sf = sum(x['stockfrac'] * sn for x, sn in zip(sts[c], sleeve_navs)) / n
                R['avg_stockfrac'] = float(np.nanmean(sf.to_numpy()))
                x = lg.copy(); x['run'] = c
                if 'sleeve' not in x:
                    x['sleeve'] = c
                trades.append(x[['run', 'sleeve', 'code', 'sig_date', 'ent_date', 'exit_date', 'hold_days', 'shares', 'cost', 'proceeds', 'pnl', 'ret', 'why', 'score']])
                # 成本加倍、禁買最賺 5 筆
                blk, top = s2lib.block_from(lg, 5)
                if c in ('E0', 'Q', 'QF', 'QR'):
                    nx2 = sim(sigS[c][0], 1e6, cost='strict_x2')[0]; nb = sim(sigS[c][0], 1e6, block=blk)[0]
                elif c == 'C0':
                    nx2 = sim(SE, 5e5, cost='strict_x2')[0] + half0050(1e6); nb = sim(SE, 5e5, block=blk)[0] + half0050(1e6)
                else:
                    nx2 = sim(sigS[c][0], 5e5, cost='strict_x2')[0] + sim(sigS[c][1], 5e5, cost='strict_x2')[0]
                    nb = sim(sigS[c][0], 5e5, block=blk)[0] + sim(sigS[c][1], 5e5, block=blk)[0]
                R['costx2'] = slim(W(nx2), ('since2019', 'full'))
                R['block5'] = slim(W(nb), ('since2019', 'full'))
                R['block5_top'] = [dict(code=y.code, sleeve=getattr(y, 'sleeve', c), ent=str(y.ent_date.date()), exit=str(y.exit_date.date()), ret=float(y.ret), pnl=float(y.pnl)) for y in top.itertuples()]
                del nx2, nb
            res['sleeve_weight_C_E0_end'] = float(eh[0].iloc[-1] / nv['C'].iloc[-1])
            res['sleeve_weight_C_E0_series_yearend'] = {int(y): float(v) for y, v in (eh[0] / nv['C']).groupby(nv['C'].index.year).last().items()}
        del r, eh, qh, qfh, nv
    print('位移', s, f'{time.time()-tc:.0f}s 累計 {time.time()-t_all:.0f}s {rss()}MB', flush=True)
    del SE, SQ, SQF, SQR
    if s == 0:
        del SIG0, SE0
    gc.collect()

# ---------------------------------------------------------------- 彙整
b19 = bw['since2019']['cagr']
for c in CELLS:
    R = res['cells'][c]; pv = R['paths']
    a = np.array([v['r19'] for v in pv.values()]); d = np.array([v['maxdd'] for v in pv.values()])
    assert len(a) == 20
    R['paths20'] = dict(min=float(a.min()), median=float(np.median(a)), max=float(a.max()), n_above_0050=int((a > b19).sum()),
                        maxdd_min=float(d.min()), maxdd_median=float(np.median(d)), maxdd_max=float(d.max()),
                        r1923_median=float(np.median([v['r1923'] for v in pv.values()])), r24_median=float(np.median([v['r24'] for v in pv.values()])),
                        full_median=float(np.median([v['full'] for v in pv.values()])),
                        seg_median={k: float(np.median([v['seg'][k] for v in pv.values()])) for k in s11lib.SEGMENTS},
                        corr_E0_median=float(np.median([v['corr_E0'] for v in pv.values()])))
    w = R['windows']; x2 = R['costx2']; b5 = R['block5']
    chk = dict(net_since2019=w['since2019']['diff'] > 0, net_full=w['full']['diff'] > 0, costx2_since2019=x2['since2019']['diff'] > 0, costx2_full=x2['full']['diff'] > 0,
               block5_since2019=b5['since2019']['diff'] > 0, block5_full=b5['full']['diff'] > 0, half1=w['half1']['diff'] > 0, half2=w['half2']['diff'] > 0)
    R['checks'] = {k: bool(v) for k, v in chk.items()}
    R['class'] = '值得上XQ測' if all(chk.values()) else ('不值得' if not (chk['net_since2019'] or chk['net_full']) else '有條件')
    R['fails'] = [k for k, v in chk.items() if not v]
    R['block5_loss_since2019_pp'] = (w['since2019']['cagr'] - b5['since2019']['cagr']) * 100

keys = list(res['cells']['E0']['paths'].keys())


def paired(a, b):
    A, B = res['cells'][a]['paths'], res['cells'][b]['paths']
    dr = np.array([A[k]['r19'] - B[k]['r19'] for k in keys]); dd = np.array([A[k]['maxdd'] - B[k]['maxdd'] for k in keys])
    seg = {}
    for sg in s11lib.SEGMENTS:
        x = np.array([A[k]['seg'][sg] - B[k]['seg'][sg] for k in keys])
        seg[sg] = dict(n_shallower=int((x > 0).sum()), median_pp=float(np.median(x)) * 100, pass_=bool((x > 0).sum() >= 15 and np.median(x) * 100 >= 1.0))
    d1923 = np.array([A[k]['r1923'] - B[k]['r1923'] for k in keys]); d24 = np.array([A[k]['r24'] - B[k]['r24'] for k in keys])
    return dict(ret_npos=int((dr > 0).sum()), ret_median_pp=float(np.median(dr)) * 100, ret_min_pp=float(dr.min()) * 100, ret_max_pp=float(dr.max()) * 100,
                dd_n_shallower=int((dd > 0).sum()), dd_median_pp=float(np.median(dd)) * 100, seg=seg,
                r1923_npos=int((d1923 > 0).sum()), r1923_median_pp=float(np.median(d1923)) * 100, r24_npos=int((d24 > 0).sum()), r24_median_pp=float(np.median(d24)) * 100)


pairs = {f'{a}-{b}': paired(a, b) for a, b in (('C', 'E0'), ('CF', 'E0'), ('C0', 'E0'), ('C', 'C0'), ('CF', 'C0'), ('Q', 'E0'), ('QF', 'E0'), ('QR', 'Q'), ('QF', 'Q'))}
res['pairs'] = pairs


def corr_cls(x):
    return '低相關' if x < 0.3 else ('中度相關' if x < 0.6 else '高相關')


navE, navQ, navQF = navs['E0'], navs['Q'], navs['QF']
corr = dict(Q_E0_full=s11lib.corr_excess(navQ, navE, bench, D0, DE), Q_E0_since2019=s11lib.corr_excess(navQ, navE, bench, D19, DE),
            QF_E0_full=s11lib.corr_excess(navQF, navE, bench, D0, DE), QR_E0_full=s11lib.corr_excess(navs['QR'], navE, bench, D0, DE),
            Q_E0_paths_median=res['cells']['Q']['paths20']['corr_E0_median'], QF_E0_paths_median=res['cells']['QF']['paths20']['corr_E0_median'])
mr = lambda n: n.groupby(n.index.to_period('M')).last().pct_change().dropna()
corr['Q_E0_raw_monthly_full'] = float(pd.concat([mr(navQ), mr(navE)], axis=1).dropna().corr().iloc[0, 1])
corr['Q_E0_full_from2015-12'] = s11lib.corr_excess(navQ, navE, bench, pd.Timestamp('2015-12-01'), DE)
res['correlation'] = corr


def beat0050(c):
    p = res['cells'][c]['paths20']; w = res['cells'][c]['windows']
    if p['median'] > b19 and p['n_above_0050'] >= 15 and w['full']['diff'] > 0:
        return '贏 0050'
    if p['median'] < b19 and p['n_above_0050'] <= 5:
        return '輸 0050'
    return '不確定'


def diversify(cc):
    pr = pairs[f'{cc}-E0']; att = pairs[f'{cc}-C0']
    c_ok = pr['dd_n_shallower'] >= 15 and pr['ret_median_pp'] >= -2
    nseg = sum(1 for v in pr['seg'].values() if v['pass_'])
    d_ok = nseg >= 3
    q_ok = att['ret_npos'] >= 15
    if c_ok and d_ok:
        v = '有分散效果' if q_ok else '回撤變淺，但來自少押 SURGE，不是 Q'
    elif c_ok or d_ok:
        v = '部分'
    else:
        v = '沒有分散效果'
    return dict(verdict=v, c_pass=bool(c_ok), d_pass=bool(d_ok), d_segments_passed=nseg, attribution_Q_contributes=bool(q_ok))


def winners(cc):
    a, b = res['cells'][cc], res['cells']['E0']
    ok = a['top5_share_of_gain'] < b['top5_share_of_gain'] and a['block5_loss_since2019_pp'] < b['block5_loss_since2019_pp']
    return '較不依賴大贏家' if ok else '沒有降低'


dv = diversify('C')
q_vs = beat0050('Q')
xq_ok = dv['verdict'] == '有分散效果' or (q_vs == '贏 0050' and corr_cls(corr['Q_E0_full']) != '高相關')
res['verdict'] = dict(correlation_class=corr_cls(corr['Q_E0_full']), Q_vs_0050=q_vs, QF_vs_0050=beat0050('QF'), QR_vs_0050=beat0050('QR'),
                      C_diversify=dv, CF_diversify=diversify('CF'), C_winners=winners('C'), CF_winners=winners('CF'),
                      xq_recommend='可以加' if xq_ok else '不建議加進現行策略，只做紙上追蹤／XQ 交叉驗證',
                      classes={c: res['cells'][c]['class'] for c in CELLS}, fails={c: res['cells'][c]['fails'] for c in CELLS})
print('相關', json.dumps(corr, ensure_ascii=False), flush=True)
print('判定', json.dumps(res['verdict'], ensure_ascii=False), flush=True)
for c in CELLS:
    R = res['cells'][c]
    print(c, {k: round(v, 4) if isinstance(v, float) else v for k, v in R['paths20'].items() if k != 'seg_median'}, R['class'], flush=True)
for k, v in pairs.items():
    print(k, {a: b for a, b in v.items() if a != 'seg'}, {s: (x['n_shallower'], round(x['median_pp'], 2)) for s, x in v['seg'].items()}, flush=True)

res['meta']['runtime_s'] = round(time.time() - t_all, 1); res['meta']['maxrss_mb'] = rss()
res['meta']['N'] = PRE['data_snooping']
(H / 'results.json').write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str))
pd.DataFrame(navs).round(0).to_csv(H / 'navs.csv.gz', index_label='date')
tr = pd.concat(trades, ignore_index=True)
num = tr.select_dtypes('number').columns
tr[num] = tr[num].round(4)
for c in ('sig_date', 'ent_date', 'exit_date'):
    tr[c] = pd.to_datetime(tr[c]).dt.strftime('%Y-%m-%d')
tr.to_csv(H / 'trades.csv.gz', index=False)
print(res['meta']['runtime_s'], 's', res['meta']['maxrss_mb'], 'MB')
