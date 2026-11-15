#!/usr/bin/env python3
"""S14：處置事件衛星 D（上市、第一次處置，起日開盤買、處置最後一天隔天開盤賣）單獨，與和 SURGE 各半合併 C。
格子 E0、D、DA、C、CA、C0；每格 100 萬主跑、20 條路徑（5 種資金 × 位移 0～3）、成本加倍、禁買最賺 5 筆。
python3 -I -B run.py → results.json、trades.csv.gz、navs.csv.gz。PREREG.json 的 sha256 必須相符。單行程。"""
import sys, json, hashlib, time, resource, gc
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
import s14lib
from s14lib import s10lib, s9lib, s4lib, s2lib, stlib, COSTS

H = Path(__file__).resolve().parent
pre = (H / 'PREREG.json').read_bytes()
sha = hashlib.sha256(pre).hexdigest()
assert sha == (H / 'PREREG.sha256').read_text().split()[0], 'PREREG.json 與 PREREG.sha256 不符'
PRE = json.loads(pre)
t_all = time.time()
rss = lambda: round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
P = s14lib.load()
assert (P.T, P.N, str(P.IDX[-1].date())) == (2876, 2142, '2026-10-22'), 'P0 失敗'
U0 = stlib.universe(P, 'twse')
RT = s2lib.rev_tables(P)
INNOV, _ = s4lib.innov_panel(P)
U = U0 & ~INNOV
del U0, INNOV
DT = s14lib.disp_table()
CH = {'D': s14lib.disp_chains(P, DT, 'first'), 'DA': s14lib.disp_chains(P, DT, 'all')}
ADV20 = pd.DataFrame(np.nan_to_num(P.AMT)).rolling(20, min_periods=20).mean().to_numpy()
for a in ('H', 'L', 'VOL', 'CHG', 'HF', 'VOLF', 'adjH', 'HI240', 'MA60', 'nbar', 'F_HI', 'IDXC', 'AO', 'listed', 'NEARHI', 'V60', 'liq'):
    if hasattr(P, a):
        delattr(P, a)
gc.collect()
print('載入完成', f'{time.time()-t_all:.0f}s', rss(), 'MB', {k: len(v) for k, v in CH.items()}, flush=True)

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
bw['seg_dd'] = s14lib.seg_dd(bench)
YRS = (P.IDX[-1] - P.IDX[P.t0]).days / 365.25
CELLS = ['E0', 'D', 'DA', 'C', 'CA', 'C0']


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


def overlap_share(l_e, l_d):
    """兩半同時持有同一檔：每日 重疊檔數 ÷ 事件半邊持股檔數（事件半邊有持股的日子平均）。"""
    T = P.T
    he = [set() for _ in range(T)]; hd = [set() for _ in range(T)]
    for lg, hh in ((l_e, he), (l_d, hd)):
        for r in lg.itertuples():
            for t in range(int(r.ent_t), int(r.exit_t)):
                hh[t].add(r.code)
    num = den = 0; days_any = 0
    for t in range(P.t0, T):
        if hd[t]:
            den += len(hd[t]); num += len(hd[t] & he[t]); days_any += int(bool(hd[t] & he[t]))
    return dict(share=float(num / den) if den else float('nan'), days_with_overlap=int(days_any))


# ---------------------------------------------------------------- 驗收
acc = {'P0': True}


def build(s):
    SE = s14lib.surge_signal(P, U, RT, shift=s)
    SD = s14lib.make_signal_disp(P, U, CH['D'], shift=s, adv20=ADV20)
    SDA = s14lib.make_signal_disp(P, U, CH['DA'], shift=s, adv20=ADV20)
    bad = 0
    for S, ch in ((SD, CH['D']), (SDA, CH['DA'])):
        kk = set(zip(ch['k'] + s, ch['j']))
        r, c = np.nonzero(S['sig'])
        for t, j in zip(r, c):
            if (int(t), int(j)) not in kk or not U[t, j]:
                bad += 1
        if not (ch['announce0'].to_numpy() < P.IDX[ch['s_row'].to_numpy()]).all():
            bad += 1
    acc[f'A1_shift{s}_signal_violations'] = bad
    assert bad == 0, ('A1 失敗', s, bad)
    return SE, SD, SDA


def check_trades(lg, ch, s):
    """A1：D 交易的進出場日與鏈相符。"""
    m = {(r.j, r.s_row + s): r.e_row + s for r in ch.itertuples()}
    bad = defer = 0
    for r in lg.itertuples():
        e = m.get((int(r.j), int(r.ent_t)))
        if e is None:
            bad += 1; continue
        if r.why in ('open_end', 'delist'):
            continue
        if int(r.exit_t) == e + 1:
            continue
        if int(r.exit_t) > e + 1:
            defer += 1; continue
        bad += 1
    return bad, defer


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
       'bench_0050': bw, 'cells': {c: dict(paths={}) for c in CELLS}}
navs = {'0050': bench}; trades = []
for s in range(4):
    tc = time.time()
    SE, SD, SDA = SIG0 if s == 0 else build(s)
    if s == 0:
        res['signal_counts'] = {'D': dict(n_events=SD['n_events'], n_eligible=SD['n_eligible'], n_rows=len(SD['info'])),
                                'DA': dict(n_events=SDA['n_events'], n_eligible=SDA['n_eligible'], n_rows=len(SDA['info']))}
    for cap in CAPS:
        key = f'{int(cap/1e4)}萬|+{s}'
        r = {}
        r['E0'] = sim(SE, cap); r['D'] = sim(SD, cap); r['DA'] = sim(SDA, cap)
        eh = sim(SE, cap / 2); dh = sim(SD, cap / 2); dah = sim(SDA, cap / 2)
        bad, defer = check_trades(r['D'][1], CH['D'], s)
        acc.setdefault('A1_trades', {})[key] = dict(bad=bad, deferred=defer)
        assert bad == 0, ('A1 交易失敗', key, bad)
        nv = {c: r[c][0] for c in ('E0', 'D', 'DA')}
        nv['C'] = eh[0] + dh[0]; nv['CA'] = eh[0] + dah[0]; nv['C0'] = eh[0] + half0050(cap)
        for c in CELLS:
            n = nv[c]
            res['cells'][c]['paths'][key] = dict(r19=cg(n, 'since2019'), full=cg(n, 'full'), r1923=cg(n, 'y2019_2023'), r24=cg(n, 'since2024'), maxdd=mdd(n),
                                                 seg=s14lib.seg_dd(n), corr_E0=s14lib.corr_excess(n, nv['E0'], bench) if c != 'E0' else 1.0)
        big = {c: int((r[c][1]['shares'] >= 10000).sum()) for c in ('D', 'DA')}
        res['cells']['D']['paths'][key]['n_buy_ge_10lots'] = big['D']; res['cells']['DA']['paths'][key]['n_buy_ge_10lots'] = big['DA']
        if s == 0 and cap == 1e6:
            assert np.allclose(nv['C'].to_numpy(), (eh[0] + dh[0]).to_numpy(), atol=1.0)
            acc['A2'] = True
            lgs = {'E0': r['E0'][1], 'D': r['D'][1], 'DA': r['DA'][1],
                   'C': pd.concat([eh[1].assign(sleeve='E0'), dh[1].assign(sleeve='D')], ignore_index=True),
                   'CA': pd.concat([eh[1].assign(sleeve='E0'), dah[1].assign(sleeve='DA')], ignore_index=True),
                   'C0': eh[1].assign(sleeve='E0')}
            sts = {'E0': [r['E0'][2]], 'D': [r['D'][2]], 'DA': [r['DA'][2]], 'C': [eh[2], dh[2]], 'CA': [eh[2], dah[2]], 'C0': [eh[2]]}
            sigS = {'E0': [SE], 'D': [SD], 'DA': [SDA], 'C': [SE, SD], 'CA': [SE, SDA], 'C0': [SE]}
            res['overlap_C'] = overlap_share(eh[1], dh[1]); res['overlap_CA'] = overlap_share(eh[1], dah[1])
            for c in CELLS:
                R = res['cells'][c]; n = nv[c]; lg = lgs[c]
                navs[c] = n
                w = W(n)
                R['windows'] = slim(w); R['sharpe'] = {k: w[k]['sharpe'] for k in ('full', 'since2019')}
                R['t_month_excess'] = {k: w[k]['t_month_excess'] for k in ('full', 'since2019')}
                R['dd_dates'] = s10lib.dd_dates(n); R['seg_dd'] = s14lib.seg_dd(n); R['yearly'] = stlib.yearly(n, bench)
                R['trades_full'] = stlib.trade_stats(lg); R['trades_since2019'] = stlib.trade_stats(lg, '2019-01-01')
                R['top5_share_of_gain'] = top5_share(lg, n, 1e6)
                mnav = float(n.mean())
                bv = sum(x['buy_val'] for x in sts[c]); sv = sum(x['sell_val'] for x in sts[c])
                R['turnover_oneway_ann'] = float((bv + sv) / 2 / mnav / YRS)
                R['cost_ann_pct_of_nav'] = float(sum(x['fee'] + x['tax'] + x['slip'] + x['e50_cost'] for x in sts[c]) / mnav / YRS)
                R['n_buy'] = int(sum(x['n_buy'] for x in sts[c])); R['blk_min'] = int(sum(x['blk_min'] for x in sts[c]))
                R['blk_lot'] = int(sum(x['blk_lot'] for x in sts[c])); R['blk_limup'] = int(sum(x['blk_limup'] for x in sts[c]))
                R['noslot'] = int(sum(x['noslot'] for x in sts[c])); R['blk_cap'] = int(sum(x['blk_cap'] for x in sts[c]))
                R['avg_npos'] = float(sum(x['avg_npos'] for x in sts[c]))
                sleeve_navs = {'C': [eh[0], dh[0]], 'CA': [eh[0], dah[0]], 'C0': [eh[0]]}.get(c, [n])
                sf = sum(x['stockfrac'] * sn for x, sn in zip(sts[c], sleeve_navs)) / n
                R['avg_stockfrac'] = float(np.nanmean(sf.to_numpy()))
                x = lg.copy(); x['run'] = c
                if 'sleeve' not in x:
                    x['sleeve'] = c
                trades.append(x[['run', 'sleeve', 'code', 'sig_date', 'ent_date', 'exit_date', 'hold_days', 'shares', 'cost', 'proceeds', 'pnl', 'ret', 'why', 'score']])
                blk, top = s2lib.block_from(lg, 5)
                if c in ('E0', 'D', 'DA'):
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
        del r, eh, dh, dah, nv
    print('位移', s, f'{time.time()-tc:.0f}s 累計 {time.time()-t_all:.0f}s {rss()}MB', flush=True)
    del SE, SD, SDA
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
                        seg_median={k: float(np.median([v['seg'][k] for v in pv.values()])) for k in s14lib.SEGMENTS},
                        corr_E0_median=float(np.median([v['corr_E0'] for v in pv.values()])))
    w = R['windows']; x2 = R['costx2']; b5 = R['block5']
    chk = dict(net_since2019=w['since2019']['diff'] > 0, net_full=w['full']['diff'] > 0, costx2_since2019=x2['since2019']['diff'] > 0, costx2_full=x2['full']['diff'] > 0,
               block5_since2019=b5['since2019']['diff'] > 0, block5_full=b5['full']['diff'] > 0, half1=w['half1']['diff'] > 0, half2=w['half2']['diff'] > 0)
    R['checks'] = {k: bool(v) for k, v in chk.items()}
    R['class'] = '值得上XQ測' if all(chk.values()) else ('不值得' if not (chk['net_since2019'] or chk['net_full']) else '有條件')
    R['fails'] = [k for k, v in chk.items() if not v]
    R['block5_loss_since2019_pp'] = (w['since2019']['cagr'] - b5['since2019']['cagr']) * 100

keys = list(res['cells']['E0']['paths'].keys())
b_path = {'r19': bw['since2019']['cagr'], 'r1923': bw['y2019_2023']['cagr'], 'r24': bw['since2024']['cagr'], 'full': bw['full']['cagr'], 'maxdd': bw['full']['maxdd'], 'seg': bw['seg_dd']}


def paired(a, b):
    A = res['cells'][a]['paths']
    Bp = (lambda k: b_path) if b == '0050' else (lambda k: res['cells'][b]['paths'][k])
    dr = np.array([A[k]['r19'] - Bp(k)['r19'] for k in keys]); dd = np.array([A[k]['maxdd'] - Bp(k)['maxdd'] for k in keys])
    df = np.array([A[k]['full'] - Bp(k)['full'] for k in keys])
    seg = {}
    for sg in s14lib.SEGMENTS:
        x = np.array([A[k]['seg'][sg] - Bp(k)['seg'][sg] for k in keys])
        ns = int((x > 0).sum()); md = float(np.median(x)) * 100
        seg[sg] = dict(n_shallower=ns, median_pp=md, cls='較抗跌' if (ns >= 15 and md >= 0.5) else ('較深' if ns <= 5 else '差不多'))
    d1923 = np.array([A[k]['r1923'] - Bp(k)['r1923'] for k in keys]); d24 = np.array([A[k]['r24'] - Bp(k)['r24'] for k in keys])
    return dict(ret_npos=int((dr > 0).sum()), ret_median_pp=float(np.median(dr)) * 100, ret_min_pp=float(dr.min()) * 100, ret_max_pp=float(dr.max()) * 100,
                full_npos=int((df > 0).sum()), full_median_pp=float(np.median(df)) * 100,
                dd_n_shallower=int((dd > 0).sum()), dd_median_pp=float(np.median(dd)) * 100, seg=seg,
                r1923_npos=int((d1923 > 0).sum()), r1923_median_pp=float(np.median(d1923)) * 100, r24_npos=int((d24 > 0).sum()), r24_median_pp=float(np.median(d24)) * 100)


pairs = {f'{a}-{b}': paired(a, b) for a, b in (('C', 'C0'), ('CA', 'C0'), ('C', 'E0'), ('CA', 'E0'), ('C0', 'E0'), ('D', '0050'), ('DA', '0050'), ('DA', 'D'), ('E0', '0050'), ('C0', '0050'))}
res['pairs'] = pairs
navE, navD = navs['E0'], navs['D']
corr = dict(D_E0_full=s14lib.corr_excess(navD, navE, bench, D0, DE), D_E0_since2019=s14lib.corr_excess(navD, navE, bench, D19, DE),
            DA_E0_full=s14lib.corr_excess(navs['DA'], navE, bench, D0, DE), D_E0_paths_median=res['cells']['D']['paths20']['corr_E0_median'])
mr = lambda n: n.groupby(n.index.to_period('M')).last().pct_change().dropna()
corr['D_E0_raw_monthly_full'] = float(pd.concat([mr(navD), mr(navE)], axis=1).dropna().corr().iloc[0, 1])
res['correlation'] = corr


def beat0050(c):
    p = res['cells'][c]['paths20']; w = res['cells'][c]['windows']
    if p['median'] > b19 and p['n_above_0050'] >= 15 and w['full']['diff'] > 0:
        return '贏 0050'
    if p['median'] < b19 and p['n_above_0050'] <= 5:
        return '輸 0050'
    return '不確定'


def supplement(cc):
    pr = pairs[f'{cc}-C0']
    if pr['ret_npos'] >= 15:
        v = '有補充價值' if (pr['r1923_median_pp'] > 0 and pr['r24_median_pp'] > 0) else '有補充價值，但只靠一段'
    elif pr['ret_npos'] <= 5:
        v = '沒有補充價值'
    else:
        v = '不確定'
    return dict(verdict=v, npos=pr['ret_npos'], median_pp=pr['ret_median_pp'], r1923_median_pp=pr['r1923_median_pp'], r24_median_pp=pr['r24_median_pp'],
                seg={k: v2['cls'] for k, v2 in pr['seg'].items()})


sv = supplement('C')
res['verdict'] = dict(D_vs_0050=beat0050('D'), DA_vs_0050=beat0050('DA'), C_supplement=sv, CA_supplement=supplement('CA'),
                      xq_recommend='可以加（紙上追蹤＋XQ 測；不直接改現行預設）' if sv['verdict'] == '有補充價值' else '不建議加進現行策略',
                      classes={c: res['cells'][c]['class'] for c in CELLS}, fails={c: res['cells'][c]['fails'] for c in CELLS})
print('相關', json.dumps(corr, ensure_ascii=False), flush=True)
print('判定', json.dumps(res['verdict'], ensure_ascii=False), flush=True)
for c in CELLS:
    R = res['cells'][c]
    print(c, {k: round(v, 4) if isinstance(v, float) else v for k, v in R['paths20'].items() if k != 'seg_median'}, R['class'], flush=True)
for k, v in pairs.items():
    print(k, {a: (round(b, 2) if isinstance(b, float) else b) for a, b in v.items() if a != 'seg'}, {s: (x['n_shallower'], round(x['median_pp'], 2)) for s, x in v['seg'].items()}, flush=True)

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
