#!/usr/bin/env python3
"""S16：SURGE 股池加入上櫃。交易格 CUR（只上市）、ALL（上市＋上櫃）、OTC（只上櫃）、ALL_O20（上櫃成交值門檻 2,000 萬，敏感度）；資料版本對照 ALL_RAW（未補洞營收）。
全部 W10_70、F1 進場、S10 E0 出場、A10 開盤。每格 100 萬主跑、20 條路徑（5 種資金 × 位移 0～3）、成本加倍、禁買最賺 5 筆。
python3 -I -B run.py → results.json、trades.csv.gz、navs.csv.gz。PREREG.json 的 sha256 必須相符。單行程。"""
import sys, json, hashlib, time, resource, gc
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
import s16lib
from s16lib import s10lib, s9lib, s4lib, s2lib, stlib, COSTS

H = Path(__file__).resolve().parent
pre = (H / 'PREREG.json').read_bytes()
sha = hashlib.sha256(pre).hexdigest()
assert sha == (H / 'PREREG.sha256').read_text().split()[0], 'PREREG.json 與 PREREG.sha256 不符'
PRE = json.loads(pre)
t_all = time.time()
rss = lambda: round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
P = s16lib.load()
assert (P.T, P.N, str(P.IDX[-1].date())) == (2876, 2142, '2026-10-22'), 'P0 失敗'
INNOV, _ = s4lib.innov_panel(P)
UN = s16lib.universes(P, INNOV)
del INNOV
RT = s16lib.rev_tables_s16(P, patch=True)
RT0 = s2lib.rev_tables(P)
A3 = s16lib.rev_tables_s16(P, patch=False)
acc = {'P0': True, 'A3': bool(A3['S'].equals(RT0['S']))}
del A3
MKT = P.MK.copy()
V60 = P.V60.astype(np.float32)
for a in ('H', 'L', 'VOL', 'CHG', 'HF', 'VOLF', 'adjH', 'HI240', 'MA60', 'nbar', 'F_HI', 'IDXC', 'AO', 'listed', 'NEARHI', 'V60', 'liq'):   # F_MA20 留著（make_signal_s10 要用）
    if hasattr(P, a):
        delattr(P, a)
gc.collect()
# A2：股池集合關係
U = UN
acc['A2_sets'] = dict(all_eq_cur_or_otc=bool(np.array_equal(U['ALL'], U['CUR'] | U['OTC'])), cur_otc_disjoint=bool(not (U['CUR'] & U['OTC']).any()),
                      o20_subset_all=bool(not (U['ALL_O20'] & ~U['ALL']).any()), o20_twse_eq_cur=bool(np.array_equal(U['ALL_O20'] & (MKT == 1), U['CUR'])))
assert acc['A3'] and all(acc['A2_sets'].values()), ('驗收 A2／A3 失敗', acc)
print('載入完成', f'{time.time()-t_all:.0f}s', rss(), 'MB', flush=True)

bench = pd.Series(P.E50[P.t0:], index=P.IDX[P.t0:])
i_mid = (P.t0 + P.T - 1) // 2
D0, DM, DE = P.IDX[P.t0], P.IDX[i_mid], P.IDX[-1]
D19 = P.IDX[P.IDX < pd.Timestamp('2019-01-01')][-1]
D24 = P.IDX[P.IDX < pd.Timestamp('2024-01-01')][-1]
WIN = {'since2019': (D19, DE), 'full': (D0, DE), 'since2024': (D24, DE), 'half1': (D0, DM), 'half2': (DM, DE), 'y2019_2023': (D19, D24)}
W = lambda nav: {k: stlib.window(nav, bench, a, b) for k, (a, b) in WIN.items()}
cg = lambda nav, k: stlib.cagr(nav, *WIN[k])
mdd = lambda nav: stlib.maxdd(nav, D0, DE)
CAPS = (6e5, 8e5, 1e6, 1.5e6, 2e6)
bw = {k: dict(cagr=stlib.cagr(bench, a, b), maxdd=stlib.maxdd(bench, a, b)) for k, (a, b) in WIN.items()}
bw['seg'] = s16lib.seg_dd(bench)
B19 = bw['since2019']['cagr']
YRS = (P.IDX[-1] - P.IDX[P.t0]).days / 365.25
CELLS = ['CUR', 'ALL', 'OTC', 'ALL_O20', 'ALL_RAW']
CELL_U = {'CUR': 'CUR', 'ALL': 'ALL', 'OTC': 'OTC', 'ALL_O20': 'ALL_O20', 'ALL_RAW': 'ALL'}


def same_trades(l1, l2):
    a = l1[['code', 'ent_t', 'exit_t', 'why']].reset_index(drop=True); b = l2[['code', 'ent_t', 'exit_t', 'why']].reset_index(drop=True)
    return bool(len(a) == len(b) and (a == b).all().all())


def slim(w, keys=('since2019', 'full', 'since2024', 'half1', 'half2', 'y2019_2023')):
    return {k: dict(cagr=w[k]['cagr'], diff=w[k]['diff'], maxdd=w[k]['maxdd']) for k in keys}


def build(s):
    out = {}
    for c in CELLS:
        out[c] = s16lib.surge_signal(P, U[CELL_U[c]], RT0 if c == 'ALL_RAW' else RT, shift=s)
        out[c].pop('A2_mismatch', None)
        # A2：買進訊號都在該格股池內
        assert not (out[c]['sig'] & ~U[CELL_U[c]]).any(), ('A2 失敗：訊號不在股池', c, s)
    S4 = s4lib.make_signal(P, U['CUR'], RT, depth=10, mom_lo=0.10, exit='top20', shift=s, keep_info=True)
    C = out['CUR']
    acc[f'A1_shift{s}'] = bool(np.array_equal(C['sig'], S4['sig']) and np.array_equal(C['X'], S4['X'])
                               and np.array_equal(np.nan_to_num(C['rank'], nan=-9e9), np.nan_to_num(S4['rank'], nan=-9e9))
                               and [x['k'] for x in C['info']] == [x['k'] for x in S4['info']])
    assert acc[f'A1_shift{s}'], ('A1 失敗', s)
    del S4; gc.collect()
    return out


def sim(S, univ, capital=1e6, cost='strict', block=None):
    return s10lib.simulate_s10(P, S, univ, sat_w=0.7, sizing='W10', lock=False, cost=COSTS[cost], capital=capital, slots=10, block=block)


def mkt_at_sig(lg):
    return np.where(MKT[lg['sig_t'].to_numpy(int), lg['j'].to_numpy(int)] == 1, 'twse', 'tpex')


def list_stats(S):
    info = S['info']
    n = np.array([x['n_tgt'] for x in info]); nma = np.array([x['n_ma'] for x in info])
    d = pd.to_datetime([x['sig_date'] for x in info])
    m19 = d >= pd.Timestamp('2019-01-01')
    return dict(months=int(len(n)), avg_list=float(n.mean()), empty_months=int((n == 0).sum()), avg_lma=float(nma.mean()),
                avg_list_since2019=float(n[m19].mean()), empty_months_since2019=int((n[m19] == 0).sum()), avg_lma_since2019=float(nma[m19].mean()))


def cap_by_mkt(lg):
    if not len(lg):
        return {}
    b = s16lib.cap_bound(P, lg)
    mk = mkt_at_sig(lg)
    return {m: dict(buys=int((mk == m).sum()), cap_bound=int((b & (mk == m)).sum())) for m in ('twse', 'tpex')}


SIG = {0: build(0)}
SG0 = SIG[0]
# A0
n9, l9, _ = s9lib.simulate_s9(P, SG0['CUR'], U['CUR'], sat_w=0.7, sizing='W10', lock=False, cost=COSTS['strict'], exec_mode='open')
n10, l10, _ = sim(SG0['CUR'], U['CUR'])
acc['A0'] = dict(since2019=cg(n10, 'since2019'), full=cg(n10, 'full'), maxdd=mdd(n10),
                 nav_identical=bool(np.allclose(n9.to_numpy(), n10.to_numpy(), rtol=0, atol=1e-6)), trades_identical=same_trades(l9, l10))
acc['A0']['ok'] = bool(abs(acc['A0']['since2019'] - 0.4813) < 0.00025 and abs(acc['A0']['full'] - 0.3408) < 0.00025 and abs(acc['A0']['maxdd'] + 0.3340) < 0.00025
                       and acc['A0']['nav_identical'] and acc['A0']['trades_identical'])
del n9, l9, n10, l10
assert acc['A0']['ok'], ('驗收 A0 失敗', acc)
print('驗收 A0～A3 通過', acc, flush=True)

res = {'PREREG_sha256': sha, 'acceptance': acc,
       'meta': dict(panel_T=P.T, panel_N=P.N, start=str(D0.date()), mid=str(DM.date()), since2019_from=str(D19.date()), since2024_from=str(D24.date()), end=str(DE.date())),
       'bench_0050': bw, 'cells': {c: dict(paths={}) for c in CELLS}}
navs = {'0050': bench}; trades = []
for s in range(4):
    tc = time.time()
    SG = SIG.pop(s) if s in SIG else build(s)
    for cid in CELLS:
        R = res['cells'][cid]; Uc = U[CELL_U[cid]]
        if s == 0:
            R['list'] = list_stats(SG[cid])
        R.setdefault('list_shift', {})[s] = list_stats(SG[cid])['avg_list']
        for cap in CAPS:
            key = f'{int(cap/1e4)}萬|+{s}'
            nav, lg, st = sim(SG[cid], Uc, capital=cap)
            cl = lg[lg['why'] != 'open_end']
            R['paths'][key] = dict(r19=cg(nav, 'since2019'), full=cg(nav, 'full'), r1923=cg(nav, 'y2019_2023'), r24=cg(nav, 'since2024'), maxdd=mdd(nav),
                                   seg=s16lib.seg_dd(nav), top5=s16lib.top5_share(lg, nav, cap), dd_dates=[s10lib.dd_dates(nav)['peak'], s10lib.dd_dates(nav)['trough']],
                                   n_trades=int(len(cl)), share_tpex=float((mkt_at_sig(lg) == 'tpex').mean()) if len(lg) else 0.0,
                                   blk_cap=int(st['blk_cap']), blk_lot=int(st['blk_lot']), blk_min=int(st['blk_min']), blk_limup=int(st['blk_limup']), sell_partial_cap=int(st['sell_partial_cap']),
                                   cap_by_mkt=cap_by_mkt(lg))
            if s == 0 and cap == 1e6:
                navs[cid] = nav
                w = W(nav)
                R['windows'] = slim(w)
                R['sharpe'] = {k: w[k]['sharpe'] for k in ('full', 'since2019')}
                R['t_month_excess'] = {k: w[k]['t_month_excess'] for k in ('full', 'since2019')}
                R['dd_dates'] = s10lib.dd_dates(nav)
                R['seg'] = s16lib.seg_dd(nav)
                R['yearly'] = stlib.yearly(nav, bench)
                lg['market'] = mkt_at_sig(lg)
                R['trades_full'] = stlib.trade_stats(lg); R['trades_since2019'] = stlib.trade_stats(lg, '2019-01-01')
                R['trades_by_mkt'] = {m: stlib.trade_stats(lg[lg['market'] == m]) for m in ('twse', 'tpex') if (lg['market'] == m).any()}
                R['engine'] = {k: (int(v) if isinstance(v, (int, np.integer)) else float(v)) for k, v in st.items() if isinstance(v, (int, float, np.integer, np.floating))}
                R['avg_stockfrac'] = float(np.nanmean(st['stockfrac'].to_numpy()))
                mnav = float(nav.mean())
                R['turnover_oneway_ann'] = float((st['buy_val'] + st['sell_val']) / 2 / mnav / YRS)
                R['cost_ann_pct_of_nav'] = float((st['fee'] + st['tax'] + st['slip'] + st['e50_cost']) / mnav / YRS)
                R['top5_share_of_gain'] = s16lib.top5_share(lg, nav, 1e6)
                # 每筆附加：報酬貢獻、5% 上限是否綁住、進場時 60 日均成交值
                con = s16lib.trade_contrib(P, lg, nav)
                for c in con.columns:
                    lg[f'contrib_{c}'] = con[c].to_numpy()
                lg['cap_bound'] = s16lib.cap_bound(P, lg)
                lg['v60_e8'] = V60[lg['sig_t'].to_numpy(int), lg['j'].to_numpy(int)] / 1e8
                lg['buy_amt'] = lg['shares'] * P.O[lg['ent_t'].to_numpy(int), lg['j'].to_numpy(int)]
                lg['mk_exit'] = np.where(MKT[np.minimum(lg['exit_t'].to_numpy(int), P.T - 1), lg['j'].to_numpy(int)] == 1, 'twse',
                                         np.where(MKT[np.minimum(lg['exit_t'].to_numpy(int), P.T - 1), lg['j'].to_numpy(int)] == 2, 'tpex', 'none'))
                R['nav_contrib'] = s16lib.nav_contrib(nav)
                x = lg.copy(); x['run'] = cid
                trades.append(x[['run', 'code', 'market', 'mk_exit', 'sig_date', 'ent_date', 'exit_date', 'hold_days', 'shares', 'cost', 'proceeds', 'pnl', 'ret', 'why', 'score',
                                 'contrib_full', 'contrib_since2019', 'contrib_y2019_2023', 'contrib_since2024', 'cap_bound', 'v60_e8', 'buy_amt']])
                if cid != 'ALL_RAW':
                    nx2 = sim(SG[cid], Uc, cost='strict_x2')[0]
                    R['costx2'] = slim(W(nx2), ('since2019', 'full'))
                    blk, top = s2lib.block_from(lg, 5)
                    nb = sim(SG[cid], Uc, block=blk)[0]
                    R['block5'] = slim(W(nb), ('since2019', 'full'))
                    R['block5_loss_since2019_pp'] = (w['since2019']['cagr'] - R['block5']['since2019']['cagr']) * 100
                    R['block5_top'] = [dict(code=y.code, market=y.market, ent=str(y.ent_date.date()), exit=str(y.exit_date.date()), ret=float(y.ret), pnl=float(y.pnl)) for y in top.itertuples()]
                    del nx2, nb
                    if cid == 'OTC':
                        assert (lg['market'] == 'tpex').all(), 'A2 失敗：OTC 有上市交易'
                    if cid == 'CUR':
                        assert (lg['market'] == 'twse').all(), 'A2 失敗：CUR 有上櫃交易'
            del nav, lg, st, cl
    print('位移', s, f'{time.time()-tc:.0f}s 累計 {time.time()-t_all:.0f}s {rss()}MB', flush=True)
    del SG; gc.collect()
acc['A2_trade_market'] = True

# ---------------------------------------------------------------- 彙整、八項
for cid, R in res['cells'].items():
    pv = R['paths']
    r19 = np.array([v['r19'] for v in pv.values()]); dd = np.array([v['maxdd'] for v in pv.values()])
    assert len(r19) == 20
    R['paths20'] = dict(min=float(r19.min()), median=float(np.median(r19)), max=float(r19.max()), n_above_0050=int((r19 > B19).sum()),
                        maxdd_min=float(dd.min()), maxdd_median=float(np.median(dd)), maxdd_max=float(dd.max()),
                        r1923_median=float(np.median([v['r1923'] for v in pv.values()])), r24_median=float(np.median([v['r24'] for v in pv.values()])),
                        full_median=float(np.median([v['full'] for v in pv.values()])),
                        top5_median=float(np.median([v['top5'] for v in pv.values()])),
                        share_tpex_median=float(np.median([v['share_tpex'] for v in pv.values()])),
                        seg_median={k: float(np.median([v['seg'][k] for v in pv.values()])) for k in s16lib.SEGMENTS},
                        cap_tpex_total=int(sum(v['cap_by_mkt'].get('tpex', {}).get('cap_bound', 0) for v in pv.values())),
                        cap_twse_total=int(sum(v['cap_by_mkt'].get('twse', {}).get('cap_bound', 0) for v in pv.values())),
                        buys_tpex_total=int(sum(v['cap_by_mkt'].get('tpex', {}).get('buys', 0) for v in pv.values())),
                        buys_twse_total=int(sum(v['cap_by_mkt'].get('twse', {}).get('buys', 0) for v in pv.values())))
    if cid == 'ALL_RAW':
        R['class'] = '（資料版本對照，不判定）'
        continue
    w = R['windows']; x2 = R['costx2']; b5 = R['block5']
    chk = dict(net_since2019=w['since2019']['diff'] > 0, net_full=w['full']['diff'] > 0, costx2_since2019=x2['since2019']['diff'] > 0, costx2_full=x2['full']['diff'] > 0,
               block5_since2019=b5['since2019']['diff'] > 0, block5_full=b5['full']['diff'] > 0, half1=w['half1']['diff'] > 0, half2=w['half2']['diff'] > 0)
    R['checks'] = {k: bool(v) for k, v in chk.items()}
    R['class'] = '值得上XQ測' if all(chk.values()) else ('不值得' if not (chk['net_since2019'] or chk['net_full']) else '有條件')
    R['fails'] = [k for k, v in chk.items() if not v]
keys = list(res['cells']['CUR']['paths'].keys())


def paired(a, b='CUR'):
    A, B = res['cells'][a]['paths'], res['cells'][b]['paths']
    d = lambda f: np.array([A[k][f] - B[k][f] for k in keys])
    dr, dfull, dd, d1923, d24, dt5 = d('r19'), d('full'), d('maxdd'), d('r1923'), d('r24'), d('top5')
    seg = {}
    for sg in s16lib.SEGMENTS:
        x = np.array([A[k]['seg'][sg] - B[k]['seg'][sg] for k in keys])
        n, m = int((x > 0).sum()), float(np.median(x)) * 100
        seg[sg] = dict(n_shallower=n, median_pp=m, verdict='較淺' if (n >= 15 and m >= 1) else ('較深' if (n <= 5 and m <= -1) else '差不多'))
    out = dict(ret_npos=int((dr > 0).sum()), ret_median_pp=float(np.median(dr)) * 100, ret_min_pp=float(dr.min()) * 100, ret_max_pp=float(dr.max()) * 100,
               full_npos=int((dfull > 0).sum()), full_median_pp=float(np.median(dfull)) * 100,
               dd_n_shallower=int((dd > 0).sum()), dd_median_pp=float(np.median(dd)) * 100, seg=seg,
               r1923_npos=int((d1923 > 0).sum()), r1923_median_pp=float(np.median(d1923)) * 100, r24_npos=int((d24 > 0).sum()), r24_median_pp=float(np.median(d24)) * 100,
               top5_npos_lower=int((dt5 < 0).sum()), top5_median_diff=float(np.median(dt5)),
               by_path={k: dict(r19_pp=float(A[k]['r19'] - B[k]['r19']) * 100, dd_pp=float(A[k]['maxdd'] - B[k]['maxdd']) * 100) for k in keys})
    n = out['ret_npos']; s_ok = [out['r1923_median_pp'] >= 0, out['r24_median_pp'] >= 0]; dd_ok = out['dd_median_pp'] >= -2
    if n >= 15 and all(s_ok) and dd_ok:
        v = '加上櫃較好'
    elif n >= 15 and not dd_ok:
        v = '報酬較高，但回撤代價超過 2 個百分點'
    elif n >= 15 and any(s_ok):
        v = '報酬較高，但只靠一段'
    elif n >= 15:
        v = '報酬較高，但兩段中位數都 < 0（矛盾，需查）'
    elif n <= 5:
        v = '加上櫃較差'
    else:
        v = '差不多／不確定'
    out['verdict'] = v
    t5 = out['top5_npos_lower']; md = out['top5_median_diff']
    out['concentration'] = '較分散' if (t5 >= 15 and md < 0) else ('較集中' if t5 <= 5 else '差不多')
    return out


pairs = {'ALL-CUR': paired('ALL'), 'OTC-CUR': paired('OTC'), 'ALL_O20-CUR': paired('ALL_O20'), 'ALL_O20-ALL': paired('ALL_O20', 'ALL'), 'ALL_RAW-ALL': paired('ALL_RAW', 'ALL'),
         'ALL_RAW-CUR': paired('ALL_RAW')}
res['pairs'] = pairs


def beat0050(c):
    p = res['cells'][c]['paths20']
    if p['median'] > B19 and p['n_above_0050'] >= 15:
        return 'SURGE 在上櫃有效'
    if p['median'] < B19 and p['n_above_0050'] <= 5:
        return '在上櫃無效'
    return '不確定'


v = dict(main_ALL_vs_CUR=pairs['ALL-CUR']['verdict'], concentration_ALL_vs_CUR=pairs['ALL-CUR']['concentration'],
         bear_segments_ALL_vs_CUR={sg: pairs['ALL-CUR']['seg'][sg]['verdict'] for sg in s16lib.SEGMENTS},
         OTC_alone=beat0050('OTC'), OTC_vs_CUR=pairs['OTC-CUR']['verdict'],
         ALL_O20_vs_CUR=pairs['ALL_O20-CUR']['verdict'], ALL_O20_vs_ALL=pairs['ALL_O20-ALL']['verdict'],
         classes={c: res['cells'][c]['class'] for c in CELLS}, class_fails={c: res['cells'][c].get('fails') for c in CELLS if res['cells'][c].get('fails')})
v['xq'] = ('寫給 XQ 的做法（20／21 號腳本股池改上市＋上櫃）' if (v['main_ALL_vs_CUR'] == '加上櫃較好' and res['cells']['ALL']['class'] == '值得上XQ測')
           else ('可選、紙上追蹤' if v['main_ALL_vs_CUR'] in ('報酬較高，但只靠一段', '報酬較高，但回撤代價超過 2 個百分點') else '現行不改（股池維持只上市）'))
res['verdict'] = v
print('判定', json.dumps(v, ensure_ascii=False), flush=True)
for c in CELLS:
    R = res['cells'][c]
    print(c, {k: round(R['windows'][k]['cagr'] * 100, 2) for k in ('since2019', 'full', 'since2024', 'y2019_2023')}, 'dd', round(R['windows']['full']['maxdd'] * 100, 2),
          'p20', round(R['paths20']['median'] * 100, 2), R['paths20']['n_above_0050'], round(R['paths20']['maxdd_median'] * 100, 2), 'top5', round(R['paths20']['top5_median'], 3),
          'tpex', round(R['paths20']['share_tpex_median'], 3), R['class'], flush=True)
for k, p in pairs.items():
    print(k, p['ret_npos'], round(p['ret_median_pp'], 2), 'dd', p['dd_n_shallower'], round(p['dd_median_pp'], 2), '1923', round(p['r1923_median_pp'], 2), '24', round(p['r24_median_pp'], 2),
          'top5', p['top5_npos_lower'], round(p['top5_median_diff'], 3), p['verdict'], flush=True)

res['meta']['runtime_s'] = round(time.time() - t_all, 1); res['meta']['maxrss_mb'] = rss()
res['meta']['N'] = PRE['data_snooping']
(H / 'results.json').write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str))
pd.DataFrame(navs).round(0).to_csv(H / 'navs.csv.gz', index_label='date')
tr = pd.concat(trades, ignore_index=True)
num = tr.select_dtypes('number').columns
tr[num] = tr[num].round(6)
for c in ('sig_date', 'ent_date', 'exit_date'):
    tr[c] = pd.to_datetime(tr[c]).dt.strftime('%Y-%m-%d')
tr.to_csv(H / 'trades.csv.gz', index=False)
print(res['meta']['runtime_s'], 's', res['meta']['maxrss_mb'], 'MB')
