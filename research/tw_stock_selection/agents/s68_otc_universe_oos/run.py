#!/usr/bin/env python3
"""S68 主跑（只跑一次）。先驗 SPEC／PREREG／code.sha256；results.json 已存在就拒跑。python3 -I -B run.py
輸出 results.json、navs.csv.gz（主跑日淨值）、monthly.csv、trades.csv.gz。"""
import sys, json, time, resource, hashlib, gc
from pathlib import Path
import numpy as np, pandas as pd
H = Path(__file__).resolve().parent
sys.path.insert(0, str(H))
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert not (H / 'results.json').exists(), 'results.json 已存在：S68 只跑一次'
for f in ('SPEC', 'PREREG'):
    assert sha(H / f'{f}.json') == (H / f'{f}.sha256').read_text().split()[0], f'{f}.json 與登記的 sha256 不符'
for line in (H / 'code.sha256').read_text().strip().split('\n'):
    h, f = line.split()
    assert sha(H / f) == h, f'{f} 與登記的 sha256 不符'
PRE = json.loads((H / 'PREREG.json').read_text()); SPEC = json.loads((H / 'SPEC.json').read_text())
import s68lib as L                                          # noqa: E402
from s68lib import s10lib, s17lib, stlib, cslib             # noqa: E402
assert SPEC['frozen_rules']['src_sha256'] == L.SRC_SHA
t0 = time.time()
rss = lambda: round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
WIN = {k: (v['frm'], v['to']) for k, v in PRE['primary']['windows'].items()}
THR = PRE['decision']['t_threshold']


def verdict(w):
    a, b = w['W2015'], w['W2019']
    if a['ann_excess_pp'] > 0 and b['ann_excess_pp'] > 0:
        return '支持' if (a['t_nw6'] > THR and b['t_nw6'] > THR) else '方向一致但不顯著'
    return '不支持'


def run_cell(P, U, RT, E, full=True):
    """一格：主跑（100 萬、位移 0）＋20 條路徑。回傳 (結果 dict, 主跑 nav, 主跑交易, 月報酬)。"""
    lastb = lambda d: P.IDX[P.IDX < pd.Timestamp(d)][-1]
    D0, DE, D19 = P.IDX[P.t0], P.IDX[-1], lastb('2019-01-01')
    e19 = stlib.cagr(E, D19, DE)
    o = dict(paths={}); tot = dict(n_sig=0, blk_limup=0, n_sized=0, n_capped=0, n_buy=0, sell_partial_cap=0, sell_defer_limdn=0, blk_noopen=0)
    for s in L.SHIFTS:
        sg = s10lib.make_signal_s10(P, U, RT, shift=s)
        for cap in L.CAPS:
            nav, lg, st = L.sim(P, sg, U, cap=cap)
            c = L.capped(st)
            for k in tot:
                tot[k] += c[k]
            o['paths'][f'{int(cap/1e4)}萬|+{s}'] = dict(since2019=stlib.cagr(nav, D19, DE), full=stlib.cagr(nav, D0, DE), maxdd=stlib.maxdd(nav, D0, DE),
                                                        d_since2019_pp=(stlib.cagr(nav, D19, DE) - e19) * 100)
            if s == 0 and cap == 1e6:
                N0, LG0, ST0, SG0 = nav, lg, st, sg
    d = np.array([v['d_since2019_pp'] for v in o['paths'].values()]); c19 = np.array([v['since2019'] for v in o['paths'].values()])
    o['paths20'] = dict(n=int(len(d)), median_since2019=float(np.median(c19)), min_since2019=float(c19.min()), max_since2019=float(c19.max()), e50_since2019=e19,
                        median_diff_pp=float(np.median(d)), n_positive=int((d > 0).sum()), min_diff_pp=float(d.min()), max_diff_pp=float(d.max()),
                        aux_support=bool((d > 0).sum() >= 15 and np.median(d) > 0),
                        median_maxdd=float(np.median([v['maxdd'] for v in o['paths'].values()])))
    o['paths20_exec'] = dict(tot, limup_rate=tot['blk_limup'] / tot['n_sig'] if tot['n_sig'] else None, capped_rate=tot['n_capped'] / tot['n_sized'] if tot['n_sized'] else None)
    M = L.monthly(N0, E)
    o['windows'] = {k: L.excess_stats(M.loc[a:b]) for k, (a, b) in WIN.items()}
    o['verdict'] = verdict(o['windows'])
    o['main'] = dict(since2019=stlib.cagr(N0, D19, DE), full=stlib.cagr(N0, D0, DE), maxdd=stlib.maxdd(N0, D0, DE), maxdd_since2019=stlib.maxdd(N0, D19, DE),
                     e50_since2019=e19, e50_full=stlib.cagr(E, D0, DE), e50_maxdd=stlib.maxdd(E, D0, DE), end_nav=float(N0.iloc[-1]))
    if full:
        x = LG0[LG0['why'] != 'open_end']
        mk_exit = P.MK[np.minimum(LG0['exit_t'].to_numpy(int), P.T - 1), LG0['j'].to_numpy(int)]
        mk_ent = P.MK[LG0['ent_t'].to_numpy(int), LG0['j'].to_numpy(int)]
        o['trades'] = dict(n=int(len(LG0)), n_closed=int(len(x)), win_rate=float((x['ret'] > 0).mean()), mean_ret=float(x['ret'].mean()), median_ret=float(x['ret'].median()),
                           avg_hold_days=float(x['hold_days'].mean()), why={str(k): int(v) for k, v in LG0['why'].value_counts().items()},
                           share_otc_entries=float((mk_ent == 2).mean()), n_market_changed_while_held=int((mk_exit != mk_ent).sum() - int(((mk_exit == 0) & (mk_ent != 0)).sum())),
                           top5_share=float(s17lib.top5_share(LG0, N0, 1e6)) if N0.iloc[-1] > 1e6 else None,
                           top10=[dict(code=str(r.code), ent=str(r.ent_date.date()), exit=str(r.exit_date.date()), ret=float(r.ret), pnl=float(r.pnl), market=r.market)
                                  for r in LG0.sort_values('pnl', ascending=False).head(10).itertuples()],
                           avg_npos=float(ST0['avg_npos']), avg_idle_frac=float(ST0['avg_idle_frac']))
        o['exec_main'] = L.capped(ST0)
        o['yearly'] = {str(k): v for k, v in stlib.yearly(N0, E).items()}
        o['seg_dd'] = s17lib.seg_dd(N0)
        o['n_tgt'] = dict(avg=float(np.mean([i['n_tgt'] for i in SG0['info']])), months=len(SG0['info']))
        # 容量：1,000 萬、位移 0
        nav, lg, st = L.sim(P, SG0, U, cap=1e7)
        o['capacity_1000wan'] = dict(L.capped(st), since2019=stlib.cagr(nav, D19, DE), full=stlib.cagr(nav, D0, DE), maxdd=stlib.maxdd(nav, D0, DE),
                                     avg_npos=float(st['avg_npos']), avg_idle_frac=float(st['avg_idle_frac']),
                                     shrunk_cash=int(st['shrunk']), d_since2019_vs_100wan_pp=(stlib.cagr(nav, D19, DE) - o['main']['since2019']) * 100)
    return o, N0, LG0, M


res = dict(spec_sha256=sha(H / 'SPEC.json'), prereg_sha256=sha(H / 'PREREG.json'), cells={}, sens={})
navs = {}; mon = {}; trades = []
for mode in ('main', 'none', 'zero'):
    P = L.load(mode)
    UU, RT = L.base(P)
    E = pd.Series(P.E50, index=P.IDX)
    if mode == 'main':
        res['otc_info'] = P.otc_info; res['exright_info_twse'] = P.exright_info
        for c in ('OTC', 'ALL', 'CUR'):
            o, N0, LG0, M = run_cell(P, UU[c], RT, E)
            res['cells'][c] = o; navs[c] = N0; mon[c] = M
            t = LG0[['code', 'market', 'sig_date', 'ent_date', 'exit_date', 'why', 'shares', 'cost', 'proceeds', 'ret', 'pnl', 'hold_days']].copy(); t.insert(0, 'cell', c); trades.append(t)
            print(c, o['verdict'], {k: (round(v['ann_excess_pp'], 2), round(v['t_nw6'], 2)) for k, v in o['windows'].items()}, o['paths20']['median_diff_pp'], o['paths20']['n_positive'], f'{time.time()-t0:.0f}s', rss(), flush=True)
        navs['0050'] = E.reindex(N0.index)
        cur = res['cells']['CUR']['main']
        assert cur['since2019'] == L.ACC['since2019'] and cur['full'] == L.ACC['full'] and res['cells']['CUR']['trades']['n_closed'] == 381, 'CUR 驗收失敗'
    else:
        o, N0, LG0, M = run_cell(P, UU['OTC'], RT, E, full=False)
        o['otc_info'] = P.otc_info
        res['sens'][mode] = o; navs[f'OTC_{mode}'] = N0
        print('OTC', mode, o['verdict'], {k: (round(v['ann_excess_pp'], 2), round(v['t_nw6'], 2)) for k, v in o['windows'].items()}, o['paths20']['median_diff_pp'], o['paths20']['n_positive'], flush=True)
    del P, UU, RT; gc.collect()

# 判定（PREREG）
po = res['cells']['OTC']
res['decision'] = dict(rule=PRE['decision']['rule'], verdict=po['verdict'],
                       W2015=dict(ann_excess_pp=po['windows']['W2015']['ann_excess_pp'], t_nw6=po['windows']['W2015']['t_nw6']),
                       W2019=dict(ann_excess_pp=po['windows']['W2019']['ann_excess_pp'], t_nw6=po['windows']['W2019']['t_nw6']),
                       aux=dict(median_diff_pp=po['paths20']['median_diff_pp'], n_positive=po['paths20']['n_positive'], aux_support=po['paths20']['aux_support']),
                       aux_consistent=bool(po['paths20']['aux_support'] == (po['verdict'] == '支持')))
# 描述：ALL、OTC 對 CUR 逐條差；月超額相關
for c in ('OTC', 'ALL'):
    d = np.array([(res['cells'][c]['paths'][k]['since2019'] - res['cells']['CUR']['paths'][k]['since2019']) * 100 for k in res['cells']['CUR']['paths']])
    res['cells'][c]['vs_CUR'] = dict(median_pp=float(np.median(d)), n_higher=int((d > 0).sum()), min_pp=float(d.min()), max_pp=float(d.max()),
                                     corr_monthly_excess=float(np.corrcoef(mon[c]['excess'], mon['CUR']['excess'])[0, 1]))
    md = mon[c]['strategy'] - mon['CUR']['strategy']
    res['cells'][c]['vs_CUR']['monthly_diff'] = {k: dict(ann_pp=float(md.loc[a:b].mean() * 1200), t_nw6=L.nw_t(md.loc[a:b].to_numpy())) for k, (a, b) in WIN.items()}
res['runtime_s'] = round(time.time() - t0, 1); res['maxrss_mb'] = rss()
pd.DataFrame(navs).to_csv(H / 'navs.csv.gz')
pd.concat({c: m for c, m in mon.items()}, axis=1).to_csv(H / 'monthly.csv')
pd.concat(trades, ignore_index=True).to_csv(H / 'trades.csv.gz', index=False)
(H / 'results.json').write_text(json.dumps(res, ensure_ascii=False, indent=1, default=lambda o: o.item() if hasattr(o, 'item') else str(o)))
print(json.dumps(res['decision'], ensure_ascii=False, indent=1)); print('runtime', res['runtime_s'], 'rss', res['maxrss_mb'])
