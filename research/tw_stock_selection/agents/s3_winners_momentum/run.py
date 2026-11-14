#!/usr/bin/env python3
"""S3：D1 基準＋E1～E5＋資金規模（D1、E1 × 30 萬／300 萬）。每格 {輕, 嚴, 嚴×2} × {空位現金, 空位 0050（e50_fix）}
＋禁買最賺 5／10 筆＋換股日位移 3／5／10＋相對同資金 D1 的配對差；兩個診斷。
python3 -I -B run.py → results.json、trades.csv.gz、navs.csv.gz。必須先有 PREREG.json（sha256 寫進 results.json）。"""
import sys, json, hashlib, time, resource, gc
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
import s3lib
from s3lib import s2lib, stlib, COSTS

H = Path(__file__).resolve().parent
pre = (H / 'PREREG.json').read_bytes()
sha = hashlib.sha256(pre).hexdigest()
assert sha == (H / 'PREREG.sha256').read_text().split()[0], 'PREREG.json 與 PREREG.sha256 不符'
t_all = time.time()
P = s3lib.load()
rss_after_load = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
U = stlib.universe(P, 'twse')
RT = s2lib.rev_tables(P)
for a in ('H', 'L', 'VOL', 'CHG', 'HF', 'VOLF', 'adjH', 'HI240', 'MA60', 'nbar', 'F_HI', 'IDXC'):   # 之後用不到的陣列先釋放（記憶體 2GB 內）
    if hasattr(P, a):
        delattr(P, a)
gc.collect()
bench = pd.Series(P.E50[P.t0:], index=P.IDX[P.t0:])
i_mid = (P.t0 + P.T - 1) // 2
D0, DM, DE = P.IDX[P.t0], P.IDX[i_mid], P.IDX[-1]
D19 = P.IDX[P.IDX < pd.Timestamp('2019-01-01')][-1]
D24 = P.IDX[P.IDX < pd.Timestamp('2024-01-01')][-1]
WIN = {'since2019': (D19, DE), 'full': (D0, DE), 'since2024': (D24, DE), 'half1': (D0, DM), 'half2': (DM, DE)}

GRID = {
    'D1':     dict(sigkw=dict(), capital=1e6, base=None),
    'E1':     dict(sigkw=dict(hold_n=20), capital=1e6, base='D1'),
    'E2':     dict(sigkw=dict(mom=0.10), capital=1e6, base='D1'),
    'E3':     dict(sigkw=dict(mom=0.10, hold_n=20), capital=1e6, base='D1'),
    'E4':     dict(sigkw=dict(hold_n=30), capital=1e6, base='D1'),
    'E5a':    dict(sigkw=dict(mom=0.05), capital=1e6, base='D1'),
    'E5b':    dict(sigkw=dict(mom=0.15), capital=1e6, base='D1'),
    'D1_30':  dict(sigkw=dict(), capital=3e5, base=None),
    'E1_30':  dict(sigkw=dict(hold_n=20), capital=3e5, base='D1_30'),
    'D1_300': dict(sigkw=dict(), capital=3e6, base=None),
    'E1_300': dict(sigkw=dict(hold_n=20), capital=3e6, base='D1_300'),
}

# ---- 驗收：D1 訊號矩陣與 s2lib 完全相同；嚴成本 2019 起 41.56%、全期 30.63%、914 筆
S_ref = s2lib.make_signal(P, U, RT, kind='surge', nsel=10, filt=P.F_MA60)
S_d1 = s3lib.make_signal(P, U, RT)
assert (S_ref['sig'] == S_d1['sig']).all() and (S_ref['X'] == S_d1['X']).all() and np.array_equal(S_ref['rank'], S_d1['rank'], equal_nan=True)
n0, l0, _ = s2lib.simulate(P, S_d1, U, idle='cash', cost=COSTS['strict'])
acc = dict(cagr_since2019=stlib.cagr(n0, D19, DE), cagr_full=stlib.cagr(n0, D0, DE), n_closed=int((l0['why'] != 'open_end').sum()), maxdd=stlib.maxdd(n0, D0, DE))
assert abs(acc['cagr_since2019'] - 0.4156) < 0.0002 and abs(acc['cagr_full'] - 0.3063) < 0.0002 and acc['n_closed'] == 914, ('D1 驗收失敗', acc)
print('驗收通過', acc, flush=True)


def W(nav):
    return {k: stlib.window(nav, bench, a, b) for k, (a, b) in WIN.items()}


def slim(w, keys=('since2019', 'full', 'since2024')):
    return {k: dict(cagr=w[k]['cagr'], cagr_0050=w[k]['cagr_0050'], diff=w[k]['diff'], maxdd=w[k]['maxdd']) for k in keys}


res = {'PREREG_sha256': sha, 'acceptance_D1_strict_cash': acc,
       'meta': dict(panel_T=P.T, panel_N=P.N, start=str(D0.date()), mid=str(DM.date()), since2019_from=str(D19.date()), since2024_from=str(D24.date()), end=str(DE.date()),
                    n_signal_days=len(S_d1['info']), first_signal=S_d1['info'][0]['sig_date'], last_signal=S_d1['info'][-1]['sig_date']),
       'bench_0050': {k: dict(cagr=stlib.cagr(bench, a, b), maxdd=stlib.maxdd(bench, a, b)) for k, (a, b) in WIN.items()},
       'bench_0050_yearly': {y: v['e0050'] for y, v in stlib.yearly(bench, bench).items()},
       'costs': {k: COSTS[k] for k in ('light', 'strict', 'strict_x2')}, 'grid': {}, 'targets': {}, 'diagnostics': {}}
navs = {'0050': bench}; trades = []; keep = {}
NSD = len(S_d1['info'])

for gid, g in GRID.items():
    S = s3lib.make_signal(P, U, RT, **g['sigkw'])
    exd = sorted(k + 1 for k in S['pos_ma'])
    if g['capital'] == 1e6:
        res['targets'][gid] = [dict(sig_date=x['sig_date'], month=x['month'], n_ma=x['n_ma'], n_ent=x['n_ent'], targets=x['targets'], scores=x['scores'],
                                    surge_10th=x['surge_10th'], surge_20th=x['surge_20th'], surge_30th=x['surge_30th']) for x in S['info']]
    R = {'runs': {}, 'capital': g['capital'], 'base': g['base']}
    for idle in ('cash', 'e50'):
        for cn in ('light', 'strict', 'strict_x2'):
            nav, lg, st = s2lib.simulate(P, S, U, idle=idle, cost=COSTS[cn], capital=g['capital'], e50_fix=True)
            key = f'{gid}|{idle}|{cn}'
            navs[key] = nav
            r = dict(windows=W(nav), final_nav=float(nav.iloc[-1]), extra=s3lib.extra_stats(P, nav, lg, st, NSD))
            if cn != 'strict_x2':
                r['yearly'] = stlib.yearly(nav, bench)
                r['trades_full'] = stlib.trade_stats(lg); r['trades_since2019'] = stlib.trade_stats(lg, '2019-01-01')
                F = s3lib.feats(P, S, lg, exd)
                trades.append(F.assign(run=key))
                if idle == 'cash' and cn == 'strict':
                    keep[gid] = (S if gid in ('D1', 'E1', 'E2', 'E3') else None, nav, F, st)
            R['runs'][f'{idle}|{cn}'] = r
    _, nav, lg, st = keep[gid]
    for n in (5, 10):                                         # 整段持有期禁買該格自己最賺的 n 筆
        blk, top = s2lib.block_from(lg, n)
        nb, lb, sb = s2lib.simulate(P, S, U, idle='cash', cost=COSTS['strict'], block=blk, capital=g['capital'])
        navs[f'{gid}|cash|strict|block{n}'] = nb
        R[f'block{n}'] = dict(windows=slim(W(nb)), n_dropped=int(sb['blk_dropped']),
                              top=[dict(code=x.code, ent=str(x.ent_date.date()), exit=str(x.exit_date.date()), hold=int(x.hold_days), ret=float(x.ret), pnl=float(x.pnl)) for x in top.itertuples()],
                              top_pnl_share_of_closed=float(top['pnl'].sum() / lg[lg['why'] != 'open_end']['pnl'].sum()))
    R['shift'] = {'0': slim(R['runs']['cash|strict']['windows'])}
    for s in (3, 5, 10):                                      # 換股日位移
        Ss = s3lib.make_signal(P, U, RT, shift=s, **g['sigkw'])
        ns, _, _ = s2lib.simulate(P, Ss, U, idle='cash', cost=COSTS['strict'], capital=g['capital'])
        navs[f'{gid}|cash|strict|shift{s}'] = ns
        R['shift'][str(s)] = slim(W(ns))
    res['grid'][gid] = R
    del S, Ss; gc.collect()
    w = R['runs']['cash|strict']['windows']; ts = R['runs']['cash|strict']['trades_full']
    print(f"{gid} 嚴 2019起 {w['since2019']['cagr']*100:.2f}% 全期 {w['full']['cagr']*100:.2f}% 2024起 {w['since2024']['cagr']*100:.2f}% dd {w['full']['maxdd']*100:.1f}% | 輕 {R['runs']['cash|light']['windows']['since2019']['cagr']*100:.2f}% | x2 {R['runs']['cash|strict_x2']['windows']['since2019']['cagr']*100:.2f}% | blk5 {R['block5']['windows']['since2019']['cagr']*100:.2f}% | n {ts['n']} hold {ts['avg_hold_days']:.1f} npos {R['runs']['cash|strict']['extra']['avg_npos_from_first']:.2f} | {time.time()-t_all:.0f}s", flush=True)

# ---- 分類（對 0050 八項）與相對同資金 D1
for gid, g in GRID.items():
    R = res['grid'][gid]; w = R['runs']['cash|strict']['windows']; x2 = R['runs']['cash|strict_x2']['windows']; b5 = R['block5']['windows']
    chk = dict(net_since2019=w['since2019']['diff'] > 0, net_full=w['full']['diff'] > 0, costx2_since2019=x2['since2019']['diff'] > 0, costx2_full=x2['full']['diff'] > 0,
               block5_since2019=b5['since2019']['diff'] > 0, block5_full=b5['full']['diff'] > 0, half1=w['half1']['diff'] > 0, half2=w['half2']['diff'] > 0)
    cls = '值得上XQ測' if all(chk.values()) else ('不值得' if not (chk['net_since2019'] or chk['net_full']) else '有條件')
    R['checks'] = {k: bool(v) for k, v in chk.items()}; R['class'] = cls; R['fails'] = [k for k, v in chk.items() if not v]
    b = g['base']
    if b:
        B = res['grid'][b]; wb = B['runs']['cash|strict']['windows']
        pr = {'since2016_03': s2lib.paired(keep[gid][1], keep[b][1], '2016-03'), 'since2019_01': s2lib.paired(keep[gid][1], keep[b][1], '2019-01'),
              'idle0050_since2016_03': s2lib.paired(navs[f'{gid}|e50|strict'], navs[f'{b}|e50|strict'], '2016-03'),
              'light_since2016_03': s2lib.paired(navs[f'{gid}|cash|light'], navs[f'{b}|cash|light'], '2016-03')}
        five = dict(paired_mean=pr['since2016_03']['mean'], **{k: w[k]['cagr'] - wb[k]['cagr'] for k in ('since2019', 'full', 'half1', 'half2')})
        lab = '改善' if all(v > 0 for v in five.values()) else ('變差' if all(v < 0 for v in five.values()) else '沒差')
        R['paired_vs_D1'] = pr; R['vs_D1_five'] = five; R['vs_D1'] = lab
        y1 = R['runs']['cash|strict']['yearly']; y0 = B['runs']['cash|strict']['yearly']
        R['yearly_diff_vs_D1'] = {y: y1[y]['strat'] - y0[y]['strat'] for y in y1}
        R['block_sym_diff_vs_D1'] = {f'block{n}': {k: R[f'block{n}']['windows'][k]['cagr'] - B[f'block{n}']['windows'][k]['cagr'] for k in ('since2019', 'full', 'since2024')} for n in (5, 10)}
        R['shift_diff_vs_D1'] = {s: {k: R['shift'][s][k]['cagr'] - B['shift'][s][k]['cagr'] for k in ('since2019', 'full')} for s in ('3', '5', '10')}
        R['xs'] = bool(cls != '不值得' and lab == '改善' and gid in ('E1', 'E2', 'E3'))
    print(gid, cls, R['fails'], R.get('vs_D1'), {k: round(v * 100, 3) for k, v in R.get('vs_D1_five', {}).items()}, flush=True)

# ---- 診斷 1：E1 續抱的部位（嚴、空位現金）
S, nav, lg, st = keep['E1']
d1, sg = s3lib.diag1(P, S, lg)
res['diagnostics']['1_E1_continuation'] = d1
# 參照：D1 自己的續抱（全在 L_MA 前 10 名）
dd, _ = s3lib.diag1(P, keep['D1'][0], keep['D1'][2], hold_n=10)
res['diagnostics']['1_ref_D1_continuation'] = dict(by_month_no=dd['by_month_no'], by_month_no_since2019=dd['by_month_no_since2019'], months_per_trade_hist=dd['months_per_trade_hist'], residual_segments=dd['residual_segments'])
# E3 字面規則的副作用：當月在買進名單卻被賣（不在 L_MA 前 20 名）
S3_, _, lg3, _ = keep['E3']
x = lg3[lg3['why'] == 'rule']
res['diagnostics']['E3_sold_while_target'] = int(S3_['sig'][x['exit_t'].to_numpy() - 1, x['j'].to_numpy()].sum())
# L_MA 第 10／20／30 名 SURGE 的分布（XS 出場門檻用）
inf = pd.DataFrame(S['info'])
res['diagnostics']['surge_nth_quartiles'] = {c: dict(q25=float(inf[c].quantile(0.25)), median=float(inf[c].median()), q75=float(inf[c].quantile(0.75)), min=float(inf[c].min()), max=float(inf[c].max()),
                                                     n=int(inf[c].notna().sum())) for c in ('surge_10th', 'surge_20th', 'surge_30th')}
res['diagnostics']['n_ma_per_month'] = dict(median=float(inf['n_ma'].median()), min=int(inf['n_ma'].min()), months_lt20=int((inf['n_ma'] < 20).sum()), months_lt30=int((inf['n_ma'] < 30).sum()))

# ---- 診斷 2：E2 擋掉的是什麼（嚴、空位現金）
e2x = res['grid']['E2']['runs']['cash|strict']['extra']
res['diagnostics']['2_E2_blocked'] = s3lib.diag2(keep['D1'][2], keep['E2'][2], keep['E2'][0],
                                                  dict(avg_npos_from_first=e2x['avg_npos_from_first'], avg_idle_from_first=e2x['avg_idle_from_first'],
                                                       D1_avg_npos_from_first=res['grid']['D1']['runs']['cash|strict']['extra']['avg_npos_from_first'],
                                                       D1_avg_idle_from_first=res['grid']['D1']['runs']['cash|strict']['extra']['avg_idle_from_first']))

res['meta']['rss_after_load_mb'] = rss_after_load; res['meta']['runtime_s'] = round(time.time() - t_all, 1); res['meta']['maxrss_mb'] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
(H / 'results.json').write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str))
pd.DataFrame(navs).round(2).to_csv(H / 'navs.csv.gz', index_label='date')
tr = pd.concat(trades, ignore_index=True)
tr = tr[['run', 'code', 'market', 'sig_date', 'ent_date', 'exit_date', 'hold_days', 'n_months', 'shares', 'cost', 'proceeds', 'pnl', 'ret', 'why', 'surge', 'pos_ma', 'pre20', 'close_sig', 'v60', 'near_hi240']]
num = tr.select_dtypes('number').columns
tr[num] = tr[num].round(6)
tr.to_csv(H / 'trades.csv.gz', index=False)
sg.assign(code=P.COLS[sg['j'].to_numpy()], d0=P.IDX[sg['t0'].to_numpy()], d1=P.IDX[sg['t1'].to_numpy()]).drop(columns=['j', 't0', 't1']).round({'ret': 6}).to_csv(H / 'e1_segments.csv.gz', index=False)
print(res['meta'])
