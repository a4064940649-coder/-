#!/usr/bin/env python3
"""S19：XQ 門檻版回測。7 策略（E0、XR、XU、XT、XT0、XDYN、XS3）× 2 資金格（W10_70、SAT）＝14 組合；
每個組合：主跑、20 條路徑（5 種資金 × 位移 0～3）、成本加倍、禁買最賺 5 筆。驗收 P0、A0～A3。判定照 PREREG.json。
python3 -I -B run.py → results.json、trades.csv.gz、navs.csv.gz。PREREG.json 的 sha256 必須相符。單一行程。"""
import sys, json, hashlib, time, resource, gc
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
import s19lib
from s19lib import s10lib, s4lib, s2lib, stlib, COSTS

H = Path(__file__).resolve().parent
pre = (H / 'PREREG.json').read_bytes()
sha = hashlib.sha256(pre).hexdigest()
assert sha == (H / 'PREREG.sha256').read_text().split()[0], 'PREREG.json 與 PREREG.sha256 不符'
PRE = json.loads(pre)
t_all = time.time()
rss = lambda: round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)

# ---------------------------------------------------------------- 資料
P = s19lib.load()
assert (P.T, P.N, str(P.IDX[-1].date())) == (2876, 2142, '2026-10-22'), 'P0 失敗'
fix = s19lib.e50_fix_suspension(P)
INNOV, _ = s4lib.innov_panel(P)
U = stlib.universe(P, 'twse') & ~INNOV
UX, Cadj, liqX = s19lib.script_universe(P, INNOV)
del Cadj, liqX, INNOV
S, n_sigma0 = s19lib.rev_std(P)
months = list(S.index)
avA = s19lib.a10_rows(P, months)
avB, _dl = s19lib.b11_rows(P, months, 11)
RT_A = dict(S=S, av=avA); RT_B = dict(S=S, av=avB)
for a in ('H', 'L', 'VOL', 'CHG', 'HF', 'VOLF', 'adjH', 'HI240', 'MA60', 'nbar', 'F_HI', 'IDXC', 'listed', 'NEARHI', 'V60', 'liq', 'has'):
    if hasattr(P, a):
        delattr(P, a)
gc.collect()
print('載入完成', f'{time.time()-t_all:.0f}s', rss(), 'MB', flush=True)

bench = pd.Series(P.E50[P.t0:], index=P.IDX[P.t0:])
i_mid = (P.t0 + P.T - 1) // 2
D0, DM, DE = P.IDX[P.t0], P.IDX[i_mid], P.IDX[-1]
D19 = P.IDX[P.IDX < pd.Timestamp('2019-01-01')][-1]
D24 = P.IDX[P.IDX < pd.Timestamp('2024-01-01')][-1]
WIN = {'since2019': (D19, DE), 'y2019_2023': (D19, D24), 'full': (D0, DE), 'since2024': (D24, DE), 'half1': (D0, DM), 'half2': (DM, DE)}
W = lambda nav: {k: stlib.window(nav, bench, a, b) for k, (a, b) in WIN.items()}
cg = lambda nav, k: stlib.cagr(nav, *WIN[k])
mdd = lambda nav: stlib.maxdd(nav, D0, DE)
bw = {k: dict(cagr=stlib.cagr(bench, a, b), maxdd=stlib.maxdd(bench, a, b)) for k, (a, b) in WIN.items()}
bw['seg'] = s19lib.seg_dd(bench)
YRS = (P.IDX[-1] - P.IDX[P.t0]).days / 365.25

STRATS = ['E0', 'XR', 'XU', 'XT', 'XT0', 'XDYN', 'XS3']
FUNDS = ['W10_70', 'SAT']
CAPS = {'W10_70': (6e5, 8e5, 1e6, 1.5e6, 2e6), 'SAT': (4.2e5, 5.6e5, 7e5, 1.05e6, 1.4e6)}
MAINCAP = {'W10_70': 1e6, 'SAT': 7e5}
cells = {c['id']: c for c in PRE['cells（策略）']}
DYN = cells['XDYN']['table']
DYN_MIN = {int(y): v['min_surge'] for y, v in DYN.items()}; DYN_EXIT = {int(y): v['exit_surge'] for y, v in DYN.items()}
THR = {'XU': (2.5, 2.0), 'XT': (2.5, 2.0), 'XT0': (0.0, 2.0), 'XDYN': (DYN_MIN, DYN_EXIT), 'XS3': (3.0, 2.5)}
UNIV = {'E0': U, 'XR': U, 'XU': U, 'XT': UX, 'XT0': UX, 'XDYN': UX, 'XS3': UX}


def build(sid, s):
    if sid == 'E0':
        return s10lib.make_signal_s10(P, U, RT_A, shift=s)
    if sid == 'XR':
        return s10lib.make_signal_s10(P, U, RT_B, shift=s)
    ms, es = THR[sid]
    if sid == 'XU':
        return s19lib.make_signal_thr(P, U, U, U, S, avB, ms, es, shift=s)
    return s19lib.make_signal_thr(P, U, UX, None, S, avB, ms, es, shift=s)


def sim(sid, Sg, fund, capital=None, cost='strict', block=None):
    if fund == 'W10_70':
        return s10lib.simulate_s10(P, Sg, UNIV[sid], sat_w=0.7, sizing='W10', lock=False, cost=COSTS[cost], block=block, capital=capital or 1e6)
    return s10lib.simulate_s10(P, Sg, UNIV[sid], sat_w=1.0, sizing='W10', lock=False, cost=COSTS[cost], block=block, capital=capital or 7e5, sweep_min=float('inf'))


def slim(w, keys):
    return {k: dict(cagr=w[k]['cagr'], diff=w[k]['diff'], maxdd=w[k]['maxdd']) for k in keys}


def holdings_at(lg, rows):
    """每個成交日（訊號列＋1）收盤後的持股檔數。"""
    e = lg['ent_t'].to_numpy(); x = lg['exit_t'].to_numpy(); oe = (lg['why'] == 'open_end').to_numpy()
    out = []
    for k in rows:
        t = k + 1
        if t >= P.T:
            continue
        out.append(int(((e <= t) & ((x > t) | oe)).sum()))
    return out


# ---------------------------------------------------------------- 驗收 A0、A1、A2
acc = {'P0': True, 'e50_fix': fix, 'n_sigma0': int(n_sigma0)}
S_E0 = build('E0', 0)
S4 = s4lib.make_signal(P, U, RT_A, shift=0, depth=10, mom_lo=0.10, exit='top20', keep_info=True)
acc['A0_signal_same_as_s4lib'] = bool(np.array_equal(S_E0['sig'], S4['sig']) and np.array_equal(S_E0['X'], S4['X'])
                                      and np.array_equal(np.nan_to_num(S_E0['rank'], nan=-9e9), np.nan_to_num(S4['rank'], nan=-9e9)))
del S4; gc.collect()
n, l, _ = sim('E0', S_E0, 'W10_70')
acc['A0'] = dict(since2019=cg(n, 'since2019'), full=cg(n, 'full'), maxdd=mdd(n))
acc['A0']['ok'] = bool(abs(acc['A0']['since2019'] - 0.48126) < 0.00025 and abs(acc['A0']['full'] - 0.34082) < 0.00025 and abs(acc['A0']['maxdd'] + 0.33404) < 0.00025)
S_XR = build('XR', 0)
n, l, _ = sim('XR', S_XR, 'W10_70')
acc['A1'] = dict(since2019=cg(n, 'since2019'), full=cg(n, 'full'), maxdd=mdd(n))
acc['A1']['ok'] = bool(abs(acc['A1']['since2019'] - 0.4948) < 0.0010 and abs(acc['A1']['full'] - 0.3524) < 0.0010 and abs(acc['A1']['maxdd'] + 0.3006) < 0.0010)
nthB = s19lib.lma_nth(P, U, S, avB)
d10 = {r.month: (r.s10 if np.isfinite(r.s10) else -1e18) for r in nthB.itertuples()}
d20 = {r.month: (r.s20 if np.isfinite(r.s20) else -1e18) for r in nthB.itertuples()}
S_a2 = s19lib.make_signal_thr(P, U, U, U, S, avB, d10, d20, shift=0)
acc['A2'] = dict(sig_same=bool(np.array_equal(S_a2['sig'], S_XR['sig'])), X_same=bool(np.array_equal(S_a2['X'], S_XR['X'])),
                 sig_diff_cells=int((S_a2['sig'] != S_XR['sig']).sum()), X_diff_cells=int((S_a2['X'] != S_XR['X']).sum()))
del S_a2, n, l; gc.collect()
print('驗收', {k: v for k, v in acc.items() if k != 'e50_fix'}, flush=True)
assert acc['A0_signal_same_as_s4lib'] and acc['A0']['ok'] and acc['A1']['ok'] and acc['A2']['sig_same'] and acc['A2']['X_same'], ('驗收失敗', acc)

# 位移 0 的 XR 名單與 X（名單差異、S5 用）
XR_tg = {x['month']: set(x['targets']) for x in S_XR['info']}
XR_k = {x['month']: x['k'] for x in S_XR['info']}
XR_X = {x['k']: S_XR['X'][x['k']].copy() for x in S_XR['info']}
del S_E0, S_XR; gc.collect()

# ---------------------------------------------------------------- 14 組合
res = {'PREREG_sha256': sha, 'acceptance': acc,
       'meta': dict(panel_T=P.T, panel_N=P.N, start=str(D0.date()), mid=str(DM.date()), since2019_from=str(D19.date()), since2024_from=str(D24.date()), end=str(DE.date())),
       'bench_0050': bw, 'combos': {}, 'lists': {}}
navs = {'0050': bench}; trades = []
for sid in STRATS:
    for f in FUNDS:
        res['combos'][f'{sid}|{f}'] = dict(strat=sid, fund=f, paths={}, paths_1923={}, paths_24={}, paths_dd={}, paths_seg={})
keep_lg = {}
for s in range(4):
    tc = time.time()
    for sid in STRATS:
        Sg = build(sid, s)
        if s == 0:
            info = Sg['info']
            L = dict(n_months=len(info), months=[x['month'] for x in info], sig_dates=[x['sig_date'] for x in info], n_cand=[int(x['n_tgt']) for x in info])
            if sid not in ('E0', 'XR'):
                L['min_s'] = [x['min_s'] for x in info]; L['exit_s'] = [x['exit_s'] for x in info]; L['n_keep'] = [x['n_keep'] for x in info]
            inter, cov_n, cov_d, jac = [], 0, 0, []
            for x in info:
                a = set(x['targets']); b = XR_tg.get(x['month'], set())
                inter.append(len(a & b)); cov_n += len(a & b); cov_d += len(b)
                if a | b:
                    jac.append(len(a & b) / len(a | b))
            L['inter_with_XR'] = inter
            L['share_XR_in_list'] = float(cov_n / max(cov_d, 1)); L['jaccard_mean'] = float(np.mean(jac)); L['n_jaccard_months'] = len(jac)
            L['n_cand_mean'] = float(np.mean(L['n_cand'])); L['n_cand_median'] = float(np.median(L['n_cand'])); L['n_cand_max'] = int(max(L['n_cand']))
            L['months_cand_gt10'] = int(sum(1 for v in L['n_cand'] if v > 10)); L['months_cand_0'] = int(sum(1 for v in L['n_cand'] if v == 0))
            L['targets'] = {x['month']: x['targets'] for x in info}
            res['lists'][sid] = L
            if sid == 'XT':
                XT_X = {x['k']: Sg['X'][x['k']].copy() for x in info}
                XT_k = {x['month']: x['k'] for x in info}
        for f in FUNDS:
            R = res['combos'][f'{sid}|{f}']
            for cap in CAPS[f]:
                key = f'{int(cap/1e4)}萬|+{s}'
                nav, lg, st = sim(sid, Sg, f, capital=cap)
                R['paths'][key] = cg(nav, 'since2019'); R['paths_1923'][key] = cg(nav, 'y2019_2023'); R['paths_24'][key] = cg(nav, 'since2024')
                R['paths_dd'][key] = mdd(nav); R['paths_seg'][key] = s19lib.seg_dd(nav)
                if s == 0 and cap == MAINCAP[f]:
                    name = f'{sid}|{f}'
                    navs[name] = nav
                    w = W(nav)
                    R['windows'] = slim(w, tuple(WIN))
                    R['sharpe'] = {k: w[k]['sharpe'] for k in ('full', 'since2019')}
                    R['t_month_excess'] = {k: w[k]['t_month_excess'] for k in ('full', 'since2019')}
                    R['dd_dates'] = s10lib.dd_dates(nav); R['seg'] = s19lib.seg_dd(nav)
                    R['yearly'] = stlib.yearly(nav, bench)
                    R['trades_full'] = stlib.trade_stats(lg); R['trades_since2019'] = stlib.trade_stats(lg, '2019-01-01')
                    R['n_buy'] = int(st['n_buy']); R['noslot'] = int(st['noslot']); R['avg_npos'] = float(st['avg_npos']); R['blk_min'] = int(st['blk_min'])
                    R['blk_limup'] = int(st['blk_limup']); R['skip_cash'] = int(st['skip_cash'])
                    mnav = float(nav.mean())
                    R['turnover_oneway_ann'] = float((st['buy_val'] + st['sell_val']) / 2 / mnav / YRS)
                    R['cost_ann_pct_of_nav'] = float((st['fee'] + st['tax'] + st['slip'] + st['e50_cost']) / mnav / YRS)
                    R['avg_stockfrac'] = float(np.nanmean(st['stockfrac'].to_numpy()))
                    hk = holdings_at(lg, [x['k'] for x in Sg['info']])
                    R['holdings_at_exec'] = dict(mean=float(np.mean(hk)), median=float(np.median(hk)), dist={int(a): int(b) for a, b in zip(*np.unique(hk, return_counts=True))},
                                                 by_month=dict(zip([x['month'] for x in Sg['info']][:len(hk)], hk)))
                    x = lg.copy(); x['run'] = name
                    trades.append(x[['run', 'code', 'sig_date', 'ent_date', 'exit_date', 'hold_days', 'shares', 'cost', 'proceeds', 'pnl', 'ret', 'why', 'score']])
                    if f == 'W10_70' and sid in ('XT', 'XR', 'E0'):
                        keep_lg[sid] = lg.copy()
                    nx2 = sim(sid, Sg, f, capital=cap, cost='strict_x2')[0]
                    R['costx2'] = slim(W(nx2), ('since2019', 'full'))
                    blk, top = s2lib.block_from(lg, 5)
                    nb = sim(sid, Sg, f, capital=cap, block=blk)[0]
                    R['block5'] = slim(W(nb), ('since2019', 'full'))
                    R['block5_top'] = [dict(code=y.code, ent=str(y.ent_date.date()), exit=str(y.exit_date.date()), ret=float(y.ret), pnl=float(y.pnl)) for y in top.itertuples()]
                    del nx2, nb
                del nav, lg, st
        del Sg; gc.collect()
    print('位移', s, f'{time.time()-tc:.0f}s 累計 {time.time()-t_all:.0f}s {rss()}MB', flush=True)

# ---------------------------------------------------------------- A3（XT｜W10_70 主跑）
lgT = keep_lg['XT']
pos_max = max(holdings_at(lgT, list(range(P.t0, P.T - 1)))) if len(lgT) else 0
bad_buy = 0; bad_exit = 0
k2m_T = {k: m for m, k in XT_k.items()}
for r in lgT.itertuples():
    k = int(r.sig_t); j = int(r.j); m = k2m_T.get(k)
    if m is None:
        bad_buy += 1; continue
    sv = S.loc[pd.Period(m, 'M')].to_numpy()[j]
    if not (np.isfinite(sv) and sv >= 2.5 and P.RET20[k, j] > 0.10 and P.F_MA60[k, j] and UX[k, j]):
        bad_buy += 1
    if r.why == 'rule':
        kk = max(q for q in XT_X if q < r.exit_t)
        sv2 = S.loc[pd.Period(k2m_T[kk], 'M')].to_numpy()[j]
        if not XT_X[kk][j] or not ((not np.isfinite(sv2)) or sv2 < 2.0 or not P.F_MA60[kk, j]):
            bad_exit += 1
acc['A3'] = dict(max_positions=int(pos_max), bad_buy=int(bad_buy), bad_exit=int(bad_exit), n_trades=int(len(lgT)), ok=bool(pos_max <= 10 and bad_buy == 0 and bad_exit == 0))
print('驗收 A3', acc['A3'], flush=True)
assert acc['A3']['ok'], ('驗收 A3 失敗', acc['A3'])

# ---------------------------------------------------------------- S5：多買、多續抱、少買（XT｜W10_70 主跑）
exec_rows = sorted(XT_k.values())
nxt = {k: (exec_rows[i + 1] if i + 1 < len(exec_rows) else None) for i, k in enumerate(exec_rows)}
k2m = {k: m for m, k in XT_k.items()}


def fwd(k, j):
    """訊號列 k 的次一交易日開盤 → 下一個訊號列的次一交易日開盤：還原開盤報酬 − 0050 還原開盤報酬。最後一個月不算。"""
    k2 = nxt.get(k)
    if k2 is None or k2 + 1 >= P.T:
        return None
    a, b = k + 1, k2 + 1
    r = P.AO[b, j] / P.AO[a, j] - 1; r0 = P.E50O[b] / P.E50O[a] - 1
    return float(r - r0) if np.isfinite(r) and np.isfinite(r0) else None


def tstat(v):
    v = np.array([x for x in v if x is not None], float)
    if len(v) < 3:
        return dict(n=int(len(v)), mean=float(v.mean()) if len(v) else float('nan'), t=float('nan'))
    return dict(n=int(len(v)), mean=float(v.mean()), median=float(np.median(v)), t=float(v.mean() / (v.std(ddof=1) / np.sqrt(len(v)))), share_pos=float((v > 0).mean()))


extra_buy, extra_buy_trade, same_buy = [], [], []
for r in lgT.itertuples():
    k = int(r.sig_t); m = k2m.get(k)
    if m is None:
        continue
    if r.code in XR_tg.get(m, set()):
        same_buy.append(fwd(k, int(r.j)))
    else:
        extra_buy.append(fwd(k, int(r.j))); extra_buy_trade.append(float(r.ret))
e = lgT['ent_t'].to_numpy(); x = lgT['exit_t'].to_numpy(); oe = (lgT['why'] == 'open_end').to_numpy(); jj = lgT['j'].to_numpy()
extra_keep, both_keep, xt_sell_xr_keep = [], [], []
miss_buy = []
for k in exec_rows:
    held = jj[(e <= k) & ((x > k) | oe)]
    xr = XR_X.get(k)
    for j in held:
        if xr is None:
            continue
        if not XT_X[k][j] and xr[j]:
            extra_keep.append(fwd(k, int(j)))
        elif not XT_X[k][j] and not xr[j]:
            both_keep.append(fwd(k, int(j)))
        elif XT_X[k][j] and not xr[j]:
            xt_sell_xr_keep.append(fwd(k, int(j)))
    m = k2m[k]
    bought = set(lgT.loc[lgT['sig_t'] == k, 'code'])
    heldc = set(P.COLS[held])
    for c in XR_tg.get(m, set()):
        if c not in bought and c not in heldc:
            miss_buy.append(fwd(k, int(P.ci[c])))
res['S5'] = dict(extra_buy=tstat(extra_buy), extra_buy_trade_ret=tstat(extra_buy_trade), same_buy=tstat(same_buy), extra_keep=tstat(extra_keep),
                 both_keep=tstat(both_keep), xt_sell_xr_keep=tstat(xt_sell_xr_keep), miss_buy=tstat(miss_buy),
                 note='每筆＝次一交易日開盤到下一個成交日開盤（還原開盤）減同期 0050；t 是簡單 t（沒有按月份聚類）。same_buy、both_keep、xt_sell_xr_keep 是對照。')

# ---------------------------------------------------------------- 彙整、八項、判定
for key, R in res['combos'].items():
    pv = np.array(list(R['paths'].values())); pdd = np.array(list(R['paths_dd'].values()))
    assert len(pv) == 20
    R['paths20'] = dict(min=float(pv.min()), median=float(np.median(pv)), max=float(pv.max()), n_above_0050=int((pv > bw['since2019']['cagr']).sum()),
                        median_1923=float(np.median(list(R['paths_1923'].values()))), median_24=float(np.median(list(R['paths_24'].values()))),
                        maxdd_min=float(pdd.min()), maxdd_median=float(np.median(pdd)), maxdd_max=float(pdd.max()),
                        seg_median={sg: float(np.median([v[sg] for v in R['paths_seg'].values()])) for sg in s19lib.SEGMENTS})
    w = R['windows']; x2 = R['costx2']; b5 = R['block5']
    chk = dict(net_since2019=w['since2019']['diff'] > 0, net_full=w['full']['diff'] > 0, costx2_since2019=x2['since2019']['diff'] > 0, costx2_full=x2['full']['diff'] > 0,
               block5_since2019=b5['since2019']['diff'] > 0, block5_full=b5['full']['diff'] > 0, half1=w['half1']['diff'] > 0, half2=w['half2']['diff'] > 0)
    R['checks'] = {k: bool(v) for k, v in chk.items()}
    R['class'] = '值得上XQ測' if all(chk.values()) else ('不值得' if not (chk['net_since2019'] or chk['net_full']) else '有條件')
    R['fails'] = [k for k, v in chk.items() if not v]


def pdiff(a, b, f, fld='paths'):
    A = res['combos'][f'{a}|{f}'][fld]; B = res['combos'][f'{b}|{f}'][fld]
    return np.array([A[k] - B[k] for k in A]) * 100


def summ(d):
    return dict(median=float(np.median(d)), npos=int((d > 0).sum()), nneg=int((d < 0).sum()), min=float(d.min()), max=float(d.max()))


def j1(d):
    med = float(np.median(d)); same = max(int((d > 0).sum()), int((d < 0).sum()))
    if abs(med) >= 3:
        return ('明顯不同（' + ('較高' if med > 0 else '較低') + '）') if same >= 15 else '明顯不同但方向不穩'
    if abs(med) >= 1:
        return '略有不同（' + ('較高' if med > 0 else '較低') + '）'
    return '接近'


pairs = {}
for f in FUNDS:
    for a in STRATS[1:]:
        for b in ('E0', 'XR', 'XU', 'XT'):
            if a == b:
                continue
            d19, d1923, d24 = pdiff(a, b, f), pdiff(a, b, f, 'paths_1923'), pdiff(a, b, f, 'paths_24')
            ddd = pdiff(a, b, f, 'paths_dd')
            Ra, Rb = res['combos'][f'{a}|{f}'], res['combos'][f'{b}|{f}']
            seg = {}
            for sg in s19lib.SEGMENTS:
                ds = np.array([Ra['paths_seg'][k][sg] - Rb['paths_seg'][k][sg] for k in Ra['paths_seg']]) * 100
                med = float(np.median(ds)); nsh = int((ds > 0).sum()); ndp = int((ds < 0).sum())
                v = '較淺' if (med >= 1 and nsh >= 15) else ('較深' if (med <= -1 and ndp >= 15) else '差不多')
                seg[sg] = dict(median=med, n_shallower=nsh, n_deeper=ndp, verdict=v)
            pairs[f'{a}-{b}|{f}'] = dict(since2019=summ(d19), y2019_2023=summ(d1923), since2024=summ(d24), maxdd=summ(ddd), seg=seg,
                                         J1=j1(d19), J1_1923=j1(d1923), J1_24=j1(d24),
                                         main_diff_pp=dict(since2019=(Ra['windows']['since2019']['cagr'] - Rb['windows']['since2019']['cagr']) * 100,
                                                           full=(Ra['windows']['full']['cagr'] - Rb['windows']['full']['cagr']) * 100,
                                                           maxdd=(Ra['windows']['full']['maxdd'] - Rb['windows']['full']['maxdd']) * 100))
res['pairs'] = pairs
verdict = {}
for f in FUNDS:
    p = pairs[f'XT-E0|{f}']
    comp = {'訊號日（XR−E0）': pairs[f'XR-E0|{f}']['since2019']['median'], '門檻（XU−XR）': pairs[f'XU-XR|{f}']['since2019']['median'], '腳本細節（XT−XU）': pairs[f'XT-XU|{f}']['since2019']['median']}
    main = max(comp, key=lambda k: abs(comp[k]))
    verdict[f] = dict(J1=p['J1'], J1_1923=p['J1_1923'], J1_24=p['J1_24'], XT_minus_E0=p['since2019'], decomposition=comp,
                      J2=(main if abs(comp[main]) >= 1 else '三者都小（< 1 個百分點）'), J3={sg: v['verdict'] for sg, v in p['seg'].items()})
# J4、J5（W10_70）
cl = lambda sid, fld='since2019': abs(pairs[f'{sid}-E0|W10_70'][fld]['median'])
jac = {sid: res['lists'][sid]['jaccard_mean'] for sid in ('XT', 'XT0', 'XDYN', 'XS3', 'XU')}
j4 = (cl('XDYN') <= cl('XT') - 1) and (cl('XDYN', 'y2019_2023') <= cl('XT', 'y2019_2023')) and (cl('XDYN', 'since2024') <= cl('XT', 'since2024')) and (jac['XDYN'] >= jac['XT'])
closeness = {sid: cl(sid) for sid in ('XT', 'XT0', 'XDYN', 'XS3')}
best = min(closeness, key=closeness.get)
rec = best if (best != 'XT' and closeness[best] <= closeness['XT'] - 1 and jac[best] >= jac['XT'] - 0.05) else 'XT'
verdict['J4_dynamic'] = dict(verdict='建議改動態' if j4 else '不必改動態', closeness_XDYN=cl('XDYN'), closeness_XT=cl('XT'),
                             closeness_1923=dict(XDYN=cl('XDYN', 'y2019_2023'), XT=cl('XT', 'y2019_2023')), closeness_24=dict(XDYN=cl('XDYN', 'since2024'), XT=cl('XT', 'since2024')), jaccard=jac)
verdict['J5_setting'] = dict(recommend=rec, closeness=closeness, jaccard=jac, warn_XT0=bool(cl('XT0') >= 3))
verdict['J6_expect'] = {f: dict(main=res['combos'][f'XT|{f}']['windows'], paths20=res['combos'][f'XT|{f}']['paths20']) for f in FUNDS}
verdict['J7_class'] = {k: R['class'] for k, R in res['combos'].items()}
verdict['J7_fails'] = {k: R['fails'] for k, R in res['combos'].items() if R['fails']}
res['verdict'] = verdict
print('判定', json.dumps({k: v for k, v in verdict.items() if k != 'J6_expect'}, ensure_ascii=False, default=str), flush=True)
for k, R in res['combos'].items():
    w = R['windows']; p = R['paths20']
    print(k, 'main %.2f/%.2f/%.2f/%.2f' % (w['since2019']['cagr'] * 100, w['full']['cagr'] * 100, w['since2024']['cagr'] * 100, w['full']['maxdd'] * 100),
          'paths %.2f/%.2f/%.2f dd %.2f' % (p['min'] * 100, p['median'] * 100, p['max'] * 100, p['maxdd_median'] * 100), 'npos %.2f' % R['avg_npos'], R['class'], flush=True)
print('S5', json.dumps(res['S5'], ensure_ascii=False), flush=True)

res['meta']['runtime_s'] = round(time.time() - t_all, 1); res['meta']['maxrss_mb'] = rss()
res['meta']['N'] = PRE['data_snooping']
for R in res['combos'].values():
    for k in ('paths_seg',):
        R[k] = {kk: {a: round(b, 5) for a, b in v.items()} for kk, v in R[k].items()}
(H / 'results.json').write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str))
pd.DataFrame(navs).round(0).to_csv(H / 'navs.csv.gz', index_label='date')
tr = pd.concat(trades, ignore_index=True)
num = tr.select_dtypes('number').columns
tr[num] = tr[num].round(4)
for c in ('sig_date', 'ent_date', 'exit_date'):
    tr[c] = pd.to_datetime(tr[c]).dt.strftime('%Y-%m-%d')
tr.to_csv(H / 'trades.csv.gz', index=False)
print(res['meta']['runtime_s'], 's', res['meta']['maxrss_mb'], 'MB')
