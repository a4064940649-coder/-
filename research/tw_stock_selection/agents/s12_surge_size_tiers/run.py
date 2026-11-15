#!/usr/bin/env python3
"""S12：SURGE 在不同市值層。交易格 CUR、TL、TM、TS、Q55（大 5＋中小 5，兩子帳戶）、TL_ADV（敏感度），全部 W10_70、E0（層內版）、A10 開盤；
每格 100 萬主跑、20 條路徑（5 種資金 × 位移 0～3）、成本加倍、禁買最賺 5 筆；同層等權／名單等權（扣成本前）與 S11 複製員拆法。
python3 -I -B run.py → results.json、trades.csv.gz、navs.csv.gz。PREREG.json 的 sha256 必須相符。單行程。"""
import sys, json, hashlib, time, resource, gc, math
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
import s12lib
from s12lib import s10lib, s9lib, s4lib, s2lib, stlib, COSTS

H = Path(__file__).resolve().parent
pre = (H / 'PREREG.json').read_bytes()
sha = hashlib.sha256(pre).hexdigest()
assert sha == (H / 'PREREG.sha256').read_text().split()[0], 'PREREG.json 與 PREREG.sha256 不符'
PRE = json.loads(pre)
t_all = time.time()
rss = lambda: round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
P = s12lib.load()
assert (P.T, P.N, str(P.IDX[-1].date())) == (2876, 2142, '2026-10-22'), 'P0 失敗'
U0 = stlib.universe(P, 'twse')
RT = s2lib.rev_tables(P)
INNOV, _ = s4lib.innov_panel(P)
U = U0 & ~INNOV
del U0, INNOV
SH = s12lib.load_shares(P)
MC, MCSRC = s12lib.mcap_panel(P, SH, s12lib.LAG)
del SH, MCSRC
ADV = P.V60.copy()
for a in ('H', 'L', 'VOL', 'CHG', 'HF', 'VOLF', 'adjH', 'HI240', 'MA60', 'nbar', 'F_HI', 'IDXC', 'AO', 'listed', 'NEARHI', 'V60', 'liq', 'F_MA20'):
    if hasattr(P, a):
        delattr(P, a)
gc.collect()
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
KW4 = dict(depth=10, mom_lo=0.10, exit='top20', keep_info=True)
bw = {k: dict(cagr=stlib.cagr(bench, a, b), maxdd=stlib.maxdd(bench, a, b)) for k, (a, b) in WIN.items()}
bw['seg'] = s12lib.seg_dd(bench)
B19 = bw['since2019']['cagr']
YRS = (P.IDX[-1] - P.IDX[P.t0]).days / 365.25
CELLS = ['CUR', 'TL', 'TM', 'TS', 'Q55', 'TL_ADV']
SIGSPEC = {'CUR': (None, None), 'L': ('MC', {0}), 'M': ('MC', {1}), 'S': ('MC', {2}), 'MS': ('MC', {1, 2}), 'L_ADV': ('ADV', {0})}
CELL_SIG = {'CUR': 'CUR', 'TL': 'L', 'TM': 'M', 'TS': 'S', 'TL_ADV': 'L_ADV'}


def same_trades(l1, l2):
    a = l1[['code', 'ent_t', 'exit_t', 'why']].reset_index(drop=True); b = l2[['code', 'ent_t', 'exit_t', 'why']].reset_index(drop=True)
    return bool(len(a) == len(b) and (a == b).all().all())


def slim(w, keys=('since2019', 'full', 'since2024', 'half1', 'half2', 'y2019_2023')):
    return {k: dict(cagr=w[k]['cagr'], diff=w[k]['diff'], maxdd=w[k]['maxdd']) for k in keys}


acc = {'P0': True}


def build(s):
    out = {}
    for key, (vk, ts) in SIGSPEC.items():
        VAL = {'MC': MC, 'ADV': ADV, None: None}[vk]
        out[key] = s12lib.make_signal_tier(P, U, RT, VAL, ts, shift=s)
    # A1：全股池版＝s4lib
    S4 = s4lib.make_signal(P, U, RT, shift=s, **KW4)
    C = out['CUR']
    acc[f'A1_shift{s}'] = bool(np.array_equal(C['sig'], S4['sig']) and np.array_equal(C['X'], S4['X'])
                               and np.array_equal(np.nan_to_num(C['rank'], nan=-9e9), np.nan_to_num(S4['rank'], nan=-9e9))
                               and [x['k'] for x in C['info']] == [x['k'] for x in S4['info']])
    del S4
    # A2：層內續抱＝層內前 20；買進名單屬於該層
    a2 = {}
    for key, (vk, ts) in SIGSPEC.items():
        S = out[key]; bad_in = 0
        if ts is not None:
            for k, tr in S['TIER'].items():
                js = np.flatnonzero(S['sig'][k])
                bad_in += int((~np.isin(tr[js], list(ts))).sum())
        a2[key] = dict(mismatch=int(S['A2_mismatch']), buy_outside_tier=bad_in)
    acc[f'A2_shift{s}'] = a2
    assert acc[f'A1_shift{s}'] and all(v['mismatch'] == 0 and v['buy_outside_tier'] == 0 for v in a2.values()), ('驗收 A1／A2 失敗', s, acc)
    for S in out.values():
        S.pop('A2_mismatch', None)
    gc.collect()
    return out


def sim(S, capital=1e6, cost='strict', block=None, slots=10, sizing='W10', **kw):
    return s10lib.simulate_s10(P, S, U, sat_w=0.7, sizing=sizing, lock=False, cost=COSTS[cost], capital=capital, slots=slots, block=block, **kw)


def run_cell(cid, SG, capital=1e6, cost='strict', block=None):
    """回傳 (nav, lg, st_list)。Q55＝兩子帳戶（各 capital/2、5 檔、每檔＝子帳戶衛星 ÷ 5）淨值相加。"""
    if cid != 'Q55':
        n, l, st = sim(SG[CELL_SIG[cid]], capital, cost, block)
        l['sleeve'] = cid
        return n, l, [st]
    nL, lL, sL = sim(s12lib.as_w5(SG['L']), capital / 2, cost, block, slots=5, sizing='WN')
    nM, lM, sM = sim(s12lib.as_w5(SG['MS']), capital / 2, cost, block, slots=5, sizing='WN')
    lL['sleeve'] = 'Q55_L'; lM['sleeve'] = 'Q55_MS'
    return nL + nM, pd.concat([lL, lM], ignore_index=True), [sL, sM, nL, nM]


SIG = {0: build(0)}
SG0 = SIG[0]
# A0
n9, l9, _ = s9lib.simulate_s9(P, SG0['CUR'], U, sat_w=0.7, sizing='W10', lock=False, cost=COSTS['strict'], exec_mode='open')
n10, l10, _ = sim(SG0['CUR'])
acc['A0'] = dict(since2019=cg(n10, 'since2019'), full=cg(n10, 'full'), maxdd=mdd(n10),
                 nav_identical=bool(np.allclose(n9.to_numpy(), n10.to_numpy(), rtol=0, atol=1e-6)), trades_identical=same_trades(l9, l10))
acc['A0']['ok'] = bool(abs(acc['A0']['since2019'] - 0.4813) < 0.00025 and abs(acc['A0']['full'] - 0.3408) < 0.00025 and abs(acc['A0']['maxdd'] + 0.3340) < 0.00025
                       and acc['A0']['nav_identical'] and acc['A0']['trades_identical'])
# A3
S10x = dict(SG0['CUR']); S10x['info'] = [dict(x, n_tgt=10) for x in SG0['CUR']['info']]
nA3, lA3, _ = sim(S10x, sizing='WN')
acc['A3'] = bool(np.allclose(n10.to_numpy(), nA3.to_numpy(), rtol=0, atol=1e-6) and same_trades(l10, lA3))
del n9, l9, n10, l10, nA3, lA3, S10x
assert acc['A0']['ok'] and acc['A3'], ('驗收 A0／A3 失敗', acc)
print('驗收 A0～A3 通過', {k: v for k, v in acc.items()}, flush=True)

# ---------------------------------------------------------------- 同層等權／名单等權（扣成本前）
E50O = P.E50O


def stock_ret(e, e2, js):
    """執行日 e 開盤買、e2 開盤賣（e2＝None：持有到最後一天收盤）的含息報酬；e 沒有開盤價的剔除。"""
    js = np.asarray(js, int)
    if len(js) == 0:
        return np.array([]), js
    o = P.O[e, js]
    ok = np.isfinite(o) & (o > 0)
    js = js[ok]; o = o[ok]
    r = P.CF[e, js] / o
    if e2 is None:
        r = r * P.G[P.T - 1, js] / P.G[e, js]
    else:
        r = r * P.G[e2 - 1, js] / P.G[e, js]
        ox = P.O[e2, js] / P.REF[e2, js]
        r = r * np.where(np.isfinite(ox) & (ox > 0), ox, 1.0)
    return r - 1, js


def ew_tables(SGs, tier_keys):
    """每個訊號月：0050、各層同層等權、名單等權、名單 W10（每檔 1/10、空位 0050）。"""
    info = SGs['CUR']['info']
    ks = [x['k'] for x in info]
    rows = []
    for i, k in enumerate(ks):
        e = k + 1
        e2 = ks[i + 1] + 1 if i + 1 < len(ks) else None
        if e >= P.T:
            break
        r50 = (E50O[e2] if e2 is not None else P.E50[P.T - 1]) / E50O[e] - 1
        row = dict(k=k, e=e, e2=e2 if e2 is not None else P.T - 1, date=P.IDX[e], r50=r50)
        for tk in tier_keys:
            S = SGs[tk]
            if tk == 'CUR':
                mem = np.flatnonzero(U[k])
            else:
                vk, ts = SIGSPEC[tk]
                mem = np.flatnonzero(U[k] & np.isin(S['TIER'][k], list(ts)))
            rew, _ = stock_ret(e, e2, mem)
            lst = np.flatnonzero(S['sig'][k])
            rl, jl = stock_ret(e, e2, lst)
            ew = float(rew.mean()) if len(rew) else 0.0
            li = float(rl.mean()) if len(rl) else ew
            n = min(len(rl), 10)
            w10 = float(rl[:n].sum() / 10 + (1 - n / 10) * r50) if n else r50
            row.update({f'ew_{tk}': ew, f'list_{tk}': li, f'w10_{tk}': w10, f'n_{tk}': int(len(rl)), f'nmem_{tk}': int(len(rew))})
        rows.append(row)
    return pd.DataFrame(rows)


def ann(rets, d0, d1):
    yrs = (d1 - d0).days / 365.25
    return float(np.prod(1 + np.asarray(rets)) ** (1 / yrs) - 1) if yrs > 0 and len(rets) else float('nan')


PERIODS = {'full': ('2000-01-01', '2100-01-01'), 'since2019': ('2019-01-01', '2100-01-01'), 'y2019_2023': ('2019-01-01', '2023-12-31'), 'since2024': ('2024-01-01', '2100-01-01')}


def period_slices(df):
    out = {}
    for p, (a, b) in PERIODS.items():
        m = (df['date'] >= pd.Timestamp(a)) & (df['date'] <= pd.Timestamp(b))
        idx = np.flatnonzero(m.to_numpy())
        i0, i1 = idx[0], idx[-1]
        d0 = df['date'].iloc[i0]
        e_end = int(df['e2'].iloc[i1])
        d1 = P.IDX[e_end]
        out[p] = (idx, d0, d1, int(df['e'].iloc[i0]), e_end, i1 == len(df) - 1)
    return out


def decompose(df, tk, nav):
    """S11 複製員拆法（30/70 每時窗再平衡、扣成本前）＋實際淨值。"""
    sl = period_slices(df)
    out = {}
    for p, (idx, d0, d1, e0, e_end, last) in sl.items():
        sub = df.iloc[idx]
        r50 = ann(sub['r50'], d0, d1)
        a_ew = ann(0.3 * sub['r50'] + 0.7 * sub[f'ew_{tk}'], d0, d1)
        a_li = ann(0.3 * sub['r50'] + 0.7 * sub[f'list_{tk}'], d0, d1)
        a_w10 = ann(0.3 * sub['r50'] + 0.7 * sub[f'w10_{tk}'], d0, d1)
        v0 = nav.loc[P.IDX[e0 - 1]]
        v1 = nav.loc[P.IDX[P.T - 1]] if last else nav.loc[P.IDX[e_end - 1]]
        d1n = P.IDX[P.T - 1] if last else P.IDX[e_end - 1]
        act = float((v1 / v0) ** (1 / ((d1n - P.IDX[e0 - 1]).days / 365.25)) - 1)
        out[p] = dict(start=str(d0.date()), end=str(d1.date()), e0050=r50, ew_raw=ann(sub[f'ew_{tk}'], d0, d1), list_raw=ann(sub[f'list_{tk}'], d0, d1),
                      w3070_ew=a_ew, w3070_list=a_li, w3070_w10=a_w10, actual=act,
                      tier_lag=(a_ew - r50) * 100, selection=(a_li - a_ew) * 100, execution=(act - a_li) * 100,
                      exec_w10_dilution=(a_w10 - a_li) * 100, exec_rest=(act - a_w10) * 100, total=(act - r50) * 100)
    return out


def selection_stats(df, tk):
    x = (df[f'list_{tk}'] - df[f'ew_{tk}']).to_numpy()
    sl = period_slices(df)
    idx, d0, d1, *_ = sl['full']
    a = ann(df[f'list_{tk}'], d0, d1) - ann(df[f'ew_{tk}'], d0, d1)
    t = float(x.mean() / x.std(ddof=1) * math.sqrt(len(x)))
    yr = {}
    for y, g in df.groupby(df['date'].dt.year):
        yr[int(y)] = float(np.prod(1 + g[f'list_{tk}']) - np.prod(1 + g[f'ew_{tk}']))
    wins = sum(1 for v in yr.values() if v > 0)
    v = '選股超額存在' if (a > 0 and wins >= 7 and t >= 2) else ('弱' if a > 0 else '不存在')
    per = {p: (ann(df.iloc[ix][f'list_{tk}'], a0, a1) - ann(df.iloc[ix][f'ew_{tk}'], a0, a1)) * 100 for p, (ix, a0, a1, *_r) in sl.items()}
    return dict(ann_excess_pp=a * 100, t_month=t, mean_month_pp=float(x.mean()) * 100, yearly_pp={k: v * 100 for k, v in yr.items()}, years_positive=wins, n_years=len(yr),
                months_empty_list=int((df[f'n_{tk}'] == 0).sum()), avg_list=float(df[f'n_{tk}'].mean()), avg_members=float(df[f'nmem_{tk}'].mean()),
                period_excess_pp=per, verdict=v)


def beta_small(df, nav):
    """月超額（訊號時窗）對（全股池等權 − 0050）的迴歸斜率。"""
    v = np.array([nav.loc[P.IDX[e - 1]] for e in df['e']] + [nav.loc[P.IDX[P.T - 1]]])
    rn = v[1:] / v[:-1] - 1
    y = rn - df['r50'].to_numpy(); f = (df['ew_CUR'] - df['r50']).to_numpy()
    X = np.column_stack([np.ones_like(f), f])
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    res_ = y - X @ b
    s2 = res_ @ res_ / (len(y) - 2)
    se = np.sqrt(np.diag(s2 * np.linalg.inv(X.T @ X)))
    return dict(beta=float(b[1]), t_beta=float(b[1] / se[1]), alpha_month_pp=float(b[0]) * 100, t_alpha=float(b[0] / se[0]), corr=float(np.corrcoef(y, f)[0, 1]), n=len(y))


# ---------------------------------------------------------------- 交易格
res = {'PREREG_sha256': sha, 'acceptance': acc,
       'meta': dict(panel_T=P.T, panel_N=P.N, start=str(D0.date()), mid=str(DM.date()), since2019_from=str(D19.date()), since2024_from=str(D24.date()), end=str(DE.date())),
       'bench_0050': bw, 'cells': {c: dict(paths={}) for c in CELLS}, 'selection': {}, 'selection_shift': {}}
navs = {'0050': bench}; trades = []
EWT = {}
for s in range(4):
    tc = time.time()
    SG = SIG.pop(s) if s in SIG else build(s)
    ew = ew_tables(SG, ['CUR', 'L', 'M', 'S', 'MS', 'L_ADV'])
    EWT[s] = ew
    for tk in ('CUR', 'L', 'M', 'S', 'MS', 'L_ADV'):
        res['selection_shift'].setdefault(tk, {})[s] = selection_stats(ew, tk)['ann_excess_pp']
    if s == 0:
        r50a = ew['r50']; sl = period_slices(ew)
        idx, d0, d1, *_ = sl['full']
        acc['A4'] = dict(chain_0050_full=ann(r50a, d0, d1), official=stlib.cagr(bench, P.IDX[int(ew['e'].iloc[0]) - 1], DE))
        acc['A4']['ok'] = bool(abs(acc['A4']['chain_0050_full'] - acc['A4']['official']) < 0.005)
        assert acc['A4']['ok'], ('驗收 A4 失敗', acc['A4'])
        for tk in ('CUR', 'L', 'M', 'S', 'MS', 'L_ADV'):
            res['selection'][tk] = selection_stats(ew, tk)
    for cid in CELLS:
        R = res['cells'][cid]
        for cap in CAPS:
            key = f'{int(cap/1e4)}萬|+{s}'
            nav, lg, sts = run_cell(cid, SG, capital=cap)
            R['paths'][key] = dict(r19=cg(nav, 'since2019'), full=cg(nav, 'full'), r1923=cg(nav, 'y2019_2023'), r24=cg(nav, 'since2024'), maxdd=mdd(nav),
                                   seg=s12lib.seg_dd(nav), top5=s12lib.top5_share(lg, nav, cap), dd_dates=[s10lib.dd_dates(nav)['peak'], s10lib.dd_dates(nav)['trough']])
            if s == 0 and cap == 1e6:
                navs[cid] = nav
                w = W(nav)
                R['windows'] = slim(w)
                R['sharpe'] = {k: w[k]['sharpe'] for k in ('full', 'since2019')}
                R['t_month_excess'] = {k: w[k]['t_month_excess'] for k in ('full', 'since2019')}
                R['dd_dates'] = s10lib.dd_dates(nav)
                R['seg'] = s12lib.seg_dd(nav)
                R['yearly'] = stlib.yearly(nav, bench)
                R['trades_full'] = stlib.trade_stats(lg); R['trades_since2019'] = stlib.trade_stats(lg, '2019-01-01')
                R['n_buy'] = int(sum(x['n_buy'] for x in sts[:2] if isinstance(x, dict)))
                stl = [x for x in sts if isinstance(x, dict)]
                R['avg_npos'] = float(sum(x['avg_npos'] for x in stl))
                R['blk_min'] = int(sum(x['blk_min'] for x in stl)); R['blk_lot'] = int(sum(x['blk_lot'] for x in stl)); R['blk_limup'] = int(sum(x['blk_limup'] for x in stl))
                mnav = float(nav.mean())
                R['turnover_oneway_ann'] = float(sum((x['buy_val'] + x['sell_val']) for x in stl) / 2 / mnav / YRS)
                R['cost_ann_pct_of_nav'] = float(sum((x['fee'] + x['tax'] + x['slip'] + x['e50_cost']) for x in stl) / mnav / YRS)
                if cid == 'Q55':
                    sf = (stl[0]['stockfrac'] * sts[2] + stl[1]['stockfrac'] * sts[3]) / nav
                    R['L_book_share'] = {str(d.date()): float(sts[2].loc[d] / nav.loc[d]) for d in nav.index[::250].append(nav.index[-1:])}
                    navs['Q55_L'] = sts[2]; navs['Q55_MS'] = sts[3]
                else:
                    sf = stl[0]['stockfrac']
                R['avg_stockfrac'] = float(np.nanmean(sf.to_numpy()))
                R['top5_share_of_gain'] = s12lib.top5_share(lg, nav, 1e6)
                R['beta_small'] = beta_small(ew, nav)
                tk = {'CUR': 'CUR', 'TL': 'L', 'TM': 'M', 'TS': 'S', 'Q55': None, 'TL_ADV': 'L_ADV'}[cid]
                if tk is not None:
                    R['decomposition'] = decompose(ew, tk, nav)
                # 交易：進場時的市值層（市值分層，sig_t 當日）與市值（億）
                tiers = {k: s12lib.tiers_at(U[k], MC[k]) for k in set(lg['sig_t'].astype(int))}
                lg['tier_mc'] = [int(tiers[int(a)][int(b)]) for a, b in zip(lg['sig_t'], lg['j'])]
                lg['mc_e8'] = MC[lg['sig_t'].to_numpy(int), lg['j'].to_numpy(int)] / 1e8
                x = lg.copy(); x['run'] = cid
                trades.append(x[['run', 'sleeve', 'code', 'sig_date', 'ent_date', 'exit_date', 'hold_days', 'shares', 'cost', 'proceeds', 'pnl', 'ret', 'why', 'score', 'tier_mc', 'mc_e8']])
                nx2 = run_cell(cid, SG, cost='strict_x2')[0]
                R['costx2'] = slim(W(nx2), ('since2019', 'full'))
                blk, top = s2lib.block_from(lg, 5)
                nb = run_cell(cid, SG, block=blk)[0]
                R['block5'] = slim(W(nb), ('since2019', 'full'))
                R['block5_loss_since2019_pp'] = (w['since2019']['cagr'] - R['block5']['since2019']['cagr']) * 100
                R['block5_top'] = [dict(code=y.code, sleeve=y.sleeve, ent=str(y.ent_date.date()), exit=str(y.exit_date.date()), ret=float(y.ret), pnl=float(y.pnl)) for y in top.itertuples()]
                del nx2, nb
            del nav, lg, sts
    print('位移', s, f'{time.time()-tc:.0f}s 累計 {time.time()-t_all:.0f}s {rss()}MB', flush=True)
    del SG; gc.collect()

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
                        seg_median={k: float(np.median([v['seg'][k] for v in pv.values()])) for k in s12lib.SEGMENTS})
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
    dr, dd, d1923, d24, dt5 = d('r19'), d('maxdd'), d('r1923'), d('r24'), d('top5')
    seg = {}
    for sg in s12lib.SEGMENTS:
        x = np.array([A[k]['seg'][sg] - B[k]['seg'][sg] for k in keys])
        n, m = int((x > 0).sum()), float(np.median(x)) * 100
        seg[sg] = dict(n_shallower=n, median_pp=m, verdict='較淺' if (n >= 15 and m >= 1) else ('較深' if (n <= 5 and m <= -1) else '差不多'))
    return dict(ret_npos=int((dr > 0).sum()), ret_median_pp=float(np.median(dr)) * 100, ret_min_pp=float(dr.min()) * 100, ret_max_pp=float(dr.max()) * 100,
                dd_n_shallower=int((dd > 0).sum()), dd_median_pp=float(np.median(dd)) * 100, seg=seg,
                r1923_npos=int((d1923 > 0).sum()), r1923_median_pp=float(np.median(d1923)) * 100, r24_npos=int((d24 > 0).sum()), r24_median_pp=float(np.median(d24)) * 100,
                top5_npos_lower=int((dt5 < 0).sum()), top5_median_diff=float(np.median(dt5)))


pairs = {f'{c}-CUR': paired(c) for c in CELLS if c != 'CUR'}
res['pairs'] = pairs


def beat0050(c):
    p = res['cells'][c]['paths20']
    if p['median'] > B19 and p['n_above_0050'] >= 15:
        return 'SURGE 在大型股有效'
    if p['median'] < B19 and p['n_above_0050'] <= 5:
        return '在大型股無效'
    return '不確定'


def dependence(c):
    a, b = res['cells'][c]['paths20'], res['cells']['CUR']['paths20']
    pr = pairs[f'{c}-CUR']
    ok_a = a['top5_median'] < b['top5_median']
    ok_b = pr['r1923_median_pp'] >= -3 and pr['r24_median_pp'] >= -3
    v = '降低依賴' if (ok_a and ok_b) else ('較分散，但代價超過 3 個百分點' if ok_a else '沒有降低依賴')
    return dict(verdict=v, a_top5_lower=bool(ok_a), b_segments_within_3pp=bool(ok_b), top5_median=a['top5_median'], top5_median_CUR=b['top5_median'],
                cost_r19_median_pp=pr['ret_median_pp'], cost_maxdd_median_pp=pr['dd_median_pp'], r1923_median_pp=pr['r1923_median_pp'], r24_median_pp=pr['r24_median_pp'])


def small_exp(c):
    bc, b0 = res['cells'][c]['beta_small']['beta'], res['cells']['CUR']['beta_small']['beta']
    return dict(verdict='降低中小型股曝險' if bc <= b0 - 0.10 else '沒有明顯降低', beta=bc, beta_CUR=b0)


res['verdict'] = dict(
    TL_large_cap=beat0050('TL'), TL_ADV_large_cap_sensitivity=beat0050('TL_ADV'),
    selection={tk: res['selection'][tk]['verdict'] for tk in res['selection']},
    Q55_dependence=dependence('Q55'), TL_dependence_secondary=dependence('TL'),
    bear_segments={c: {sg: pairs[f'{c}-CUR']['seg'][sg]['verdict'] for sg in s12lib.SEGMENTS} for c in CELLS if c != 'CUR'},
    small_exposure={c: small_exp(c) for c in ('Q55', 'TL', 'TM', 'TS')},
    classes={c: res['cells'][c]['class'] for c in CELLS}, class_fails={c: res['cells'][c]['fails'] for c in CELLS if res['cells'][c]['fails']})
v = res['verdict']
v['xq'] = ('寫 20／21 號腳本的條件或參數' if ((v['Q55_dependence']['verdict'] == '降低依賴' and res['cells']['Q55']['class'] == '值得上XQ測')
                                    or (v['TL_large_cap'] == 'SURGE 在大型股有效' and res['cells']['TL']['class'] == '值得上XQ測')) else '不建議改現行；XS 草稿只供交叉驗證與紙上追蹤')
print('判定', json.dumps(v, ensure_ascii=False), flush=True)
for c in CELLS:
    R = res['cells'][c]
    print(c, {k: round(R['windows'][k]['cagr'] * 100, 2) for k in ('since2019', 'full', 'since2024', 'y2019_2023')}, 'dd', round(R['windows']['full']['maxdd'] * 100, 2),
          'p20', round(R['paths20']['median'] * 100, 2), R['paths20']['n_above_0050'], round(R['paths20']['maxdd_median'] * 100, 2), 'top5', round(R['paths20']['top5_median'], 3), R['class'], flush=True)
for tk, sv in res['selection'].items():
    print('選股', tk, round(sv['ann_excess_pp'], 2), round(sv['t_month'], 2), sv['years_positive'], '/', sv['n_years'], sv['verdict'], flush=True)

res['ew_monthly_shift0'] = EWT[0].assign(date=EWT[0]['date'].dt.strftime('%Y-%m-%d')).round(6).to_dict('records')
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
