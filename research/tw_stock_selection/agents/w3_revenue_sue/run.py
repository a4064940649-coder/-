#!/usr/bin/env python3
"""標準化月營收驚奇 SURGE（季節性隨機漫步加漂移）長歷史可成交回測。
先跑 prereg.py（PREREG 落地），再跑本檔：python3 -I run.py
成交、成本、漲停判定、60 日均值修補全部沿用 momentum_family/mlib.py → exec_rebuild/exlib.py（只讀取）。
權重版引擎 sim_w 與 w1_rev_record_high 相同（含 0050 欄），並以等權情形與 exlib.sim_portfolio 逐日比對。"""
import json, math, sys, time, hashlib
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'momentum_family'))
import mlib as M  # noqa: E402
L = M.L
T0 = time.time()
LONG = Path('/home/user/niklavsd/bottleneck-signals/data/processed/tw_revenue.csv')
XR = Path('/home/user/stevenhsiehsheng-ctrl/tw-stock-screener/data/extras')

res = json.loads((HERE / 'results.json').read_text(encoding='utf-8'))
assert hashlib.sha256(json.dumps(res['PREREG'], ensure_ascii=False, sort_keys=True).encode()).hexdigest() == res['PREREG_sha256'], 'PREREG 被改過'
res = {'PREREG': res['PREREG'], 'PREREG_sha256': res['PREREG_sha256']}

IDX, COLS, N, pos = L.IDX, L.COLS, L.N, L.pos
CI = L.CI

# ---------------- 營收與 SURGE ----------------
d = pd.read_csv(LONG, dtype={'code': str, 'market': str, 'name': str})
d['p'] = pd.PeriodIndex(pd.to_datetime(d['month']), freq='M')
d = d.sort_values(['p', 'market']).drop_duplicates(['p', 'code'], keep='first')
d = d[d['code'].isin(COLS)]
REV = d.pivot_table(index='p', columns='code', values='revenue_k_twd', aggfunc='last').sort_index()
REV = REV.reindex(pd.period_range(REV.index.min(), REV.index.max(), freq='M')).reindex(columns=COLS)
REV = REV.where(REV > 0)
IFRS_T = (REV.index >= pd.Period('2013-01', 'M')) & (REV.index <= pd.Period('2013-12', 'M'))


def mk_D(R):
    return R - R.shift(12)


def mk_G(R):
    p = R.shift(12)
    return (R / p - 1).where(p > 0)


def surge(X, drift=True, skip=False):
    if not skip:
        Xl = X.shift(1)
        mu = Xl.rolling(24, min_periods=24).mean(); sd = Xl.rolling(24, min_periods=24).std(ddof=1)
    else:   # 最近 24 個有效值（跳過缺值），最早一筆須在 t−36 之內
        mu = pd.DataFrame(np.nan, index=X.index, columns=X.columns); sd = mu.copy()
        pnum = pd.Series(np.arange(len(X.index)), index=X.index)
        for c in X.columns:
            s = X[c].dropna()
            if len(s) < 25: continue
            sl = s.shift(1)
            m_ = sl.rolling(24, min_periods=24).mean(); s_ = sl.rolling(24, min_periods=24).std(ddof=1)
            first = pd.Series(pnum.reindex(s.index).to_numpy(), index=s.index).shift(24)
            ok = (pnum.reindex(s.index) - first) <= 36
            mu.loc[s.index, c] = m_.where(ok); sd.loc[s.index, c] = s_.where(ok)
        # 注意：skip 版 μ/σ 只定義在 X_t 有值的月份，正好就是需要的
    num = (X - mu) if drift else X
    return (num / sd).where((sd > 0) & np.isfinite(sd) & X.notna())


D_ = mk_D(REV); D_.loc[IFRS_T] = np.nan
G_ = mk_G(REV); G_.loc[IFRS_T] = np.nan
SC = {'S_main': surge(D_), 'a_noDrift': surge(D_, drift=False), 'd_growth': surge(G_)}
SC['b_hold3'] = SC['S_main']; SC['c_consecutive'] = SC['S_main']
SC['diag_ifrs_skip'] = surge(D_, skip=True)
print('surge built', round(time.time() - T0), flush=True)

# ---------------- 日曆、股池 ----------------
Dt = [x for x in M.LDATES if (pd.Period(x, 'M') - 1) in REV.index]
Dt = Dt + [IDX[-1]] if IDX[-1] > Dt[-1] else Dt
DATES = pd.DatetimeIndex(Dt)
mm = [pd.Period(x, 'M') - 1 for x in Dt]
common = np.array([not c.startswith('91') for c in COLS])
v60 = M.V60.reindex(DATES); px = L.C.reindex(DATES)
liq = ((v60 >= 1e7) & (px >= 10)).to_numpy() & common[None, :]

E50 = L.E50.ffill()
AC2 = np.hstack([L.AC, E50.to_numpy()[:, None]])
AO2 = np.hstack([L.AO, E50.shift(1).bfill().to_numpy()[:, None]])
J50 = N


def wvec(pick):
    kk = len(pick)
    if kk >= 5: return {j: 1.0 / kk for j in pick}
    o = {j: 0.2 for j in pick}; o[J50] = (5 - kk) * 0.2; return o


def build(name):
    variant = name
    Sm = SC[name].reindex(mm).to_numpy()
    events, pools, info = {}, {}, []
    held = set(); nblk = 0; prev_rf = np.full(N, np.nan); tranches = []
    for k, dd in enumerate(Dt[:-1]):
        i = pos(dd); ex = i + 1
        trad = np.isfinite(L.OPEN_RAW[ex]) & np.isfinite(L.ADJ_RAW[ex])
        pool = liq[k] & np.isfinite(Sm[k]) & trad
        pp = np.where(pool)[0]; n = len(pp)
        rf = np.full(N, np.nan); cand = []
        if n > 0:
            s = Sm[k][pp]; order = pp[np.lexsort((pp, -s))]
            rf[order] = np.arange(n) / n
            ntop = max(1, int(round(0.10 * n)))
            cand = list(order[:ntop])
            if variant == 'c_consecutive':
                cand = [j for j in cand if np.isfinite(prev_rf[j]) and prev_rf[j] < 0.30]
        pick = [j for j in cand if not (M.GAP_UP[ex, j] and j not in held)]
        nblk += len(cand) - len(pick)
        if variant == 'b_hold3':
            tranches.append(wvec(pick) if n > 0 else {J50: 1.0})
            last3 = tranches[-3:] + [{J50: 1.0}] * (3 - len(tranches[-3:]))
            tw = {}
            for tr in last3:
                for j, w in tr.items(): tw[j] = tw.get(j, 0) + w / 3
        else:
            tw = wvec(pick)
        idx = np.array(list(tw.keys()), dtype=int); w = np.array(list(tw.values()))
        events[ex] = (idx, w)
        pools[ex] = (pp, np.full(n, 1.0 / n)) if n else (np.array([J50]), np.array([1.0]))
        info.append({'d': dd, 'ex': ex, 'rev_month': str(mm[k]), 'pool': n, 'n_cand': len(cand), 'n_pick': len(pick), 'n_blocked': len(cand) - len(pick),
                     'w50': float(tw.get(J50, 0.0)), 'picks': [COLS[j] for j in pick]})
        held = set(j for j in tw if j != J50)
        prev_rf = rf
    return events, pools, pd.DataFrame(info), nblk


def sim_w(events, buy, sell, t0, t1):
    """exlib.sim_portfolio 的權重版（同 w1_rev_record_high）。"""
    NN = N + 1; u = np.zeros(NN); cash = 1.0; nav = np.empty(t1 - t0 + 1); tov = []
    for t in range(t0, t1 + 1):
        if t in events:
            pk, w = events[t]
            Pz = AO2[t]
            h = np.where(u > 0, u * np.nan_to_num(Pz), 0.0)
            Vt = h.sum() + cash; Vn = Vt; tgt = np.zeros(NN)
            for _ in range(6):
                tgt[:] = 0; tgt[pk] = w * Vn
                cost = buy * np.clip(tgt - h, 0, None).sum() + sell * np.clip(h - tgt, 0, None).sum()
                Vn = Vt - cost
            tov.append(np.clip(h - tgt, 0, None).sum() / Vt if Vt > 0 else 0)
            u = np.zeros(NN); u[pk] = tgt[pk] / Pz[pk]; cash = 0.0
        nav[t - t0] = np.nansum(u * AC2[t]) + cash
    return pd.Series(nav, index=IDX[t0:t1 + 1]), float(np.mean(tov[1:])) if len(tov) > 1 else float('nan')


def daily_stats(nav, segs, bench_nav=None, ew_nav=None):
    """同 w1_rev_record_high.daily_stats：日頻 (a,b]，月窗 [a,b)。"""
    r = nav.pct_change(); mask = np.zeros(len(r), bool); span = 0.0
    for a, b in segs:
        m = (r.index > pd.Timestamp(a)) & (r.index <= pd.Timestamp(b)) & r.notna().to_numpy()
        if m.sum() == 0: continue
        first = np.argmax(m); last = len(m) - 1 - np.argmax(m[::-1])
        span += (r.index[last] - r.index[first - 1]).days / 365.25
        mask |= m
    rr = r[mask]
    if len(rr) < 20: return None
    eq = (1 + rr).cumprod()
    o = {'CAGR': round(float(eq.iloc[-1] ** (1 / span) - 1), 4), 'vol': round(float(rr.std() * math.sqrt(252)), 4),
         'maxDD': round(float((eq / eq.cummax() - 1).min()), 4), 'years': round(span, 2)}
    nd = nav.reindex(DATES); pr = (nd.shift(-1) / nd - 1).iloc[:-1].dropna()
    inw = np.zeros(len(pr), bool)
    for a, b in segs: inw |= (pr.index >= pd.Timestamp(a)) & (pr.index < pd.Timestamp(b))
    pr = pr[inw]; o['n_months'] = int(len(pr))
    for nm, bn in (('0050', bench_nav), ('EW', ew_nav)):
        if bn is None: continue
        bd = bn.reindex(DATES); bp = (bd.shift(-1) / bd - 1).iloc[:-1].reindex(pr.index); ex = pr - bp
        o[f'excess_vs_{nm}_pct_m'] = round(float(ex.mean() * 100), 2); o[f't_vs_{nm}_m'] = L.tstat(ex)
        o[f'hit_vs_{nm}_m'] = round(float((ex > 0).mean()), 3)
        b = bn.pct_change().reindex(rr.index); eb = (1 + b.fillna(0)).cumprod()
        o[f'{nm}_CAGR'] = round(float(eb.iloc[-1] ** (1 / span) - 1), 4); o[f'{nm}_maxDD'] = round(float((eb / eb.cummax() - 1).min()), 4)
    return o


def yearly(nav):
    r = nav.pct_change().dropna(); return {int(y): round(float((1 + g).prod() - 1) * 100, 1) for y, g in r.groupby(r.index.year)}


# ---- 引擎驗證 ----
_ev, _pl, _, _ = build('S_main')
_t0 = pos(Dt[0]); _t1 = pos(Dt[-1])
_pe = {t: ('open', v[0]) for t, v in _pl.items() if J50 not in v[0]}
_pw = {t: v for t, v in _pl.items() if J50 not in v[0]}
b1, s1 = L.cost_rates(0.001)
_k = min(_pe); na, _ = L.sim_portfolio(_pe, b1, s1, _k, _t1); nb_, _ = sim_w(_pw, b1, s1, _k, _t1)
res['engine_check_max_abs_nav_diff'] = float((na - nb_).abs().max())
print('engine check', res['engine_check_max_abs_nav_diff'], flush=True)


def first_on_or_after(s): return DATES[DATES >= pd.Timestamp(s)][0]
d13 = first_on_or_after('2013-02-01'); d16 = first_on_or_after('2016-02-01')


def windows(start):
    a0 = str(start.date())
    return {
        'select_official': [(a0, str(d13.date())), (str(d16.date()), '2019-04-30')],
        'valid_2019-05_2026-10': [('2019-05-01', '2026-10-31')],
        'y2024-01_2025-12': [('2024-01-01', '2025-12-31')],
        'A_preIFRS_to_2013-01sig': [(a0, str(d13.date()))],
        'B_postIFRS_2016-02_2019-04': [(str(d16.date()), '2019-04-30')],
        'IFRS_block_0050_2013-02_2016-01': [(str(d13.date()), str(d16.date()))],
        'valid_survivorPanel_2019-05_2023-12': [('2019-05-01', '2023-12-31')],
        'r21_select_2021-03_2023-12': [('2021-03-01', '2023-12-31')],
        'r21_valid_2024-01_2026-10': [('2024-01-01', '2026-10-31')],
        'valid_officialPx_only': [('2019-05-01', '2024-02-14'), ('2025-09-24', '2026-10-31')],
        'full_ex_IFRS': [(a0, str(d13.date())), (str(d16.date()), '2026-10-31')],
    }


FL = pd.read_parquet(L.P / 'flag.parquet').reindex(index=IDX, columns=COLS).to_numpy()
navs = {}; INFO = {}
START = None
for name in ('S_main', 'a_noDrift', 'b_hold3', 'c_consecutive', 'd_growth', 'diag_ifrs_skip'):
    ev, pools, info, nblk = build(name); INFO[name] = info
    st = info[info.pool > 0]
    k0 = int(st.index[0])
    start = Dt[k0]
    if START is None: START = start
    t0 = pos(start); t1 = pos(Dt[-1])
    ew, _ = sim_w(pools, 0.0, 0.0, t0, t1); navs[f'{name}_EW'] = ew
    W = windows(start)
    o = {'first_signal_date': str(start.date()), 'skipped_limit_up_total': int(nblk),
         'n_pick_by_year_mean': {int(y): round(float(g.n_pick.mean()), 1) for y, g in info.groupby(pd.DatetimeIndex(info.d).year)},
         'pool_by_year_mean': {int(y): round(float(g.pool.mean()), 1) for y, g in info.groupby(pd.DatetimeIndex(info.d).year)},
         'signal_months_with_pool0_after_start': [str(x.date()) for x in info.d[(info.pool == 0) & (info.index >= k0)]][:60],
         'months_lt5_picks_select': int(((info.n_pick < 5) & (info.pool > 0) & (info.d >= start) & (info.d < pd.Timestamp('2019-05-01'))).sum()),
         'months_lt5_picks_valid': int(((info.n_pick < 5) & (info.d >= pd.Timestamp('2019-05-01'))).sum()),
         'EW_windows': {k: daily_stats(ew, s, E50) for k, s in W.items()}, 'EW_yearly': yearly(ew)}
    for cost in ('gross', 0.001, 'stress'):
        b, s = (0.0, 0.0) if cost == 'gross' else L.cost_rates(cost)
        nav, tov = sim_w(ev, b, s, t0, t1); navs[f'{name}_{cost}'] = nav
        o[f'cost_{cost}'] = {'turnover_per_rebalance': round(tov, 3), 'windows': {k: daily_stats(nav, s_, E50, ew) for k, s_, in W.items()}, 'yearly_pct': yearly(nav)}
    # 事件層級（主成本；b 不做，因持有 3 個月重疊）
    if name != 'b_hold3':
        b, s = L.cost_rates(0.001)
        exs = list(info.ex) + [pos(Dt[-1])]
        rows = []
        for k in range(k0, len(info)):
            e0, e1 = exs[k], exs[k + 1]
            p0 = L.AO[e0]; p1 = L.AO[e1] if k + 1 < len(info) else L.AC[e1]
            rr = p1 / p0 - 1
            pl = pools[e0][0]; pl = pl[pl != J50]
            if len(pl) == 0: continue
            ewr = np.nanmean(rr[pl])
            for c in info.picks[k]:
                j = CI[c]
                if not np.isfinite(rr[j]): continue
                net = (1 + rr[j]) * (1 - s) / (1 + b) - 1
                rows.append({'d': info.d[k], 'code': c, 'ret': rr[j], 'net': net, 'abn': net - ewr})
        evd = pd.DataFrame(rows); evs = {}
        for kname, segs in W.items():
            m = np.zeros(len(evd), bool)
            for a, b_ in segs: m |= (evd.d >= pd.Timestamp(a)) & (evd.d < pd.Timestamp(b_))
            x = evd[m]
            if len(x) < 10: continue
            g = x.groupby('d').abn.mean()
            evs[kname] = {'n_events': int(len(x)), 'n_months': int(len(g)), 'mean_net_ret_pct': round(float(x.net.mean() * 100), 2),
                          'mean_abn_vs_poolEW_pct': round(float(x.abn.mean() * 100), 2), 'month_cluster_mean_abn_pct': round(float(g.mean() * 100), 2),
                          't_month_cluster': L.tstat(g), 'hit_abn': round(float((x.abn > 0).mean()), 3)}
        big = evd[evd.ret > 1.0]
        o['events_ret_gt_100pct'] = {'n': int(len(big)), 'of': int(len(evd)), 'sample': big.assign(d=big.d.astype(str)).round(3).head(15).to_dict('records')}
        o['event_study'] = evs
        nfl = 0; ntot = 0
        for k in range(k0, len(info)):
            e0, e1 = exs[k], min(exs[k + 1], len(IDX) - 1)
            for c in info.picks[k]:
                j = CI[c]; ntot += 1
                if np.isin(FL[e0:e1 + 1, j], [3, 4, 5]).any(): nfl += 1
        o['picks_with_flag345_in_hold'] = {'n': nfl, 'of': ntot}
    o['last_signal'] = {'d': str(info.d.iloc[-1].date()), 'rev_month': info.rev_month.iloc[-1], 'pool': int(info.pool.iloc[-1]), 'n_pick': int(info.n_pick.iloc[-1]), 'picks': info.picks.iloc[-1][:30]}
    res[name] = o
    w = o['cost_0.001']['windows']
    print(name, start.date(), {k: (w[k]['CAGR'], w[k]['maxDD'], w[k].get('t_vs_0050_m'), w[k].get('t_vs_EW_m'), w[k].get('0050_CAGR'), w[k].get('EW_CAGR'))
                               for k in ('select_official', 'valid_2019-05_2026-10', 'y2024-01_2025-12', 'A_preIFRS_to_2013-01sig', 'B_postIFRS_2016-02_2019-04') if w.get(k)}, round(time.time() - T0), flush=True)
    info.assign(d=info.d.astype(str), picks=info.picks.map(lambda x: ' '.join(x))).to_csv(HERE / f'info_{name}.csv.gz', index=False)

res['0050_yearly'] = yearly(E50[(E50.index >= START) & (E50.index <= Dt[-1])])

# ---- 十分位（主版本，未扣成本，事件層級）----
Sm = SC['S_main'].reindex(mm).to_numpy()
info = INFO['S_main']; exs = list(info.ex) + [pos(Dt[-1])]
rowsq = []
for k in range(len(info)):
    e0, e1 = exs[k], exs[k + 1]
    pool = liq[k] & np.isfinite(Sm[k]) & np.isfinite(L.OPEN_RAW[e0]) & np.isfinite(L.ADJ_RAW[e0])
    pp = np.where(pool)[0]; n = len(pp)
    if n < 50: continue
    p1 = L.AO[e1] if k + 1 < len(info) else L.AC[e1]
    rr = p1[pp] / L.AO[e0][pp] - 1
    s = Sm[k][pp]; rk = pd.Series(s).rank(method='first', ascending=False).to_numpy() - 1
    dec = np.floor(rk / n * 10).astype(int) + 1   # 1 = 最高 SURGE
    ewr = np.nanmean(rr)
    row = {'d': info.d[k]}
    for q in range(1, 11): row[q] = np.nanmean(rr[dec == q]) - ewr
    rowsq.append(row)
Q = pd.DataFrame(rowsq).set_index('d')
W = windows(START); dq = {}
for kname in ('select_official', 'valid_2019-05_2026-10', 'A_preIFRS_to_2013-01sig', 'B_postIFRS_2016-02_2019-04', 'y2024-01_2025-12'):
    m = np.zeros(len(Q), bool)
    for a, b_ in W[kname]: m |= (Q.index >= pd.Timestamp(a)) & (Q.index < pd.Timestamp(b_))
    x = Q[m]
    dq[kname] = {'n_months': int(len(x)), 'decile_abn_vs_EW_pct_m (1=最高SURGE)': {int(q): round(float(x[q].mean() * 100), 2) for q in range(1, 11)},
                 't_D1': L.tstat(x[1]), 't_D1_minus_D10': L.tstat(x[1] - x[10])}
res['diag_deciles_gross'] = dq
print('deciles', {k: (v['decile_abn_vs_EW_pct_m (1=最高SURGE)'][1], v['t_D1'], v['t_D1_minus_D10']) for k, v in dq.items()}, flush=True)

# ---- SURGE 分布健檢 ----
sm = SC['S_main'].stack()
res['diag_surge_distribution'] = {'n': int(len(sm)), 'quantiles': {str(q): round(float(sm.quantile(q)), 3) for q in (0.001, 0.01, 0.1, 0.5, 0.9, 0.99, 0.999)},
                                  'share_abs_gt_10': round(float((sm.abs() > 10).mean()), 5)}

# ---- rev_5y 交叉檢查 ----
r5 = pd.read_csv(XR / 'rev_5y.csv.gz', dtype={'code': str}); r5 = r5[r5.code.isin(COLS)]
r5['p'] = pd.PeriodIndex(r5['ym'], freq='M')
REV5 = r5.pivot_table(index='p', columns='code', values='revenue', aggfunc='last').sort_index()
REV5 = REV5.reindex(pd.period_range(REV5.index.min(), REV5.index.max(), freq='M')).reindex(columns=COLS)
REV5 = REV5.where(REV5 > 0)
both = REV.reindex(REV5.index).stack().to_frame('long').join(REV5.stack().to_frame('r5'), how='inner')
rel = (both.r5 / both.long - 1).abs()
S5 = surge(mk_D(REV5))
ov = []
for k, dd in enumerate(Dt[:-1]):
    if mm[k] < pd.Period('2023-12', 'M') or mm[k] not in S5.index: continue
    a = SC['S_main'].loc[mm[k]].to_numpy(); b = S5.loc[mm[k]].to_numpy()
    pl = liq[k] & np.isfinite(a) & np.isfinite(b)
    pp = np.where(pl)[0]
    if len(pp) < 50: continue
    nt = int(round(0.1 * len(pp)))
    ta = set(pp[np.argsort(-a[pp])[:nt]]); tb = set(pp[np.argsort(-b[pp])[:nt]])
    ov.append({'d': str(dd.date()), 'n': len(pp), 'jaccard_top10': round(len(ta & tb) / len(ta | tb), 3),
               'spearman': round(float(pd.Series(a[pp]).corr(pd.Series(b[pp]), method='spearman')), 3)})
ov = pd.DataFrame(ov)
res['diag_cross_check_rev5y'] = {'n_pairs': int(len(both)), 'share_within_0.1pct': round(float((rel < 1e-3).mean()), 4), 'share_within_1pct': round(float((rel < 1e-2).mean()), 4),
                                 'by_year_within_0.1pct': {int(y): round(float((g < 1e-3).mean()), 4) for y, g in rel.groupby(rel.index.get_level_values(0).year)},
                                 'surge_overlap_months': int(len(ov)), 'mean_jaccard_top10': round(float(ov.jaccard_top10.mean()), 3) if len(ov) else None,
                                 'mean_spearman': round(float(ov.spearman.mean()), 3) if len(ov) else None, 'min_jaccard': float(ov.jaccard_top10.min()) if len(ov) else None}
# ---- ann_date ----
rh = pd.read_csv(XR / 'rev_hist.csv.gz', dtype={'code': str})
ad = rh[(rh.ym == '2026-09') & rh.ann_date.notna()].set_index('code')['ann_date']
last = INFO['S_main'].iloc[-1]; dsig = str(last.d.date()); a_ = ad.reindex(last.picks)
res['diag_ann_date_2026-09'] = {'signal_date': dsig, 'n_picks': len(last.picks), 'n_with_ann_date': int(a_.notna().sum()),
                                'n_ann_after_signal_date': int((a_ > dsig).sum()), 'late_codes': list(a_[a_ > dsig].index)}
(HERE / 'results.json').write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str), encoding='utf-8')
pd.DataFrame(navs).to_parquet(HERE / 'nav.parquet')
print('done', round(time.time() - T0))
