#!/usr/bin/env python3
"""價格延遲 D1（Hou-Moskowitz 2005）月換股（M2）＋變體 (a)(b)(c)＋TAIEX 穩健性＋與 Amihud 重疊／漲跌停診斷。
規則見 results.json 的 PREREG（prereg.py，事前寫入）。"""
import json, math, sys
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'exec_rebuild'))
import exlib as L  # noqa: E402  成交引擎價格陣列與成本（只讀取）
from btlib import tstat  # noqa: E402

P = L.P
D = {k: pd.read_parquet(P / f'{k}.parquet') for k in ['adj', 'close_raw', 'open', 'high', 'value', 'flag', 'ret']}
good = D['adj'].notna().sum(axis=1) >= 300
BAD = [str(x.date()) for x in D['adj'].index[~good]]
for k in D: D[k] = D[k].loc[good]
cols = [c for c in D['adj'].columns if len(c) == 4 and c.isdigit() and not c.startswith('91')]
for k in D: D[k] = D[k][cols]
IDX = D['adj'].index; N = len(cols)
F = D['flag']; ADJ = D['adj']; CR = D['close_raw']; V = D['value']; RET = D['ret']; HI = D['high']
v0 = V.fillna(0).where(V.notna().cumsum() > 0)
V60 = v0.rolling(60, min_periods=60).mean()
BASEU = (CR >= 10) & F.isin([0, 1, 2]) & ADJ.notna()
LIQ = (V60 >= 1e7) & (CR >= 10)                       # Rm 用的流動基本池

# ---- Amihud（照 w3_amihud_illiquidity 定義重算，只為比較重疊） ----
valid = V.notna() & (V > 0) & RET.notna() & ~F.isin([3, 4, 5])
ILLIQ250 = (RET.abs() / (V / 1e6)).where(valid).rolling(250, min_periods=200).mean()
V250 = V.where(V > 0).rolling(250, min_periods=200).mean()

# ---- 漲跌停日 ----
thr_d = pd.Series(np.where(IDX < pd.Timestamp('2015-06-01'), 0.07, 0.10) - L.LTOL, index=IDX)
LU = RET.ge(thr_d, axis=0) & (CR >= HI * 0.999)
LDn = RET.le(-thr_d, axis=0)
LIMHITS = (LU | LDn).astype(float).where(ADJ.notna()).rolling(250, min_periods=200).sum()

# ---- 週報酬（W-WED） ----
_s = pd.Series(IDX, index=IDX)
WD = pd.DatetimeIndex(_s.groupby(IDX.to_period('W-WED')).last().to_numpy())
AW = ADJ.ffill(limit=4).loc[WD]
RW = (AW / AW.shift(1) - 1)
RW = RW.where(RW.abs() <= 2.0)
LIQW = LIQ.loc[WD].shift(1).fillna(False).astype(bool)
RWn = RW.to_numpy()
_m = LIQW.to_numpy() & np.isfinite(RWn)
_cnt = _m.sum(1)
RM_EW = np.where(_cnt >= 50, np.where(_m, RWn, 0).sum(1) / np.maximum(_cnt, 1), np.nan)
tx = pd.read_csv('/home/user/jonatw/finlab-txd/data/raw/taiex_twii.csv', parse_dates=['date']).set_index('date')['close'].sort_index()
TXW = tx.reindex(WD, method='ffill')
RM_TX = (TXW / TXW.shift(1) - 1).to_numpy()

# ---- 日報酬（變體 c） ----
RD = RET.where(RET.abs() <= 0.5).to_numpy()
LIQD = LIQ.shift(1).fillna(False).astype(bool).to_numpy()
_md = LIQD & np.isfinite(RD); _cd = _md.sum(1)
RM_D = np.where(_cd >= 50, np.where(_md, RD, 0).sum(1) / np.maximum(_cd, 1), np.nan)

def delay_D1(Y, rm, nlag=4, minobs=40):
    """Y: (T, N) 個股報酬（T 期，最後一期 = 最近），rm: (T+nlag,) 市場報酬（前 nlag 期為落後期用）。"""
    T = Y.shape[0]
    Xl = np.column_stack([rm[nlag - k: nlag - k + T] for k in range(nlag + 1)])   # col0 = Rm_t, col k = Rm_{t-k}
    xok = np.isfinite(Xl).all(1)
    M = (np.isfinite(Y) & xok[:, None]).astype(float)
    n = M.sum(0)
    Xf = np.column_stack([np.ones(T), np.nan_to_num(Xl)])                         # (T, 6)
    Yz = np.nan_to_num(Y) * M
    def r2(X):
        XtX = np.einsum('tn,ti,tj->nij', M, X, X); Xty = np.einsum('tn,ti->ni', Yz, X)
        XtX = XtX + 1e-12 * np.eye(X.shape[1])[None]
        b = np.linalg.solve(XtX, Xty[..., None])[..., 0]
        yy = (Yz ** 2).sum(0); sy = Yz.sum(0)
        ssr = yy - (b * Xty).sum(1); sst = yy - sy ** 2 / np.maximum(n, 1)
        with np.errstate(invalid='ignore', divide='ignore'):
            return 1 - ssr / sst
    with np.errstate(invalid='ignore', divide='ignore'):
        R2r = r2(Xf[:, :2]); R2u = r2(Xf)
        d1 = 1 - R2r / R2u
    d1[(n < minobs) | ~(R2u > 0) | ~np.isfinite(d1)] = np.nan
    return d1, R2r, R2u

ME = list(_s.groupby([IDX.year, IDX.month]).last())
ME = [d for d in ME if pd.Timestamp('2004-12-01') <= d <= pd.Timestamp('2026-09-30')]
NEXT = {d: IDX[IDX.get_loc(d) + 1] for d in ME}
WPOS = {d: int(np.searchsorted(WD, d, side='right') - 1) for d in ME}

def d1_at(d, kind):
    if kind in ('W_EW', 'W_TX'):
        k = WPOS[d]; lo = k - 51
        if lo - 4 < 0: lo = max(lo, 4)
        Y = RWn[lo:k + 1]; rm = (RM_EW if kind == 'W_EW' else RM_TX)[lo - 4:k + 1]
        return delay_D1(Y, rm, 4, 40)
    if kind == 'DAILY':
        i = IDX.get_loc(d); lo = max(i - 249, 4)
        return delay_D1(RD[lo:i + 1], RM_D[lo - 4:i + 1], 4, 200)
    raise KeyError(kind)

SIG = {}
for kind in ('W_EW', 'W_TX', 'DAILY'):
    SIG[kind] = {d: d1_at(d, kind)[0] for d in ME}

LPOS = {d: i for i, d in enumerate(L.IDX)}
LCOL = np.array([L.CI[x] for x in cols])
thrL = np.where(L.IDX < pd.Timestamp('2015-06-01'), 0.07, 0.10)[:, None] - L.LTOL
GAP_UP_L = (L.AO_raw / L.AC_prev - 1) >= thrL

def pool_at(d):
    ex = LPOS[NEXT[d]]
    u = (BASEU.loc[d] & (V60.loc[d] >= 1e7)).to_numpy()
    trad = np.isfinite(L.OPEN_RAW[ex, LCOL]) & np.isfinite(L.ADJ_RAW[ex, LCOL])
    return ex, u & trad

def qcut(s, ok, nq):
    g = np.full(N, -1); idx = np.where(ok)[0]
    if len(idx) == 0: return g
    g[idx] = pd.qcut(pd.Series(s[idx]).rank(method='first'), nq, labels=False).to_numpy()
    return g

def build(kind, dates=ME, terc=False, group=9, exlim=False):
    """group=9 → D10（遞補）；group in 0..8 → 診斷組（不遞補，剔除開盤漲停未持有）。"""
    events, pools, info = {}, {}, []
    held = set(); nblk = 0; ncand = 0
    for d in dates:
        ex, pool = pool_at(d)
        s = SIG[kind][d]
        ok = pool & np.isfinite(s)
        if terc: ok &= np.isfinite(V250.loc[d].to_numpy())
        if exlim: ok &= (LIMHITS.loc[d].to_numpy() < 5)
        if ok.sum() < 50: continue
        blocked = GAP_UP_L[ex, LCOL]
        if terc:
            tg = qcut(V250.loc[d].to_numpy(), ok, 3); pick = []
            for k in range(3):
                okk = ok & (tg == k); gq = qcut(s, okk, 10); nk = int((gq == 9).sum()); ncand += nk
                order = np.where(okk)[0]; order = order[np.argsort(-s[order], kind='stable')]; got = 0
                for j in order:
                    if got >= nk: break
                    if blocked[j] and LCOL[j] not in held: nblk += 1; continue
                    pick.append(j); got += 1
        else:
            gq = qcut(s, ok, 10); nk = int((gq == group).sum()); ncand += nk
            if group == 9:
                order = np.where(ok)[0]; order = order[np.argsort(-s[order], kind='stable')]; pick = []
                for j in order:
                    if len(pick) >= nk: break
                    if blocked[j] and LCOL[j] not in held: nblk += 1; continue
                    pick.append(j)
            else:
                mem = np.where(gq == group)[0]
                pick = [j for j in mem if not (blocked[j] and LCOL[j] not in held)]; nblk += nk - len(pick)
        pick = np.array(pick, dtype=int)
        events[ex] = ('open', LCOL[pick]); pools[ex] = ('open', LCOL[ok])
        held = set(LCOL[pick])
        pv = V60.loc[d].to_numpy()[pick] if len(pick) else np.array([np.nan])
        lh = LIMHITS.loc[d].to_numpy()
        info.append((d, int(ok.sum()), len(pick), float(np.nanmedian(pv)), float(np.nanpercentile(pv, 10)),
                     float(np.nanmean(lh[pick])) if len(pick) else np.nan, float(np.nanmean(lh[ok])),
                     float(np.nanmedian(s[pick])) if len(pick) else np.nan, float(np.nanmedian(s[ok]))))
    inf = pd.DataFrame(info, columns=['d', 'pool', 'held', 'v60_med', 'v60_p10', 'limhits_pick', 'limhits_pool', 'd1_med_pick', 'd1_med_pool']).set_index('d')
    return events, pools, inf, nblk, ncand

def sim(events, buy, sell, t0, t1):
    u = np.zeros(L.N); cash = 1.0; nav = np.empty(t1 - t0 + 1); tov = []
    for t in range(t0, t1 + 1):
        if t in events:
            mode, pk = events[t]; Pz = L.AO[t]
            h = np.where(u > 0, u * np.nan_to_num(Pz), 0.0); Vt = h.sum() + cash
            if len(pk) == 0:
                cash = Vt - sell * h.sum(); tov.append(h.sum() / Vt if Vt > 0 else 0); u = np.zeros(L.N)
            else:
                tgt = np.zeros(L.N); w = 1.0 / len(pk); Vn = Vt
                for _ in range(6):
                    tgt[:] = 0; tgt[pk] = w * Vn
                    cost = buy * np.clip(tgt - h, 0, None).sum() + sell * np.clip(h - tgt, 0, None).sum()
                    Vn = Vt - cost
                tov.append(np.clip(h - tgt, 0, None).sum() / Vt if Vt > 0 else 0)
                u = np.zeros(L.N); u[pk] = tgt[pk] / Pz[pk]; cash = 0.0
        nav[t - t0] = np.nansum(u * L.AC[t]) + cash
    return pd.Series(nav, index=L.IDX[t0:t1 + 1]), (float(np.mean(tov[1:])) if len(tov) > 1 else float('nan'))

WIN = {'select_2005-01_2019-04': ('2005-01-01', '2019-04-30'), 'valid_2019-05_2026-10': ('2019-05-01', '2026-10-31'),
       'valid_2024-01_2025-12': ('2024-01-01', '2025-12-31'), 'valid_survivor_2019-05_2023-12': ('2019-05-01', '2023-12-31'),
       'valid_2024-01_2026-10': ('2024-01-01', '2026-10-31')}
MAINW = list(WIN)[:3]

def mo(x): return (1 + x).groupby(x.index.to_period('M')).prod() - 1

def pstats(r, bench, ew, a, b):
    m = (r.index >= pd.Timestamp(a)) & (r.index <= pd.Timestamp(b)) & r.notna().to_numpy()
    rr = r[m]; bb = bench.reindex(rr.index).fillna(0); ee = ew.reindex(rr.index).fillna(0)
    yrs = (rr.index[-1] - r.index[max(np.argmax(m) - 1, 0)]).days / 365.25
    def st(x):
        eq = (1 + x).cumprod(); return round(float(eq.iloc[-1] ** (1 / yrs) - 1), 4), round(float((eq / eq.cummax() - 1).min()), 4)
    o = {}
    o['CAGR'], o['maxDD'] = st(rr); o['0050_CAGR'], o['0050_maxDD'] = st(bb); o['EW_CAGR'], o['EW_maxDD'] = st(ee)
    o['vol'] = round(float(rr.std() * math.sqrt(252)), 4)
    ms, mb, me = mo(rr), mo(bb), mo(ee)
    o['n_months'] = int(len(ms))
    o['excess_vs_0050_pct_m'] = round(float((ms - mb).mean() * 100), 3); o['t_vs_0050_m'] = tstat(ms - mb)
    o['excess_vs_EW_pct_m'] = round(float((ms - me).mean() * 100), 3); o['t_vs_EW_m'] = tstat(ms - me)
    o['hit_vs_EW_m'] = round(float((ms > me).mean()), 3)
    o['yearly_pct'] = {int(y): round(float((1 + g).prod() - 1) * 100, 1) for y, g in rr.groupby(rr.index.year)}
    o['yearly_0050_pct'] = {int(y): round(float((1 + g).prod() - 1) * 100, 1) for y, g in bb.groupby(bb.index.year)}
    o['yearly_EW_pct'] = {int(y): round(float((1 + g).prod() - 1) * 100, 1) for y, g in ee.groupby(ee.index.year)}
    return o

def rank_pct(x): return pd.Series(x).rank(pct=True).to_numpy() - 0.5

def fama_macbeth():
    rows = []
    for k, d in enumerate(ME[:-1]):
        ex0, pool = pool_at(d); ex1 = LPOS[NEXT[ME[k + 1]]]
        fr = L.AO[ex1, LCOL] / L.AO[ex0, LCOL] - 1
        d1 = SIG['W_EW'][d]; il = ILLIQ250.loc[d].to_numpy(); lh = LIMHITS.loc[d].to_numpy()
        ok = pool & np.isfinite(fr) & np.isfinite(d1) & np.isfinite(il) & np.isfinite(lh)
        if ok.sum() < 50: continue
        y = np.clip(fr[ok], -0.5, 1.0); rd = rank_pct(d1[ok]); ri = rank_pct(il[ok]); rl = rank_pct(lh[ok])
        one = np.ones(len(y))
        b1 = np.linalg.lstsq(np.column_stack([one, rd]), y, rcond=None)[0]
        b3 = np.linalg.lstsq(np.column_stack([one, rd, ri, rl]), y, rcond=None)[0]
        rows.append((d, b1[1], b3[1], b3[2], b3[3],
                     pd.Series(d1[ok]).corr(pd.Series(il[ok]), method='spearman'),
                     pd.Series(d1[ok]).corr(pd.Series(lh[ok]), method='spearman'),
                     pd.Series(d1[ok]).corr(pd.Series(V250.loc[d].to_numpy()[ok]), method='spearman'), int(ok.sum())))
    return pd.DataFrame(rows, columns=['d', 'b_d1', 'b_d1_ctrl', 'b_illiq_ctrl', 'b_lim_ctrl', 'sp_d1_illiq', 'sp_d1_limhits', 'sp_d1_v250', 'n']).set_index('d')

def fm_stats(fm, a, b):
    x = fm[(fm.index >= pd.Timestamp(a)) & (fm.index <= pd.Timestamp(b))]
    o = {'n_months': int(len(x)), 'avg_n': round(float(x['n'].mean()), 1)}
    for c in ('sp_d1_illiq', 'sp_d1_limhits', 'sp_d1_v250'): o[c] = round(float(x[c].mean()), 3)
    for c in ('b_d1', 'b_d1_ctrl', 'b_illiq_ctrl', 'b_lim_ctrl'):
        o[c + '_pct_m'] = round(float(x[c].mean() * 100), 3); o['t_' + c] = tstat(x[c])
    return o

def amihud_overlap(events_d10):
    rows = []
    for d in ME:
        ex, pool = pool_at(d)
        if ex not in events_d10: continue
        il = ILLIQ250.loc[d].to_numpy(); okA = pool & np.isfinite(il)
        if okA.sum() < 50: continue
        gq = qcut(il, okA, 5); q5 = set(LCOL[gq == 4])
        pk = set(events_d10[ex][1])
        if not pk: continue
        rows.append((d, len(pk & q5) / len(pk), len(q5) / okA.sum()))
    return pd.DataFrame(rows, columns=['d', 'share_D10_in_AmihudQ5', 'Q5_base_rate']).set_index('d')

def main():
    BUY, SELL = L.cost_rates(0.001); BUY_S, SELL_S = L.cost_rates('stress')
    out = {'meta': {'bad_days_dropped': BAD, 'n_stocks': N, 'cost_buy': BUY, 'cost_sell': SELL, 'stress': [BUY_S, SELL_S],
                    'n_weeks': int(len(WD)), 'first_rebalance_signal': str(ME[0].date()), 'last_rebalance_signal': str(ME[-1].date()),
                    'last_day': str(L.IDX[-1].date()), 'web_sources_accessible': False}}
    t0 = LPOS[NEXT[ME[0]]]; t1 = len(L.IDX) - 1
    # D1 分布
    dd = {}
    for kind in SIG:
        vals = np.concatenate([SIG[kind][d][pool_at(d)[1]] for d in ME[::12]])
        vals = vals[np.isfinite(vals)]
        dd[kind] = {q: round(float(np.percentile(vals, q)), 3) for q in (5, 25, 50, 75, 90, 95)}
    out['D1_distribution_in_pool'] = dd; print('D1 dist', dd, flush=True)
    ME_ANN = [d for d in ME if d.month == 6 and d.year >= 2005]
    specs = {'W_EW_D10': dict(kind='W_EW'), 'W_TX_D10': dict(kind='W_TX'), 'W_EW_D10_ANNUAL': dict(kind='W_EW', dates=ME_ANN),
             'W_EW_D10_TERC': dict(kind='W_EW', terc=True), 'DAILY_EW_D10': dict(kind='DAILY'), 'W_EW_D10_EXLIM_diag': dict(kind='W_EW', exlim=True)}
    for g in range(9): specs[f'W_EW_D{g+1}_diag'] = dict(kind='W_EW', group=g)
    b50 = L.E50.reindex(L.IDX[t0:t1 + 1]).pct_change()
    ewcache = {}
    def ew_for(kind, dates):
        key = (kind, len(dates))
        if key not in ewcache:
            _, pools, _, _, _ = build(kind, dates)
            g_, _ = sim(pools, 0.0, 0.0, t0, t1); n_, _ = sim(pools, BUY, SELL, t0, t1); ewcache[key] = (g_, n_)
        return ewcache[key]
    ewP = ew_for('W_EW', ME)
    navs = {'EW_gross': ewP[0], 'EW_net': ewP[1]}
    res = {}; evsave = {}
    for key, sp in specs.items():
        ev, _, info, nb, nc = build(**sp)
        if key == 'W_EW_D10': evsave = ev
        nav, tov = sim(ev, BUY, SELL, t0, t1); nav_s, _ = sim(ev, BUY_S, SELL_S, t0, t1); nav_g, _ = sim(ev, 0.0, 0.0, t0, t1)
        navs[key] = nav; navs[key + '_gross'] = nav_g
        own_ew = ew_for(sp['kind'], sp.get('dates', ME)) if not sp.get('terc') and not sp.get('exlim') else ewP
        ewr = ewP[0].pct_change(); ewn = own_ew[1].pct_change(); ewo = own_ew[0].pct_change()
        r = nav.pct_change()
        o = {'blocked_gap_up': nb, 'group_members_total': nc, 'blocked_share': round(nb / nc, 4) if nc else None,
             'turnover_value_per_rebalance': round(tov, 3), 'avg_pool': round(float(info['pool'].mean()), 1), 'avg_held': round(float(info['held'].mean()), 1)}
        for w, (a, b) in WIN.items():
            if 'ANNUAL' in key: a = max(a, str(L.IDX[min(ev)].date()))
            o[w] = pstats(r, b50, ewr, a, b)
            ps_ = pstats(nav_s.pct_change(), b50, ewr, a, b); pg = pstats(nav_g.pct_change(), b50, ewr, a, b)
            o[w]['CAGR_stress'] = ps_['CAGR']; o[w]['t_stress_vs_EW_m'] = ps_['t_vs_EW_m']; o[w]['t_stress_vs_0050_m'] = ps_['t_vs_0050_m']
            o[w]['CAGR_gross'] = pg['CAGR']; o[w]['t_gross_vs_EW_m'] = pg['t_vs_EW_m']
            o[w]['EW_net_CAGR_ownpool'] = pstats(ewn, b50, ewr, a, b)['CAGR']
            po = pstats(r, b50, ewo, a, b); o[w]['EW_ownpool_CAGR'] = po['EW_CAGR']; o[w]['t_vs_EW_ownpool_m'] = po['t_vs_EW_m']
            sub = info[(info.index >= pd.Timestamp(a)) & (info.index <= pd.Timestamp(b))]
            if len(sub):
                o[w]['avg_pool'] = round(float(sub['pool'].mean()), 1); o[w]['avg_held'] = round(float(sub['held'].mean()), 1)
                o[w]['holding_v60_median_NTD_m'] = round(float(sub['v60_med'].median()) / 1e6, 1)
                o[w]['capacity_AUM_NTD_m_10pct_p10'] = round(float((sub['held'] * sub['v60_p10'] * 0.1).median()) / 1e6, 1)
                o[w]['limhits250_pick_mean'] = round(float(sub['limhits_pick'].mean()), 2); o[w]['limhits250_pool_mean'] = round(float(sub['limhits_pool'].mean()), 2)
                o[w]['D1_median_pick'] = round(float(sub['d1_med_pick'].median()), 3); o[w]['D1_median_pool'] = round(float(sub['d1_med_pool'].median()), 3)
        res[key] = o
        ps, pv, p2 = (o[w] for w in MAINW)
        print(key, 'tov', o['turnover_value_per_rebalance'], 'blk', o['blocked_share'], 'held', o['avg_held'],
              '| S', ps['CAGR'], 'g', ps['CAGR_gross'], 'st', ps['CAGR_stress'], '50', ps['0050_CAGR'], 'EW', ps['EW_CAGR'], 'tEW', ps['t_vs_EW_m'], 't50', ps['t_vs_0050_m'], 'DD', ps['maxDD'],
              '| V', pv['CAGR'], 'g', pv['CAGR_gross'], 'st', pv['CAGR_stress'], '50', pv['0050_CAGR'], 'EW', pv['EW_CAGR'], 'tEW', pv['t_vs_EW_m'], 't50', pv['t_vs_0050_m'], 'DD', pv['maxDD'],
              '| 24-25', p2['CAGR'], p2['0050_CAGR'], p2['EW_CAGR'], p2['t_vs_EW_m'], p2['t_vs_0050_m'], flush=True)
        if key == 'W_EW_D10': info.to_csv(HERE / 'monthly_counts_W_EW_D10.csv')
    # 年換股等權（同股池）
    ewA = ew_for('W_EW', ME_ANN)
    rA = navs['W_EW_D10_ANNUAL'].pct_change()
    a0 = str(rA.index[rA.ne(0).argmax()].date())
    res['W_EW_D10_ANNUAL']['vs_annual_EW'] = {w: {k: v for k, v in pstats(rA, b50, ewA[0].pct_change(), max(a, a0), b).items() if k in ('CAGR', 'EW_CAGR', 'excess_vs_EW_pct_m', 't_vs_EW_m')} for w, (a, b) in WIN.items()}
    out['portfolios'] = res
    # D10−D1 毛價差
    sp = {}
    q10 = navs['W_EW_D10_gross'].pct_change(); q1 = navs['W_EW_D1_diag_gross'].pct_change()
    for w, (a, b) in WIN.items():
        m = (q10.index >= pd.Timestamp(a)) & (q10.index <= pd.Timestamp(b))
        x = mo(q10[m].fillna(0)) - mo(q1[m].fillna(0))
        sp[w] = {'D10_minus_D1_gross_pct_m': round(float(x.mean() * 100), 3), 't': tstat(x), 'n_months': int(len(x))}
    out['D10_minus_D1_gross'] = sp
    # 十分位單調性（毛，對 EW 月超額）
    mono = {}
    for w, (a, b) in list(WIN.items())[:2]:
        mono[w] = {}
        for g in range(10):
            k = f'W_EW_D{g+1}_diag_gross' if g < 9 else 'W_EW_D10_gross'
            mono[w][f'D{g+1}'] = pstats(navs[k].pct_change(), b50, navs['EW_gross'].pct_change(), a, b)['excess_vs_EW_pct_m']
    out['decile_gross_excess_vs_EW_pct_m'] = mono; print('MONO', json.dumps(mono), flush=True)
    fm = fama_macbeth()
    out['fama_macbeth'] = {w: fm_stats(fm, a, b) for w, (a, b) in WIN.items()}
    for w in MAINW: print('FM', w, json.dumps(out['fama_macbeth'][w], ensure_ascii=False), flush=True)
    print('SPREAD', json.dumps(sp), flush=True)
    # Amihud 重疊
    ov = amihud_overlap(evsave)
    amh = pd.read_parquet(HERE.parent / 'w3_amihud_illiquidity' / 'nav.parquet')
    ovo = {}
    for w, (a, b) in WIN.items():
        x = ov[(ov.index >= pd.Timestamp(a)) & (ov.index <= pd.Timestamp(b))]
        o = {'share_D10_in_AmihudQ5': round(float(x['share_D10_in_AmihudQ5'].mean()), 3), 'base_rate': round(float(x['Q5_base_rate'].mean()), 3)}
        ra = amh['A250_Q5'].pct_change(); rew = amh['EW_gross'].pct_change(); rd = navs['W_EW_D10'].pct_change(); rew2 = navs['EW_gross'].pct_change()
        m = (rd.index >= pd.Timestamp(a)) & (rd.index <= pd.Timestamp(b))
        ix = rd.index[m]
        ea = mo(ra.reindex(ix).fillna(0)) - mo(rew.reindex(ix).fillna(0)); ed = mo(rd.reindex(ix).fillna(0)) - mo(rew2.reindex(ix).fillna(0))
        o['corr_monthly_excess_vs_EW_D10_vs_AmihudQ5'] = round(float(ea.corr(ed)), 3)
        o['amihud_A250_Q5_CAGR'] = round(float((1 + ra.reindex(ix).fillna(0)).prod() ** (1 / ((ix[-1] - ix[0]).days / 365.25)) - 1), 4)
        ovo[w] = o
    out['amihud_overlap'] = ovo; print('OVERLAP', json.dumps(ovo), flush=True)
    def passes(o):
        t = o['t_vs_EW_m'] if np.isfinite(o['t_vs_EW_m']) else -9
        ok_ew = (o['excess_vs_EW_pct_m'] > 0) and (t >= 2); ok_50 = o['CAGR'] >= o['0050_CAGR']
        return {'beat_EW_t2': bool(ok_ew), 'not_lose_0050': bool(ok_50), 'pass': bool(ok_ew and ok_50)}
    pr = res['W_EW_D10']
    vs = {'select': passes(pr['select_2005-01_2019-04']), 'valid': passes(pr['valid_2019-05_2026-10'])}
    k = vs['select']['pass'] + vs['valid']['pass']
    vs['verdict'] = '可用' if k == 2 else ('待觀察' if k == 1 else '不可用')
    vs['variants_pass'] = {kk: {w: passes(o[w])['pass'] for w in MAINW[:2]} for kk, o in res.items()}
    out['verdict'] = vs
    pd.DataFrame(navs).to_parquet(HERE / 'nav.parquet')
    f = HERE / 'results.json'
    d = json.loads(f.read_text(encoding='utf-8')); d.update(out)
    f.write_text(json.dumps(d, ensure_ascii=False, indent=1, default=lambda x: None if isinstance(x, float) and not np.isfinite(x) else str(x)), encoding='utf-8')
    print(json.dumps(vs, ensure_ascii=False))

if __name__ == '__main__':
    main()
