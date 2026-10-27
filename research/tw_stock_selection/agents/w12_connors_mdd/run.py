#!/usr/bin/env python3
"""Connors 多日下跌（MDD）：SMA200 之上、收盤 < SMA5、最近 5 日至少 4 日收黑 → 隔日開盤買；收盤 > SMA5 → 隔日開盤賣；積極版加碼。
規則見 results.json → PREREG（prereg.py 事前寫入）。執行：python3 -I run.py
成本只從 exec_rebuild/exlib 匯入；股池／flag／處置／事件／組合引擎沿用 w3_connors_double7/run.py（複製後改訊號、出場、加碼、20 槽）。"""
import json, math, sys, time
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'exec_rebuild'))
import exlib as L  # noqa: E402
from btlib import tstat  # noqa: E402
T0 = time.time()
res = json.loads((HERE / 'results.json').read_text(encoding='utf-8')); assert 'PREREG' in res
res = {'PREREG': res['PREREG']}

P = L.P
names = ['adj', 'close_raw', 'open', 'high', 'value', 'flag', 'src']
D = {k: pd.read_parquet(P / f'{k}.parquet') for k in names}
good = D['adj'].notna().sum(axis=1) >= 300
BAD = [str(x.date()) for x in D['adj'].index[~good]]
cols = [c for c in D['adj'].columns if not c.startswith('91')]
for k in D: D[k] = D[k].loc[good, cols]
IDX = D['adj'].index; T = len(IDX); N = len(cols)
gi = np.array([L.IDX.get_loc(d) for d in IDX]); colj = np.array([L.CI[c] for c in cols])
F = D['flag']; bad5 = (F == 5)
ADJ = D['adj'].mask(bad5); CR = D['close_raw'].mask(bad5); OP = D['open'].mask(bad5); HI = D['high'].mask(bad5); V = D['value']
fac = ADJ / CR
AO_raw = OP * fac
AH = (HI * fac).to_numpy()
ACf = ADJ.ffill()
v0 = V.fillna(0).where(V.notna().cumsum() > 0)
V60 = v0.rolling(60, min_periods=60).mean()
UNIV = (V60 >= 1e7) & (CR >= 10) & F.isin([0, 1, 2]) & ADJ.notna()
mkt4 = (F == 4).sum(axis=1) >= 50
FL = (F.isin([3, 4, 5]) & ~((F == 4) & mkt4.to_numpy()[:, None])).astype(float)
dsp = pd.read_csv(L.X / 'disposal_5y.csv.gz', dtype={'code': str}, parse_dates=['start', 'end'])
INDISP = np.zeros((T, N), bool); CJ = {c: j for j, c in enumerate(cols)}
for _, r in dsp.iterrows():
    if r['code'] in CJ and pd.notna(r['start']) and pd.notna(r['end']):
        INDISP[(IDX >= r['start']) & (IDX <= r['end']), CJ[r['code']]] = True
DISP_TT1 = INDISP | np.vstack([INDISP[1:], np.zeros((1, N), bool)])

lim = np.where(IDX < pd.Timestamp('2015-06-01'), 0.07, 0.10)[:, None]
AC_prev = ACf.shift(1).to_numpy()
AOn = AO_raw.to_numpy(); ACn = ACf.to_numpy(); ADJn = ADJ.to_numpy()
with np.errstate(invalid='ignore', divide='ignore'):
    GAP_UP = (AOn / AC_prev - 1) >= lim - 0.006
    LOCK_DN = (AH / AC_prev - 1) <= -(lim - 0.006)
SELLABLE = np.isfinite(AOn) & ~LOCK_DN
OPOK = np.isfinite(AOn)
BUYOK = OPOK & ~GAP_UP
lv_mat = np.maximum.accumulate(np.where(np.isfinite(ADJn), np.arange(T)[:, None], -1), axis=0)
LASTV = lv_mat[T - 1]
SRC3 = ((IDX >= pd.Timestamp('2024-02-15')) & (IDX <= pd.Timestamp('2025-09-23')))

# ---------------- 指標（只用還原收盤） ----------------
A = ADJ.ffill(limit=10)
SMA200 = A.rolling(200, min_periods=200).mean().to_numpy()
SMA5 = A.rolling(5, min_periods=5).mean().to_numpy()
dn = (A < A.shift(1)).astype(float).where(A.shift(1).notna() & A.notna())
DOWN5 = dn.rolling(5, min_periods=5).sum().to_numpy()
HAS = np.isfinite(ADJn)
FL6 = FL.rolling(6, min_periods=1).max().to_numpy() > 0
with np.errstate(invalid='ignore'):
    BASE = UNIV.to_numpy() & HAS & (ADJn > SMA200) & ~FL6 & ~DISP_TT1
    SIG = BASE & (ADJn < SMA5) & (DOWN5 >= 4)
    EXIT = HAS & (ADJn > SMA5)
    DIST5 = ADJn / SMA5 - 1
    # 重疊診斷：Double 7（7 日最低收盤）與 RSI(2)<10（Wilder）
    LO7 = A.rolling(7, min_periods=7).min().to_numpy()
    FL7 = FL.rolling(7, min_periods=1).max().to_numpy() > 0
    D7 = UNIV.to_numpy() & HAS & (ADJn > SMA200) & ~FL7 & ~DISP_TT1 & (ADJn <= LO7 + 1e-12)
    dA = A.diff(); AG = dA.clip(lower=0).ewm(alpha=0.5, adjust=False).mean(); AL_ = (-dA).clip(lower=0).ewm(alpha=0.5, adjust=False).mean()
    RSI = (100 - 100 / (1 + AG / AL_)).to_numpy()
    R2 = BASE & (RSI < 10)
sel = IDX >= pd.Timestamp('2005-01-01')
res['meta'] = {'bad_days_dropped': BAD, 'n_stocks': N, 'market_flag4_days_ignored': [str(x.date()) for x in IDX[mkt4.to_numpy()]],
               'signal_stock_days': int(SIG.sum()), 'mean_signals_per_day': round(float(SIG.sum(1)[sel].mean()), 2),
               'mean_univ_per_day': round(float(UNIV.to_numpy().sum(1)[sel].mean()), 1),
               'share_univ_above_sma200': round(float(np.nanmean(np.where(UNIV.to_numpy() & np.isfinite(SMA200), ADJn > SMA200, np.nan))), 3),
               'overlap_MDD_signal_also_Double7_7day_low': round(float((SIG & D7).sum() / SIG.sum()), 3),
               'overlap_MDD_signal_also_RSI2_lt10': round(float((SIG & R2).sum() / SIG.sum()), 3),
               'overlap_Double7_signal_also_MDD': round(float((SIG & D7).sum() / D7.sum()), 3)}
print('setup', round(time.time() - T0), res['meta'], flush=True)

# 同池等權（開盤對開盤，事件用）
oo = (AO_raw.shift(-1) / AO_raw - 1).where(UNIV.shift(1, fill_value=False)).clip(-0.5, 1.0)
LEW = np.concatenate([[0.0], np.cumsum(np.log1p(oo.mean(axis=1).fillna(0).to_numpy()))])
E50n = L.e50.reindex(IDX.union(L.e50.index)).ffill().reindex(IDX).to_numpy()

def exit_px(x, j):
    """x 日開盤賣；鎖跌停／無開盤 → 順延；之後無價格（下市）→ 最後收盤。回傳 (價, 實際日, 是否下市)。"""
    if x > LASTV[j]: return ACn[T - 1, j], min(LASTV[j] + 1, T - 1), True
    k = x
    while k < T and not SELLABLE[k, j]:
        if k > LASTV[j]: return ACn[T - 1, j], min(LASTV[j] + 1, T - 1), True
        k += 1
    if k >= T: return ACn[T - 1, j], T - 1, False
    return AOn[k, j], k, False

def build_events(agg, ts_lim):
    rows = []; nblk = nno = 0
    for j in range(N):
        sigs = np.flatnonzero(SIG[:, j]); free = -1
        for t in sigs:
            if t < free: continue
            e = t + 1
            if e >= T: break
            if not OPOK[e, j]: nno += 1; continue
            if GAP_UP[e, j]: nblk += 1; continue
            p0 = AOn[e, j]
            end = min(T, e + ts_lim) if ts_lim else T
            hit = np.flatnonzero(EXIT[e:end, j])
            why = 'sma5'
            if len(hit):
                sx = e + hit[0]; x = sx + 1
            elif ts_lim:
                sx = e + ts_lim - 1; x = e + ts_lim; why = 'time'
            else:
                sx = min(LASTV[j], T - 1); x = LASTV[j] + 1 if LASTV[j] < T - 1 else T; why = 'none'
            # 加碼：s ∈ [e, sx-1]（sx 當日出場優先），收盤 < p0 且 s+1 可買
            a = -1
            if agg and sx > e:
                s_rng = np.arange(e, min(sx, T - 1))
                ok = (ADJn[s_rng, j] < p0) & BUYOK[s_rng + 1, j]
                if ok.any(): a = int(s_rng[np.argmax(ok)] + 1)
            if x >= T:
                px, xs, dl = ACn[T - 1, j], T - 1, False; why = 'open_at_end'
            else:
                px, xs, dl = exit_px(x, j)
                if dl: why = 'delist'
            pa = AOn[a, j] if a >= 0 else np.nan
            rows.append((j, t, e, xs, px, why, dl, a, pa))
            free = xs if not dl else T
    df = pd.DataFrame(rows, columns=['j', 's', 'e', 'x', 'px', 'why', 'delist', 'a', 'pa'])
    df['p0'] = AOn[df.e, df.j]; df['added'] = df.a >= 0
    r1 = df.px / df.p0 - 1; r2 = df.px / df.pa - 1
    df['raw1'] = r1; df['raw_add'] = r2
    df['raw'] = np.where(df.added, (r1 + r2) / 2, r1); df['hold'] = df.x - df.e
    ew1 = np.expm1(LEW[df.x.to_numpy()] - LEW[df.e.to_numpy()])
    ew2 = np.expm1(LEW[df.x.to_numpy()] - LEW[np.where(df.added, df.a, df.e)])
    df['ew'] = np.where(df.added, (ew1 + ew2) / 2, ew1)
    df['b50'] = E50n[np.maximum(df.x - 1, df.s)] / E50n[df.s] - 1
    df['gap'] = df.p0 / ACn[df.s, df.j] - 1
    df['src3'] = [bool(SRC3[a_:b + 1].any()) for a_, b in zip(df.s, df.x)]
    df['sig'] = IDX[df.s.to_numpy()]; df['entry'] = IDX[df.e.to_numpy()]
    df['dist5'] = DIST5[df.s, df.j]
    build_events.cnt = {'blocked_gap_up': nblk, 'no_open': nno}
    return df

COSTS = {'gross': (0.0, 0.0), 'rt0.5': (0.0025, 0.0025), 'slip0.1': L.cost_rates(0.001), 'stress': L.cost_rates('stress')}
res['cost_roundtrip_pct'] = {k: round((1 - (1 - s) / (1 + b)) * 100, 3) for k, (b, s) in COSTS.items()}
def add_cost(df, cost):
    b, s = COSTS[cost]
    return (1 + df.raw) * (1 - s) / (1 + b) - 1

WIN = {'select_2005-01_2019-04': [('2005-01-01', '2019-04-30')], 'valid_2019-05_2026-10': [('2019-05-01', '2026-10-31')],
       'y2024-01_2025-12': [('2024-01-01', '2025-12-31')],
       'select_sub_2005-2011': [('2005-01-01', '2011-12-31')], 'select_sub_2012-2019-04': [('2012-01-01', '2019-04-30')],
       'valid_survivor_2019-05_2023-12': [('2019-05-01', '2023-12-31')],
       'valid_officialPx_only': [('2019-05-01', '2024-02-14'), ('2025-09-24', '2026-10-31')],
       'r21_select_2021-01_2023-12': [('2021-01-01', '2023-12-31')], 'r21_valid_2024-01_2026-10': [('2024-01-01', '2026-10-31')]}

def ev_stats(df, segs):
    m = np.zeros(len(df), bool)
    for a, b in segs: m |= ((df.sig >= pd.Timestamp(a)) & (df.sig <= pd.Timestamp(b))).to_numpy()
    x = df[m]
    if len(x) < 10: return {'n': int(len(x))}
    net = add_cost(x, 'slip0.1'); ab = net - x.ew; abg = x.raw - x.ew; ab5 = add_cost(x, 'rt0.5') - x.ew
    abs_ = add_cost(x, 'stress') - x.ew; ab50 = net - x.b50
    em = x.entry.dt.to_period('M'); gm = ab.groupby(em).mean()
    o = {'n': int(len(x)), 'n_months': int(gm.size), 'avg_hold_days': round(float(x.hold.mean()), 2),
         'share_added': round(float(x.added.mean()), 3),
         'mean_raw_pct': round(float(x.raw.mean() * 100), 3), 'mean_net_pct': round(float(net.mean() * 100), 3),
         'median_net_pct': round(float(net.median() * 100), 3), 'win_rate_net': round(float((net > 0).mean()), 3),
         'mean_EW_same_window_pct': round(float(x.ew.mean() * 100), 3),
         'mean_ab_gross_pct': round(float(abg.mean() * 100), 3), 't_month_ab_gross': tstat(abg.groupby(em).mean()),
         'mean_ab_rt0.5_pct': round(float(ab5.mean() * 100), 3), 't_month_ab_rt0.5': tstat(ab5.groupby(em).mean()),
         'mean_ab_net_pct': round(float(ab.mean() * 100), 3), 'month_cluster_mean_ab_net_pct': round(float(gm.mean() * 100), 3),
         't_month_ab_net': tstat(gm), 't_date_ab_net_ref': tstat(ab.groupby(x.entry).mean()),
         'mean_ab_net_stress_pct': round(float(abs_.mean() * 100), 3), 't_month_ab_net_stress': tstat(abs_.groupby(em).mean()),
         'mean_net_vs_0050_pct': round(float(ab50.mean() * 100), 3), 't_month_vs_0050': tstat(ab50.groupby(em).mean()),
         'hit_ab_net': round(float((ab > 0).mean()), 3),
         'mean_gap_pct': round(float(x.gap.mean() * 100), 3),
         'mean_ab_gross_if_close_entry_pct_NOT_TRADABLE': round(float((((1 + x.raw1) * (1 + x.gap) - 1) - np.expm1(LEW[x.x.to_numpy()] - LEW[x.e.to_numpy()])).mean() * 100), 3),
         'exit_mix': {k: round(float(v), 3) for k, v in x.why.value_counts(normalize=True).items()},
         'n_delist': int(x.delist.sum()), 'p01_raw_pct': round(float(x.raw.quantile(0.01) * 100), 2), 'p99_raw_pct': round(float(x.raw.quantile(0.99) * 100), 2)}
    if x.added.any():
        xa = x[x.added]; ewa = np.expm1(LEW[xa.x.to_numpy()] - LEW[xa.a.to_numpy()])
        ew1 = np.expm1(LEW[xa.x.to_numpy()] - LEW[xa.e.to_numpy()])
        o['added_trades'] = {'n': int(len(xa)), 'first_unit_raw_pct': round(float(xa.raw1.mean() * 100), 3),
                             'add_unit_raw_pct': round(float(xa.raw_add.mean() * 100), 3),
                             'add_unit_ab_gross_pct': round(float((xa.raw_add - ewa).mean() * 100), 3),
                             'first_unit_ab_gross_pct': round(float((xa.raw1 - ew1).mean() * 100), 3),
                             'avg_hold_days': round(float(xa.hold.mean()), 2)}
        nx = x[~x.added]
        o['not_added_trades'] = {'n': int(len(nx)), 'ab_gross_pct': round(float((nx.raw - nx.ew).mean() * 100), 3), 'avg_hold_days': round(float(nx.hold.mean()), 2)}
    return o

# ---------------- 20 槽組合 ----------------
t_start = int(np.searchsorted(IDX, pd.Timestamp('2004-07-01')))
V60n = V60.to_numpy()
CANDS = []
for t in range(T):
    js = np.flatnonzero(SIG[t])
    if len(js): js = js[np.lexsort((-np.nan_to_num(V60n[t, js]), DIST5[t, js]))]
    CANDS.append(js)

def sim(agg, ts_lim, buy, sell, slots=20, unit_div=20, seed=None):
    rng = np.random.default_rng(seed) if seed is not None else None
    cash = 1.0; Pp = {}
    nav = np.empty(T - t_start); npos = np.empty(T - t_start); expo = np.empty(T - t_start)
    cnt = {'entries': 0, 'adds': 0, 'add_skip_cash': 0, 'add_skip_gap': 0, 'skip_no_slot': 0, 'skip_gap': 0,
           'exit_signal': 0, 'exit_time': 0, 'delist': 0, 'exit_delayed': 0}
    for t in range(t_start, T):
        sold = set()
        for j in list(Pp):
            p = Pp[j]
            if t > LASTV[j]:
                cash += p['u'] * ACn[T - 1, j] * (1 - sell); del Pp[j]; cnt['delist'] += 1; sold.add(j)
            elif p['px']:
                if SELLABLE[t, j]:
                    cash += p['u'] * AOn[t, j] * (1 - sell); cnt[p['why']] += 1; del Pp[j]; sold.add(j)
                else: cnt['exit_delayed'] += 1
        navo = cash + sum(p['u'] * (AOn[t, j] if OPOK[t, j] else ACn[t - 1, j]) for j, p in Pp.items())
        unit = navo / unit_div
        for j, p in Pp.items():
            if not p['padd']: continue
            p['padd'] = False
            if not BUYOK[t, j]: cnt['add_skip_gap'] += 1; continue
            if cash < 0.25 * unit: cnt['add_skip_cash'] += 1; continue
            amt = min(unit, cash); p['u'] += amt / (1 + buy) / AOn[t, j]; cash -= amt; p['added'] = True; cnt['adds'] += 1
        if t > t_start:
            cands = CANDS[t - 1]
            if rng is not None and len(cands): cands = rng.permutation(cands)
            for j in cands:
                if j in Pp or j in sold or not OPOK[t, j]: continue
                if GAP_UP[t, j]: cnt['skip_gap'] += 1; continue
                if len(Pp) >= slots or cash < 0.25 * unit: cnt['skip_no_slot'] += 1; continue
                amt = min(unit, cash)
                Pp[j] = {'u': amt / (1 + buy) / AOn[t, j], 'e': t, 'p0': AOn[t, j], 'px': False, 'why': None, 'added': False, 'padd': False}
                cash -= amt; cnt['entries'] += 1
        val = 0.0
        for j, p in Pp.items():
            val += p['u'] * ACn[t, j]
            if p['px']: continue
            if EXIT[t, j]: p['px'] = True; p['why'] = 'exit_signal'
            elif ts_lim and t - p['e'] >= ts_lim - 1: p['px'] = True; p['why'] = 'exit_time'
            elif agg and not p['added'] and ADJn[t, j] < p['p0'] and t < T - 1: p['padd'] = True
        nav[t - t_start] = cash + val; npos[t - t_start] = len(Pp); expo[t - t_start] = val / (cash + val)
    ix = IDX[t_start:]
    return pd.Series(nav, index=ix), pd.Series(npos, index=ix), pd.Series(expo, index=ix), cnt

# 月窗與同池等權（月換股、隔日開盤、無成本；沿用 w3_connors_double7 作法）
LIQ = UNIV.to_numpy()
_s = pd.Series(IDX, index=IDX)
MD = pd.DatetimeIndex([g[g.dt.day >= 11].iloc[0] for _, g in _s.groupby([IDX.year, IDX.month]) if (g.dt.day >= 11).any()])
MD = MD[MD >= pd.Timestamp('2004-07-01')]
gpos = {d: i for i, d in enumerate(IDX)}
ev_ew = {}
for d in MD[:-1]:
    i = gpos[d]
    if i + 1 >= T: continue
    pl = np.flatnonzero(LIQ[i] & OPOK[i + 1])
    if len(pl) >= 50: ev_ew[int(gi[i + 1])] = ('open', colj[pl])
t0f = min(ev_ew); ewf, _ = L.sim_portfolio(ev_ew, 0.0, 0.0, t0f, len(L.IDX) - 1)
EW = pd.concat([pd.Series([1.0], index=[L.IDX[t0f - 1]]), ewf]).reindex(IDX).ffill().dropna()
E50 = L.E50.reindex(IDX).ffill()
MDX = MD.append(pd.DatetimeIndex([IDX[-1]])) if MD[-1] != IDX[-1] else MD
def mwin(nav):
    nd = nav.reindex(MDX); return (nd.shift(-1) / nd - 1).iloc[:-1]
def daily_stats(nav, segs):
    r = nav.pct_change(); mask = np.zeros(len(r), bool); span = 0.0
    for a, b in segs:
        m = (r.index >= pd.Timestamp(a)) & (r.index <= pd.Timestamp(b)) & r.notna().to_numpy()
        if m.sum() == 0: continue
        first = np.argmax(m); last = len(m) - 1 - np.argmax(m[::-1])
        span += (r.index[last] - r.index[max(first - 1, 0)]).days / 365.25; mask |= m
    rr = r[mask]
    if len(rr) < 20: return None
    eq = (1 + rr).cumprod()
    o = {'CAGR': round(float(eq.iloc[-1] ** (1 / span) - 1), 4), 'vol': round(float(rr.std() * math.sqrt(252)), 4),
         'maxDD': round(float((eq / eq.cummax() - 1).min()), 4), 'years': round(span, 2)}
    pr = mwin(nav).dropna(); inw = np.zeros(len(pr), bool)
    for a, b in segs: inw |= (pr.index >= pd.Timestamp(a)) & (pr.index <= pd.Timestamp(b))
    pr = pr[inw]; o['n_months'] = int(len(pr))
    for nm_, bn in (('0050', E50), ('EW', EW)):
        if bn is nav: continue
        ex = pr - mwin(bn).reindex(pr.index)
        o[f'excess_vs_{nm_}_pct_m'] = round(float(ex.mean() * 100), 2); o[f't_vs_{nm_}_m'] = tstat(ex)
        bd = bn.pct_change().reindex(rr.index); eb = (1 + bd.fillna(0)).cumprod()
        o[f'{nm_}_CAGR'] = round(float(eb.iloc[-1] ** (1 / span) - 1), 4); o[f'{nm_}_maxDD'] = round(float((eb / eb.cummax() - 1).min()), 4)
    return o
def yearly(nav):
    r = nav.pct_change().dropna(); return {int(y): round(float((1 + g).prod() - 1) * 100, 1) for y, g in r.groupby(r.index.year)}

# ---------------- 執行全部變體 ----------------
VARS = {'AGG_NT': (True, 0, 20), 'STD_NT': (False, 0, 20), 'AGG_T10': (True, 10, 20), 'STD_T10': (False, 10, 20), 'AGG40_NT': (True, 0, 40)}
PRIMARY = 'AGG_NT'
EVS = {}; out_ev = {}; out_pt = {}; navs = {'EW_pool': EW, '0050': E50}; yearly_ev = {}
for vn, (agg, tl, ud) in VARS.items():
    if vn != 'AGG40_NT':
        df = build_events(agg, tl); EVS[vn] = df
        out_ev[vn] = {'counts': build_events.cnt, 'n_events_total': int(len(df)), 'windows': {k: ev_stats(df, sg) for k, sg in WIN.items()}}
        net = add_cost(df, 'slip0.1')
        yearly_ev[vn] = {int(y): {'n': int(len(g)), 'net_pct': round(float(net[g.index].mean() * 100), 2),
                                  'ab_gross_pct': round(float((g.raw - g.ew).mean() * 100), 2),
                                  'ab_net_pct': round(float((net[g.index] - g.ew).mean() * 100), 2)} for y, g in df.groupby(df.sig.dt.year)}
    out_pt[vn] = {}
    for cost, (b, s) in COSTS.items():
        nav, npos, expo, cnt = sim(agg, tl, b, s, unit_div=ud)
        navs[f'{vn}_{cost}'] = nav
        o = {'windows': {k: daily_stats(nav, sg) for k, sg in WIN.items()}, 'counts': cnt,
             'mean_positions': round(float(npos[npos.index >= '2005-01-01'].mean()), 2),
             'mean_exposure': round(float(expo[expo.index >= '2005-01-01'].mean()), 3),
             'yearly_pct': yearly(nav[nav.index >= '2004-12-31'])}
        if cost == 'slip0.1': o['positions_by_year'] = {int(y): round(float(g.mean()), 2) for y, g in npos.groupby(npos.index.year)}
        out_pt[vn][cost] = o
    ps = out_pt[vn]['slip0.1']['windows']; pg = out_pt[vn]['gross']['windows']
    line = f"{vn} PT S: CAGRg {pg['select_2005-01_2019-04']['CAGR']} CAGR {ps['select_2005-01_2019-04']['CAGR']} 50 {ps['select_2005-01_2019-04']['0050_CAGR']} EW {ps['select_2005-01_2019-04']['EW_CAGR']} tEW {ps['select_2005-01_2019-04']['t_vs_EW_m']}"
    line += f" | V: CAGRg {pg['valid_2019-05_2026-10']['CAGR']} CAGR {ps['valid_2019-05_2026-10']['CAGR']} 50 {ps['valid_2019-05_2026-10']['0050_CAGR']} EW {ps['valid_2019-05_2026-10']['EW_CAGR']} tEW {ps['valid_2019-05_2026-10']['t_vs_EW_m']}"
    if vn in out_ev:
        es = out_ev[vn]['windows']['select_2005-01_2019-04']; ev_ = out_ev[vn]['windows']['valid_2019-05_2026-10']
        line += f" || EV S n {es['n']} hold {es['avg_hold_days']} add {es['share_added']} abG {es['mean_ab_gross_pct']} ({es['t_month_ab_gross']}) abN {es['mean_ab_net_pct']} ({es['t_month_ab_net']})"
        line += f" V n {ev_['n']} abG {ev_['mean_ab_gross_pct']} ({ev_['t_month_ab_gross']}) abN {ev_['mean_ab_net_pct']} ({ev_['t_month_ab_net']})"
    print(line, round(time.time() - T0), flush=True)
res['events'] = out_ev; res['events_by_year_slip0.1'] = yearly_ev; res['portfolio'] = out_pt
res['EW_pool_windows'] = {k: daily_stats(EW, sg) for k, sg in WIN.items()}
res['EW_pool_yearly'] = yearly(EW[EW.index >= '2004-12-31']); res['0050_yearly'] = yearly(E50[E50.index >= '2004-12-31'])

# 隨機排序診斷（PRIMARY、滑價 0.1%）
b1, s1 = COSTS['slip0.1']; rnd = {}
for seed in range(10):
    nav, _, _, _ = sim(True, 0, b1, s1, seed=seed)
    for k in ('select_2005-01_2019-04', 'valid_2019-05_2026-10', 'y2024-01_2025-12'):
        st = daily_stats(nav, WIN[k]); rnd.setdefault(k, []).append((st['CAGR'], st['t_vs_EW_m'], st['t_vs_0050_m']))
res['primary_random_order_diag'] = {k: {'CAGR_mean': round(float(np.mean([x[0] for x in v])), 4), 'CAGR_min': round(float(np.min([x[0] for x in v])), 4),
                                        'CAGR_max': round(float(np.max([x[0] for x in v])), 4), 't_vs_EW_mean': round(float(np.mean([x[1] for x in v])), 2),
                                        't_vs_0050_mean': round(float(np.mean([x[2] for x in v])), 2)} for k, v in rnd.items()}
print('random', res['primary_random_order_diag'], flush=True)

# ---------------- 0050 單一標的（只有收盤 → 隔日收盤成交） ----------------
e = L.e50.copy(); e = e[~e.index.duplicated()].sort_index()
ei = e.index; ev = e.to_numpy(); TE = len(ev)
es = pd.Series(ev, index=ei)
s200 = es.rolling(200, min_periods=200).mean().to_numpy(); s5 = es.rolling(5, min_periods=5).mean().to_numpy()
d5 = (es < es.shift(1)).astype(float).where(es.shift(1).notna()).rolling(5, min_periods=5).sum().to_numpy()
with np.errstate(invalid='ignore'):
    SIG50 = (ev > s200) & (ev < s5) & (d5 >= 4); EX50 = ev > s5
COST50 = {'gross': (0.0, 0.0), 'rt0.5': (0.0025, 0.0025), 'etf_slip0.1': (L.FEE + 0.001, L.FEE + 0.002), 'stress': L.cost_rates('stress')}
def run50(agg, ts_lim, buy, sell):
    """日頻：權重 w 於收盤成交後生效。回傳 NAV、事件表。"""
    unit = 0.5 if agg else 1.0
    nav = np.empty(TE); cash = 1.0; sh = 0.0; st = None; ev_rows = []
    pend = None   # ('buy'|'add'|'sell')，於下一收盤執行
    for t in range(TE):
        px = ev[t]
        if pend is not None:
            navc = cash + sh * px
            if pend == 'buy':
                amt = unit * navc; sh = amt / (1 + buy) / px; cash = navc - amt
                st = {'e': t, 'p0': px, 'added': False, 'a': -1, 'pa': np.nan, 'sig': t - 1}
            elif pend == 'add':
                amt = min(unit * navc, cash); sh += amt / (1 + buy) / px; cash -= amt; st['added'] = True; st['a'] = t; st['pa'] = px
            elif pend == 'sell':
                cash += sh * px * (1 - sell); sh = 0.0
                r1 = px / st['p0'] - 1; r2 = px / st['pa'] - 1 if st['added'] else np.nan
                raw = (r1 + r2) / 2 if st['added'] else r1
                ev_rows.append((st['sig'], st['e'], t, raw, st['added'])); st = None
            pend = None
        nav[t] = cash + sh * px
        if t >= TE - 1: continue
        if st is None:
            if SIG50[t]: pend = 'buy'
        else:
            if EX50[t]: pend = 'sell'
            elif ts_lim and t - st['e'] >= ts_lim - 1: pend = 'sell'
            elif agg and not st['added'] and px < st['p0']: pend = 'add'
    df = pd.DataFrame(ev_rows, columns=['s', 'e', 'x', 'raw', 'added'])
    df['sig'] = ei[df.s.to_numpy()]; df['entry'] = ei[df.e.to_numpy()]
    df['bh'] = ev[df.x.to_numpy()] / ev[df.e.to_numpy()] - 1; df['hold'] = df.x - df.e
    return pd.Series(nav, index=ei), df
WIN50 = {k: WIN[k] for k in ('select_2005-01_2019-04', 'valid_2019-05_2026-10', 'y2024-01_2025-12', 'r21_select_2021-01_2023-12', 'r21_valid_2024-01_2026-10')}
out50 = {}
for vn, (agg, tl, _) in VARS.items():
    if vn == 'AGG40_NT': continue
    out50[vn] = {}
    for cn, (b, s) in COST50.items():
        nav, df = run50(agg, tl, b, s)
        navp = nav.reindex(IDX.union(nav.index)).ffill().reindex(IDX)
        o = {'windows': {}}
        for k, sg in WIN50.items():
            st = daily_stats(navp, sg)
            if st is not None:
                st.pop('excess_vs_EW_pct_m', None); st.pop('t_vs_EW_m', None); st.pop('EW_CAGR', None); st.pop('EW_maxDD', None)
            m = np.zeros(len(df), bool)
            for a, bb in sg: m |= ((df.sig >= pd.Timestamp(a)) & (df.sig <= pd.Timestamp(bb))).to_numpy()
            x = df[m]; net = (1 + x.raw) * (1 - s) / (1 + b) - 1; d = net - x.bh
            inpos = None
            st['trades'] = int(len(x)); st['avg_hold_days'] = round(float(x.hold.mean()), 2) if len(x) else None
            st['trade_mean_net_pct'] = round(float(net.mean() * 100), 3) if len(x) else None
            st['trade_net_minus_0050_same_window_pct'] = round(float(d.mean() * 100), 3) if len(x) else None
            st['t_month_trade_net'] = tstat(net.groupby(x.entry.dt.to_period('M')).mean()) if len(x) > 3 else None
            st['win_rate_net'] = round(float((net > 0).mean()), 3) if len(x) else None
            st['share_added'] = round(float(x.added.mean()), 3) if len(x) else None
            o['windows'][k] = st
        if cn == 'etf_slip0.1': o['yearly_pct'] = yearly(navp[navp.index >= '2004-12-31'])
        out50[vn][cn] = o
    w = out50[vn]['etf_slip0.1']['windows']
    print('0050', vn, {k: (v['CAGR'], v['0050_CAGR'], v['t_vs_0050_m'], v['trades'], v['trade_mean_net_pct']) for k, v in w.items()}, flush=True)
res['single_0050'] = out50
res['single_0050_cost_roundtrip_pct'] = {k: round((1 - (1 - s) / (1 + b)) * 100, 3) for k, (b, s) in COST50.items()}

# 資料檢查
Fn = F.to_numpy(); chk = {}
for vn in ('AGG_NT', 'STD_NT'):
    df = EVS[vn]
    f345 = np.array([bool(np.isin(Fn[r.e:r.x + 1, r.j], [3, 4, 5]).any()) for r in df.itertuples()])
    big = df[(df.raw > 1.0) | (df.raw < -0.5)]
    keep = ~f345 & (df.raw.abs() <= 1.0).to_numpy()
    st = {k: ev_stats(df[keep].reset_index(drop=True), WIN[k]) for k in ('select_2005-01_2019-04', 'valid_2019-05_2026-10')}
    chk[vn] = {'trades_with_flag345_in_hold': int(f345.sum()), 'n': int(len(df)), 'n_ret_gt_100pct': int((df.raw > 1).sum()), 'n_ret_lt_-50pct': int((df.raw < -0.5).sum()),
               'extreme_examples': big.assign(code=[cols[j] for j in big.j], sig=big.sig.astype(str))[['sig', 'code', 'raw', 'why', 'hold']].round(3).head(10).to_dict('records'),
               'ab_net_ex_flag345_extremes': {k: (v.get('mean_ab_net_pct'), v.get('t_month_ab_net')) for k, v in st.items()},
               'valid_events_touching_src3_share': round(float(df[df.sig >= '2019-05-01'].src3.mean()), 3)}
    s = df[(df.sig >= '2005-01-01') & (df.sig <= '2019-04-30')]; net = add_cost(s, 'slip0.1')
    chk[vn]['select_ab_net_by_dist5_bucket'] = {str(k): [int(len(g)), round(float((s.raw - s.ew)[g.index].mean() * 100), 3), round(float((net - s.ew)[g.index].mean() * 100), 3)]
                                                for k, g in s.groupby(pd.cut(s.dist5, [-10, -0.06, -0.04, -0.02, -0.01, 0]), observed=True)}
    for k in ('select_2005-01_2019-04', 'valid_2019-05_2026-10', 'y2024-01_2025-12'):
        chk[vn][f'breakeven_roundtrip_pct_{k}'] = out_ev[vn]['windows'][k]['mean_ab_gross_pct']
res['data_checks'] = chk

# 判定（PRIMARY 決定；其他變體列出）
def passes(vn, k):
    e_ = out_ev[vn]['windows'][k]; p = out_pt[vn]['slip0.1']['windows'][k]
    ok_ew = e_['mean_ab_net_pct'] > 0 and (e_['t_month_ab_net'] or -9) >= 2
    ok_50 = p['CAGR'] >= p['0050_CAGR']
    return {'event_ab_net_pct': e_['mean_ab_net_pct'], 'event_t_month': e_['t_month_ab_net'], 'port_CAGR': p['CAGR'], '0050_CAGR': p['0050_CAGR'],
            'EW_CAGR': p['EW_CAGR'], 'port_maxDD': p['maxDD'], 'port_t_vs_EW_m': p['t_vs_EW_m'], 'port_t_vs_0050_m': p['t_vs_0050_m'],
            'beat_EW_t2_event': bool(ok_ew), 'port_beat_EW_t2': bool((p['t_vs_EW_m'] or -9) >= 2), 'not_lose_0050': bool(ok_50), 'pass': bool(ok_ew and ok_50)}
J = {}
for vn in out_ev:
    J[vn] = {k: passes(vn, k) for k in ('select_2005-01_2019-04', 'valid_2019-05_2026-10', 'y2024-01_2025-12')}
    n = J[vn]['select_2005-01_2019-04']['pass'] + J[vn]['valid_2019-05_2026-10']['pass']
    J[vn]['verdict'] = '可用' if n == 2 else ('待觀察' if n == 1 else '不可用')
J['PRIMARY'] = PRIMARY; J['overall'] = J[PRIMARY]['verdict']
res['judgement'] = J
print('judgement', J['overall'], json.dumps(J[PRIMARY], ensure_ascii=False), {vn: J[vn]['verdict'] for vn in out_ev}, flush=True)
(HERE / 'results.json').write_text(json.dumps(res, ensure_ascii=False, indent=1, default=lambda x: None if isinstance(x, float) and not np.isfinite(x) else str(x)), encoding='utf-8')
pd.DataFrame(navs).to_parquet(HERE / 'nav.parquet')
EVS[PRIMARY].assign(code=[cols[j] for j in EVS[PRIMARY].j]).drop(columns=['j']).to_csv(HERE / 'primary_events.csv.gz', index=False)
print('done', round(time.time() - T0))
