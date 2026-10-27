#!/usr/bin/env python3
"""ConnorsRSI（CRSI）超賣＋限價低接。規則見 results.json → PREREG（prereg.py 事前寫入）。執行：python3 -I run.py
成本只從 exec_rebuild/exlib 匯入；股池／flag／處置／事件與組合引擎沿用 w3_connors_rsi2_sma5/run.py（只讀取、複製後修改）。"""
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
LIQDF = (V60 >= 1e7) & (CR >= 10) & F.isin([0, 1, 2]) & ADJ.notna()
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
lv_mat = np.maximum.accumulate(np.where(np.isfinite(ADJn), np.arange(T)[:, None], -1), axis=0)
LASTV = lv_mat[T - 1]
SRC3 = ((IDX >= pd.Timestamp('2024-02-15')) & (IDX <= pd.Timestamp('2025-09-23')))

# ---------------- 指標（只用還原收盤） ----------------
A = ADJ.ffill(limit=10)
def wilder_rsi(x, n):
    d = x.diff()
    ag = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean(); al = (-d).clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    with np.errstate(invalid='ignore', divide='ignore'):
        r = (100 - 100 / (1 + ag / al)).to_numpy()
        r = np.where((al.to_numpy() == 0) & (ag.to_numpy() > 0), 100.0, r)
        r = np.where((al.to_numpy() == 0) & (ag.to_numpy() == 0), 50.0, r)
    r[~np.isfinite(d.to_numpy())] = np.nan
    return r
RSI3 = wilder_rsi(A, 3)
An = A.to_numpy(); dA = np.vstack([np.full((1, N), np.nan), np.diff(An, axis=0)])
STK = np.full((T, N), np.nan); prev = np.zeros(N)
for t in range(T):
    d = dA[t]
    s = np.where(d > 0, np.where(prev > 0, prev + 1, 1.0), np.where(d < 0, np.where(prev < 0, prev - 1, -1.0), 0.0))
    s = np.where(np.isfinite(d), s, np.nan); STK[t] = s; prev = np.nan_to_num(s)
RSIS = wilder_rsi(pd.DataFrame(STK, index=IDX, columns=cols), 2)
with np.errstate(invalid='ignore', divide='ignore'):
    R1 = An / np.vstack([np.full((1, N), np.nan), An[:-1]]) - 1
CNT = np.zeros((T, N)); VAL = np.ones((T, N), bool); VAL[:100] = False
for k in range(1, 101):
    with np.errstate(invalid='ignore'):
        CNT[k:] += (R1[:-k] < R1[k:])
    VAL[k:] &= np.isfinite(R1[:-k])
PR = np.where(VAL & np.isfinite(R1), CNT, np.nan)  # = 比例×100（100 個值）
CRSI = (RSI3 + RSIS + PR) / 3
with np.errstate(invalid='ignore', divide='ignore'):
    HV = (np.log(A).diff().rolling(100, min_periods=100).std() * math.sqrt(252)).to_numpy()
SMA200 = A.rolling(200, min_periods=200).mean().to_numpy()
SMA5 = A.rolling(5, min_periods=5).mean().to_numpy()
HAS = np.isfinite(ADJn)
FL5 = FL.rolling(5, min_periods=1).max().to_numpy() > 0
LIQ = LIQDF.to_numpy()
with np.errstate(invalid='ignore'):
    POOL = LIQ & (HV > 0.30)
    BASE = POOL & HAS & np.isfinite(CRSI) & ~FL5 & ~DISP_TT1
    F200 = BASE & (ADJn > SMA200)
    SIG = {'F200_c10': F200 & (CRSI < 10), 'F200_c5': F200 & (CRSI < 5), 'NOF_c10': BASE & (CRSI < 10), 'NOF_c5': BASE & (CRSI < 5)}
    E70 = HAS & (CRSI > 70); E50 = HAS & (CRSI > 50); ESMA = HAS & (ADJn > SMA5)
EXIT = {'E70': E70, 'E50': E50, 'SMA5': ESMA, 'E70orSMA5': E70 | ESMA}
LP = 0.04
def lim_mat(lp): return ACn * (1 - lp)          # 訊號日 t 的限價（還原）
LIMM = lim_mat(LP)
_m = IDX >= '2005-01-01'
res['meta'] = {'bad_days_dropped': BAD, 'n_stocks': N, 'market_flag4_days_ignored': [str(x.date()) for x in IDX[mkt4.to_numpy()]],
               'signal_days': {k: int(v.sum()) for k, v in SIG.items()},
               'mean_signals_per_day': {k: round(float(v.sum(1)[_m].mean()), 2) for k, v in SIG.items()},
               'pool_size_mean_LIQ': round(float(LIQ.sum(1)[_m].mean()), 1), 'pool_size_mean_LIQ_HV30': round(float(POOL.sum(1)[_m].mean()), 1),
               'share_LIQ_with_HV_gt30': round(float(POOL.sum() / max(LIQ[_m].sum(), 1)), 3),
               'crsi_median_in_pool': round(float(np.nanmedian(np.where(POOL, CRSI, np.nan))), 1),
               'crsi_share_lt10_in_pool': round(float(np.nanmean(np.where(POOL & np.isfinite(CRSI), CRSI < 10, np.nan))), 4)}
print('setup', round(time.time() - T0), res['meta'], flush=True)

# 同池等權（開盤對開盤，事件用）：rule_pool 與 LIQ
def lew(pool):
    oo = (AO_raw.shift(-1) / AO_raw - 1).where(pd.DataFrame(pool, index=IDX, columns=cols).shift(1, fill_value=False)).clip(-0.5, 1.0)
    return np.concatenate([[0.0], np.cumsum(np.log1p(oo.mean(axis=1).fillna(0).to_numpy()))])
LEW_HV = lew(POOL); LEW_LQ = lew(LIQ)
E50n = L.e50.reindex(IDX.union(L.e50.index)).ffill().reindex(IDX).to_numpy()

def exit_px(x, j):
    if x > LASTV[j]: return ACn[T - 1, j], min(LASTV[j] + 1, T - 1), True
    k = x
    while k < T and not SELLABLE[k, j]:
        if k > LASTV[j]: return ACn[T - 1, j], min(LASTV[j] + 1, T - 1), True
        k += 1
    if k >= T: return ACn[T - 1, j], T - 1, False
    return AOn[k, j], k, False

def build_events(sig, ex, fill, lp=LP, ts_lim=10):
    EX = EXIT[ex]; LM = lim_mat(lp); rows = []; c = {'signals_considered': 0, 'no_open': 0, 'blocked_gap_up': 0, 'no_fill': 0, 'fill_open': 0, 'fill_limit_touch': 0}
    for j in range(N):
        sigs = np.flatnonzero(sig[:, j]); free = -1
        for t in sigs:
            if t < free: continue
            e = t + 1
            if e >= T: break
            c['signals_considered'] += 1
            if not OPOK[e, j]: c['no_open'] += 1; continue
            if fill == 'B':
                if GAP_UP[e, j]: c['blocked_gap_up'] += 1; continue
                p0 = AOn[e, j]; how = 'open'
            else:
                if AOn[e, j] <= LM[t, j]: p0 = AOn[e, j]; how = 'open'
                elif fill == 'A2' and HAS[e, j] and ADJn[e, j] <= LM[t, j]: p0 = LM[t, j]; how = 'limit'
                else: c['no_fill'] += 1; continue
            c['fill_open' if how == 'open' else 'fill_limit_touch'] += 1
            end = min(T, e + ts_lim)
            hit = np.flatnonzero(EX[e:end, j])
            if len(hit): x = e + hit[0] + 1; why = 'signal'
            else: x = e + ts_lim; why = 'time'
            if x >= T: px, xs, dl = ACn[T - 1, j], T - 1, False; why = 'open_at_end'
            else:
                px, xs, dl = exit_px(x, j)
                if dl: why = 'delist'
            rows.append((j, t, e, xs, px, p0, why, dl, how))
            free = xs if not dl else T
    df = pd.DataFrame(rows, columns=['j', 's', 'e', 'x', 'px', 'p0', 'why', 'delist', 'how'])
    df['raw'] = df.px / df.p0 - 1; df['hold'] = df.x - df.e
    df['ew'] = np.expm1(LEW_HV[df.x.to_numpy()] - LEW_HV[df.e.to_numpy()])
    df['ewl'] = np.expm1(LEW_LQ[df.x.to_numpy()] - LEW_LQ[df.e.to_numpy()])
    df['b50'] = E50n[np.maximum(df.x - 1, df.s)] / E50n[df.s] - 1
    df['gap'] = AOn[df.e, df.j] / ACn[df.s, df.j] - 1
    df['src3'] = [bool(SRC3[a:b + 1].any()) for a, b in zip(df.s, df.x)]
    df['sig'] = IDX[df.s.to_numpy()]; df['entry'] = IDX[df.e.to_numpy()]
    df['crsi'] = CRSI[df.s, df.j]
    build_events.cnt = c
    return df

def add_cost(df, cost):
    b, s = (0.0, 0.0) if cost == 'gross' else L.cost_rates(cost)
    return (1 + df.raw) * (1 - s) / (1 + b) - 1

WIN = {'select_2005-01_2019-04': [('2005-01-01', '2019-04-30')], 'valid_2019-05_2026-10': [('2019-05-01', '2026-10-31')],
       'y2024-01_2025-12': [('2024-01-01', '2025-12-31')],
       'select_sub_2005-2011': [('2005-01-01', '2011-12-31')], 'select_sub_2012-2019-04': [('2012-01-01', '2019-04-30')],
       'valid_survivor_2019-05_2023-12': [('2019-05-01', '2023-12-31')],
       'valid_officialPx_only': [('2019-05-01', '2024-02-14'), ('2025-09-24', '2026-10-31')],
       'r21_select_2021-01_2023-12': [('2021-01-01', '2023-12-31')], 'r21_valid_2024-01_2026-10': [('2024-01-01', '2026-10-31')]}
MAINW = ('select_2005-01_2019-04', 'valid_2019-05_2026-10', 'y2024-01_2025-12')

def ev_stats(df, segs):
    m = np.zeros(len(df), bool)
    for a, b in segs: m |= ((df.sig >= pd.Timestamp(a)) & (df.sig <= pd.Timestamp(b))).to_numpy()
    x = df[m]
    if len(x) < 10: return {'n': int(len(x)), 'n_months': int(x.entry.dt.to_period('M').nunique()) if len(x) else 0}
    net = add_cost(x, 0.001); ab = net - x.ew; abg = x.raw - x.ew; abs_ = add_cost(x, 'stress') - x.ew; ab50 = net - x.b50; abl = net - x.ewl
    em = x.entry.dt.to_period('M'); gm = ab.groupby(em).mean()
    return {'n': int(len(x)), 'n_months': int(gm.size), 'avg_hold_days': round(float(x.hold.mean()), 2),
            'mean_raw_pct': round(float(x.raw.mean() * 100), 3), 'mean_net_pct': round(float(net.mean() * 100), 3),
            'median_net_pct': round(float(net.median() * 100), 3), 'win_rate_net': round(float((net > 0).mean()), 3),
            'mean_EW_same_window_pct': round(float(x.ew.mean() * 100), 3),
            'mean_ab_gross_pct': round(float(abg.mean() * 100), 3), 't_month_ab_gross': tstat(abg.groupby(em).mean()),
            'mean_ab_net_pct': round(float(ab.mean() * 100), 3), 'month_cluster_mean_ab_net_pct': round(float(gm.mean() * 100), 3),
            't_month_ab_net': tstat(gm), 't_date_ab_net_ref': tstat(ab.groupby(x.entry).mean()),
            'mean_ab_net_stress_pct': round(float(abs_.mean() * 100), 3), 't_month_ab_net_stress': tstat(abs_.groupby(em).mean()),
            'mean_ab_net_vs_EWliq_pct': round(float(abl.mean() * 100), 3), 't_month_vs_EWliq': tstat(abl.groupby(em).mean()),
            'mean_net_vs_0050_pct': round(float(ab50.mean() * 100), 3), 't_month_vs_0050': tstat(ab50.groupby(em).mean()),
            'mean_gap_pct': round(float(x.gap.mean() * 100), 3),
            'exit_mix': {k: round(float(v), 3) for k, v in x.why.value_counts(normalize=True).items()},
            'fill_mix': {k: round(float(v), 3) for k, v in x.how.value_counts(normalize=True).items()},
            'n_delist': int(x.delist.sum()), 'p01_raw_pct': round(float(x.raw.quantile(0.01) * 100), 2), 'p99_raw_pct': round(float(x.raw.quantile(0.99) * 100), 2)}

# ---------------- 10 槽組合 ----------------
t_start = int(np.searchsorted(IDX, pd.Timestamp('2004-07-01')))
V60n = V60.to_numpy()
def cand_lists(sig):
    out = []
    for t in range(T):
        js = np.flatnonzero(sig[t])
        if len(js): js = js[np.lexsort((-np.nan_to_num(V60n[t, js]), CRSI[t, js]))]
        out.append(js)
    return out
CANDS = {k: cand_lists(v) for k, v in SIG.items()}

def sim(sk, ex, fill, buy, sell, slots=10, ts_lim=10, strict=False):
    EX = EXIT[ex]
    cash = 1.0; Pp = {}
    nav = np.empty(T - t_start); npos = np.empty(T - t_start)
    cnt = {'entries': 0, 'skip_no_slot': 0, 'skip_gap': 0, 'no_fill': 0, 'exit_signal': 0, 'exit_time': 0, 'delist': 0, 'exit_delayed': 0}
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
        unit = navo / slots
        if t > t_start:
            cands = [j for j in CANDS[sk][t - 1] if j not in Pp and j not in sold]
            if strict: cands = cands[:max(slots - len(Pp), 0)]
            for j in cands:
                if not OPOK[t, j]: continue
                if fill == 'B':
                    if GAP_UP[t, j]: cnt['skip_gap'] += 1; continue
                elif AOn[t, j] > LIMM[t - 1, j]: cnt['no_fill'] += 1; continue
                if len(Pp) >= slots or cash < 0.25 * unit: cnt['skip_no_slot'] += 1; continue
                amt = min(unit, cash)
                Pp[j] = {'u': amt / (1 + buy) / AOn[t, j], 'e': t, 'px': False, 'why': None}
                cash -= amt; cnt['entries'] += 1
        val = 0.0
        for j, p in Pp.items():
            val += p['u'] * ACn[t, j]
            if p['px']: continue
            if EX[t, j]: p['px'] = True; p['why'] = 'exit_signal'
            elif ts_lim and t - p['e'] >= ts_lim - 1: p['px'] = True; p['why'] = 'exit_time'
        nav[t - t_start] = cash + val; npos[t - t_start] = len(Pp)
    ix = IDX[t_start:]
    return pd.Series(nav, index=ix), pd.Series(npos, index=ix), cnt

# 月窗與同池等權（月換股、隔日開盤、無成本）
_s = pd.Series(IDX, index=IDX)
MD = pd.DatetimeIndex([g[g.dt.day >= 11].iloc[0] for _, g in _s.groupby([IDX.year, IDX.month]) if (g.dt.day >= 11).any()])
MD = MD[MD >= pd.Timestamp('2004-07-01')]
gpos = {d: i for i, d in enumerate(IDX)}
def ew_nav(pool):
    ev = {}
    for d in MD[:-1]:
        i = gpos[d]
        if i + 1 >= T: continue
        pl = np.flatnonzero(pool[i] & OPOK[i + 1])
        if len(pl) >= 30: ev[int(gi[i + 1])] = ('open', colj[pl])
    t0f = min(ev); ewf, _ = L.sim_portfolio(ev, 0.0, 0.0, t0f, len(L.IDX) - 1)
    return pd.concat([pd.Series([1.0], index=[L.IDX[t0f - 1]]), ewf]).reindex(IDX).ffill().dropna()
EW = ew_nav(POOL); EWL = ew_nav(LIQ)
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
    for nm_, bn in (('0050', E50), ('EW', EW), ('EWliq', EWL)):
        if bn is nav: continue
        ex = pr - mwin(bn).reindex(pr.index)
        o[f'excess_vs_{nm_}_pct_m'] = round(float(ex.mean() * 100), 2); o[f't_vs_{nm_}_m'] = tstat(ex)
        bd = bn.pct_change().reindex(rr.index); eb = (1 + bd.fillna(0)).cumprod()
        o[f'{nm_}_CAGR'] = round(float(eb.iloc[-1] ** (1 / span) - 1), 4); o[f'{nm_}_maxDD'] = round(float((eb / eb.cummax() - 1).min()), 4)
    return o
def yearly(nav):
    r = nav.pct_change().dropna(); return {int(y): round(float((1 + g).prod() - 1) * 100, 1) for y, g in r.groupby(r.index.year)}

# ---------------- 執行全部 32 組 ----------------
VARS = {f'{sk}_{ex}_{fl}': (sk, ex, fl) for sk in SIG for ex in EXIT for fl in ('A', 'B')}
PRIMARY = 'F200_c10_E70_A'
b1, s1 = L.cost_rates(0.001)
EVS = {}; out_ev = {}; out_pt = {}; navs = {'EW_pool_HV30': EW, 'EW_liq': EWL, '0050': E50}; yearly_ev = {}
for vn, (sk, ex, fl) in VARS.items():
    df = build_events(SIG[sk], ex, fl); EVS[vn] = df
    out_ev[vn] = {'counts': build_events.cnt, 'n_events_total': int(len(df)), 'windows': {k: ev_stats(df, sg) for k, sg in WIN.items()}}
    net = add_cost(df, 0.001)
    yearly_ev[vn] = {int(y): {'n': int(len(g)), 'net_pct': round(float(net[g.index].mean() * 100), 2),
                              'ab_net_pct': round(float((net[g.index] - g.ew).mean() * 100), 2)} for y, g in df.groupby(df.sig.dt.year)}
    out_pt[vn] = {}
    for cost in ((('gross', 0.001, 'stress')) if vn in (PRIMARY, 'NOF_c10_E70_A', 'F200_c10_E70_B', 'NOF_c10_E70_B') else (0.001,)):
        b, s = (0.0, 0.0) if cost == 'gross' else L.cost_rates(cost)
        nav, npos, cnt = sim(sk, ex, fl, b, s)
        navs[f'{vn}_{cost}'] = nav
        o = {'counts': cnt, 'mean_positions': round(float(npos[npos.index >= '2005-01-01'].mean()), 2),
             'windows': {k: daily_stats(nav, sg) for k, sg in WIN.items()}, 'yearly_pct': yearly(nav[nav.index >= '2004-12-31'])}
        if cost == 0.001: o['positions_by_year'] = {int(y): round(float(g.mean()), 2) for y, g in npos.groupby(npos.index.year)}
        out_pt[vn][str(cost)] = o
    es, ev_ = out_ev[vn]['windows'][MAINW[0]], out_ev[vn]['windows'][MAINW[1]]
    ps, pv = out_pt[vn]['0.001']['windows'][MAINW[0]], out_pt[vn]['0.001']['windows'][MAINW[1]]
    print(vn, 'n', len(df), '| S n', es.get('n'), 'ab', es.get('mean_ab_net_pct'), 't', es.get('t_month_ab_net'), 'gross', es.get('mean_ab_gross_pct'), es.get('t_month_ab_gross'),
          'CAGR', ps['CAGR'], '50', ps['0050_CAGR'], 'EW', ps['EW_CAGR'], 'tEW', ps['t_vs_EW_m'], 'pos', out_pt[vn]['0.001']['mean_positions'],
          '| V n', ev_.get('n'), 'ab', ev_.get('mean_ab_net_pct'), 't', ev_.get('t_month_ab_net'), 'gross', ev_.get('mean_ab_gross_pct'), 'CAGR', pv['CAGR'], '50', pv['0050_CAGR'], 'EW', pv['EW_CAGR'], 'tEW', pv['t_vs_EW_m'],
          round(time.time() - T0), flush=True)
res['events'] = out_ev; res['events_by_year_slip0.1'] = yearly_ev; res['portfolio'] = out_pt
res['EW_pool_HV30_windows'] = {k: daily_stats(EW, sg) for k, sg in WIN.items()}
res['EW_liq_windows'] = {k: daily_stats(EWL, sg) for k, sg in WIN.items()}
res['EW_pool_HV30_yearly'] = yearly(EW[EW.index >= '2004-12-31']); res['EW_liq_yearly'] = yearly(EWL[EWL.index >= '2004-12-31'])
res['0050_yearly'] = yearly(E50[E50.index >= '2004-12-31'])

# ---------------- 事前列出的診斷 ----------------
diag = {}
# fill A2（確定觸價成交下界）：PRIMARY 與 NOF 版
for sk in ('F200_c10', 'NOF_c10'):
    df = build_events(SIG[sk], 'E70', 'A2')
    diag[f'{sk}_E70_A2'] = {'counts': build_events.cnt, 'windows': {k: ev_stats(df, WIN[k]) for k in MAINW}}
# 限價深度 2/4/6/8/10%（fill A 與 A2）
for lp in (0.02, 0.04, 0.06, 0.08, 0.10):
    for fl in ('A', 'A2'):
        df = build_events(SIG['F200_c10'], 'E70', fl, lp=lp)
        diag[f'limit_{int(lp*100)}pct_F200_c10_E70_{fl}'] = {'counts': build_events.cnt,
            'windows': {k: {kk: ev_stats(df, WIN[k]).get(kk) for kk in ('n', 'n_months', 'mean_ab_gross_pct', 'mean_ab_net_pct', 't_month_ab_net', 'mean_net_pct', 'win_rate_net')} for k in MAINW}}
# 嚴格掛單版組合
for vn in (PRIMARY, 'NOF_c10_E70_A'):
    sk, ex, fl = VARS[vn]; nav, npos, cnt = sim(sk, ex, fl, b1, s1, strict=True)
    diag[f'{vn}_strict_orders_portfolio'] = {'counts': cnt, 'mean_positions': round(float(npos[npos.index >= '2005-01-01'].mean()), 2),
                                            'windows': {k: daily_stats(nav, WIN[k]) for k in MAINW}}
# fill 成交率（訊號 → 成交）
diag['fill_rate'] = {vn: round(out_ev[vn]['counts']['fill_open'] / max(out_ev[vn]['counts']['signals_considered'], 1), 4) for vn in VARS if vn.endswith('_A')}
# 依 CRSI 分層（挑選期，B 版，看訊號本身）
for vn in ('F200_c10_E70_B', 'NOF_c10_E70_B', PRIMARY):
    df = EVS[vn]; s = df[(df.sig >= '2005-01-01') & (df.sig <= '2019-04-30')]; net = add_cost(s, 0.001)
    diag[f'{vn}_select_ab_by_crsi_bucket'] = {str(k): [int(len(g)), round(float((net - s.ew)[g.index].mean() * 100), 3), round(float((s.raw - s.ew)[g.index].mean() * 100), 3)]
                                              for k, g in s.groupby(pd.cut(s.crsi, [0, 2.5, 5, 7.5, 10]), observed=True)}
res['diagnostics'] = diag
print('diag done', round(time.time() - T0), flush=True)

# 資料檢查
Fn = F.to_numpy(); chk = {}
for vn in (PRIMARY, 'F200_c10_E70_B', 'NOF_c10_E70_A'):
    df = EVS[vn]
    f345 = np.array([bool(np.isin(Fn[r.e:r.x + 1, r.j], [3, 4, 5]).any()) for r in df.itertuples()])
    big = df[(df.raw > 1.0) | (df.raw < -0.5)]
    keep = ~f345 & (df.raw.abs() <= 1.0).to_numpy()
    st = {k: ev_stats(df[keep].reset_index(drop=True), WIN[k]) for k in MAINW[:2]}
    chk[vn] = {'trades_with_flag345_in_hold': int(f345.sum()), 'n': int(len(df)), 'n_ret_gt_100pct': int((df.raw > 1).sum()), 'n_ret_lt_-50pct': int((df.raw < -0.5).sum()),
               'extreme_examples': big.assign(code=[cols[j] for j in big.j], sig=big.sig.astype(str))[['sig', 'code', 'raw', 'why', 'hold']].round(3).head(10).to_dict('records'),
               'ab_net_ex_flag345_extremes': {k: (v.get('mean_ab_net_pct'), v.get('t_month_ab_net')) for k, v in st.items()},
               'valid_events_touching_src3_share': round(float(df[df.sig >= '2019-05-01'].src3.mean()), 3) if (df.sig >= '2019-05-01').any() else None}
res['data_checks'] = chk

# 判定
def passes(vn, k):
    e = out_ev[vn]['windows'][k]; p = out_pt[vn]['0.001']['windows'][k]
    insufficient = e.get('n', 0) < 100 or e.get('n_months', 0) < 24
    if 'mean_ab_net_pct' not in e:
        return {'n': e.get('n'), 'insufficient': True, 'pass': False}
    ok_ew = e['mean_ab_net_pct'] > 0 and (e['t_month_ab_net'] if e['t_month_ab_net'] == e['t_month_ab_net'] else -9) >= 2
    ok_50 = p['CAGR'] >= p['0050_CAGR']
    return {'n': e['n'], 'n_months': e['n_months'], 'insufficient': bool(insufficient), 'event_ab_net_pct': e['mean_ab_net_pct'], 'event_t_month': e['t_month_ab_net'],
            'event_ab_gross_pct': e['mean_ab_gross_pct'], 'port_CAGR': p['CAGR'], 'port_maxDD': p['maxDD'], '0050_CAGR': p['0050_CAGR'],
            'EW_CAGR': p['EW_CAGR'], 'port_t_vs_EW_m': p['t_vs_EW_m'], 'port_t_vs_0050_m': p['t_vs_0050_m'],
            'beat_EW_t2_event': bool(ok_ew), 'port_beat_EW_t2': bool((p['t_vs_EW_m'] or -9) >= 2), 'not_lose_0050': bool(ok_50), 'pass': bool(ok_ew and ok_50 and not insufficient)}
J = {vn: {k: passes(vn, k) for k in MAINW} for vn in VARS}
for vn in VARS:
    s_, v_ = J[vn][MAINW[0]], J[vn][MAINW[1]]
    if s_['insufficient'] and v_['insufficient']: J[vn]['verdict'] = '無法測試'
    else:
        n = s_['pass'] + v_['pass']; J[vn]['verdict'] = '可用' if n == 2 else ('待觀察' if n == 1 else '不可用')
J['PRIMARY'] = PRIMARY; J['overall'] = J[PRIMARY]['verdict']
J['secondary_ref'] = {v: J[v]['verdict'] for v in ('NOF_c10_E70_A', 'F200_c10_E70_B', 'NOF_c10_E70_B')}
J['verdict_counts'] = pd.Series([J[v]['verdict'] for v in VARS]).value_counts().to_dict()
res['judgement'] = J
print('judgement', J['overall'], json.dumps(J[PRIMARY], ensure_ascii=False), J['verdict_counts'], flush=True)
(HERE / 'results.json').write_text(json.dumps(res, ensure_ascii=False, indent=1, default=lambda x: None if isinstance(x, float) and not np.isfinite(x) else str(x)), encoding='utf-8')
pd.DataFrame(navs).to_parquet(HERE / 'nav.parquet')
for vn in (PRIMARY, 'F200_c10_E70_B'):
    EVS[vn].assign(code=[cols[j] for j in EVS[vn].j]).drop(columns=['j']).to_csv(HERE / f'events_{vn}.csv.gz', index=False)
print('done', round(time.time() - T0))
