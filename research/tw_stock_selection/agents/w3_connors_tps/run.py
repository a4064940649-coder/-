#!/usr/bin/env python3
"""Connors TPS（時間、價格、分批加碼）。規則見 results.json → PREREG（prereg.py 事前寫入）。執行：python3 -I run.py
成本與月換股等權引擎從 exec_rebuild/exlib 匯入；股池／flag／處置／統計沿用 w3_connors_crsi/run.py（複製後修改，不改原檔）。"""
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
BUYOK = OPOK & ~GAP_UP
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
RSI2 = wilder_rsi(A, 2)
SMA200 = A.rolling(200, min_periods=200).mean().to_numpy()
An = A.to_numpy()
HAS = np.isfinite(ADJn)
HAS_prev = np.vstack([np.zeros((1, N), bool), HAS[:-1]])
RSI_prev = np.vstack([np.full((1, N), np.nan), RSI2[:-1]])
FL5 = FL.rolling(5, min_periods=1).max().to_numpy() > 0
LIQ = LIQDF.to_numpy()
with np.errstate(invalid='ignore'):
    SIG = LIQ & HAS & HAS_prev & ~FL5 & ~DISP_TT1 & (ADJn > SMA200) & (RSI2 < 25) & (RSI_prev < 25)
    EXIT = HAS & (RSI2 > 70)
    ABOVE = HAS & (ADJn >= SMA200)            # P 版加碼濾網（需真實收盤）
    ABOVE_D = np.isfinite(An) & (An >= SMA200)  # D 版濾網（用 ffill 收盤）
_m = IDX >= '2005-01-01'
res['meta'] = {'bad_days_dropped': BAD, 'n_stocks': N, 'market_flag4_days_ignored': [str(x.date()) for x in IDX[mkt4.to_numpy()]],
               'signal_days': int(SIG.sum()), 'mean_signals_per_day': round(float(SIG.sum(1)[_m].mean()), 2),
               'pool_size_mean_LIQ': round(float(LIQ.sum(1)[_m].mean()), 1),
               'share_pool_above_sma200': round(float(np.nanmean(np.where(LIQ & np.isfinite(SMA200), ADJn > SMA200, np.nan)[_m])), 3)}
print('setup', round(time.time() - T0), res['meta'], flush=True)

def lew(pool):
    oo = (AO_raw.shift(-1) / AO_raw - 1).where(pd.DataFrame(pool, index=IDX, columns=cols).shift(1, fill_value=False)).clip(-0.5, 1.0)
    return np.concatenate([[0.0], np.cumsum(np.log1p(oo.mean(axis=1).fillna(0).to_numpy()))])
LEW_LQ = lew(LIQ)            # LEW[k] = 累積 log 報酬，開盤 0 → 開盤 k
E50n = L.e50.reindex(IDX.union(L.e50.index)).ffill().reindex(IDX).to_numpy()

def exit_px(x, j):
    if x > LASTV[j]: return ACn[T - 1, j], min(LASTV[j] + 1, T - 1), True
    k = x
    while k < T and not SELLABLE[k, j]:
        if k > LASTV[j]: return ACn[T - 1, j], min(LASTV[j] + 1, T - 1), True
        k += 1
    if k >= T: return ACn[T - 1, j], T - 1, False
    return AOn[k, j], k, False

FR = {'P': (0.1, 0.2, 0.3, 0.4), 'D': (0.1, 0.2, 0.3, 0.4), 'ALL': (1.0,), 'ONE': (0.1,)}
DDAY = {2: 0.2, 3: 0.3, 4: 0.4}      # D 版：e+2/e+3/e+4 開盤（第 3/4/5 天）

def build_events(scheme, filt, ts_lim, mode='open'):
    """mode='open'：t+1 開盤成交（可成交）；'close'：決策日收盤成交（不可成交，只作對照）。"""
    rows = []; c = {'signals_considered': 0, 'no_open': 0, 'blocked_gap_up': 0, 'add_blocked': 0, 'add_filtered': 0}
    for j in range(N):
        sigs = np.flatnonzero(SIG[:, j]); free = -1
        for t in sigs:
            if t < free: continue
            e = t + 1
            if e >= T: break
            c['signals_considered'] += 1
            if mode == 'open':
                if not OPOK[e, j]: c['no_open'] += 1; continue
                if GAP_UP[e, j]: c['blocked_gap_up'] += 1; continue
            fx = (lambda d: AOn[d, j]) if mode == 'open' else (lambda d: ACn[d - 1, j])
            end = min(T, e + ts_lim) if ts_lim else T
            hit = np.flatnonzero(EXIT[e:end, j])
            if len(hit): x = e + hit[0] + 1; why = 'signal'
            else: x = e + ts_lim if ts_lim else T; why = 'time'
            # 加碼
            tr = [(e, fx(e), FR[scheme][0])]
            if scheme == 'P':
                k = 1; last = tr[0][1]
                for s in range(e, min(x, T) - 1):
                    if k >= 4: break
                    if not HAS[s, j] or not (ADJn[s, j] < last): continue
                    if filt and not ABOVE[s, j]: c['add_filtered'] += 1; continue
                    d = s + 1
                    if mode == 'open' and not BUYOK[d, j]: c['add_blocked'] += 1; continue
                    p = fx(d); tr.append((d, p, FR['P'][k])); last = p; k += 1
            elif scheme == 'D':
                for off, fr in DDAY.items():
                    d = e + off
                    if d >= min(x, T): break
                    if filt and not ABOVE_D[d - 1, j]: c['add_filtered'] += 1; continue
                    if mode == 'open' and not BUYOK[d, j]: c['add_blocked'] += 1; continue
                    tr.append((d, fx(d), fr))
            if x >= T: px, xs, dl = ACn[T - 1, j], T - 1, False; why = 'open_at_end'
            else:
                if mode == 'open': px, xs, dl = exit_px(x, j)
                else: px, xs, dl = ACn[x - 1, j], x, False
                if dl: why = 'delist'
            di = np.array([a for a, _, _ in tr]); pi = np.array([b for _, b, _ in tr]); fi = np.array([f for _, _, f in tr])
            Cf = fi.sum(); U = (fi / pi).sum()
            ew = (fi * np.expm1(LEW_LQ[xs] - LEW_LQ[di])).sum() / Cf
            b50 = (fi * (E50n[max(xs - 1, t)] / E50n[di - 1] - 1)).sum() / Cf
            capdays = (fi * (xs - di)).sum()          # 資金×天
            rows.append((j, t, e, xs, px, U, Cf, len(tr), why, dl, ew, b50, capdays))
            free = xs if not dl else T
    df = pd.DataFrame(rows, columns=['j', 's', 'e', 'x', 'px', 'U', 'Cf', 'ntr', 'why', 'delist', 'ew', 'b50', 'capdays'])
    df['raw'] = df.U * df.px / df.Cf - 1; df['hold'] = df.x - df.e
    df['src3'] = [bool(SRC3[a:b + 1].any()) for a, b in zip(df.s, df.x)]
    df['sig'] = IDX[df.s.to_numpy()]; df['entry'] = IDX[df.e.to_numpy()]
    df['rsi'] = RSI2[df.s, df.j]
    build_events.cnt = c
    return df

def add_cost(df, cost):
    b, s = (0.0, 0.0) if cost == 'gross' else L.cost_rates(cost)
    return df.U * df.px * (1 - s) / (1 + b) / df.Cf - 1

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
    net = add_cost(x, 0.001); ab = net - x.ew; abg = x.raw - x.ew; abs_ = add_cost(x, 'stress') - x.ew; ab50 = net - x.b50
    em = x.entry.dt.to_period('M'); gm = ab.groupby(em).mean()
    budget_ret = net * x.Cf      # 損益／滿倉預算（預算=1）
    return {'n': int(len(x)), 'n_months': int(gm.size), 'avg_hold_days': round(float(x.hold.mean()), 2),
            'avg_tranches': round(float(x.ntr.mean()), 2), 'tranche_dist': {int(k): round(float(v), 3) for k, v in x.ntr.value_counts(normalize=True).sort_index().items()},
            'avg_invested_frac_of_budget': round(float(x.Cf.mean()), 3),
            'mean_raw_pct': round(float(x.raw.mean() * 100), 3), 'mean_net_pct': round(float(net.mean() * 100), 3),
            'median_net_pct': round(float(net.median() * 100), 3), 'win_rate_net': round(float((net > 0).mean()), 3),
            'mean_net_on_full_budget_pct_ref': round(float(budget_ret.mean() * 100), 3),
            'capital_weighted_net_pct': round(float((net * x.Cf).sum() / x.Cf.sum() * 100), 3),
            'mean_EW_same_window_pct': round(float(x.ew.mean() * 100), 3),
            'mean_ab_gross_pct': round(float(abg.mean() * 100), 3), 't_month_ab_gross': tstat(abg.groupby(em).mean()),
            'mean_ab_net_pct': round(float(ab.mean() * 100), 3), 'month_cluster_mean_ab_net_pct': round(float(gm.mean() * 100), 3),
            't_month_ab_net': tstat(gm), 't_date_ab_net_ref': tstat(ab.groupby(x.entry).mean()),
            'mean_ab_net_stress_pct': round(float(abs_.mean() * 100), 3), 't_month_ab_net_stress': tstat(abs_.groupby(em).mean()),
            'mean_net_vs_0050_pct': round(float(ab50.mean() * 100), 3), 't_month_vs_0050': tstat(ab50.groupby(em).mean()),
            'breakeven_roundtrip_cost_pct': round(float(abg.mean() * 100), 3),
            'exit_mix': {k: round(float(v), 3) for k, v in x.why.value_counts(normalize=True).items()},
            'n_delist': int(x.delist.sum()), 'p01_raw_pct': round(float(x.raw.quantile(0.01) * 100), 2), 'p99_raw_pct': round(float(x.raw.quantile(0.99) * 100), 2)}

# ---------------- 10 槽組合 ----------------
t_start = int(np.searchsorted(IDX, pd.Timestamp('2004-07-01')))
V60n = V60.to_numpy()
CANDS = []
for t in range(T):
    js = np.flatnonzero(SIG[t])
    if len(js): js = js[np.lexsort((-np.nan_to_num(V60n[t, js]), RSI2[t, js]))]
    CANDS.append(js)

def sim(scheme, filt, ts_lim, buy, sell, slots=10):
    fr = FR[scheme]; cap = sum(fr) if scheme != 'ONE' else 0.1
    cash = 1.0; Pp = {}
    nav = np.empty(T - t_start); npos = np.empty(T - t_start); inv = np.empty(T - t_start)
    cnt = {'entries': 0, 'adds': 0, 'add_blocked': 0, 'add_filtered': 0, 'add_cash_short': 0, 'skip_no_slot_or_cash': 0, 'skip_gap': 0,
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
        # 加碼（前一日收盤決定）
        for j, p in Pp.items():
            if p['add'] is None: continue
            f = p['add']; p['add'] = None
            if not BUYOK[t, j]: cnt['add_blocked'] += 1; continue
            amt = min(f * p['B'], cash)
            if amt <= 0: cnt['add_cash_short'] += 1; continue
            p['u'] += amt / (1 + buy) / AOn[t, j]; cash -= amt; p['cf'] += f; p['k'] += 1; p['last'] = AOn[t, j]; cnt['adds'] += 1
        navo = cash + sum(p['u'] * (AOn[t, j] if OPOK[t, j] else ACn[t - 1, j]) for j, p in Pp.items())
        unit = navo / slots
        if t > t_start:
            reserved = sum(max(cap - p['cf'], 0) * p['B'] for p in Pp.values() if not p['px'])
            for j in CANDS[t - 1]:
                if j in Pp or j in sold or not OPOK[t, j]: continue
                if GAP_UP[t, j]: cnt['skip_gap'] += 1; continue
                freec = cash - reserved
                if len(Pp) >= slots or freec < 0.25 * unit: cnt['skip_no_slot_or_cash'] += 1; continue
                B = min(unit, freec)      # ONE 不跑組合；P/D/ALL 與原式相同
                amt = fr[0] * B
                Pp[j] = {'u': amt / (1 + buy) / AOn[t, j], 'e': t, 'px': False, 'why': None, 'B': B, 'cf': fr[0], 'k': 1, 'last': AOn[t, j], 'add': None}
                cash -= amt; reserved += max(cap - fr[0], 0) * B; cnt['entries'] += 1
        val = 0.0
        for j, p in Pp.items():
            val += p['u'] * ACn[t, j]
            if p['px']: continue
            if EXIT[t, j]: p['px'] = True; p['why'] = 'exit_signal'; continue
            if ts_lim and t - p['e'] >= ts_lim - 1: p['px'] = True; p['why'] = 'exit_time'; continue
            if scheme == 'P' and p['k'] < 4 and HAS[t, j] and ADJn[t, j] < p['last']:
                if filt and not ABOVE[t, j]: cnt['add_filtered'] += 1
                elif t + 1 < T: p['add'] = FR['P'][p['k']]
            elif scheme == 'D' and (t + 1 - p['e']) in DDAY:
                if filt and not ABOVE_D[t, j]: cnt['add_filtered'] += 1
                elif t + 1 < T: p['add'] = DDAY[t + 1 - p['e']]
        nav[t - t_start] = cash + val; npos[t - t_start] = len(Pp); inv[t - t_start] = val / (cash + val)
    ix = IDX[t_start:]
    return pd.Series(nav, index=ix), pd.Series(npos, index=ix), pd.Series(inv, index=ix), cnt

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
EWL = ew_nav(LIQ)
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
    for nm_, bn in (('0050', E50), ('EW', EWL)):
        if bn is nav: continue
        ex = pr - mwin(bn).reindex(pr.index)
        o[f'excess_vs_{nm_}_pct_m'] = round(float(ex.mean() * 100), 2); o[f't_vs_{nm_}_m'] = tstat(ex)
        bd = bn.pct_change().reindex(rr.index); eb = (1 + bd.fillna(0)).cumprod()
        o[f'{nm_}_CAGR'] = round(float(eb.iloc[-1] ** (1 / span) - 1), 4); o[f'{nm_}_maxDD'] = round(float((eb / eb.cummax() - 1).min()), 4)
    return o
def yearly(nav):
    r = nav.pct_change().dropna(); return {int(y): round(float((1 + g).prod() - 1) * 100, 1) for y, g in r.groupby(r.index.year)}

# ---------------- 執行 8 組 ＋ 診斷 ----------------
VARS = {f'{sc}_{"F" if fl else "NF"}_{"T20" if ts else "NT"}': (sc, fl, ts) for sc in ('P', 'D') for fl in (True, False) for ts in (20, 0)}
DIAGV = {f'{sc}_{"T20" if ts else "NT"}': (sc, False, ts) for sc in ('ALL', 'ONE') for ts in (20, 0)}
PRIMARY = 'P_F_T20'; CO = 'D_F_T20'
EVS = {}; out_ev = {}; out_pt = {}; navs = {'EW_liq': EWL, '0050': E50}; yearly_ev = {}
for vn, (sc, fl, ts) in {**VARS, **DIAGV}.items():
    df = build_events(sc, fl, ts); EVS[vn] = df
    out_ev[vn] = {'counts': build_events.cnt, 'n_events_total': int(len(df)), 'windows': {k: ev_stats(df, sg) for k, sg in WIN.items()}}
    net = add_cost(df, 0.001)
    yearly_ev[vn] = {int(y): {'n': int(len(g)), 'net_pct': round(float(net[g.index].mean() * 100), 2),
                              'ab_net_pct': round(float((net[g.index] - g.ew).mean() * 100), 2),
                              'ab_gross_pct': round(float((g.raw - g.ew).mean() * 100), 2)} for y, g in df.groupby(df.sig.dt.year)}
    if sc == 'ONE': continue
    out_pt[vn] = {}
    for cost in (('gross', 0.001, 'stress') if vn in (PRIMARY, CO, 'ALL_T20') else (0.001,)):
        b, s = (0.0, 0.0) if cost == 'gross' else L.cost_rates(cost)
        nav, npos, inv, cnt = sim(sc, fl, ts, b, s)
        navs[f'{vn}_{cost}'] = nav
        o = {'counts': cnt, 'mean_positions': round(float(npos[npos.index >= '2005-01-01'].mean()), 2),
             'mean_invested_frac_of_NAV': round(float(inv[inv.index >= '2005-01-01'].mean()), 3),
             'windows': {k: daily_stats(nav, sg) for k, sg in WIN.items()}, 'yearly_pct': yearly(nav[nav.index >= '2004-12-31'])}
        if cost == 0.001:
            o['positions_by_year'] = {int(y): round(float(g.mean()), 2) for y, g in npos.groupby(npos.index.year)}
            o['invested_by_year'] = {int(y): round(float(g.mean()), 3) for y, g in inv.groupby(inv.index.year)}
        out_pt[vn][str(cost)] = o
    es, ev_ = out_ev[vn]['windows'][MAINW[0]], out_ev[vn]['windows'][MAINW[1]]
    ps, pv = out_pt[vn]['0.001']['windows'][MAINW[0]], out_pt[vn]['0.001']['windows'][MAINW[1]]
    print(vn, 'n', len(df), '| S n', es.get('n'), 'tr', es.get('avg_tranches'), 'ab', es.get('mean_ab_net_pct'), 't', es.get('t_month_ab_net'), 'gross', es.get('mean_ab_gross_pct'), es.get('t_month_ab_gross'),
          'CAGR', ps['CAGR'], '50', ps['0050_CAGR'], 'EW', ps['EW_CAGR'], 'tEW', ps['t_vs_EW_m'], 'inv', out_pt[vn]['0.001']['mean_invested_frac_of_NAV'],
          '| V n', ev_.get('n'), 'ab', ev_.get('mean_ab_net_pct'), 't', ev_.get('t_month_ab_net'), 'gross', ev_.get('mean_ab_gross_pct'), 'CAGR', pv['CAGR'], '50', pv['0050_CAGR'], 'EW', pv['EW_CAGR'], 'tEW', pv['t_vs_EW_m'],
          round(time.time() - T0), flush=True)
res['events'] = out_ev; res['portfolio'] = out_pt; res['events_yearly'] = yearly_ev
res['EW_liq_yearly'] = yearly(EWL[EWL.index >= '2004-12-31']); res['0050_yearly'] = yearly(E50[E50.index >= '2004-12-31'])

# ---------------- 診斷 ----------------
diag = {}
for vn in (PRIMARY, CO, 'ALL_T20'):
    sc, fl, ts = {**VARS, **DIAGV}[vn]
    dfc = build_events(sc, fl, ts, mode='close')
    diag[f'{vn}_close_entry_untradable'] = {k: {kk: ev_stats(dfc, WIN[k]).get(kk) for kk in ('n', 'mean_ab_gross_pct', 't_month_ab_gross', 'mean_ab_net_pct', 't_month_ab_net')} for k in MAINW}
# PRIMARY 依加碼批數分層
for vn in (PRIMARY, CO):
    df = EVS[vn]; d2 = {}
    for k in MAINW[:2]:
        a, b = WIN[k][0]; s = df[(df.sig >= a) & (df.sig <= b)]; net = add_cost(s, 0.001)
        d2[k] = {int(n): [int(len(g)), round(float(net[g.index].mean() * 100), 3), round(float((net - s.ew)[g.index].mean() * 100), 3), round(float(g.hold.mean()), 1)]
                 for n, g in s.groupby('ntr')}
    diag[f'{vn}_by_ntranches_[n,net,ab_net,hold]'] = d2
# 事件：資金×天 加權（每單位資金每日異常）
for vn in (PRIMARY, CO, 'ALL_T20'):
    df = EVS[vn]; d3 = {}
    for k in MAINW:
        a, b = WIN[k][0]; s = df[(df.sig >= a) & (df.sig <= b)]; net = add_cost(s, 0.001)
        d3[k] = {'ab_net_per_capital_day_bp': round(float(((net - s.ew) * s.Cf).sum() / s.capdays.sum() * 1e4), 2),
                 'gross_ab_per_capital_day_bp': round(float(((s.raw - s.ew) * s.Cf).sum() / s.capdays.sum() * 1e4), 2)}
    diag[f'{vn}_per_capital_day'] = d3
res['diagnostics'] = diag
print('diag done', round(time.time() - T0), flush=True)

# 資料檢查
Fn = F.to_numpy(); chk = {}
for vn in (PRIMARY, CO):
    df = EVS[vn]
    f345 = np.array([bool(np.isin(Fn[r.e:r.x + 1, r.j], [3, 4, 5]).any()) for r in df.itertuples()])
    big = df[(df.raw > 1.0) | (df.raw < -0.5)]
    keep = ~f345 & (df.raw.abs() <= 1.0).to_numpy()
    st = {k: ev_stats(df[keep].reset_index(drop=True), WIN[k]) for k in MAINW[:2]}
    chk[vn] = {'trades_with_flag345_in_hold': int(f345.sum()), 'n': int(len(df)), 'n_ret_gt_100pct': int((df.raw > 1).sum()), 'n_ret_lt_-50pct': int((df.raw < -0.5).sum()),
               'extreme_examples': big.assign(code=[cols[j] for j in big.j], sig=big.sig.astype(str))[['sig', 'code', 'raw', 'why', 'hold']].round(3).head(10).to_dict('records'),
               'ab_net_ex_flag345_extremes': {k: (v.get('mean_ab_net_pct'), v.get('t_month_ab_net')) for k, v in st.items()},
               'valid_events_touching_src3_share': round(float(df[df.sig >= '2019-05-01'].src3.mean()), 3)}
res['data_checks'] = chk

def passes(vn, k):
    e = out_ev[vn]['windows'][k]; p = out_pt[vn]['0.001']['windows'][k]
    if 'mean_ab_net_pct' not in e: return {'n': e.get('n'), 'insufficient': True, 'pass': False}
    insufficient = e['n'] < 100 or e['n_months'] < 24
    tm = e['t_month_ab_net'] if e['t_month_ab_net'] == e['t_month_ab_net'] else -9
    ok_ew = e['mean_ab_net_pct'] > 0 and tm >= 2; ok_50 = p['CAGR'] >= p['0050_CAGR']
    return {'n': e['n'], 'n_months': e['n_months'], 'insufficient': bool(insufficient), 'event_ab_net_pct': e['mean_ab_net_pct'], 'event_t_month': e['t_month_ab_net'],
            'event_ab_gross_pct': e['mean_ab_gross_pct'], 'port_CAGR': p['CAGR'], 'port_maxDD': p['maxDD'], '0050_CAGR': p['0050_CAGR'],
            'EW_CAGR': p['EW_CAGR'], 'port_t_vs_EW_m': p['t_vs_EW_m'], 'port_t_vs_0050_m': p['t_vs_0050_m'],
            'beat_EW_t2_event': bool(ok_ew), 'port_beat_EW_t2': bool((p['t_vs_EW_m'] or -9) >= 2), 'not_lose_0050': bool(ok_50), 'pass': bool(ok_ew and ok_50 and not insufficient)}
J = {vn: {k: passes(vn, k) for k in MAINW} for vn in list(VARS) + ['ALL_T20', 'ALL_NT']}
for vn in J:
    s_, v_ = J[vn][MAINW[0]], J[vn][MAINW[1]]
    if s_['insufficient'] and v_['insufficient']: J[vn]['verdict'] = '無法測試'
    else:
        n = s_['pass'] + v_['pass']; J[vn]['verdict'] = '可用' if n == 2 else ('待觀察' if n == 1 else '不可用')
J['PRIMARY'] = PRIMARY; J['overall'] = J[PRIMARY]['verdict']; J['co_reported_D'] = J[CO]['verdict']
J['verdict_counts_8'] = pd.Series([J[v]['verdict'] for v in VARS]).value_counts().to_dict()
res['judgement'] = J
print('judgement', J['overall'], json.dumps(J[PRIMARY], ensure_ascii=False), 'D', J[CO]['verdict'], J['verdict_counts_8'], flush=True)
(HERE / 'results.json').write_text(json.dumps(res, ensure_ascii=False, indent=1, default=lambda x: None if isinstance(x, float) and not np.isfinite(x) else str(x)), encoding='utf-8')
pd.DataFrame(navs).to_parquet(HERE / 'nav.parquet')
for vn in (PRIMARY, CO):
    EVS[vn].assign(code=[cols[j] for j in EVS[vn].j]).drop(columns=['j']).to_csv(HERE / f'events_{vn}.csv.gz', index=False)
print('done', round(time.time() - T0))
