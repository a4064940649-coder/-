#!/usr/bin/env python3
"""Connors %b（SMA200 之上、%b 連 3 日 < 0.2 買、%b > 0.8 賣）。規則見 results.json → PREREG（prereg.py 事前寫入）。執行：python3 -I run.py
骨架沿用 w3_connors_r3/run.py（股池、flag、處置、下市處理、事件引擎、月窗統計），改訊號、出場、20 槽與積極加碼；成本只從 exec_rebuild/exlib 匯入。"""
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
lv_mat = np.maximum.accumulate(np.where(np.isfinite(ADJn), np.arange(T)[:, None], -1), axis=0)
LASTV = lv_mat[T - 1]
DEAD = np.zeros((T, N), bool); GAPS = []
for j in range(N):
    ixv = np.flatnonzero(np.isfinite(ADJn[:, j]))
    if len(ixv) < 2: continue
    dd = np.diff(ixv)
    for q in np.flatnonzero(dd > 250):
        DEAD[ixv[q] + 1:ixv[q + 1], j] = True; GAPS.append((cols[j], str(IDX[ixv[q]].date()), str(IDX[ixv[q + 1]].date())))
SRC3 = ((IDX >= pd.Timestamp('2024-02-15')) & (IDX <= pd.Timestamp('2025-09-23')))

# ---------------- 指標（只用還原收盤） ----------------
A = ADJ.ffill(limit=10)
HAS = np.isfinite(ADJn)
SMA200 = A.rolling(200, min_periods=200).mean().to_numpy()
def pctb(a, n, k):
    m = a.rolling(n, min_periods=n).mean(); sd = a.rolling(n, min_periods=n).std(ddof=0)
    with np.errstate(invalid='ignore', divide='ignore'):
        pb = ((a - (m - k * sd)) / (2 * k * sd)).to_numpy().copy()
    pb[~(sd.to_numpy() > 1e-12 * np.abs(m.to_numpy()))] = np.nan
    pb[~HAS] = np.nan
    return pb
PB = {'P5_1': pctb(A, 5, 1.0), 'P20_2': pctb(A, 20, 2.0)}
FL5 = FL.rolling(5, min_periods=1).max().to_numpy() > 0
def lag(a, k):
    return np.vstack([np.full((k, a.shape[1]), np.nan if a.dtype.kind == 'f' else False), a[:-k]])
with np.errstate(invalid='ignore'):
    BASE = UNIV.to_numpy() & HAS & (ADJn > SMA200) & ~FL5 & ~DISP_TT1
    SIG, ADD, EXIT = {}, {}, {}
    for p, pb in PB.items():
        SIG[p] = BASE & (pb < 0.2) & (lag(pb, 1) < 0.2) & (lag(pb, 2) < 0.2)
        ADD[p] = HAS & (pb < 0.2)
        EXIT[p] = HAS & (pb > 0.8)
    # 參考：RSI(2) < 10（w3_connors_rsi2_sma5 訊號）只用於重疊度
    dA = A.diff(); AG = dA.clip(lower=0).ewm(alpha=0.5, adjust=False).mean(); AL_ = (-dA).clip(lower=0).ewm(alpha=0.5, adjust=False).mean()
    with np.errstate(divide='ignore'):
        RSI = (100 - 100 / (1 + AG / AL_)).to_numpy()
    RSI = np.where((AL_.to_numpy() == 0) & (AG.to_numpy() > 0), 100.0, RSI); RSI = np.where((AL_.to_numpy() == 0) & (AG.to_numpy() == 0), 50.0, RSI)
    RSI[~np.isfinite(dA.to_numpy())] = np.nan
    SIG_REF = BASE & (RSI < 10)
m05 = (IDX >= pd.Timestamp('2005-01-01'))
res['meta'] = {'bad_days_dropped': BAD, 'post_hoc_gap_delist_fix_n': len(GAPS), 'n_stocks': N,
               'signal_days': {k: int(v[m05].sum()) for k, v in SIG.items()},
               'mean_signals_per_day': {k: round(float(v.sum(1)[m05].mean()), 3) for k, v in SIG.items()},
               'share_days_with_any_signal': {k: round(float((v.sum(1) > 0)[m05].mean()), 3) for k, v in SIG.items()},
               'share_univ_above_sma200_with_pb_lt_0.2': {k: round(float(np.nanmean(np.where(BASE[m05], PB[k][m05] < 0.2, np.nan))), 3) for k in PB},
               'signal_overlap_with_rsi2_lt10': {k: {'share_pctb_sig_also_rsi2<10': round(float((v & SIG_REF)[m05].sum() / max(v[m05].sum(), 1)), 3),
                                                     'share_rsi2<10_also_pctb_sig': round(float((v & SIG_REF)[m05].sum() / max(SIG_REF[m05].sum(), 1)), 3)} for k, v in SIG.items()}}
print('setup', round(time.time() - T0), res['meta']['signal_days'], flush=True)

# ---------------- 時點對齊半日等權指數（EW_match） ----------------
pool = UNIV.shift(1, fill_value=False).to_numpy()
with np.errstate(invalid='ignore', divide='ignore'):
    on = AOn / AC_prev - 1; intra = ACn / AOn - 1
ok = pool & np.isfinite(on) & np.isfinite(intra) & np.isfinite(ADJn)
on = np.clip(on, -0.5, 1.0); intra = np.clip(intra, -0.5, 1.0)
r_on = np.nan_to_num(np.nanmean(np.where(ok, on, np.nan), axis=1)); r_in = np.nan_to_num(np.nanmean(np.where(ok, intra, np.nan), axis=1))
steps = np.empty(2 * T); steps[0::2] = r_on; steps[1::2] = r_in
Hh = np.cumprod(1 + steps); EWO = Hh[0::2]; EWC = Hh[1::2]
E50n = L.e50.reindex(IDX.union(L.e50.index)).ffill().reindex(IDX).to_numpy()

def exit_px(x, j):
    """x 日開盤賣；鎖跌停／無開盤 → 順延；之後無價格（下市）→ 最後收盤。回傳 (價, 實際日, 是否下市, 是否收盤價)。"""
    if x > LASTV[j]: return ACn[T - 1, j], min(LASTV[j] + 1, T - 1), True, True
    k = x
    while k < T and not SELLABLE[k, j]:
        if k > LASTV[j]: return ACn[T - 1, j], min(LASTV[j] + 1, T - 1), True, True
        if DEAD[k, j]: return ACn[k - 1, j], k, True, True
        k += 1
    if k >= T: return ACn[T - 1, j], T - 1, False, True
    return AOn[k, j], k, False, False

def build_events(p, aggr, ts_lim):
    sig, EX, AD = SIG[p], EXIT[p], ADD[p]; rows = []; nblk = nno = 0; nadd_blk = 0
    for j in range(N):
        sigs = np.flatnonzero(sig[:, j]); free = -1
        for t in sigs:
            if t < free: continue
            e = t + 1
            if e >= T: break
            if not OPOK[e, j]: nno += 1; continue
            if GAP_UP[e, j]: nblk += 1; continue
            end = min(T, e + ts_lim) if ts_lim else T
            dz = np.flatnonzero(DEAD[e:end, j])        # 代碼中斷（>250 日）→ 中斷前最後收盤下市出場，不跨中斷找出場訊號
            if len(dz): end = e + dz[0]
            hit = np.flatnonzero(EX[e:end, j])
            why = 'pb>0.8'
            if len(hit):
                x = e + hit[0] + 1
            elif len(dz):
                x = end
            elif ts_lim:
                x = e + ts_lim; why = 'time'
            else:
                x = LASTV[j] + 1 if LASTV[j] < T - 1 else T; why = 'none'
            if x >= T:
                px, xs, dl, xc = ACn[T - 1, j], T - 1, False, True; why = 'open_at_end'
            else:
                px, xs, dl, xc = exit_px(x, j)
                if dl: why = 'delist'
            rows.append((j, t, e, xs, px, why, dl, xc, 1, AOn[e, j]))
            if aggr:   # 第 2 單位：收盤 s∈[e, x−2] 的 %b<0.2 → s+1 開盤加碼（s+1 < x）
                xe = min(x, xs)
                for s in np.flatnonzero(AD[e:xe - 1, j]) + e:
                    a = s + 1
                    if a >= xe: break
                    if not OPOK[a, j] or GAP_UP[a, j]: nadd_blk += 1; continue
                    rows.append((j, s, a, xs, px, why, dl, xc, 2, AOn[a, j])); break
            free = xs if not dl else T
    df = pd.DataFrame(rows, columns=['j', 's', 'e', 'x', 'px', 'why', 'delist', 'xclose', 'unit', 'p0'])
    df['raw'] = df.px / df.p0 - 1; df['hold'] = df.x - df.e
    ewx = np.where(df.xclose, EWC[np.maximum(df.x.to_numpy() - 1, 0)], EWO[df.x.to_numpy()])
    ewx = np.where(df.xclose & (df.why == 'open_at_end'), EWC[df.x.to_numpy()], ewx)
    df['ew'] = ewx / EWO[df.e.to_numpy()] - 1
    df['b50'] = E50n[np.maximum(df.x - 1, df.s)] / E50n[df.s] - 1
    df['gap'] = df.p0 / ACn[df.s, df.j] - 1
    df['ew_gap'] = EWO[df.e.to_numpy()] / EWC[df.s.to_numpy()] - 1
    df['src3'] = [bool(SRC3[a:b + 1].any()) for a, b in zip(df.s, df.x)]
    df['sig'] = IDX[df.s.to_numpy()]; df['entry'] = IDX[df.e.to_numpy()]
    df['pb'] = PB[p][df.s, df.j]
    build_events.cnt = {'blocked_gap_up': nblk, 'no_open': nno, 'addon_blocked': nadd_blk}
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

def ev_stats(df, segs, offpx=False):
    m = np.zeros(len(df), bool)
    for a, b in segs: m |= ((df.sig >= pd.Timestamp(a)) & (df.sig <= pd.Timestamp(b))).to_numpy()
    if offpx: m &= ~df.src3.to_numpy()
    x = df[m]
    if len(x) < 10: return {'n': int(len(x))}
    net = add_cost(x, 0.001); ab = net - x.ew; abg = x.raw - x.ew; abs_ = add_cost(x, 'stress') - x.ew; ab50 = net - x.b50
    em = x.entry.dt.to_period('M'); gm = ab.groupby(em).mean()
    closeab = ((1 + x.raw) * (1 + x.gap) - 1) - ((1 + x.ew) * (1 + x.ew_gap) - 1)
    return {'n': int(len(x)), 'n_unit2': int((x.unit == 2).sum()), 'n_months': int(gm.size), 'avg_hold_days': round(float(x.hold.mean()), 2),
            'mean_raw_pct': round(float(x.raw.mean() * 100), 3), 'mean_net_pct': round(float(net.mean() * 100), 3),
            'median_net_pct': round(float(net.median() * 100), 3), 'win_rate_net': round(float((net > 0).mean()), 3), 'win_rate_raw': round(float((x.raw > 0).mean()), 3),
            'mean_EW_match_pct': round(float(x.ew.mean() * 100), 3),
            'mean_ab_gross_pct': round(float(abg.mean() * 100), 3), 't_month_ab_gross': tstat(abg.groupby(em).mean()),
            'mean_ab_net_pct': round(float(ab.mean() * 100), 3), 'month_cluster_mean_ab_net_pct': round(float(gm.mean() * 100), 3),
            't_month_ab_net': tstat(gm), 't_date_ab_net_ref': tstat(ab.groupby(x.entry).mean()),
            'mean_ab_net_stress_pct': round(float(abs_.mean() * 100), 3), 't_month_ab_net_stress': tstat(abs_.groupby(em).mean()),
            'mean_net_vs_0050_pct': round(float(ab50.mean() * 100), 3), 't_month_vs_0050': tstat(ab50.groupby(em).mean()),
            'hit_ab_net': round(float((ab > 0).mean()), 3),
            'mean_gap_pct': round(float(x.gap.mean() * 100), 3), 'mean_gap_minus_EW_overnight_pct': round(float((x.gap - x.ew_gap).mean() * 100), 3),
            'ab_gross_if_close_entry_pct_NOT_TRADABLE': round(float(closeab.mean() * 100), 3), 't_month_close_entry_NOT_TRADABLE': tstat(closeab.groupby(em).mean()),
            'exit_mix': {k: round(float(v), 3) for k, v in x.why.value_counts(normalize=True).items()},
            'n_delist': int(x.delist.sum()), 'p01_raw_pct': round(float(x.raw.quantile(0.01) * 100), 2), 'p99_raw_pct': round(float(x.raw.quantile(0.99) * 100), 2)}

# ---------------- 20 槽組合 ----------------
t_start = int(np.searchsorted(IDX, pd.Timestamp('2004-07-01')))
V60n = V60.to_numpy()
def cand_lists(sig, pb):
    out = []
    for t in range(T):
        js = np.flatnonzero(sig[t])
        if len(js): js = js[np.lexsort((-np.nan_to_num(V60n[t, js]), pb[t, js]))]
        out.append(js)
    return out
CANDS = {k: cand_lists(v, PB[k]) for k, v in SIG.items()}
EWO_b = np.broadcast_to(EWO[:, None], (T, N)); EWC_b = np.broadcast_to(EWC[:, None], (T, N))

def sim(p, aggr, ts_lim, buy, sell, slots=20, shadow=False):
    EX, AD, pb = EXIT[p], ADD[p], PB[p]
    PO, PC = (EWO_b, EWC_b) if shadow else (AOn, ACn)
    cash = 1.0; Pp = {}
    nav = np.empty(T - t_start); nunits = np.empty(T - t_start)
    cnt = {'entries': 0, 'addons': 0, 'skip_no_slot': 0, 'skip_gap': 0, 'exit_signal': 0, 'exit_time': 0, 'delist': 0, 'exit_delayed': 0}
    for t in range(t_start, T):
        sold = set()
        for j in list(Pp):
            q = Pp[j]
            if t > LASTV[j] or DEAD[t, j]:
                cash += q['u'] * PC[t - 1, j] * (1 - sell); del Pp[j]; cnt['delist'] += 1; sold.add(j)
            elif q['px']:
                if SELLABLE[t, j]:
                    cash += q['u'] * PO[t, j] * (1 - sell); cnt[q['why']] += 1; del Pp[j]; sold.add(j)
                else: cnt['exit_delayed'] += 1
        navo = cash + sum(q['u'] * (PO[t, j] if OPOK[t, j] else PC[t - 1, j]) for j, q in Pp.items())
        unit = navo / slots; used = sum(q['n'] for q in Pp.values())
        if t > t_start:
            cands = [(pb[t - 1, j], -np.nan_to_num(V60n[t - 1, j]), j, 0) for j in CANDS[p][t - 1] if j not in Pp and j not in sold]
            if aggr:
                cands += [(pb[t - 1, j], -np.nan_to_num(V60n[t - 1, j]), j, 1) for j, q in Pp.items()
                          if q['n'] == 1 and not q['px'] and t - 1 >= q['e'] and AD[t - 1, j]]
                cands.sort(key=lambda z: (z[0], z[1]))
            for _, _, j, isadd in cands:
                if not OPOK[t, j]: continue
                if GAP_UP[t, j]: cnt['skip_gap'] += 1; continue
                if used >= slots or cash < 0.25 * unit: cnt['skip_no_slot'] += 1; continue
                amt = min(unit, cash); u = amt / (1 + buy) / PO[t, j]
                if isadd:
                    Pp[j]['u'] += u; Pp[j]['n'] = 2; cnt['addons'] += 1
                else:
                    Pp[j] = {'u': u, 'n': 1, 'e': t, 'px': False, 'why': None}; cnt['entries'] += 1
                cash -= amt; used += 1
        val = 0.0
        for j, q in Pp.items():
            val += q['u'] * PC[t, j]
            if q['px']: continue
            if EX[t, j]: q['px'] = True; q['why'] = 'exit_signal'
            elif ts_lim and t - q['e'] >= ts_lim - 1: q['px'] = True; q['why'] = 'exit_time'
        nav[t - t_start] = cash + val; nunits[t - t_start] = sum(q['n'] for q in Pp.values())
    ix = IDX[t_start:]
    return pd.Series(nav, index=ix), pd.Series(nunits, index=ix), cnt

# 月窗與同池等權（月換股、隔日開盤、無成本；沿用 w3_connors_r3）
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
def mwin(nav, md=None):
    nd = nav.reindex(MDX if md is None else md); return (nd.shift(-1) / nd - 1).iloc[:-1]
def daily_stats(nav, segs, shadow=None):
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
    bl = [('0050', E50), ('EW', EW)] + ([('shadow', shadow)] if shadow is not None else [])
    for nm_, bn in bl:
        if bn is nav: continue
        ex = pr - mwin(bn).reindex(pr.index)
        o[f'excess_vs_{nm_}_pct_m'] = round(float(ex.mean() * 100), 2); o[f't_vs_{nm_}_m'] = tstat(ex)
        bd = bn.pct_change().reindex(rr.index); eb = (1 + bd.fillna(0)).cumprod()
        o[f'{nm_}_CAGR'] = round(float(eb.iloc[-1] ** (1 / span) - 1), 4); o[f'{nm_}_maxDD'] = round(float((eb / eb.cummax() - 1).min()), 4)
    return o
def yearly(nav):
    r = nav.pct_change().dropna(); return {int(y): round(float((1 + g).prod() - 1) * 100, 1) for y, g in r.groupby(r.index.year)}

# ---------------- 個股 8 組 ----------------
VARS = {f'{p}_{ag}_{tn}': (p, ag == 'AGG', tl) for p in ('P5_1', 'P20_2') for ag in ('BASE', 'AGG') for tn, tl in (('NT', 0), ('T20', 20))}
PRIMARY = 'P5_1_BASE_NT'
EVS = {}; out_ev = {}; out_pt = {}; navs = {'EW_pool': EW, '0050': E50}; yearly_ev = {}
for vn, (p, ag, tl) in VARS.items():
    df = build_events(p, ag, tl); EVS[vn] = df
    out_ev[vn] = {'counts': build_events.cnt, 'n_events_total': int(len(df)),
                  'windows': {k: ev_stats(df, sg, offpx=(k == 'valid_officialPx_only')) for k, sg in WIN.items()}}
    net = add_cost(df, 0.001)
    yearly_ev[vn] = {int(y): {'n': int(len(g)), 'net_pct': round(float(net[g.index].mean() * 100), 2),
                              'ab_gross_pct': round(float((g.raw - g.ew).mean() * 100), 2),
                              'ab_net_pct': round(float((net[g.index] - g.ew).mean() * 100), 2)} for y, g in df.groupby(df.sig.dt.year)}
    shnav, _, _ = sim(p, ag, tl, 0.0, 0.0, shadow=True); navs[f'{vn}_shadowEW'] = shnav
    out_pt[vn] = {}
    for cost in ('gross', 0.001, 'stress'):
        b, s = (0.0, 0.0) if cost == 'gross' else L.cost_rates(cost)
        nav, nun, cnt = sim(p, ag, tl, b, s)
        navs[f'{vn}_{cost}'] = nav
        o = {'counts': cnt, 'mean_units': round(float(nun[nun.index >= '2005-01-01'].mean()), 2),
             'windows': {k: daily_stats(nav, sg, shadow=shnav) for k, sg in WIN.items()}, 'yearly_pct': yearly(nav[nav.index >= '2004-12-31'])}
        if cost == 0.001: o['units_by_year'] = {int(y): round(float(g.mean()), 2) for y, g in nun.groupby(nun.index.year)}
        out_pt[vn][str(cost)] = o
    out_pt[vn]['shadowEW_yearly_pct'] = yearly(shnav[shnav.index >= '2004-12-31'])
    es, ev_ = out_ev[vn]['windows']['select_2005-01_2019-04'], out_ev[vn]['windows']['valid_2019-05_2026-10']
    ps, pv = out_pt[vn]['0.001']['windows']['select_2005-01_2019-04'], out_pt[vn]['0.001']['windows']['valid_2019-05_2026-10']
    print(vn, 'n', len(df), '| S ab', es['mean_ab_net_pct'], 't', es['t_month_ab_net'], 'gross', es['mean_ab_gross_pct'], es['t_month_ab_gross'],
          'CAGR', ps['CAGR'], '50', ps['0050_CAGR'], 'EW', ps['EW_CAGR'], 'tEW', ps['t_vs_EW_m'], 'tSh', ps['t_vs_shadow_m'],
          '| V ab', ev_['mean_ab_net_pct'], 't', ev_['t_month_ab_net'], 'gross', ev_['mean_ab_gross_pct'], 'CAGR', pv['CAGR'], '50', pv['0050_CAGR'], 'EW', pv['EW_CAGR'], 'tEW', pv['t_vs_EW_m'],
          round(time.time() - T0), flush=True)
res['events'] = out_ev; res['events_by_year_slip0.1'] = yearly_ev; res['portfolio'] = out_pt
res['EW_pool_windows'] = {k: daily_stats(EW, sg) for k, sg in WIN.items()}
res['EW_pool_yearly'] = yearly(EW[EW.index >= '2004-12-31']); res['0050_yearly'] = yearly(E50[E50.index >= '2004-12-31'])
res['portfolio_cash_share_mean'] = {vn: round(float(1 - out_pt[vn]['0.001']['mean_units'] / 20), 3) for vn in VARS}

# 資料檢查
Fn = F.to_numpy(); chk = {}
for vn in (PRIMARY, 'P20_2_BASE_NT'):
    df = EVS[vn]
    f345 = np.array([bool(np.isin(Fn[r.e:r.x + 1, r.j], [3, 4, 5]).any()) for r in df.itertuples()])
    big = df[(df.raw > 1.0) | (df.raw < -0.5)]
    keep = ~f345 & (df.raw.abs() <= 1.0).to_numpy()
    st = {k: ev_stats(df[keep].reset_index(drop=True), WIN[k]) for k in ('select_2005-01_2019-04', 'valid_2019-05_2026-10')}
    chk[vn] = {'trades_with_flag345_in_hold': int(f345.sum()), 'n': int(len(df)), 'n_ret_gt_100pct': int((df.raw > 1).sum()), 'n_ret_lt_-50pct': int((df.raw < -0.5).sum()),
               'extreme_examples': big.assign(code=[cols[j] for j in big.j], sig=big.sig.astype(str))[['sig', 'code', 'raw', 'why', 'hold']].round(3).head(10).to_dict('records'),
               'ab_net_ex_flag345_extremes': {k: (v.get('mean_ab_net_pct'), v.get('t_month_ab_net')) for k, v in st.items()},
               'valid_events_touching_src3_share': round(float(df[df.sig >= '2019-05-01'].src3.mean()), 3),
               'longest_hold_days': int(df.hold.max()), 'share_hold_gt_60d': round(float((df.hold > 60).mean()), 4)}
    for wk in ('select_2005-01_2019-04', 'valid_2019-05_2026-10'):
        a, b = WIN[wk][0]
        s = df[(df.sig >= a) & (df.sig <= b)]; net = add_cost(s, 0.001)
        chk[vn][f'{wk}_ab_net_by_pb_bucket'] = {str(k): [int(len(g)), round(float((net - s.ew)[g.index].mean() * 100), 3)]
                                                for k, g in s.groupby(pd.cut(s.pb, [-10, -0.5, -0.25, 0, 0.1, 0.2]), observed=True)}
res['data_checks'] = chk

# ---------------- 0050 單一標的 ----------------
e = L.e50.copy().sort_index().drop(index=[pd.Timestamp('2026-07-10'), pd.Timestamp('2026-09-20')], errors='ignore')
e = e[(e.index >= '2003-07-01') & (e.index <= IDX[-1])]
EI = e.index; TE = len(EI); EP = e.to_numpy()
es_ = pd.Series(EP, index=EI)
SMA200e = es_.rolling(200, min_periods=200).mean().to_numpy()
def pctb1(n, k):
    m = es_.rolling(n, min_periods=n).mean(); sd = es_.rolling(n, min_periods=n).std(ddof=0)
    return ((es_ - (m - k * sd)) / (2 * k * sd)).to_numpy().copy()
PBe = {'P5_1': pctb1(5, 1.0), 'P20_2': pctb1(20, 2.0)}
with np.errstate(invalid='ignore'):
    SIGe = {p: (EP > SMA200e) & (pb < 0.2) & (np.r_[np.nan, pb[:-1]] < 0.2) & (np.r_[np.nan, np.nan, pb[:-2]] < 0.2) for p, pb in PBe.items()}
FEE = L.FEE
COST50 = {'etf': (FEE + 0.001, FEE + 0.001 + 0.001), 'exlib_stock': L.cost_rates(0.001), 'stress': L.cost_rates('stress'), 'gross': (0.0, 0.0)}

def trades_0050(p, aggr, ts_lim, lagk):
    """訊號 t 收盤；進場 t+lagk 收盤；出場訊號 s>=t+1 收盤 %b>0.8 → s+lagk 收盤；T20：進場後 ts_lim 日收盤。每單位一列。"""
    sig, pb = SIGe[p], PBe[p]; rows = []; t = 0
    while t < TE:
        if not sig[t]: t += 1; continue
        ent = t + lagk
        if ent >= TE: break
        x = None; why = 'pb>0.8'
        for s in range(t + 1, TE):
            if pb[s] > 0.8: x = s + lagk; break
            if ts_lim and s + lagk >= ent + ts_lim: x = ent + ts_lim; why = 'time'; break
        if x is None or x >= TE: x = TE - 1; why = 'open_at_end'
        rows.append((t, ent, x, 1, why))
        if aggr:
            for s in range(t + 1, x - lagk):
                if pb[s] < 0.2 and s + lagk < x: rows.append((s, s + lagk, x, 2, why)); break
        t = max(x, t + 1)   # 賣出日當天收盤起可再觸發（新進場最早在賣出日之後／同日先賣後買）
    df = pd.DataFrame(rows, columns=['s', 'e', 'x', 'unit', 'why'])
    df['raw'] = EP[df.x] / EP[df.e] - 1; df['hold'] = df.x - df.e
    df['sig'] = EI[df.s.to_numpy()]; df['entry'] = EI[df.e.to_numpy()]
    return df

def nav_0050(df, aggr, buy, sell):
    """收盤成交；基本版 1 單位＝100%；積極版每單位 50%（第 1 單位時以當時 NAV 的 50% 買，加碼用剩餘現金）。"""
    cash = 1.0; sh = 0.0; nav = np.empty(TE)
    ent = {}; ext = {}
    for r in df.itertuples():
        ent.setdefault(r.e, []).append(r.unit); ext.setdefault(r.x, True)
    for t in range(TE):
        if t in ext and sh > 0:
            cash += sh * EP[t] * (1 - sell); sh = 0.0
        if t in ent:
            for u in sorted(ent[t]):
                v = cash + sh * EP[t]
                amt = cash if (not aggr or u == 2) else min(cash, 0.5 * v)
                sh += amt / (1 + buy) / EP[t]; cash -= amt
        nav[t] = cash + sh * EP[t]
    return pd.Series(nav, index=EI)

_se = pd.Series(EI, index=EI)
MDe = pd.DatetimeIndex([g[g.dt.day >= 11].iloc[0] for _, g in _se.groupby([EI.year, EI.month]) if (g.dt.day >= 11).any()])
MDe = MDe.append(pd.DatetimeIndex([EI[-1]])) if MDe[-1] != EI[-1] else MDe
BH = pd.Series(EP / EP[0], index=EI)
def stats_0050(nav, segs):
    r = nav.pct_change(); mask = np.zeros(len(r), bool); span = 0.0
    for a, b in segs:
        m = (r.index >= pd.Timestamp(a)) & (r.index <= pd.Timestamp(b)) & r.notna().to_numpy()
        first = np.argmax(m); last = len(m) - 1 - np.argmax(m[::-1])
        span += (r.index[last] - r.index[max(first - 1, 0)]).days / 365.25; mask |= m
    rr = r[mask]; bb = BH.pct_change()[mask]
    def blk(x):
        eq = (1 + x).cumprod()
        return {'CAGR': round(float(eq.iloc[-1] ** (1 / span) - 1), 4), 'maxDD': round(float((eq / eq.cummax() - 1).min()), 4),
                'vol': round(float(x.std() * math.sqrt(252)), 4), 'Sharpe': round(float(x.mean() / x.std() * math.sqrt(252)), 3) if x.std() > 0 else None}
    o = blk(rr); o['BH'] = blk(bb); o['exposure_days_share'] = None
    pr = mwin(nav, MDe).dropna(); pb_ = mwin(BH, MDe).reindex(pr.index); inw = np.zeros(len(pr), bool)
    for a, b in segs: inw |= (pr.index >= pd.Timestamp(a)) & (pr.index <= pd.Timestamp(b))
    ex = (pr - pb_)[inw]; o['n_months'] = int(inw.sum()); o['excess_vs_BH_pct_m'] = round(float(ex.mean() * 100), 2); o['t_vs_BH_m'] = tstat(ex)
    return o

def tr_stats(df, segs, buy, sell):
    m = np.zeros(len(df), bool); sm = np.zeros(TE, bool)
    for a, b in segs:
        m |= ((df.sig >= pd.Timestamp(a)) & (df.sig <= pd.Timestamp(b))).to_numpy(); sm |= (EI >= pd.Timestamp(a)) & (EI <= pd.Timestamp(b))
    x = df[m]
    if len(x) < 5: return {'n': int(len(x))}
    net = (1 + x.raw) * (1 - sell) / (1 + buy) - 1
    st = np.flatnonzero(sm)
    def unc(h):
        i = st[st + h < TE]; return float(np.mean(EP[i + h] / EP[i] - 1))
    cache = {}
    u = np.array([cache.setdefault(h, unc(h)) for h in x.hold])
    xs = net - u; em = x.entry.dt.to_period('M')
    return {'n': int(len(x)), 'n_unit2': int((x.unit == 2).sum()), 'avg_hold_days': round(float(x.hold.mean()), 2), 'win_rate_net': round(float((net > 0).mean()), 3),
            'mean_raw_pct': round(float(x.raw.mean() * 100), 3), 'mean_net_pct': round(float(net.mean() * 100), 3),
            'mean_uncond_same_len_pct': round(float(u.mean() * 100), 3), 'mean_xs_net_vs_uncond_pct': round(float(xs.mean() * 100), 3),
            't_month_xs_net': tstat(xs.groupby(em).mean()), 'mean_xs_gross_vs_uncond_pct': round(float((x.raw - u).mean() * 100), 3),
            't_month_xs_gross': tstat((x.raw - u).groupby(em).mean()), 'exit_mix': {k: round(float(v), 3) for k, v in x.why.value_counts(normalize=True).items()}}

W50 = {k: WIN[k] for k in ('select_2005-01_2019-04', 'valid_2019-05_2026-10', 'y2024-01_2025-12', 'select_sub_2005-2011', 'select_sub_2012-2019-04', 'r21_select_2021-01_2023-12', 'r21_valid_2024-01_2026-10')}
out50 = {}
for vn, (p, ag, tl) in VARS.items():
    out50[vn] = {}
    for lagk, lab in ((1, 'nextclose_TRADABLE'), (0, 'sameclose_NOT_TRADABLE')):
        df = trades_0050(p, ag, tl, lagk); blk = {'n_trades_total': int(len(df))}
        for cn, (b, s) in COST50.items():
            if lab.startswith('same') and cn not in ('etf', 'gross'): continue
            nav = nav_0050(df, ag, b, s); navs[f'0050single_{vn}_{lab}_{cn}'] = nav.reindex(IDX)
            blk[cn] = {'portfolio': {k: stats_0050(nav, sg) for k, sg in W50.items()}, 'trades': {k: tr_stats(df, sg, b, s) for k, sg in W50.items()},
                       'yearly_pct': yearly(nav[nav.index >= '2004-12-31'])}
            inpos = pd.Series(0.0, index=EI)
            for r in df[df.unit == 1].itertuples(): inpos.iloc[r.e + 1:r.x + 1] = 1.0
            blk[cn]['exposure_share_days_from2005'] = round(float(inpos[EI >= '2005-01-01'].mean()), 3)
        out50[vn][lab] = blk
    a = out50[vn]['nextclose_TRADABLE']['etf']
    print('0050', vn, 'n', out50[vn]['nextclose_TRADABLE']['n_trades_total'],
          '| S', a['trades']['select_2005-01_2019-04'].get('mean_xs_net_vs_uncond_pct'), a['trades']['select_2005-01_2019-04'].get('t_month_xs_net'),
          'CAGR', a['portfolio']['select_2005-01_2019-04']['CAGR'], 'Sh', a['portfolio']['select_2005-01_2019-04']['Sharpe'], 'BH', a['portfolio']['select_2005-01_2019-04']['BH'],
          '| V', a['trades']['valid_2019-05_2026-10'].get('mean_xs_net_vs_uncond_pct'), a['trades']['valid_2019-05_2026-10'].get('t_month_xs_net'),
          'CAGR', a['portfolio']['valid_2019-05_2026-10']['CAGR'], 'Sh', a['portfolio']['valid_2019-05_2026-10']['Sharpe'], 'BH', a['portfolio']['valid_2019-05_2026-10']['BH'], flush=True)
res['single_0050'] = out50
res['single_0050_BH_yearly'] = yearly(BH[BH.index >= '2004-12-31'])

# ---------------- 判定 ----------------
def passes(vn, k):
    e_ = out_ev[vn]['windows'][k]; p_ = out_pt[vn]['0.001']['windows'][k]
    ok_ew = e_['mean_ab_net_pct'] > 0 and (e_['t_month_ab_net'] or -9) >= 2
    ok_50 = p_['CAGR'] >= p_['0050_CAGR']
    return {'event_n': e_['n'], 'event_ab_net_pct': e_['mean_ab_net_pct'], 'event_t_month': e_['t_month_ab_net'], 'event_ab_gross_pct': e_['mean_ab_gross_pct'], 'event_t_gross': e_['t_month_ab_gross'],
            'port_CAGR': p_['CAGR'], 'port_maxDD': p_['maxDD'], '0050_CAGR': p_['0050_CAGR'], 'EW_CAGR': p_['EW_CAGR'], 'port_t_vs_EW_m': p_['t_vs_EW_m'], 'port_t_vs_0050_m': p_['t_vs_0050_m'],
            'port_t_vs_shadow_m': p_['t_vs_shadow_m'],
            'beat_EW_t2_event': bool(ok_ew), 'port_beat_EW_t2': bool((p_['t_vs_EW_m'] or -9) >= 2), 'not_lose_0050': bool(ok_50), 'pass': bool(ok_ew and ok_50)}
J = {vn: {k: passes(vn, k) for k in ('select_2005-01_2019-04', 'valid_2019-05_2026-10', 'y2024-01_2025-12')} for vn in VARS}
for vn in VARS:
    n = J[vn]['select_2005-01_2019-04']['pass'] + J[vn]['valid_2019-05_2026-10']['pass']
    few = min(out_ev[vn]['windows'][k]['n'] for k in ('select_2005-01_2019-04', 'valid_2019-05_2026-10')) < 100
    J[vn]['verdict'] = '無法測試' if few else ('可用' if n == 2 else ('待觀察' if n == 1 else '不可用'))
def pass50(vn, k):
    a = out50[vn]['nextclose_TRADABLE']['etf']; tr = a['trades'][k]; pt = a['portfolio'][k]
    if tr['n'] < 30: return {'n': tr['n'], 'pass': None}
    ok1 = tr['mean_xs_net_vs_uncond_pct'] > 0 and (tr['t_month_xs_net'] or -9) >= 2
    ok2 = (pt['Sharpe'] or -9) >= (pt['BH']['Sharpe'] or -9)
    return {'n': tr['n'], 'xs_net_pct': tr['mean_xs_net_vs_uncond_pct'], 't': tr['t_month_xs_net'], 'Sharpe': pt['Sharpe'], 'BH_Sharpe': pt['BH']['Sharpe'],
            'CAGR': pt['CAGR'], 'BH_CAGR': pt['BH']['CAGR'], 't_vs_BH_m': pt['t_vs_BH_m'], 'pass': bool(ok1 and ok2)}
J50 = {}
for vn in VARS:
    J50[vn] = {k: pass50(vn, k) for k in ('select_2005-01_2019-04', 'valid_2019-05_2026-10', 'y2024-01_2025-12')}
    ps_ = [J50[vn][k]['pass'] for k in ('select_2005-01_2019-04', 'valid_2019-05_2026-10')]
    J50[vn]['verdict'] = '無法測試' if None in ps_ else ('可用' if all(ps_) else ('待觀察' if any(ps_) else '不可用'))
J['PRIMARY'] = PRIMARY; J['overall'] = J[PRIMARY]['verdict']
J['n_variants_both_pass'] = sum(J[v]['verdict'] == '可用' for v in VARS); J['n_variants_one_pass'] = sum(J[v]['verdict'] == '待觀察' for v in VARS)
J['single_0050'] = J50; J['single_0050_PRIMARY_verdict'] = J50[PRIMARY]['verdict']
res['judgement'] = J
print('judgement', J['overall'], json.dumps(J[PRIMARY], ensure_ascii=False), '| 0050', J50[PRIMARY]['verdict'], json.dumps(J50[PRIMARY], ensure_ascii=False), flush=True)
(HERE / 'results.json').write_text(json.dumps(res, ensure_ascii=False, indent=1, default=lambda x: None if isinstance(x, float) and not np.isfinite(x) else str(x)), encoding='utf-8')
pd.DataFrame(navs).to_parquet(HERE / 'nav.parquet')
EVS[PRIMARY].assign(code=[cols[j] for j in EVS[PRIMARY].j]).drop(columns=['j']).to_csv(HERE / 'primary_events.csv.gz', index=False)
print('done', round(time.time() - T0))
