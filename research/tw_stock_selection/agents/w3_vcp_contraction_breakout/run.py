#!/usr/bin/env python3
"""VCP 突破（程式化近似）長歷史可成交回測。執行：python3 -I run.py
規則見 prereg.py（PREREG 已先寫入 results.json，本檔只附加結果）。
成交、成本、漲停判定、60 日均值缺漏修補沿用 momentum_family/mlib.py → exec_rebuild/exlib.py（只讀取、不修改）。
組合引擎與統計骨架改寫自 w3_ath_breakout_duke/run.py（只讀取）。"""
import json, math, sys, time
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'momentum_family'))
import mlib as M  # noqa: E402
L = M.L
T0 = time.time()
res = json.loads((HERE / 'results.json').read_text(encoding='utf-8'))
assert 'PREREG' in res
res = {'PREREG': res['PREREG']}

IDX, COLS, N = L.IDX, L.COLS, L.N
GOOD = (L.adj.notna().sum(axis=1) >= 300).to_numpy()
gi = np.where(GOOD)[0]; G = IDX[gi]; T = len(G)
res['dropped_sparse_days'] = [str(d.date()) for d in IDX[~GOOD]]
ACg = L.AC[gi]; AOg = L.AO[gi]; ADJg = L.ADJ_RAW[gi]
OPOK = np.isfinite(L.OPEN_RAW[gi]) & np.isfinite(L.ADJ_RAW[gi])
GAPg = M.GAP_UP[gi]
adjG = L.adj.loc[GOOD]; CG = L.C.loc[GOOD]; VG = L.V.loc[GOOD]
Cg = CG.to_numpy(); Hg = L.H.loc[GOOD].to_numpy(); RETg = L.RET.loc[GOOD].to_numpy()
V60g = M.V60.to_numpy()[gi]
common = np.array([not c.startswith('91') for c in COLS])
LIQ = (V60g >= 1e7) & (Cg >= 10) & common[None, :] & np.isfinite(ADJg)
POOL = LIQ
ALL = common[None, :] & np.isfinite(ADJg)

AF = adjG.ffill(limit=10)
A = AF.to_numpy()
def sma(n): return AF.rolling(n, min_periods=n).mean().to_numpy()
S50, S150, S200 = sma(50), sma(150), sma(200)
MX252 = AF.rolling(252, min_periods=240).max().to_numpy()
MN252 = AF.rolling(252, min_periods=240).min().to_numpy()
VM10 = VG.rolling(10, min_periods=10).mean().to_numpy()
VM50 = VG.rolling(50, min_periods=50).mean().to_numpy()
VM50p = np.vstack([np.full((1, N), np.nan), VM50[:-1]])
VALg = VG.to_numpy()
with np.errstate(invalid='ignore', divide='ignore'):
    TREND = (ADJg > S50) & (S50 > S150) & (S150 > S200) & (ADJg >= 0.75 * MX252)
    VSURGE = VALg > 1.5 * VM50p
    VRATIO = VALg / VM50p
lim = np.where(G < pd.Timestamp('2015-06-01'), 0.07, 0.10)[:, None]
with np.errstate(invalid='ignore'):
    LOCK = (RETg >= lim - L.LTOL) & (Cg >= Hg * 0.999)
PRE = LIQ & TREND & VSURGE & ~LOCK & np.isfinite(ADJg)
res['prefilter_counts'] = {'liq_stock_days': int(LIQ.sum()), 'liq_trend': int((LIQ & TREND).sum()), 'liq_trend_vsurge_nolock': int(PRE.sum())}
print('setup', round(time.time() - T0), res['prefilter_counts'], flush=True)

# ---------------- Minervini 模板 ----------------
AGE = adjG.notna().cumsum().to_numpy()
def ret_k(k): return (AF / AF.shift(k) - 1)
RS = 0.4 * ret_k(63) + 0.2 * ret_k(126) + 0.2 * ret_k(189) + 0.2 * ret_k(252)
RSpool = LIQ & (AGE >= 260)
RSR = (RS.where(pd.DataFrame(RSpool, index=G, columns=COLS)).rank(axis=1, pct=True) * 100).to_numpy()
S200_21 = np.vstack([np.full((21, N), np.nan), S200[:-21]])
with np.errstate(invalid='ignore'):
    MINV = ((ADJg > S150) & (ADJg > S200) & (S150 > S200) & (S200 > S200_21) & (S50 > S150) & (S50 > S200) & (ADJg > S50)
            & (ADJg >= 1.30 * MN252) & (ADJg >= 0.75 * MX252) & (RSR >= 70) & RSpool)
res['minervini_template_stock_days'] = int(MINV.sum())

# ---------------- zigzag + 收縮 ----------------
THR = 0.05
def scan_stock(j):
    """回傳 list of (t, pivot, stop, dry_at_L, Lidx, k_used, depths)。只在 PRE[t,j] 的日子評估型態。"""
    c = A[:, j]; pre = PRE[:, j]; out = []
    fin = np.flatnonzero(np.isfinite(c))
    if len(fin) < 260: return out
    s = fin[0]
    piv = []          # (idx, price, type)
    mode = 0; hi = lo = c[s]; hi_i = lo_i = s; ext = c[s]; ext_i = s
    for t in range(s + 1, T):
        x = c[t]
        if x != x: continue
        if mode == 0:
            if x > hi: hi, hi_i = x, t
            if x < lo: lo, lo_i = x, t
            if x >= lo * (1 + THR) and lo_i < t:
                piv.append((lo_i, lo, 1)); mode = 1; ext, ext_i = x, t
            elif x <= hi * (1 - THR) and hi_i < t:
                piv.append((hi_i, hi, 0)); mode = -1; ext, ext_i = x, t
        elif mode == 1:
            if x > ext: ext, ext_i = x, t
            elif x <= ext * (1 - THR):
                piv.append((ext_i, ext, 0)); mode = -1; ext, ext_i = x, t
        else:
            if x < ext: ext, ext_i = x, t
            elif x >= ext * (1 + THR):
                piv.append((ext_i, ext, 1)); mode = 1; ext, ext_i = x, t
        if not pre[t] or len(piv) < 4 or piv[-1][2] != 1: continue
        # 回檔對（H_i, L_i）由近到遠
        pairs = []; q = len(piv) - 1
        while q >= 1 and len(pairs) < 4:
            if piv[q][2] == 1 and piv[q - 1][2] == 0:
                pairs.append((piv[q - 1], piv[q])); q -= 2
            else:
                break
        if len(pairs) < 2: continue
        Hk, Lk = pairs[0]
        pivot = Hk[1]
        if not (x > pivot and c[t - 1] <= pivot): continue
        dk = 1 - Lk[1] / Hk[1]
        if dk > 0.10: continue
        ok_k = None; deps = [dk]
        for k in range(2, len(pairs) + 1):
            H1, L1 = pairs[k - 1]
            d1 = 1 - L1[1] / H1[1]
            if H1[0] < t - 150: break
            if deps[-1] > 0.7 * d1: break
            deps.append(d1)
            if t - H1[0] >= 60: ok_k = k
        if ok_k is None: continue
        li = Lk[0]
        dry = bool(VM10[li, j] < VM50[li, j]) if np.isfinite(VM10[li, j]) and np.isfinite(VM50[li, j]) else False
        out.append((t, pivot, Lk[1], dry, li, ok_k, [round(float(v), 4) for v in deps[:ok_k][::-1]]))
    return out

RAW = []
for j in range(N):
    for r in scan_stock(j):
        RAW.append((r[0], j) + r[1:])
SIGDF = pd.DataFrame(RAW, columns=['t', 'j', 'pivot', 'stop', 'dryL', 'Lidx', 'k', 'depths'])
with np.errstate(invalid='ignore'):
    DRYT1 = np.vstack([np.zeros((1, N), bool), (VM10 < VM50)[:-1]])
SIGDF['dryT1'] = DRYT1[SIGDF.t, SIGDF.j]
print('zigzag scan', len(SIGDF), round(time.time() - T0), flush=True)

# 對照：趨勢 + 20 日突破
AFv = AF
HH20 = AFv.rolling(20, min_periods=20).max().shift(1).to_numpy()
LL20 = AFv.rolling(20, min_periods=20).min().shift(1).to_numpy()
with np.errstate(invalid='ignore'):
    BRK = A > HH20
BRKp = np.vstack([np.zeros((1, N), bool), BRK[:-1]])
TB = PRE & BRK & ~BRKp
tt, jj = np.nonzero(TB)
TBDF = pd.DataFrame({'t': tt, 'j': jj, 'pivot': HH20[tt, jj], 'stop': LL20[tt, jj]})

def mkvar(df):
    df = df.sort_values(['t', 'j']).reset_index(drop=True)
    return df
VAR = {
    'VCP': mkvar(SIGDF[SIGDF.dryL][['t', 'j', 'pivot', 'stop']]),
    'DRY_T1': mkvar(SIGDF[SIGDF.dryT1][['t', 'j', 'pivot', 'stop']]),
    'NO_DRY': mkvar(SIGDF[['t', 'j', 'pivot', 'stop']]),
    'TREND_BRK': mkvar(TBDF),
}
res['signal_counts'] = {k: int(len(v)) for k, v in VAR.items()}
res['vcp_k_mix'] = {int(k): int(v) for k, v in SIGDF[SIGDF.dryL].k.value_counts().items()}
res['vcp_last_depth_pct_mean'] = round(float((1 - SIGDF[SIGDF.dryL]['stop'] / SIGDF[SIGDF.dryL]['pivot']).mean() * 100), 2)
print('signals', res['signal_counts'], res['vcp_k_mix'], flush=True)

lastvalid = np.array([np.where(np.isfinite(ADJg[:, j]))[0].max() if np.isfinite(ADJg[:, j]).any() else -1 for j in range(N)])
MAXH = 60
def find_exit(j, e, p0, stop):
    lv = lastvalid[j]
    end = min(e + MAXH, lv + 1)
    c = ADJg[e:end, j]
    with np.errstate(invalid='ignore'):
        st = c < stop
        ma = c < S50[e:end, j]
    hit = st | ma
    k = np.flatnonzero(hit)
    if len(k):
        t = e + k[0]; why = 'stop_last_low' if st[k[0]] else 'below_sma50'
    elif e + MAXH <= lv:
        t = e + MAXH - 1; why = 'max_hold'
    else:
        if lv < T - 1: return min(lv + 1, T - 1), ACg[lv, j], 'delist', None
        return T - 1, ACg[T - 1, j], 'open_at_end', None
    for x in range(t + 1, lv + 1):
        if OPOK[x, j]:
            return x, AOg[x, j], why, t
    if t + 1 > T - 1: return T - 1, ACg[T - 1, j], 'open_at_end', t
    return min(lv + 1, T - 1), ACg[lv, j], 'delist', t

def build_events(df):
    rows = []
    for j, g in df.groupby('j'):
        nxt = 0
        for r in g.itertuples():
            s = r.t
            if s < nxt: continue
            e = s + 1
            if e >= T: break
            if not OPOK[e, j] or GAPg[e, j]: continue
            p0 = AOg[e, j]
            x, p1, why, _ = find_exit(j, e, p0, r.stop)
            rows.append((s, e, x, j, p0, p1, why))
            nxt = x
    tr = pd.DataFrame(rows, columns=['s', 'e', 'x', 'j', 'p0', 'p1', 'why']).sort_values(['s', 'j']).reset_index(drop=True)
    tr['ret'] = tr.p1 / tr.p0 - 1; tr['hold'] = tr.x - tr.e
    ew = np.empty(len(tr)); ewa = np.empty(len(tr))
    for i_, r in enumerate(tr.itertuples()):
        for arr, msk in ((ew, POOL), (ewa, ALL)):
            pl = np.flatnonzero(msk[r.s] & OPOK[r.s + 1])
            px1 = ACg[r.x, pl] if r.why == 'open_at_end' else AOg[r.x, pl]
            arr[i_] = np.nanmean(px1 / AOg[r.e, pl] - 1)
    tr['ew'] = ew; tr['ewall'] = ewa
    tr['d'] = G[tr.s.to_numpy()]; tr['em'] = G[tr.e.to_numpy()].to_period('M')
    tr['minv'] = MINV[tr.s, tr.j]
    return tr

W = {
    'select_2005-01_2019-04': [('2005-01-01', '2019-04-30')],
    'valid_2019-05_2026-10': [('2019-05-01', '2026-10-31')],
    'y2024-01_2025-12': [('2024-01-01', '2025-12-31')],
    'r21_select_2021-01_2023-12': [('2021-01-01', '2023-12-31')],
    'r21_valid_2024-01_2026-10': [('2024-01-01', '2026-10-31')],
    'valid_survivor_2019-05_2023-12': [('2019-05-01', '2023-12-31')],
    'valid_officialPx_only': [('2019-05-01', '2024-02-14'), ('2025-09-24', '2026-10-31')],
    'full_2005-01_2026-10': [('2005-01-01', '2026-10-31')],
}
def netr(tr, cost):
    b, s = (0.0, 0.0) if cost == 'gross' else L.cost_rates(cost)
    return (1 + tr.ret) * (1 - s) / (1 + b) - 1

def event_stats(tr, cost, windows=W):
    net = netr(tr, cost); abn = net - tr.ew; abna = net - tr.ewall
    out = {}
    for k, segs in windows.items():
        m = np.zeros(len(tr), bool)
        for a, b_ in segs: m |= ((tr.d >= pd.Timestamp(a)) & (tr.d <= pd.Timestamp(b_))).to_numpy()
        if m.sum() < 10: out[k] = {'n_trades': int(m.sum())}; continue
        x = tr[m]; g = abn[m].groupby(x.em).mean(); ga = abna[m].groupby(x.em).mean()
        nm_ = net[m]
        out[k] = {'n_trades': int(m.sum()), 'n_months': int(len(g)), 'win_rate_net': round(float((nm_ > 0).mean()), 3),
                  'mean_ret_gross_pct': round(float(x.ret.mean() * 100), 2), 'mean_net_ret_pct': round(float(nm_.mean() * 100), 2),
                  'median_net_ret_pct': round(float(nm_.median() * 100), 2), 'mean_poolEW_same_window_pct': round(float(x.ew.mean() * 100), 2),
                  'mean_abn_pct': round(float(abn[m].mean() * 100), 2), 'month_cluster_mean_abn_pct': round(float(g.mean() * 100), 2),
                  't_month_cluster': L.tstat(g), 'mean_abn_vs_allListedEW_pct': round(float(abna[m].mean() * 100), 2), 't_month_vs_allListedEW': L.tstat(ga),
                  'mean_hold_days': round(float(x.hold.mean()), 1),
                  'exit_mix': {k_: round(float(v), 3) for k_, v in x.why.value_counts(normalize=True).items()}}
    return out

EVENTS = {}
for nm, df in VAR.items():
    EVENTS[nm] = build_events(df)
    print('events', nm, len(EVENTS[nm]), round(time.time() - T0), flush=True)
COSTS = ('gross', 0.0, 0.001, 'stress')
res['events'] = {nm: {str(c): event_stats(EVENTS[nm], c) for c in COSTS} for nm in VAR}
res['events_by_year_slip0.1'] = {}
for nm, tr in EVENTS.items():
    net = netr(tr, 0.001)
    res['events_by_year_slip0.1'][nm] = {int(y): {'n': int(len(g)), 'win': round(float((net[g.index] > 0).mean()), 3),
                                               'mean_net_pct': round(float(net[g.index].mean() * 100), 2),
                                               'mean_abn_pct': round(float((net[g.index] - g.ew).mean() * 100), 2)}
                                         for y, g in tr.groupby(tr.d.dt.year)}
for nm in VAR:
    print(nm, {k: (v.get('mean_abn_pct'), v.get('t_month_cluster'), v.get('n_trades')) for k, v in res['events'][nm]['0.001'].items()}, flush=True)

# Minervini 重疊
ov = {}
tr = EVENTS['VCP']
for k in ('select_2005-01_2019-04', 'valid_2019-05_2026-10', 'full_2005-01_2026-10'):
    a, b_ = W[k][0]; m = ((tr.d >= a) & (tr.d <= b_)).to_numpy(); x = tr[m]
    o = {'n_events': int(m.sum()), 'share_events_meeting_template': round(float(x.minv.mean()), 3)}
    for lab, sub in (('template_yes', x[x.minv]), ('template_no', x[~x.minv])):
        st = event_stats(sub, 0.001, {k: W[k]})[k]
        o[lab] = {kk: st.get(kk) for kk in ('n_trades', 'mean_abn_pct', 't_month_cluster', 'mean_net_ret_pct')}
    ov[k] = o
tdays = G >= pd.Timestamp('2005-01-01')
ov['vcp_raw_signals_per_template_stock_day'] = round(float(len(SIGDF[SIGDF.dryL]) / max(1, MINV[tdays].sum())), 5)
ov['vcp_raw_signals_meeting_template_share'] = round(float(MINV[SIGDF[SIGDF.dryL].t, SIGDF[SIGDF.dryL].j].mean()), 3)
res['minervini_overlap'] = ov
print('overlap', ov, flush=True)

# ---------------- 組合（最多 20 檔） ----------------
t_start = int(np.searchsorted(G, pd.Timestamp('2005-01-01')))
CAND = {}
for nm, df in VAR.items():
    lst = [np.array([], int)] * T; stp = {}
    for t, g in df.groupby('t'):
        js = g.j.to_numpy(); vr = VRATIO[t, js]
        lst[t] = js[np.argsort(-vr)]
        for r in g.itertuples(): stp[(t, r.j)] = r.stop
    CAND[nm] = (lst, stp)

def sim(nm, buy, sell, maxpos=20, seed=None):
    rng = np.random.default_rng(seed) if seed is not None else None
    lst, stp = CAND[nm]
    cash = 1.0; P = {}
    nav = np.empty(T - t_start); npos = np.empty(T - t_start); inv = np.empty(T - t_start)
    cnt = {'entries': 0, 'stop_last_low': 0, 'below_sma50': 0, 'max_hold': 0, 'delist': 0, 'open_at_end': 0, 'skip_no_slot': 0, 'skip_gap': 0}
    trades = []
    for t in range(t_start, T):
        for j in list(P):
            p = P[j]
            if t == p['x']:
                if p['why'] == 'open_at_end': continue
                px = p['p1'] if p['why'] == 'delist' else AOg[t, j]
                cash += p['u'] * px * (1 - sell); cnt[p['why']] += 1
                trades.append((p['s'], p['e'], t, j, px / p['p0'] - 1, p['why'])); del P[j]
        navo = cash + sum(p['u'] * AOg[t, j] for j, p in P.items())
        unit = navo / maxpos
        if t > t_start:
            cands = lst[t - 1]
            if rng is not None and len(cands): cands = rng.permutation(cands)
            for j in cands:
                if j in P: continue
                if not OPOK[t, j]: continue
                if GAPg[t, j]: cnt['skip_gap'] += 1; continue
                if len(P) >= maxpos or cash < 0.25 * unit: cnt['skip_no_slot'] += 1; continue
                amt = min(unit, cash); px = AOg[t, j]
                x, p1, why, _ = find_exit(j, t, px, stp[(t - 1, j)])
                P[j] = {'u': amt / (1 + buy) / px, 'p0': px, 's': t - 1, 'e': t, 'x': x, 'p1': p1, 'why': why}
                cash -= amt; cnt['entries'] += 1
        val = sum(p['u'] * ACg[t, j] for j, p in P.items())
        k = t - t_start
        nav[k] = cash + val; npos[k] = len(P); inv[k] = val / (cash + val)
    ix = G[t_start:]
    return pd.Series(nav, index=ix), pd.Series(npos, index=ix), pd.Series(inv, index=ix), cnt, trades

_s = pd.Series(G, index=G)
MD = pd.DatetimeIndex([g[g.dt.day >= 11].iloc[0] for _, g in _s.groupby([G.year, G.month]) if (g.dt.day >= 11).any()])
MD = MD[MD >= pd.Timestamp('2004-12-01')]
gpos = {d: i for i, d in enumerate(G)}
ev_ew = {}
for dd in MD[:-1]:
    i = gpos[dd]
    if i + 1 >= T: continue
    pl = np.flatnonzero(POOL[i] & OPOK[i + 1])
    if len(pl) >= 50: ev_ew[int(gi[i + 1])] = ('open', pl)
t0f = min(ev_ew); ewf, _ = L.sim_portfolio(ev_ew, 0.0, 0.0, t0f, len(IDX) - 1)
EW = pd.concat([pd.Series([1.0], index=[IDX[t0f - 1]]), ewf]).reindex(G).ffill().dropna()
E50 = L.E50.reindex(G).ffill()
MDX = MD.append(pd.DatetimeIndex([G[-1]])) if MD[-1] != G[-1] else MD

def mwin(nav):
    nd = nav.reindex(MDX); return (nd.shift(-1) / nd - 1).iloc[:-1]

def daily_stats(nav, segs, extra=None):
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
    bl = [('0050', E50), ('EW', EW)] + (extra or [])
    for nm_, bn in bl:
        if bn is nav: continue
        ex = pr - mwin(bn).reindex(pr.index)
        o[f'excess_vs_{nm_}_pct_m'] = round(float(ex.mean() * 100), 2); o[f't_vs_{nm_}_m'] = L.tstat(ex)
        bd = bn.pct_change().reindex(rr.index); eb = (1 + bd.fillna(0)).cumprod()
        o[f'{nm_}_CAGR'] = round(float(eb.iloc[-1] ** (1 / span) - 1), 4); o[f'{nm_}_maxDD'] = round(float((eb / eb.cummax() - 1).min()), 4)
    return o

def yearly(nav):
    r = nav.pct_change().dropna(); return {int(y): round(float((1 + g).prod() - 1) * 100, 1) for y, g in r.groupby(r.index.year)}

navs = {'EW_pool': EW, '0050': E50}
port = {}
b1, s1 = L.cost_rates(0.001)
for nm in VAR:
    port[nm] = {}
    for cost in COSTS:
        b, s = (0.0, 0.0) if cost == 'gross' else L.cost_rates(cost)
        nav, npos, inv, cnt, trd = sim(nm, b, s)
        navs[f'{nm}_{cost}'] = nav
        o = {'counts': cnt, 'mean_open_positions': round(float(npos.mean()), 2), 'mean_invested_share': round(float(inv.mean()), 3)}
        if cost == 0.001:
            o['open_positions_by_year'] = {int(y): round(float(g.mean()), 2) for y, g in npos.groupby(npos.index.year)}
            o['invested_share_by_year'] = {int(y): round(float(g.mean()), 3) for y, g in inv.groupby(inv.index.year)}
            tdf = pd.DataFrame(trd, columns=['s', 'e', 'x', 'j', 'ret', 'why'])
            o['entries_per_year'] = round(cnt['entries'] / ((G[-1] - G[t_start]).days / 365.25), 1)
            o['trade_mean_ret_pct'] = round(float(tdf.ret.mean() * 100), 2); o['trade_win_rate'] = round(float((tdf.ret > 0).mean()), 3)
            o['trade_mean_hold'] = round(float((tdf.x - tdf.e).mean()), 1)
            o['entries_meeting_minervini_template_share'] = round(float(MINV[tdf.s, tdf.j].mean()), 3) if len(tdf) else None
            for k in ('select_2005-01_2019-04', 'valid_2019-05_2026-10'):
                a, b_ = W[k][0]; m = (G[tdf.e] >= pd.Timestamp(a)) & (G[tdf.e] <= pd.Timestamp(b_))
                o[f'n_trades_{k}'] = int(m.sum())
                hold_m = (npos.index >= a) & (npos.index <= b_)
                o[f'mean_open_positions_{k}'] = round(float(npos[hold_m].mean()), 2)
        port[nm][str(cost)] = o
for nm in VAR:
    for cost in COSTS:
        nav = navs[f'{nm}_{cost}']
        extra = [('TREND_BRK_same_cost', navs[f'TREND_BRK_{cost}'])] if nm != 'TREND_BRK' else []
        port[nm][str(cost)]['windows'] = {k: daily_stats(nav, sg, extra) for k, sg in W.items()}
        port[nm][str(cost)]['yearly_pct'] = yearly(nav)
    w = port[nm]['0.001']['windows']
    print(nm, {k: (w[k]['CAGR'], w[k]['0050_CAGR'], w[k]['EW_CAGR'], w[k]['t_vs_EW_m'], w[k]['t_vs_0050_m'], w[k]['maxDD'])
               for k in ('select_2005-01_2019-04', 'valid_2019-05_2026-10', 'y2024-01_2025-12')}, port[nm]['0.001']['mean_open_positions'], round(time.time() - T0), flush=True)
res['portfolio'] = port
res['EW_pool_windows'] = {k: daily_stats(EW, sg) for k, sg in W.items()}
res['EW_pool_yearly'] = yearly(EW[EW.index >= '2004-12-31'])
res['0050_yearly'] = yearly(E50[E50.index >= '2004-12-31'])
res['pool_size_by_year'] = {int(y): round(float(np.mean([POOL[gpos[dd]].sum() for dd in MD if dd.year == y])), 0) for y in sorted(set(MD.year))}
res['signals_per_year'] = {nm: {int(y): int((G[df.t] .year == y).sum()) for y in range(2005, 2027)} for nm, df in VAR.items()}

rnd = {}
for seed in range(10):
    nav, *_ = sim('VCP', b1, s1, seed=seed)
    for k in ('select_2005-01_2019-04', 'valid_2019-05_2026-10', 'y2024-01_2025-12'):
        st = daily_stats(nav, W[k]); rnd.setdefault(k, []).append((st['CAGR'], st['t_vs_EW_m'], st['t_vs_0050_m']))
res['primary_random_order_diag'] = {k: {'CAGR_mean': round(float(np.mean([x[0] for x in v])), 4), 'CAGR_min': round(float(np.min([x[0] for x in v])), 4),
                                        'CAGR_max': round(float(np.max([x[0] for x in v])), 4), 't_vs_EW_mean': round(float(np.mean([x[1] for x in v])), 2),
                                        't_vs_EW_max': round(float(np.max([x[1] for x in v])), 2), 't_vs_0050_mean': round(float(np.mean([x[2] for x in v])), 2)}
                                    for k, v in rnd.items()}
print('random', res['primary_random_order_diag'], round(time.time() - T0), flush=True)

# ---------------- 資料檢查 ----------------
FL = pd.read_parquet(L.P / 'flag.parquet').reindex(index=IDX, columns=COLS).to_numpy()[gi]
SRC = pd.read_parquet(L.P / 'src.parquet').reindex(index=IDX, columns=COLS).to_numpy()[gi]
chk = {}
for nm, tr in EVENTS.items():
    f345 = np.array([bool(np.isin(FL[r.s:r.x + 1, r.j], [3, 4, 5]).any()) for r in tr.itertuples()])
    big = tr[(tr.ret > 1.0) | (tr.ret < -0.5)]
    c = {'n_trades': int(len(tr)), 'trades_with_flag345': int(f345.sum()), 'entry_src3_share': round(float((SRC[tr.e, tr.j] == 3).mean()), 3),
         'n_ret_gt_100pct': int((tr.ret > 1.0).sum()), 'n_ret_lt_-50pct': int((tr.ret < -0.5).sum()),
         'extreme_examples': big.assign(d=big.d.astype(str), code=[COLS[j] for j in big.j])[['d', 'code', 'ret', 'why', 'hold']].round(3).head(12).to_dict('records')}
    st = event_stats(tr[~f345 & (tr.ret.abs() <= 1.0).to_numpy()], 0.001)
    c['ex_flag345_extremes'] = {k: (st[k].get('mean_abn_pct'), st[k].get('t_month_cluster'), st[k].get('n_trades')) for k in ('select_2005-01_2019-04', 'valid_2019-05_2026-10')}
    net = netr(tr, 0.001); abn = (net - tr.ew)
    mv = ((tr.d >= '2019-05-01')).to_numpy()
    a = abn[mv].sort_values(ascending=False)
    c['top10_abn_share_valid'] = round(float(a.iloc[:10].sum() / a.sum()), 3) if a.sum() > 0 else None
    c['valid_median_abn_pct'] = round(float(a.median() * 100), 2)
    chk[nm] = c
res['data_checks'] = chk

# ---------------- 判定 ----------------
def passes(nm, k):
    w = port[nm]['0.001']['windows'][k]
    return {'CAGR': w['CAGR'], '0050_CAGR': w['0050_CAGR'], 'EW_CAGR': w['EW_CAGR'], 't_vs_EW_m': w['t_vs_EW_m'], 't_vs_0050_m': w['t_vs_0050_m'],
            'maxDD': w['maxDD'], 'pass': bool(w['t_vs_EW_m'] >= 2 and w['CAGR'] >= w['0050_CAGR'])}
J = {}
for nm in VAR:
    J[nm] = {k: passes(nm, k) for k in ('select_2005-01_2019-04', 'valid_2019-05_2026-10')}
    npass = sum(v['pass'] for v in J[nm].values())
    po = port[nm]['0.001']
    insuff = any(po[f'mean_open_positions_{k}'] < 2 or po[f'n_trades_{k}'] < 30 for k in ('select_2005-01_2019-04', 'valid_2019-05_2026-10'))
    J[nm]['verdict'] = '無法測試' if insuff else ('可用' if npass == 2 else ('待觀察' if npass == 1 else '不可用'))
J['overall'] = J['VCP']['verdict']
res['judgement'] = J
print('judgement', json.dumps(J, ensure_ascii=False), flush=True)
(HERE / 'results.json').write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str), encoding='utf-8')
pd.DataFrame(navs).to_parquet(HERE / 'nav.parquet')
for nm, tr in EVENTS.items():
    tr.assign(code=[COLS[j] for j in tr.j], d=tr.d.astype(str), em=tr.em.astype(str)).drop(columns=['j']).to_csv(HERE / f'events_{nm}.csv.gz', index=False)
SIGDF.assign(code=[COLS[j] for j in SIGDF.j], d=G[SIGDF.t].astype(str)).to_csv(HERE / 'vcp_raw_signals.csv.gz', index=False)
print('done', round(time.time() - T0))
