#!/usr/bin/env python3
"""林則行 鉛筆型平穩期後創新高 長歷史可成交回測。執行：python3 -I run.py
規則見 prereg.py（PREREG 已先寫入 results.json，本檔只附加結果，不改 PREREG）。
成交、成本、漲停判定、60 日均值缺漏修補沿用 momentum_family/mlib.py → exec_rebuild/exlib.py（只讀取、不修改）。
組合引擎與統計骨架改寫自 w3_vcp_contraction_breakout/run.py（只讀取）。"""
import json, math, sys, time
from pathlib import Path
import numpy as np, pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
from scipy.stats import rankdata

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
Cg = CG.to_numpy(); Hg = L.H.loc[GOOD].to_numpy(); Og = L.O.loc[GOOD].to_numpy()
V60g = M.V60.to_numpy()[gi]
common = np.array([not c.startswith('91') for c in COLS])
BASEOK = np.isfinite(ADJg) & np.isfinite(Cg) & np.isfinite(Hg) & np.isfinite(Og)
with np.errstate(invalid='ignore'):
    LIQ = (V60g >= 1e7) & (Cg >= 10) & common[None, :] & BASEOK
POOL = (V60g >= 1e7) & (Cg >= 10) & common[None, :] & np.isfinite(ADJg)

def price_conds(PX):
    """PX: DataFrame（有效日），ffill(limit=10)。回傳 (NH, NOTATH, PENCIL) bool 陣列。"""
    A = PX.ffill(limit=10)
    prev = A.shift(1)
    mx250 = prev.rolling(250, min_periods=240).max().to_numpy()
    mx2500 = prev.rolling(2500, min_periods=1).max().to_numpy()
    mx120 = prev.rolling(120, min_periods=115).max().to_numpy(); mn120 = prev.rolling(120, min_periods=115).min().to_numpy()
    x = PX.to_numpy()
    with np.errstate(invalid='ignore', divide='ignore'):
        return x > mx250, x < mx2500, (mx120 / mn120 - 1) <= 0.30

NH, NATH, PEN = price_conds(adjG)
NH_r, NATH_r, PEN_r = price_conds(CG)
VOL = VG / CG
VOLp = VOL.shift(1).rolling(20, min_periods=18).mean().to_numpy()
VOLn = VOL.to_numpy()
with np.errstate(invalid='ignore', divide='ignore'):
    VRATIO = VOLn / VOLp
    VUP = VRATIO >= 1.2
    Cprev = np.vstack([np.full((1, N), np.nan), Cg[:-1]])
    RANGE = (Hg - np.minimum(Og, Cg)) / Cprev <= 0.04

# ---------------- 月營收 ----------------
LONG = Path('/home/user/niklavsd/bottleneck-signals/data/processed/tw_revenue.csv')
d = pd.read_csv(LONG, dtype={'code': str, 'market': str, 'name': str})
d['p'] = pd.PeriodIndex(pd.to_datetime(d['month']), freq='M')
d = d.sort_values(['p', 'market']).drop_duplicates(['p', 'code'], keep='first'); d = d[d.code.isin(COLS)]
RL = d.pivot_table(index='p', columns='code', values='revenue_k_twd', aggfunc='last')
r5 = pd.read_csv(L.X / 'rev_5y.csv.gz', dtype={'code': str}); r5 = r5[r5.code.isin(COLS)]; r5['p'] = pd.PeriodIndex(r5['ym'], freq='M')
R5 = r5.pivot_table(index='p', columns='code', values='revenue', aggfunc='last')
pr = pd.period_range(min(RL.index.min(), R5.index.min()), max(RL.index.max(), R5.index.max()), freq='M')
RL = RL.reindex(index=pr, columns=COLS); R5 = R5.reindex(index=pr, columns=COLS)
late = pr >= pd.Period('2021-01', 'M')
REV = RL.copy(); REV.loc[late] = R5.loc[late].combine_first(RL.loc[late]); REV.loc[~late] = RL.loc[~late].combine_first(R5.loc[~late])
REV = REV.where(REV > 0)
S3 = REV.rolling(3, min_periods=3).sum()
Y3 = S3 / S3.shift(12) - 1
km = pd.PeriodIndex([pd.Period(t, 'M') - (1 if t.day >= 11 else 2) for t in G], freq='M')
Y3t = Y3.reindex(km).to_numpy()
with np.errstate(invalid='ignore'):
    REVOK = Y3t >= 0.10
res['revenue_coverage'] = {int(y): round(float(np.isfinite(Y3t[(G.year == y)][:, :]).any(axis=0)[LIQ[G.year == y].any(axis=0)].mean()), 3) for y in range(2005, 2027)}
print('setup', round(time.time() - T0), flush=True)

# ---------------- Spearman（26 日）----------------
def spearman_neg(PX, w=26):
    A = PX.ffill(limit=10).to_numpy(); out = np.zeros((T, N), bool)
    tt = np.arange(w) - (w - 1) / 2
    for c0 in range(0, N, 100):
        sub = A[:, c0:c0 + 100]
        win = sliding_window_view(sub, w, axis=0)          # (T-w+1, n, w)
        ok = np.isfinite(win).all(axis=-1)
        rk = rankdata(np.where(np.isfinite(win), win, 0), axis=-1)
        S = ((rk - (w + 1) / 2) * tt).sum(-1)
        out[w - 1:, c0:c0 + 100] = ok & (S < -1e-9)
    return out
SPNEG = spearman_neg(adjG)
print('spearman', round(time.time() - T0), round(float(SPNEG[LIQ].mean()), 3), flush=True)

# ---------------- 訊號 ----------------
lim = np.where(G < pd.Timestamp('2015-06-01'), 0.07, 0.10)[:, None]
SIG = {
    'BASE': LIQ & NH & NATH & PEN & VUP & REVOK & RANGE,
    'NO_REV': LIQ & NH & NATH & PEN & VUP & RANGE,
    'NO_PENCIL': LIQ & NH & NATH & VUP & REVOK & RANGE,
    'ALLOW_ATH': LIQ & NH & PEN & VUP & REVOK & RANGE,
    'NO_RANGE': LIQ & NH & NATH & PEN & VUP & REVOK,
    'NH250_ONLY': LIQ & NH,
    'BASE_RAWPX': LIQ & NH_r & NATH_r & PEN_r & VUP & REVOK & RANGE,
}
SIG['BASE_STOPONLY'] = SIG['BASE']
EXITMODE = {k: ('stoponly' if k == 'BASE_STOPONLY' else 'full') for k in SIG}
t05 = G >= pd.Timestamp('2005-01-01')
fun = {}; cum = LIQ.copy()
for nm_, c in (('liq', LIQ), ('(1)new_high_250', NH), ('(2)not_ATH', NATH), ('(3)pencil_30pct', PEN), ('(4)vol_x1.2', VUP), ('(6)range_4pct', RANGE), ('(5)rev3m_yoy_10pct', REVOK)):
    cum = cum & c; fun[nm_] = int(cum[t05].sum())
res['funnel_stock_days_2005on_cumulative'] = fun
res['single_condition_share_in_liq'] = {nm_: round(float(c[LIQ & t05[:, None]].mean()), 4) for nm_, c in (('NH', NH), ('NATH', NATH), ('PEN', PEN), ('VUP', VUP), ('RANGE', RANGE), ('REVOK', REVOK))}
res['signal_counts_2005on'] = {k: int(v[t05].sum()) for k, v in SIG.items()}
print('signals', res['signal_counts_2005on'], fun, flush=True)

lastvalid = np.array([np.where(np.isfinite(ADJg[:, j]))[0].max() if np.isfinite(ADJg[:, j]).any() else -1 for j in range(N)])
MAXH = 120
def find_exit(j, e, p0, mode):
    lv = lastvalid[j]
    end = min(e + MAXH, lv + 1)
    c = ADJg[e:end, j]
    with np.errstate(invalid='ignore'):
        st = c < p0 * 0.92
        sp = SPNEG[e:end, j] & np.isfinite(c) if mode == 'full' else np.zeros(len(c), bool)
    k = np.flatnonzero(st | sp)
    if len(k):
        t = e + k[0]; why = 'stop_8pct' if st[k[0]] else 'spearman_neg'
    elif e + MAXH <= lv:
        t = e + MAXH - 1; why = 'max_hold'
    else:
        if lv < T - 1: return min(lv + 1, T - 1), ACg[lv, j], 'delist'
        return T - 1, ACg[T - 1, j], 'open_at_end'
    for x in range(t + 1, lv + 1):
        if OPOK[x, j]:
            return x, AOg[x, j], why
    if t + 1 > T - 1: return T - 1, ACg[T - 1, j], 'open_at_end'
    return min(lv + 1, T - 1), ACg[lv, j], 'delist'

def build_events(S, mode):
    rows = []
    tt, jj = np.nonzero(S)
    df = pd.DataFrame({'t': tt, 'j': jj})
    df = df[df.t >= np.searchsorted(G, pd.Timestamp('2005-01-01'))]
    for j, g in df.groupby('j'):
        nxt = 0
        for s in g.t.to_numpy():
            if s < nxt: continue
            e = s + 1
            if e >= T: break
            if not OPOK[e, j] or GAPg[e, j]: continue
            p0 = AOg[e, j]
            x, p1, why = find_exit(j, e, p0, mode)
            rows.append((s, e, x, j, p0, p1, why)); nxt = x
    tr = pd.DataFrame(rows, columns=['s', 'e', 'x', 'j', 'p0', 'p1', 'why']).sort_values(['s', 'j']).reset_index(drop=True)
    tr['ret'] = tr.p1 / tr.p0 - 1; tr['hold'] = tr.x - tr.e
    ew = np.empty(len(tr))
    for i_, r in enumerate(tr.itertuples()):
        pl = np.flatnonzero(POOL[r.s] & OPOK[r.s + 1])
        px1 = ACg[r.x, pl] if r.why in ('open_at_end', 'delist') else AOg[r.x, pl]
        ew[i_] = np.nanmean(px1 / AOg[r.e, pl] - 1)
    tr['ew'] = ew
    tr['d'] = G[tr.s.to_numpy()]; tr['em'] = G[tr.e.to_numpy()].to_period('M')
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
    net = netr(tr, cost); abn = net - tr.ew
    out = {}
    for k, segs in windows.items():
        m = np.zeros(len(tr), bool)
        for a, b_ in segs: m |= ((tr.d >= pd.Timestamp(a)) & (tr.d <= pd.Timestamp(b_))).to_numpy()
        if m.sum() < 10: out[k] = {'n_trades': int(m.sum())}; continue
        x = tr[m]; g = abn[m].groupby(x.em).mean(); nm_ = net[m]
        out[k] = {'n_trades': int(m.sum()), 'n_months': int(len(g)), 'win_rate_net': round(float((nm_ > 0).mean()), 3),
                  'mean_ret_gross_pct': round(float(x.ret.mean() * 100), 2), 'mean_net_ret_pct': round(float(nm_.mean() * 100), 2),
                  'median_net_ret_pct': round(float(nm_.median() * 100), 2), 'mean_poolEW_same_window_pct': round(float(x.ew.mean() * 100), 2),
                  'mean_abn_pct': round(float(abn[m].mean() * 100), 2), 'month_cluster_mean_abn_pct': round(float(g.mean() * 100), 2),
                  't_month_cluster': L.tstat(g), 'mean_hold_days': round(float(x.hold.mean()), 1),
                  'exit_mix': {k_: round(float(v), 3) for k_, v in x.why.value_counts(normalize=True).items()}}
    return out

EVENTS = {}
for nm, S in SIG.items():
    EVENTS[nm] = build_events(S, EXITMODE[nm])
    print('events', nm, len(EVENTS[nm]), round(time.time() - T0), flush=True)
COSTS = ('gross', 0.0, 0.001, 'stress')
res['events'] = {nm: {str(c): event_stats(EVENTS[nm], c) for c in COSTS} for nm in SIG}
res['events_by_year_slip0.1'] = {}
for nm, tr in EVENTS.items():
    net = netr(tr, 0.001)
    res['events_by_year_slip0.1'][nm] = {int(y): {'n': int(len(g)), 'win': round(float((net[g.index] > 0).mean()), 3),
                                               'mean_net_pct': round(float(net[g.index].mean() * 100), 2),
                                               'mean_abn_pct': round(float((net[g.index] - g.ew).mean() * 100), 2)}
                                         for y, g in tr.groupby(tr.d.dt.year)}
for nm in SIG:
    print(nm, {k: (v.get('mean_abn_pct'), v.get('t_month_cluster'), v.get('n_trades')) for k, v in res['events'][nm]['0.001'].items()}, flush=True)

# ---------------- 組合（最多 20 檔） ----------------
t_start = int(np.searchsorted(G, pd.Timestamp('2005-01-01')))
CAND = {}
for nm, S in SIG.items():
    lst = [np.array([], int)] * T
    tt, jj = np.nonzero(S)
    for t, js in pd.Series(jj).groupby(tt):
        js = js.to_numpy(); lst[t] = js[np.argsort(-np.nan_to_num(VRATIO[t, js]))]
    CAND[nm] = lst

def sim(nm, buy, sell, maxpos=20, seed=None):
    rng = np.random.default_rng(seed) if seed is not None else None
    lst = CAND[nm]; mode = EXITMODE[nm]
    cash = 1.0; P = {}
    nav = np.empty(T - t_start); npos = np.empty(T - t_start); inv = np.empty(T - t_start)
    cnt = {'entries': 0, 'stop_8pct': 0, 'spearman_neg': 0, 'max_hold': 0, 'delist': 0, 'open_at_end': 0, 'skip_no_slot': 0, 'skip_gap': 0}
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
                x, p1, why = find_exit(j, t, px, mode)
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
gpos = {dd: i for i, dd in enumerate(G)}
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
    pr_ = mwin(nav).dropna(); inw = np.zeros(len(pr_), bool)
    for a, b in segs: inw |= (pr_.index >= pd.Timestamp(a)) & (pr_.index <= pd.Timestamp(b))
    pr_ = pr_[inw]; o['n_months'] = int(len(pr_))
    bl = [('0050', E50), ('EW', EW)] + (extra or [])
    for nm_, bn in bl:
        if bn is nav: continue
        ex = pr_ - mwin(bn).reindex(pr_.index)
        o[f'excess_vs_{nm_}_pct_m'] = round(float(ex.mean() * 100), 2); o[f't_vs_{nm_}_m'] = L.tstat(ex)
        bd = bn.pct_change().reindex(rr.index); eb = (1 + bd.fillna(0)).cumprod()
        o[f'{nm_}_CAGR'] = round(float(eb.iloc[-1] ** (1 / span) - 1), 4); o[f'{nm_}_maxDD'] = round(float((eb / eb.cummax() - 1).min()), 4)
    return o

def yearly(nav):
    r = nav.pct_change().dropna(); return {int(y): round(float((1 + g).prod() - 1) * 100, 1) for y, g in r.groupby(r.index.year)}

navs = {'EW_pool': EW, '0050': E50}
port = {}
b1, s1 = L.cost_rates(0.001)
for nm in SIG:
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
            for k in ('select_2005-01_2019-04', 'valid_2019-05_2026-10', 'r21_select_2021-01_2023-12', 'r21_valid_2024-01_2026-10'):
                a, b_ = W[k][0]; m = (G[tdf.e] >= pd.Timestamp(a)) & (G[tdf.e] <= pd.Timestamp(b_))
                o[f'n_trades_{k}'] = int(m.sum())
                hold_m = (npos.index >= a) & (npos.index <= b_)
                o[f'mean_open_positions_{k}'] = round(float(npos[hold_m].mean()), 2)
                o[f'mean_invested_{k}'] = round(float(inv[hold_m].mean()), 3)
        port[nm][str(cost)] = o
for nm in SIG:
    for cost in COSTS:
        nav = navs[f'{nm}_{cost}']
        extra = [('NH250_same_cost', navs[f'NH250_ONLY_{cost}'])] if nm != 'NH250_ONLY' else []
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
res['signals_per_year'] = {nm: {int(y): int(S[G.year == y].sum()) for y in range(2005, 2027)} for nm, S in SIG.items()}

rnd = {}
for seed in range(10):
    nav, *_ = sim('BASE', b1, s1, seed=seed)
    for k in ('select_2005-01_2019-04', 'valid_2019-05_2026-10', 'y2024-01_2025-12'):
        st = daily_stats(nav, W[k]); rnd.setdefault(k, []).append((st['CAGR'], st['t_vs_EW_m'], st['t_vs_0050_m']))
res['base_random_order_diag'] = {k: {'CAGR_mean': round(float(np.mean([x[0] for x in v])), 4), 'CAGR_min': round(float(np.min([x[0] for x in v])), 4),
                                     'CAGR_max': round(float(np.max([x[0] for x in v])), 4), 't_vs_EW_mean': round(float(np.mean([x[1] for x in v])), 2),
                                     't_vs_EW_max': round(float(np.max([x[1] for x in v])), 2), 't_vs_0050_mean': round(float(np.mean([x[2] for x in v])), 2)}
                                 for k, v in rnd.items()}
print('random', res['base_random_order_diag'], round(time.time() - T0), flush=True)

# ---------------- 資料檢查 ----------------
FL = pd.read_parquet(L.P / 'flag.parquet').reindex(index=IDX, columns=COLS).to_numpy()[gi]
SRC = pd.read_parquet(L.P / 'src.parquet').reindex(index=IDX, columns=COLS).to_numpy()[gi]
chk = {}
for nm, tr in EVENTS.items():
    if not len(tr): continue
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
    c['valid_median_abn_pct'] = round(float(a.median() * 100), 2) if len(a) else None
    chk[nm] = c
res['data_checks'] = chk

# ---------------- 判定 ----------------
def passes(nm, k):
    w = port[nm]['0.001']['windows'][k]
    return {'CAGR': w['CAGR'], '0050_CAGR': w['0050_CAGR'], 'EW_CAGR': w['EW_CAGR'], 't_vs_EW_m': w['t_vs_EW_m'], 't_vs_0050_m': w['t_vs_0050_m'],
            'maxDD': w['maxDD'], 'pass': bool(w['t_vs_EW_m'] >= 2 and w['CAGR'] >= w['0050_CAGR'])}
J = {}
for nm in SIG:
    J[nm] = {k: passes(nm, k) for k in ('select_2005-01_2019-04', 'valid_2019-05_2026-10')}
    npass = sum(v['pass'] for v in J[nm].values())
    po = port[nm]['0.001']
    insuff = any(po[f'mean_open_positions_{k}'] < 2 or po[f'n_trades_{k}'] < 30 for k in ('select_2005-01_2019-04', 'valid_2019-05_2026-10'))
    J[nm]['verdict'] = '無法測試' if insuff else ('可用' if npass == 2 else ('待觀察' if npass == 1 else '不可用'))
    J[nm]['r21_split'] = {k: passes(nm, k) for k in ('r21_select_2021-01_2023-12', 'r21_valid_2024-01_2026-10')}
J['overall'] = J['BASE']['verdict']
res['judgement'] = J
print('judgement', json.dumps(J, ensure_ascii=False), flush=True)
(HERE / 'results.json').write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str), encoding='utf-8')
pd.DataFrame(navs).to_parquet(HERE / 'nav.parquet')
for nm in ('BASE', 'NO_REV', 'BASE_RAWPX'):
    tr = EVENTS[nm]
    tr.assign(code=[COLS[j] for j in tr.j], d=tr.d.astype(str), em=tr.em.astype(str), entry=G[tr.e].astype(str), exit=G[tr.x].astype(str)).drop(columns=['j']).to_csv(HERE / f'events_{nm}.csv.gz', index=False)
print('done', round(time.time() - T0))
