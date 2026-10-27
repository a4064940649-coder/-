#!/usr/bin/env python3
"""DUKE 新高値ブレイク（上市以來新高＋營收成長）長歷史可成交回測。執行：python3 -I run.py
規則見 prereg.py（PREREG 已先寫入 results.json，本檔只附加結果）。
成交、成本、漲停判定、60 日均值缺漏修補沿用 momentum_family/mlib.py → exec_rebuild/exlib.py（只讀取、不修改）。"""
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
adjG = L.adj.loc[GOOD]; CG = L.C.loc[GOOD]
Cg = CG.to_numpy(); Hg = L.H.loc[GOOD].to_numpy(); RETg = L.RET.loc[GOOD].to_numpy()
V60g = M.V60.to_numpy()[gi]
common = np.array([not c.startswith('91') for c in COLS])
AGE = adjG.notna().cumsum().to_numpy()
LIQ = (V60g >= 1e7) & (Cg >= 10) & common[None, :] & np.isfinite(ADJg)
POOL = LIQ & (AGE >= 750)
ALL = common[None, :] & np.isfinite(ADJg)

# ---------------- 價格訊號 ----------------
with np.errstate(invalid='ignore'):
    SIG = {
        'ATH_raw': Cg > CG.cummax().ffill().shift(1).to_numpy(),
        'ATH_adj': ADJg > adjG.cummax().ffill().shift(1).to_numpy(),
        'H52_raw': Cg > CG.rolling(252, min_periods=240).max().shift(1).to_numpy(),
        'H52_adj': ADJg > adjG.rolling(252, min_periods=240).max().shift(1).to_numpy(),
    }
MA50 = adjG.rolling(50, min_periods=50).mean().to_numpy()
tx = pd.read_csv('/home/user/jonatw/finlab-txd/data/raw/taiex_twii.csv', parse_dates=['date']).set_index('date')['close'].sort_index()
TXg = tx.reindex(IDX, method='ffill').to_numpy()[gi]
TX20 = np.full(T, np.nan); TX20[20:] = TXg[20:] / TXg[:-20] - 1

# 漲停鎖死（收盤）
lim = np.where(G < pd.Timestamp('2015-06-01'), 0.07, 0.10)[:, None]
with np.errstate(invalid='ignore'):
    LOCK = (RETg >= lim - L.LTOL) & (Cg >= Hg * 0.999)

# ---------------- 營收 ----------------
d = pd.read_csv('/home/user/niklavsd/bottleneck-signals/data/processed/tw_revenue.csv', dtype={'code': str, 'market': str, 'name': str})
d['p'] = pd.PeriodIndex(pd.to_datetime(d['month']), freq='M')
d = d.sort_values(['p', 'market']).drop_duplicates(['p', 'code'], keep='first'); d = d[d['code'].isin(COLS)]
RV = d.pivot_table(index='p', columns='code', values='revenue_k_twd', aggfunc='last').sort_index()
RV = RV.reindex(pd.period_range(RV.index.min(), RV.index.max(), freq='M')).reindex(columns=COLS)
RVp = RV.where(RV > 0)
S3 = RVp.rolling(3, min_periods=3).sum()
Y3 = S3 / S3.shift(12) - 1
Y1 = RVp / RVp.shift(12) - 1
REVOK = (Y3 > 0.10) & (Y1 > 0)
ifrs = (REVOK.index >= pd.Period('2013-01', 'M')) & (REVOK.index <= pd.Period('2014-02', 'M'))
REVOK.loc[ifrs] = False
mstar = pd.PeriodIndex([pd.Period(x, 'M') - (1 if x.day >= 11 else 2) for x in G])
REVg = REVOK.reindex(mstar).fillna(False).to_numpy().astype(bool)
Y3g = Y3.reindex(mstar).to_numpy()
res['revenue_coverage'] = {'first_month': str(RV.index.min()), 'last_month': str(RV.index.max()),
                           'pool_share_with_Y3_by_year': {int(y): round(float(np.nanmean([np.isfinite(Y3g[t][POOL[t]]).mean() for t in np.where(G.year == y)[0][::21] if POOL[t].any()])), 3) for y in range(2007, 2027)}}

# ---------------- 處置 ----------------
DISP = np.zeros((T, N), bool)
gpos_all = {i: k for k, i in enumerate(gi)}
dsp = pd.read_csv(L.X / 'disposal_5y.csv.gz', dtype={'code': str}, parse_dates=['date', 'start', 'end'])
for r in dsp.itertuples():
    if r.code in L.CI and pd.notna(r.start) and pd.notna(r.end):
        a = np.searchsorted(G, r.start); b = np.searchsorted(G, r.end, 'right')
        DISP[a:b, L.CI[r.code]] = True
cut = np.searchsorted(G, pd.Timestamp('2021-09-24'))
A6 = adjG / adjG.shift(6) - 1; m6 = pd.Series(TXg, index=G); m6 = m6 / m6.shift(6) - 1
att = ((A6 >= 0.32) & (A6.sub(m6, axis=0) >= 0.20)).to_numpy()
trig = att.copy(); trig[1:] &= att[:-1]; trig[2:] &= att[:-2]; trig[:2] = False
nprox = 0
for j in range(N):
    i = 0; col = trig[:, j]
    while i < cut:
        if col[i]:
            DISP[i + 1:min(i + 11, cut), j] = True; nprox += 1; i = i + 11
        else:
            i += 1
DISPX = DISP.copy(); DISPX[:-1] |= DISP[1:]          # t 或 t+1 處置
res['disposal_proxy_events_pre2021_09'] = nprox

BASE = POOL & REVg & ~LOCK & ~DISPX
ENT = {k: BASE & v for k, v in SIG.items()}
FAC = (adjG / CG).to_numpy()
with np.errstate(invalid='ignore', divide='ignore'):
    fd = FAC[1:] / FAC[:-1] < 0.95
FDROP = np.zeros((T, N), bool); FDROP[1:] = fd
FD20 = pd.DataFrame(FDROP).rolling(20, min_periods=1).max().to_numpy().astype(bool)
FDEVER = np.cumsum(FDROP, axis=0) > 0
lastvalid = np.array([np.where(np.isfinite(ADJg[:, j]))[0].max() if np.isfinite(ADJg[:, j]).any() else -1 for j in range(N)])
res['signal_counts'] = {k: {'raw_price_signal': int((POOL & v).sum()), 'with_rev': int((POOL & REVg & v).sum()), 'final_ENT': int(ENT[k].sum())} for k, v in SIG.items()}
print('setup', round(time.time() - T0), res['signal_counts'], flush=True)

MAXH = 250
def find_exit(j, e, p0):
    lv = lastvalid[j]
    end = min(e + MAXH, lv + 1)          # 判斷日 t ∈ [e, end)
    c = ADJg[e:end, j]
    with np.errstate(invalid='ignore'):
        st10 = c <= 0.90 * p0
        rw = (TX20[e:end] > 0) & (c <= 0.95 * p0)
        prof = np.maximum.accumulate(np.nan_to_num(c, nan=-1) > p0)
        ma = prof & (c < MA50[e:end, j])
    hit = st10 | rw | ma
    k = np.flatnonzero(hit)
    if len(k):
        t = e + k[0]; why = 'stop10' if st10[k[0]] else ('rel_weak5' if rw[k[0]] else 'ma50')
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

# 預先算每檔的出場決策日（組合用）：給定 (j, e, p0) 直接呼叫 find_exit
def build_events(name):
    E = ENT[name]; rows = []
    for j in range(N):
        sigs = np.flatnonzero(E[:, j]); nxt = 0
        for s in sigs:
            if s < nxt: continue
            e = s + 1
            if e >= T: break
            if not OPOK[e, j] or GAPg[e, j]: continue
            p0 = AOg[e, j]
            x, p1, why, _ = find_exit(j, e, p0)
            rows.append((s, e, x, j, p0, p1, why))
            nxt = x
    tr = pd.DataFrame(rows, columns=['s', 'e', 'x', 'j', 'p0', 'p1', 'why'])
    tr['ret'] = tr.p1 / tr.p0 - 1; tr['hold'] = tr.x - tr.e
    ew = np.empty(len(tr)); ewa = np.empty(len(tr))
    for i_, r in enumerate(tr.itertuples()):
        for arr, msk in ((ew, POOL), (ewa, ALL)):
            pl = np.flatnonzero(msk[r.s] & OPOK[r.s + 1])
            px1 = ACg[r.x, pl] if r.why == 'open_at_end' else AOg[r.x, pl]
            arr[i_] = np.nanmean(px1 / AOg[r.e, pl] - 1)
    tr['ew'] = ew; tr['ewall'] = ewa
    tr['d'] = G[tr.s.to_numpy()]; tr['em'] = G[tr.e.to_numpy()].to_period('M')
    tr['isATH_raw'] = SIG['ATH_raw'][tr.s, tr.j]; tr['isATH_adj'] = SIG['ATH_adj'][tr.s, tr.j]
    tr['fd20'] = FD20[tr.s, tr.j]; tr['fdever'] = FDEVER[tr.s, tr.j]
    tr['v60'] = V60g[tr.s, tr.j]; tr['y3'] = Y3g[tr.s, tr.j]
    return tr

W = {
    'select_2007-01_2019-04': [('2007-01-01', '2019-04-30')],
    'valid_2019-05_2026-10': [('2019-05-01', '2026-10-31')],
    'y2024-01_2025-12': [('2024-01-01', '2025-12-31')],
    'r21_select_2021-01_2023-12': [('2021-01-01', '2023-12-31')],
    'r21_valid_2024-01_2026-10': [('2024-01-01', '2026-10-31')],
    'valid_survivor_2019-05_2023-12': [('2019-05-01', '2023-12-31')],
    'valid_officialPx_only': [('2019-05-01', '2024-02-14'), ('2025-09-24', '2026-10-31')],
    'select_cleanA_2007-01_2012-12': [('2007-01-01', '2012-12-31')],
    'full_2007-01_2026-10': [('2007-01-01', '2026-10-31')],
}
def netr(tr, cost):
    b, s = (0.0, 0.0) if cost == 'gross' else L.cost_rates(cost)
    return (1 + tr.ret) * (1 - s) / (1 + b) - 1

def event_stats(tr, cost):
    net = netr(tr, cost); abn = net - tr.ew; abna = net - tr.ewall
    out = {}
    for k, segs in W.items():
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
                  'payoff_ratio': round(float(nm_[nm_ > 0].mean() / -nm_[nm_ <= 0].mean()), 2) if (nm_ <= 0).any() and (nm_ > 0).any() else None,
                  'exit_mix': {k_: round(float(v), 3) for k_, v in x.why.value_counts(normalize=True).items()}}
    return out

def incr_stats(tr, flag, cost=0.001):
    """H52 事件拆成 flag（同時為 ATH）與非 ATH：月份分群異常差。"""
    net = netr(tr, cost); abn = net - tr.ew; out = {}
    for k, segs in W.items():
        m = np.zeros(len(tr), bool)
        for a, b_ in segs: m |= ((tr.d >= pd.Timestamp(a)) & (tr.d <= pd.Timestamp(b_))).to_numpy()
        x = tr[m]; f = x[flag].to_numpy().astype(bool)
        if f.sum() < 10 or (~f).sum() < 10: continue
        ga = abn[m][f].groupby(x.em[f]).mean(); gb = abn[m][~f].groupby(x.em[~f]).mean()
        dif = (ga - gb).dropna()
        out[k] = {'n_ATH': int(f.sum()), 'n_52wNotATH': int((~f).sum()), 'abn_ATH_pct': round(float(abn[m][f].mean() * 100), 2),
                  'abn_52wNotATH_pct': round(float(abn[m][~f].mean() * 100), 2), 't_ATH': L.tstat(ga), 't_52wNotATH': L.tstat(gb),
                  'diff_month_paired_pct': round(float(dif.mean() * 100), 2), 't_diff_paired': L.tstat(dif), 'n_paired_months': int(len(dif))}
    return out

EVENTS = {}
for nm in SIG:
    EVENTS[nm] = build_events(nm)
    print('events', nm, len(EVENTS[nm]), round(time.time() - T0), flush=True)
COSTS = ('gross', 0.0, 0.001, 'stress')
res['events'] = {nm: {str(c): event_stats(EVENTS[nm], c) for c in COSTS} for nm in SIG}
res['events_incremental_ATH_within_H52'] = {'raw (H52_raw 事件依 isATH_raw 拆)': incr_stats(EVENTS['H52_raw'], 'isATH_raw'),
                                           'adj (H52_adj 事件依 isATH_adj 拆)': incr_stats(EVENTS['H52_adj'], 'isATH_adj')}
b1, s1 = L.cost_rates(0.001)
res['events_by_year_slip0.1'] = {}
for nm, tr in EVENTS.items():
    net = netr(tr, 0.001)
    res['events_by_year_slip0.1'][nm] = {int(y): {'n': int(len(g)), 'win': round(float((net[g.index] > 0).mean()), 3),
                                               'mean_net_pct': round(float(net[g.index].mean() * 100), 2),
                                               'mean_abn_pct': round(float((net[g.index] - g.ew).mean() * 100), 2)}
                                         for y, g in tr.groupby(tr.d.dt.year)}
for nm in SIG:
    print(nm, {k: (v.get('mean_abn_pct'), v.get('t_month_cluster'), v.get('n_trades')) for k, v in res['events'][nm]['0.001'].items()}, flush=True)
print('incr', json.dumps(res['events_incremental_ATH_within_H52'], ensure_ascii=False)[:1500], flush=True)

# ---------------- 組合（最多 5 檔） ----------------
t_start = int(np.searchsorted(G, pd.Timestamp('2006-07-01')))
CAND = {}
for nm, E in ENT.items():
    lst = []
    for t in range(T):
        js = np.flatnonzero(E[t])
        lst.append(js[np.argsort(V60g[t, js])] if len(js) else js)    # 60 日均成交值由小到大
    CAND[nm] = lst

def sim(nm, buy, sell, maxpos=5, seed=None):
    rng = np.random.default_rng(seed) if seed is not None else None
    cash = 1.0; P = {}
    nav = np.empty(T - t_start); npos = np.empty(T - t_start); inv = np.empty(T - t_start)
    cnt = {'entries': 0, 'stop10': 0, 'rel_weak5': 0, 'ma50': 0, 'max_hold': 0, 'delist': 0, 'open_at_end': 0, 'skip_no_slot': 0, 'skip_gap': 0}
    trades = []
    for t in range(t_start, T):
        for j in list(P):
            p = P[j]
            if t == p['x']:
                px = p['p1'] if p['why'] in ('delist',) else AOg[t, j]
                if p['why'] == 'open_at_end': continue
                cash += p['u'] * px * (1 - sell); cnt[p['why']] += 1
                trades.append((p['e'], t, j, px / p['p0'] - 1, p['why'])); del P[j]
        navo = cash + sum(p['u'] * AOg[t, j] for j, p in P.items())
        unit = navo / maxpos
        if t > t_start:
            cands = CAND[nm][t - 1]
            if rng is not None and len(cands): cands = rng.permutation(cands)
            for j in cands:
                if j in P: continue
                if not OPOK[t, j]: continue
                if GAPg[t, j]: cnt['skip_gap'] += 1; continue
                if len(P) >= maxpos or cash < 0.25 * unit: cnt['skip_no_slot'] += 1; continue
                amt = min(unit, cash); px = AOg[t, j]
                x, p1, why, _ = find_exit(j, t, px)
                P[j] = {'u': amt / (1 + buy) / px, 'p0': px, 'e': t, 'x': x, 'p1': p1, 'why': why}
                cash -= amt; cnt['entries'] += 1
        val = sum(p['u'] * ACg[t, j] for j, p in P.items())
        k = t - t_start
        nav[k] = cash + val; npos[k] = len(P); inv[k] = val / (cash + val)
    ix = G[t_start:]
    return pd.Series(nav, index=ix), pd.Series(npos, index=ix), pd.Series(inv, index=ix), cnt, trades

_s = pd.Series(G, index=G)
MD = pd.DatetimeIndex([g[g.dt.day >= 11].iloc[0] for _, g in _s.groupby([G.year, G.month]) if (g.dt.day >= 11).any()])
MD = MD[MD >= pd.Timestamp('2006-07-01')]
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
port = {}; TRADES = {}
for nm in SIG:
    port[nm] = {}
    for cost in COSTS:
        b, s = (0.0, 0.0) if cost == 'gross' else L.cost_rates(cost)
        nav, npos, inv, cnt, trd = sim(nm, b, s)
        navs[f'{nm}_{cost}'] = nav
        o = {'counts': cnt, 'mean_open_positions': round(float(npos[npos.index >= '2007-01-01'].mean()), 2),
             'mean_invested_share': round(float(inv[inv.index >= '2007-01-01'].mean()), 3)}
        if cost == 0.001:
            TRADES[nm] = trd
            o['open_positions_by_year'] = {int(y): round(float(g.mean()), 2) for y, g in npos.groupby(npos.index.year)}
            o['entries_per_year'] = round(cnt['entries'] / ((G[-1] - G[t_start]).days / 365.25), 1)
            tdf = pd.DataFrame(trd, columns=['e', 'x', 'j', 'ret', 'why'])
            o['trade_mean_ret_pct'] = round(float(tdf.ret.mean() * 100), 2); o['trade_win_rate'] = round(float((tdf.ret > 0).mean()), 3)
            o['trade_mean_hold'] = round(float((tdf.x - tdf.e).mean()), 1)
        port[nm][str(cost)] = o
for nm in SIG:
    for cost in COSTS:
        nav = navs[f'{nm}_{cost}']
        extra = [] if nm.startswith('H52') else [('H52_same_basis', navs[f"H52_{nm.split('_')[1]}_{cost}"])]
        port[nm][str(cost)]['windows'] = {k: daily_stats(nav, sg, extra) for k, sg in W.items()}
        port[nm][str(cost)]['yearly_pct'] = yearly(nav[nav.index >= '2006-12-29'])
    w = port[nm]['0.001']['windows']
    print(nm, {k: (w[k]['CAGR'], w[k]['0050_CAGR'], w[k]['EW_CAGR'], w[k]['t_vs_EW_m'], w[k]['t_vs_0050_m'], w[k]['maxDD'])
               for k in ('select_2007-01_2019-04', 'valid_2019-05_2026-10', 'y2024-01_2025-12')}, port[nm]['0.001']['mean_open_positions'], round(time.time() - T0), flush=True)
res['portfolio'] = port
res['EW_pool_windows'] = {k: daily_stats(EW, sg) for k, sg in W.items()}
res['EW_pool_yearly'] = yearly(EW[EW.index >= '2006-12-29'])
res['0050_yearly'] = yearly(E50[E50.index >= '2006-12-29'])
res['pool_size_by_year'] = {int(y): round(float(np.mean([POOL[gpos[dd]].sum() for dd in MD if dd.year == y])), 0) for y in sorted(set(MD.year))}
res['signals_per_year_ENT'] = {nm: {int(y): int(E[G.year == y].sum()) for y in range(2007, 2027)} for nm, E in ENT.items()}
res['days_with_signal_share'] = {nm: round(float(E[G >= pd.Timestamp('2007-01-01')].any(axis=1).mean()), 3) for nm, E in ENT.items()}

rnd = {}
for seed in range(10):
    nav, *_ = sim('ATH_raw', b1, s1, seed=seed)
    for k in ('select_2007-01_2019-04', 'valid_2019-05_2026-10', 'y2024-01_2025-12'):
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
         'signal_with_factor_drop_20d': int(tr.fd20.sum()), 'signal_with_factor_drop_ever': int(tr.fdever.sum()),
         'extreme_examples': big.assign(d=big.d.astype(str), code=[COLS[j] for j in big.j])[['d', 'code', 'ret', 'why', 'hold']].round(3).head(12).to_dict('records'),
         'stop10_exit_mean_ret_pct': round(float(tr[tr.why == 'stop10'].ret.mean() * 100), 2),
         'top10_abn_share_valid': None}
    for lab, keep in (('ex_flag345_extremes', ~f345 & (tr.ret.abs() <= 1.0).to_numpy()), ('ex_factor_drop_20d', ~tr.fd20.to_numpy()), ('ex_factor_drop_ever', ~tr.fdever.to_numpy())):
        st = event_stats(tr[keep], 0.001)
        c[lab] = {k: (st[k].get('mean_abn_pct'), st[k].get('t_month_cluster'), st[k].get('n_trades')) for k in ('select_2007-01_2019-04', 'valid_2019-05_2026-10')}
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
            'pass': bool(w['t_vs_EW_m'] >= 2 and w['CAGR'] >= w['0050_CAGR'])}
J = {}
for nm in SIG:
    J[nm] = {k: passes(nm, k) for k in ('select_2007-01_2019-04', 'valid_2019-05_2026-10')}
    npass = sum(v['pass'] for v in J[nm].values())
    J[nm]['verdict'] = '可用' if npass == 2 else ('待觀察' if npass == 1 else '不可用')
v = J['ATH_raw']['verdict']
if v == '不可用' and J['ATH_adj']['verdict'] == '可用': v = '待觀察'
J['overall'] = v
res['judgement'] = J
print('judgement', json.dumps(J, ensure_ascii=False), flush=True)
(HERE / 'results.json').write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str), encoding='utf-8')
pd.DataFrame(navs).to_parquet(HERE / 'nav.parquet')
for nm, tr in EVENTS.items():
    tr.assign(code=[COLS[j] for j in tr.j], d=tr.d.astype(str), em=tr.em.astype(str)).drop(columns=['j']).to_csv(HERE / f'events_{nm}.csv.gz', index=False)
print('done', round(time.time() - T0))
