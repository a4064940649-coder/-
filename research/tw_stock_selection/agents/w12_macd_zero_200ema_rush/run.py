#!/usr/bin/env python3
"""Trading Rush MACD（200EMA 之上、零軸下黃金交叉、1.5R）長歷史可成交回測。執行：python3 -I run.py
規則見 prereg.py（PREREG 已先寫入 results.json，本檔只附加結果）。引擎骨架沿用 w1_turtle_donchian/run.py；
成交、成本、漲停、60 日均值缺漏修補用 momentum_family/mlib.py → exec_rebuild/exlib.py（只讀取、不修改）。"""
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
adjG = L.adj.loc[GOOD]
Cg = L.C.to_numpy()[gi]
V60g = M.V60.to_numpy()[gi]
common = np.array([(len(c) == 4) and not c.startswith('91') for c in COLS])
LIQ = (V60g >= 1e7) & (Cg >= 10) & common[None, :] & np.isfinite(ADJg)

def ema(X, n): return X.ewm(span=n, adjust=False, min_periods=n).mean()
# EMA 只在有值的日子遞推（逐欄 dropna），避免停牌 NaN 影響；再對回日曆
def ema_valid(X, n):
    out = pd.DataFrame(np.nan, index=X.index, columns=X.columns)
    for c in X.columns:
        s = X[c].dropna()
        if len(s): out.loc[s.index, c] = s.ewm(span=n, adjust=False, min_periods=n).mean()
    return out
E200 = ema_valid(adjG, 200)
E12 = ema_valid(adjG, 12); E26 = ema_valid(adjG, 26)
DIFdf = E12 - E26
DEAdf = ema_valid(DIFdf, 9)
DIF = DIFdf.to_numpy(); DEA = DEAdf.to_numpy(); EMA200 = E200.to_numpy()
# 前一個有效值（交叉用前一個有效交易日）
DIFp = DIFdf.apply(lambda s: s.dropna().shift(1).reindex(s.index)).to_numpy()
DEAp = DEAdf.apply(lambda s: s.dropna().shift(1).reindex(s.index)).to_numpy()
with np.errstate(invalid='ignore'):
    XUP = (DIF > DEA) & (DIFp <= DEAp)
    XDN = (DIF < DEA) & (DIFp >= DEAp)
    ENT_RUSH = LIQ & XUP & (DIF < 0) & (DEA < 0) & (ADJg > EMA200)
    ENT_PLAIN = LIQ & XUP
    DEPTH = DIF / ADJg
STOPP = adjG.rolling(10, min_periods=10).min().to_numpy()   # adj_{t-9..t} 最低
res['signal_counts'] = {'rush': int(ENT_RUSH.sum()), 'plain_cross': int(ENT_PLAIN.sum()),
                        'rush_share_of_plain': round(float(ENT_RUSH.sum() / ENT_PLAIN.sum()), 4)}
lastvalid = np.array([np.where(np.isfinite(ADJg[:, j]))[0].max() if np.isfinite(ADJg[:, j]).any() else -1 for j in range(N)])
print('setup', round(time.time() - T0), res['signal_counts'], flush=True)

EXITS = {  # name: (entry matrix, use_stop, use_target, use_xdn)
    'A_1.5R': (ENT_RUSH, True, True, False),
    'B_xdn_stop': (ENT_RUSH, True, False, True),
    'diag_B0_pure_xdn': (ENT_RUSH, False, False, True),
    'diag_plain_cross_1.5R': (ENT_PLAIN, True, True, False),
}
HOLD = 60

def exit_signal_day(j, e, stop, tgt, use_s, use_t, use_x):
    """回傳出場訊號日 t（收盤判斷）與原因；None 表示資料結束/下市前未觸發。時間出場：t=e+59。"""
    lv = lastvalid[j]; end = min(lv, e + HOLD - 1, T - 1)
    seg = ADJg[e:end + 1, j]
    with np.errstate(invalid='ignore'):
        hs = (seg < stop) if use_s else np.zeros(len(seg), bool)
        ht = (seg >= tgt) if use_t else np.zeros(len(seg), bool)
        hx = XDN[e:end + 1, j] if use_x else np.zeros(len(seg), bool)
    hit = hs | ht | hx; k = np.flatnonzero(hit)
    if len(k):
        k = k[0]; why = 'stop' if hs[k] else ('target' if ht[k] else 'xdn'); return e + k, why
    if e + HOLD - 1 <= min(lv, T - 1): return e + HOLD - 1, 'time'
    return None, None

def exec_exit(j, t, why):
    lv = lastvalid[j]
    if t is None:
        if lv < T - 1: return min(lv + 1, T - 1), ACg[lv, j], 'delist'
        return T - 1, ACg[T - 1, j], 'open_at_end'
    for x in range(t + 1, lv + 1):
        if OPOK[x, j]: return x, AOg[x, j], why
    if t + 1 > T - 1: return T - 1, ACg[T - 1, j], 'open_at_end'
    return min(lv + 1, T - 1), ACg[lv, j], 'delist'

ALLPOOL = {}
def build_events(name):
    ENT, us, ut, ux = EXITS[name]
    rows = []; ninv = 0; ngap = 0
    for j in range(N):
        sigs = np.flatnonzero(ENT[:, j]); nxt = 0
        for s in sigs:
            if s < nxt: continue
            e = s + 1
            if e >= T: break
            if not OPOK[e, j]: continue
            if GAPg[e, j]: ngap += 1; continue
            p0 = AOg[e, j]; stop = STOPP[s, j]
            if not np.isfinite(stop):
                continue
            if us and p0 <= stop: ninv += 1; continue
            tgt = p0 + 1.5 * (p0 - stop)
            t, why = exit_signal_day(j, e, stop, tgt, us, ut, ux)
            x, p1, why = exec_exit(j, t, why)
            rows.append((s, e, x, j, p0, p1, why, (p0 - stop) / p0, DEPTH[s, j]))
            nxt = x
    tr = pd.DataFrame(rows, columns=['s', 'e', 'x', 'j', 'p0', 'p1', 'why', 'risk_pct', 'depth'])
    tr['ret'] = tr.p1 / tr.p0 - 1; tr['hold'] = tr.x - tr.e
    ew = np.empty(len(tr)); ewa = np.empty(len(tr)); pools = {}
    for i_, r in enumerate(tr.itertuples()):
        pl = pools.get(r.s)
        if pl is None:
            pl = np.flatnonzero(LIQ[r.s] & OPOK[r.s + 1]); pools[r.s] = pl
        pa = ALLPOOL.get(r.s)
        if pa is None:
            pa = np.flatnonzero(common & OPOK[r.s + 1] & np.isfinite(ADJg[r.s])); ALLPOOL[r.s] = pa
        endp = ACg if r.why == 'open_at_end' else AOg
        with np.errstate(invalid='ignore', divide='ignore'):
            ew[i_] = np.nanmean(endp[r.x, pl] / AOg[r.e, pl] - 1)
            ewa[i_] = np.nanmean(endp[r.x, pa] / AOg[r.e, pa] - 1)
    tr['ew'] = ew; tr['ew_all'] = ewa
    tr['d'] = G[tr.s.to_numpy()]; tr['em'] = G[tr.e.to_numpy()].to_period('M')
    build_events.info = {'skipped_invalid_risk': ninv, 'skipped_gap_up': ngap}
    return tr

W = {
    'select_2005-01_2019-04': [('2005-01-01', '2019-04-30')],
    'valid_2019-05_2026-10': [('2019-05-01', '2026-10-31')],
    'y2024-01_2025-12': [('2024-01-01', '2025-12-31')],
    'valid_survivor_2019-05_2023-12': [('2019-05-01', '2023-12-31')],
    'valid_officialPx_only': [('2019-05-01', '2024-02-14'), ('2025-09-24', '2026-10-31')],
    'full_2005-01_2026-10': [('2005-01-01', '2026-10-31')],
}

def event_stats(tr, cost):
    b, s = (0.0, 0.0) if cost == 'gross' else L.cost_rates(cost)
    net = (1 + tr.ret) * (1 - s) / (1 + b) - 1; abn = net - tr.ew; abna = net - tr.ew_all
    out = {}
    for k, segs in W.items():
        m = np.zeros(len(tr), bool)
        for a, b_ in segs: m |= ((tr.d >= pd.Timestamp(a)) & (tr.d <= pd.Timestamp(b_))).to_numpy()
        if m.sum() < 10: continue
        x = tr[m]; g = abn[m].groupby(x.em).mean(); ga = abna[m].groupby(x.em).mean()
        out[k] = {'n_trades': int(m.sum()), 'n_months': int(len(g)), 'win_rate_net': round(float((net[m] > 0).mean()), 3),
                  'mean_ret_gross_pct': round(float(x.ret.mean() * 100), 2), 'mean_net_ret_pct': round(float(net[m].mean() * 100), 2),
                  'median_net_ret_pct': round(float(net[m].median() * 100), 2), 'mean_poolEW_same_window_pct': round(float(x.ew.mean() * 100), 2),
                  'mean_abn_pct': round(float(abn[m].mean() * 100), 2), 'month_cluster_mean_abn_pct': round(float(g.mean() * 100), 2),
                  't_month_cluster': L.tstat(g),
                  'mean_abn_vs_allListedEW_pct': round(float(abna[m].mean() * 100), 2), 't_month_cluster_vs_allListedEW': L.tstat(ga),
                  'mean_hold_days': round(float(x.hold.mean()), 1), 'median_risk_pct': round(float(x.risk_pct.median() * 100), 2),
                  'payoff_ratio': round(float(net[m][net[m] > 0].mean() / -net[m][net[m] <= 0].mean()), 2),
                  'exit_mix': {k_: round(float(v), 3) for k_, v in x.why.value_counts(normalize=True).items()}}
    return out

EVENTS = {}; res['event_build_info'] = {}
for nm in EXITS:
    EVENTS[nm] = build_events(nm); res['event_build_info'][nm] = build_events.info
    print('events', nm, len(EVENTS[nm]), build_events.info, round(time.time() - T0), flush=True)
COSTS = ('gross', 0.0, 0.001, 'stress')
res['events'] = {nm: {str(c): event_stats(EVENTS[nm], c) for c in COSTS} for nm in EXITS}
b1, s1 = L.cost_rates(0.001)
res['events_by_year_slip0.1'] = {}
for nm, tr in EVENTS.items():
    net = (1 + tr.ret) * (1 - s1) / (1 + b1) - 1
    res['events_by_year_slip0.1'][nm] = {int(y): {'n': int(len(g)), 'win': round(float((net[g.index] > 0).mean()), 3),
                                               'mean_net_pct': round(float(net[g.index].mean() * 100), 2),
                                               'mean_abn_pct': round(float((net[g.index] - g.ew).mean() * 100), 2)}
                                         for y, g in tr.groupby(tr.d.dt.year)}
for nm in EXITS:
    print(nm, {k: (v['mean_abn_pct'], v['t_month_cluster']) for k, v in res['events'][nm]['0.001'].items()}, flush=True)

# ---------------- 組合（最多 20 檔） ----------------
t_start = int(np.searchsorted(G, pd.Timestamp('2004-07-01')))
def cand_lists(ENT, key):
    lst = []
    for t in range(T):
        js = np.flatnonzero(ENT[t])
        if len(js): js = js[np.argsort(key[t, js], kind='stable')]
        lst.append(js)
    return lst
CAND = {'rush': cand_lists(ENT_RUSH, DEPTH), 'plain': cand_lists(ENT_PLAIN, DEPTH)}

def sim(name, buy, sell, maxpos=20, seed=None, log=None):
    ENT, us, ut, ux = EXITS[name]
    cands_all = CAND['plain' if ENT is ENT_PLAIN else 'rush']
    rng = np.random.default_rng(seed) if seed is not None else None
    cash = 1.0; P = {}
    nav = np.empty(T - t_start); npos = np.empty(T - t_start); inv = np.empty(T - t_start)
    cnt = {'entries': 0, 'stop': 0, 'target': 0, 'xdn': 0, 'time': 0, 'delist': 0, 'skip_no_slot': 0, 'skip_gap': 0, 'skip_invalid_risk': 0}
    for t in range(t_start, T):
        for j in list(P):
            p = P[j]
            if t > lastvalid[j]:
                cash += p['u'] * ACg[t, j] * (1 - sell); del P[j]; cnt['delist'] += 1
            elif p['px'] and OPOK[t, j]:
                cash += p['u'] * AOg[t, j] * (1 - sell); cnt[p['why']] += 1
                if log is not None: log.append((p['e'], t, j, AOg[t, j] / p['p0'] - 1, p['why']))
                del P[j]
        navo = cash + sum(p['u'] * AOg[t, j] for j, p in P.items())
        unit = navo / maxpos
        if t > t_start:
            cands = cands_all[t - 1]
            if rng is not None and len(cands): cands = rng.permutation(cands)
            for j in cands:
                if j in P or not OPOK[t, j]: continue
                if GAPg[t, j]: cnt['skip_gap'] += 1; continue
                px = AOg[t, j]; stop = STOPP[t - 1, j]
                if not np.isfinite(stop): continue
                if us and px <= stop: cnt['skip_invalid_risk'] += 1; continue
                if len(P) >= maxpos or cash < 0.25 * unit: cnt['skip_no_slot'] += 1; continue
                amt = min(unit, cash)
                P[j] = {'u': amt / (1 + buy) / px, 'e': t, 'p0': px, 'stop': stop, 'tgt': px + 1.5 * (px - stop), 'px': False, 'why': None}
                cash -= amt; cnt['entries'] += 1
        val = 0.0
        for j, p in P.items():
            val += p['u'] * ACg[t, j]
            if p['px']: continue
            c = ADJg[t, j]
            if not np.isfinite(c): continue
            if us and c < p['stop']: p['px'] = True; p['why'] = 'stop'
            elif ut and c >= p['tgt']: p['px'] = True; p['why'] = 'target'
            elif ux and XDN[t, j]: p['px'] = True; p['why'] = 'xdn'
            elif t - p['e'] >= HOLD - 1: p['px'] = True; p['why'] = 'time'
        k = t - t_start
        nav[k] = cash + val; npos[k] = len(P); inv[k] = val / (cash + val)
    ix = G[t_start:]
    return pd.Series(nav, index=ix), pd.Series(npos, index=ix), pd.Series(inv, index=ix), cnt

_s = pd.Series(G, index=G)
MD = pd.DatetimeIndex([g[g.dt.day >= 11].iloc[0] for _, g in _s.groupby([G.year, G.month]) if (g.dt.day >= 11).any()])
MD = MD[MD >= pd.Timestamp('2004-07-01')]
gpos = {d: i for i, d in enumerate(G)}
ev_ew = {}
for d in MD[:-1]:
    i = gpos[d]
    if i + 1 >= T: continue
    pl = np.flatnonzero(LIQ[i] & OPOK[i + 1])
    if len(pl) >= 50: ev_ew[int(gi[i + 1])] = ('open', pl)
t0f = min(ev_ew); ewf, _ = L.sim_portfolio(ev_ew, 0.0, 0.0, t0f, len(IDX) - 1)
EW = pd.concat([pd.Series([1.0], index=[IDX[t0f - 1]]), ewf]).reindex(G).ffill().dropna()
E50 = L.E50.reindex(G).ffill()
MDX = MD.append(pd.DatetimeIndex([G[-1]])) if MD[-1] != G[-1] else MD

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
        o[f'excess_vs_{nm_}_pct_m'] = round(float(ex.mean() * 100), 2); o[f't_vs_{nm_}_m'] = L.tstat(ex)
        o[f'hit_vs_{nm_}_m'] = round(float((ex > 0).mean()), 3)
        bd = bn.pct_change().reindex(rr.index); eb = (1 + bd.fillna(0)).cumprod()
        o[f'{nm_}_CAGR'] = round(float(eb.iloc[-1] ** (1 / span) - 1), 4); o[f'{nm_}_maxDD'] = round(float((eb / eb.cummax() - 1).min()), 4)
    return o

def yearly(nav):
    r = nav.pct_change().dropna(); return {int(y): round(float((1 + g).prod() - 1) * 100, 1) for y, g in r.groupby(r.index.year)}

navs = {'EW_pool': EW, '0050': E50}
port = {}
for vn in EXITS:
    port[vn] = {}
    for cost in COSTS:
        b, s = (0.0, 0.0) if cost == 'gross' else L.cost_rates(cost)
        nav, npos, inv, cnt = sim(vn, b, s)
        navs[f'{vn}_{cost}'] = nav
        o = {'counts': cnt, 'mean_open_positions': round(float(npos[npos.index >= '2005-01-01'].mean()), 1),
             'mean_invested_share': round(float(inv[inv.index >= '2005-01-01'].mean()), 3),
             'windows': {k: daily_stats(nav, sg) for k, sg in W.items()}, 'yearly_pct': yearly(nav[nav.index >= '2004-12-31'])}
        if cost == 0.001:
            o['open_positions_by_year'] = {int(y): round(float(g.mean()), 1) for y, g in npos.groupby(npos.index.year)}
            o['invested_by_year'] = {int(y): round(float(g.mean()), 3) for y, g in inv.groupby(inv.index.year)}
            o['entries_per_year'] = round(cnt['entries'] / ((G[-1] - G[t_start]).days / 365.25), 0)
        port[vn][str(cost)] = o
    w = port[vn]['0.001']['windows']
    print(vn, {k: (w[k]['CAGR'], w[k]['0050_CAGR'], w[k]['EW_CAGR'], w[k]['t_vs_EW_m'], w[k]['t_vs_0050_m'], w[k]['maxDD'])
               for k in ('select_2005-01_2019-04', 'valid_2019-05_2026-10', 'y2024-01_2025-12')}, round(time.time() - T0), flush=True)
res['portfolio'] = port
res['EW_pool_windows'] = {k: daily_stats(EW, sg) for k, sg in W.items()}
res['0050_windows'] = {k: daily_stats(E50, sg) for k, sg in W.items()}
res['EW_pool_yearly'] = yearly(EW[EW.index >= '2004-12-31'])
res['0050_yearly'] = yearly(E50[E50.index >= '2004-12-31'])
res['pool_size_by_year'] = {int(y): round(float(np.mean([LIQ[gpos[d]].sum() for d in MD if d.year == y])), 0) for y in sorted(set(MD.year))}

rnd = {}
for seed in range(10):
    nav, *_ = sim('A_1.5R', b1, s1, seed=seed)
    for k in ('select_2005-01_2019-04', 'valid_2019-05_2026-10', 'y2024-01_2025-12'):
        st = daily_stats(nav, W[k]); rnd.setdefault(k, []).append((st['CAGR'], st['t_vs_EW_m'], st['t_vs_0050_m']))
res['primary_random_order_diag'] = {k: {'CAGR_mean': round(float(np.mean([x[0] for x in v])), 4), 'CAGR_min': round(float(np.min([x[0] for x in v])), 4),
                                        'CAGR_max': round(float(np.max([x[0] for x in v])), 4), 't_vs_EW_mean': round(float(np.mean([x[1] for x in v])), 2),
                                        't_vs_EW_max': round(float(np.max([x[1] for x in v])), 2), 't_vs_0050_mean': round(float(np.mean([x[2] for x in v])), 2)}
                                    for k, v in rnd.items()}
# 診斷：組合實際成交的交易 vs 全部事件（毛報酬、同池同窗等權）
plog = []; sim('A_1.5R', b1, s1, log=plog)
pt = pd.DataFrame(plog, columns=['e', 'x', 'j', 'ret', 'why'])
pew = []
for r in pt.itertuples():
    pl = np.flatnonzero(LIQ[r.e - 1] & OPOK[r.e])
    with np.errstate(invalid='ignore'): pew.append(np.nanmean(AOg[r.x, pl] / AOg[r.e, pl] - 1))
pt['ew'] = pew; pt['d'] = G[pt.e.to_numpy()]
dg = {}
for k in ('select_2005-01_2019-04', 'valid_2019-05_2026-10'):
    a, b_ = W[k][0]; m = (pt.d >= a) & (pt.d <= b_); x = pt[m]
    dg[k] = {'n': int(m.sum()), 'mean_gross_ret_pct': round(float(x.ret.mean() * 100), 2), 'mean_poolEW_pct': round(float(x.ew.mean() * 100), 2),
             'mean_gross_abn_pct': round(float((x.ret - x.ew).mean() * 100), 2), 'mean_hold': round(float((x.x - x.e).mean()), 1),
             'exit_mix': {k_: round(float(v), 3) for k_, v in x.why.value_counts(normalize=True).items()}}
res['portfolio_trades_diag_A_slip0.1'] = dg
print('ptrades', dg, flush=True)
print('random', res['primary_random_order_diag'], round(time.time() - T0), flush=True)

FL = pd.read_parquet(L.P / 'flag.parquet').reindex(index=IDX, columns=COLS).to_numpy()[gi]
SRC = pd.read_parquet(L.P / 'src.parquet').reindex(index=IDX, columns=COLS).to_numpy()[gi]
chk = {}
for nm, tr in EVENTS.items():
    f345 = np.array([bool(np.isin(FL[r.e:r.x + 1, r.j], [3, 4, 5]).any()) for r in tr.itertuples()])
    big = tr[(tr.ret > 1.0) | (tr.ret < -0.5)]
    keep = ~f345 & (tr.ret.abs() <= 1.0).to_numpy()
    st = event_stats(tr[keep], 0.001)
    chk[nm] = {'trades_with_flag345': int(f345.sum()), 'n_trades': int(len(tr)), 'entry_src3_share': round(float((SRC[tr.e, tr.j] == 3).mean()), 3),
               'n_ret_gt_100pct': int((tr.ret > 1.0).sum()), 'n_ret_lt_-50pct': int((tr.ret < -0.5).sum()),
               'extreme_examples': big.assign(d=big.d.astype(str), code=[COLS[j] for j in big.j])[['d', 'code', 'ret', 'why', 'hold']].round(3).head(10).to_dict('records'),
               'mean_ret_by_exit_pct': {k: round(float(g.ret.mean() * 100), 2) for k, g in tr.groupby('why')},
               'abn_ex_flag345_and_extremes': {k: (st[k]['mean_abn_pct'], st[k]['t_month_cluster']) for k in ('select_2005-01_2019-04', 'valid_2019-05_2026-10') if k in st}}
res['data_checks'] = chk

def passes(vn, k):
    w = port[vn]['0.001']['windows'][k]
    return {'CAGR': w['CAGR'], '0050_CAGR': w['0050_CAGR'], 'EW_CAGR': w['EW_CAGR'], 't_vs_EW_m': w['t_vs_EW_m'], 't_vs_0050_m': w['t_vs_0050_m'],
            'pass': bool(w['t_vs_EW_m'] >= 2 and w['CAGR'] >= w['0050_CAGR'])}
J = {}
for vn in ('A_1.5R', 'B_xdn_stop'):
    J[vn] = {k: passes(vn, k) for k in ('select_2005-01_2019-04', 'valid_2019-05_2026-10')}
    npass = sum(v['pass'] for v in J[vn].values())
    J[vn]['verdict'] = '可用' if npass == 2 else ('待觀察' if npass == 1 else '不可用')
v = J['A_1.5R']['verdict']
if v == '不可用' and J['B_xdn_stop']['verdict'] == '可用': v = '待觀察'
J['overall'] = v
res['judgement'] = J
print('judgement', json.dumps(J, ensure_ascii=False), flush=True)
(HERE / 'results.json').write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str), encoding='utf-8')
pd.DataFrame(navs).to_parquet(HERE / 'nav.parquet')
tr = EVENTS['A_1.5R']
tr.assign(code=[COLS[j] for j in tr.j], d=tr.d.astype(str), em=tr.em.astype(str)).drop(columns=['j']).to_csv(HERE / 'events_A.csv.gz', index=False)
print('done', round(time.time() - T0))
