#!/usr/bin/env python3
"""Weinstein Stage 2 突破：週線訊號、日頻可成交回測。執行：python3 -I run.py
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
Cg = L.C.to_numpy()[gi]; V60g = M.V60.to_numpy()[gi]; VALg = L.V.to_numpy()[gi]
common = np.array([len(c) == 4 and c.isdigit() and not c.startswith('91') for c in COLS])
LIQ = (V60g >= 1e7) & (Cg >= 10) & common[None, :] & np.isfinite(ADJg)

# ---------------- 週線 ----------------
wk = G.to_period('W-FRI')                      # 週一～週五
wkey = pd.Index(wk)
first_of_week = np.r_[True, wkey[1:] != wkey[:-1]]
last_of_week = np.r_[wkey[1:] != wkey[:-1], True]
WE = np.flatnonzero(last_of_week)               # 每週最後交易日（G 索引）
K = len(WE)
week_of_day = np.cumsum(first_of_week) - 1      # 每日屬於第幾週
adjG = pd.DataFrame(ADJg, index=G, columns=COLS)
Wc_df = adjG.groupby(week_of_day).last()       # 週內最後有效還原收盤
Wc = Wc_df.to_numpy()
TRADED_WE = np.isfinite(ADJg[WE])               # 該週最後交易日有收盤
Wv_df = pd.DataFrame(VALg, index=G, columns=COLS).groupby(week_of_day).sum(min_count=1)
tx = pd.read_csv('/home/user/jonatw/finlab-txd/data/raw/taiex_twii.csv', parse_dates=['date']).set_index('date')['close'].sort_index()
TXg = tx.reindex(G, method='ffill').to_numpy()
TXw = TXg[WE]
MA30 = Wc_df.rolling(30, min_periods=30).mean()
RS = Wc_df.div(TXw, axis=0)
MRS = ((RS / RS.rolling(52, min_periods=52).mean()) - 1) * 100
slope10 = MA30 / MA30.shift(10) - 1
BASE = (slope10.shift(1).abs() < 0.02).to_numpy()
UPPER = Wc_df.rolling(10, min_periods=10).max().shift(1).to_numpy()
L10 = Wc_df.rolling(10, min_periods=10).min().shift(1).to_numpy()
VOLC = {'V1': (Wv_df >= 2 * Wv_df.shift(1).rolling(4, min_periods=4).mean()).to_numpy(),
        'V2': (Wv_df >= 1.5 * Wv_df.shift(1).rolling(20, min_periods=20).median()).to_numpy()}
MA30n = MA30.to_numpy(); MRSn = MRS.to_numpy()
with np.errstate(invalid='ignore'):
    CORE = (LIQ[WE] & TRADED_WE & (Wc > UPPER) & (Wc > MA30n) & (MA30n >= MA30.shift(1).to_numpy()) & (MRSn > 0) & BASE)
    EXA = Wc < MA30n                            # X1：週收盤跌破 MA30w
ENTRY = {v: CORE & VOLC[v] for v in VOLC}
# 下週第一個交易日須存在
ENTRY = {v: e & (WE + 1 < T)[:, None] for v, e in ENTRY.items()}
res['condition_pass_share'] = {nm: round(float(np.nanmean(np.where(LIQ[WE] & TRADED_WE, x, np.nan))), 4) for nm, x in (
    ('close_gt_upper', Wc > UPPER), ('close_gt_MA30', Wc > MA30n), ('MA30_up', MA30n >= MA30.shift(1).to_numpy()), ('MRS_pos', MRSn > 0),
    ('base_flat', BASE), ('V1', VOLC['V1']), ('V2', VOLC['V2']), ('CORE_wo_vol', CORE), ('ENTRY_V1', ENTRY['V1']), ('ENTRY_V2', ENTRY['V2']))}
res['signals_per_year'] = {v: round(float(e.sum() / ((G[-1] - G[0]).days / 365.25)), 0) for v, e in ENTRY.items()}
lastvalid = np.array([np.where(np.isfinite(ADJg[:, j]))[0].max() if np.isfinite(ADJg[:, j]).any() else -1 for j in range(N)])
print('setup', round(time.time() - T0), res['signals_per_year'], flush=True)

def exit_week(j, s, xmode):
    """從訊號週 s 之後的第一個週（進場週 s+1）起找出場週。"""
    if xmode == 'X1':
        hit = EXA[s + 1:, j]
    else:
        stop = np.fmax.accumulate(np.nan_to_num(L10[s:, j], nan=-np.inf))[1:]
        with np.errstate(invalid='ignore'):
            hit = Wc[s + 1:, j] < stop
    k = np.flatnonzero(hit & TRADED_WE[s + 1:, j])
    return s + 1 + k[0] if len(k) else None

def exit_fill(j, kx):
    """出場週 kx → (日索引, 價, 原因)。"""
    lv = lastvalid[j]
    if kx is None:
        if lv < T - 1: return min(lv + 1, T - 1), ACg[lv, j], 'delist'
        return T - 1, ACg[T - 1, j], 'open_at_end'
    t = WE[kx]
    for x in range(t + 1, lv + 1):
        if OPOK[x, j]: return x, AOg[x, j], 'signal'
    if t + 1 > T - 1: return T - 1, ACg[T - 1, j], 'open_at_end'
    return min(lv + 1, T - 1), ACg[lv, j], 'delist'

# ---------------- 事件（全部訊號） ----------------
def build_events(v, xmode):
    ENT = ENTRY[v]; rows = []
    for j in range(N):
        sigs = np.flatnonzero(ENT[:, j]); nxt = -1
        for s in sigs:
            if WE[s] < nxt: continue
            e = WE[s] + 1
            if not OPOK[e, j] or GAPg[e, j]: continue
            kx = exit_week(j, s, xmode)
            x, p1, why = exit_fill(j, kx)
            p0 = AOg[e, j]
            rows.append((s, WE[s], e, x, j, p0, p1, why, MRSn[s, j]))
            nxt = x            # 出場日（含）之前的訊號忽略
    tr = pd.DataFrame(rows, columns=['sw', 's', 'e', 'x', 'j', 'p0', 'p1', 'why', 'mrs'])
    tr['ret'] = tr.p1 / tr.p0 - 1; tr['hold'] = tr.x - tr.e
    ew = np.empty(len(tr)); pools = {}
    for i_, r in enumerate(tr.itertuples()):
        pl = pools.get(r.s)
        if pl is None:
            pl = np.flatnonzero(LIQ[r.s] & OPOK[r.s + 1]); pools[r.s] = pl
        px1 = ACg[r.x, pl] if r.why == 'open_at_end' else AOg[r.x, pl]
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

def event_stats(tr, cost):
    b, s = (0.0, 0.0) if cost == 'gross' else L.cost_rates(cost)
    net = (1 + tr.ret) * (1 - s) / (1 + b) - 1; abn = net - tr.ew
    out = {}
    for k, segs in W.items():
        m = np.zeros(len(tr), bool)
        for a, b_ in segs: m |= ((tr.d >= pd.Timestamp(a)) & (tr.d <= pd.Timestamp(b_))).to_numpy()
        if m.sum() < 10: continue
        x = tr[m]; g = abn[m].groupby(x.em).mean()
        out[k] = {'n_trades': int(m.sum()), 'n_months': int(len(g)), 'win_rate_net': round(float((net[m] > 0).mean()), 3),
                  'mean_ret_gross_pct': round(float(x.ret.mean() * 100), 2), 'mean_net_ret_pct': round(float(net[m].mean() * 100), 2),
                  'median_net_ret_pct': round(float(net[m].median() * 100), 2), 'mean_poolEW_same_window_pct': round(float(x.ew.mean() * 100), 2),
                  'mean_abn_pct': round(float(abn[m].mean() * 100), 2), 'month_cluster_mean_abn_pct': round(float(g.mean() * 100), 2),
                  't_month_cluster': L.tstat(g), 'mean_hold_days': round(float(x.hold.mean()), 1),
                  'payoff_ratio': round(float(net[m][net[m] > 0].mean() / -net[m][net[m] <= 0].mean()), 2),
                  'exit_mix': {k_: round(float(v), 3) for k_, v in x.why.value_counts(normalize=True).items()}}
    return out

VARS = {'V1X1': ('V1', 'X1'), 'V2X1': ('V2', 'X1'), 'V1X2': ('V1', 'X2'), 'V2X2': ('V2', 'X2')}
EVENTS = {}
for vn, (v, xm) in VARS.items():
    EVENTS[vn] = build_events(v, xm)
    print('events', vn, len(EVENTS[vn]), round(time.time() - T0), flush=True)
res['events'] = {vn: {str(c): event_stats(EVENTS[vn], c) for c in ('gross', 0.001, 'stress')} for vn in VARS}
b1, s1 = L.cost_rates(0.001)
res['events_by_year_slip0.1'] = {}
for vn, tr in EVENTS.items():
    net = (1 + tr.ret) * (1 - s1) / (1 + b1) - 1
    res['events_by_year_slip0.1'][vn] = {int(y): {'n': int(len(g)), 'win': round(float((net[g.index] > 0).mean()), 3),
                                               'mean_net_pct': round(float(net[g.index].mean() * 100), 2),
                                               'mean_abn_pct': round(float((net[g.index] - g.ew).mean() * 100), 2)}
                                         for y, g in tr.groupby(tr.d.dt.year)}
for vn in VARS:
    print(vn, {k: (v['n_trades'], v['mean_abn_pct'], v['t_month_cluster']) for k, v in res['events'][vn]['0.001'].items()}, flush=True)

# ---------------- 組合（最多 20 檔） ----------------
t_start = int(np.searchsorted(G, pd.Timestamp('2004-07-01')))
# 執行日 t（= 訊號週最後交易日 + 1）→ 候選（依 MRS 由大到小）
CAND = {}
for v, ENT in ENTRY.items():
    d = {}
    for k in range(K):
        js = np.flatnonzero(ENT[k])
        if len(js): d[WE[k] + 1] = (js[np.argsort(-MRSn[k, js])], k)
    CAND[v] = d
WEEK_END_AT = {int(t): k for k, t in enumerate(WE)}

def sim(v, xmode, buy, sell, maxpos=20, seed=None):
    rng = np.random.default_rng(seed) if seed is not None else None
    cash = 1.0; P = {}
    nav = np.empty(T - t_start); npos = np.empty(T - t_start); inv = np.empty(T - t_start)
    cnt = {'entries': 0, 'exits_signal': 0, 'delist': 0, 'skip_no_slot': 0, 'skip_gap': 0, 'skip_held': 0}
    for t in range(t_start, T):
        for j in list(P):
            p = P[j]
            if t > lastvalid[j]:
                cash += p['u'] * ACg[t, j] * (1 - sell); del P[j]; cnt['delist'] += 1
            elif p['px'] and OPOK[t, j]:
                cash += p['u'] * AOg[t, j] * (1 - sell); cnt['exits_signal'] += 1; del P[j]
        navo = cash + sum(p['u'] * AOg[t, j] for j, p in P.items())
        unit = navo / maxpos
        if t in CAND[v] and t > t_start:
            cands, ks = CAND[v][t]
            if rng is not None: cands = rng.permutation(cands)
            for j in cands:
                if j in P: cnt['skip_held'] += 1; continue
                if not OPOK[t, j]: continue
                if GAPg[t, j]: cnt['skip_gap'] += 1; continue
                if len(P) >= maxpos or cash < 0.25 * unit: cnt['skip_no_slot'] += 1; continue
                amt = min(unit, cash); px = AOg[t, j]
                st0 = L10[ks, j] if np.isfinite(L10[ks, j]) else -np.inf
                P[j] = {'u': amt / (1 + buy) / px, 'stop': st0, 'px': False}
                cash -= amt; cnt['entries'] += 1
        val = 0.0
        kw = WEEK_END_AT.get(t)
        for j, p in P.items():
            val += p['u'] * ACg[t, j]
            if kw is None or p['px'] or not TRADED_WE[kw, j]: 
                if kw is not None and xmode == 'X2' and np.isfinite(L10[kw, j]): p['stop'] = max(p['stop'], L10[kw, j])
                continue
            if xmode == 'X1':
                if EXA[kw, j]: p['px'] = True
            else:
                if np.isfinite(L10[kw, j]): p['stop'] = max(p['stop'], L10[kw, j])
                if Wc[kw, j] < p['stop']: p['px'] = True
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
for vn, (v, xm) in VARS.items():
    port[vn] = {}
    for cost in ('gross', 0.001, 'stress'):
        b, s = (0.0, 0.0) if cost == 'gross' else L.cost_rates(cost)
        nav, npos, inv, cnt = sim(v, xm, b, s)
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
res['EW_pool_yearly'] = yearly(EW[EW.index >= '2004-12-31'])
res['0050_yearly'] = yearly(E50[E50.index >= '2004-12-31'])
res['pool_size_by_year'] = {int(y): round(float(np.mean([LIQ[gpos[d]].sum() for d in MD if d.year == y])), 0) for y in sorted(set(MD.year))}

rnd = {}
for seed in range(10):
    nav, *_ = sim('V1', 'X1', b1, s1, seed=seed)
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
for vn, tr in EVENTS.items():
    f345 = np.array([bool(np.isin(FL[r.e:r.x + 1, r.j], [3, 4, 5]).any()) for r in tr.itertuples()])
    big = tr[(tr.ret > 1.0) | (tr.ret < -0.5)]
    keep = ~f345 & (tr.ret.abs() <= 1.0).to_numpy()
    st = event_stats(tr[keep], 0.001)
    chk[vn] = {'trades_with_flag345': int(f345.sum()), 'n_trades': int(len(tr)), 'entry_src3_share': round(float((SRC[tr.e, tr.j] == 3).mean()), 3),
               'n_ret_gt_100pct': int((tr.ret > 1.0).sum()), 'n_ret_lt_-50pct': int((tr.ret < -0.5).sum()),
               'extreme_examples': big.assign(d=big.d.astype(str), code=[COLS[j] for j in big.j])[['d', 'code', 'ret', 'why', 'hold']].round(3).head(12).to_dict('records'),
               'abn_ex_flag345_and_extremes': {k: (st[k]['mean_abn_pct'], st[k]['t_month_cluster']) for k in ('select_2005-01_2019-04', 'valid_2019-05_2026-10') if k in st}}
res['data_checks'] = chk

def passes(vn, k):
    w = port[vn]['0.001']['windows'][k]
    return {'CAGR': w['CAGR'], '0050_CAGR': w['0050_CAGR'], 'EW_CAGR': w['EW_CAGR'], 't_vs_EW_m': w['t_vs_EW_m'], 't_vs_0050_m': w['t_vs_0050_m'],
            'pass': bool(w['t_vs_EW_m'] >= 2 and w['CAGR'] >= w['0050_CAGR'])}
J = {}
for vn in VARS:
    J[vn] = {k: passes(vn, k) for k in ('select_2005-01_2019-04', 'valid_2019-05_2026-10')}
    npass = sum(x['pass'] for x in J[vn].values())
    J[vn]['verdict'] = '可用' if npass == 2 else ('待觀察' if npass == 1 else '不可用')
v = J['V1X1']['verdict']
if v == '不可用' and any(J[x]['verdict'] == '可用' for x in ('V2X1', 'V1X2', 'V2X2')): v = '待觀察'
J['overall'] = v
res['judgement'] = J
print('judgement', json.dumps(J, ensure_ascii=False), flush=True)
(HERE / 'results.json').write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str), encoding='utf-8')
pd.DataFrame(navs).to_parquet(HERE / 'nav.parquet')
for vn, tr in EVENTS.items():
    tr.assign(code=[COLS[j] for j in tr.j], d=tr.d.astype(str), em=tr.em.astype(str)).drop(columns=['j']).to_csv(HERE / f'events_{vn}.csv.gz', index=False)
print('done', round(time.time() - T0))
