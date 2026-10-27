#!/usr/bin/env python3
"""海龜通道（S1 20/10、S2 55/20、2N 停損、加碼）長歷史可成交回測。執行：python3 -I run.py
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
# ---------------- 只用有值檔數 >= 300 的日子 ----------------
GOOD = (L.adj.notna().sum(axis=1) >= 300).to_numpy()
gi = np.where(GOOD)[0]; G = IDX[gi]; T = len(G)
res['dropped_sparse_days'] = [str(d.date()) for d in IDX[~GOOD]]
ACg = L.AC[gi]; AOg = L.AO[gi]; ADJg = L.ADJ_RAW[gi]
OPOK = np.isfinite(L.OPEN_RAW[gi]) & np.isfinite(L.ADJ_RAW[gi])
GAPg = M.GAP_UP[gi]
adjG = L.adj.loc[GOOD]
AHG = (L.H * (L.adj / L.C)).loc[GOOD]
res['high_coverage_given_adj'] = round(float((AHG.notna() & adjG.notna()).sum().sum() / adjG.notna().sum().sum()), 5)
Cg = L.C.to_numpy()[gi]
V60g = M.V60.to_numpy()[gi]
common = np.array([not c.startswith('91') for c in COLS])
LIQ = (V60g >= 1e7) & (Cg >= 10) & common[None, :] & np.isfinite(ADJg)
NV = adjG.diff().abs().rolling(20, min_periods=20).mean().to_numpy()

def chan(nin, nout):
    hh = AHG.rolling(nin, min_periods=nin).max().shift(1).to_numpy()
    ll = adjG.rolling(nout, min_periods=nout).min().shift(1).to_numpy()
    with np.errstate(invalid='ignore'):
        ent = LIQ & (ADJg > hh) & np.isfinite(NV) & (NV > 0)
        ext = ADJg < ll
    return ent, ext

SYS = {'S1': chan(20, 10), 'S2': chan(55, 20)}
lastvalid = np.array([np.where(np.isfinite(ADJg[:, j]))[0].max() if np.isfinite(ADJg[:, j]).any() else -1 for j in range(N)])
print('setup', round(time.time() - T0), flush=True)

def find_exit(j, e, stop, EXc, ADc):
    """從 e 收盤起找第一個出場訊號日 t，回傳 (出場日 x, 價, 原因)。"""
    lv = lastvalid[j]; start = e; why = None; t = None
    for w in (32, 256, 100000):
        a, b = start, min(lv + 1, e + w)
        if a >= b: break
        seg_x = EXc[j, a:b]; seg_p = ADc[j, a:b]
        with np.errstate(invalid='ignore'):
            hit = seg_x | (seg_p < stop)
        k = np.flatnonzero(hit)
        if len(k):
            t = a + k[0]; why = 'channel' if seg_x[k[0]] else 'stop2N'; break
        start = b
    if t is None:
        if lv < T - 1:
            return min(lv + 1, T - 1), ACg[lv, j], 'delist'
        return T - 1, ACg[T - 1, j], 'open_at_end'
    for x in range(t + 1, lv + 1):
        if OPOK[x, j]:
            return x, AOg[x, j], why
    if t + 1 > T - 1:
        return T - 1, ACg[T - 1, j], 'open_at_end'
    return min(lv + 1, T - 1), ACg[lv, j], 'delist'

# ---------------- 事件（不加碼、全部訊號） ----------------
def build_events(name):
    ENT, EXT = SYS[name]
    EXc = np.ascontiguousarray(EXT.T); ADc = np.ascontiguousarray(ADJg.T)
    rows = []
    for j in range(N):
        sigs = np.flatnonzero(ENT[:, j]); nxt = 0
        for s in sigs:
            if s < nxt: continue
            e = s + 1
            if e >= T: break
            if not OPOK[e, j] or GAPg[e, j]: continue
            p0 = AOg[e, j]; n0 = NV[s, j]
            x, p1, why = find_exit(j, e, p0 - 2 * n0, EXc, ADc)
            rows.append((s, e, x, j, p0, p1, why, n0 / p0))
            nxt = x            # 出場日（含）之前的新訊號忽略；出場日收盤可再觸發
    tr = pd.DataFrame(rows, columns=['s', 'e', 'x', 'j', 'p0', 'p1', 'why', 'n_over_p'])
    tr['ret'] = tr.p1 / tr.p0 - 1; tr['hold'] = tr.x - tr.e
    ew = np.empty(len(tr))
    pools = {}
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

EVENTS = {}
for nm in SYS:
    EVENTS[nm] = build_events(nm)
    print('events', nm, len(EVENTS[nm]), round(time.time() - T0), flush=True)
res['events_no_pyramid'] = {nm: {str(c): event_stats(EVENTS[nm], c) for c in ('gross', 0.001, 'stress')} for nm in SYS}
b1, s1 = L.cost_rates(0.001)
res['events_by_year_slip0.1'] = {}
for nm, tr in EVENTS.items():
    net = (1 + tr.ret) * (1 - s1) / (1 + b1) - 1
    res['events_by_year_slip0.1'][nm] = {int(y): {'n': int(len(g)), 'win': round(float((net[g.index] > 0).mean()), 3),
                                               'mean_net_pct': round(float(net[g.index].mean() * 100), 2),
                                               'mean_abn_pct': round(float((net[g.index] - g.ew).mean() * 100), 2)}
                                         for y, g in tr.groupby(tr.d.dt.year)}
for nm in SYS:
    print(nm, {k: (v['mean_abn_pct'], v['t_month_cluster']) for k, v in res['events_no_pyramid'][nm]['0.001'].items()}, flush=True)

# ---------------- 組合（最多 20 檔） ----------------
t_start = int(np.searchsorted(G, pd.Timestamp('2004-07-01')))
CAND = {}
for nm, (ENT, EXT) in SYS.items():
    lst = []
    for t in range(T):
        js = np.flatnonzero(ENT[t])
        lst.append(js[np.argsort(-V60g[t, js])] if len(js) else js)
    CAND[nm] = lst

def sim(nm, buy, sell, div=None, maxpos=20, order='liq', seed=None):
    """div=None 不加碼（每檔 NAV/20）；div=80/40 加碼（1 單位 = NAV/div，最多 4 單位）。"""
    ENT, EXT = SYS[nm]; pyr = div is not None; unit_div = div if pyr else maxpos
    rng = np.random.default_rng(seed) if seed is not None else None
    cash = 1.0; P = {}
    nav = np.empty(T - t_start); npos = np.empty(T - t_start); inv = np.empty(T - t_start)
    cnt = {'entries': 0, 'adds': 0, 'channel': 0, 'stop2N': 0, 'delist': 0, 'skip_no_slot': 0, 'skip_gap': 0}
    for t in range(t_start, T):
        for j in list(P):
            p = P[j]
            if t > lastvalid[j]:
                cash += p['u'] * ACg[t, j] * (1 - sell); del P[j]; cnt['delist'] += 1
            elif p['px'] and OPOK[t, j]:
                cash += p['u'] * AOg[t, j] * (1 - sell); cnt[p['why']] += 1; del P[j]
        navo = cash + sum(p['u'] * AOg[t, j] for j, p in P.items())
        unit = navo / unit_div
        if pyr:
            for j, p in P.items():
                if p['pa']:
                    p['pa'] = False
                    if OPOK[t, j] and not GAPg[t, j]:
                        amt = min(unit, cash)
                        if amt >= 0.25 * unit:
                            px = AOg[t, j]; p['u'] += amt / (1 + buy) / px; cash -= amt
                            p['units'] += 1; p['last'] = px; p['stop'] = px - 2 * p['n0']; cnt['adds'] += 1
        if t > t_start:
            cands = CAND[nm][t - 1]
            if rng is not None and len(cands): cands = rng.permutation(cands)
            for j in cands:
                if j in P: continue
                if not OPOK[t, j]: continue
                if GAPg[t, j]: cnt['skip_gap'] += 1; continue
                if len(P) >= maxpos or cash < 0.25 * unit: cnt['skip_no_slot'] += 1; continue
                amt = min(unit, cash); px = AOg[t, j]; n0 = NV[t - 1, j]
                P[j] = {'u': amt / (1 + buy) / px, 'last': px, 'n0': n0, 'stop': px - 2 * n0, 'units': 1, 'px': False, 'pa': False, 'why': None}
                cash -= amt; cnt['entries'] += 1
        val = 0.0
        for j, p in P.items():
            val += p['u'] * ACg[t, j]
            if p['px']: continue
            c = ADJg[t, j]
            if not np.isfinite(c): continue
            if EXT[t, j]: p['px'] = True; p['why'] = 'channel'
            elif c < p['stop']: p['px'] = True; p['why'] = 'stop2N'
            elif pyr and p['units'] < 4 and c >= p['last'] + 0.5 * p['n0']: p['pa'] = True
        k = t - t_start
        nav[k] = cash + val; npos[k] = len(P); inv[k] = val / (cash + val)
    ix = G[t_start:]
    return pd.Series(nav, index=ix), pd.Series(npos, index=ix), pd.Series(inv, index=ix), cnt

# 月曆（每月第一個 >= 11 日交易日）與同池等權
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

VARS = {'S1_noPyr': ('S1', None), 'S1_pyr_U80': ('S1', 80), 'S2_noPyr': ('S2', None), 'S2_pyr_U80': ('S2', 80),
        'diag_S1_pyr_U40': ('S1', 40), 'diag_S2_pyr_U40': ('S2', 40)}
navs = {'EW_pool': EW, '0050': E50}
port = {}
for vn, (nm, div) in VARS.items():
    port[vn] = {}
    for cost in ('gross', 0.001, 'stress'):
        b, s = (0.0, 0.0) if cost == 'gross' else L.cost_rates(cost)
        nav, npos, inv, cnt = sim(nm, b, s, div)
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

# 隨機優先順序診斷（PRIMARY、滑價 0.1%）
rnd = {}
for seed in range(10):
    nav, *_ = sim('S1', b1, s1, None, order='random', seed=seed)
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
    nfl = sum(bool(np.isin(FL[r.e:r.x + 1, r.j], [3, 4, 5]).any()) for r in tr.itertuples())
    big = tr[(tr.ret > 1.0) | (tr.ret < -0.5)]
    chk[nm] = {'trades_with_flag345': int(nfl), 'n_trades': int(len(tr)), 'entry_src3_share': round(float((SRC[tr.e, tr.j] == 3).mean()), 3),
               'n_ret_gt_100pct': int((tr.ret > 1.0).sum()), 'n_ret_lt_-50pct': int((tr.ret < -0.5).sum()),
               'extreme_examples': big.assign(d=big.d.astype(str), code=[COLS[j] for j in big.j])[['d', 'code', 'ret', 'why', 'hold']].round(3).head(15).to_dict('records'),
               'stop_exit_mean_ret_pct': round(float(tr[tr.why == 'stop2N'].ret.mean() * 100), 2),
               'channel_exit_mean_ret_pct': round(float(tr[tr.why == 'channel'].ret.mean() * 100), 2),
               'median_2N_over_price_pct': round(float(2 * tr.n_over_p.median() * 100), 2),
               'abn_ex_flag345_and_extremes_select_valid': {}}
    # 剔除 flag345 與極端值後的事件異常（滑價 0.1%）
    keep = np.array([not np.isin(FL[r.e:r.x + 1, r.j], [3, 4, 5]).any() for r in tr.itertuples()]) & (tr.ret.abs() <= 1.0).to_numpy()
    st = event_stats(tr[keep], 0.001)
    chk[nm]['abn_ex_flag345_and_extremes_select_valid'] = {k: (st[k]['mean_abn_pct'], st[k]['t_month_cluster']) for k in ('select_2005-01_2019-04', 'valid_2019-05_2026-10') if k in st}
res['data_checks'] = chk

# ---------------- 判定 ----------------
def passes(vn, k):
    w = port[vn]['0.001']['windows'][k]
    return {'CAGR': w['CAGR'], '0050_CAGR': w['0050_CAGR'], 'EW_CAGR': w['EW_CAGR'], 't_vs_EW_m': w['t_vs_EW_m'], 't_vs_0050_m': w['t_vs_0050_m'],
            'pass': bool(w['t_vs_EW_m'] >= 2 and w['CAGR'] >= w['0050_CAGR'])}
J = {}
for vn in ('S1_noPyr', 'S1_pyr_U80', 'S2_noPyr', 'S2_pyr_U80'):
    J[vn] = {k: passes(vn, k) for k in ('select_2005-01_2019-04', 'valid_2019-05_2026-10')}
    npass = sum(v['pass'] for v in J[vn].values())
    J[vn]['verdict'] = '可用' if npass == 2 else ('待觀察' if npass == 1 else '不可用')
v = J['S1_noPyr']['verdict']
if v == '不可用' and any(J[x]['verdict'] == '可用' for x in ('S1_pyr_U80', 'S2_noPyr', 'S2_pyr_U80')): v = '待觀察'
J['overall'] = v
res['judgement'] = J
print('judgement', json.dumps(J, ensure_ascii=False), flush=True)
(HERE / 'results.json').write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str), encoding='utf-8')
pd.DataFrame(navs).to_parquet(HERE / 'nav.parquet')
for nm, tr in EVENTS.items():
    tr.assign(code=[COLS[j] for j in tr.j], d=tr.d.astype(str), em=tr.em.astype(str)).drop(columns=['j']).to_csv(HERE / f'events_{nm}.csv.gz', index=False)
print('done', round(time.time() - T0))
