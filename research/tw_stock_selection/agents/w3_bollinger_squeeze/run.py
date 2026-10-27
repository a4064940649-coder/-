#!/usr/bin/env python3
"""布林通道壓縮（Bollinger Squeeze）長歷史可成交回測。執行：python3 -I run.py
規則見 prereg.py（PREREG 已先寫入 results.json，本檔只附加結果）。
成交、成本、漲停判定、V60 缺漏修補沿用 momentum_family/mlib.py → exec_rebuild/exlib.py（只讀取）。
事件／組合引擎骨架改寫自 w3_vcp_contraction_breakout/run.py（只讀取）。"""
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
adjG = L.adj.loc[GOOD]; CG = L.C.loc[GOOD]; VG = L.V.loc[GOOD]; HG = L.H.loc[GOOD]
Cg = CG.to_numpy(); Hg = HG.to_numpy(); RETg = L.RET.loc[GOOD].to_numpy()
V60g = M.V60.to_numpy()[gi]
common = np.array([not c.startswith('91') for c in COLS])
with np.errstate(invalid='ignore'):
    LIQ = (V60g >= 1e7) & (Cg >= 10) & common[None, :] & np.isfinite(ADJg)
POOL = LIQ
ALL = common[None, :] & np.isfinite(ADJg)

# ---------------- 布林、量能、Keltner ----------------
MBd = adjG.rolling(20, min_periods=20).mean()
SDd = adjG.rolling(20, min_periods=20).std(ddof=0)
BWd = 4 * SDd / MBd
MB = MBd.to_numpy(); UB = (MBd + 2 * SDd).to_numpy(); LB = (MBd - 2 * SDd).to_numpy(); BW = BWd.to_numpy()
BWMIN = BWd.rolling(125, min_periods=120).min().to_numpy()
print('bb', round(time.time() - T0), flush=True)
BWQ5 = BWd.rolling(250, min_periods=240).quantile(0.05).to_numpy()
print('q5', round(time.time() - T0), flush=True)
SH = VG / CG.where(CG > 0)
VOL20 = SH.rolling(20, min_periods=20).mean()
VRATIO = (SH / VOL20).to_numpy()
AF = adjG.ffill(limit=10)
HA = HG * (adjG / CG.where(CG > 0))
cp = AF.shift(1)
TR = np.maximum((adjG - cp).abs(), HA - cp)
ATR = TR.rolling(20, min_periods=20).mean()
EMA = AF.ewm(span=20, adjust=False, min_periods=20).mean().where(adjG.notna())
KCU = (EMA + 1.5 * ATR).to_numpy(); KCL = (EMA - 1.5 * ATR).to_numpy()
with np.errstate(invalid='ignore'):
    SQ_MAIN = np.isfinite(BW) & np.isfinite(BWMIN) & (BW <= BWMIN * (1 + 1e-9))
    SQ_PCT5 = np.isfinite(BW) & np.isfinite(BWQ5) & (BW <= BWQ5)
    SQ_TTM = np.isfinite(UB) & np.isfinite(KCU) & (UB < KCU) & (LB > KCL)
    ABOVE = ADJg > UB; BELOW = ADJg < LB; ABOVE_MB = ADJg > MB; BELOW_MB = ADJg < MB
    VOLOK = VRATIO > 1.5
    UNIV = LIQ
    ABOVEp = np.vstack([np.zeros((1, N), bool), ABOVE[:-1]])
    lim = np.where(G < pd.Timestamp('2015-06-01'), 0.07, 0.10)[:, None]
    LOCK = (RETg >= lim - L.LTOL) & (Cg >= Hg * 0.999)
t05 = G >= pd.Timestamp('2005-01-01')
res['squeeze_share_of_liquid_stock_days'] = {nm: round(float((A_ & LIQ)[t05].sum() / LIQ[t05].sum()), 4) for nm, A_ in
                                             (('MAIN', SQ_MAIN), ('PCT5', SQ_PCT5), ('TTM', SQ_TTM))}
res['median_BW_liquid'] = round(float(np.nanmedian(np.where(LIQ & t05[:, None], BW, np.nan))), 4)
print('setup', round(time.time() - T0), res['squeeze_share_of_liquid_stock_days'], flush=True)

def scan_breakout(SQ, needvol):
    out = []
    for j in range(N):
        sq = SQ[:, j]
        if not sq.any(): continue
        ab = ABOVE[:, j]; vo = VOLOK[:, j]; un = UNIV[:, j]
        s = -999; used = True
        for t in np.flatnonzero(sq | ab):
            if sq[t]: s = t; used = False
            if used or t - s > 10: continue
            if ab[t]:
                used = True
                if un[t] and (vo[t] or not needvol): out.append((t, j))
    return pd.DataFrame(out, columns=['t', 'j'])

def scan_headfake(SQ, needvol):
    out = []
    for j in range(N):
        sq = SQ[:, j]
        if not sq.any(): continue
        ab = ABOVE[:, j]; be = BELOW[:, j]; am = ABOVE_MB[:, j]; vo = VOLOK[:, j]; un = UNIV[:, j]
        state = 0; s = b = -999
        for t in range(int(np.argmax(sq)), T):
            if sq[t]: s = t; state = 1
            if state == 1:
                if t - s > 10: state = 0
                elif ab[t]: state = 0
                elif be[t]: state = 2; b = t; continue
            if state == 2:
                if t - b > 10: state = 0
                elif am[t]:
                    state = 0
                    if un[t] and (vo[t] or not needvol): out.append((t, j))
    return pd.DataFrame(out, columns=['t', 'j'])

tt, jj = np.nonzero(ABOVE & ~ABOVEp & VOLOK & UNIV)
VAR = {
    'MAIN': scan_breakout(SQ_MAIN, True),
    'a_NOVOL': scan_breakout(SQ_MAIN, False),
    'b_PCT5': scan_breakout(SQ_PCT5, True),
    'c_TTM': scan_breakout(SQ_TTM, True),
    'd_HEADFAKE': scan_headfake(SQ_MAIN, False),
    'd_HEADFAKE_VOL': scan_headfake(SQ_MAIN, True),
    'BB_BREAK': pd.DataFrame({'t': tt, 'j': jj}),
}
VAR = {k: v.sort_values(['t', 'j']).reset_index(drop=True) for k, v in VAR.items()}
VAR = {k: v[t05[v.t.to_numpy()]].reset_index(drop=True) for k, v in VAR.items()}
res['signal_counts'] = {k: int(len(v)) for k, v in VAR.items()}
res['signal_diag'] = {k: {'lockup_close_on_signal_day_share': round(float(LOCK[v.t, v.j].mean()), 3) if len(v) else None,
                          'signal_src3_share': None} for k, v in VAR.items()}
print('signals', res['signal_counts'], round(time.time() - T0), flush=True)

lastvalid = np.array([np.where(np.isfinite(ADJg[:, j]))[0].max() if np.isfinite(ADJg[:, j]).any() else -1 for j in range(N)])
MAXH = 40
def find_exit(j, e):
    lv = lastvalid[j]
    end = min(e + MAXH, lv + 1)
    k = np.flatnonzero(BELOW_MB[e:end, j])
    if len(k):
        t = e + k[0]; why = 'below_MB'
    elif e + MAXH <= lv:
        t = e + MAXH - 1; why = 'time_40d'
    else:
        if lv < T - 1: return min(lv + 1, T - 1), ACg[lv, j], 'delist'
        return T - 1, ACg[T - 1, j], 'open_at_end'
    for x in range(t + 1, lv + 1):
        if OPOK[x, j]:
            return x, AOg[x, j], why
    if t + 1 > T - 1: return T - 1, ACg[T - 1, j], 'open_at_end'
    return min(lv + 1, T - 1), ACg[lv, j], 'delist'

E50 = L.E50.reindex(G).ffill()
E50n = E50.to_numpy()
def build_events(df):
    rows = []; nblk = 0
    for j, g in df.groupby('j'):
        nxt = 0
        for r in g.itertuples():
            s = r.t
            if s < nxt: continue
            e = s + 1
            if e >= T: break
            if not OPOK[e, j] or GAPg[e, j]: nblk += 1; continue
            p0 = AOg[e, j]
            x, p1, why = find_exit(j, e)
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
    xe = np.where(tr.why == 'open_at_end', tr.x, tr.x - 1)
    tr['r50'] = E50n[xe] / E50n[tr.e - 1] - 1
    tr['d'] = G[tr.s.to_numpy()]; tr['em'] = G[tr.e.to_numpy()].to_period('M')
    tr.attrs['n_blocked'] = nblk
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
    net = netr(tr, cost); abn = net - tr.ew; abna = net - tr.ewall; ab50 = net - tr.r50
    out = {}
    for k, segs in windows.items():
        m = np.zeros(len(tr), bool)
        for a, b_ in segs: m |= ((tr.d >= pd.Timestamp(a)) & (tr.d <= pd.Timestamp(b_))).to_numpy()
        if m.sum() < 10: out[k] = {'n_trades': int(m.sum())}; continue
        x = tr[m]; g = abn[m].groupby(x.em).mean(); ga = abna[m].groupby(x.em).mean(); g5 = ab50[m].groupby(x.em).mean()
        nm_ = net[m]
        out[k] = {'n_trades': int(m.sum()), 'n_months': int(len(g)), 'win_rate_net': round(float((nm_ > 0).mean()), 3),
                  'mean_ret_gross_pct': round(float(x.ret.mean() * 100), 2), 'mean_net_ret_pct': round(float(nm_.mean() * 100), 2),
                  'median_net_ret_pct': round(float(nm_.median() * 100), 2), 'mean_poolEW_same_window_pct': round(float(x.ew.mean() * 100), 2),
                  'mean_abn_pct': round(float(abn[m].mean() * 100), 2), 'month_cluster_mean_abn_pct': round(float(g.mean() * 100), 2),
                  't_month_cluster': L.tstat(g), 'mean_abn_vs_allListedEW_pct': round(float(abna[m].mean() * 100), 2), 't_month_vs_allListedEW': L.tstat(ga),
                  'mean_ex_vs_0050_pct': round(float(ab50[m].mean() * 100), 2), 'month_cluster_ex_vs_0050_pct': round(float(g5.mean() * 100), 2),
                  't_month_vs_0050': L.tstat(g5),
                  'mean_hold_days': round(float(x.hold.mean()), 1),
                  'exit_mix': {k_: round(float(v), 3) for k_, v in x.why.value_counts(normalize=True).items()}}
    return out

EVENTS = {}
for nm, df in VAR.items():
    EVENTS[nm] = build_events(df)
    print('events', nm, len(EVENTS[nm]), round(time.time() - T0), flush=True)
COSTS = ('gross', 0.0, 0.001, 'stress')
res['events'] = {nm: {str(c): event_stats(EVENTS[nm], c) for c in COSTS} for nm in VAR}
res['events_blocked_gap_or_noopen'] = {nm: tr.attrs['n_blocked'] for nm, tr in EVENTS.items()}
res['events_by_year_slip0.1'] = {}
for nm, tr in EVENTS.items():
    net = netr(tr, 0.001)
    res['events_by_year_slip0.1'][nm] = {int(y): {'n': int(len(g)), 'win': round(float((net[g.index] > 0).mean()), 3),
                                               'mean_net_pct': round(float(net[g.index].mean() * 100), 2),
                                               'mean_abn_pct': round(float((net[g.index] - g.ew).mean() * 100), 2)}
                                         for y, g in tr.groupby(tr.d.dt.year)}
for nm in VAR:
    print(nm, {k: (v.get('mean_abn_pct'), v.get('t_month_cluster'), v.get('month_cluster_ex_vs_0050_pct'), v.get('n_trades')) for k, v in res['events'][nm]['0.001'].items()}, flush=True)

# ---------------- 組合（最多 20 檔，參考） ----------------
t_start = int(np.searchsorted(G, pd.Timestamp('2005-01-01')))
CAND = {}
for nm, df in VAR.items():
    lst = [np.array([], int)] * T
    for t, g in df.groupby('t'):
        js = g.j.to_numpy(); vr = np.nan_to_num(VRATIO[t, js], nan=0)
        lst[t] = js[np.argsort(-vr)]
    CAND[nm] = lst

def sim(nm, buy, sell, maxpos=20):
    lst = CAND[nm]
    cash = 1.0; P = {}
    nav = np.empty(T - t_start); npos = np.empty(T - t_start); inv = np.empty(T - t_start)
    cnt = {'entries': 0, 'below_MB': 0, 'time_40d': 0, 'delist': 0, 'open_at_end': 0, 'skip_no_slot': 0, 'skip_gap': 0}
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
            for j in lst[t - 1]:
                if j in P: continue
                if not OPOK[t, j]: continue
                if GAPg[t, j]: cnt['skip_gap'] += 1; continue
                if len(P) >= maxpos or cash < 0.25 * unit: cnt['skip_no_slot'] += 1; continue
                amt = min(unit, cash); px = AOg[t, j]
                x, p1, why = find_exit(j, t)
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
    for nm_, bn in [('0050', E50), ('EW', EW)] + (extra or []):
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
for nm in VAR:
    port[nm] = {}
    for cost in ('gross', 0.001, 'stress'):
        b, s = (0.0, 0.0) if cost == 'gross' else L.cost_rates(cost)
        nav, npos, inv, cnt, trd = sim(nm, b, s)
        navs[f'{nm}_{cost}'] = nav
        o = {'counts': cnt, 'mean_open_positions': round(float(npos.mean()), 2), 'mean_invested_share': round(float(inv.mean()), 3)}
        if cost == 0.001:
            o['invested_share_by_year'] = {int(y): round(float(g.mean()), 3) for y, g in inv.groupby(inv.index.year)}
            tdf = pd.DataFrame(trd, columns=['s', 'e', 'x', 'j', 'ret', 'why'])
            o['entries_per_year'] = round(cnt['entries'] / ((G[-1] - G[t_start]).days / 365.25), 1)
            o['trade_mean_ret_pct'] = round(float(tdf.ret.mean() * 100), 2); o['trade_win_rate'] = round(float((tdf.ret > 0).mean()), 3)
            for k in ('select_2005-01_2019-04', 'valid_2019-05_2026-10'):
                a, b_ = W[k][0]; hold_m = (npos.index >= a) & (npos.index <= b_)
                o[f'mean_open_positions_{k}'] = round(float(npos[hold_m].mean()), 2)
                o[f'mean_invested_{k}'] = round(float(inv[hold_m].mean()), 3)
        port[nm][str(cost)] = o
for nm in VAR:
    for cost in ('gross', 0.001, 'stress'):
        nav = navs[f'{nm}_{cost}']
        extra = [('BB_BREAK_same_cost', navs[f'BB_BREAK_{cost}'])] if nm != 'BB_BREAK' else []
        port[nm][str(cost)]['windows'] = {k: daily_stats(nav, sg, extra) for k, sg in W.items()}
        port[nm][str(cost)]['yearly_pct'] = yearly(nav)
    w = port[nm]['0.001']['windows']
    print(nm, {k: (w[k]['CAGR'], w[k]['0050_CAGR'], w[k]['EW_CAGR'], w[k]['t_vs_EW_m'], w[k]['t_vs_0050_m'], w[k]['maxDD'])
               for k in ('select_2005-01_2019-04', 'valid_2019-05_2026-10', 'y2024-01_2025-12')}, port[nm]['0.001']['mean_open_positions'], round(time.time() - T0), flush=True)
res['portfolio'] = port
res['EW_pool_windows'] = {k: daily_stats(EW, sg) for k, sg in W.items()}
res['EW_pool_yearly'] = yearly(EW[EW.index >= '2004-12-31'])
res['0050_yearly'] = yearly(E50[E50.index >= '2004-12-31'])
res['signals_per_year'] = {nm: {int(y): int((G[df.t].year == y).sum()) for y in range(2005, 2027)} for nm, df in VAR.items()}

# ---------------- 資料檢查 ----------------
FL = pd.read_parquet(L.P / 'flag.parquet').reindex(index=IDX, columns=COLS).to_numpy()[gi]
SRC = pd.read_parquet(L.P / 'src.parquet').reindex(index=IDX, columns=COLS).to_numpy()[gi]
for nm, v in VAR.items():
    if len(v): res['signal_diag'][nm]['signal_src3_share'] = round(float((SRC[v.t, v.j] == 3).mean()), 3)
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
    for lab, a_, b_ in (('select', '2005-01-01', '2019-04-30'), ('valid', '2019-05-01', '2026-10-31')):
        mv = ((tr.d >= a_) & (tr.d <= b_)).to_numpy(); a = abn[mv].sort_values(ascending=False)
        c[f'top10_abn_share_{lab}'] = round(float(a.iloc[:10].sum() / a.sum()), 3) if a.sum() > 0 else None
        c[f'median_abn_pct_{lab}'] = round(float(a.median() * 100), 2) if len(a) else None
    chk[nm] = c
res['data_checks'] = chk

# ---------------- 判定（事件、MAIN、滑價 0.1%） ----------------
def ppass(nm, k):
    w = res['events'][nm]['0.001'][k]
    if w.get('n_trades', 0) < 30 or w.get('n_months', 0) < 12:
        return {'n_trades': w.get('n_trades'), 'pass': None}
    return {'n_trades': w['n_trades'], 'mean_abn_pct': w['mean_abn_pct'], 't_month_cluster': w['t_month_cluster'],
            'month_cluster_ex_vs_0050_pct': w['month_cluster_ex_vs_0050_pct'], 't_month_vs_0050': w['t_month_vs_0050'],
            'pass': bool(w['t_month_cluster'] >= 2 and w['mean_abn_pct'] > 0 and w['month_cluster_ex_vs_0050_pct'] >= 0)}
J = {}
for nm in VAR:
    J[nm] = {k: ppass(nm, k) for k in ('select_2005-01_2019-04', 'valid_2019-05_2026-10')}
    ps = [v['pass'] for v in J[nm].values()]
    J[nm]['verdict'] = '無法測試' if any(p is None for p in ps) else ('可用' if all(ps) else ('待觀察' if any(ps) else '不可用'))
J['overall'] = J['MAIN']['verdict']
res['judgement'] = J
print('judgement', json.dumps(J, ensure_ascii=False), flush=True)
(HERE / 'results.json').write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str), encoding='utf-8')
pd.DataFrame(navs).to_parquet(HERE / 'nav.parquet')
for nm, tr in EVENTS.items():
    if nm == 'BB_BREAK': continue
    tr.assign(code=[COLS[j] for j in tr.j], d=tr.d.astype(str), em=tr.em.astype(str)).drop(columns=['j']).to_csv(HERE / f'events_{nm}.csv.gz', index=False)
print('done', round(time.time() - T0))
