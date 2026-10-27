#!/usr/bin/env python3
"""0050 定期定額＋跌深分批加碼：回測主程式。先核對 PREREG 雜湊，結果寫入 results.json['results']。"""
import json, hashlib, math, sys
from pathlib import Path
import numpy as np, pandas as pd
H = Path(__file__).resolve().parent; AG = H.parent
sys.path.insert(0, str(AG.parent)); from btlib import tstat  # noqa
RJ = json.loads((H / 'results.json').read_text())
assert hashlib.sha256(json.dumps(RJ['PREREG'], ensure_ascii=False, sort_keys=True).encode()).hexdigest() == RJ['PREREG_sha256'], 'PREREG changed'
PR = RJ['PREREG']
FEE = 0.001425 * 0.6
COSTS = {'slip0.001': (FEE + 0.001, FEE + 0.001 + 0.001), 'stress': (0.005, 0.005), 'gross': (0.0, 0.0)}

e50 = pd.read_csv('/home/user/jonatw/finlab-txd/data/raw/etf_0050.csv', parse_dates=['date']).set_index('date')['adj'].sort_index()
e50.loc['2026-09-20'] = np.nan
E = e50.ffill(); IDX = E.index; EV = E.to_numpy(); NT = len(IDX)
R1 = E.pct_change().to_numpy()
LOCK = np.nan_to_num(R1) >= 0.094
CDAYS = np.r_[0, np.diff(IDX.values).astype('timedelta64[D]').astype(int)]
ew = pd.read_parquet(AG / 'momentum_family' / 'long_nav.parquet')['EW']
EW = ew.reindex(ew.index.union(IDX)).ffill().reindex(IDX)
EWV = EW.to_numpy()
S0 = int(np.searchsorted(IDX, pd.Timestamp('2004-01-02'))); TEND = NT - 1
ym = IDX.year * 12 + IDX.month
FIRST = np.r_[True, ym[1:] != ym[:-1]]           # 每月第一個交易日

# ---------- 回落訊號狀態機（全歷史，從 2003-07 起） ----------
TH = [0.10, 0.20, 0.30]
cm = np.maximum.accumulate(EV); DD = EV / cm - 1
sig = []                 # (signal_t, exec_t, k)
fired = [False] * 3
for t in range(NT):
    if DD[t] >= -1e-12: fired = [False] * 3
    for k, th in enumerate(TH):
        if not fired[k] and DD[t] <= -th:
            fired[k] = True; sig.append((t, t + 1, k))
EXEC = {}
for t, x, k in sig:
    if x < NT: EXEC.setdefault(x, []).append(k)
# 觸發時每個循環剩幾個門檻：依序觸發，k=0→3 個剩、k=1→2、k=2→1（同一循環內若跳過門檻也以「未觸發數」計）
def cycle_remaining():
    rem = {}; fired = [False] * 3
    for t in range(NT):
        if DD[t] >= -1e-12: fired = [False] * 3
        for k, th in enumerate(TH):
            if not fired[k] and DD[t] <= -th:
                rem[(t + 1, k)] = 3 - sum(fired); fired[k] = True
    return rem
REM = cycle_remaining()

# ---------- KD（月收盤近似） ----------
mclose = E.groupby([IDX.year, IDX.month]).last()
lo = mclose.rolling(9, min_periods=1).min(); hi = mclose.rolling(9, min_periods=1).max()
rsv = ((mclose - lo) / (hi - lo) * 100).fillna(50)
K = []; kp = 50.0
for v in rsv: kp = 2 / 3 * kp + 1 / 3 * v; K.append(kp)
K = pd.Series(K, index=mclose.index)
Kprev = K.shift(1)                                # 上月底 K
def kd_mult(t):
    k = Kprev.get((IDX[t].year, IDX[t].month), np.nan)
    if not np.isfinite(k): return 1.0
    return 2.0 if k < 20 else (0.5 if k > 80 else 1.0)

# ---------- 模擬 ----------
def sim(strat, cost='slip0.001', start=S0, end=TEND, dep=0.5, upfront=0.0, rate=0.01, lump_total=None):
    buy, sell = COSTS[cost]
    u = 0.0; c = 0.0; pend = 0.0; pend_trig = []
    V = np.zeros(end - start + 1); CASH = np.zeros_like(V); FLOW = np.zeros_like(V); UNITS = np.zeros_like(V)
    log = []
    for i, t in enumerate(range(start, end + 1)):
        if i > 0: c *= (1 + rate) ** (CDAYS[t] / 365.25)
        f = 0.0
        if i == 0:
            if upfront: c += upfront; f += upfront
            if strat == 'LUMP': pend += lump_total; f += lump_total
        if FIRST[t] and strat != 'LUMP':
            if strat == 'ADD': pend += 1.0; c += dep; f += 1.0 + dep
            elif strat == 'DCA_same': pend += 1.0 + dep; f += 1.0 + dep
            elif strat == 'DCA_1': pend += 1.0; c += dep; f += 1.0 + dep
            elif strat == 'KD': m = kd_mult(t); pend += m; f += m
            elif strat == 'KD_same':
                c += 1.5; f += 1.5; m = min(kd_mult(t), c); c -= m; pend += m
        if strat == 'ADD' and t in EXEC:
            pend_trig += EXEC[t]
        if not LOCK[t]:
            if pend_trig:
                for k in sorted(pend_trig):
                    rem = REM.get((t, k), None) or max(1, 3 - k)
                    amt = c / rem; c -= amt; pend += amt; log.append((str(IDX[t].date()), k, round(amt, 4)))
                pend_trig = []
            if pend > 0: u += pend * (1 - buy) / EV[t]; pend = 0.0
        V[i] = u * EV[t] + c + pend; CASH[i] = c + pend; FLOW[i] = f; UNITS[i] = u
    d = pd.DataFrame({'V': V, 'cash': CASH, 'flow': FLOW, 'units': UNITS}, index=IDX[start:end + 1])
    d['liq'] = d['units'] * EV[start:end + 1] * (1 - sell) + d['cash']
    return d, log

def xirr(dates, flows):
    t = np.array([(d - dates[0]).days / 365.25 for d in dates]); f = np.array(flows)
    npv = lambda r: (f / (1 + r) ** t).sum()
    lo, hi = -0.99, 5.0
    if npv(lo) * npv(hi) > 0: return float('nan')
    for _ in range(200):
        mid = (lo + hi) / 2
        if npv(lo) * npv(mid) <= 0: hi = mid
        else: lo = mid
    return round((lo + hi) / 2, 4)

def win_idx(a, b):
    i0 = int(np.searchsorted(IDX, pd.Timestamp(a))); i1 = int(np.searchsorted(IDX, pd.Timestamp(b), side='right')) - 1
    return max(i0, S0), min(i1, TEND)

def win_stats(d, i0, i1):
    """連續模擬中視窗 [i0,i1] 的統計。"""
    s = d.loc[IDX[i0]:IDX[i1]]
    prev = d.loc[:IDX[i0]].iloc[:-1]
    v0 = float(prev['V'].iloc[-1]) if len(prev) else 0.0
    dates = []; flows = []
    if v0 > 0: dates.append(IDX[i0 - 1]); flows.append(-v0)
    for dt, f in s['flow'].items():
        if f: dates.append(dt); flows.append(-f)
    dates.append(IDX[i1]); flows.append(float(s['liq'].iloc[-1]))
    o = {'XIRR': xirr(dates, flows), 'V_start': round(v0, 2), 'contrib': round(float(s['flow'].sum()), 2),
         'V_end': round(float(s['V'].iloc[-1]), 2), 'liq_end': round(float(s['liq'].iloc[-1]), 2)}
    o['multiple_end_over_in'] = round(o['liq_end'] / (v0 + o['contrib']), 3) if v0 + o['contrib'] > 0 else None
    o['maxDD_account'] = round(float((s['V'] / s['V'].cummax() - 1).min()), 4)
    full_v = d['V']; full_f = d['flow']
    r = ((full_v - full_f) / full_v.shift(1) - 1).loc[IDX[i0]:IDX[i1]].replace([np.inf, -np.inf], np.nan).dropna()
    nav = (1 + r).cumprod(); o['maxDD_unitNAV'] = round(float((nav / nav.cummax() - 1).min()), 4) if len(nav) else None
    yrs = (IDX[i1] - IDX[i0]).days / 365.25
    o['TWR_CAGR'] = round(float(nav.iloc[-1] ** (1 / yrs) - 1), 4) if len(nav) and yrs > 0 else None
    o['idle_cash_ratio'] = round(float((s['cash'] / s['V'].replace(0, np.nan)).mean()), 4)
    return o

def lump(i0, i1, cost, dep=0.5):
    nmon = int(FIRST[i0:i1 + 1].sum()) + (0 if FIRST[i0] else 0)
    tot = (1.0 + dep) * max(nmon, 1)
    d, _ = sim('LUMP', cost, i0, i1, lump_total=tot)
    o = {'XIRR': xirr([IDX[i0], IDX[i1]], [-tot, float(d['liq'].iloc[-1])]), 'contrib': round(tot, 2), 'liq_end': round(float(d['liq'].iloc[-1]), 2),
         'multiple_end_over_in': round(float(d['liq'].iloc[-1]) / tot, 3), 'maxDD_account': round(float((d['V'] / d['V'].cummax() - 1).min()), 4), 'idle_cash_ratio': 0.0}
    return o

WIN = dict(PR['windows'])
# 空頭子段
BEAR = {}
for y in (2008, 2015, 2020, 2022):
    m = (IDX.year == y); tr = int(np.flatnonzero(m)[np.argmin(EV[m])])
    pk_val = cm[tr]; pk = int(np.flatnonzero(EV[:tr + 1] >= pk_val - 1e-9)[-1])
    rec = np.flatnonzero((np.arange(NT) > tr) & (EV >= pk_val)); rc = int(rec[0]) if len(rec) else TEND
    BEAR[f'bear_{y}'] = [str(IDX[pk].date()), str(IDX[rc].date())]
    BEAR[f'bear_{y}_trough'] = str(IDX[tr].date()); BEAR[f'bear_{y}_DD'] = round(float(DD[tr]), 4)
    WIN[f'cal_{y}'] = [f'{y}-01-01', f'{y}-12-31']
for k, v in BEAR.items():
    if isinstance(v, list): WIN[k] = v

STRATS = ['ADD', 'DCA_same', 'DCA_1', 'KD', 'KD_same']
res = {'bear_segments': BEAR, 'n_signals_total': len(sig),
       'signals': [{'signal': str(IDX[t].date()), 'exec': str(IDX[min(x, NT - 1)].date()), 'threshold': TH[k]} for t, x, k in sig if t >= S0 - 1]}
runs = {}
for cost in COSTS:
    for s in STRATS:
        runs[(s, cost, 'main')] = sim(s, cost)
for nm, kw in (('S1_upfront12', dict(dep=0.0, upfront=12.0)), ('S2_monthly1', dict(dep=1.0)), ('S3_cash0', dict(rate=0.0))):
    for s in ('ADD', 'DCA_same', 'DCA_1'):
        runs[(s, 'slip0.001', nm)] = sim(s, 'slip0.001', **kw)
# S1 的同金額對照：起始 12 單位立即買進（用 LUMP 機制疊加 DCA 1）
def sim_s1_same(cost='slip0.001'):
    buy, sell = COSTS[cost]; d1, _ = sim('DCA_same', cost, dep=0.0)
    u0 = 12 * (1 - buy) / EV[S0]; ex = u0 * EV[S0:TEND + 1]
    d = d1.copy(); d['V'] += ex; d['units'] += u0; d.iloc[0, d.columns.get_loc('flow')] += 12; d['liq'] = d['units'] * EV[S0:TEND + 1] * (1 - sell) + d['cash']
    return d, []
runs[('DCA_same', 'slip0.001', 'S1_upfront12')] = sim_s1_same()

out = {}
for (s, cost, spec), (d, lg) in runs.items():
    key = f'{s}|{cost}|{spec}'
    out[key] = {w: win_stats(d, *win_idx(a, b)) for w, (a, b) in WIN.items()}
    if s == 'ADD': out[key]['_trades'] = lg
for cost in COSTS:
    out[f'LUMP|{cost}|main'] = {w: lump(*win_idx(a, b), cost) for w, (a, b) in WIN.items()}
# 全新起跑（主成本）
fresh = {}
for w in ('select_price', 'valid_price', 'select_2021', 'valid_2024', 'y2024_2025'):
    i0, i1 = win_idx(*WIN[w])
    for s in STRATS:
        d, _ = sim(s, 'slip0.001', i0, i1)
        fresh[f'{s}|{w}'] = win_stats(d, i0, i1)
    for s, kw in (('ADD_S1', dict(dep=0.0, upfront=12.0)),):
        d, _ = sim('ADD', 'slip0.001', i0, i1, **kw); fresh[f'{s}|{w}'] = win_stats(d, i0, i1)
# 逐年 XIRR（連續模擬）
yearly = {}
for s in STRATS:
    d, _ = runs[(s, 'slip0.001', 'main')]
    yearly[s] = {int(y): win_stats(d, *win_idx(f'{y}-01-01', f'{y}-12-31'))['XIRR'] for y in range(2004, 2027)}
yearly['LUMP_0050'] = {int(y): lump(*win_idx(f'{y}-01-01', f'{y}-12-31'), 'slip0.001')['XIRR'] for y in range(2004, 2027)}
res['continuous'] = out; res['fresh_start'] = fresh; res['yearly_XIRR'] = yearly

# ---------- 事件檢定 ----------
def events(cost):
    buy, sell = COSTS[cost]; rows = []
    for t, x, k in sig:
        if t < S0 - 1 or x >= NT: continue
        while x < NT and LOCK[x]: x += 1
        for Hd in (63, 126, 252, 504):
            if x + Hd > TEND: continue
            r = EV[x + Hd] * (1 - sell) / (EV[x] * (1 + buy)) - 1
            re = EWV[x + Hd] / EWV[x] - 1 if np.isfinite(EWV[x]) and np.isfinite(EWV[x + Hd]) else np.nan
            rows.append({'entry': IDX[x], 'k': TH[k], 'H': Hd, 'ret': r, 'ew': re, 'ab_ew': r - re})
    return pd.DataFrame(rows)
def uncond(i0, i1, Hd, cost):
    buy, sell = COSTS[cost]
    xs = np.arange(i0, min(i1, TEND - Hd) + 1)
    return float(np.mean(EV[xs + Hd] * (1 - sell) / (EV[xs] * (1 + buy)) - 1)) if len(xs) else np.nan
ev_out = {}
for cost in ('slip0.001', 'stress'):
    df = events(cost)
    for w in ('full_2004', 'select_price', 'valid_price', 'select_2021', 'valid_2024', 'y2024_2025'):
        a, b = WIN[w]; i0, i1 = win_idx(a, b)
        for Hd in (63, 126, 252, 504):
            g = df[(df.H == Hd) & (df.entry >= IDX[i0]) & (df.entry <= IDX[i1])].copy()
            if not len(g): ev_out[f'{cost}|{w}|H{Hd}'] = {'n_events': 0}; continue
            un = uncond(i0, i1, Hd, cost); g['ab_un'] = g['ret'] - un
            gm = g.groupby(g.entry.dt.to_period('M'))[['ret', 'ab_ew', 'ab_un']].mean()
            ge = gm['ab_ew'].dropna()
            ev_out[f'{cost}|{w}|H{Hd}'] = {'n_events': int(len(g)), 'n_months': int(len(gm)), 'n_months_ew': int(len(ge)),
                'mean_ret': round(float(g.ret.mean()), 4), 'mean_ab_ew': round(float(ge.mean()), 4) if len(ge) else None, 't_ab_ew_month': tstat(ge),
                'uncond_0050': round(un, 4), 'mean_ab_0050uncond': round(float(gm.ab_un.mean()), 4), 't_ab_0050uncond_month': tstat(gm.ab_un),
                'by_threshold': {str(k): {'n': int(len(gg)), 'ret': round(float(gg.ret.mean()), 4), 'ab_ew': round(float(gg.ab_ew.mean()), 4) if gg.ab_ew.notna().any() else None} for k, gg in g.groupby('k')}}
    if cost == 'slip0.001':
        res['event_list_H252'] = [{'entry': str(r.entry.date()), 'k': r.k, 'ret': round(r.ret, 4), 'ab_ew': None if not np.isfinite(r.ab_ew) else round(r.ab_ew, 4)} for r in df[df.H == 252].itertuples()]
res['events'] = ev_out

# ---------- 判定 ----------
ver = {}
for w in ('select_price', 'valid_price', 'select_2021', 'valid_2024', 'y2024_2025'):
    e = ev_out.get(f'slip0.001|{w}|H252', {})
    beat_ew = bool(e.get('n_months_ew', 0) >= 3 and (e.get('mean_ab_ew') or -1) > 0 and np.isfinite(e.get('t_ab_ew_month', np.nan)) and e['t_ab_ew_month'] >= 2)
    xa = out['ADD|slip0.001|main'][w]['XIRR']; xd = out['DCA_same|slip0.001|main'][w]['XIRR']
    ver[w] = {'beat_EW_event_t>=2': beat_ew, 'ADD_XIRR': xa, 'DCA_same_XIRR': xd, 'not_lose_0050': bool(xa >= xd), 'pass': bool(beat_ew and xa >= xd),
              't_ab_ew': e.get('t_ab_ew_month'), 'mean_ab_ew': e.get('mean_ab_ew')}
np_ = sum(ver[w]['pass'] for w in ('select_price', 'valid_price'))
ver['verdict'] = {2: '可用', 1: '待觀察', 0: '不可用'}[np_]
res['verdict'] = ver
RJ['results'] = res
(H / 'results.json').write_text(json.dumps(RJ, ensure_ascii=False, indent=1, default=str))
print(json.dumps(ver, ensure_ascii=False, indent=1))
