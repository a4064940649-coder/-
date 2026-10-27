#!/usr/bin/env python3
"""Bensdorp 週輪動 長歷史可成交回測。執行：python3 -I run.py
規則見 prereg.py（PREREG 已先寫入 results.json，本檔只附加結果）。
成交、成本、漲停判定、60 日均值缺漏修補沿用 momentum_family/mlib.py → exec_rebuild/exlib.py；主策略對手用 revenue_longhistory/lhlib.py（只讀取、不修改）。"""
import json, math, sys, time
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'momentum_family'))
sys.path.insert(0, str(HERE.parent / 'revenue_longhistory'))
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
Cg = L.C.to_numpy()[gi]
V60g = M.V60.to_numpy()[gi]
common = np.array([not c.startswith('91') for c in COLS])
lastvalid = np.array([np.where(np.isfinite(ADJg[:, j]))[0].max() if np.isfinite(ADJg[:, j]).any() else -1 for j in range(N)])

# ---------------- 指標 ----------------
ADF = pd.DataFrame(ADJg)
AF5 = ADF.ffill(limit=5).to_numpy()
with np.errstate(invalid='ignore', divide='ignore'):
    ROC = np.full_like(ADJg, np.nan); ROC[200:] = ADJg[200:] / AF5[:-200] - 1
dlt = ADF.ffill(limit=5).diff().where(ADF.notna())
up = dlt.clip(lower=0); dn = (-dlt).clip(lower=0)
ag = up.ewm(alpha=1 / 3, adjust=False, ignore_na=True, min_periods=20).mean()
al = dn.ewm(alpha=1 / 3, adjust=False, ignore_na=True, min_periods=20).mean()
with np.errstate(invalid='ignore', divide='ignore'):
    RSI = (100 - 100 / (1 + ag / al)).to_numpy()
RSI = np.where(np.isfinite(ADJg), RSI, np.nan)
RSI = np.where((al.to_numpy() == 0) & (ag.to_numpy() > 0), 100.0, RSI)

tx = pd.read_csv('/home/user/jonatw/finlab-txd/data/raw/taiex_twii.csv', parse_dates=['date']).set_index('date')['close'].sort_index()
tx = tx[tx.index.dayofweek != 6]
sma = tx.rolling(200, min_periods=200).mean()
miss_idx = int((~G.isin(tx.index)).sum())
REG = (tx > 0.98 * sma).reindex(G).ffill().fillna(False).to_numpy().astype(bool)
REGRAW = (tx > sma).reindex(G).ffill().fillna(False).to_numpy().astype(bool)

# ---------------- 週線與股池 ----------------
wk = G.to_period('W-SUN')
WE = np.array([np.flatnonzero(wk == p)[-1] for p in pd.unique(wk)])
WE = WE[WE + 1 < T]
LIQ = (V60g >= 1e7) & (Cg >= 10) & common[None, :] & np.isfinite(ADJg)
POOL = {}; ORDER = {}
for t in WE:
    v = np.where(LIQ[t], V60g[t], -np.inf)
    nl = int(LIQ[t].sum())
    top = np.argsort(-v)[:min(300, nl)]
    top = top[np.isfinite(ROC[t, top])]
    if len(top) < 50: continue
    POOL[int(t)] = top
    ORDER[int(t)] = top[np.argsort(-ROC[t, top], kind='stable')]
SIGW = np.array(sorted(POOL))
res['setup_info'] = {'n_weeks_with_signal': int(len(SIGW)), 'first_signal': str(G[SIGW[0]].date()), 'last_signal': str(G[SIGW[-1]].date()),
                     'index_days_missing_on_panel_calendar': miss_idx,
                     'regime_on_share_weeks_2005_2026': round(float(REG[SIGW][G[SIGW] >= '2005-01-01'].mean()), 3),
                     'regime_on_share_by_year': {int(y): round(float(REG[SIGW][G[SIGW].year == y].mean()), 2) for y in sorted(set(G[SIGW].year))},
                     'mean_pool_size': round(float(np.mean([len(POOL[t]) for t in SIGW])), 1),
                     'mean_share_rsi_lt50_in_pool': round(float(np.mean([(RSI[t, POOL[t]] < 50).mean() for t in SIGW])), 3),
                     'mean_share_rsi_lt50_in_top10': round(float(np.mean([(RSI[t, ORDER[t][:10]] < 50).mean() for t in SIGW])), 3)}
print('setup', round(time.time() - T0), res['setup_info'], flush=True)

EXEC = {int(t + 1): int(t) for t in SIGW}

# ---------------- 模擬 ----------------
def sim(version, buy, sell, nslot=10, w=0.10):
    cash = 1.0; P = {}; pend = set(); trades = []
    t_start = int(SIGW[0])
    nav = np.empty(T - t_start); npos = np.empty(T - t_start); inv = np.empty(T - t_start)
    traded = np.zeros(T - t_start)
    cnt = {'buys': 0, 'sells_rank': 0, 'sells_regime': 0, 'delist': 0, 'skip_gap': 0, 'skip_no_open': 0, 'empty_slot_weeks': 0, 'cash_slots_sum': 0}
    def do_sell(j, t, px, why):
        nonlocal cash
        p = P.pop(j); cash += p['u'] * px * (1 - sell)
        trades.append((p['e'], t, j, p['p0'], px, why)); pend.discard(j)
        return p['u'] * px
    for t in range(t_start, T):
        k = t - t_start; tv = 0.0
        for j in list(P):
            if t > lastvalid[j]:
                tv += do_sell(j, t, ACg[lastvalid[j], j], 'delist'); cnt['delist'] += 1
        for j in list(pend):
            if j in P and OPOK[t, j]:
                tv += do_sell(j, t, AOg[t, j], P[j]['pwhy'])
        if t in EXEC:
            s = EXEC[t]; order = ORDER[s]; held = set(P)
            reg = REG[s]
            if version == 'A':
                top = list(order[:10]); topset = set(top)
                sells = {j: 'rank' for j in held if j not in topset}
                cands = [j for j in top if RSI[s, j] < 50] if reg else []
                ext = False
            else:
                if not reg:
                    sells = {j: 'regime' for j in held}; cands = []
                else:
                    elig = (RSI[s, order] < 50) | (np.isin(order, list(held)) if version == 'B' else False)
                    lb = list(order[elig]); topset = set(lb[:10])
                    sells = {j: 'rank' for j in held if j not in topset}
                    cands = lb
                ext = True
            navo = cash + sum(p['u'] * AOg[t, j] for j, p in P.items())
            for j, why in sells.items():
                if j in pend: continue
                if OPOK[t, j]:
                    tv += do_sell(j, t, AOg[t, j], why)
                else:
                    P[j]['pwhy'] = why; pend.add(j)
                cnt['sells_rank' if why == 'rank' else 'sells_regime'] += 1
            slots = nslot - len(P)
            unit = navo * w; limit = 10; nb = 0
            for ii, j in enumerate(cands):
                if slots <= 0: break
                if ii >= limit: break
                if j in P: continue
                if not OPOK[t, j]:
                    cnt['skip_no_open'] += 1
                    if ext: limit += 1
                    continue
                if GAPg[t, j]:
                    cnt['skip_gap'] += 1
                    if ext: limit += 1
                    continue
                if cash < 0.25 * unit: break
                amt = min(unit, cash); px = AOg[t, j]
                P[j] = {'u': amt / (1 + buy) / px, 'p0': px, 'e': t}
                cash -= amt; cnt['buys'] += 1; slots -= 1; tv += amt; nb += 1
            if reg and slots > 0:
                cnt['empty_slot_weeks'] += 1; cnt['cash_slots_sum'] += slots
        val = sum(p['u'] * ACg[t, j] for j, p in P.items())
        nav[k] = cash + val; npos[k] = len(P); inv[k] = val / (cash + val); traded[k] = tv / (cash + val)
    for j, p in P.items(): trades.append((p['e'], T - 1, j, p['p0'], ACg[T - 1, j], 'open_at_end'))
    ix = G[t_start:]
    tr = pd.DataFrame(trades, columns=['e', 'x', 'j', 'p0', 'p1', 'why'])
    return pd.Series(nav, index=ix), pd.Series(npos, index=ix), pd.Series(inv, index=ix), pd.Series(traded, index=ix), cnt, tr

# ---------------- 對手 ----------------
ev_ew = {}
for s in SIGW:
    pl = POOL[s][OPOK[s + 1, POOL[s]]]
    if len(pl) >= 50: ev_ew[int(gi[s + 1])] = ('open', pl)
t0f = min(ev_ew); ewf, ew_tov = L.sim_portfolio(ev_ew, 0.0, 0.0, t0f, len(IDX) - 1)
EW = pd.concat([pd.Series([1.0], index=[IDX[t0f - 1]]), ewf]).reindex(G).ffill().dropna()
# 月頻同池等權（參考）
_s = pd.Series(G, index=G)
MD = pd.DatetimeIndex([g[g.dt.day >= 11].iloc[0] for _, g in _s.groupby([G.year, G.month]) if (g.dt.day >= 11).any()])
gpos = {d: i for i, d in enumerate(G)}
ev_ewm = {}
for d in MD:
    i = gpos[d]
    if i + 1 >= T or i < SIGW[0]: continue
    v = np.where(LIQ[i], V60g[i], -np.inf); top = np.argsort(-v)[:min(300, int(LIQ[i].sum()))]
    top = top[np.isfinite(ROC[i, top]) & OPOK[i + 1, top]]
    if len(top) >= 50: ev_ewm[int(gi[i + 1])] = ('open', top)
t0m = min(ev_ewm); ewm_, _ = L.sim_portfolio(ev_ewm, 0.0, 0.0, t0m, len(IDX) - 1)
EWM = pd.concat([pd.Series([1.0], index=[IDX[t0m - 1]]), ewm_]).reindex(G).ffill().dropna()
E50 = L.E50.reindex(G).ffill()
print('benchmarks', round(time.time() - T0), flush=True)

import lhlib as LH  # noqa: E402
S = LH.Setup(LH.yoy_long(), start='2005-01-01')
MAIN = {}
for n in (30, 10):
    evm, _ = S.selection(n)
    b, s_ = LH.cost_rates(0.001)
    t0 = min(evm); nv, _ = LH.sim_portfolio(evm, b, s_, t0, len(IDX) - 1)
    MAIN[f'MAIN_52w_rev_n{n}_0.001'] = pd.concat([pd.Series([1.0], index=[IDX[t0 - 1]]), nv]).reindex(G).ffill().dropna()
LN = pd.read_parquet(HERE.parent / 'momentum_family' / 'long_nav.parquet')
for c in ('P1_52wHigh_n10_0.001', 'P1_52wHigh_n30_0.001'):
    MAIN[c] = LN[c].reindex(G).ffill().dropna()
print('main', round(time.time() - T0), flush=True)

# ---------------- 統計 ----------------
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
KEYW = ('select_2005-01_2019-04', 'valid_2019-05_2026-10', 'y2024-01_2025-12')
ME = pd.DatetimeIndex(_s.groupby([G.year, G.month]).last().to_numpy())

def mrets(nav):
    nd = nav.reindex(ME); return (nd / nd.shift(1) - 1)

def inwin(ix, segs):
    m = np.zeros(len(ix), bool)
    for a, b in segs: m |= (ix >= pd.Timestamp(a)) & (ix <= pd.Timestamp(b))
    return m

def daily_stats(nav, segs, comps=None):
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
    pr = mrets(nav); pr = pr[inwin(pr.index, segs)].dropna(); o['n_months'] = int(len(pr))
    if comps is None: comps = {'0050': E50, 'EW': EW}
    for nm_, bn in comps.items():
        if bn is nav: continue
        bm = mrets(bn).reindex(pr.index); ex = (pr - bm).dropna()
        if len(ex) < 6: continue
        o[f'excess_vs_{nm_}_pct_m'] = round(float(ex.mean() * 100), 2); o[f't_vs_{nm_}_m'] = L.tstat(ex)
        o[f'hit_vs_{nm_}_m'] = round(float((ex > 0).mean()), 3)
        bd = bn.pct_change().reindex(rr.index); eb = (1 + bd.fillna(0)).cumprod()
        o[f'{nm_}_CAGR'] = round(float(eb.iloc[-1] ** (1 / span) - 1), 4); o[f'{nm_}_maxDD'] = round(float((eb / eb.cummax() - 1).min()), 4)
    return o

def yearly(nav):
    r = nav.pct_change().dropna(); return {int(y): round(float((1 + g).prod() - 1) * 100, 1) for y, g in r.groupby(r.index.year)}

navs = {'EW_pool_weekly': EW, 'EW_pool_monthly': EWM, '0050': E50, **MAIN}
port = {}; TRADES = {}
def run_variant(vn, version, costs):
    port[vn] = {}
    for cost in costs:
        b, s = (0.0, 0.0) if cost == 'gross' else L.cost_rates(cost)
        nav, npos, inv, trd, cnt, tr = sim(version, b, s)
        navs[f'{vn}_{cost}'] = nav
        tr['ret'] = tr.p1 / tr.p0 - 1; tr['hold'] = tr.x - tr.e
        nt = (1 + tr.ret) * (1 - s) / (1 + b) - 1
        yrs = (G[-1] - G[int(SIGW[0])]).days / 365.25
        o = {'counts': cnt, 'mean_open_positions': round(float(npos[npos.index >= '2005-01-01'].mean()), 2),
             'mean_invested_share': round(float(inv[inv.index >= '2005-01-01'].mean()), 3),
             'annual_turnover_one_way_x': round(float(trd.sum() / 2 / yrs), 2),
             'trades': {'n': int(len(tr)), 'win_rate_net': round(float((nt > 0).mean()), 3), 'mean_net_pct': round(float(nt.mean() * 100), 2),
                        'median_net_pct': round(float(nt.median() * 100), 2), 'mean_hold_days': round(float(tr.hold.mean()), 1),
                        'median_hold_days': float(tr.hold.median()),
                        'payoff_ratio': round(float(nt[nt > 0].mean() / -nt[nt <= 0].mean()), 2), 'exit_mix': tr.why.value_counts().to_dict()},
             'windows': {k: daily_stats(nav, sg) for k, sg in W.items()}, 'yearly_pct': yearly(nav[nav.index >= '2004-12-31'])}
        if cost == 0.001:
            o['open_positions_by_year'] = {int(y): round(float(g.mean()), 1) for y, g in npos.groupby(npos.index.year)}
            o['invested_by_year'] = {int(y): round(float(g.mean()), 3) for y, g in inv.groupby(inv.index.year)}
            o['vs_main'] = {k: {mk: daily_stats(nav, W[k], {mk: mv}) for mk, mv in MAIN.items()} for k in W}
            TRADES[vn] = tr
        port[vn][str(cost)] = o
        w_ = o['windows']
        print(vn, cost, {k: (w_[k]['CAGR'], w_[k]['0050_CAGR'], w_[k]['EW_CAGR'], w_[k]['t_vs_EW_m'], w_[k]['t_vs_0050_m'], w_[k]['maxDD']) for k in KEYW},
              'tov', o['annual_turnover_one_way_x'], 'pos', o['mean_open_positions'], round(time.time() - T0), flush=True)

run_variant('A', 'A', ('gross', 0.0, 0.001, 'stress'))
run_variant('B', 'B', ('gross', 0.0, 0.001, 'stress'))
run_variant('B_literal', 'Blit', (0.001,))
res['portfolio'] = port
res['benchmark_windows'] = {nm: {k: daily_stats(nv, sg) for k, sg in W.items()} for nm, nv in navs.items() if nm in ('EW_pool_weekly', 'EW_pool_monthly', '0050') or nm in MAIN}
res['benchmark_yearly'] = {nm: yearly(nv[nv.index >= '2004-12-31']) for nm, nv in navs.items() if nm in ('EW_pool_weekly', 'EW_pool_monthly', '0050') or nm in MAIN}
res['EW_weekly_turnover'] = round(float(ew_tov), 4)

# ---------------- 資料檢查 ----------------
FL = pd.read_parquet(L.P / 'flag.parquet').reindex(index=IDX, columns=COLS).to_numpy()[gi]
SRC = pd.read_parquet(L.P / 'src.parquet').reindex(index=IDX, columns=COLS).to_numpy()[gi]
chk = {}
for vn in ('A', 'B'):
    tr = TRADES[vn]
    has = np.array([bool(np.isin(FL[r.e:r.x + 1, r.j], [3, 4, 5]).any()) for r in tr.itertuples()])
    big = tr[(tr.ret > 1.0) | (tr.ret < -0.5)]
    tp = tr.assign(code=[COLS[j] for j in tr.j], e=G[tr.e.to_numpy()].astype(str), x=G[tr.x.to_numpy()].astype(str))
    sret = tr.ret.sort_values(ascending=False)
    chk[vn] = {'n': int(len(tr)), 'trades_with_flag345': int(has.sum()), 'entry_src3_share': round(float((SRC[tr.e, tr.j] == 3).mean()), 3),
               'n_ret_gt_100pct': int((tr.ret > 1.0).sum()), 'n_ret_lt_-50pct': int((tr.ret < -0.5).sum()),
               'max_ret': round(float(tr.ret.max()), 3), 'min_ret': round(float(tr.ret.min()), 3),
               'top_trades': tp.sort_values('ret', ascending=False)[['e', 'x', 'code', 'ret', 'why', 'hold']].round(3).head(8).to_dict('records'),
               'worst_trades': tp.sort_values('ret')[['e', 'x', 'code', 'ret', 'why', 'hold']].round(3).head(5).to_dict('records'),
               'share_sum_ret_top10_trades': round(float(sret.head(10).sum() / tr.ret.sum()), 3) if tr.ret.sum() > 0 else None,
               'flag345_examples': tp[has][['e', 'x', 'code', 'ret']].round(3).head(8).to_dict('records'),
               'extreme_examples': tp.loc[big.index, ['e', 'x', 'code', 'ret', 'why']].round(3).head(10).to_dict('records')}
res['data_checks'] = chk

# ---------------- 判定 ----------------
def passes(vn, k):
    w_ = port[vn]['0.001']['windows'][k]
    return {'CAGR': w_['CAGR'], '0050_CAGR': w_['0050_CAGR'], 'EW_CAGR': w_['EW_CAGR'], 't_vs_EW_m': w_['t_vs_EW_m'], 't_vs_0050_m': w_['t_vs_0050_m'],
            'pass': bool(w_['t_vs_EW_m'] >= 2 and w_['CAGR'] >= w_['0050_CAGR'])}
J = {}
for vn in ('A', 'B'):
    j_ = {k: passes(vn, k) for k in KEYW[:2]}
    npass = sum(v['pass'] for v in j_.values())
    j_['verdict'] = '可用' if npass == 2 else ('待觀察' if npass == 1 else '不可用'); J[vn] = j_
rank = {'不可用': 0, '待觀察': 1, '可用': 2}
best = max(('A', 'B'), key=lambda v: rank[J[v]['verdict']])
ov = J[best]['verdict']
if ov != '不可用' and J['A']['verdict'] != J['B']['verdict'] and min(J[best][k]['t_vs_EW_m'] for k in KEYW[:2] if J[best][k]['pass']) < 2.24:
    ov = {'可用': '待觀察', '待觀察': '不可用'}[ov]
J['overall'] = ov
res['judgement'] = J
print('judgement', json.dumps(J, ensure_ascii=False), flush=True)
(HERE / 'results.json').write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str), encoding='utf-8')
pd.DataFrame(navs).to_parquet(HERE / 'nav.parquet')
for vn in ('A', 'B'):
    TRADES[vn].assign(code=[COLS[j] for j in TRADES[vn].j], e=G[TRADES[vn].e.to_numpy()].astype(str),
                      x=G[TRADES[vn].x.to_numpy()].astype(str)).drop(columns=['j']).to_csv(HERE / f'trades_{vn}.csv.gz', index=False)
print('done', round(time.time() - T0))
