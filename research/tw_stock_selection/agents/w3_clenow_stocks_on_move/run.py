#!/usr/bin/env python3
"""Clenow《Stocks on the Move》長歷史可成交回測。執行：python3 -I run.py
規則見 prereg.py（PREREG 已先寫入 results.json，本檔只附加結果）。
成交價、成本、漲停判定、60 日均值缺漏修補沿用 momentum_family/mlib.py → exec_rebuild/exlib.py（只讀取、不修改）。
部位為 ATR 風險平價（非等權），故組合引擎自寫，但成本率 = L.cost_rates、成交價 = L.AO/L.AC、漲停 = M.GAP_UP。"""
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
Cg = L.C.to_numpy()[gi]
V60g = M.V60.to_numpy()[gi]
common = np.array([not c.startswith('91') for c in COLS])
lastvalid = np.array([np.where(np.isfinite(ADJg[:, j]))[0].max() if np.isfinite(ADJg[:, j]).any() else -1 for j in range(N)])

# ---------------- 指標（G 日曆） ----------------
LOGA = np.log(ADJg)
SMA100 = adjG.rolling(100, min_periods=100).mean().to_numpy()
ATR20 = adjG.diff().abs().rolling(20, min_periods=20).mean().to_numpy()
r1 = adjG.pct_change(fill_method=None)
GAP90 = (r1.abs() > 0.15).astype(float).rolling(90, min_periods=1).max().to_numpy() > 0
H52 = (adjG / adjG.rolling(252, min_periods=240).max()).to_numpy()
tx = M.TX.reindex(G)
TXv = tx.to_numpy(); TXS200 = tx.rolling(200, min_periods=200).mean().to_numpy()
BULL = TXv > TXS200

# ---------------- 週訊號日 ----------------
wk = pd.Series(G, index=G)
SIG = []
for _, g in wk.groupby([G.isocalendar().year.to_numpy(), G.isocalendar().week.to_numpy()]):
    w = g[g.dt.weekday >= 2]
    if len(w): SIG.append(w.iloc[0])
SIG = pd.DatetimeIndex(SIG); SIG = SIG[SIG >= pd.Timestamp('2004-07-01')]
gpos = {d: i for i, d in enumerate(G)}
SIGI = np.array([gpos[d] for d in SIG])
SIGI = SIGI[SIGI + 1 < T]
res['n_signal_weeks'] = int(len(SIGI))
res['signal_weekday_counts'] = {int(k): int(v) for k, v in pd.Series(G[SIGI].weekday).value_counts().sort_index().items()}

# 處置
DISP = L.in_disposal(G[SIGI]).to_numpy()

xc = np.arange(90) - 44.5; SXX = float((xc ** 2).sum())
def clenow_score(i):
    Y = LOGA[i - 89:i + 1]
    ok = np.isfinite(Y).all(0)
    Yc = Y - np.nanmean(Y, 0)
    with np.errstate(invalid='ignore', divide='ignore'):
        b = (xc[:, None] * Yc).sum(0) / SXX
        syy = (Yc ** 2).sum(0)
        r2 = b ** 2 * SXX / syy
        sc = (np.exp(b) ** 250 - 1) * r2
    sc[~ok] = np.nan
    return sc, b, r2

S = {}   # 每個訊號日：pool、scores
SC = {'clenow': np.full((len(SIGI), N), np.nan), 'h52': np.full((len(SIGI), N), np.nan), 'ret90': np.full((len(SIGI), N), np.nan)}
POOL = np.zeros((len(SIGI), N), bool)
R2 = np.full((len(SIGI), N), np.nan)
for k, i in enumerate(SIGI):
    with np.errstate(invalid='ignore'):
        base = (V60g[i] >= 1e7) & (Cg[i] >= 10) & common & np.isfinite(ADJg[i]) & ~DISP[k]
    v = np.where(base, V60g[i], -np.inf)
    nb = int(base.sum())
    top = np.argsort(-v)[:min(200, nb)]
    POOL[k, top] = True
    if i >= 89:
        sc, b, r2 = clenow_score(i)
        SC['clenow'][k] = sc; R2[k] = r2
        with np.errstate(invalid='ignore'):
            r90 = ADJg[i] / ADJg[i - 89] - 1
        r90[~np.isfinite(sc)] = np.nan
        SC['ret90'][k] = r90
        h = H52[i].copy(); h[~np.isfinite(sc)] = np.nan
        SC['h52'][k] = h
res['pool_size_by_year'] = {int(y): round(float(POOL[G[SIGI].year == y].sum(1).mean()), 0) for y in sorted(set(G[SIGI].year))}

def keep_sets(name):
    """回傳每個訊號 k：top20 & 通過排除 的布林（可持有／可買），以及排名用分數。"""
    sc = np.where(POOL, SC[name], np.nan)
    with np.errstate(invalid='ignore'):
        filt = np.isfinite(SMA100[SIGI]) & (ADJg[SIGI] >= SMA100[SIGI]) & ~GAP90[SIGI]
    ok_rank = np.isfinite(sc)
    cnt = ok_rank.sum(1)
    order = np.argsort(-np.nan_to_num(sc, nan=-np.inf), axis=1)
    rank = np.empty_like(order); rows = np.arange(len(SIGI))[:, None]
    rank[rows, order] = np.arange(N)[None, :] + 1
    top20 = ok_rank & (rank <= np.ceil(0.2 * cnt)[:, None])
    return top20 & filt, sc, top20, filt

KS = {nm: keep_sets(nm) for nm in SC}
res['diag_top20_counts'] = {nm: {'mean_top20': round(float(KS[nm][2].sum(1)[SIGI > 200].mean()), 1),
                                 'mean_top20_pass_filters': round(float(KS[nm][0].sum(1)[SIGI > 200].mean()), 1)} for nm in KS}
res['bull_share_signal_weeks'] = round(float(BULL[SIGI].mean()), 3)
print('setup', round(time.time() - T0), flush=True)

t_start = int(np.searchsorted(G, pd.Timestamp('2004-07-01')))
SIGPOS = {int(i): k for k, i in enumerate(SIGI)}

def sim(name, buy, sell, index_filter=True, sizing='atr', maxpos=30, log=False):
    ok, sc, _, _ = KS[name]
    cash = 1.0; P = {}; pend = set()
    nav = np.empty(T - t_start); npos = np.empty(T - t_start); inv = np.empty(T - t_start)
    cnt = {'new_buys': 0, 'sells_rule': 0, 'delist': 0, 'blocked_limitup': 0, 'cash_limited_stop': 0, 'weeks_bear_no_buy': 0, 'rebalances': 0}
    traded = 0.0; trades = []; entry = {}
    for t in range(t_start, T):
        # 1) 已下市
        for j in list(P):
            if t > lastvalid[j]:
                amt = P[j] * ACg[t, j]; cash += amt * (1 - sell); traded += amt; del P[j]; pend.discard(j); cnt['delist'] += 1
                if log: trades.append((entry.pop(j), t, j, 'delist'))
        k = SIGPOS.get(t - 1)
        if k is not None:
            for j in P:
                if not ok[k, j]: pend.add(j)
        # 2) 賣出
        for j in list(pend):
            if j in P and OPOK[t, j]:
                amt = P[j] * AOg[t, j]; cash += amt * (1 - sell); traded += amt; del P[j]; pend.discard(j); cnt['sells_rule'] += 1
                if log: trades.append((entry.pop(j), t, j, 'rule'))
            elif j not in P: pend.discard(j)
        if k is not None:
            i = t - 1
            navo = cash + sum(u * AOg[t, j] for j, u in P.items())
            def tgt_sh(j):
                if sizing == 'atr':
                    a = ATR20[i, j]
                    return navo * 0.001 / a if np.isfinite(a) and a > 0 else np.nan
                return navo / maxpos / AOg[t, j]
            # 3) 兩週調整（先減後加）
            if k % 2 == 0:
                cnt['rebalances'] += 1
                adds = []
                for j in list(P):
                    if j in pend or not OPOK[t, j]: continue
                    ts = tgt_sh(j)
                    if not np.isfinite(ts): continue
                    d = ts - P[j]
                    if d < 0:
                        amt = -d * AOg[t, j]; cash += amt * (1 - sell); traded += amt; P[j] = ts
                    elif d > 0: adds.append((j, d))
                for j, d in adds:
                    if GAPg[t, j]: cnt['blocked_limitup'] += 1; continue
                    need = d * AOg[t, j] * (1 + buy); use = min(need, cash)
                    if use <= 0: continue
                    P[j] += use / (1 + buy) / AOg[t, j]; cash -= use; traded += use / (1 + buy)
            # 4) 新買
            if index_filter and not BULL[i]:
                cnt['weeks_bear_no_buy'] += 1
            else:
                cand = np.flatnonzero(ok[k]); cand = cand[np.argsort(-sc[k, cand])]
                for j in cand:
                    if len(P) >= maxpos: break
                    if j in P or not OPOK[t, j]: continue
                    if GAPg[t, j]: cnt['blocked_limitup'] += 1; continue
                    ts = tgt_sh(j)
                    if not np.isfinite(ts) or ts <= 0: continue
                    need = ts * AOg[t, j] * (1 + buy)
                    if cash < 0.25 * need: cnt['cash_limited_stop'] += 1; break
                    use = min(need, cash)
                    P[j] = use / (1 + buy) / AOg[t, j]; cash -= use; traded += use / (1 + buy); cnt['new_buys'] += 1
                    if log: entry[j] = t
        val = sum(u * ACg[t, j] for j, u in P.items())
        kk = t - t_start
        nav[kk] = cash + val; npos[kk] = len(P); inv[kk] = val / (cash + val)
    ix = G[t_start:]
    navs = pd.Series(nav, index=ix)
    yrs = (ix[-1] - ix[0]).days / 365.25
    cnt['annual_turnover_one_side'] = round(float(traded / 2 / navs.mean() / yrs), 2)   # 粗估：(買+賣)/2 ÷ 平均 NAV ÷ 年
    if log:
        for j, e in entry.items(): trades.append((e, T - 1, j, 'open'))
    return navs, pd.Series(npos, index=ix), pd.Series(inv, index=ix), cnt, trades

# ---------------- 同池等權 ----------------
_s = pd.Series(G, index=G)
MD = pd.DatetimeIndex([g[g.dt.day >= 11].iloc[0] for _, g in _s.groupby([G.year, G.month]) if (g.dt.day >= 11).any()])
MD = MD[MD >= pd.Timestamp('2004-07-01')]
# 月度股池：同樣 200 檔規則（處置名單在月度日另算）
DISP_M = L.in_disposal(MD).to_numpy()
ev_ew = {}
for m, d in enumerate(MD[:-1]):
    i = gpos[d]
    if i + 1 >= T: continue
    with np.errstate(invalid='ignore'):
        base = (V60g[i] >= 1e7) & (Cg[i] >= 10) & common & np.isfinite(ADJg[i]) & ~DISP_M[m]
    v = np.where(base, V60g[i], -np.inf); top = np.argsort(-v)[:min(200, int(base.sum()))]
    pl = top[OPOK[i + 1, top]]
    if len(pl) >= 50: ev_ew[int(gi[i + 1])] = ('open', pl)
t0f = min(ev_ew); ewf, _ = L.sim_portfolio(ev_ew, 0.0, 0.0, t0f, len(IDX) - 1)
EW = pd.concat([pd.Series([1.0], index=[IDX[t0f - 1]]), ewf]).reindex(G).ffill().dropna()
ev_eww = {}
for k, i in enumerate(SIGI):
    pl = np.flatnonzero(POOL[k] & OPOK[i + 1])
    if len(pl) >= 50: ev_eww[int(gi[i + 1])] = ('open', pl)
t0w = min(ev_eww); eww, _ = L.sim_portfolio(ev_eww, 0.0, 0.0, t0w, len(IDX) - 1)
EWW = pd.concat([pd.Series([1.0], index=[IDX[t0w - 1]]), eww]).reindex(G).ffill().dropna()
E50 = L.E50.reindex(G).ffill()
MDX = MD.append(pd.DatetimeIndex([G[-1]])) if MD[-1] != G[-1] else MD
print('benchmarks', round(time.time() - T0), flush=True)

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
    for nm_, bn in (('0050', E50), ('EW', EW), ('EWweekly', EWW)):
        if bn is nav: continue
        ex = pr - mwin(bn).reindex(pr.index)
        o[f'excess_vs_{nm_}_pct_m'] = round(float(ex.mean() * 100), 2); o[f't_vs_{nm_}_m'] = L.tstat(ex)
        o[f'hit_vs_{nm_}_m'] = round(float((ex > 0).mean()), 3)
        bd = bn.pct_change().reindex(rr.index); eb = (1 + bd.fillna(0)).cumprod()
        o[f'{nm_}_CAGR'] = round(float(eb.iloc[-1] ** (1 / span) - 1), 4); o[f'{nm_}_maxDD'] = round(float((eb / eb.cummax() - 1).min()), 4)
    return o

def yearly(nav):
    r = nav.pct_change().dropna(); return {int(y): round(float((1 + g).prod() - 1) * 100, 1) for y, g in r.groupby(r.index.year)}

VARS = {'PRIMARY': dict(name='clenow'),
        'D1_score_52wHigh': dict(name='h52'),
        'D2_score_ret90': dict(name='ret90'),
        'D3_no_index_filter': dict(name='clenow', index_filter=False),
        'D4_equal_weight': dict(name='clenow', sizing='ew')}
navs = {'EW_pool': EW, 'EW_pool_weekly': EWW, '0050': E50}
port = {}; KEYW = ('select_2005-01_2019-04', 'valid_2019-05_2026-10', 'y2024-01_2025-12')
TRADES = None
for vn, kw in VARS.items():
    port[vn] = {}
    for cost in ('gross', 0.001, 'stress'):
        if vn != 'PRIMARY' and cost == 'stress': continue
        b, s = (0.0, 0.0) if cost == 'gross' else L.cost_rates(cost)
        nav, npos, inv, cnt, trades = sim(buy=b, sell=s, log=(vn == 'PRIMARY' and cost == 0.001), **kw)
        if trades: TRADES = trades
        navs[f'{vn}_{cost}'] = nav
        o = {'counts': cnt, 'mean_open_positions': round(float(npos[npos.index >= '2005-01-01'].mean()), 1),
             'mean_invested_share': round(float(inv[inv.index >= '2005-01-01'].mean()), 3),
             'windows': {k: daily_stats(nav, sg) for k, sg in W.items()}, 'yearly_pct': yearly(nav[nav.index >= '2004-12-31'])}
        if cost == 0.001:
            o['open_positions_by_year'] = {int(y): round(float(g.mean()), 1) for y, g in npos.groupby(npos.index.year)}
            o['invested_by_year'] = {int(y): round(float(g.mean()), 3) for y, g in inv.groupby(inv.index.year)}
        port[vn][str(cost)] = o
        w = o['windows']
        print(vn, cost, {k: (w[k]['CAGR'], w[k]['0050_CAGR'], w[k]['EW_CAGR'], w[k]['t_vs_EW_m'], w[k]['t_vs_0050_m'], w[k]['maxDD']) for k in KEYW},
              o['mean_open_positions'], o['mean_invested_share'], cnt['annual_turnover_one_side'], round(time.time() - T0), flush=True)
res['portfolio'] = port
res['EW_pool_windows'] = {k: daily_stats(EW, sg) for k, sg in W.items()}
res['EW_pool_yearly'] = yearly(EW[EW.index >= '2004-12-31'])
res['EW_pool_weekly_yearly'] = yearly(EWW[EWW.index >= '2004-12-31'])
res['0050_yearly'] = yearly(E50[E50.index >= '2004-12-31'])

# 相對 52 週高點／90 日報酬的增量（月窗差，滑價 0.1%）
inc = {}
pn = navs['PRIMARY_0.001']
for dn in ('D1_score_52wHigh', 'D2_score_ret90', 'D3_no_index_filter', 'D4_equal_weight'):
    dif = mwin(pn) - mwin(navs[f'{dn}_0.001']); inc[dn] = {}
    for k in KEYW + ('r21_select_2021-01_2023-12', 'r21_valid_2024-01_2026-10'):
        a, b_ = W[k][0]; x = dif[(dif.index >= a) & (dif.index <= b_)].dropna()
        inc[dn][k] = {'PRIMARY_minus_diag_pct_m': round(float(x.mean() * 100), 2), 't': L.tstat(x), 'n': int(len(x))}
res['increment_PRIMARY_vs_diag'] = inc

# ---------------- 訊號層 IC（週）：分數 vs d+1 開盤 → 下一個 d+1 開盤 ----------------
ic = {}
for nm in ('clenow', 'h52', 'ret90'):
    sc = np.where(POOL, SC[nm], np.nan); vals = []
    for k in range(len(SIGI) - 1):
        e0, e1 = SIGI[k] + 1, SIGI[k + 1] + 1
        fr = AOg[e1] / AOg[e0] - 1
        m = np.isfinite(sc[k]) & np.isfinite(fr)
        if m.sum() < 50: vals.append(np.nan); continue
        vals.append(pd.Series(sc[k][m]).rank().corr(pd.Series(fr[m]).rank()))
    s = pd.Series(vals, index=G[SIGI[:-1]])
    ic[nm] = {k: {'mean_IC': round(float(s[(s.index >= W[k][0][0]) & (s.index <= W[k][0][1])].mean()), 4),
                  't_weekly': L.tstat(s[(s.index >= W[k][0][0]) & (s.index <= W[k][0][1])])} for k in KEYW}
res['weekly_IC'] = ic
print('IC', ic, flush=True)

# ---------------- 資料檢查（PRIMARY 交易） ----------------
FL = pd.read_parquet(L.P / 'flag.parquet').reindex(index=IDX, columns=COLS).to_numpy()[gi]
SRCa = pd.read_parquet(L.P / 'src.parquet').reindex(index=IDX, columns=COLS).to_numpy()[gi]
tr = pd.DataFrame(TRADES, columns=['e', 'x', 'j', 'why'])
tr['ret'] = [ACg[x, j] / AOg[e, j] - 1 if w == 'delist' else (AOg[x, j] / AOg[e, j] - 1 if w == 'rule' else ACg[x, j] / AOg[e, j] - 1) for e, x, j, w in tr[['e', 'x', 'j', 'why']].itertuples(index=False)]
tr['hold'] = tr.x - tr.e
tr['flag345'] = [bool(np.isin(FL[e:x + 1, j], [3, 4, 5]).any()) for e, x, j in tr[['e', 'x', 'j']].itertuples(index=False)]
tr['d'] = G[tr.e]; tr['code'] = [COLS[j] for j in tr.j]
big = tr[(tr.ret > 1.0) | (tr.ret < -0.5)]
res['data_checks'] = {
    'n_trades': int(len(tr)), 'median_hold_days': float(tr.hold.median()), 'mean_hold_days': round(float(tr.hold.mean()), 1),
    'win_rate': round(float((tr.ret > 0).mean()), 3), 'mean_trade_ret_pct': round(float(tr.ret.mean() * 100), 2),
    'trades_with_flag345': int(tr.flag345.sum()), 'entry_src3_share': round(float((SRCa[tr.e, tr.j] == 3).mean()), 3),
    'n_ret_gt_100pct': int((tr.ret > 1.0).sum()), 'n_ret_lt_-50pct': int((tr.ret < -0.5).sum()),
    'extreme_examples': big.assign(d=big.d.astype(str))[['d', 'code', 'ret', 'why', 'hold']].round(3).head(15).to_dict('records'),
    'exit_reasons': tr.why.value_counts().to_dict(),
    'median_R2_top20': round(float(np.nanmedian(np.where(KS['clenow'][2], R2, np.nan))), 3),
    'median_weight_target_pct': round(float(np.nanmedian(np.where(KS['clenow'][0], 0.001 * ADJg[SIGI] / ATR20[SIGI], np.nan)) * 100), 2),
}
print('checks', res['data_checks'], flush=True)

# ---------------- 判定 ----------------
def passes(vn, k):
    w = port[vn]['0.001']['windows'][k]
    return {'CAGR': w['CAGR'], '0050_CAGR': w['0050_CAGR'], 'EW_CAGR': w['EW_CAGR'], 't_vs_EW_m': w['t_vs_EW_m'], 't_vs_0050_m': w['t_vs_0050_m'],
            'pass': bool(w['t_vs_EW_m'] >= 2 and w['CAGR'] >= w['0050_CAGR'])}
J = {k: passes('PRIMARY', k) for k in ('select_2005-01_2019-04', 'valid_2019-05_2026-10')}
npass = sum(v['pass'] for v in J.values())
J['verdict'] = '可用' if npass == 2 else ('待觀察' if npass == 1 else '不可用')
res['judgement'] = J
print('judgement', json.dumps(J, ensure_ascii=False), flush=True)
(HERE / 'results.json').write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str), encoding='utf-8')
pd.DataFrame(navs).to_parquet(HERE / 'nav.parquet')
tr.assign(d=tr.d.astype(str), exit=G[tr.x].astype(str)).drop(columns=['j', 'e', 'x']).to_csv(HERE / 'trades_PRIMARY.csv.gz', index=False)
print('done', round(time.time() - T0))
