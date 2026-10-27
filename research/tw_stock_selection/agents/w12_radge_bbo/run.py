#!/usr/bin/env python3
"""Nick Radge 布林突破（BBO）長歷史可成交回測。執行：python3 -I run.py
規則見 prereg.py（PREREG 已先寫入 results.json，本檔只附加結果）。
成交、成本、漲停判定、60 日均值缺漏修補沿用 momentum_family/mlib.py → exec_rebuild/exlib.py（只讀取、不修改）。
組合／事件骨架改寫自 w1_turtle_donchian/run.py。"""
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
Cg = L.C.to_numpy()[gi]; Hg = L.H.to_numpy()[gi]; RETg = L.RETN[gi]
thr = np.where(G < pd.Timestamp('2015-06-01'), 0.07, 0.10)[:, None] - L.LTOL
with np.errstate(invalid='ignore'):
    LOCKg = (RETg >= thr) & (Cg >= Hg * 0.999)            # 訊號日收盤鎖漲停
V60g = M.V60.to_numpy()[gi]
common = np.array([not c.startswith('91') for c in COLS])
LIQ = (V60g >= 1e7) & (Cg >= 10) & common[None, :] & np.isfinite(ADJg)

MID = adjG.rolling(100, min_periods=100).mean()
SD = adjG.rolling(100, min_periods=100).std(ddof=0)
UP = (MID + 3 * SD).to_numpy(); LO = (MID - 1 * SD).to_numpy()
ROC = (adjG / adjG.shift(100) - 1).to_numpy()
with np.errstate(invalid='ignore'):
    ENT_A = LIQ & (ADJg > UP)
    EXT = ADJg < LO
    prevA = np.vstack([np.full((1, N), np.nan), ADJg[:-1]]); prevU = np.vstack([np.full((1, N), np.nan), UP[:-1]])
    ENT_D = ENT_A & (prevA <= prevU)
# 大盤濾網（指數自身日曆上算 200 日均線，再對到面板日曆）
tx = pd.read_csv('/home/user/jonatw/finlab-txd/data/raw/taiex_twii.csv', parse_dates=['date']).set_index('date')['close'].sort_index()
FB = (tx > tx.rolling(200).mean()).reindex(G, method='ffill').fillna(False).to_numpy()
e50 = L.e50.sort_index()
FC = (e50 > e50.rolling(200).mean()).reindex(G, method='ffill').fillna(False).to_numpy()
ENTS = {'A_noFilter': ENT_A, 'B_taiex200': ENT_A & FB[:, None], 'diag_C_0050_200': ENT_A & FC[:, None], 'diag_D_cross_noFilter': ENT_D}
res['filter_on_share'] = {'taiex200_2005+': round(float(FB[G >= pd.Timestamp('2005-01-01')].mean()), 3),
                          '0050_200_2005+': round(float(FC[G >= pd.Timestamp('2005-01-01')].mean()), 3),
                          'taiex200_by_year': {int(y): round(float(FB[G.year == y].mean()), 2) for y in sorted(set(G.year))}}
lastvalid = np.array([np.where(np.isfinite(ADJg[:, j]))[0].max() if np.isfinite(ADJg[:, j]).any() else -1 for j in range(N)])
res['signal_density'] = {'mean_daily_signals_A_2005+': round(float(ENT_A[G >= pd.Timestamp('2005-01-01')].sum(1).mean()), 2),
                         'mean_daily_pool_2005+': round(float(LIQ[G >= pd.Timestamp('2005-01-01')].sum(1).mean()), 1),
                         'mean_daily_cross_signals_D_2005+': round(float(ENT_D[G >= pd.Timestamp('2005-01-01')].sum(1).mean()), 2)}
print('setup', round(time.time() - T0), res['signal_density'], flush=True)

def find_exit(j, e, EXc):
    lv = lastvalid[j]; start = e; t = None
    for w in (64, 512, 100000):
        a, b = start, min(lv + 1, e + w)
        if a >= b: break
        k = np.flatnonzero(EXc[j, a:b])
        if len(k): t = a + k[0]; break
        start = b
    if t is None:
        if lv < T - 1: return min(lv + 1, T - 1), ACg[lv, j], 'delist'
        return T - 1, ACg[T - 1, j], 'open_at_end'
    for x in range(t + 1, lv + 1):
        if OPOK[x, j]: return x, AOg[x, j], 'band'
    if t + 1 > T - 1: return T - 1, ACg[T - 1, j], 'open_at_end'
    return min(lv + 1, T - 1), ACg[lv, j], 'delist'

EXc = np.ascontiguousarray(EXT.T)

def build_events(ENT):
    rows = []; att = {'signal_days_not_held': 0, 'blocked_gap': 0, 'no_open': 0, 'signal_day_locked': 0, 'blocked_and_signal_locked': 0}
    for j in range(N):
        sigs = np.flatnonzero(ENT[:, j]); nxt = 0
        for s in sigs:
            if s < nxt: continue
            e = s + 1
            if e >= T: break
            att['signal_days_not_held'] += 1
            if LOCKg[s, j]: att['signal_day_locked'] += 1
            if not OPOK[e, j]: att['no_open'] += 1; continue
            if GAPg[e, j]:
                att['blocked_gap'] += 1; att['blocked_and_signal_locked'] += int(LOCKg[s, j]); continue
            p0 = AOg[e, j]
            x, p1, why = find_exit(j, e, EXc)
            rows.append((s, e, x, j, p0, p1, why, bool(LOCKg[s, j]), (ADJg[s, j] / UP[s, j] - 1)))
            nxt = x
    tr = pd.DataFrame(rows, columns=['s', 'e', 'x', 'j', 'p0', 'p1', 'why', 'sig_locked', 'over_upper'])
    tr['ret'] = tr.p1 / tr.p0 - 1; tr['hold'] = tr.x - tr.e
    ew = np.empty(len(tr)); pools = {}
    for i_, r in enumerate(tr.itertuples()):
        pl = pools.get(r.s)
        if pl is None: pl = np.flatnonzero(LIQ[r.s] & OPOK[r.s + 1]); pools[r.s] = pl
        px1 = ACg[r.x, pl] if r.why == 'open_at_end' else AOg[r.x, pl]
        ew[i_] = np.nanmean(px1 / AOg[r.e, pl] - 1)
    tr['ew'] = ew
    tr['d'] = G[tr.s.to_numpy()]; tr['em'] = G[tr.e.to_numpy()].to_period('M')
    return tr, att

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
                  't_month_cluster': L.tstat(g), 'mean_hold_days': round(float(x.hold.mean()), 1), 'median_hold_days': float(x.hold.median()),
                  'payoff_ratio': round(float(net[m][net[m] > 0].mean() / -net[m][net[m] <= 0].mean()), 2),
                  'exit_mix': {k_: round(float(v), 3) for k_, v in x.why.value_counts(normalize=True).items()}}
    return out

EVENTS, ATT = {}, {}
for nm in ('A_noFilter', 'B_taiex200'):
    EVENTS[nm], ATT[nm] = build_events(ENTS[nm])
    print('events', nm, len(EVENTS[nm]), ATT[nm], round(time.time() - T0), flush=True)
res['events'] = {nm: {str(c): event_stats(EVENTS[nm], c) for c in ('gross', 0.001, 'stress')} for nm in EVENTS}
b1, s1 = L.cost_rates(0.001)
res['events_by_year_slip0.1'] = {}
for nm, tr in EVENTS.items():
    net = (1 + tr.ret) * (1 - s1) / (1 + b1) - 1
    res['events_by_year_slip0.1'][nm] = {int(y): {'n': int(len(g)), 'win': round(float((net[g.index] > 0).mean()), 3),
                                               'mean_net_pct': round(float(net[g.index].mean() * 100), 2),
                                               'mean_abn_pct': round(float((net[g.index] - g.ew).mean() * 100), 2)} for y, g in tr.groupby(tr.d.dt.year)}
# 漲停重疊
lim = {}
for nm, a in ATT.items():
    tr = EVENTS[nm]; o = dict(a)
    o['share_signal_day_locked_limitup'] = round(a['signal_day_locked'] / a['signal_days_not_held'], 4)
    o['share_next_open_limitup_blocked'] = round(a['blocked_gap'] / a['signal_days_not_held'], 4)
    o['share_blocked_given_signal_locked'] = round(a['blocked_and_signal_locked'] / max(a['signal_day_locked'], 1), 4)
    o['entered_trades_with_signal_locked'] = int(tr.sig_locked.sum())
    for k in ('select_2005-01_2019-04', 'valid_2019-05_2026-10'):
        m = ((tr.d >= pd.Timestamp(W[k][0][0])) & (tr.d <= pd.Timestamp(W[k][0][1])))
        net = (1 + tr.ret) * (1 - s1) / (1 + b1) - 1
        o[f'{k}_abn_pct_sig_locked_vs_not'] = (round(float((net - tr.ew)[m & tr.sig_locked].mean() * 100), 2), round(float((net - tr.ew)[m & ~tr.sig_locked].mean() * 100), 2))
    lim[nm] = o
# 穿越型首日訊號（每段上軌突破的第一天）被開盤漲停擋掉的比例
D = ENT_D; sd_, jd_ = np.nonzero(D[:-1]); ok = OPOK[sd_ + 1, jd_]
lim['cross_first_day'] = {'n': int(len(sd_)), 'share_signal_day_locked': round(float(LOCKg[sd_, jd_].mean()), 4),
                          'share_next_open_limitup_blocked': round(float((GAPg[sd_ + 1, jd_] & ok).sum() / max(ok.sum(), 1)), 4),
                          'share_by_era_blocked': {era: round(float((GAPg[sd_ + 1, jd_] & ok)[msk].sum() / max(ok[msk].sum(), 1)), 4)
                                                   for era, msk in (('2005-2015.05_7pct', (G[sd_] >= pd.Timestamp('2005-01-01')) & (G[sd_] < pd.Timestamp('2015-06-01'))),
                                                                    ('2015.06+_10pct', G[sd_] >= pd.Timestamp('2015-06-01')))}}
res['limit_up_overlap'] = lim
for nm in EVENTS:
    print(nm, {k: (v['mean_abn_pct'], v['t_month_cluster'], v['n_trades']) for k, v in res['events'][nm]['0.001'].items()}, flush=True)
print('limit', json.dumps(lim, ensure_ascii=False), flush=True)

# ---------------- 組合（最多 20 檔） ----------------
t_start = int(np.searchsorted(G, pd.Timestamp('2004-07-01')))
def cands(ENT):
    lst = []
    for t in range(T):
        js = np.flatnonzero(ENT[t])
        if len(js):
            r = np.nan_to_num(ROC[t, js], nan=-9e9); js = js[np.argsort(-r, kind='stable')]
        lst.append(js)
    return lst
CAND = {nm: cands(E) for nm, E in ENTS.items()}

def sim(nm, buy, sell, maxpos=20, log=False):
    cash = 1.0; P = {}; trades = []
    nav = np.empty(T - t_start); npos = np.empty(T - t_start); inv = np.empty(T - t_start)
    cnt = {'entries': 0, 'band': 0, 'delist': 0, 'skip_no_slot': 0, 'skip_gap': 0, 'cand_considered': 0}
    for t in range(t_start, T):
        for j in list(P):
            p = P[j]; sold = None
            if t > lastvalid[j]: sold = ACg[t, j]; cnt['delist'] += 1
            elif p['px'] and OPOK[t, j]: sold = AOg[t, j]; cnt['band'] += 1
            if sold is not None:
                cash += p['u'] * sold * (1 - sell)
                if log: trades.append((p['e'], t, j, p['p0'], sold))
                del P[j]
        navo = cash + sum(p['u'] * AOg[t, j] for j, p in P.items()); unit = navo / maxpos
        if t > t_start:
            for j in CAND[nm][t - 1]:
                if j in P or not OPOK[t, j]: continue
                if len(P) >= maxpos or cash < 0.25 * unit: cnt['skip_no_slot'] += 1; continue
                cnt['cand_considered'] += 1
                if GAPg[t, j]: cnt['skip_gap'] += 1; continue
                amt = min(unit, cash); px = AOg[t, j]
                P[j] = {'u': amt / (1 + buy) / px, 'px': False, 'e': t, 'p0': px}; cash -= amt; cnt['entries'] += 1
        val = 0.0
        for j, p in P.items():
            val += p['u'] * ACg[t, j]
            if not p['px'] and EXT[t, j]: p['px'] = True
        k = t - t_start; nav[k] = cash + val; npos[k] = len(P); inv[k] = val / (cash + val)
    ix = G[t_start:]
    return pd.Series(nav, index=ix), pd.Series(npos, index=ix), pd.Series(inv, index=ix), cnt, trades

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
port = {}; TRADES = None
for vn in ENTS:
    port[vn] = {}
    for cost in ('gross', 0.001, 'stress'):
        b, s = (0.0, 0.0) if cost == 'gross' else L.cost_rates(cost)
        nav, npos, inv, cnt, trades = sim(vn, b, s, log=(vn == 'A_noFilter' and cost == 0.001))
        navs[f'{vn}_{cost}'] = nav
        o = {'counts': cnt, 'mean_open_positions': round(float(npos[npos.index >= '2005-01-01'].mean()), 1),
             'mean_invested_share': round(float(inv[inv.index >= '2005-01-01'].mean()), 3),
             'share_cand_blocked_gap': round(cnt['skip_gap'] / max(cnt['cand_considered'], 1), 4),
             'windows': {k: daily_stats(nav, sg) for k, sg in W.items()}, 'yearly_pct': yearly(nav[nav.index >= '2004-12-31'])}
        if cost == 0.001:
            o['open_positions_by_year'] = {int(y): round(float(g.mean()), 1) for y, g in npos.groupby(npos.index.year)}
            o['invested_by_year'] = {int(y): round(float(g.mean()), 3) for y, g in inv.groupby(inv.index.year)}
            o['entries_per_year'] = round(cnt['entries'] / ((G[-1] - G[t_start]).days / 365.25), 0)
            if trades:
                TRADES = pd.DataFrame(trades, columns=['e', 'x', 'j', 'p0', 'p1']); TRADES['ret'] = TRADES.p1 / TRADES.p0 - 1; TRADES['hold'] = TRADES.x - TRADES.e
                o['trade_summary'] = {'n': int(len(TRADES)), 'win_gross': round(float((TRADES.ret > 0).mean()), 3), 'mean_ret_gross_pct': round(float(TRADES.ret.mean() * 100), 2),
                                      'median_ret_gross_pct': round(float(TRADES.ret.median() * 100), 2), 'mean_hold_days': round(float(TRADES.hold.mean()), 1),
                                      'median_hold_days': float(TRADES.hold.median())}
        port[vn][str(cost)] = o
    w = port[vn]['0.001']['windows']
    print(vn, {k: (w[k]['CAGR'], w[k]['0050_CAGR'], w[k]['EW_CAGR'], w[k]['t_vs_EW_m'], w[k]['t_vs_0050_m'], w[k]['maxDD'])
               for k in ('select_2005-01_2019-04', 'valid_2019-05_2026-10', 'y2024-01_2025-12')}, round(time.time() - T0), flush=True)
res['portfolio'] = port
res['EW_pool_windows'] = {k: daily_stats(EW, sg) for k, sg in W.items()}
res['EW_pool_yearly'] = yearly(EW[EW.index >= '2004-12-31'])
res['0050_yearly'] = yearly(E50[E50.index >= '2004-12-31'])
res['pool_size_by_year'] = {int(y): round(float(np.mean([LIQ[gpos[d]].sum() for d in MD if d.year == y])), 0) for y in sorted(set(MD.year))}

# ---------------- 資料檢查 ----------------
FL = pd.read_parquet(L.P / 'flag.parquet').reindex(index=IDX, columns=COLS).to_numpy()[gi]
SRC = pd.read_parquet(L.P / 'src.parquet').reindex(index=IDX, columns=COLS).to_numpy()[gi]
chk = {}
for nm, tr in EVENTS.items():
    f345 = np.array([np.isin(FL[r.e:r.x + 1, r.j], [3, 4, 5]).any() for r in tr.itertuples()])
    big = tr[(tr.ret > 1.0) | (tr.ret < -0.5)]
    keep = ~f345 & (tr.ret.abs() <= 1.0).to_numpy()
    st = event_stats(tr[keep], 0.001)
    chk[nm] = {'trades_with_flag345': int(f345.sum()), 'n_trades': int(len(tr)), 'entry_src3_share': round(float((SRC[tr.e, tr.j] == 3).mean()), 3),
               'n_ret_gt_100pct': int((tr.ret > 1.0).sum()), 'n_ret_lt_-50pct': int((tr.ret < -0.5).sum()),
               'extreme_examples': big.assign(d=big.d.astype(str), code=[COLS[j] for j in big.j])[['d', 'code', 'ret', 'why', 'hold']].round(3).head(15).to_dict('records'),
               'abn_ex_flag345_and_extremes': {k: (st[k]['mean_abn_pct'], st[k]['t_month_cluster']) for k in ('select_2005-01_2019-04', 'valid_2019-05_2026-10') if k in st}}
if TRADES is not None:
    cs = TRADES.groupby('j').ret.apply(lambda r: (1 + r).prod() - 1).sort_values(ascending=False)
    chk['A_portfolio_top_contrib_codes'] = {COLS[j]: round(float(v), 2) for j, v in cs.head(10).items()}
    TRADES.assign(code=[COLS[j] for j in TRADES.j], ed=G[TRADES.e.to_numpy()].astype(str), xd=G[TRADES.x.to_numpy()].astype(str)).drop(columns=['j']).to_csv(HERE / 'trades_A_slip0.1.csv.gz', index=False)
res['data_checks'] = chk

# ---------------- 判定 ----------------
def passes(vn, k):
    w = port[vn]['0.001']['windows'][k]
    return {'CAGR': w['CAGR'], '0050_CAGR': w['0050_CAGR'], 'EW_CAGR': w['EW_CAGR'], 't_vs_EW_m': w['t_vs_EW_m'], 't_vs_0050_m': w['t_vs_0050_m'],
            'pass': bool(w['t_vs_EW_m'] >= 2 and w['CAGR'] >= w['0050_CAGR'])}
J = {}; RANK = {'不可用': 0, '待觀察': 1, '可用': 2}
for vn in ('A_noFilter', 'B_taiex200'):
    J[vn] = {k: passes(vn, k) for k in ('select_2005-01_2019-04', 'valid_2019-05_2026-10')}
    npass = sum(v['pass'] for v in J[vn].values())
    J[vn]['verdict'] = '可用' if npass == 2 else ('待觀察' if npass == 1 else '不可用')
v = J['A_noFilter']['verdict']
if v == '不可用' and J['B_taiex200']['verdict'] == '可用': v = '待觀察'
J['overall'] = v
res['judgement'] = J
print('judgement', json.dumps(J, ensure_ascii=False), flush=True)
(HERE / 'results.json').write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str), encoding='utf-8')
pd.DataFrame(navs).to_parquet(HERE / 'nav.parquet')
for nm, tr in EVENTS.items():
    tr.assign(code=[COLS[j] for j in tr.j], d=tr.d.astype(str), em=tr.em.astype(str)).drop(columns=['j']).to_csv(HERE / f'events_{nm}.csv.gz', index=False)
print('done', round(time.time() - T0))
