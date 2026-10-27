#!/usr/bin/env python3
"""Nick Radge 週末趨勢交易者 長歷史可成交回測。執行：python3 -I run.py
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
Cg = L.C.to_numpy()[gi]
V60g = M.V60.to_numpy()[gi]
common = np.array([not c.startswith('91') for c in COLS])
LIQ = (V60g >= 1e7) & (Cg >= 10) & common[None, :] & np.isfinite(ADJg)
lastvalid = np.array([np.where(np.isfinite(ADJg[:, j]))[0].max() if np.isfinite(ADJg[:, j]).any() else -1 for j in range(N)])

# ---------------- 週線 ----------------
wk = G.to_period('W-SUN')
WE = np.array([np.flatnonzero(wk == p)[-1] for p in pd.unique(wk)])      # 每週最後一個交易日（G 位置）
WEP = wk[WE]
AF = pd.DataFrame(ADJg).ffill(limit=4).to_numpy()
WC = pd.DataFrame(AF[WE])
HH20 = WC.rolling(20, min_periods=20).max().to_numpy()
ROC = (WC / WC.shift(20) - 1).to_numpy()
WCn = WC.to_numpy()
# 指數濾網：在指數自己的日曆上建週線
tx = pd.read_csv('/home/user/jonatw/finlab-txd/data/raw/taiex_twii.csv', parse_dates=['date']).set_index('date')['close'].sort_index()
txw = tx.groupby(tx.index.to_period('W-SUN')).last()
txup = (txw > txw.rolling(10, min_periods=10).mean())
IDXUP = txup.reindex(WEP).fillna(False).to_numpy().astype(bool)
assert txw.reindex(WEP).notna().all()
with np.errstate(invalid='ignore'):
    SIG = LIQ[WE] & np.isfinite(ADJg[WE]) & (WCn >= HH20 - 1e-12) & (ROC >= 0.30) & IDXUP[:, None]
WEpos = {int(t): k for k, t in enumerate(WE)}
res['weekly_info'] = {'n_weeks': int(len(WE)), 'idx_up_share_2005_2026': round(float(IDXUP[WEP >= pd.Period('2005-01-03', 'W-SUN')].mean()), 3),
                      'mean_signals_per_week_when_up': round(float(SIG.sum(1)[IDXUP].mean()), 2),
                      'signals_per_year': {int(y): int(SIG.sum(1)[WEP.year == y].sum()) for y in sorted(set(WEP.year))}}
print('setup', round(time.time() - T0), res['weekly_info']['idx_up_share_2005_2026'], res['weekly_info']['mean_signals_per_week_when_up'], flush=True)

# 訊號週 k → 候選（依 ROC 由高到低）；執行日 = WE[k]+1
CAND = {}
for k in range(len(WE)):
    js = np.flatnonzero(SIG[k])
    if len(js) and WE[k] + 1 < T:
        CAND[int(WE[k] + 1)] = js[np.argsort(-ROC[k, js])]

def exit_fill(j, t_sig):
    """週末 t_sig 停損觸發 → 下一個有開盤價的日子賣；回傳 (x, 價, why)。"""
    lv = lastvalid[j]
    for x in range(t_sig + 1, lv + 1):
        if OPOK[x, j]: return x, AOg[x, j], 'stop'
    if t_sig + 1 > T - 1: return T - 1, ACg[T - 1, j], 'open_at_end'
    return min(lv + 1, T - 1), ACg[lv, j], 'delist'

# ---------------- 事件（全部訊號，不設上限） ----------------
def build_events(fixed40=False):
    rows = []; pools = {}
    for j in range(N):
        ks = np.flatnonzero(SIG[:, j]); nxt = -1
        for k in ks:
            s = WE[k]
            if s < nxt: continue
            e = s + 1
            if e >= T: break
            if not OPOK[e, j] or GAPg[e, j]: continue
            p0 = AOg[e, j]; hh = -np.inf; stop = -np.inf; res_ = None
            for m in range(k + 1, len(WE)):
                t = WE[m]
                if t > lastvalid[j]:
                    lv = lastvalid[j]; res_ = (min(lv + 1, T - 1), ACg[lv, j], 'delist'); break
                c = ACg[t, j]; hh = max(hh, c)
                p = 0.4 if (IDXUP[m] or fixed40) else 0.1
                stop = max(stop, hh * (1 - p))
                if c < stop:
                    res_ = exit_fill(j, t); break
            if res_ is None: res_ = (T - 1, ACg[T - 1, j], 'open_at_end')
            x, p1, why = res_
            rows.append((s, e, x, j, p0, p1, why, ROC[k, j]))
            nxt = x
    tr = pd.DataFrame(rows, columns=['s', 'e', 'x', 'j', 'p0', 'p1', 'why', 'roc'])
    tr['ret'] = tr.p1 / tr.p0 - 1; tr['hold'] = tr.x - tr.e
    ew = np.empty(len(tr))
    for i_, r in enumerate(tr.itertuples()):
        pl = pools.get(r.s)
        if pl is None:
            pl = np.flatnonzero(LIQ[r.s] & OPOK[r.s + 1]); pools[r.s] = pl
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
                  'mean_abn_pct': round(float(abn[m].mean() * 100), 2), 'median_abn_pct': round(float(abn[m].median() * 100), 2),
                  'month_cluster_mean_abn_pct': round(float(g.mean() * 100), 2),
                  't_month_cluster': L.tstat(g), 'mean_hold_days': round(float(x.hold.mean()), 1), 'median_hold_days': float(x.hold.median()),
                  'payoff_ratio': round(float(net[m][net[m] > 0].mean() / -net[m][net[m] <= 0].mean()), 2),
                  'exit_mix': {k_: round(float(v), 3) for k_, v in x.why.value_counts(normalize=True).items()}}
    return out

EV = build_events()
print('events', len(EV), round(time.time() - T0), flush=True)
res['events_all_signals'] = {str(c): event_stats(EV, c) for c in ('gross', 0.001, 'stress')}
b1, s1 = L.cost_rates(0.001)
net1 = (1 + EV.ret) * (1 - s1) / (1 + b1) - 1
res['events_by_year_slip0.1'] = {int(y): {'n': int(len(g)), 'win': round(float((net1[g.index] > 0).mean()), 3),
                                          'mean_net_pct': round(float(net1[g.index].mean() * 100), 2),
                                          'mean_abn_pct': round(float((net1[g.index] - g.ew).mean() * 100), 2)}
                                 for y, g in EV.groupby(EV.d.dt.year)}
print({k: (v['mean_abn_pct'], v['t_month_cluster'], v['n_trades']) for k, v in res['events_all_signals']['0.001'].items()}, flush=True)

# ---------------- 組合 ----------------
t_start = int(np.searchsorted(G, pd.Timestamp('2004-07-01')))
ISWE = np.zeros(T, bool); ISWE[WE] = True
WEK = np.full(T, -1); WEK[WE] = np.arange(len(WE))

def sim(buy, sell, maxpos=20, w=0.05, fixed40=False, seed=None):
    rng = np.random.default_rng(seed) if seed is not None else None
    cash = 1.0; P = {}; trades = []
    nav = np.empty(T - t_start); npos = np.empty(T - t_start); inv = np.empty(T - t_start)
    cnt = {'entries': 0, 'stop': 0, 'delist': 0, 'skip_no_slot': 0, 'skip_gap': 0, 'skip_no_open': 0}
    for t in range(t_start, T):
        for j in list(P):
            p = P[j]
            if t > lastvalid[j]:
                px = ACg[lastvalid[j], j]; cash += p['u'] * px * (1 - sell); cnt['delist'] += 1
                trades.append((p['e'], t, j, p['p0'], px, 'delist')); del P[j]
            elif p['px'] and OPOK[t, j]:
                px = AOg[t, j]; cash += p['u'] * px * (1 - sell); cnt['stop'] += 1
                trades.append((p['e'], t, j, p['p0'], px, 'stop')); del P[j]
        if t > t_start and t in CAND:
            navo = cash + sum(p['u'] * AOg[t, j] for j, p in P.items())
            unit = navo * w
            cands = CAND[t]
            if rng is not None: cands = rng.permutation(cands)
            for j in cands:
                if j in P: continue
                if not OPOK[t, j]: cnt['skip_no_open'] += 1; continue
                if GAPg[t, j]: cnt['skip_gap'] += 1; continue
                if len(P) >= maxpos or cash < 0.25 * unit: cnt['skip_no_slot'] += 1; continue
                amt = min(unit, cash); px = AOg[t, j]
                P[j] = {'u': amt / (1 + buy) / px, 'p0': px, 'e': t, 'hh': -np.inf, 'stop': -np.inf, 'px': False}
                cash -= amt; cnt['entries'] += 1
        val = 0.0
        for j, p in P.items(): val += p['u'] * ACg[t, j]
        if ISWE[t]:
            up = IDXUP[WEK[t]] or fixed40
            for j, p in P.items():
                if p['px'] or t > lastvalid[j]: continue
                c = ACg[t, j]; p['hh'] = max(p['hh'], c)
                p['stop'] = max(p['stop'], p['hh'] * (0.6 if up else 0.9))
                if c < p['stop']: p['px'] = True
        k = t - t_start
        nav[k] = cash + val; npos[k] = len(P); inv[k] = val / (cash + val)
    for j, p in P.items(): trades.append((p['e'], T - 1, j, p['p0'], ACg[T - 1, j], 'open_at_end'))
    ix = G[t_start:]
    tr = pd.DataFrame(trades, columns=['e', 'x', 'j', 'p0', 'p1', 'why'])
    return pd.Series(nav, index=ix), pd.Series(npos, index=ix), pd.Series(inv, index=ix), cnt, tr

# 月曆與同池等權（同 w1_turtle_donchian）
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

KEYW = ('select_2005-01_2019-04', 'valid_2019-05_2026-10', 'y2024-01_2025-12')
navs = {'EW_pool': EW, '0050': E50}
port = {}; TRADES = {}
def run_variant(vn, costs, **kw):
    port[vn] = {}
    for cost in costs:
        b, s = (0.0, 0.0) if cost == 'gross' else L.cost_rates(cost)
        nav, npos, inv, cnt, tr = sim(b, s, **kw)
        navs[f'{vn}_{cost}'] = nav
        tr['ret'] = tr.p1 / tr.p0 - 1; tr['hold'] = tr.x - tr.e
        nt = (1 + tr.ret) * (1 - s) / (1 + b) - 1
        o = {'counts': cnt, 'mean_open_positions': round(float(npos[npos.index >= '2005-01-01'].mean()), 1),
             'mean_invested_share': round(float(inv[inv.index >= '2005-01-01'].mean()), 3),
             'trades': {'n': int(len(tr)), 'win_rate_net': round(float((nt > 0).mean()), 3), 'mean_net_pct': round(float(nt.mean() * 100), 2),
                        'median_net_pct': round(float(nt.median() * 100), 2), 'mean_hold_days': round(float(tr.hold.mean()), 1),
                        'median_hold_days': float(tr.hold.median()),
                        'payoff_ratio': round(float(nt[nt > 0].mean() / -nt[nt <= 0].mean()), 2)},
             'windows': {k: daily_stats(nav, sg) for k, sg in W.items()}, 'yearly_pct': yearly(nav[nav.index >= '2004-12-31'])}
        if cost == 0.001:
            o['open_positions_by_year'] = {int(y): round(float(g.mean()), 1) for y, g in npos.groupby(npos.index.year)}
            o['invested_by_year'] = {int(y): round(float(g.mean()), 3) for y, g in inv.groupby(inv.index.year)}
            o['entries_per_year'] = round(cnt['entries'] / ((G[-1] - G[t_start]).days / 365.25), 0)
            TRADES[vn] = tr
        port[vn][str(cost)] = o
        w_ = o['windows']
        print(vn, cost, {k: (w_[k]['CAGR'], w_[k]['0050_CAGR'], w_[k]['EW_CAGR'], w_[k]['t_vs_EW_m'], w_[k]['t_vs_0050_m'], w_[k]['maxDD']) for k in KEYW},
              round(time.time() - T0), flush=True)

run_variant('PRIMARY', ('gross', 0.001, 'stress'))
run_variant('D2_10x10pct', (0.001,), maxpos=10, w=0.10)
run_variant('D3_fixed40_stop', (0.001,), fixed40=True)
res['portfolio'] = port
res['EW_pool_windows'] = {k: daily_stats(EW, sg) for k, sg in W.items()}
res['EW_pool_yearly'] = yearly(EW[EW.index >= '2004-12-31'])
res['0050_yearly'] = yearly(E50[E50.index >= '2004-12-31'])
res['0050_windows'] = {k: daily_stats(E50, sg) for k, sg in W.items()}
res['pool_size_by_year'] = {int(y): round(float(np.mean([LIQ[gpos[d]].sum() for d in MD if d.year == y])), 0) for y in sorted(set(MD.year))}

rnd = {}
for seed in range(10):
    nav, *_ = sim(b1, s1, seed=seed)
    for k in KEYW:
        st = daily_stats(nav, W[k]); rnd.setdefault(k, []).append((st['CAGR'], st['t_vs_EW_m'], st['t_vs_0050_m']))
res['D1_random_order'] = {k: {'CAGR_mean': round(float(np.mean([x[0] for x in v])), 4), 'CAGR_min': round(float(np.min([x[0] for x in v])), 4),
                              'CAGR_max': round(float(np.max([x[0] for x in v])), 4), 't_vs_EW_mean': round(float(np.mean([x[1] for x in v])), 2),
                              't_vs_EW_max': round(float(np.max([x[1] for x in v])), 2), 't_vs_0050_mean': round(float(np.mean([x[2] for x in v])), 2)}
                          for k, v in rnd.items()}
print('random', res['D1_random_order'], round(time.time() - T0), flush=True)

# ---------------- 資料檢查 ----------------
FL = pd.read_parquet(L.P / 'flag.parquet').reindex(index=IDX, columns=COLS).to_numpy()[gi]
SRC = pd.read_parquet(L.P / 'src.parquet').reindex(index=IDX, columns=COLS).to_numpy()[gi]
chk = {}
for nm, tr in (('events', EV), ('PRIMARY_trades', TRADES['PRIMARY'])):
    has = np.array([bool(np.isin(FL[r.e:r.x + 1, r.j], [3, 4, 5]).any()) for r in tr.itertuples()])
    big = tr[(tr.ret > 2.0) | (tr.ret < -0.6)]
    chk[nm] = {'n': int(len(tr)), 'trades_with_flag345': int(has.sum()), 'entry_src3_share': round(float((SRC[tr.e, tr.j] == 3).mean()), 3),
               'n_ret_gt_200pct': int((tr.ret > 2.0).sum()), 'n_ret_lt_-60pct': int((tr.ret < -0.6).sum()),
               'max_ret': round(float(tr.ret.max()), 3), 'min_ret': round(float(tr.ret.min()), 3),
               'extreme_examples': big.assign(e=G[big.e.to_numpy()].astype(str), code=[COLS[j] for j in big.j])[['e', 'code', 'ret', 'why', 'hold']].round(3).head(15).to_dict('records'),
               'exit_why': tr.why.value_counts().to_dict()}
    if nm == 'events':
        keep = ~has & (tr.ret.abs() <= 2.0).to_numpy()
        st = event_stats(tr[keep], 0.001)
        chk[nm]['abn_ex_flag345_and_extremes'] = {k: (st[k]['mean_abn_pct'], st[k]['t_month_cluster']) for k in KEYW if k in st}
# 組合：去除前 5 大獲利交易的影響（粗估：該交易的貢獻 = 5% × 報酬）
tp = TRADES['PRIMARY'].copy(); tp['ey'] = G[tp.e.to_numpy()].year
chk['PRIMARY_top_trades'] = tp.assign(code=[COLS[j] for j in tp.j], e=G[tp.e.to_numpy()].astype(str), x=G[tp.x.to_numpy()].astype(str)).sort_values('ret', ascending=False)[['e', 'x', 'code', 'ret', 'why']].round(3).head(10).to_dict('records')
res['data_checks'] = chk

# ---------------- 判定 ----------------
def passes(k):
    w_ = port['PRIMARY']['0.001']['windows'][k]
    return {'CAGR': w_['CAGR'], '0050_CAGR': w_['0050_CAGR'], 'EW_CAGR': w_['EW_CAGR'], 't_vs_EW_m': w_['t_vs_EW_m'], 't_vs_0050_m': w_['t_vs_0050_m'],
            'pass': bool(w_['t_vs_EW_m'] >= 2 and w_['CAGR'] >= w_['0050_CAGR'])}
J = {k: passes(k) for k in ('select_2005-01_2019-04', 'valid_2019-05_2026-10')}
npass = sum(v['pass'] for v in J.values())
J['verdict'] = '可用' if npass == 2 else ('待觀察' if npass == 1 else '不可用')
res['judgement'] = J
print('judgement', json.dumps(J, ensure_ascii=False), flush=True)
(HERE / 'results.json').write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str), encoding='utf-8')
pd.DataFrame(navs).to_parquet(HERE / 'nav.parquet')
EV.assign(code=[COLS[j] for j in EV.j], d=EV.d.astype(str), em=EV.em.astype(str)).drop(columns=['j']).to_csv(HERE / 'events.csv.gz', index=False)
TRADES['PRIMARY'].assign(code=[COLS[j] for j in TRADES['PRIMARY'].j], e=G[TRADES['PRIMARY'].e.to_numpy()].astype(str),
                         x=G[TRADES['PRIMARY'].x.to_numpy()].astype(str)).drop(columns=['j']).to_csv(HERE / 'trades_PRIMARY.csv.gz', index=False)
print('done', round(time.time() - T0))
