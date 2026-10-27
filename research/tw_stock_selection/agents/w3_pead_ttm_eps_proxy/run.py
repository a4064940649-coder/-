#!/usr/bin/env python3
"""PEAD（本益比反推近四季 EPS 的季節性差額 SUE 近似）可成交回測。
先跑 prereg.py，再跑：python3 -I run.py
成交、成本、漲停、60 日均值沿用 momentum_family/mlib.py → exec_rebuild/exlib.py（只讀取）；權重版引擎 sim_w 同 w3_revenue_sue。"""
import json, math, sys, time, hashlib
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'momentum_family'))
import mlib as M  # noqa: E402
L = M.L
T0 = time.time()
XR = Path('/home/user/stevenhsiehsheng-ctrl/tw-stock-screener/data/extras')
LONG = Path('/home/user/niklavsd/bottleneck-signals/data/processed/tw_revenue.csv')

res = json.loads((HERE / 'results.json').read_text(encoding='utf-8'))
assert hashlib.sha256(json.dumps(res['PREREG'], ensure_ascii=False, sort_keys=True).encode()).hexdigest() == res['PREREG_sha256'], 'PREREG 被改過'
res = {'PREREG': res['PREREG'], 'PREREG_sha256': res['PREREG_sha256']}

IDX, COLS, N, pos, CI = L.IDX, L.COLS, L.N, L.pos, L.CI
common = np.array([not c.startswith('91') for c in COLS])

# ---------------- PE → TTM EPS ----------------
pe = pd.read_csv(XR / 'pe_5y.csv.gz', dtype={'code': str}, parse_dates=['date'])
pe = pe[(pe.market == 'TWSE') & pe.code.isin(COLS)]
PE = pe.pivot_table(index='date', columns='code', values='pe', aggfunc='last').sort_index().reindex(columns=COLS)
SNAP = PE.index
EPS = L.C.reindex(SNAP) / PE.where(PE > 0)          # 快照不在面板 → NaN

# ---------------- 事件 ----------------
rows = []
for y in range(2021, 2027):
    for q, md, fy in (('Q4', '03-31', y - 1), ('Q1', '05-15', y), ('Q2', '08-14', y), ('Q3', '11-14', y)):
        dl = pd.Timestamp(f'{y}-{md}')
        post = SNAP[SNAP > dl]
        if len(post) == 0 or dl < pd.Timestamp('2021-08-01'): continue
        post = post[0]
        if (post - dl).days > 40: continue
        rows.append({'q': f'{fy}{q}', 'fy': fy, 'qn': int(q[1]), 'deadline': dl, 'post': post})
EV = pd.DataFrame(rows).sort_values('deadline').reset_index(drop=True)
EV['prev'] = EV['post'].shift(1)
EV = EV.iloc[1:].reset_index(drop=True)            # 2021Q2 只當 prev
EV['d_main'] = [IDX[IDX > x][0] for x in EV.deadline]
EV['d_pit'] = EV['post']
assert all(x in IDX for x in EV.d_pit)
DT = pd.DataFrame([(EPS.loc[r.post] - EPS.loc[r.prev]).to_numpy() for r in EV.itertuples()], columns=COLS)
EPS_UPD = [float(((EPS.loc[r.post] / EPS.loc[r.prev] - 1).abs() > 1e-3).sum() / max(1, (EPS.loc[r.post].notna() & EPS.loc[r.prev].notna()).sum())) for r in EV.itertuples()]
SD8 = DT.shift(1).rolling(8, min_periods=8).std(ddof=1)
STD_SUE = (DT / SD8).where(SD8 > 0)
Cf = L.C.ffill(limit=5)
def pref(d):
    m0 = pd.Timestamp(d.year, d.month, 1); return IDX[IDX < m0][-1]

# ---------------- 季 SURGE（變體 b）----------------
d = pd.read_csv(LONG, dtype={'code': str, 'market': str, 'name': str})
d['p'] = pd.PeriodIndex(pd.to_datetime(d['month']), freq='M')
d = d.sort_values(['p', 'market']).drop_duplicates(['p', 'code'], keep='first'); d = d[d['code'].isin(COLS)]
REV = d.pivot_table(index='p', columns='code', values='revenue_k_twd', aggfunc='last').sort_index()
REV = REV.reindex(pd.period_range(REV.index.min(), REV.index.max(), freq='M')).reindex(columns=COLS).where(lambda x: x > 0)
Dr = REV - REV.shift(12); Drl = Dr.shift(1)
SURGE = ((Dr - Drl.rolling(24, min_periods=24).mean()) / Drl.rolling(24, min_periods=24).std(ddof=1))
SURGE = SURGE.where(np.isfinite(SURGE))
def q_surge(fy, qn):
    ms = [pd.Period(f'{fy}-{3 * (qn - 1) + k:02d}', 'M') for k in (1, 2, 3)]
    X = SURGE.reindex(ms)
    return X.mean(axis=0).where(X.notna().sum(axis=0) >= 2).to_numpy()
QS = np.vstack([q_surge(r.fy, r.qn) for r in EV.itertuples()])
print('signals built', round(time.time() - T0), flush=True)

# ---------------- 交易日曆 ----------------
TD = pd.DatetimeIndex(pd.read_csv('/home/user/jonatw/finlab-txd/data/raw/etf_0050.csv', parse_dates=['date'])['date']).sort_values()
TD = TD[TD <= IDX[-1]]
def exit_of(ex, H):
    e = IDX[ex]; k = TD.get_indexer([e])[0]
    assert k >= 0, e
    if k + H >= len(TD): return len(IDX) - 1, True
    t = TD[k + H]; j = IDX.searchsorted(t)
    if j >= len(IDX): return len(IDX) - 1, True
    return int(j), False

E50 = L.E50.ffill(); E50o = E50.shift(1).bfill()
AC2 = np.hstack([L.AC, E50.to_numpy()[:, None]]); AO2 = np.hstack([L.AO, E50o.to_numpy()[:, None]]); J50 = N
v60_all = M.V60

def build(variant):
    H = 20 if variant == 'a_hold20' else 60
    dcol = 'd_pit' if variant == 'PIT' else 'd_main'
    out = []
    for k, r in EV.iterrows():
        dd = r[dcol]; i = pos(dd); ex = i + 1
        if ex >= len(IDX) - 1: continue
        P0 = Cf.loc[pref(dd)].to_numpy()
        if variant == 'c_std_sue': sc = STD_SUE.iloc[k].to_numpy()
        else: sc = DT.iloc[k].to_numpy() / np.where(P0 > 0, P0, np.nan)
        trad = np.isfinite(L.OPEN_RAW[ex]) & np.isfinite(L.ADJ_RAW[ex])
        liq = (v60_all.loc[dd].to_numpy() >= 1e7) & (L.C.loc[dd].to_numpy() >= 10) & common & trad
        pool = liq & np.isfinite(sc)
        if variant == 'b_rev_confirm': pool = pool & np.isfinite(QS[k])
        pp = np.where(pool)[0]; n = len(pp)
        if n == 0: continue
        order = pp[np.lexsort((pp, -sc[pp]))]; ntop = max(1, int(round(0.10 * n)))
        cand = list(order[:ntop])
        if variant == 'b_rev_confirm':
            rq = pd.Series(QS[k][pp], index=pp).rank(pct=True)
            cand = [j for j in cand if rq[j] >= 0.70]
        pick = [j for j in cand if not M.GAP_UP[ex, j]]
        xt, trunc = exit_of(ex, H)
        out.append({'k': k, 'q': r.q, 'd': dd, 'ex': ex, 'exit': xt, 'trunc': trunc, 'pool': pp, 'liq': np.where(liq)[0],
                    'cand': cand, 'pick': pick, 'n_pool': n, 'n_cand': len(cand), 'n_pick': len(pick), 'sc': sc})
    return out

def wvec(pick):
    kk = len(pick)
    if kk >= 5: return {j: 1.0 / kk for j in pick}
    o = {j: 0.2 for j in pick}; o[J50] = (5 - kk) * 0.2; return o

def sim_w(events, buy, sell, t0, t1):
    NN = N + 1; u = np.zeros(NN); cash = 1.0; nav = np.empty(t1 - t0 + 1)
    for t in range(t0, t1 + 1):
        if t in events:
            pk, w = events[t]; Pz = AO2[t]
            h = np.where(u > 0, u * np.nan_to_num(Pz), 0.0)
            Vt = h.sum() + cash; Vn = Vt; tgt = np.zeros(NN)
            for _ in range(6):
                tgt[:] = 0; tgt[pk] = w * Vn
                cost = buy * np.clip(tgt - h, 0, None).sum() + sell * np.clip(h - tgt, 0, None).sum()
                Vn = Vt - cost
            u = np.zeros(NN); u[pk] = tgt[pk] / Pz[pk]; cash = 0.0
        nav[t - t0] = np.nansum(u * AC2[t]) + cash
    return pd.Series(nav, index=IDX[t0:t1 + 1])

def port_events(evs, kind):
    E = {}
    for z, e in enumerate(evs):
        if kind == 'pick': tw = wvec(e['pick'])
        else: tw = {j: 1.0 / len(e['pool']) for j in e['pool']}
        E[e['ex']] = (np.array(list(tw.keys()), dtype=int), np.array(list(tw.values())))
        nxt = evs[z + 1]['ex'] if z + 1 < len(evs) else 10 ** 9
        if not e['trunc'] and e['exit'] < nxt: E[e['exit']] = (np.array([J50]), np.array([1.0]))
    return E

PERIODS = {'select_2021-09_2023-12': ('2021-09-01', '2023-12-31'), 'valid_2024-01_2026-10': ('2024-01-01', '2026-10-31'),
           'task_select_2022-01_2023-12': ('2022-01-01', '2023-12-31'), 'y2024-01_2025-12': ('2024-01-01', '2025-12-31'),
           'valid_officialPx_sig_ge_2025-09-24': ('2025-09-24', '2026-10-31')}
def inper(dd, p): a, b = PERIODS[p]; return pd.Timestamp(a) <= dd <= pd.Timestamp(b)

FL = pd.read_parquet(L.P / 'flag.parquet').reindex(index=IDX, columns=COLS).to_numpy()

def event_table(evs, cost):
    b, s = (0.0, 0.0) if cost == 'gross' else L.cost_rates(cost)
    rows = []
    for e in evs:
        p0 = L.AO[e['ex']]; p1 = L.AC[e['exit']] if e['trunc'] else L.AO[e['exit']]
        rr = p1 / p0 - 1
        ewp = np.nanmean(rr[e['pool']]); ewl = np.nanmean(rr[e['liq']])
        r50 = (E50.iloc[e['exit']] if e['trunc'] else E50o.iloc[e['exit']]) / E50o.iloc[e['ex']] - 1
        for j in e['pick']:
            if not np.isfinite(rr[j]): continue
            net = (1 + rr[j]) * (1 - s) / (1 + b) - 1
            rows.append({'q': e['q'], 'd': e['d'], 'code': COLS[j], 'ret': rr[j], 'net': net, 'abn': net - ewp, 'abn_liqEW': net - ewl, 'ex50': net - r50,
                         'poolEW': ewp, 'r50': r50, 'trunc': e['trunc'], 'flag345': bool(np.isin(FL[e['ex']:e['exit'] + 1, j], [3, 4, 5]).any())})
    return pd.DataFrame(rows)

def ev_stats(T, p):
    x = T[[inper(dd, p) for dd in T.d]] if len(T) else T
    if len(x) == 0: return None
    g = x.groupby('d')
    ga, g50, gl, gn = g.abn.mean(), g.ex50.mean(), g.abn_liqEW.mean(), g.net.mean()
    return {'n_events': int(len(x)), 'n_quarters': int(len(ga)), 'mean_picks_per_q': round(len(x) / len(ga), 1),
            'q_mean_net_ret_pct': round(float(gn.mean() * 100), 2), 'q_mean_poolEW_ret_pct': round(float(g.poolEW.first().mean() * 100), 2),
            'q_mean_0050_ret_pct': round(float(g.r50.first().mean() * 100), 2),
            'q_mean_abn_vs_poolEW_pct': round(float(ga.mean() * 100), 2), 't_abn_vs_poolEW_q': L.tstat(ga), 'q_hit_abn': round(float((ga > 0).mean()), 3),
            'q_mean_abn_vs_liqEW_pct': round(float(gl.mean() * 100), 2), 't_abn_vs_liqEW_q': L.tstat(gl),
            'q_mean_ex_vs_0050_pct': round(float(g50.mean() * 100), 2), 't_vs_0050_q': L.tstat(g50),
            'event_hit_abn': round(float((x.abn > 0).mean()), 3), 'per_quarter_abn_pct': {str(k.date()): round(float(v * 100), 2) for k, v in ga.items()}}

def port_stats(nav, ew, evs, p):
    E = [e for e in evs if inper(e['d'], p)]
    if len(E) < 2: return None
    zs = [evs.index(e) for e in E]
    a = E[0]['ex'] - 1; z_last = zs[-1]
    b = evs[z_last + 1]['ex'] - 1 if z_last + 1 < len(evs) else len(IDX) - 1
    sl = slice(IDX[a], IDX[b]); out = {}
    span = (IDX[b] - IDX[a]).days / 365.25
    for nm, s in (('strat', nav), ('EW', ew), ('0050', E50)):
        x = s.loc[sl]; eq = x / x.iloc[0]; r = x.pct_change().dropna()
        out[f'{nm}_CAGR'] = round(float(eq.iloc[-1] ** (1 / span) - 1), 4); out[f'{nm}_maxDD'] = round(float((eq / eq.cummax() - 1).min()), 4)
        if nm == 'strat': out['vol'] = round(float(r.std() * math.sqrt(252)), 4)
    seg = []
    for z in zs:
        s0 = evs[z]['ex'] - 1; s1 = evs[z + 1]['ex'] - 1 if z + 1 < len(evs) else len(IDX) - 1
        seg.append({'st': nav.iloc[s1 - t0g] / nav.iloc[s0 - t0g] - 1, 'ew': ew.iloc[s1 - t0g] / ew.iloc[s0 - t0g] - 1, 'e50': E50.iloc[s1] / E50.iloc[s0] - 1})
    seg = pd.DataFrame(seg)
    out.update({'n_segments': len(seg), 'years': round(span, 2), 'seg_mean_ex_vs_EW_pct': round(float((seg.st - seg.ew).mean() * 100), 2), 't_vs_EW_seg': L.tstat(seg.st - seg.ew),
                'seg_mean_ex_vs_0050_pct': round(float((seg.st - seg.e50).mean() * 100), 2), 't_vs_0050_seg': L.tstat(seg.st - seg.e50)})
    return out

def yearly(s):
    r = s.pct_change().dropna(); return {int(y): round(float((1 + g).prod() - 1) * 100, 1) for y, g in r.groupby(r.index.year)}

VARS = ('S_main', 'a_hold20', 'b_rev_confirm', 'c_std_sue', 'PIT')
NAVS = {}; ALL = {}
for v in VARS:
    evs = build(v); ALL[v] = evs
    t0g = evs[0]['ex'] - 1; t1g = len(IDX) - 1
    ew = sim_w(port_events(evs, 'pool'), 0.0, 0.0, t0g, t1g); NAVS[f'{v}_EW'] = ew
    o = {'events': [{'q': e['q'], 'd': str(e['d'].date()), 'entry': str(IDX[e['ex']].date()), 'exit': str(IDX[e['exit']].date()), 'trunc': e['trunc'],
                     'n_pool': e['n_pool'], 'n_cand': e['n_cand'], 'n_pick': e['n_pick'], 'blocked_limit_up': e['n_cand'] - e['n_pick'],
                     'picks': [COLS[j] for j in e['pick']][:40]} for e in evs]}
    for cost in ('gross', 0.001, 'stress'):
        T = event_table(evs, cost)
        b, s = (0.0, 0.0) if cost == 'gross' else L.cost_rates(cost)
        nav = sim_w(port_events(evs, 'pick'), b, s, t0g, t1g); NAVS[f'{v}_{cost}'] = nav
        o[f'cost_{cost}'] = {'event': {p: ev_stats(T, p) for p in PERIODS}, 'portfolio': {p: port_stats(nav, ew, evs, p) for p in PERIODS},
                             'portfolio_yearly_pct': yearly(nav)}
        if cost == 0.001:
            big = T[T.ret > 1.0]
            o['sanity'] = {'events_ret_gt_100pct': big.assign(d=big.d.astype(str)).round(3)[['q', 'code', 'ret']].to_dict('records'),
                           'n_flag345_in_hold': int(T.flag345.sum()), 'n_events_total': int(len(T)), 'n_truncated_events': int(T.trunc.sum())}
            T.assign(d=T.d.astype(str)).round(5).to_csv(HERE / f'events_{v}.csv.gz', index=False)
    o['EW_yearly_pct'] = yearly(ew)
    res[v] = o
    m = o['cost_0.001']['event']
    print(v, {p: (m[p]['q_mean_abn_vs_poolEW_pct'], m[p]['t_abn_vs_poolEW_q'], m[p]['q_mean_ex_vs_0050_pct'], m[p]['n_quarters'], m[p]['mean_picks_per_q']) for p in PERIODS if m.get(p)}, round(time.time() - T0), flush=True)

res['0050_yearly_pct'] = yearly(E50.loc[IDX[ALL['S_main'][0]['ex'] - 1]:])
res['diag_eps_updated_share_by_event'] = {r.q: {'prev': str(r.prev.date()), 'post': str(r.post.date()), 'share_eps_changed_gt_0.1pct': round(x, 3)} for r, x in zip(EV.itertuples(), EPS_UPD)}

# ---- 十分位（主版本、未扣成本）----
rowsq = []
for e in ALL['S_main']:
    pp = e['pool']; n = len(pp)
    if n < 50: continue
    p1 = L.AC[e['exit']] if e['trunc'] else L.AO[e['exit']]
    rr = p1[pp] / L.AO[e['ex']][pp] - 1
    rk = pd.Series(e['sc'][pp]).rank(method='first', ascending=False).to_numpy() - 1
    dec = np.floor(rk / n * 10).astype(int) + 1
    ewr = np.nanmean(rr); row = {'d': e['d']}
    for q in range(1, 11): row[q] = np.nanmean(rr[dec == q]) - ewr
    rowsq.append(row)
Q = pd.DataFrame(rowsq).set_index('d'); dq = {}
for p in ('select_2021-09_2023-12', 'valid_2024-01_2026-10'):
    x = Q[[inper(dd, p) for dd in Q.index]]
    dq[p] = {'n_q': int(len(x)), 'decile_abn_pct (1=最高SUE)': {int(q): round(float(x[q].mean() * 100), 2) for q in range(1, 11)},
             't_D1': L.tstat(x[1]), 't_D1_minus_D10': L.tstat(x[1] - x[10])}
res['diag_deciles_gross_60d'] = dq

# ---- EPS 雜訊（非季報月）----
nz = {'official': [], 'src3': []}
for a, b in zip(SNAP[:-1], SNAP[1:]):
    if b.month not in (1, 2, 7, 8, 10, 11): continue
    rel = (EPS.loc[b] / EPS.loc[a] - 1).abs().dropna()
    if len(rel) == 0: continue
    key = 'src3' if pd.Timestamp('2024-02-15') <= b <= pd.Timestamp('2025-09-23') else 'official'
    nz[key].append(rel.to_numpy())
res['diag_eps_noise_nonreport_months'] = {k: {'median_abs_rel_change': round(float(np.median(np.concatenate(v))), 5), 'p90': round(float(np.quantile(np.concatenate(v), 0.9)), 5), 'n': int(sum(len(z) for z in v))} for k, v in nz.items() if v}
sp = pd.concat([pd.Series(DT.iloc[k].to_numpy() / Cf.loc[pref(r.d_main)].to_numpy()) for k, r in EV.iterrows()]).dropna()
res['diag_sue_price_dist'] = {str(q): round(float(sp.quantile(q)), 4) for q in (0.01, 0.1, 0.5, 0.9, 0.99)}
(HERE / 'results.json').write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str), encoding='utf-8')
pd.DataFrame(NAVS).to_parquet(HERE / 'nav.parquet')
print('done', round(time.time() - T0))
