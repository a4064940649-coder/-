#!/usr/bin/env python3
"""w12 假日前效應：依 results.json 的 PREREG 執行（不改 PREREG）。"""
import json, math, sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'exec_rebuild'))
import exlib as L  # 只讀：cost_rates、FEE、LTOL、面板

RES = json.loads((HERE / 'results.json').read_text(encoding='utf-8'))
PRE = RES['PREREG']
OUT = {'PREREG': PRE}
UNS = set(PRE['calendar']['unscheduled_days'])

# ---------------- 資料與日曆 ----------------
BAD = pd.Timestamp('2018-01-13')
adj = L.adj.drop(index=BAD); C = L.C.drop(index=BAD); H = L.H.drop(index=BAD); O = L.O.drop(index=BAD); V = L.V.drop(index=BAD)
PIDX = adj.index; N = len(adj.columns)
e = pd.read_csv('/home/user/jonatw/finlab-txd/data/raw/etf_0050.csv', parse_dates=['date']).set_index('date')['adj'].sort_index()
e = e.drop(index=[pd.Timestamp('2026-07-10'), pd.Timestamp('2026-09-20')])
gapd = e.index[(e.index >= '2024-01-02') & (e.index <= '2024-02-14')]
CAL = PIDX.union(gapd); CAL = CAL[(CAL >= '2004-01-02') & (CAL <= '2026-10-22')]
E = e.reindex(CAL).ffill()
tx = pd.read_csv('/home/user/jonatw/finlab-txd/data/raw/taiex_twii.csv', parse_dates=['date']).set_index('date')['close'].sort_index()
TX = tx.reindex(CAL).ffill()
NC = len(CAL); cpos = {d: i for i, d in enumerate(CAL)}; inpanel = CAL.isin(PIDX); ppos = {d: i for i, d in enumerate(PIDX)}

AC = adj.ffill().to_numpy(); ADJ = adj.to_numpy(); CN = C.to_numpy(); HN = H.to_numpy()
RN = adj.pct_change(fill_method=None).to_numpy()
AOr = (O * (adj / C)).to_numpy()
V60 = V.rolling(60, min_periods=50).mean().to_numpy()
thr = np.where(PIDX < pd.Timestamp('2015-06-01'), 0.07, 0.10)[:, None] - L.LTOL
LOCK = (RN >= thr) & (CN >= HN * 0.999)
AC_prev = np.vstack([np.full((1, N), np.nan), AC[:-1]])
GAPUP = (AOr / AC_prev - 1) >= thr

# ---------------- 假日空檔 ----------------
twh = json.loads(Path('/home/user/stevenhsiehsheng-ctrl/tw-stock-screener/data/extras/tw_holidays.json').read_text())
twh = {pd.Timestamp(d) for y in twh['years'].values() for d in y}
gaps = []
for i in range(NC - 1):
    a, b = CAL[i], CAL[i + 1]
    wd = pd.bdate_range(a + pd.Timedelta(days=1), b - pd.Timedelta(days=1))
    if len(wd): gaps.append((i, [str(x.date()) for x in wd]))
# 資料最後一日之後的事前公告假日
nxt = pd.bdate_range(CAL[-1] + pd.Timedelta(days=1), periods=1)[0]
if nxt in twh: gaps.append((NC - 1, [str(nxt.date()) + '（tw_holidays.json）']))
G = []
for i, days in gaps:
    core = [d[:10] for d in days]
    sched = [d for d in core if d not in UNS]; typh = [d for d in core if d in UNS]
    G.append({'T1': i, 'T1_date': str(CAL[i].date()), 'days': core, 'n_sched': len(sched), 'n_unsched': len(typh)})
G = pd.DataFrame(G)
# 農曆年：每年 1/10～2/28 最長空檔
G['len'] = G.days.map(len); G['first'] = pd.to_datetime(G.days.map(lambda x: x[0]))
G['LNY'] = False
for y, g in G.groupby(G['first'].dt.year):
    w = g[(g['first'] >= f'{y}-01-10') & (g['first'] <= f'{y}-02-28')]
    if len(w): G.loc[w['len'].idxmax(), 'LNY'] = True
EVH = G[G.n_sched > 0].reset_index(drop=True)          # 事前已知假日
TYP = G[G.n_sched == 0].reset_index(drop=True)          # 只有颱風
# 檢查：tw_holidays 2026 與推得的 2026 空檔
der26 = sorted(d for ds in G.days for d in ds if d.startswith('2026') and d not in UNS)
off26 = sorted(str(d.date()) for d in twh if d.year == 2026 and d.weekday() < 5 and d <= pd.Timestamp('2026-10-23'))
# 颱風落在 T-4～T-1 之間（會改變 T-k 的實際日期）
uns_gap_T1 = set(TYP.T1)
win_typh = [r.T1_date for r in EVH.itertuples() if any((r.T1 - k) in uns_gap_T1 for k in range(1, 4))]
# 下一個空檔也有排程假日（兩個事件之間的交易日數）
OUT['calendar_check'] = {
 'n_gaps': int(len(G)), 'n_holiday_events': int(len(EVH)), 'n_typhoon_only_gaps': int(len(TYP)),
 'typhoon_only_gaps': [f"{r.T1_date}→{r.days}" for r in TYP.itertuples()],
 'mixed_gaps': [f"{r.T1_date}→{r.days}" for r in EVH[EVH.n_unsched > 0].itertuples()],
 'LNY_T1': [r.T1_date for r in EVH[EVH.LNY].itertuples()],
 'events_per_year': {int(k): int(v) for k, v in EVH.groupby(pd.to_datetime(EVH.T1_date).dt.year).size().items()},
 'unscheduled_gap_within_T4_T1_of_event': win_typh,
 '2026_derived_vs_tw_holidays_json': {'derived': der26, 'json_weekdays_to_10-23': off26, 'match': der26 == [d for d in off26]},
 'all_events': [f"{r.T1_date}{'(LNY)' if r.LNY else ''}→{r.days[0]}..{r.days[-1]}({r.len}d)" for r in EVH.itertuples()]}

# ---------------- 共用 ----------------
def net(g, buy, sell): return (1 - buy) * (1 + g) * (1 - sell) - 1
COST50 = {'zero': (0, 0), 'etf_base': (L.FEE, L.FEE + 0.001), 'etf_slip0.1': (L.FEE + 0.001, L.FEE + 0.002), 'stress': L.cost_rates('stress')}
COSTP = {'zero': (0, 0), 'rt0.5': (0.0025, 0.0025), 'exlib_slip0.1': L.cost_rates(0.001), 'stress_rt1': L.cost_rates('stress')}
OUT['cost_check_roundtrip_pct'] = {k: round((1 - (1 - b) * (1 - s)) * 100, 3) for k, (b, s) in {**COST50, **{'P_' + k: v for k, v in COSTP.items()}}.items()}
PER = {'IS': ('2005-01-01', '2019-04-30'), 'OOS': ('2019-05-01', '2026-10-22'), 'y2024_2025': ('2024-01-01', '2025-12-31'),
       'y2021_2023': ('2021-01-01', '2023-12-31'), 'y2024_2026': ('2024-01-01', '2026-10-22')}

def tt(x):
    x = pd.Series(x).dropna()
    if len(x) < 3 or x.std() == 0: return float('nan')
    return round(float(x.mean() / (x.std(ddof=1) / math.sqrt(len(x)))), 2)
def pm(x): return round(float(pd.Series(x).dropna().mean() * 100), 3)
def cl(x, months):
    """按進場月份分群：同月加總後的月平均與 t（回傳 每事件平均, 月平均, t, 月數）"""
    s = pd.Series(np.asarray(x, float), index=months).dropna()
    g = s.groupby(level=0).sum()
    return pm(s), pm(g), tt(g), int(len(g))

def universe(d):
    i = ppos[d]; v, px = V60[i - 1], CN[i - 1]
    base = (v >= 1e7) & (px >= 10)
    order = np.argsort(-np.where(base, v, -np.inf))[:300]
    u = np.zeros(N, bool); u[order] = True; return u & base

def basket(p, q, mode='close', excl_lock=True, which='pool'):
    if not inpanel[p:q + 1].all(): return None
    i = ppos[CAL[p]]; j = ppos[CAL[q]]
    u = universe(CAL[p]) if which == 'pool' else np.ones(N, bool)
    u = u & np.isfinite(ADJ[i])
    if which == 'pool' and excl_lock and mode == 'close': u &= ~LOCK[i]
    if mode == 'open':
        u &= np.isfinite(AOr[i + 1]) & ~GAPUP[i + 1]; base_px = AOr[i + 1]
    else:
        base_px = AC[i]
    idx = np.where(u)[0]
    if len(idx) < 30: return None
    vals = np.nanmean(AC[i + 1:j + 1, idx] / base_px[idx], axis=1)
    return float(vals[-1] - 1), vals / np.r_[1.0, vals[:-1]] - 1, len(idx)

r50 = E.pct_change().to_numpy().copy(); rtx = TX.pct_change().to_numpy().copy(); r50[0] = rtx[0] = np.nan
# 同池等權、上市等權日報酬（每月第一個交易日換池）
poolEW_d = np.full(NC, np.nan); listEW_d = np.full(NC, np.nan)
mon = CAL.to_period('M'); u = None
for t in range(1, NC):
    if not inpanel[t]: continue
    if (u is None or mon[t] != mon[t - 1]) and ppos[CAL[t]] > 0: u = universe(CAL[t])
    row = RN[ppos[CAL[t]]]; poolEW_d[t] = np.nanmean(row[u]); listEW_d[t] = np.nanmean(row)
DAILY = {'0050': r50, 'TAIEX': rtx, 'poolEW': poolEW_d, 'listedEW': listEW_d}

def make_events(h, lny_only=False, src=EVH):
    ev = src[src.LNY] if lny_only else src
    ev = pd.DataFrame({'q': ev.T1.to_numpy(), 'p': ev.T1.to_numpy() - h, 'LNY': ev.LNY.to_numpy()})
    ev = ev[ev.p >= 1].reset_index(drop=True)
    ev['p_next'] = ev.p.shift(-1)
    ev['entry'] = [CAL[int(p)] for p in ev.p]; ev['exit'] = [CAL[int(q)] for q in ev.q]
    ev['m'] = ev.entry.dt.to_period('M')
    return ev

def build(ev, asset):
    g, daily, nst = [], {}, []
    for k, r in ev.iterrows():
        p, q = int(r.p), int(r.q)
        if asset == '0050': gg, dl, ns = E.iloc[q] / E.iloc[p] - 1, r50[p + 1:q + 1], 1
        elif asset == 'TAIEX': gg, dl, ns = TX.iloc[q] / TX.iloc[p] - 1, rtx[p + 1:q + 1], 1
        else:
            o = basket(p, q, mode='open' if asset == 'pool_open' else 'close')
            if o is None: g.append(np.nan); nst.append(0); continue
            gg, dl, ns = o
        g.append(gg); nst.append(ns); daily[k] = (p, q, dl)
    ev = ev.copy(); ev['gross'] = g; ev['n'] = nst
    lew, pcyc, c50 = [], [], []
    for k, r in ev.iterrows():
        p, q = int(r.p), int(r.q)
        o = basket(p, q, which='all'); lew.append(o[0] if o else np.nan)
        if np.isfinite(r.p_next) and r.p_next > p:
            pn = int(r.p_next); o2 = basket(p, pn, which='pool', excl_lock=False)
            pcyc.append(o2[0] if o2 else np.nan); c50.append(E.iloc[pn] / E.iloc[p] - 1)
        else: pcyc.append(np.nan); c50.append(np.nan)
    ev['listedEW_win'] = lew; ev['poolEW_cycle'] = pcyc; ev['e50_cycle'] = c50
    ev['expo'] = (ev.q - ev.p) / (ev.p_next - ev.p)
    return ev, daily

def psel(ev, per):
    a, b = PER[per]; return ev[(ev.entry >= a) & (ev.entry <= b)]

def winmask(ev):
    m = np.zeros(NC, bool)
    for r in ev.itertuples(): m[int(r.p) + 1:int(r.q) + 1] = True
    return m

# ---------------- S1：假日前持有日 vs 其他日 ----------------
def s1(ev):
    im = winmask(ev); res = {}
    yrs = CAL.year
    for per, (da, db) in PER.items():
        dm = (CAL >= da) & (CAL <= db); evp = psel(ev, per)
        for nm, x in DAILY.items():
            ok = dm & np.isfinite(x); xi, xo = x[ok & im], x[ok & ~im]
            if len(xi) < 3: continue
            tw = stats.ttest_ind(xi, xo, equal_var=False).statistic
            exc = []
            for r in evp.itertuples():
                p, q = int(r.p), int(r.q); seg = x[p + 1:q + 1]
                yo = np.isfinite(x) & ~im & (yrs == CAL[q].year)
                exc.append(np.nansum(seg) - (q - p) * np.nanmean(x[yo]) if np.isfinite(seg).all() else np.nan)
            a1, a2, t2, nm_ = cl(exc, evp.m.to_numpy())
            res[f'{nm}|{per}'] = {'in_mean_pct': round(float(xi.mean() * 100), 3), 'out_mean_pct': round(float(xo.mean() * 100), 3),
                                  'diff_pct': round(float((xi.mean() - xo.mean()) * 100), 3), 't_welch_daily': round(float(tw), 2),
                                  'n_in_days': int(len(xi)), 'n_out_days': int(len(xo)),
                                  'event_excess_gross_pct': a1, 'month_cluster_mean_pct': a2, 't_cluster_month': t2, 'n_months': nm_}
    return res

def by_rel_day():
    """各相對日（T-5..T-1、T+1..T+3，T+1 = 假日後第一個交易日）日均報酬（描述）"""
    res = {}
    for nm in ('0050', 'poolEW', 'TAIEX'):
        x = DAILY[nm]
        for per in ('IS', 'OOS'):
            da, db = PER[per]; out = {}
            for k in [-5, -4, -3, -2, -1, 1, 2, 3]:
                vals = []
                for r in EVH.itertuples():
                    t = r.T1 + k + 1 if k < 0 else r.T1 + k
                    if 0 < t < NC and da <= str(CAL[t].date()) <= db: vals.append(x[t])
                v = pd.Series(vals).dropna(); out[k] = {'mean_pct': round(float(v.mean() * 100), 3), 't0': tt(v), 'n': int(len(v)), 'hit': round(float((v > 0).mean()), 3)}
            v = pd.Series(x[(CAL >= da) & (CAL <= db)]).dropna(); out['all_days_mean_pct'] = round(float(v.mean() * 100), 3)
            res[f'{nm}|{per}'] = out
    return res

# ---------------- S2/S3 ----------------
def event_stats(ev, costs):
    res = {}
    for per in PER:
        x = psel(ev, per).dropna(subset=['gross'])
        if len(x) < 3: continue
        M = x.m.to_numpy()
        a1, a2, t2, nmo = cl(x.gross, M)
        o = {'n_events': int(len(x)), 'n_months': nmo, 'gross_mean_pct': a1, 't_gross_month': t2, 'hit_gross': round(float((x.gross > 0).mean()), 3),
             'AR_vs_listedEW_gross_pct': pm(x.gross - x.listedEW_win), 't_AR_gross_month': cl(x.gross - x.listedEW_win, M)[2],
             'expo_mean': round(float(x.expo.mean()), 4)}
        d0 = x.gross - x.expo * x.e50_cycle; o['gross_vs_0050xExpo_pct'] = pm(d0); o['t_gross_vs_0050xExpo'] = cl(d0, M)[2]
        dp0 = x.gross - x.expo * x.poolEW_cycle; o['gross_vs_poolEWxExpo_pct'] = pm(dp0); o['t_gross_vs_poolEWxExpo'] = cl(dp0, M)[2]
        for cn, (bu, se) in costs.items():
            nt = net(x.gross, bu, se); d50 = nt - x.expo * x.e50_cycle; dp = nt - x.expo * x.poolEW_cycle
            o[cn] = {'net_mean_pct': pm(nt), 't_net_month': cl(nt, M)[2], 'hit_net': round(float((nt > 0).mean()), 3),
                     'AR_vs_listedEW_pct': pm(nt - x.listedEW_win), 't_AR_month': cl(nt - x.listedEW_win, M)[2],
                     'vs_0050xExpo_pct': pm(d50), 't_vs_0050xExpo': cl(d50, M)[2], 'n_0050': int(d50.notna().sum()),
                     'vs_poolEWxExpo_pct': pm(dp), 't_vs_poolEWxExpo': cl(dp, M)[2], 'n_pool': int(dp.notna().sum())}
        res[per] = o
    return res

# ---------------- S4：日頻 NAV ----------------
def cash_daily(rate): return (1 + rate) ** (1 / 252) - 1
def nav_daily(daily, buy, sell, cash_rate):
    s = np.full(NC, cash_daily(cash_rate)); held = np.zeros(NC, bool)
    for k in sorted(daily):
        p, q, dl = daily[k]; dl = np.nan_to_num(np.asarray(dl, float))
        for j, t in enumerate(range(p + 1, q + 1)):
            if not held[t]: s[t] = dl[j]; held[t] = True        # 重疊日沿用先進場的持股
    t = 1
    while t < NC:
        if held[t]:
            st = t
            while t + 1 < NC and held[t + 1]: t += 1
            s[st - 1] = (1 + s[st - 1]) * (1 - buy) - 1; s[t] = (1 + s[t]) * (1 - sell) - 1
        t += 1
    return s, held

def perf(s, da, db, inm=None):
    dm = (CAL >= da) & (CAL <= db); x = pd.Series(s[dm], index=CAL[dm]).fillna(0)
    eq = (1 + x).cumprod(); yrs = (x.index[-1] - x.index[0]).days / 365.25
    o = {'CAGR': round(float(eq.iloc[-1] ** (1 / yrs) - 1), 4), 'maxDD': round(float((eq / eq.cummax() - 1).min()), 4)}
    if inm is not None: o['exposure'] = round(float(inm[dm].mean()), 4)
    o['yearly_pct'] = {int(y): round(float((1 + g).prod() - 1) * 100, 2) for y, g in x.groupby(x.index.year)}
    return o

def nav_block(daily, costs):
    res = {}
    for per, (da, db) in PER.items():
        o = {}
        for cr in (0.0, 0.01):
            for cn, (bu, se) in costs.items():
                s, inm = nav_daily(daily, bu, se, cr); o[f'{cn}|cash{cr}'] = perf(s, da, db, inm)
        _, inm = nav_daily(daily, 0, 0, 0.0); dm = (CAL >= da) & (CAL <= db); ex = float(inm[dm].mean())
        for bn, bd in (('0050', r50), ('poolEW', poolEW_d)):
            o[f'BH_{bn}'] = perf(np.nan_to_num(bd), da, db)
            o[f'{bn}_x_expo_cash0'] = perf(np.nan_to_num(bd) * ex, da, db)
        res[per] = o
    return res

# ---------------- S5：隨機同曝險 ----------------
def random_ctrl(ev, series, buy, sell, nsim=2000, seed=83):
    rng = np.random.default_rng(seed)
    Lc = np.r_[0.0, np.cumsum(np.log1p(np.nan_to_num(series[1:])))]
    res = {}
    for per in ('IS', 'OOS', 'y2024_2025'):
        x = psel(ev, per)
        hs = (x.q - x.p).to_numpy().astype(int); ps0 = x.p.to_numpy().astype(int)
        act = net(np.exp(Lc[ps0 + hs] - Lc[ps0]) - 1, buy, sell).mean()
        cand = [np.where((mon == m) & (np.arange(NC) + h < NC))[0] for m, h in zip(x.m, hs)]
        sims = np.empty(nsim)
        for k in range(nsim):
            ps = np.array([rng.choice(c) for c in cand])
            sims[k] = net(np.exp(Lc[ps + hs] - Lc[ps]) - 1, buy, sell).mean()
        res[per] = {'actual_net_mean_pct': round(float(act * 100), 3), 'random_net_mean_pct': round(float(sims.mean() * 100), 3),
                    'random_p95_pct': round(float(np.percentile(sims, 95) * 100), 3), 'pctile_of_actual': round(float((sims < act).mean()), 3),
                    'p_value_one_sided': round(float((sims >= act).mean()), 4)}
    return res

# ---------------- 執行 ----------------
VARS = {'main_h1': (1, False), 'var_h2': (2, False), 'LNY_h1': (1, True), 'LNY_h2': (2, True), 'LNY_h3': (3, True)}
OUT['S1_daily'] = {}; OUT['S2S3_event'] = {}; OUT['S4_nav'] = {}; OUT['S5_random'] = {}; EV = {}
for vn, (h, lo) in VARS.items():
    ev0 = make_events(h, lo)
    OUT['S1_daily'][vn] = s1(ev0)
    for asset in ('0050', 'TAIEX', 'pool', 'pool_open'):
        if asset == 'pool_open' and vn not in ('main_h1', 'var_h2'): continue
        ev, daily = build(ev0, asset); EV[(asset, vn)] = ev
        costs = COST50 if asset in ('0050', 'TAIEX') else COSTP
        OUT['S2S3_event'][f'{asset}|{vn}'] = event_stats(ev, costs)
        if asset in ('0050', 'pool'): OUT['S4_nav'][f'{asset}|{vn}'] = nav_block(daily, costs)
        if asset == '0050':
            for cn in ('etf_base', 'etf_slip0.1'):
                OUT['S5_random'][f'0050|{vn}|{cn}'] = random_ctrl(ev, r50, *COST50[cn])
        print(vn, asset, 'done', flush=True)
OUT['by_relative_day'] = by_rel_day()

# 前視偏誤示範：把颱風空檔也當假日（不能事前知道）
GALL = G.copy(); evt = make_events(1, False, src=GALL); evt, _ = build(evt, '0050')
OUT['S2S3_event']['0050|main_h1|LOOKAHEAD_incl_typhoon'] = event_stats(evt, {'etf_base': COST50['etf_base']})
evty = make_events(1, False, src=TYP); evty, _ = build(evty, '0050')
OUT['typhoon_eve_only_0050_gross'] = {'n': int(evty.gross.notna().sum()), 'mean_pct': pm(evty.gross), 't': tt(evty.gross),
    'detail': {str(r.exit.date()): round(r.gross * 100, 2) for r in evty.itertuples()}}

# 主版本事件明細與子期間、逐年
ev = EV[('0050', 'main_h1')]; bu, se = COST50['etf_base']
ev['net'] = net(ev.gross, bu, se); ev['d50'] = ev.net - ev.expo * ev.e50_cycle; ev['dp'] = ev.net - ev.expo * ev.poolEW_cycle
ev['dg50'] = ev.gross - ev.expo * ev.e50_cycle
OUT['events_0050_main_h1'] = {f"{r.exit.date()}{'(LNY)' if r.LNY else ''}": {'entry': str(r.entry.date()), 'gross_pct': round(r.gross * 100, 3),
    'net_pct': round(r.net * 100, 3), 'listedEW_pct': round(r.listedEW_win * 100, 3) if np.isfinite(r.listedEW_win) else None,
    'expo': round(r.expo, 4) if np.isfinite(r.expo) else None, 'd_vs_0050xExpo_pct': round(r.d50 * 100, 3) if np.isfinite(r.d50) else None}
    for r in ev.itertuples()}
OUT['yearly_0050_main_h1'] = {int(y): {'n': int(len(g)), 'gross_sum_pct': round(float(g.gross.sum() * 100), 2), 'net_sum_pct': round(float(g.net.sum() * 100), 2),
    'd_vs_0050xExpo_sum_pct': round(float(g.d50.sum() * 100), 2), 'd_vs_poolxExpo_sum_pct': round(float(g.dp.sum() * 100), 2),
    'hit_gross': round(float((g.gross > 0).mean()), 2)} for y, g in ev.groupby(ev.entry.dt.year)}
OUT['subperiods_0050_main_h1'] = {f'{a}~{b}': {'n': int(len(x)), 'gross_pct': pm(x.gross), 't_gross': cl(x.gross, x.m.to_numpy())[2],
    'gross_vs_0050xExpo_pct': pm(x.dg50), 't': cl(x.dg50, x.m.to_numpy())[2], 'net_vs_0050xExpo_pct': pm(x.d50), 't_net': cl(x.d50, x.m.to_numpy())[2]}
    for a, b in [('2005-01-01', '2009-12-31'), ('2010-01-01', '2014-12-31'), ('2015-01-01', '2019-04-30'), ('2019-05-01', '2026-10-22')]
    for x in [ev[(ev.entry >= a) & (ev.entry <= b)]]}
# 穩健性（描述）：中位數、截尾、剔除 2008-10～2009-12
def robust(x):
    o = {}
    for nm, col in (('gross', x.gross), ('gross_vs_0050xExpo', x.dg50), ('net_vs_0050xExpo', x.d50), ('net_vs_poolxExpo', x.dp)):
        c = col.dropna(); lo, hi = c.quantile(0.05), c.quantile(0.95); tr = c[(c >= lo) & (c <= hi)]
        o[nm] = {'mean_pct': pm(c), 'median_pct': round(float(c.median() * 100), 3), 'trim5_mean_pct': pm(tr), 't_trim5': tt(tr)}
    return o
OUT['robust_0050_main_h1'] = {per: robust(psel(ev, per)) for per in ('IS', 'OOS')}
_x = psel(ev, 'IS'); _x = _x[(_x.entry < '2008-10-01') | (_x.entry > '2009-12-31')]
OUT['robust_0050_main_h1']['IS_ex_2008Q4_2009'] = {'n': int(len(_x)), 'gross_pct': pm(_x.gross), 't_gross': cl(_x.gross, _x.m.to_numpy())[2],
    'gross_vs_0050xExpo_pct': pm(_x.dg50), 't': cl(_x.dg50, _x.m.to_numpy())[2], 'net_vs_poolxExpo_pct': pm(_x.dp), 't_net_pool': cl(_x.dp, _x.m.to_numpy())[2]}
# 依假日類型（描述）
def htype(r):
    if r.LNY: return 'LNY'
    d = pd.Timestamp(EVH.loc[EVH.T1 == r.q, 'days'].iloc[0][0])
    if d.month == 1 and d.day <= 3 or (d.month == 12 and d.day >= 29): return 'NewYear'
    if d.month in (2, 3) and d.day <= 2 or (d.month == 2 and d.day >= 26): return '228'
    if d.month == 4 and d.day <= 7 or (d.month == 3 and d.day >= 31): return 'Apr4-5'
    if (d.month == 5 and d.day <= 2) or (d.month == 4 and d.day == 30): return 'LaborDay'
    if d.month in (5, 6): return 'DragonBoat'
    if d.month == 10 and 8 <= d.day <= 12: return 'Oct10'
    if (d.month, d.day) in ((9, 29),) and d.year >= 2025: return 'Teacher'
    if d.month == 10 and d.day >= 23: return 'Retrocession'
    if d.month == 12 and d.day == 25: return 'Constitution'
    if d.month in (9, 10): return 'MidAutumn'
    return 'Other'
ev['type'] = [htype(r) for r in ev.itertuples()]
OUT['by_type_0050_main_h1'] = {f'{t}|{per}': {'n': int(len(x)), 'gross_pct': pm(x.gross), 't': tt(x.gross), 'hit': round(float((x.gross > 0).mean()), 2)}
    for t in sorted(ev.type.unique()) for per in ('IS', 'OOS') for x in [psel(ev, per)[psel(ev, per).type == t]] if len(x)}
# 0050 與加權在事件日的差異（資料品質）
dd = pd.Series(r50 - rtx, index=CAL)
qs = ev.q.astype(int).to_numpy()
OUT['data_check'] = {'cal_days': int(NC), 'first': str(CAL[0].date()), 'last': str(CAL[-1].date()),
    'event_days_0050_minus_TAIEX_abs_gt_2pct': {str(CAL[q].date()): round(float(dd.iloc[q]) * 100, 2) for q in qs if abs(dd.iloc[q]) > 0.02},
    'poolEW_daily_abs_max': round(float(np.nanmax(np.abs(poolEW_d))), 4)}
pv = EV[('pool', 'main_h1')]
OUT['pool_main_h1_n_stocks'] = {'mean': round(float(pv.n[pv.n > 0].mean()), 1), 'min': int(pv.n[pv.n > 0].min()),
                                'dropped_events': [str(d.date()) for d in pv.exit[pv.n == 0]]}

# 判定
def vpart(per):
    o = OUT['S2S3_event']['0050|main_h1'][per]['etf_base']
    c1 = o['vs_poolEWxExpo_pct'] > 0 and o['t_vs_poolEWxExpo'] >= 2; c2 = o['vs_0050xExpo_pct'] >= 0
    return {'C1_vs_poolEWxExpo': [o['vs_poolEWxExpo_pct'], o['t_vs_poolEWxExpo'], bool(c1)],
            'C2_vs_0050xExpo': [o['vs_0050xExpo_pct'], o['t_vs_0050xExpo'], bool(c2)], 'pass': bool(c1 and c2)}
vp = {p: vpart(p) for p in ('IS', 'OOS')}
npass = sum(v['pass'] for v in vp.values())
vp['verdict'] = '可用' if npass == 2 else ('待觀察' if npass == 1 else '不可用')
OUT['DECISION'] = vp
(HERE / 'results.json').write_text(json.dumps(OUT, ensure_ascii=False, indent=1, default=str), encoding='utf-8')
print(json.dumps(vp, ensure_ascii=False))
