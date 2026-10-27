#!/usr/bin/env python3
"""w12 月底月初效應：依 results.json 的 PREREG 執行（不改 PREREG）。"""
import json, math, sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'exec_rebuild'))
import exlib as L  # 只讀：cost_rates、FEE、LTOL、面板

RES = json.loads((HERE / 'results.json').read_text(encoding='utf-8'))
assert 'PREREG' in RES
OUT = {'PREREG': RES['PREREG']}

# ---------------- 資料與日曆 ----------------
BAD = pd.Timestamp('2018-01-13')
adj = L.adj.drop(index=BAD); C = L.C.drop(index=BAD); H = L.H.drop(index=BAD); O = L.O.drop(index=BAD); V = L.V.drop(index=BAD)
PIDX = adj.index; COLS = adj.columns; N = len(COLS)
e = pd.read_csv('/home/user/jonatw/finlab-txd/data/raw/etf_0050.csv', parse_dates=['date']).set_index('date')['adj'].sort_index()
e = e.drop(index=[pd.Timestamp('2026-07-10'), pd.Timestamp('2026-09-20')])
gapd = e.index[(e.index >= '2024-01-02') & (e.index <= '2024-02-14')]
CAL = PIDX.union(gapd)
CAL = CAL[(CAL >= '2004-01-02') & (CAL <= '2026-10-22')]
assert not e.reindex(CAL).isna().iloc[100:].all()
E = e.reindex(CAL).ffill()
tx = pd.read_csv('/home/user/jonatw/finlab-txd/data/raw/taiex_twii.csv', parse_dates=['date']).set_index('date')['close'].sort_index()
TX = tx.reindex(CAL).ffill()
NC = len(CAL); cpos = {d: i for i, d in enumerate(CAL)}; inpanel = CAL.isin(PIDX)
ppos = {d: i for i, d in enumerate(PIDX)}

# 面板陣列
AC = adj.ffill().to_numpy(); ADJ = adj.to_numpy(); CN = C.to_numpy(); HN = H.to_numpy()
Rp = adj.pct_change(fill_method=None)
RN = Rp.to_numpy()
fac = adj / C; AOr = (O * fac).to_numpy()
V60 = V.rolling(60, min_periods=50).mean().to_numpy()
thr = np.where(PIDX < pd.Timestamp('2015-06-01'), 0.07, 0.10)[:, None] - L.LTOL
LOCK = (RN >= thr) & (CN >= HN * 0.999)
AC_prev = np.vstack([np.full((1, N), np.nan), AC[:-1]])
GAPUP = (AOr / AC_prev - 1) >= thr

def cash_daily(rate): return (1 + rate) ** (1 / 252) - 1

# ---------------- 事件 ----------------
mon = CAL.to_period('M')
last_pos = pd.Series(np.arange(NC), index=mon).groupby(level=0).max()

def events(a, b):
    rows = []
    for m, lp in last_pos.items():
        p, q = lp - a, lp + b
        if p < 1 or q >= NC: continue
        rows.append((m, p, q))
    ev = pd.DataFrame(rows, columns=['m', 'p', 'q']).set_index('m')
    ev['p_next'] = ev['p'].shift(-1)
    return ev

def universe(d):
    """進場日 d 的股池（前一面板日資料）與可買集合。"""
    i = ppos[d]
    v, px = V60[i - 1], CN[i - 1]
    base = (v >= 1e7) & (px >= 10)
    vv = np.where(base, v, -np.inf); order = np.argsort(-vv)[:300]
    u = np.zeros(N, bool); u[order] = True; u &= base
    return u

def basket(p_cal, q_cal, mode='close', excl_lock=True, which='pool'):
    """回傳 (毛報酬, 每日報酬陣列 len q-p, 檔數)；p,q 為 CAL 位置。任何日不在面板 → None。"""
    if not inpanel[p_cal:q_cal + 1].all(): return None
    d = CAL[p_cal]; i = ppos[d]; j = ppos[CAL[q_cal]]
    if which == 'pool':
        u = universe(d)
    else:
        u = np.ones(N, bool)
    u &= np.isfinite(ADJ[i])
    if which == 'pool' and excl_lock and mode == 'close':
        u &= ~LOCK[i]
    if mode == 'open':          # 進場日隔日開盤買
        u &= np.isfinite(AOr[i + 1]) & ~GAPUP[i + 1]
        base_px = AOr[i + 1]
    else:
        base_px = AC[i]
    idx = np.where(u)[0]
    if len(idx) < 30: return None
    path = AC[i + 1:j + 1, idx] / base_px[idx]          # 每日收盤價值（相對進場）
    vals = np.nanmean(path, axis=1)
    if mode == 'open':
        prev = np.r_[1.0, vals[:-1]]
    else:
        prev = np.r_[1.0, vals[:-1]]
    return float(vals[-1] - 1), vals / prev - 1, len(idx)

def net(g, buy, sell): return (1 - buy) * (1 + g) * (1 - sell) - 1

COST50 = {'zero': (0, 0), 'etf_base': (L.FEE, L.FEE + 0.001), 'etf_slip0.1': (L.FEE + 0.001, L.FEE + 0.001 + 0.001), 'stress': L.cost_rates('stress')}
COSTP = {'zero': (0, 0), 'exlib_slip0.1': L.cost_rates(0.001), 'rt0.5': (0.0025, 0.0025), 'stress_rt1': L.cost_rates('stress')}
OUT['cost_check'] = {k: round((1 - (1 - b) * (1 - s)) * 100, 3) for k, (b, s) in {**COST50, **{'P_' + k: v for k, v in COSTP.items()}}.items()}

PER_M = {'IS': ('2005-01', '2019-04'), 'OOS': ('2019-05', '2026-09'), 'y2024_2025': ('2024-01', '2025-12'),
         'y2021_2023': ('2021-01', '2023-12'), 'y2024_2026': ('2024-01', '2026-09')}
PER_D = {'IS': ('2005-01-01', '2019-04-30'), 'OOS': ('2019-05-01', '2026-10-22'), 'y2024_2025': ('2024-01-01', '2025-12-31'),
         'y2021_2023': ('2021-01-01', '2023-12-31'), 'y2024_2026': ('2024-01-01', '2026-10-22')}

def tt(x):
    x = pd.Series(x).dropna()
    if len(x) < 3 or x.std() == 0: return float('nan')
    return round(float(x.mean() / (x.std(ddof=1) / math.sqrt(len(x)))), 2)

def pm(x): return round(float(pd.Series(x).dropna().mean() * 100), 3)

def msel(df, per):
    a, b = PER_M[per]; return df[(df.index >= pd.Period(a, 'M')) & (df.index <= pd.Period(b, 'M'))]

# ---------------- 每日序列 ----------------
r50 = E.pct_change().to_numpy().copy(); rtx = TX.pct_change().to_numpy().copy()
r50[0] = rtx[0] = np.nan

def build(asset, a, b):
    """回傳 events DataFrame（含 gross、cycle 基準報酬）與 in-window 日報酬 dict。"""
    ev = events(a, b); W = a + b
    g, daily, nst = [], {}, []
    for m, r in ev.iterrows():
        p, q = int(r.p), int(r.q)
        if asset == '0050':
            gg = E.iloc[q] / E.iloc[p] - 1; dl = r50[p + 1:q + 1]; ns = 1
        elif asset == 'TAIEX':
            gg = TX.iloc[q] / TX.iloc[p] - 1; dl = rtx[p + 1:q + 1]; ns = 1
        else:
            mode = 'open' if asset == 'pool_open' else 'close'
            out = basket(p, q, mode=mode, which='pool')
            if out is None: g.append(np.nan); nst.append(0); continue
            gg, dl, ns = out
        g.append(gg); nst.append(ns); daily[m] = (p, q, dl)
    ev['gross'] = g; ev['n'] = nst
    return ev, daily

# 基準：上市等權窗口報酬、同池等權週期報酬、0050 週期報酬
def bench_cols(ev):
    lew, pcyc, c50 = [], [], []
    for m, r in ev.iterrows():
        p, q = int(r.p), int(r.q)
        o = basket(p, q, which='all'); lew.append(o[0] if o else np.nan)
        if np.isfinite(r.p_next):
            pn = int(r.p_next)
            o2 = basket(p, pn, which='pool', excl_lock=False); pcyc.append(o2[0] if o2 else np.nan)
            c50.append(E.iloc[pn] / E.iloc[p] - 1)
        else:
            pcyc.append(np.nan); c50.append(np.nan)
    ev['listedEW_win'] = lew; ev['poolEW_cycle'] = pcyc; ev['e50_cycle'] = c50
    ev['expo'] = (ev.q - ev.p) / (ev.p_next - ev.p)
    return ev

# 同池等權日報酬（每週期固定股池、每日等權）與上市等權日報酬，CAL 上
poolEW_d = np.full(NC, np.nan); listEW_d = np.full(NC, np.nan)
ev_main = events(1, 3)
for m, r in ev_main.iterrows():
    p = int(r.p); pn = int(r.p_next) if np.isfinite(r.p_next) else NC - 1
    if not inpanel[p]: continue
    u = universe(CAL[p]) & np.isfinite(ADJ[ppos[CAL[p]]])
    for t in range(p + 1, pn + 1):
        if not inpanel[t]: continue
        row = RN[ppos[CAL[t]]]
        poolEW_d[t] = np.nanmean(row[u]); listEW_d[t] = np.nanmean(row)
first_p = int(ev_main.p.iloc[0])
for t in range(1, first_p + 1):
    if inpanel[t]: listEW_d[t] = np.nanmean(RN[ppos[CAL[t]]])

DAILY = {'0050': r50, 'TAIEX': rtx, 'poolEW': poolEW_d, 'listedEW': listEW_d}
OUT['data_check'] = {'cal_days': int(NC), 'first': str(CAL[0].date()), 'last': str(CAL[-1].date()),
                     'gap_days_from_0050': int(len(gapd)),
                     'poolEW_daily_abs_max': round(float(np.nanmax(np.abs(poolEW_d))), 4),
                     'listedEW_daily_abs_max': round(float(np.nanmax(np.abs(listEW_d))), 4)}

# ---------------- S1：窗口內外日均 ----------------
def inmask(a, b):
    m = np.zeros(NC, bool)
    for lp in last_pos.values:
        m[max(lp - a + 1, 0):min(lp + b + 1, NC)] = True
    return m

def s1(a, b):
    im = inmask(a, b); res = {}
    ev = events(a, b)
    for per, (da, db) in PER_D.items():
        dm = (CAL >= da) & (CAL <= db)
        for nm, x in DAILY.items():
            ok = dm & np.isfinite(x)
            xi, xo = x[ok & im], x[ok & ~im]
            tw = stats.ttest_ind(xi, xo, equal_var=False).statistic
            # 按進場月份分群
            evp = msel(ev, per); diffs = []
            for m, r in evp.iterrows():
                if not np.isfinite(r.p_next): continue
                p, pn = int(r.p), int(r.p_next); seg = x[p + 1:pn + 1]; W = a + b
                if np.isfinite(seg).sum() < len(seg) - 1: continue
                diffs.append(np.nanmean(seg[:W]) - np.nanmean(seg[W:]))
            res[f'{nm}|{per}'] = {'in_mean_pct': round(float(xi.mean() * 100), 3), 'out_mean_pct': round(float(xo.mean() * 100), 3),
                                  'diff_pct': round(float((xi.mean() - xo.mean()) * 100), 3), 't_welch_daily': round(float(tw), 2),
                                  'n_in': int(len(xi)), 'n_out': int(len(xo)), 'diff_cluster_pct': pm(diffs), 't_cluster_month': tt(diffs), 'n_cycles': len(diffs)}
    return res

# 每個相對日的平均報酬（描述）
def by_day():
    res = {}
    for nm in ('0050', 'poolEW'):
        x = DAILY[nm]
        for per in ('IS', 'OOS'):
            da, db = PER_D[per]; out = {}
            for k in list(range(-6, 0)) + list(range(1, 7)):
                vals = []
                for lp in last_pos.values:
                    t = lp + k + 1 if k < 0 else lp + k
                    if 0 < t < NC and da <= str(CAL[t].date()) <= db: vals.append(x[t])
                v = pd.Series(vals).dropna(); out[k] = {'mean_pct': round(float(v.mean() * 100), 3), 't0': tt(v)}
            v = pd.Series(x[(CAL >= da) & (CAL <= db)]).dropna(); out['all_days_mean_pct'] = round(float(v.mean() * 100), 3)
            res[f'{nm}|{per}'] = out
    return res

# ---------------- S2/S3：事件與同曝險 ----------------
def event_stats(ev, costs):
    res = {}
    for per in PER_M:
        x = msel(ev, per).dropna(subset=['gross'])
        o = {'n_events': int(len(x)), 'gross_mean_pct': pm(x.gross), 't_gross': tt(x.gross), 'hit_gross': round(float((x.gross > 0).mean()), 3),
             'AR_vs_listedEW_gross_pct': pm(x.gross - x.listedEW_win), 't_AR_gross': tt(x.gross - x.listedEW_win),
             'expo_mean': round(float(x.expo.mean()), 4)}
        for cn, (bu, se) in costs.items():
            nt = net(x.gross, bu, se)
            d50 = nt - x.expo * x.e50_cycle; dp = nt - x.expo * x.poolEW_cycle
            o[cn] = {'net_mean_pct': pm(nt), 't_net': tt(nt), 'hit_net': round(float((nt > 0).mean()), 3),
                     'AR_vs_listedEW_pct': pm(nt - x.listedEW_win), 't_AR': tt(nt - x.listedEW_win),
                     'vs_0050xExpo_pct': pm(d50), 't_vs_0050xExpo': tt(d50), 'n_0050': int(d50.notna().sum()),
                     'vs_poolEWxExpo_pct': pm(dp), 't_vs_poolEWxExpo': tt(dp), 'n_pool': int(dp.notna().sum())}
        res[per] = o
    return res

# ---------------- S4：日頻 NAV ----------------
def nav_daily(ev, daily, buy, sell, cash_rate):
    c = cash_daily(cash_rate); s = np.full(NC, c); inm = np.zeros(NC, bool)
    for m, (p, q, dl) in daily.items():
        dl = np.nan_to_num(np.asarray(dl, float))
        s[p] = (1 + s[p]) * (1 - buy) - 1
        s[p + 1:q + 1] = dl; inm[p + 1:q + 1] = True
        s[q] = (1 + s[q]) * (1 - sell) - 1
    return s, inm

def perf(s, da, db, inm=None):
    dm = (CAL >= da) & (CAL <= db); x = pd.Series(s[dm], index=CAL[dm]).fillna(0)
    eq = (1 + x).cumprod(); yrs = (x.index[-1] - x.index[0]).days / 365.25
    o = {'CAGR': round(float(eq.iloc[-1] ** (1 / yrs) - 1), 4), 'maxDD': round(float((eq / eq.cummax() - 1).min()), 4),
         'vol': round(float(x.std() * math.sqrt(252)), 4)}
    if inm is not None: o['exposure'] = round(float(inm[dm].mean()), 4)
    o['yearly_pct'] = {int(y): round(float((1 + g).prod() - 1) * 100, 1) for y, g in x.groupby(x.index.year)}
    return o

def nav_block(ev, daily, costs, bench_daily, cash_rates=(0.0, 0.01)):
    res = {}
    for per, (da, db) in PER_D.items():
        o = {}
        for cr in cash_rates:
            for cn, (bu, se) in costs.items():
                s, inm = nav_daily(ev, daily, bu, se, cr)
                o[f'{cn}|cash{cr}'] = perf(s, da, db, inm)
        s0, inm = nav_daily(ev, daily, 0, 0, 0.0)
        dm = (CAL >= da) & (CAL <= db); ex = float(inm[dm].mean())
        for bn, bd in bench_daily.items():
            o[f'BH_{bn}'] = perf(np.nan_to_num(bd), da, db)
            o[f'{bn}_x_expo_cash0'] = perf(np.nan_to_num(bd) * ex, da, db)
            o[f'{bn}_x_expo_cash0.01'] = perf(np.nan_to_num(bd) * ex + (1 - ex) * cash_daily(0.01), da, db)
        res[per] = o
    return res

# ---------------- S5：隨機同曝險 ----------------
def random_ctrl(series_daily, a, b, buy, sell, nsim=2000, seed=12):
    rng = np.random.default_rng(seed); W = a + b
    Lc = np.r_[0.0, np.cumsum(np.log1p(np.nan_to_num(series_daily[1:])))]   # Lc[t] = log 累積到 t
    res = {}
    for per in ('IS', 'OOS', 'y2024_2025'):
        a_, b_ = PER_M[per]
        months = [m for m in last_pos.index if pd.Period(a_, 'M') <= m <= pd.Period(b_, 'M')]
        cand = [np.where(mon == m)[0] for m in months]
        cand = [c[c + W < NC] for c in cand]
        act = []
        for m in months:
            lp = last_pos[m]; p = lp - a
            act.append(net(math.exp(Lc[p + W] - Lc[p]) - 1, buy, sell))
        act = np.array(act)
        sims = np.empty(nsim)
        for k in range(nsim):
            ps = np.array([rng.choice(c) for c in cand])
            g = np.exp(Lc[ps + W] - Lc[ps]) - 1
            sims[k] = net(g, buy, sell).mean()
        res[per] = {'actual_net_mean_pct': round(float(act.mean() * 100), 3), 'random_net_mean_pct': round(float(sims.mean() * 100), 3),
                    'random_p95_pct': round(float(np.percentile(sims, 95) * 100), 3), 'pctile_of_actual': round(float((sims < act.mean()).mean()), 3),
                    'p_value_one_sided': round(float((sims >= act.mean()).mean()), 4)}
    return res

# ---------------- 執行 ----------------
WINS = {'m1p3': (1, 3), 'm1p2': (1, 2), 'm2p3': (2, 3)}
OUT['S1_daily_in_vs_out'] = {w: s1(*ab) for w, ab in WINS.items()}
OUT['by_relative_day'] = by_day()
OUT['S2S3_event'] = {}; OUT['S4_nav'] = {}; OUT['S5_random'] = {}
EV = {}
for w, (a, b) in WINS.items():
    for asset in ('0050', 'TAIEX', 'pool', 'pool_open'):
        if asset in ('TAIEX', 'pool_open') and w != 'm1p3': continue
        ev, daily = build(asset, a, b); ev = bench_cols(ev); EV[(asset, w)] = ev
        costs = COST50 if asset in ('0050', 'TAIEX') else COSTP
        OUT['S2S3_event'][f'{asset}|{w}'] = event_stats(ev, costs)
        bd = {'0050': r50, 'poolEW': poolEW_d}
        if asset != 'pool_open':
            OUT['S4_nav'][f'{asset}|{w}'] = nav_block(ev, daily, costs, bd)
        if asset in ('0050', 'TAIEX'):
            ser = r50 if asset == '0050' else rtx
            bu, se = costs['etf_slip0.1']
            OUT['S5_random'][f'{asset}|{w}'] = random_ctrl(ser, a, b, bu, se)
        print(asset, w, 'done', flush=True)
# 剔除 2026-07-31 事件
ev = EV[('0050', 'm1p3')].drop(index=pd.Period('2026-07', 'M'))
OUT['S2S3_event']['0050|m1p3|ex_2026-07'] = event_stats(ev, {'etf_slip0.1': COST50['etf_slip0.1']})

# 事件明細（主版本）
ev = EV[('0050', 'm1p3')]; bu, se = COST50['etf_slip0.1']
det = pd.DataFrame({'entry': [str(CAL[int(p)].date()) for p in ev.p], 'exit': [str(CAL[int(q)].date()) for q in ev.q],
                    'gross_pct': (ev.gross * 100).round(2), 'net_pct': (net(ev.gross, bu, se) * 100).round(2),
                    'listedEW_win_pct': (ev.listedEW_win * 100).round(2), 'expo': ev.expo.round(3),
                    'e50_cycle_pct': (ev.e50_cycle * 100).round(2), 'poolEW_cycle_pct': (ev.poolEW_cycle * 100).round(2)}, index=ev.index.astype(str))
OUT['events_0050_m1p3'] = det.to_dict(orient='index')

# 描述：主版本在不同子期間的同曝險超額（事後描述，不影響判定）
_d = det.copy(); _d.index = ev.index
_d['dg'] = ev.gross - ev.expo * ev.e50_cycle; _d['dn'] = net(ev.gross, bu, se) - ev.expo * ev.e50_cycle
OUT['descriptive_subperiods_0050_m1p3'] = {f'{a}~{b}': {'n': int(len(x)), 'gross_pct': pm(x.gross_pct / 100), 't_gross': tt(x.gross_pct),
    'vs_0050xExpo_gross_pct': pm(x.dg), 't_gross_vs_0050xExpo': tt(x.dg), 'vs_0050xExpo_net_pct': pm(x.dn), 't_net_vs_0050xExpo': tt(x.dn)}
    for a, b in [('2005-01', '2009-12'), ('2010-01', '2014-12'), ('2015-01', '2019-04'), ('2019-05', '2026-09')]
    for x in [_d[(_d.index >= pd.Period(a, 'M')) & (_d.index <= pd.Period(b, 'M'))]]}
pv = EV[('pool', 'm1p3')]
OUT['pool_m1p3_n_stocks'] = {'mean': round(float(pv.n[pv.n > 0].mean()), 1), 'min': int(pv.n[pv.n > 0].min()), 'dropped_events': [str(m) for m in pv.index[pv.n == 0]]}

# 判定
def verdict_part(per):
    o = OUT['S2S3_event']['0050|m1p3'][per]['etf_slip0.1']
    c1 = o['vs_poolEWxExpo_pct'] > 0 and o['t_vs_poolEWxExpo'] >= 2
    c2 = o['vs_0050xExpo_pct'] >= 0
    return {'C1_vs_poolEWxExpo': [o['vs_poolEWxExpo_pct'], o['t_vs_poolEWxExpo'], bool(c1)], 'C2_vs_0050xExpo': [o['vs_0050xExpo_pct'], o['t_vs_0050xExpo'], bool(c2)], 'pass': bool(c1 and c2)}
vp = {p: verdict_part(p) for p in ('IS', 'OOS')}
npass = sum(v['pass'] for v in vp.values())
vp['verdict'] = '可用' if npass == 2 else ('待觀察' if npass == 1 else '不可用')
OUT['DECISION'] = vp
(HERE / 'results.json').write_text(json.dumps(OUT, ensure_ascii=False, indent=1, default=str), encoding='utf-8')
print(json.dumps(vp, ensure_ascii=False))
