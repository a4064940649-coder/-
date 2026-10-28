#!/usr/bin/env python3
"""o4 V4 獨立複製：核心（只依凍結規格與 BRIEF 的資料從頭寫）。"""
import json, math
from pathlib import Path
import numpy as np, pandas as pd

AG = Path('/home/user/-/research/tw_stock_selection/agents')
O4 = AG / 'o_v4_revenue_patch_official'
PNL = Path('/tmp/claude-0/-home-user--/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/panel')
EXT = Path('/home/user/stevenhsiehsheng-ctrl/tw-stock-screener/data/extras')
END = pd.Timestamp('2026-10-12')
A0 = pd.Timestamp('2006-04-01')
SWITCH = pd.Timestamp('2021-03-01')

# ---------------- 面板＋P ----------------
F = {k: pd.read_parquet(PNL / f'{k}.parquet') for k in ['close_raw', 'open', 'high', 'value', 'ret', 'flag']}
C0 = F['close_raw']
IDX, COLS = C0.index, C0.columns
N = len(COLS); CI = {c: j for j, c in enumerate(COLS)}
P = pd.read_parquet(AG / 'o_infra_v3_rebuild/delist_prices.parquet')
P['date'] = pd.to_datetime(P['date'])
def _fill(df, col):
    w = P.pivot(index='date', columns='code', values=col).reindex(index=IDX, columns=COLS)
    return df.where(df.notna(), w)
C = _fill(F['close_raw'], 'close'); O = _fill(F['open'], 'open'); H = _fill(F['high'], 'high')
V = _fill(F['value'], 'value'); R = _fill(F['ret'], 'ret')
pclose = P.pivot(index='date', columns='code', values='close').reindex(index=IDX, columns=COLS)
filled = F['close_raw'].isna() & pclose.notna()
FL = F['flag'].where(~filled, 0)
Cn, On, Vn, Rn, FLn = C.to_numpy(), O.to_numpy(), V.to_numpy(), R.to_numpy(), FL.to_numpy()
TRADED = np.isfinite(Cn)
GROW = np.where(TRADED, 1.0 + np.nan_to_num(Rn, nan=0.0), 1.0)
MYADJ = np.where(TRADED, np.cumprod(GROW, axis=0), np.nan)
NOBS = TRADED.sum(1); GOOD = NOBS >= 300
PREVC = C.ffill().shift(1).to_numpy()
TPOS = {d: i for i, d in enumerate(IDX)}
PANEL_END = IDX[-1]

# 0050
E = pd.read_csv('/home/user/jonatw/finlab-txd/data/raw/etf_0050.csv', parse_dates=['date']).set_index('date')['adj'].sort_index()
E50 = E.reindex(IDX.union(E.index)).ffill().reindex(IDX).to_numpy()

# ---------------- 滾動窗 gapfix ----------------
def rolling():
    g = np.where(GOOD)[0]; dg = IDX[g]
    A = MYADJ[g]; Vv = Vn[g]
    ex = dg.isin(pd.DatetimeIndex(['2017-08-03', '2018-07-12'])) | ((dg >= '2024-02-15') & (dg <= '2024-03-15'))
    mA = pd.DataFrame((np.isnan(A) & ~ex[:, None]).astype(np.int16)).rolling(252, min_periods=252).sum().to_numpy() == 0
    mV = pd.DataFrame((np.isnan(Vv) & ~ex[:, None]).astype(np.int16)).rolling(60, min_periods=60).sum().to_numpy() == 0
    hi = pd.DataFrame(A).rolling(252, min_periods=1).max().to_numpy()
    v60 = pd.DataFrame(Vv).rolling(60, min_periods=1).mean().to_numpy()
    with np.errstate(invalid='ignore', divide='ignore'):
        s = np.where(mA, A / hi, np.nan)
    S52 = np.full((len(IDX), N), np.nan); V60 = np.full((len(IDX), N), np.nan)
    S52[g] = s; V60[g] = np.where(mV, v60, np.nan)
    return S52, V60
S52, V60 = rolling()

# ---------------- 名單 ----------------
SL = pd.read_csv(EXT.parent / 'stock_list.csv', dtype=str).drop_duplicates('code').set_index('code')
FIN = set(SL.index[SL['industry'] == '金融保險業']) | {c for c in COLS if c[:2] == '28' or c in ('5876', '5880')}
DO = pd.read_csv(AG / 'o_infra_v3_rebuild/delist_official.csv', dtype=str)
DO['date'] = pd.to_datetime(DO['date'])
FINKW = ['銀', '證', '金控', '證金', '產保', '壽', '票券', '產險', '人壽', '投信', '合庫', '信託']
def isfinname(nm): return any(k in nm for k in FINKW)
FIN2 = {r.code for r in DO.itertuples() if r.code in CI and r.in_panel == 'True' and isfinname(r.name) and r.code not in FIN}
fz = json.loads((AG / 'o_rev_survivorship/data_freeze.json').read_text())
FOREIGN = set(fz['foreign_codes']) | {r.code for r in DO.itertuples() if r.name.startswith('F-') or 'KY' in r.name}
FINMASK = np.array([c in FIN for c in COLS]); FIN2MASK = np.array([c in FIN2 for c in COLS])
FORMASK = np.array([c in FOREIGN for c in COLS])

# gap 股
def _gapexcl(r):
    nm = r.name
    return ('KY' in nm or 'DR' in nm or nm.startswith('F-') or r.code[:2] == '91' or r.code in FIN or isfinname(nm))
GAP = {}
for r in DO.itertuples():
    if r.four == 'True' and r.code in CI and not _gapexcl(r):
        GAP[r.code] = max(GAP.get(r.code, pd.Timestamp('1900-01-01')), r.date)
GAPEND = np.full(N, np.datetime64('NaT'), dtype='datetime64[ns]')
for c, d in GAP.items(): GAPEND[CI[c]] = np.datetime64(d)

# ---------------- 處置 ----------------
dh = pd.read_csv(AG / 'o_disposal_rejudge/disposal_hist_rejudge.csv', dtype=str)
dh = dh[dh['in_panel'] == 'True'][['announce', 'code', 'start', 'end']]
d5 = pd.read_csv(EXT / 'disposal_5y.csv.gz', dtype=str).rename(columns={'date': 'announce'})
k = set(zip(dh.code, dh.start))
d5 = d5[[(c, s) not in k for c, s in zip(d5.code, d5.start)]][['announce', 'code', 'start', 'end']]
DISP = pd.concat([dh, d5], ignore_index=True)
for c in ['announce', 'start', 'end']: DISP[c] = pd.to_datetime(DISP[c])
DISP = DISP[DISP.code.isin(CI)]
DISP_J = DISP.code.map(CI).to_numpy(); DISP_A = DISP.announce.to_numpy(); DISP_S = DISP.start.to_numpy(); DISP_E = DISP.end.to_numpy()
def disposal_mask(d, x):
    """公告日 ≤ d 且 start ≤ x ≤ end。"""
    d = np.datetime64(d); x = np.datetime64(x)
    m = (DISP_A <= d) & (DISP_S <= x) & (x <= DISP_E)
    out = np.zeros(N, bool); out[DISP_J[m]] = True
    return out

# ---------------- 營收 ----------------
def load_y(path):
    y = pd.read_parquet(path)
    y.index = pd.PeriodIndex(y.index, freq='M')
    return y.reindex(columns=COLS)
YF = load_y(O4 / 'yoy_long_R_FIX.parquet')
YD = load_y(AG / 'o_rev_survivorship/yoy_long_R_DEL.parquet')
Y5 = load_y(O4 / 'yoy_rev5y_FIX.parquet')
YQ = load_y(O4 / 'yoy_qproxy.parquet')
def lastfull(Y):
    raw = pd.read_parquet(Y) if isinstance(Y, (str, Path)) else Y
    cnt = raw.notna().sum(1)
    return cnt[cnt >= 0.8 * cnt.median()].index.max()
_LF = {}
def lf_of(path):
    if path not in _LF:
        y = pd.read_parquet(path); y.index = pd.PeriodIndex(y.index, freq='M'); _LF[path] = lastfull(y)
    return _LF[path]
LF_L = {'F': lf_of(O4 / 'yoy_long_R_FIX.parquet'), 'D': lf_of(AG / 'o_rev_survivorship/yoy_long_R_DEL.parquet')}
LF_5 = lf_of(O4 / 'yoy_rev5y_FIX.parquet')

def y3(Y, m, back=0):
    ks = [m - 1 - back, m - 2 - back, m - 3 - back]
    if any(k not in Y.index for k in ks): return np.full(N, np.nan)
    a = Y.loc[ks].to_numpy()
    return np.where(np.isfinite(a).all(0), a.mean(0), np.nan)

# ---------------- 換股日 ----------------
def signal_dates(longkey='F'):
    out = []; seen = set()
    for i, d in enumerate(IDX):
        if d < pd.Timestamp('2005-01-01') or d >= END: continue
        if d.day < 11: continue
        key = (d.year, d.month)
        if key in seen: continue
        seen.add(key)
        m = pd.Period(d, 'M')
        long = d < SWITCH
        Y = (YF if longkey == 'F' else YD) if long else Y5
        lf = LF_L[longkey] if long else LF_5
        if m - 1 > lf: continue
        if np.isfinite(y3(Y, m)).sum() <= 100: continue
        out.append((i, i + 1))
    return out

# ---------------- 漲停 ----------------
def tick(p):
    return np.select([p < 10, p < 50, p < 100, p < 500, p < 1000], [0.01, 0.05, 0.1, 0.5, 1.0], 5.0)
def limit_up(ie, j):
    o = On[ie, j]; c = Cn[ie, j]; r = Rn[ie, j]
    if FLn[ie, j] == 1 or not np.isfinite(r): ref = PREVC[ie, j]
    else: ref = c / (1 + r)
    if not np.isfinite(ref): return False
    lim = 0.07 if IDX[ie] < pd.Timestamp('2015-06-01') else 0.10
    x = ref * (1 + lim); t = tick(x)
    lp = math.floor(x / t + 1e-9) * t
    return o >= lp - 1e-6

# ---------------- 選股計畫 ----------------
def build_plan(rev='F', disp='trade', fin2=True, forex=True, unavail=None, bound=None, tie='official', seed=None, n=10, dates=None, info=False):
    """回傳 list of dict(i_d, i_e, picks, pool)。
    rev: 'F'＝R_FIX、'D'＝R_DEL、'NO2013'＝傳入的 YF 改版（見 make_no2013）或 DataFrame。
    bound: None／'B05'／'B10'／'BQ'。"""
    Ylong = {'F': YF, 'D': YD}.get(rev, rev) if isinstance(rev, str) else rev
    rng = np.random.default_rng(seed) if tie == 'rand' else None
    plan = []; held = set(); stats = []
    for (i_d, i_e) in (dates or signal_dates('F' if not isinstance(rev, str) else ('D' if rev == 'D' else 'F'))):
        d, e = IDX[i_d], IDX[i_e]; m = pd.Period(d, 'M')
        Y = Ylong if d < SWITCH else Y5
        yv = y3(Y, m)
        U = unavail(d) if unavail is not None else None
        if U is not None:
            yb = y3(Y, m, back=1)
            yv = np.where(U, yb, yv)
        v = V60[i_d]; c = Cn[i_d]
        base = np.isfinite(v) & (v >= 1e7) & np.isfinite(c) & (c >= 10)
        bi = np.where(base)[0]
        bi = bi[np.argsort(-v[bi], kind='stable')][:300]
        top = np.zeros(N, bool); top[bi] = True
        ok = top & np.isfinite(S52[i_d]) & np.isfinite(On[i_e]) & np.isfinite(Cn[i_e])
        ok &= ~FINMASK
        if fin2: ok &= ~FIN2MASK
        if forex: ok &= ~FORMASK
        dm = disposal_mask(d, e if disp == 'trade' else d)
        ok &= ~dm
        nunav = int((U & top).sum()) if U is not None else 0
        gapadd = np.zeros(N, bool)
        if bound is not None:
            ge = GAPEND
            isgap = ~np.isnat(ge) & (ge >= np.datetime64(d))
            gapadd = ok & ~np.isfinite(yv) & isgap
            if bound == 'BQ':
                q = y3(YQ, m) if m - 4 >= YQ.index[0] else np.full(N, np.nan)
                yv = np.where(gapadd & np.isfinite(q), q, yv)
                gapadd = np.zeros(N, bool)
        pool = ok & (np.isfinite(yv) | gapadd)
        idx = np.where(pool)[0]
        if len(idx) < 50:
            stats.append((d, len(idx), 'skip')); continue
        s = pd.Series(S52[i_d, idx]).rank(pct=True).to_numpy()
        yy = yv[idx]; g = gapadd[idx]
        py = np.full(len(idx), np.nan)
        if g.any():
            py[~g] = pd.Series(yy[~g]).rank(pct=True).to_numpy()
            py[g] = 0.5 if bound == 'B05' else 1.0
            ytie = yy.copy(); ytie[g] = (np.inf if bound == 'B10' else -np.inf)
            if bound == 'B05': ytie[g] = np.nan
        else:
            py = pd.Series(yy).rank(pct=True).to_numpy(); ytie = yy
        sc = np.round(0.5 * s + 0.5 * py, 10)
        yt = np.where(np.isfinite(ytie) | np.isinf(ytie), ytie, -np.inf)
        if tie == 'rand':   # 隨機同分：分數同分時完全隨機（不再看 y3）
            perm = np.random.default_rng(int(seed) * 100003 + i_d).permutation(N)[idx]
            order = np.lexsort((perm, -sc))
        else:
            order = np.lexsort((idx, -yt, -sc))
        picks = []
        for k in order:
            j = idx[k]
            if j not in held and limit_up(i_e, j): continue
            picks.append(j)
            if len(picks) == n: break
        rec = dict(i_d=i_d, i_e=i_e, picks=picks, pool=idx, nunav=nunav, gap=list(idx[g]) if g.any() else [])
        if info: rec['score'] = dict(zip(idx, sc)); rec['order'] = idx[order]
        plan.append(rec); held = set(picks)
    return plan

# ---------------- X ----------------
MC = pd.read_csv(AG / 'o_infra_v3_rebuild/merger_consideration.csv', dtype=str)
LASTT = np.full(N, -1)
for j in range(N):
    w = np.where(TRADED[:, j])[0]
    if len(w): LASTT[j] = w[-1]
# 代碼重用：舊證券終止日前最後一筆、到新證券第一筆
REUSE = {}
for r in DO[DO.in_panel == 'True'].itertuples():
    if r.code not in CI: continue
    j = CI[r.code]; w = np.where(TRADED[:, j])[0]
    bef = w[IDX[w] < r.date]; aft = w[IDX[w] >= r.date]
    if len(bef) and len(aft): REUSE[j] = (bef[-1], aft[0], r.date)
DEL_DATE = {}
for r in DO[DO.in_panel == 'True'].itertuples():
    if r.code in CI: DEL_DATE.setdefault(CI[r.code], []).append(r.date)
XINFO = {}
for r in MC.itertuples():
    if r.code in CI:
        XINFO.setdefault(CI[r.code], []).append(r)

def dead_info(j, ie):
    """若持股 j 在 ie 已死，回傳 (last_idx, delist_date_or_None)；否則 None。"""
    if j in REUSE:
        lb, fa, dd = REUSE[j]
        if lb < ie < fa: return (lb, dd)
        if ie >= fa: return None
    lt = LASTT[j]
    if lt < ie and IDX[lt] < PANEL_END - pd.Timedelta(days=30):
        dds = [x for x in DEL_DATE.get(j, []) if x >= IDX[lt] - pd.Timedelta(days=60)]
        return (lt, min(dds) if dds else None)
    return None

def x_action(j, ie):
    """回傳 ('hold'|'cash'|'zero', mult)。"""
    di = dead_info(j, ie)
    if di is None: return None
    lt, dd = di; e = IDX[ie]
    recs = [r for r in XINFO.get(j, []) if abs((pd.Timestamp(r.last_trade) - IDX[lt]).days) <= 40] if j in XINFO else []
    if recs:
        r = recs[0]; typ = r.type
        ddate = pd.Timestamp(r.delist_date)
        if typ == 'cash' and isinstance(r.cash_over_last_close, str) and r.cash_over_last_close:
            pay = pd.Timestamp(r.pay_date) if isinstance(r.pay_date, str) and r.pay_date else ddate + pd.Timedelta(days=14)
            return ('cash', float(r.cash_over_last_close)) if e >= pay else ('hold', 1.0)  # 付款日當天即可（B10 對照確認）
        if typ in ('swap', 'cash'):
            return ('cash', 1.0) if e > ddate else ('hold', 1.0)
        if typ == 'failure':
            return ('zero', 0.0)
    # unknown
    return ('cash', 1.0) if e > IDX[lt] + pd.Timedelta(days=14) else ('hold', 1.0)

# ---------------- 引擎 ----------------
FEE = 0.001425 * 0.6
def rates(kind='normal', slip=0.001, fee=FEE):
    if kind == 'stress': return 0.005, 0.005
    return fee + slip, fee + 0.003 + slip

def run(plan, buy, sell, X=True, ew=False, end=END):
    """回傳日淨值 Series（IDX 到 end），以及統計。"""
    iend = TPOS[end]
    reb = {p['i_e']: p for p in plan}
    h = np.zeros(N); cash = 1.0
    nav = np.full(iend + 1, np.nan)
    first = plan[0]['i_e']
    nav[:first] = 1.0
    nx = 0; nfrz = 0; turn = []
    for t in range(first, iend + 1):
        if t in reb:
            p = reb[t]
            tradable = np.isfinite(On[t]) & np.isfinite(Cn[t]) & (On[t] > 0)
            f1 = np.where(tradable, GROW[t] * On[t] / np.where(tradable, Cn[t], 1), GROW[t])
            h = h * f1
            held = np.where(h > 0)[0]
            frozen = np.zeros(N, bool)
            for j in held:
                if tradable[j]: continue
                if X and not ew:
                    a = x_action(j, t)
                    if a is not None:
                        if a[0] == 'cash': cash += h[j] * a[1]; h[j] = 0; nx += 1; continue
                        if a[0] == 'zero': h[j] = 0; nx += 1; continue
                frozen[j] = True
            nfrz += int(frozen.sum())
            tgt = np.zeros(N, bool)
            pk = list(p['pool']) if ew else p['picks']
            pk = [j for j in pk if tradable[j]]
            tgt[pk] = True
            free = ~frozen
            W = cash + h[free].sum()
            cur = np.where(free, h, 0.0)
            n = len(pk)
            if n > 0:
                lo, hi = 0.0, W / n
                for _ in range(60):
                    T = 0.5 * (lo + hi)
                    newv = np.where(tgt, T, 0.0)
                    dlt = (newv - cur)[free]
                    cost = buy * dlt[dlt > 0].sum() - sell * dlt[dlt < 0].sum()
                    if n * T + cost <= W: lo = T
                    else: hi = T
                T = lo
                newv = np.where(tgt, T, 0.0)
                dlt = (newv - cur)[free]
                cost = buy * dlt[dlt > 0].sum() - sell * dlt[dlt < 0].sum()
                turn.append(dlt[dlt > 0].sum() / (W + h[frozen].sum()))
                h = np.where(free, newv, h)
                cash = W - n * T - cost
            f2 = np.where(tradable, Cn[t] / np.where(tradable, On[t], 1), 1.0)
            h = h * f2
        else:
            h = h * GROW[t]
        nav[t] = cash + h.sum()
    s = pd.Series(nav, index=IDX[:iend + 1])
    return s, dict(nx=nx, nfrz=nfrz, turn=float(np.mean(turn)) if turn else 0.0)

# ---------------- 統計 ----------------
def cagr(nav, a=A0, b=END):
    s = nav[(nav.index >= a) & (nav.index <= b)]
    i0 = nav.index.get_loc(s.index[0])
    base = nav.iloc[i0 - 1]; bd = nav.index[i0 - 1]
    yrs = (s.index[-1] - bd).days / 365.25
    return (s.iloc[-1] / base) ** (1 / yrs) - 1

def mdd(nav, a=A0, b=END):
    s = nav[(nav.index >= a) & (nav.index <= b)]
    i0 = nav.index.get_loc(s.index[0]); s = nav.iloc[i0 - 1:nav.index.get_loc(s.index[-1]) + 1]
    return float((s / s.cummax() - 1).min())

def windows(plan, end=END):
    ds = [p['i_d'] for p in plan]
    return ds, ds[1:] + [TPOS[end]]

def win_ret(nav, plan, a=A0, b=END, bench=None):
    ds, de = windows(plan)
    out = []
    for i0, i1 in zip(ds, de):
        if not (a <= IDX[i0] <= b): continue
        r = nav.iloc[i1] / nav.iloc[i0] - 1
        bb = (E50[i1] / E50[i0] - 1) if bench is None else (bench.iloc[i1] / bench.iloc[i0] - 1)
        out.append((IDX[i0], r - bb))
    return pd.Series(dict(out))

def tstat(x):
    x = np.asarray(x); return x.mean() / (x.std(ddof=1) / math.sqrt(len(x)))

def nw_t(x, L):
    x = np.asarray(x, float); n = len(x); mu = x.mean(); z = x - mu
    s = (z * z).sum() / n
    for l in range(1, L + 1):
        s += 2 * (1 - l / (L + 1)) * (z[l:] * z[:-l]).sum() / n
    return mu / math.sqrt(s / n)

def boot_p(x, B=10000, blk=12, seed=0):
    """12 個月區塊 bootstrap（循環），H0 平均＝0：去均值後抽樣，單尾 p。"""
    x = np.asarray(x, float); n = len(x); rng = np.random.default_rng(seed)
    t0 = tstat(x); z = x - x.mean(); nb = int(math.ceil(n / blk)); cnt = 0
    for _ in range(B):
        st = rng.integers(0, n, nb)
        ii = (st[:, None] + np.arange(blk)[None, :]).ravel()[:n] % n
        if tstat(z[ii]) >= t0: cnt += 1
    return (cnt + 1) / (B + 1)

# ---------------- 報告 ----------------
SEGS = {'S1': ('2006-04-01', '2013-12-31'), 'S2': ('2014-01-01', '2020-12-31'), 'S3': ('2021-03-01', '2026-10-12'),
        'S4': ('2024-01-01', '2026-10-12'), 'SEL': ('2006-04-01', '2018-12-31'), 'VAL': ('2019-01-01', '2023-12-31')}
E50S = pd.Series(E50, index=IDX)

def cagr_d(nav, plan, a, b):
    """以「區間內第一個訊號日收盤」為基期，到 b 的最後一日。"""
    a = pd.Timestamp(a); b = pd.Timestamp(b)
    ds = [IDX[p['i_d']] for p in plan if IDX[p['i_d']] >= a]
    a0 = ds[0]; s = nav[(nav.index <= b)]; b0 = s.index[-1]
    return (nav[b0] / nav[a0]) ** (365.25 / (b0 - a0).days) - 1

def mdd_d(nav, plan):
    a0 = [IDX[p['i_d']] for p in plan if IDX[p['i_d']] >= A0][0]
    s = nav[(nav.index >= a0) & (nav.index <= END)]
    return float((s / s.cummax() - 1).min())

def summary(nav, plan, ew=None, full=False):
    w = win_ret(nav, plan)
    out = dict(cagr=cagr_d(nav, plan, A0, END), t=tstat(w), n=len(w), mean=w.mean() * 100, sd=w.std() * 100, mdd=mdd_d(nav, plan))
    if ew is not None: out['t_ew'] = tstat(win_ret(nav, plan, bench=ew))
    if not full: return out
    seg = {}
    for k, (a, b) in SEGS.items():
        ww = win_ret(nav, plan, pd.Timestamp(a), pd.Timestamp(b))
        seg[k] = (cagr_d(nav, plan, a, b), tstat(ww), cagr_d(E50S, plan, a, b), len(ww))
    out['seg'] = seg
    out['nw'] = [nw_t(w.values, L) for L in (3, 6, 12)]
    from math import erf, sqrt
    out['p1'] = 0.5 * (1 - erf(out['t'] / sqrt(2)))
    out['bonf'] = 10 * out['p1']
    v = w.values
    out['drop_last'] = tstat(v[:-1])
    srt = np.argsort(-v)
    out['drop_best'] = [tstat(np.delete(v, srt[:k])) for k in (1, 3, 5)]
    out['best'] = (w.index[srt[0]], v[srt[0]] * 100)
    out['drop2021'] = tstat(w[(w.index.year != 2021)].values)
    out['to2025'] = tstat(w[w.index <= '2025-12-31'].values)
    # 日曆月
    me = nav[(nav.index <= END)].groupby(nav[(nav.index <= END)].index.to_period('M')).tail(1)
    e5 = E50S.reindex(me.index)
    a0 = [IDX[p['i_d']] for p in plan if IDX[p['i_d']] >= A0][0]
    cm = (me.pct_change() - e5.pct_change())
    out['cal_t_from_apr'] = tstat(cm[cm.index > a0].dropna().values)
    out['cal_t_from_may'] = tstat(cm[cm.index >= '2006-05-01'].dropna().values)
    # 逐年
    yr = {}
    for y in range(2006, 2027):
        s = nav[(nav.index.year == y) & (nav.index <= END)]; e = E50S[s.index]
        if y == 2006:
            s = s[s.index >= a0]; e = E50S[s.index]
            r, rb = s.iloc[-1] / s.iloc[0] - 1, e.iloc[-1] / e.iloc[0] - 1
        else:
            p = nav[nav.index < s.index[0]].index[-1]
            r, rb = s.iloc[-1] / nav[p] - 1, e.iloc[-1] / E50S[p] - 1
        yr[y] = (r, rb)
    out['lose_years'] = [y for y, (r, rb) in yr.items() if r < rb]
    out['yearly'] = yr
    out['boot_p'] = boot_p(v)
    return out
