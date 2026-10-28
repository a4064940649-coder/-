#!/usr/bin/env python3
"""o5 V5 獨立複製：核心。只依凍結規格（frozen_spec.txt）與 data_freeze_v5.json 列出的資料從頭寫；
不 import 研究中心任何程式（AB 抽樣規則依規格所指 v2_core.make_unavail／st_core.unavail_for 的定義自行重寫）。"""
import json, math, hashlib
from pathlib import Path
import numpy as np, pandas as pd

AG = Path('/home/user/-/research/tw_stock_selection/agents')
O5 = AG / 'o_v5_pit_revenue'
O4 = AG / 'o_v4_revenue_patch_official'
PNL = Path('/tmp/claude-0/-home-user--/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/panel')
EXT = Path('/home/user/stevenhsiehsheng-ctrl/tw-stock-screener/data/extras')
TAIEX = Path('/home/user/jonatw/finlab-txd/data/raw/taiex_twii.csv')
END = pd.Timestamp('2026-10-12')
SWITCH = pd.Timestamp('2021-03-01')

# ---------------- sha 檢查 ----------------
FZ = json.loads((O5 / 'data_freeze_v5.json').read_text())
def check_freeze():
    bad = []
    for k, v in FZ['files'].items():
        if k.startswith('script/') or k.startswith('topic/'): continue
        p = Path(v['path'])
        if hashlib.sha256(p.read_bytes()).hexdigest() != v['sha256']: bad.append(k)
    return bad

# ---------------- 面板＋P 補價 ----------------
_raw = {k: pd.read_parquet(PNL / f'{k}.parquet') for k in ['close_raw', 'open', 'high', 'value', 'ret', 'flag']}
IDX = _raw['close_raw'].index; COLS = _raw['close_raw'].columns; N = len(COLS)
CI = {c: j for j, c in enumerate(COLS)}
TPOS = {d: i for i, d in enumerate(IDX)}
arr = {k: v.reindex(index=IDX, columns=COLS).to_numpy().copy() for k, v in _raw.items()}
_P = pd.read_parquet(AG / 'o_infra_v3_rebuild/delist_prices.parquet')
_ii = IDX.get_indexer(pd.DatetimeIndex(pd.to_datetime(_P['date'])))
_jj = np.array([CI.get(c, -1) for c in _P['code']])
_ok = (_ii >= 0) & (_jj >= 0)
_ii, _jj, _Pk = _ii[_ok], _jj[_ok], _P[_ok]
_new = ~np.isfinite(arr['close_raw'][_ii, _jj])
_ii, _jj, _Pk = _ii[_new], _jj[_new], _Pk[_new]
for k, pk in [('close_raw', 'close'), ('open', 'open'), ('high', 'high'), ('value', 'value'), ('ret', 'ret')]:
    arr[k][_ii, _jj] = _Pk[pk].to_numpy()
arr['flag'][_ii, _jj] = 0
NPATCH = int(len(_ii))
Cn, On, Vn, Rn, Fn = arr['close_raw'], arr['open'], arr['value'], arr['ret'], arr['flag']
TRADED = np.isfinite(Cn)
GROW = np.where(TRADED, 1.0 + np.nan_to_num(Rn, nan=0.0), 1.0)
MYADJ = np.where(TRADED, np.cumprod(GROW, axis=0), np.nan)
NOBS = TRADED.sum(1); GOOD = NOBS >= 300
PREVC = pd.DataFrame(Cn).ffill().shift(1).to_numpy()
PANEL_END = IDX[-1]

# 0050 含息
_E = pd.read_csv('/home/user/jonatw/finlab-txd/data/raw/etf_0050.csv', parse_dates=['date']).set_index('date')['adj'].sort_index()
E50 = pd.Series(_E.reindex(IDX.union(_E.index)).ffill().reindex(IDX).to_numpy(), index=IDX)

# ---------------- gapfix 滾動 ----------------
def _rolling():
    g = np.where(GOOD)[0]; dg = IDX[g]
    A = MYADJ[g]; Vv = Vn[g]
    ex = dg.isin(pd.DatetimeIndex(['2017-08-03', '2018-07-12'])) | ((dg >= '2024-02-15') & (dg <= '2024-03-15'))
    okA = pd.DataFrame((np.isnan(A) & ~ex[:, None]).astype(np.int16)).rolling(252, min_periods=252).sum().to_numpy() == 0
    okV = pd.DataFrame((np.isnan(Vv) & ~ex[:, None]).astype(np.int16)).rolling(60, min_periods=60).sum().to_numpy() == 0
    hi = np.where(okA, pd.DataFrame(A).rolling(252, min_periods=1).max().to_numpy(), np.nan)
    v60 = np.where(okV, pd.DataFrame(Vv).rolling(60, min_periods=1).mean().to_numpy(), np.nan)
    with np.errstate(invalid='ignore', divide='ignore'):
        s = A / hi
    S52 = np.full((len(IDX), N), np.nan); V60 = np.full((len(IDX), N), np.nan)
    S52[g] = s; V60[g] = v60
    return S52, V60
S52, V60 = _rolling()

# ---------------- 名單：金融、外國、gap ----------------
SL = pd.read_csv(EXT.parent / 'stock_list.csv', dtype=str).drop_duplicates('code').set_index('code')
FIN = set(SL.index[SL['industry'] == '金融保險業']) | {c for c in COLS if c[:2] == '28' or c in ('5876', '5880')}
DO = pd.read_csv(AG / 'o_infra_v3_rebuild/delist_official.csv', dtype=str)
DO['date'] = pd.to_datetime(DO['date'])
FINKW = ['銀', '證', '金控', '證金', '產保', '壽', '票券', '產險', '人壽', '投信', '合庫', '信託']
def finname(nm): return any(k in nm for k in FINKW)
FIN_DEL = {r.code for r in DO.itertuples() if r.code in CI and r.in_panel == 'True' and finname(r.name)}
FINX = FIN | FIN_DEL
_fz = json.loads((AG / 'o_rev_survivorship/data_freeze.json').read_text())
def foreignname(nm): return isinstance(nm, str) and ('KY' in nm or 'DR' in nm or nm.startswith('F-'))
FOREIGN = set(_fz['foreign_codes']) | {c for c in COLS if c[:2] == '91'} \
    | {c for c, nm in SL['name'].items() if foreignname(nm)} | {r.code for r in DO.itertuples() if foreignname(r.name)}
EXCL = np.array([(c in FINX) or (c in FOREIGN) for c in COLS])
GAPEND = np.full(N, np.datetime64('NaT'), dtype='datetime64[ns]')
for r in DO.itertuples():
    if r.four == 'True' and r.code in CI and not (foreignname(r.name) or r.code[:2] == '91' or r.code in FIN or finname(r.name)):
        j = CI[r.code]
        if np.isnat(GAPEND[j]) or GAPEND[j] < np.datetime64(r.date): GAPEND[j] = np.datetime64(r.date)

# ---------------- 處置（成交日口徑） ----------------
_dh = pd.read_csv(AG / 'o_disposal_rejudge/disposal_hist_rejudge.csv', dtype=str)
_dh = _dh[_dh['in_panel'] == 'True'][['announce', 'code', 'start', 'end']]
_d5 = pd.read_csv(EXT / 'disposal_5y.csv.gz', dtype=str).rename(columns={'date': 'announce'})
_k = set(zip(_dh.code, _dh.start))
_d5 = _d5[[(c, s) not in _k for c, s in zip(_d5.code, _d5.start)]][['announce', 'code', 'start', 'end']]
DISP = pd.concat([_dh, _d5], ignore_index=True)
for c in ['announce', 'start', 'end']: DISP[c] = pd.to_datetime(DISP[c])
DISP = DISP[DISP.code.isin(CI)]
_DJ = DISP.code.map(CI).to_numpy(); _DA = DISP.announce.to_numpy(); _DS = DISP.start.to_numpy(); _DE = DISP.end.to_numpy()
def disp_mask(d, e):
    d = np.datetime64(d); e = np.datetime64(e)
    m = (_DA <= d) & (_DS <= e) & (e <= _DE)
    out = np.zeros(N, bool); out[_DJ[m]] = True
    return out

# ---------------- 營收（y 與首次值 f） ----------------
P0 = pd.Period('2003-01', 'M')
def pidx(p): return (p - P0).n
NP = pidx(pd.Period('2027-12', 'M')) + 1
def _load_y(path):
    y = pd.read_parquet(path); y.index = pd.PeriodIndex(y.index, freq='M')
    cnt = y.notna().sum(1); lf = cnt[cnt >= 0.8 * cnt.median()].index.max()
    yy = y.reindex(columns=COLS)
    M = np.full((NP, N), np.nan)
    for p, row in zip(yy.index, yy.to_numpy()): M[pidx(p)] = row
    return M, lf, set(y.index)
YL, LF_L, PS_L = _load_y(O4 / 'yoy_long_R_FIX.parquet')
Y5, LF_5, PS_5 = _load_y(O4 / 'yoy_rev5y_FIX.parquet')

FV = pd.read_csv(O5 / 'data/fv_replace.csv', dtype={'code': str, 'ym': str})
FV['first_date'] = pd.to_datetime(FV['first_date'])
NEWS = {}   # pidx → list of (j, first_date, ratio_L, ratio_5)
for r in FV.itertuples():
    if r.code in CI: NEWS.setdefault(pidx(pd.Period(r.ym, 'M')), []).append((CI[r.code], r.first_date, r.ratio_L, r.ratio_5))

def mc_matrix(name):
    """擾動格 ratio（不論日期、兩面板相同）。name=None → 全 1。"""
    F = np.ones((NP, N))
    if name is None: return F
    d = pd.read_parquet(O5 / f'data/{name}.parquet')
    for c, ym, r in zip(d['code'], d['ym'], d['ratio']):
        if c in CI: F[pidx(pd.Period(ym, 'M')), CI[c]] *= r
    return F

class Rev:
    """y3 計算器。mode：'raw'（V4，不替換）、'fv'（首次值替換）、'fv_latestbase'（分母 f 恆 1）。mc：擾動檔名或 None。"""
    def __init__(self, mode='fv', mc=None):
        self.mode = mode; self.F = mc_matrix(mc); self.hasmc = mc is not None
    def f_row(self, k, d, long):
        f = self.F[k].copy()
        if self.mode != 'raw':
            for j, fd, rL, r5 in NEWS.get(k, []):
                if fd < d: f[j] *= (rL if long else r5)
        return f
    def yprime(self, k, d, long):
        Y = YL if long else Y5
        y = Y[k]
        fn = self.f_row(k, d, long)
        fd = np.ones(N) if self.mode == 'fv_latestbase' and not self.hasmc else self.f_row(k - 12, d, long)
        if self.mode == 'fv_latestbase' and self.hasmc:
            fd = self.F[k - 12]
        adj = (fn != 1.0) | (fd != 1.0)
        if not adj.any(): return y
        out = y.copy()
        out[adj] = ((1 + y[adj] / 100) * fn[adj] / fd[adj] - 1) * 100
        return out
    def y3(self, d, m, back=0, long=None):
        if long is None: long = d < SWITCH
        PS = PS_L if long else PS_5
        ks = [m - 1 - back, m - 2 - back, m - 3 - back]
        if any(k not in PS for k in ks): return np.full(N, np.nan)
        a = np.vstack([self.yprime(pidx(k), d, long) for k in ks])
        return np.where(np.isfinite(a).all(0), a.mean(0), np.nan)

RAW = Rev('raw')

# ---------------- 換股日 ----------------
def official_dates():
    out = []; seen = set()
    for i, d in enumerate(IDX):
        if d < pd.Timestamp('2005-01-01') or d >= END or d.day < 11: continue
        if (d.year, d.month) in seen: continue
        seen.add((d.year, d.month))
        m = pd.Period(d, 'M'); long = d < SWITCH
        if m - 1 > (LF_L if long else LF_5): continue
        if np.isfinite(RAW.y3(d, m)).sum() <= 100: continue
        out.append(dict(i_d=i, i_e=i + 1, m=m))
    return out

# ---------------- 漲停（tick） ----------------
def tick(p): return np.select([p < 10, p < 50, p < 100, p < 500, p < 1000], [0.01, 0.05, 0.1, 0.5, 1.0], 5.0)
_GU = {}
def gap_up(ie):
    if ie in _GU: return _GU[ie]
    lim = 0.07 if IDX[ie] < pd.Timestamp('2015-06-01') else 0.10
    with np.errstate(invalid='ignore', divide='ignore'):
        ref = np.where(Fn[ie] == 1, PREVC[ie], Cn[ie] / (1 + Rn[ie]))
        raw = ref * (1 + lim); t = tick(np.nan_to_num(raw, nan=1.0))
        lu = np.floor(raw / t + 1e-9) * t
        g = np.isfinite(On[ie]) & np.isfinite(ref) & (On[ie] >= lu - 1e-6)
    _GU[ie] = g
    return g

# ---------------- AB ----------------
PDL = json.loads((AG / 'rev_announce/p_delay.json').read_text(encoding='utf-8'))
_FS = pd.read_csv(AG / 'rev_announce/firstseen_2026-09.csv', dtype=str, index_col=0)
OK2609 = np.array([c in set(_FS.index[_FS['cat'] == 'A_before']) for c in COLS])
_IND = [SL['industry'].get(c) if isinstance(SL['industry'].get(c), str) else '（stock_list 無）' for c in COLS]  # p_delay.json 有此鍵
PCOL = np.array([1.0 if c in FIN else PDL['p_by_industry'].get(i, PDL['p_overall']) for c, i in zip(COLS, _IND)])
IS15 = np.array([c in set(PDL['INS15']) for c in COLS])
_TX = pd.DatetimeIndex(sorted(set(pd.read_csv(TAIEX, parse_dates=['date'])['date'])))
TCAL_U = pd.DatetimeIndex(sorted(set(_TX) | set(IDX)))

def _case(d):
    rm = str(pd.Period(d, 'M') - 1)
    if rm in ('2019-01', '2023-03'): return 'late'
    if rm == '2020-01': return 'eq'
    D = TCAL_U[TCAL_U >= pd.Timestamp(d.year, d.month, 10)][0]
    return 'early' if D < d else ('eq' if D == d else 'late')

def unavail_mc(seed, worst=False):
    rng = np.random.default_rng(seed); cache = {}
    def U(d):
        if d in cache: return cache[d]
        rm = str(pd.Period(d, 'M') - 1); c = _case(d)
        u = np.zeros(N, bool)
        if rm == '2026-09': u = ~OK2609
        elif c == 'late': u[:] = True
        elif c == 'eq':
            u = np.ones(N, bool) if worst else (rng.random(N) < PCOL)
        if '2026-01' <= rm <= '2026-08': u = u | IS15
        cache[d] = u
        return u
    return U

# ---------------- 選股 ----------------
_BASE = {}
def base(i_d, i_e):
    key = (i_d, i_e)
    if key in _BASE: return _BASE[key]
    d, e = IDX[i_d], IDX[i_e]
    v = V60[i_d]; c = Cn[i_d]
    ok0 = np.isfinite(v) & (v >= 1e7) & np.isfinite(c) & (c >= 10)
    vv = np.where(ok0, v, -np.inf)
    o = np.argsort(-vv, kind='stable')[:min(300, int(ok0.sum()))]
    top = np.zeros(N, bool); top[o] = True
    ok = top & np.isfinite(S52[i_d]) & np.isfinite(On[i_e]) & TRADED[i_e] & ~EXCL & ~disp_mask(d, e)
    _BASE[key] = ok
    return ok

def build_plan(dates, rev, U=None, bound=None, n=10, long_by='d'):
    """dates：list of dict(i_d, i_e, m)。bound：None／'B05'／'B10'。回傳 list of dict(i_d,i_e,picks,pool,gap)。"""
    plan = []; held = set()
    for D in dates:
        i_d, i_e, m = D['i_d'], D['i_e'], D['m']
        d = IDX[i_d]
        long = d < SWITCH
        y = rev.y3(d, m, 0, long)
        if U is not None:
            u = U(d) if D.get('u') is None else D['u']
            if u.any(): y = np.where(u, rev.y3(d, m, 1, long), y)
        ok = base(i_d, i_e)
        gapadd = np.zeros(N, bool)
        if bound is not None:
            gapadd = ok & ~np.isfinite(y) & ~np.isnat(GAPEND) & (GAPEND >= np.datetime64(d))
        el = ok & (np.isfinite(y) | gapadd)
        idx = np.where(el)[0]
        if len(idx) < 50: continue
        s = pd.Series(S52[i_d, idx]).rank(pct=True).to_numpy()
        g = gapadd[idx]; yy = y[idx].copy()
        py = np.empty(len(idx))
        py[~g] = pd.Series(yy[~g]).rank(pct=True).to_numpy()
        if g.any():
            py[g] = 1.0 if bound == 'B10' else 0.5
            yy[g] = np.inf if bound == 'B10' else -np.inf   # B05：同分序放最後（理論上不會同分）
        sc = np.round(0.5 * s + 0.5 * py, 10)
        order = np.lexsort((idx, -yy, -sc))
        gu = gap_up(i_e)
        picks = []
        for k in order:
            j = int(idx[k])
            if gu[j] and j not in held: continue
            picks.append(j)
            if len(picks) == n: break
        plan.append(dict(i_d=i_d, i_e=i_e, m=m, picks=picks, pool=idx, gap=set(idx[g].tolist())))
        held = set(picks)
    return plan

# ---------------- X ----------------
_MC = pd.read_csv(AG / 'o_infra_v3_rebuild/merger_consideration.csv', dtype={'code': str, 'successor': str})
for c in ('delist_date', 'pay_date', 'announce_date'): _MC[c] = pd.to_datetime(_MC[c], errors='coerce')
_MC = _MC.set_index('code')
def _xrules():
    lt = np.where(TRADED.any(0), len(IDX) - 1 - np.argmax(TRADED[::-1], axis=0), -1)
    until = {}
    for c, dd in zip(DO[DO.four == 'True'].code, DO[DO.four == 'True'].date):
        j = CI.get(c)
        if j is None or lt[j] < 0 or IDX[lt[j]] <= dd: continue
        tr = np.where(TRADED[:, j])[0]
        a = tr[IDX[tr] <= dd]; b = tr[IDX[tr] > dd]
        if len(a) == 0 or len(b) == 0: continue
        lt[j] = a[-1]; until[j] = TPOS[IDX[b[0]]]
    out = {}
    for j, c in enumerate(COLS):
        if lt[j] < 0 or IDX[lt[j]] > PANEL_END - pd.Timedelta(days=30): continue
        lc = Cn[lt[j], j]; ltd = IDX[lt[j]]
        if c in _MC.index:
            e = _MC.loc[c]; typ = e['type']
            dd = e['delist_date'] if pd.notna(e['delist_date']) else ltd + pd.Timedelta(days=14)
            if typ == 'cash' and pd.notna(e['cash']):
                pay = e['pay_date'] if pd.notna(e['pay_date']) else dd + pd.Timedelta(days=14)
                rel = (float(e['cash']) / lc, pay)
            elif typ == 'failure': rel = (0.0, ltd)
            else: rel = (1.0, dd)
        else:
            rel = (1.0, ltd + pd.Timedelta(days=14))
        out[j] = (rel[0], int(np.searchsorted(IDX, rel[1])), until.get(j, 10 ** 9))
    return out
XR = _xrules()

# ---------------- 引擎 ----------------
FEE = 0.001425 * 0.6
def rates(kind='normal', slip=0.001):
    if kind == 'stress': return 0.005, 0.005
    if kind == 'fullfee': return 0.001425 + 0.001, 0.001425 + 0.003 + 0.001
    return FEE + slip, FEE + 0.003 + slip

def run(plan, buy, sell, end=END, X=True, use_pool=False, log=None):
    t1 = TPOS[end] if end in TPOS else len(IDX) - 1
    reb = {p['i_e']: (p['pool'] if use_pool else p['picks']) for p in plan}
    t0 = plan[0]['i_d']
    val = np.zeros(N); cash = 1.0
    nav = np.empty(t1 - t0 + 1)
    for t in range(t0, t1 + 1):
        if t in reb and t > t0:
            pk = np.asarray(reb[t], int)
            tr = TRADED[t] & np.isfinite(On[t])
            gapf = np.where(tr, GROW[t] * On[t] / np.where(tr, Cn[t], 1.0), 1.0)
            h = val * gapf
            frz = (h > 0) & ~tr
            if X:
                for j in np.where(frz)[0]:
                    r = XR.get(j)
                    if r is not None and r[1] <= t < r[2]:
                        cash += h[j] * r[0]; h[j] = 0.0; frz[j] = False
            htr = np.where(frz, 0.0, h)
            Vt = htr.sum() + cash
            k = len(pk)
            mask = np.zeros(N, bool); mask[pk] = True
            hk = htr[mask]; hother = htr[~mask].sum()
            lo, hi = 0.0, Vt / k
            for _ in range(60):
                mid = 0.5 * (lo + hi)
                dl = mid - hk
                cost = buy * dl[dl > 0].sum() - sell * dl[dl < 0].sum() + sell * hother
                if mid * k + cost > Vt: hi = mid
                else: lo = mid
            T = lo
            if log is not None:
                log.append((t, [j for j in pk if htr[j] <= 0]))
            newv = np.where(frz, h, 0.0); newv[pk] += T
            oc = np.where(tr, Cn[t] / np.where(tr, On[t], 1.0), 1.0)
            val = np.where(frz, newv, newv * oc)
            cash = 0.0
        elif t > t0:
            val = val * GROW[t]
        nav[t - t0] = val.sum() + cash
    return pd.Series(nav, index=IDX[t0:t1 + 1])

def avg_nav(navs):
    """多條淨值：日報酬平均後連乘。"""
    R = pd.concat([n.pct_change() for n in navs], axis=1).mean(axis=1).fillna(0.0)
    return (1 + R).cumprod() * navs[0].iloc[0]

# ---------------- 統計 ----------------
def tstat(x):
    x = np.asarray(x, float); return x.mean() / (x.std(ddof=1) / math.sqrt(len(x)))

def wins(plan, end=END):
    ds = [IDX[p['i_d']] for p in plan]
    return list(zip(ds, ds[1:] + [end]))

def wret(nav, W):
    return pd.Series([nav[b] / nav[a] - 1 for a, b in W], index=pd.DatetimeIndex([a for a, b in W]))

def excess(nav, W, bench=None):
    b = E50 if bench is None else bench
    return wret(nav, W) - wret(b, W)

def cagr(nav, a, b=END):
    s = nav[(nav.index >= a) & (nav.index <= b)]
    return (s.iloc[-1] / s.iloc[0]) ** (365.25 / (s.index[-1] - s.index[0]).days) - 1

def mdd(nav, a, b=END):
    s = nav[(nav.index >= a) & (nav.index <= b)]
    return float((s / s.cummax() - 1).min())

SEGS = {'S1': ('2006-04-01', '2013-12-31'), 'S2': ('2014-01-01', '2020-12-31'), 'S3': ('2021-03-01', '2026-10-12'),
        'S4': ('2024-01-01', '2026-10-12'), 'SEL': ('2006-04-01', '2018-12-31'), 'VAL': ('2019-01-01', '2023-12-31')}

def seg(nav, W, a, b):
    ex = excess(nav, W)
    x = ex[(ex.index >= a) & (ex.index <= b)]
    d0 = x.index[0]
    bend = min(pd.Timestamp(b), END)
    return dict(cagr=cagr(nav, d0, bend), t=tstat(x), e50=cagr(E50, d0, bend), n=len(x))

def nw_t(x, L):
    x = np.asarray(x, float); n = len(x); z = x - x.mean()
    s = (z * z).sum() / n
    for l in range(1, L + 1): s += 2 * (1 - l / (L + 1)) * (z[l:] * z[:-l]).sum() / n
    return x.mean() / math.sqrt(s / n)

def boot_p(x, B=10000, blk=12, seed=7):
    x = np.asarray(x, float); n = len(x); rng = np.random.default_rng(seed)
    t0 = tstat(x); z = x - x.mean(); nb = int(math.ceil(n / blk)); cnt = 0
    for _ in range(B):
        st = rng.integers(0, n, nb)
        ii = (st[:, None] + np.arange(blk)[None, :]).ravel()[:n] % n
        if tstat(z[ii]) >= t0: cnt += 1
    return cnt / B, (cnt + 1) / (B + 1)

def p1(t): return 0.5 * (1 - math.erf(t / math.sqrt(2)))
