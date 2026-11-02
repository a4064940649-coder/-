#!/usr/bin/env python3
"""複製員 o7 軌 B：自己重寫的零衝擊引擎（不 import 任何研究中心程式）。
只讀資料：V7 面板 data/*.parquet、P（delist_prices）、X（merger_consideration）、終止上市名單、官方處置名單、
R_FIX／Y5_FIX、AB 延後機率表、stock_list、0050。
寫法刻意不同：成本用分段線性精確解（不用二分法）；AB 亂數自己產生；統計自己算。"""
import json, math
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
O7 = HERE.parent; AG = O7.parent
PNL = O7 / 'data'
EXT = Path('/home/user/stevenhsiehsheng-ctrl/tw-stock-screener/data/extras')
END = pd.Timestamp('2026-10-12'); SWITCH = pd.Timestamp('2021-03-01')
FEE = 0.001425 * 0.6; TAX = 0.003

# ---------- 面板＋P ----------
C = pd.read_parquet(PNL / 'close_raw.parquet')
IDX, COLS = C.index, C.columns; N = len(COLS); CI = {c: j for j, c in enumerate(COLS)}
def _ld(k): return pd.read_parquet(PNL / f'{k}.parquet').reindex(index=IDX, columns=COLS).to_numpy().copy()
Cn = C.to_numpy().copy(); On = _ld('open'); Vn = _ld('value'); Rn = _ld('ret'); Fn = _ld('flag')
Pdl = pd.read_parquet(AG / 'o_infra_v3_rebuild' / 'delist_prices.parquet')
ii = IDX.get_indexer(pd.DatetimeIndex(Pdl['date'])); jj = np.array([CI.get(c, -1) for c in Pdl['code']])
ok = (ii >= 0) & (jj >= 0); ii, jj, Pdl = ii[ok], jj[ok], Pdl[ok]
nw = ~np.isfinite(Cn[ii, jj]); ii, jj, Pdl = ii[nw], jj[nw], Pdl[nw]
Cn[ii, jj] = Pdl['close'].to_numpy(); On[ii, jj] = Pdl['open'].to_numpy(); Vn[ii, jj] = Pdl['value'].to_numpy()
Rn[ii, jj] = Pdl['ret'].to_numpy(); Fn[ii, jj] = 0
NPATCH = int(len(ii))
TR = np.isfinite(Cn)
G = np.where(TR, 1.0 + np.nan_to_num(Rn, nan=0.0), 1.0)
ADJ = np.where(TR, np.cumprod(G, axis=0), np.nan)
GOOD = TR.sum(1) >= 300
PREVC = pd.DataFrame(Cn).ffill().shift(1).to_numpy()
TPOS = {d: i for i, d in enumerate(IDX)}

E50r = pd.read_csv('/home/user/jonatw/finlab-txd/data/raw/etf_0050.csv', parse_dates=['date']).set_index('date')['adj'].sort_index()
E50 = E50r.reindex(IDX.union(E50r.index)).ffill().reindex(IDX)

# ---------- 滾動（只用 GOOD 日；豁免 2017-08-03、2018-07-12、2024-02-15～03-15 的缺值）----------
def _roll():
    g = np.where(GOOD)[0]; dg = IDX[g]
    ex = dg.isin(pd.DatetimeIndex(['2017-08-03', '2018-07-12'])) | ((dg >= '2024-02-15') & (dg <= '2024-03-15'))
    A = ADJ[g]; Vv = Vn[g]
    mA = pd.DataFrame((np.isnan(A) & ~ex[:, None]).astype(np.int16)).rolling(252, min_periods=252).sum().to_numpy() == 0
    mV = pd.DataFrame((np.isnan(Vv) & ~ex[:, None]).astype(np.int16)).rolling(60, min_periods=60).sum().to_numpy() == 0
    hi = pd.DataFrame(A).rolling(252, min_periods=1).max().to_numpy()
    v60 = pd.DataFrame(Vv).rolling(60, min_periods=1).mean().to_numpy()
    with np.errstate(invalid='ignore', divide='ignore'):
        s = A / np.where(mA, hi, np.nan)
    S52 = np.full((len(IDX), N), np.nan); V60 = np.full((len(IDX), N), np.nan)
    S52[g] = s; V60[g] = np.where(mV, v60, np.nan)
    return S52, V60
S52, V60 = _roll()

# ---------- 名單 ----------
SL = pd.read_csv(EXT.parent / 'stock_list.csv', dtype=str).drop_duplicates('code').set_index('code')
FIN = set(SL.index[SL['industry'] == '金融保險業']) | {c for c in COLS if c[:2] == '28' or c in ('5876', '5880')}
FINM = np.array([c in FIN for c in COLS])
DL = pd.read_csv(AG / 'o_infra_v3_rebuild' / 'delist_official.csv', dtype=str)
DLNAME = dict(zip(DL.code, DL.name))
FW = ('銀', '證', '金控', '證金', '產保', '壽', '票券', '產險', '人壽', '投信', '合庫', '信託')
def _isfin(c):
    n = DLNAME.get(c, ''); n = n if isinstance(n, str) else ''
    return c in FIN or any(w in n for w in FW)
FIN2M = np.array([_isfin(c) for c in COLS])
FOR = set(json.loads((AG / 'o_rev_survivorship' / 'data_freeze.json').read_text())['foreign_codes'])
FOR |= {c for c in COLS if c.startswith('91')}
FOR |= {c for c, n in zip(DL.code, DL.name) if isinstance(n, str) and (n.startswith('F-') or 'KY' in n.upper())}
FORM = np.array([c in FOR for c in COLS])
_d4 = DL[DL['four'] == 'True']
GAPD = {c: pd.Timestamp(d) for c, d in zip(_d4.code, _d4.date) if c not in FOR and not _isfin(c)}
GAPM = np.array([c in GAPD for c in COLS])
GAPU = np.array([np.datetime64(GAPD[c]) if c in GAPD else np.datetime64('2100-01-01') for c in COLS])

# ---------- 處置（成交日口徑）----------
A_ = pd.read_csv(AG / 'o_disposal_rejudge' / 'disposal_hist_rejudge.csv', dtype={'code': str}, parse_dates=['announce', 'start', 'end'])
A_ = A_[A_['in_panel'] == True]
B_ = pd.read_csv(EXT / 'disposal_5y.csv.gz', dtype={'code': str}, parse_dates=['date', 'start', 'end'])
B_ = B_[B_.code.isin(set(COLS))]
ka = set(zip(A_.code, A_.start))
B_ = B_[[(c, s) not in ka for c, s in zip(B_.code, B_.start)]]
DSP = pd.concat([A_[['announce', 'code', 'start', 'end']], B_.rename(columns={'date': 'announce'})[['announce', 'code', 'start', 'end']]])
DSP = DSP[DSP.code.isin(CI)]
DJ = DSP.code.map(CI).to_numpy(); DA = DSP.announce.to_numpy(); DS = DSP.start.to_numpy(); DE = DSP.end.to_numpy()
def disp(t):
    d = np.datetime64(IDX[t]); x = np.datetime64(IDX[t + 1])
    m = np.zeros(N, bool); m[DJ[(DA <= d) & (DS <= x) & (DE >= x)]] = True
    return m

# ---------- 營收 ----------
def _y(f):
    y = pd.read_parquet(f); y.index = pd.PeriodIndex(y.index, freq='M'); return y
YL = _y(AG / 'o_v4_revenue_patch_official' / 'yoy_long_R_FIX.parquet')
Y5 = _y(AG / 'o_v4_revenue_patch_official' / 'yoy_rev5y_FIX.parquet')
def _lf(Y):
    c = Y.notna().sum(1); return c[c >= 0.8 * c.median()].index.max()
LFL, LF5 = _lf(YL), _lf(Y5)
_Y3 = {}
def y3m(Y, m, key):
    k = (key, str(m))
    if k not in _Y3:
        ms = [m - 1, m - 2, m - 3]
        if any(x not in Y.index for x in ms): out = np.full(N, np.nan)
        else:
            Z = Y.reindex(index=ms, columns=COLS).to_numpy(); out = Z.mean(0); out[np.isnan(Z).any(0)] = np.nan
        _Y3[k] = np.where(FINM, np.nan, out)
    return _Y3[k]
def y3(d, back=0, m=None):
    m = (m if m is not None else pd.Period(d, 'M')) - back
    return y3m(Y5, m, '5') if d >= SWITCH else y3m(YL, m, 'L')

def official_days():
    out, seen = [], set()
    for d in IDX[IDX >= '2005-01-01']:
        if d.day < 11 or (d.year, d.month) in seen: continue
        seen.add((d.year, d.month)); m = pd.Period(d, 'M')
        lf = LF5 if d >= SWITCH else LFL
        if m - 1 > lf: continue
        if np.isfinite(y3(d)).sum() > 100 and d < END: out.append(d)
    return out

# ---------- AB（自己重寫 make_unavail('mc', seed)）----------
RA = AG / 'rev_announce'
PDL = json.loads((RA / 'p_delay.json').read_text(encoding='utf-8'))
IND = [SL['industry'].get(c) if isinstance(SL['industry'].get(c), str) else '（stock_list 無）' for c in COLS]
PCOL = np.array([1.0 if c in FIN else PDL['p_by_industry'].get(i, PDL['p_overall']) for c, i in zip(COLS, IND)])
IS15 = np.array([c in set(PDL['INS15']) for c in COLS])
FS = pd.read_csv(RA / 'firstseen_2026-09.csv', dtype=str, index_col=0)
OK2609 = np.array([c in set(FS.index[FS['cat'] == 'A_before']) for c in COLS])
TCAL = pd.DatetimeIndex(sorted(set(pd.read_csv('/home/user/jonatw/finlab-txd/data/raw/taiex_twii.csv', parse_dates=['date'])['date']) | set(IDX)))
def case(d):
    rm = str(pd.Period(d, 'M') - 1)
    if rm in ('2019-01', '2023-03'): return 'late'
    if rm == '2020-01': return 'eq'
    D = TCAL[TCAL >= pd.Timestamp(d.year, d.month, 10)][0]
    return 'early' if D < d else ('eq' if D == d else 'late')
def unavail_mc(seed):
    rng = np.random.default_rng(seed); memo = {}
    def U(d):
        if d not in memo:
            rm = str(pd.Period(d, 'M') - 1); c = case(d); u = np.zeros(N, bool)
            if rm == '2026-09': u = ~OK2609
            elif c == 'late': u[:] = True
            elif c == 'eq': u = rng.random(N) < PCOL
            if '2026-01' <= rm <= '2026-08': u = u | IS15
            memo[d] = u
        return memo[d]
    return U

# ---------- 漲停 ----------
def tick(p): return np.select([p < 10, p < 50, p < 100, p < 500, p < 1000], [0.01, 0.05, 0.1, 0.5, 1.0], 5.0)
def gap_up(t):
    lim = 0.07 if IDX[t] < pd.Timestamp('2015-06-01') else 0.10
    with np.errstate(invalid='ignore', divide='ignore'):
        ref = np.where(Fn[t] == 1, PREVC[t], Cn[t] / (1 + Rn[t]))
        raw = ref * (1 + lim); tk = tick(raw); lu = np.floor(raw / tk + 1e-9) * tk
        return np.isfinite(On[t]) & np.isfinite(ref) & (On[t] >= lu - 1e-6)

# ---------- 選股 ----------
def plan(days=None, B10=False, unavail=None, mof=None, n=10):
    days = days or official_days(); mof = mof or {}
    out, held = [], set(); nblk = 0
    for d in days:
        t = TPOS[d]; e = t + 1; m = mof.get(d)
        v = V60[t]; px = Cn[t]
        base = np.isfinite(v) & (v >= 1e7) & np.isfinite(px) & (px >= 10)
        o = np.argsort(-np.where(base, v, -np.inf), kind='stable')
        top = np.zeros(N, bool); top[o[:min(300, int(base.sum()))]] = True; top &= base
        yy = y3(d, 0, m)
        if unavail is not None:
            U = unavail(d)
            if U is not None and U.any(): yy = np.where(U, y3(d, 1, m), yy)
        s52 = S52[t]
        okk = top & np.isfinite(s52) & ~disp(t) & np.isfinite(On[e]) & TR[e] & ~FIN2M & ~FORM
        elr = okk & np.isfinite(yy)
        gm = (GAPM & okk & ~np.isfinite(yy) & (np.datetime64(d) <= GAPU)) if B10 else np.zeros(N, bool)
        el = elr | gm
        if el.sum() < 50: continue
        p52 = pd.Series(np.where(el, s52, np.nan)).rank(pct=True).to_numpy()
        pry = pd.Series(np.where(elr, yy, np.nan)).rank(pct=True).to_numpy()
        pry = np.where(gm, 1.0, pry)
        sc = np.where(el, 0.5 * p52 + 0.5 * pry, np.nan)
        ytie = np.where(gm, np.inf, yy)
        k1 = -np.round(np.where(el, sc, -9.0), 10)
        k2 = -np.where(np.isfinite(ytie) | np.isinf(ytie), np.where(np.isinf(ytie), 1.7976931348623157e308, ytie), -1e18)
        order = np.lexsort((np.arange(N), k2, k1))[:int(el.sum())]
        gu = gap_up(e); pick = []
        for j in order:
            if gu[j] and j not in held: nblk += 1; continue
            pick.append(int(j))
            if len(pick) == n: break
        out.append(dict(d=d, e=e, picks=np.array(pick, int), pool=np.where(el)[0]))
        held = set(pick)
    return out, nblk

# ---------- X ----------
def _xrules():
    lt = np.where(TR.any(0), len(IDX) - 1 - np.argmax(TR[::-1], 0), -1)
    until = {}
    for c, dd in zip(_d4.code, pd.to_datetime(_d4.date)):
        j = CI.get(c)
        if j is None or lt[j] < 0 or IDX[lt[j]] <= dd: continue
        w = np.where(TR[:, j])[0]; a = w[IDX[w] <= dd]; b = w[IDX[w] > dd]
        if len(a) == 0 or len(b) == 0: continue
        lt[j] = a[-1]; until[j] = IDX[b[0]]
    E = pd.read_csv(AG / 'o_infra_v3_rebuild' / 'merger_consideration.csv', dtype={'code': str, 'successor': str})
    for c in ('delist_date', 'pay_date'): E[c] = pd.to_datetime(E[c], errors='coerce')
    E = E.set_index('code'); out = {}
    for j, c in enumerate(COLS):
        if lt[j] < 0 or IDX[lt[j]] > IDX[-1] - pd.Timedelta(days=30): continue
        ltd = IDX[lt[j]]; lc = Cn[lt[j], j]
        if c in E.index:
            r = E.loc[c]; dd = r['delist_date'] if pd.notna(r['delist_date']) else ltd + pd.Timedelta(days=14)
            if r['type'] == 'cash' and pd.notna(r['cash']):
                f, rel = float(r['cash']) / lc, (r['pay_date'] if pd.notna(r['pay_date']) else dd + pd.Timedelta(days=14))
            elif r['type'] == 'failure': f, rel = 0.0, ltd
            else: f, rel = 1.0, dd
        else: f, rel = 1.0, ltd + pd.Timedelta(days=14)
        out[j] = (f, int(np.searchsorted(IDX, rel)), int(np.searchsorted(IDX, until[j])) if j in until else 10 ** 9)
    return out
XR = _xrules()

# ---------- 模擬 ----------
def solve_T(h, pk, V, b, s):
    """f(T)=k·T+b·Σ_pk(T−h)+ + s·Σ_pk(h−T)+ + s·Σ_非pk h = V，分段線性精確解。"""
    k = len(pk); hp = h[pk]; other = h.sum() - hp.sum()
    const = s * other
    def f(T): return k * T + b * np.clip(T - hp, 0, None).sum() + s * np.clip(hp - T, 0, None).sum() + const
    bp = np.unique(np.concatenate([[0.0], hp[hp > 0], [V / k]]))
    fv = np.array([f(x) for x in bp])
    i = int(np.searchsorted(fv, V, side='right')) - 1
    i = min(max(i, 0), len(bp) - 2)
    x0, x1 = bp[i], bp[i + 1]; f0, f1 = fv[i], fv[i + 1]
    return x0 + (V - f0) * (x1 - x0) / (f1 - f0)

def simulate(pl, b, s, t0, t1, xr=True, pool=False):
    reb = {p['e']: (p['pool'] if pool else p['picks']) for p in pl}
    val = np.zeros(N); cash = 1.0; nav = np.empty(t1 - t0 + 1); tov = []
    for t in range(t0, t1 + 1):
        if t in reb:
            pk = reb[t]; tr = TR[t] & np.isfinite(On[t])
            h = val * np.where(tr, G[t] * On[t] / np.where(tr, Cn[t], 1.0), 1.0)
            frz = (h > 0) & ~tr
            if xr:
                for j in np.where(frz)[0]:
                    if j in XR and XR[j][1] <= t < XR[j][2]:
                        cash += h[j] * XR[j][0]; h[j] = 0.0; frz[j] = False
            htr = np.where(frz, 0.0, h); Vt = htr.sum() + cash
            T = solve_T(htr, pk, Vt, b, s) if (b or s) else Vt / len(pk)
            tgt = np.zeros(N); tgt[pk] = T
            tov.append(np.clip(tgt - htr, 0, None).sum() / (Vt + h[frz].sum()))
            nv = np.where(frz, h, 0.0); nv[pk] += T
            val = np.where(frz, nv, nv * np.where(tr, Cn[t] / np.where(tr, On[t], 1.0), 1.0)); cash = 0.0
        elif t > t0:
            val = val * G[t]
        nav[t - t0] = val.sum() + cash
    return pd.Series(nav, index=IDX[t0:t1 + 1]), float(np.mean(tov[1:]))

def rates(kind='normal', slip=0.001):
    if kind == 'stress': return 0.005, 0.005
    return FEE + slip, FEE + TAX + slip

# ---------- 統計 ----------
def tstat(x):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    return float(x.mean() / (x.std(ddof=1) / math.sqrt(len(x))))
def cagr_mdd(nav, a, b):
    s = nav[(nav.index >= pd.Timestamp(a)) & (nav.index <= pd.Timestamp(b))]
    i0 = nav.index.get_loc(s.index[0])
    base, st = (nav.iloc[i0 - 1], nav.index[i0 - 1]) if i0 > 0 else (s.iloc[0], s.index[0])
    eq = s / base; yrs = (s.index[-1] - st).days / 365.25
    return float(eq.iloc[-1] ** (1 / yrs) - 1) * 100, float((eq / eq.cummax() - 1).min()) * 100
def xs(nav, sd, a, b, bench=None):
    bench = E50 if bench is None else bench
    D = pd.DatetimeIndex([d for d in sd if d in nav.index])
    r = nav.reindex(D); rb = bench.reindex(D)
    x = ((r.shift(-1) / r - 1) - (rb.shift(-1) / rb - 1)).iloc[:-1]
    return x[(x.index >= pd.Timestamp(a)) & (x.index <= pd.Timestamp(b))].dropna()
def seg(nav, sd, a, b):
    b2 = min(pd.Timestamp(b), nav.index[-1])
    cg, dd = cagr_mdd(nav, a, b2); x = xs(nav, sd, a, b)
    return dict(CAGR=cg, maxDD=dd, t=tstat(x), n=len(x), mean_pp=float(x.mean() * 100))
def avg_nav(navs):
    return (1 + pd.concat([n.pct_change() for n in navs], axis=1).mean(1).fillna(0)).cumprod()
