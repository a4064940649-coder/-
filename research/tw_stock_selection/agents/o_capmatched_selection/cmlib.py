#!/usr/bin/env python3
"""o_capmatched_selection 共用庫（只讀 raw/ 與本資料夾 data/；不 import 其他資料夾的程式）。

內容：
  * 面板、0050 含息、加權指數日曆（raw/）
  * gapfix 滾動訊號（52 週比、60 日均成交值；照 main_v2/v2_core 的 gapfix 寫法重寫）、σ60
  * V4 只取已公告營收：R_FIX（< 2021-03-01）／Y5_FIX（之後），3 月平均年增率；未公告判定用「最壞」口徑：
    法定期限日（訊號月 10 日起第一個加權指數交易日；2019-02、2023-04 特例延後）≥ 訊號日 → 全部改用 m−2..m−4；
    2026-09 營收用 firstseen；2026-01～08 營收 INS15 一律視為未公告（同 v2_core.make_unavail('worst')）
  * 官方處置名單（disposal_hist_rejudge.csv），成交日口徑：公告日 ≤ 訊號日、成交日落在處置期間 → 排除
  * KY／DR（外國公司）與金融保險業遮罩
  * 臺灣50 成分（tw50_members.json，生效日收盤後換新名單）
  * 市值：data/shares_est.parquet（s2_capgate.py 回推）× 前值補收盤
  * 加權日頻引擎（凍結、二分法成本，買進日 close/open，同 ia_core／ocore.simulate_w）
  * 統計：訊號窗月報酬、t、IR、追蹤誤差、年化、回撤
"""
import json, math, hashlib
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
RAW = HERE / 'raw'
DATA = HERE / 'data'
DATA.mkdir(exist_ok=True)

def sha(f): return hashlib.sha256(Path(f).read_bytes()).hexdigest()

FEE = 0.001425 * 0.6
TAX = 0.003
def rates(cost):
    """回傳 (買, 賣) 單邊成本率。normal＝手續費 6 折＋滑價 0.1%；stress＝每邊 0.5%；('slip',x)；('fullfee',x)；'zero'。"""
    if cost == 'normal': return FEE + 0.001, FEE + TAX + 0.001
    if cost == 'stress': return 0.005, 0.005
    if cost == 'zero': return 0.0, 0.0
    k, x = cost
    if k == 'slip': return FEE + x, FEE + TAX + x
    if k == 'fullfee': return 0.001425 + x, 0.001425 + TAX + x
    raise ValueError(cost)

# ---------------- 面板 ----------------
def _rp(n): return pd.read_parquet(RAW / 'panel' / f'{n}.parquet')
C = _rp('close_raw'); O = _rp('open'); V = _rp('value'); RET = _rp('ret'); FLAG = _rp('flag'); SRC = _rp('src')
IDX = C.index; COLS = C.columns; N = len(COLS); CI = {c: j for j, c in enumerate(COLS)}
Cn = C.to_numpy(); On = O.to_numpy(); Vn = V.to_numpy(); Rn = RET.to_numpy(); Fn = FLAG.to_numpy(); SRCn = SRC.to_numpy()
TRADED = np.isfinite(Cn)
GROW = np.where(TRADED, 1.0 + np.nan_to_num(Rn, nan=0.0), 1.0)
LAST_TRADE = np.where(TRADED.any(axis=0), len(IDX) - 1 - np.argmax(TRADED[::-1], axis=0), -1)   # 每檔最後成交日（面板終點仍有成交者＝面板最後一日）
CFF = C.ffill().to_numpy()                       # 前值補收盤（市值用）
PREVC = C.ffill().shift(1).to_numpy()
NOBS = TRADED.sum(axis=1)
GOOD = NOBS >= 300
J2330 = CI['2330']

# 除權息日（flag 1：面板 ret 設 0）的敏感度版本：ret＝收盤/開盤 ×（1＋當日非除權息股開盤跳空中位數）−1
def _grow_exfix():
    G = GROW.copy()
    with np.errstate(invalid='ignore', divide='ignore'):
        gap = On / PREVC - 1
        normal = TRADED & np.isfinite(On) & (np.nan_to_num(Fn, nan=0) == 0) & np.isfinite(gap)
        med = np.array([np.median(gap[t][normal[t]]) if normal[t].sum() > 50 else 0.0 for t in range(len(IDX))])
        fx = (Fn == 1) & TRADED & np.isfinite(On) & (On > 0)
        alt = Cn / On * (1 + med[:, None])
    G[fx] = alt[fx]
    return G, int(fx.sum())
_GX = {}
def grow_exfix():
    if not _GX: _GX['G'], _GX['n'] = _grow_exfix()
    return _GX['G']

# ---------------- 基準 ----------------
_e = pd.read_csv(RAW / 'bench' / 'etf_0050.csv', parse_dates=['date']).set_index('date')['adj'].sort_index()
E50 = _e.reindex(IDX.union(_e.index)).ffill().reindex(IDX)
E50_RAW = _e
_tx = pd.read_csv(RAW / 'bench' / 'taiex_twii.csv', parse_dates=['date'])['date']
TCAL = pd.DatetimeIndex(sorted(set(_tx) | set(IDX)))

# ---------------- 日曆 ----------------
def official_days(start='2005-01-01', end=None):
    """每月第一個日 ≥ 11 的面板交易日。"""
    out, seen = [], set()
    ii = IDX[IDX >= pd.Timestamp(start)]
    if end is not None: ii = ii[ii <= pd.Timestamp(end)]
    for d in ii:
        if d.day >= 11 and (d.year, d.month) not in seen:
            seen.add((d.year, d.month)); out.append(d)
    return out

def deadline(m):
    """營收法定期限日 D_m：m 月 10 日起第一個加權指數交易日；2019-02、2023-04 特例（期限延後，視為 m 月 15 日起第一個交易日）。"""
    p = pd.Period(m, 'M')
    day = 15 if str(p) in ('2019-02', '2023-04') else 10
    return TCAL[TCAL >= pd.Timestamp(p.year, p.month, day)][0]

def jscan_days(j, start='2005-01-01', end=None):
    """J{j}：D_m 之後第 j 個面板交易日（j＝0 為 D_m 當日或其後第一個面板日）。每月一個。"""
    out = []
    for p in pd.period_range(pd.Timestamp(start), pd.Timestamp(end) if end else IDX[-1], freq='M'):
        D = deadline(p)
        k = IDX.searchsorted(D)
        if k + j < len(IDX):
            out.append(IDX[k + j])
    return out

# ---------------- 滾動訊號（gapfix）----------------
EXEMPT_DAYS = [pd.Timestamp('2017-08-03'), pd.Timestamp('2018-07-12')]
EXEMPT_RANGE = (pd.Timestamp('2024-02-15'), pd.Timestamp('2024-03-15'))
_RS = {}
def rolling_signals():
    if _RS: return _RS['S52'], _RS['V60'], _RS['SIG']
    g = np.where(GOOD)[0]
    A = np.where(TRADED, np.cumprod(GROW, axis=0), np.nan)[g]; Vv = Vn[g]
    Rg = np.where(TRADED, GROW - 1, np.nan)[g]
    Ad, Vd = pd.DataFrame(A), pd.DataFrame(Vv)
    dg = IDX[g]
    ex = np.isin(dg, pd.DatetimeIndex(EXEMPT_DAYS)) | ((dg >= EXEMPT_RANGE[0]) & (dg <= EXEMPT_RANGE[1]))
    missA = pd.DataFrame(np.isnan(A) & ~ex[:, None]).astype(np.int16)
    missV = pd.DataFrame(np.isnan(Vv) & ~ex[:, None]).astype(np.int16)
    okA = (missA.rolling(252, min_periods=252).sum().to_numpy() == 0)
    okV = (missV.rolling(60, min_periods=60).sum().to_numpy() == 0)
    hi = np.where(okA, Ad.rolling(252, min_periods=1).max().to_numpy(), np.nan)
    v60 = np.where(okV, Vd.rolling(60, min_periods=1).mean().to_numpy(), np.nan)
    sig = pd.DataFrame(Rg).rolling(60, min_periods=40).std().ffill().to_numpy()
    with np.errstate(invalid='ignore', divide='ignore'):
        s52 = A / hi
    S52 = np.full((len(IDX), N), np.nan); V60 = np.full((len(IDX), N), np.nan); SIG = np.full((len(IDX), N), np.nan)
    S52[g] = s52; V60[g] = v60; SIG[g] = sig
    SIG = pd.DataFrame(SIG).ffill().to_numpy()
    _RS.update(S52=S52, V60=V60, SIG=SIG)
    return S52, V60, SIG

# ---------------- 營收（V4：R_FIX／Y5_FIX）----------------
SWITCH = pd.Timestamp('2021-03-01')
def _ly(f):
    Y = pd.read_parquet(RAW / 'agents' / f); Y.index = pd.PeriodIndex(Y.index, freq='M')
    return Y.reindex(columns=COLS)
YL = _ly('yoy_long_R_FIX.parquet'); Y5 = _ly('yoy_rev5y_FIX.parquet')
def _y3(Y, m):
    ms = [m - 1, m - 2, m - 3]
    if any(x not in Y.index for x in ms): return np.full(N, np.nan)
    Z = Y.loc[ms].to_numpy(); out = Z.mean(axis=0); out[np.isnan(Z).any(axis=0)] = np.nan
    return out
def y3(d, back=0):
    Y = Y5 if d >= SWITCH else YL
    return _y3(Y, pd.Period(d, 'M') - back)

_PD = json.loads((RAW / 'agents' / 'p_delay.json').read_text(encoding='utf-8'))
INS15 = np.array([c in set(_PD['INS15']) for c in COLS])
_FS = pd.read_csv(RAW / 'agents' / 'firstseen_2026-09.csv', dtype=str, index_col=0)
ANN_OK_2609 = np.array([c in set(_FS.index[_FS['cat'] == 'A_before']) for c in COLS])
def unavail(d):
    """True＝訊號日 d 收盤時 m−1 月營收視為未公告（最壞口徑）。"""
    m = pd.Period(d, 'M'); rm = str(m - 1)
    if rm == '2026-09': u = ~ANN_OK_2609
    elif deadline(m) >= pd.Timestamp(d).normalize(): u = np.ones(N, bool)
    else: u = np.zeros(N, bool)
    if '2026-01' <= rm <= '2026-08': u = u | INS15
    return u
def y3_ab(d):
    a = y3(d); U = unavail(d)
    return np.where(U, y3(d, 1), a)

# ---------------- 產業、金融、外國公司 ----------------
SL = pd.read_csv(RAW / 'screener' / 'stock_list.csv', dtype=str).drop_duplicates('code').set_index('code')
_DL = pd.read_csv(RAW / 'agents' / 'delist_official.csv', dtype=str)
DL_NAME = dict(zip(_DL.code, _DL.name))
FIN_WORDS = ('銀', '證', '金控', '證金', '產保', '壽', '票券', '產險', '人壽', '投信', '合庫', '信託')
def _is_fin(c):
    if SL['industry'].get(c) == '金融保險業' or c[:2] == '28' or c in ('5876', '5880'): return True
    n = DL_NAME.get(c, ''); n = n if isinstance(n, str) else ''
    return any(w in n for w in FIN_WORDS)
FINMASK = np.array([_is_fin(c) for c in COLS])
_FOR = set(json.loads((RAW / 'agents' / 'rev_survivorship_data_freeze.json').read_text(encoding='utf-8'))['foreign_codes'])
_FOR |= {c for c in COLS if c.startswith('91')}
_FOR |= {c for c, n in zip(_DL.code, _DL.name) if isinstance(n, str) and (n.startswith('F-') or 'KY' in n.upper())}
_FOR |= {c for c in SL.index if 'KY' in str(SL['name'].get(c, '')).upper()}
FORMASK = np.array([c in _FOR for c in COLS])
DRMASK = np.array([c.startswith('91') for c in COLS])
ELEC_IND = {'電子零組件業', '半導體業', '光電業', '電腦及週邊設備業', '其他電子業', '通信網路業', '資訊服務業', '電子通路業', '數位雲端'}
ELECMASK = np.array([SL['industry'].get(c) in ELEC_IND for c in COLS])

# ---------------- 處置（官方名單、成交日口徑）----------------
_DSP = pd.read_csv(RAW / 'agents' / 'disposal_hist_rejudge.csv', dtype=str)
_DSP = _DSP[_DSP.code.isin(CI)]
_dann = pd.to_datetime(_DSP.announce).to_numpy(); _dst = pd.to_datetime(_DSP.start).to_numpy(); _den = pd.to_datetime(_DSP.end).to_numpy()
_dj = np.array([CI[c] for c in _DSP.code])
def disp_mask(d):
    t = IDX.get_loc(d); e = min(t + 1, len(IDX) - 1)
    dd = np.datetime64(d); de = np.datetime64(IDX[e])
    sel = (_dann <= dd) & (_dst <= de) & (_den >= de)
    m = np.zeros(N, bool); m[_dj[sel]] = True
    return m

# ---------------- 臺灣50 成分 ----------------
_TW = json.loads((RAW / 'agents' / 'tw50_members.json').read_text(encoding='utf-8'))
_TW_STEPS = [(pd.Timestamp(s['eff']), s['members_before']) for s in _TW['steps_backward']]   # 由新到舊
TW50_EARLIEST = pd.Timestamp(_TW['earliest_valid_eff'])
def tw50_at(day):
    """day 當天開盤時的成分（生效日收盤後換新名單：eff < day → 新名單）。2015-09-18 以前回傳 None。"""
    day = pd.Timestamp(day)
    if day <= TW50_EARLIEST: return None
    S = _TW['current_2026-10-09']
    for eff, mb in _TW_STEPS:
        if day <= eff: S = mb
    return list(S)

# ---------------- 市值 ----------------
_MC = {}
def shares_est():
    if 'S' not in _MC:
        S = pd.read_parquet(DATA / 'shares_est.parquet').reindex(index=IDX, columns=COLS)
        _MC['S'] = S.to_numpy()
    return _MC['S']
def mcap_row(t):
    S = shares_est()
    return S[t] * CFF[t]

# ---------------- 引擎 ----------------
def simulate_w(plan, buy, sell, t0, t1, G=None, record=False):
    """plan＝{exec_t: (idx ndarray, weight ndarray 和為 1)}。凍結：換股日開盤無成交的持股保留；二分法解成本後總值；
    買進日開盤→收盤；其他日 ×G。record=True 回傳每日持股權重（收盤後）供拆解。"""
    G = GROW if G is None else G
    val = np.zeros(N); cash = 1.0
    nav = np.empty(t1 - t0 + 1); tov = []; W = {} if record else None
    for t in range(t0, t1 + 1):
        if t in plan and t > t0:
            pk, w = plan[t]
            tr = TRADED[t] & np.isfinite(On[t])
            gap = np.where(tr, G[t] * On[t] / np.where(tr, Cn[t], 1.0), 1.0)
            h = val * gap
            frz = (h > 0) & ~tr
            # 之後再也沒有成交（下市、併購、換股轉換）的凍結持股：以最後收盤價值扣賣出成本轉成現金（PREREG 修正 v2）
            dead = frz & (LAST_TRADE < t)
            if dead.any():
                cash += h[dead].sum() * (1 - sell); h = np.where(dead, 0.0, h); frz &= ~dead
            Vfz = h[frz].sum()
            htr = np.where(frz, 0.0, h)
            Vt = htr.sum() + cash
            wv = np.zeros(N); np.add.at(wv, np.asarray(pk, int), np.asarray(w, float))
            wv[frz] = 0.0
            if wv.sum() <= 0: wv[:] = 0.0
            else: wv = wv / wv.sum()
            def cost_of(T):
                tgt = wv * T
                return buy * np.clip(tgt - htr, 0, None).sum() + sell * np.clip(htr - tgt, 0, None).sum()
            lo, hi = 0.0, Vt
            for _ in range(60):
                mid = 0.5 * (lo + hi)
                if mid + cost_of(mid) > Vt: hi = mid
                else: lo = mid
            tgt = wv * lo
            tov.append(np.clip(htr - tgt, 0, None).sum() / (Vt + Vfz))
            newval = np.where(frz, h, 0.0) + tgt
            oc = np.where(tr, Cn[t] / np.where(tr, On[t], 1.0), 1.0)
            val = np.where(frz, newval, newval * oc)
            cash = 0.0
        elif t == t0:
            pass
        else:
            val = val * G[t]
        tot = val.sum() + cash
        nav[t - t0] = tot
        if record and tot > 0: W[t] = val / tot
    out = pd.Series(nav, index=IDX[t0:t1 + 1])
    return (out, {'turnover': float(np.mean(tov[1:])) if len(tov) > 1 else np.nan, 'W': W})

def window_factor(j, e0, e1, G=None):
    """單一股票從成交日 e0 開盤到下一成交日 e1 開盤的報酬因子（買進日 C/O、中間 ×G、e1 當天 ×G×O/C）。"""
    G = GROW if G is None else G
    if not (TRADED[e0, j] and np.isfinite(On[e0, j])): return np.nan
    f = Cn[e0, j] / On[e0, j]
    f *= np.prod(G[e0 + 1:e1, j])
    if TRADED[e1, j] and np.isfinite(On[e1, j]): f *= G[e1, j] * On[e1, j] / Cn[e1, j]
    return f

def window_factors_all(e0, e1, G=None):
    G = GROW if G is None else G
    with np.errstate(invalid='ignore', divide='ignore'):
        f = np.where(TRADED[e0] & np.isfinite(On[e0]), Cn[e0] / On[e0], np.nan)
        f = f * np.prod(G[e0 + 1:e1], axis=0)
        last = np.where(TRADED[e1] & np.isfinite(On[e1]), G[e1] * On[e1] / Cn[e1], 1.0)
    return f * last

# ---------------- 漲停 ----------------
def tick(p):
    return np.select([p < 10, p < 50, p < 100, p < 500, p < 1000], [0.01, 0.05, 0.1, 0.5, 1.0], 5.0)
def limit_up_open(t):
    lim = 0.07 if IDX[t] < pd.Timestamp('2015-06-01') else 0.10
    with np.errstate(invalid='ignore', divide='ignore'):
        ref = np.where(np.nan_to_num(Fn[t], nan=0) == 1, PREVC[t], Cn[t] / (1 + Rn[t]))
        ref = np.where(np.isfinite(ref), ref, PREVC[t])
        raw = ref * (1 + lim); tk = tick(raw)
        lu = np.floor(raw / tk + 1e-9) * tk
    return np.isfinite(On[t]) & np.isfinite(lu) & (On[t] >= lu - 1e-6)

# ---------------- 統計 ----------------
def tstat(x):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    if len(x) < 3: return float('nan')
    s = x.std(ddof=1)
    return float(x.mean() / (s / math.sqrt(len(x)))) if s > 0 else float('nan')

def win_ret(nav, days):
    D = pd.DatetimeIndex([d for d in days if d in nav.index])
    nd = nav.reindex(D)
    return (nd.shift(-1) / nd - 1).iloc[:-1]

def cagr_mdd(nav, a, b):
    s = nav[(nav.index >= pd.Timestamp(a)) & (nav.index <= pd.Timestamp(b))].dropna()
    if len(s) < 5: return float('nan'), float('nan')
    yrs = (s.index[-1] - s.index[0]).days / 365.25
    return float((s.iloc[-1] / s.iloc[0]) ** (1 / yrs) - 1), float((s / s.cummax() - 1).min())

def rel_stats(nav, bench, days, a, b):
    """訊號窗（起點落在 [a, b]）月超額：均值 pp、t、追蹤誤差（年化 %）、IR；日頻年化與回撤（a～b）。"""
    ra = win_ret(nav, days); rb = win_ret(bench.reindex(nav.index).ffill(), days)
    x = (ra - rb.reindex(ra.index)); x = x[(x.index >= pd.Timestamp(a)) & (x.index <= pd.Timestamp(b))].dropna()
    te = float(x.std(ddof=1) * math.sqrt(12)) if len(x) > 2 else float('nan')
    cg, mdd = cagr_mdd(nav, a, b); cb, mb = cagr_mdd(bench.reindex(nav.index).ffill(), a, b)
    return dict(n=int(len(x)), mean_pp=float(x.mean() * 100), t=tstat(x), TE=te * 100, IR=float(x.mean() * 12 / te) if te > 0 else float('nan'),
                CAGR=cg * 100, maxDD=mdd * 100, bench_CAGR=cb * 100, bench_maxDD=mb * 100, ann_excess_pp=(cg - cb) * 100)

def excess_series(nav, bench, days, a, b):
    ra = win_ret(nav, days); rb = win_ret(bench.reindex(nav.index).ffill(), days)
    x = (ra - rb.reindex(ra.index)); return x[(x.index >= pd.Timestamp(a)) & (x.index <= pd.Timestamp(b))].dropna()

def dumps(o): return json.dumps(o, ensure_ascii=False, indent=1, default=lambda z: float(z) if isinstance(z, (np.floating, np.integer)) else str(z))
