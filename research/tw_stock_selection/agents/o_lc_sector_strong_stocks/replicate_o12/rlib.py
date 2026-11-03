#!/usr/bin/env python3
"""o12 獨立複製：題 2 產業輪動＋強勢大型股。只依凍結規格與 BRIEF 所列資料，自寫。
口徑（自行決定、照規格文字）：
  訊號日 d＝營收期限日 D_m（第一個加權指數交易日 ≥ 每月 10 日；2019-02→02-15、2023-04→04-12）後第 j 個加權指數交易日（j=0 即 D_m），
         d 須為面板 GOOD 日（有值檔數 ≥300），且 d 的下一面板日 e 也須 GOOD（AMEND v1）。
  評估窗：d_t 收盤 → d_{t+1} 收盤；d→e 開盤由舊持股承擔，e 開盤換股（成本），e 開盤 → d_{t+1} 收盤由新持股承擔；0050 同窗收盤到收盤。
"""
import json
from pathlib import Path
import numpy as np, pandas as pd

AG = Path('/home/user/-/research/tw_stock_selection/agents')
V7 = AG / 'o_v7_listing_exec_engine' / 'data'
O8 = AG / 'o_0050_tilt_pit'
T1 = AG / 'o_lc_sector_rotation_pit'
ME = AG / 'o_lc_sector_strong_stocks'
SCR = Path('/tmp/claude-0/-home-user--/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/panel')

C = pd.read_parquet(V7 / 'close_raw.parquet')
IDX = C.index; COLS = list(C.columns); N = len(COLS); CI = {c: i for i, c in enumerate(COLS)}
Cn = C.to_numpy().copy()
ADJ = pd.read_parquet(V7 / 'adj.parquet').reindex(index=IDX, columns=COLS).to_numpy().copy()
On = pd.read_parquet(V7 / 'open.parquet').reindex(index=IDX, columns=COLS).to_numpy().copy()
VAL = pd.read_parquet(SCR / 'value.parquet').reindex(index=IDX, columns=COLS).to_numpy().copy()
# 下市股補價（o_infra_v3_rebuild/delist_prices.parquet；只補 V7 面板缺值格；同 o8 tlib 作法；NaN 報酬視為 0）
def _patch_delist():
    dp = pd.read_parquet(AG / 'o_infra_v3_rebuild' / 'delist_prices.parquet')
    dp['date'] = pd.to_datetime(dp['date']); dp = dp[dp.code.isin(CI) & dp.date.isin(IDX)].sort_values(['code', 'date'])
    n = 0
    for c, g in dp.groupby('code'):
        i = CI[c]
        for r in g.itertuples():
            t = POS0[r.date]
            if np.isfinite(Cn[t, i]): continue
            Cn[t, i] = r.close; On[t, i] = r.open
            if not np.isfinite(VAL[t, i]): VAL[t, i] = r.value
            prev = np.where(np.isfinite(ADJ[:t, i]))[0]
            base = ADJ[prev[-1], i] if len(prev) else r.close
            ADJ[t, i] = base * (1 + (r.ret if np.isfinite(r.ret) else 0.0)); n += 1
    return n
POS0 = {d: i for i, d in enumerate(IDX)}
PATCHED = _patch_delist()
TRADED = np.isfinite(Cn) & np.isfinite(ADJ)
NOBS = TRADED.sum(axis=1); GOOD = NOBS >= 300
CFF = pd.DataFrame(np.where(TRADED, Cn, np.nan)).ffill().to_numpy()
# ---- 報酬用還原指數（EXFIX）：V7 在 flag==1 日 ret 一律 0（2004～2021-09 約 2.4 萬格，含 2330 每次配息）。
# 改成：當日成長＝收盤/開盤 ×（1＋當日非事件股開盤跳空中位數）（假設開盤已除息、隔夜跳空同市場）。訊號（52 週比、RS）仍用 V7 adj。
FLAGn = pd.read_parquet(V7 / 'flag.parquet').reindex(index=IDX, columns=COLS).to_numpy()
EXFIX = True
import os
SIG_EXFIX = os.environ.get('SIGEX', '0') == '1'   # 敏感度：訊號也用 EXFIX 還原指數
def _adj_ret():
    with np.errstate(invalid='ignore', divide='ignore'):
        A = np.where(TRADED, ADJ, np.nan)
        Aff = pd.DataFrame(A).ffill().to_numpy()
        g = np.ones_like(A)
        g[1:] = np.where(TRADED[1:] & np.isfinite(Aff[:-1]), A[1:] / Aff[:-1], 1.0)
        g = np.where(np.isfinite(g), g, 1.0)
        if EXFIX:
            Cff = pd.DataFrame(np.where(TRADED, Cn, np.nan)).ffill().to_numpy()
            prevc = np.vstack([np.full((1, N), np.nan), Cff[:-1]])
            gap = On / prevc - 1
            fl = np.nan_to_num(FLAGn, nan=0)
            normal = TRADED & np.isfinite(gap) & (fl == 0)
            med = np.array([np.median(gap[t][normal[t]]) if normal[t].sum() > 50 else 0.0 for t in range(len(IDX))])
            fx = (fl == 1) & TRADED & np.isfinite(On) & (On > 0)
            alt = Cn / On * (1 + med[:, None])
            g[fx] = alt[fx]
        out = np.cumprod(g, axis=0)
    return np.where(TRADED, out, np.nan), int(fx.sum()) if EXFIX else 0
ADJR, NFIX = _adj_ret()
AFF = pd.DataFrame(ADJR).ffill().to_numpy()
# 調整後開盤價（同日因子）：open × adjr/close
with np.errstate(invalid='ignore', divide='ignore'):
    AOPEN = np.where(TRADED & np.isfinite(On) & (On > 0), On * ADJR / Cn, np.nan)

# 上市身分遮罩
_L = pd.read_parquet(V7 / 'listed_twse.parquet').reindex(columns=COLS).astype(float)
LISTED = _L.reindex(IDX.union(_L.index)).sort_index().ffill().reindex(IDX).fillna(1.0).to_numpy() > 0.5

# 股數：2013-01-02 起 o8 shares_pit（短缺口 ffill≤10）＋4763／2317 修正；之前 o7 shares_est
def _shares():
    S = pd.read_parquet(O8 / 'data' / 'shares_pit.parquet').reindex(columns=COLS)
    S = S.reindex(IDX.union(S.index)).sort_index().ffill(limit=10).reindex(IDX)
    ix = S.index
    a = (ix >= '2019-09-04') & (ix < '2023-07-27'); b = (ix >= '2023-07-27') & (ix < '2025-06-30')
    S.loc[a, '4763'] = S.loc[a, '4763'] / 12.65; S.loc[b, '4763'] = S.loc[b, '4763'] / 10.0
    pre = float(S.loc[:'2018-06-29', '2317'].dropna().iloc[-1])
    m = (ix > '2018-06-29') & (ix <= '2018-10-25'); S.loc[m, '2317'] = pre
    E = pd.read_parquet(O8 / 'raw' / 'agents' / 'capmatched_shares_est.parquet').reindex(index=IDX, columns=COLS)
    early = ix < pd.Timestamp('2013-01-02')
    S[early] = E[early]
    return S.to_numpy()
SH = _shares()
def mcap(t): return SH[t] * CFF[t]

DR = np.array([c.startswith('91') for c in COLS])

# ---------------- 產業（題 1 raw/industry_pit.csv；2007-07-02 前電子 8 類→電子工業等） ----------------
_IP = pd.read_csv(T1 / 'raw' / 'industry_pit.csv', dtype=str)
ELEC8 = {'半導體業', '電腦及週邊設備業', '光電業', '通信網路業', '電子零組件業', '電子通路業', '資訊服務業', '其他電子業'}
def industry_at(d):
    d = pd.Timestamp(d); out = np.array(['未知'] * N, dtype=object)
    for r in _IP.itertuples():
        if r.code not in CI: continue
        s = pd.Timestamp(r.start) if isinstance(r.start, str) else pd.Timestamp('1900-01-01')
        e = pd.Timestamp(r.end) if isinstance(r.end, str) else pd.Timestamp('2100-01-01')
        if s <= d <= e: out[CI[r.code]] = r.industry
    if d < pd.Timestamp('2007-07-02'):
        for i, x in enumerate(out):
            if x in ELEC8: out[i] = '電子工業'
            elif x in ('化學工業', '生技醫療業'): out[i] = '化學生技醫療業'
            elif x == '油電燃氣業': out[i] = '其他業'
    return out
_IPC = {}
def ind_cached(d):
    # 改類只在少數日期；以 (年, 是否 <2007-07-02) 粗快取不安全 → 以 start/end 斷點集合快取
    key = int(np.searchsorted(_BRK, np.datetime64(pd.Timestamp(d)), side='right'))
    if key not in _IPC: _IPC[key] = industry_at(d)
    return _IPC[key]
_b = set(['2007-07-02'])
for r in _IP.itertuples():
    if isinstance(r.start, str): _b.add(r.start)
    if isinstance(r.end, str): _b.add(str((pd.Timestamp(r.end) + pd.Timedelta(days=1)).date()))
_BRK = np.array(sorted(pd.to_datetime(list(_b))), dtype='datetime64[ns]')

def finmask(ind):
    return np.array([(ind[i] == '金融保險業') or COLS[i].startswith('28') or COLS[i] in ('5876', '5880') for i in range(N)])

# ---------------- 訊號 ----------------
EXEMPT = pd.DatetimeIndex(['2017-08-03', '2018-07-12'])
def _rolling():
    g = np.where(GOOD)[0]; A = np.where(TRADED, ADJR if SIG_EXFIX else ADJ, np.nan)[g]; dg = IDX[g]
    ex = dg.isin(EXEMPT) | ((dg >= '2024-02-15') & (dg <= '2024-03-15'))
    miss = pd.DataFrame(np.isnan(A) & ~ex[:, None]).astype(np.int16)
    ok = miss.rolling(252, min_periods=252).sum().to_numpy() == 0
    hi = pd.DataFrame(A).rolling(252, min_periods=1).max().to_numpy()
    with np.errstate(invalid='ignore', divide='ignore'):
        s52 = np.where(ok, A / hi, np.nan)
        rs = np.full_like(A, np.nan)
        rs[126:] = A[126 - 21:-21] / A[:-126] - 1.0  # A[g-21]/A[g-126]-1
    S52 = np.full((len(IDX), N), np.nan); RS = np.full((len(IDX), N), np.nan)
    S52[g] = s52; RS[g] = rs
    # 非 GOOD 日（如 2018-01-13 殘列）沿用前一 GOOD 日訊號
    S52 = pd.DataFrame(S52).where(pd.Series(GOOD).to_numpy()[:, None]).ffill(limit=1).to_numpy() if False else _ffill_bad(S52)
    RS = _ffill_bad(RS)
    return S52, RS
def _ffill_bad(X):
    X = X.copy()
    for t in np.where(~GOOD)[0]:
        if t > 0: X[t] = X[t - 1]
    return X
S52, RS = _rolling()

TX = pd.DatetimeIndex(sorted(set(pd.read_csv(O8 / 'raw' / 'bench' / 'taiex_twii.csv', parse_dates=['date'])['date'])))
SPECIAL = {'2019-02': pd.Timestamp('2019-02-15'), '2023-04': pd.Timestamp('2023-04-12')}
# 訊號日用「面板交易日」計數（規格原文）；2023-04 研究員的 J0＝04-17（由其輸出日期推得，規格未載）→ 參數化
PANEL_SPECIAL = {'2019-02': pd.Timestamp('2019-02-15'), '2023-04': pd.Timestamp('2023-04-17')}
def deadline(m):
    m = pd.Period(m, 'M'); ts = SPECIAL.get(str(m), pd.Timestamp(m.year, m.month, 10))
    k = TX.searchsorted(ts); return TX[k] if k < len(TX) else None

def _ly(f):
    Y = pd.read_parquet(O8 / 'raw' / 'agents' / f); Y.index = pd.PeriodIndex(Y.index, freq='M'); return Y.reindex(columns=COLS)
YL = _ly('yoy_long_R_FIX.parquet'); Y5 = _ly('yoy_rev5y_FIX.parquet')
_PD = json.loads((O8 / 'raw' / 'agents' / 'p_delay.json').read_text(encoding='utf-8'))
INS15 = np.array([c in set(_PD['INS15']) for c in COLS])
_FS = pd.read_csv(O8 / 'raw' / 'agents' / 'firstseen_2026-09.csv', dtype=str, index_col=0)
ANN2609 = np.array([c in set(_FS.index[_FS['cat'] == 'A_before']) for c in COLS])
def _y3(Y, m):
    ms = [m - 1, m - 2, m - 3]
    if any(x not in Y.index for x in ms): return np.full(N, np.nan)
    Z = Y.loc[ms].to_numpy(); o = Z.mean(axis=0); o[np.isnan(Z).any(axis=0)] = np.nan; return o
def y3_ab(d):
    d = pd.Timestamp(d); Y = Y5 if d >= pd.Timestamp('2021-03-01') else YL; m = pd.Period(d, 'M')
    a = _y3(Y, m); b = _y3(Y, m - 1); rm = str(m - 1)
    if rm == '2026-09': U = ~ANN2609
    elif deadline(m) >= d.normalize(): U = np.ones(N, bool)
    else: U = np.zeros(N, bool)
    if '2026-01' <= rm <= '2026-08': U = U | INS15
    return np.where(U, b, a)

# ---------------- 訊號日 ----------------
POS = {d: i for i, d in enumerate(IDX)}
def sig_days(j, m0='2006-04', m1='2026-10', mode='panel'):
    out = []
    for m in pd.period_range(m0, m1, freq='M'):
        if mode == 'panel':
            ts = PANEL_SPECIAL.get(str(m), pd.Timestamp(m.year, m.month, 10))
            k0 = IDX.searchsorted(ts)
            if k0 >= len(IDX) or IDX[k0].to_period('M') != m: continue
            t = k0 + j
            if t + 1 >= len(IDX): continue
            if not GOOD[t + 1]: continue        # AMEND v1：成交日須為 GOOD
            out.append(t); continue
        D = deadline(m)
        if D is None: continue
        k = TX.get_loc(D) + j
        if k >= len(TX): continue
        s = TX[k]
        if s not in POS: continue
        t = POS[s]
        if not GOOD[t] or t + 1 >= len(IDX) or not GOOD[t + 1]: continue
        out.append(t)
    return out

# ---------------- 宇宙、分數 ----------------
def universe(t):
    mc = mcap(t)
    rec = TRADED[max(0, t - 5):t + 1].any(axis=0)
    ok = np.isfinite(mc) & (mc > 0) & rec & LISTED[t] & ~DR
    idx = np.where(ok)[0]
    o = idx[np.argsort(-mc[idx], kind='stable')][:100]
    return o, mc

PCT = os.environ.get('PCT', 'avg')
def _rk(s):
    if PCT == 'avg': return s.rank(pct=True, method='average').to_numpy()
    if PCT == 'm1': r = s.rank(method='average'); n = s.notna().sum(); return ((r - 1) / (n - 1)).to_numpy()
    if PCT == 'min': return s.rank(pct=True, method='min').to_numpy()
    if PCT == 'nall': r = s.rank(method='average'); return (r / len(s)).to_numpy()
    raise ValueError(PCT)
def pct(x, U):
    v = x[U]; s = pd.Series(v); return _rk(s)

def scores(t, U, fin):
    d = IDX[t]; rev = y3_ab(d)
    p1 = pct(S52[t], U); p2 = pct(RS[t], U)
    rv = rev[U].copy()
    if os.environ.get('FINREV', '0') != '1': rv[fin[U]] = np.nan
    p3 = _rk(pd.Series(rv))
    P = np.vstack([p1, p2, p3]).T
    P[fin[U], 2] = np.nan
    cnt = np.isfinite(P).sum(axis=1)
    sc = np.where(cnt >= 2, np.nanmean(np.where(np.isfinite(P), P, np.nan), axis=1), np.nan)
    return sc, rev[U]

def order_key(sc, rv, U):
    """排序：分數高→營收高→代碼小。回傳 U 的排列（只含有分數者）。"""
    rr = np.where(np.isfinite(rv), rv, -np.inf)
    m = np.isfinite(sc)
    idx = np.where(m)[0]
    codes = np.array([COLS[U[i]] for i in idx])
    o = np.lexsort((codes, -rr[idx], -sc[idx]))
    return idx[o]

SECT = json.loads((ME / 'data' / 't1_sector_order.json').read_text(encoding='utf-8'))['order']

def cap_weights(base, cap):
    w = base / base.sum(); K = len(w)
    if K * cap < 1 - 1e-12: return np.full(K, 1.0 / K)
    fixed = np.zeros(K, bool)
    for _ in range(100):
        over = (w > cap + 1e-12) & ~fixed
        if not over.any(): break
        fixed |= over
        free = ~fixed
        rest = 1 - cap * fixed.sum()
        w = np.where(fixed, cap, base * rest / base[free].sum())
    return w
