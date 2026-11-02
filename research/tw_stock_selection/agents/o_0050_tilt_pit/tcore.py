#!/usr/bin/env python3
"""o_0050_tilt_pit 策略核心：訊號（52 週比、近 3 月營收年增率已公告值）、PIT 成分與市值、分層傾斜權重、月頻引擎、統計。
口徑：
  訊號日 d（j 平移：營收期限日 D_m 後第 j 個面板交易日，j＝0..12；官方日＝每月第一個 ≥ 11 日）收盤算權重，下一交易日 e 開盤成交。
  評估窗：d_t 收盤 → d_{t+1} 收盤（0050 含息同窗）。d_t 收盤 → e_t 開盤由舊持股承擔（GX×O/C），e_t 開盤換股（成本），e_t 開盤 → d_{t+1} 收盤由新持股承擔。
  基準 B＝同一宇宙 PIT 市值加權（同引擎、同成本）。T1 宇宙＝e 日開盤適用的臺灣50 成分；T2＝T1 ∪ d 日 PIT 市值前 100 名（非 DR、近 6 日有成交）。
  分層：依宇宙內市值名次。2330 固定；第 2～5 名；第 6～50 名；第 51 名以後（T2）；金融保險股不論名次一律照權重、不傾斜。
  傾斜：層內 w'＝w×exp(λ z)，再等比例縮放回該層原始總權重；z＝層內分數排名的標準化（r 取 1..n，z＝(r−(n+1)/2)/√((n²−1)/12)），
        分數＝0.5×層內 52 週比百分位＋0.5×層內營收 3 月年增率百分位（缺任一項 → z＝0，倍數 1）；同分先比營收、再比代碼。
        cap 版：倍數（相對原始權重）限制在 [0.25, 2]，以反覆截斷＋重新縮放求解。層內可傾斜檔數 < 2 → 不傾斜。
"""
import sys, json, math
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
import tlib as L
RAW, DATA = L.RAW, L.DATA
IDX, COLS, CI, N = L.IDX, L.COLS, L.CI, L.N
J2330 = CI['2330']
FEE = 0.001425 * 0.6; TAX = 0.003
def rates(cost):
    if cost == 'normal': return FEE + 0.001, FEE + TAX + 0.001
    if cost == 'stress': return 0.005, 0.005
    if cost == 'zero': return 0.0, 0.0
    raise ValueError(cost)

# ---------------- PIT 股數與市值 ----------------
_S = pd.read_parquet(DATA / 'shares_pit.parquet').reindex(index=IDX, columns=COLS)
SH = _S.to_numpy()
def mcap_row(t): return SH[t] * L.CFF[t]

# ---------------- 滾動訊號（gapfix，同 cmlib.rolling_signals）----------------
EXEMPT_DAYS = [pd.Timestamp('2017-08-03'), pd.Timestamp('2018-07-12')]
EXEMPT_RANGE = (pd.Timestamp('2024-02-15'), pd.Timestamp('2024-03-15'))
def _rolling():
    g = np.where(L.GOOD)[0]
    A = np.where(L.TRADED, np.cumprod(L.GROW, axis=0), np.nan)[g]; Vv = L.Vn[g]
    Rg = np.where(L.TRADED, L.GROW - 1, np.nan)[g]
    dg = IDX[g]
    ex = np.isin(dg, pd.DatetimeIndex(EXEMPT_DAYS)) | ((dg >= EXEMPT_RANGE[0]) & (dg <= EXEMPT_RANGE[1]))
    missA = pd.DataFrame(np.isnan(A) & ~ex[:, None]).astype(np.int16)
    missV = pd.DataFrame(np.isnan(Vv) & ~ex[:, None]).astype(np.int16)
    okA = (missA.rolling(252, min_periods=252).sum().to_numpy() == 0)
    okV = (missV.rolling(60, min_periods=60).sum().to_numpy() == 0)
    hi = np.where(okA, pd.DataFrame(A).rolling(252, min_periods=1).max().to_numpy(), np.nan)
    v60 = np.where(okV, pd.DataFrame(Vv).rolling(60, min_periods=1).mean().to_numpy(), np.nan)
    sig = pd.DataFrame(Rg).rolling(60, min_periods=40).std().ffill().to_numpy()
    with np.errstate(invalid='ignore', divide='ignore'): s52 = A / hi
    S52 = np.full((len(IDX), N), np.nan); V60 = np.full((len(IDX), N), np.nan); SIG = np.full((len(IDX), N), np.nan)
    S52[g] = s52; V60[g] = v60; SIG[g] = sig
    return S52, pd.DataFrame(V60).ffill().to_numpy(), pd.DataFrame(SIG).ffill().to_numpy()
S52, V60, SIG = _rolling()

# ---------------- 營收（V4 已公告值；同 cmlib）----------------
SWITCH = pd.Timestamp('2021-03-01')
def _ly(f):
    Y = pd.read_parquet(RAW / 'agents' / f); Y.index = pd.PeriodIndex(Y.index, freq='M'); return Y.reindex(columns=COLS)
YL = _ly('yoy_long_R_FIX.parquet'); Y5 = _ly('yoy_rev5y_FIX.parquet')
def _y3(Y, m):
    ms = [m - 1, m - 2, m - 3]
    if any(x not in Y.index for x in ms): return np.full(N, np.nan)
    Z = Y.loc[ms].to_numpy(); out = Z.mean(axis=0); out[np.isnan(Z).any(axis=0)] = np.nan; return out
def y3(d, back=0):
    Y = Y5 if d >= SWITCH else YL
    return _y3(Y, pd.Period(d, 'M') - back)
_PD = json.loads((RAW / 'agents' / 'p_delay.json').read_text(encoding='utf-8'))
INS15 = np.array([c in set(_PD['INS15']) for c in COLS])
_FS = pd.read_csv(RAW / 'agents' / 'firstseen_2026-09.csv', dtype=str, index_col=0)
ANN_OK_2609 = np.array([c in set(_FS.index[_FS['cat'] == 'A_before']) for c in COLS])
def unavail(d):
    m = pd.Period(d, 'M'); rm = str(m - 1)
    if rm == '2026-09': u = ~ANN_OK_2609
    elif L.deadline(m) >= pd.Timestamp(d).normalize(): u = np.ones(N, bool)
    else: u = np.zeros(N, bool)
    if '2026-01' <= rm <= '2026-08': u = u | INS15
    return u
def y3_ab(d):
    a = y3(d); U = unavail(d); return np.where(U, y3(d, 1), a)

# ---------------- 金融、DR、產業 ----------------
SL = pd.read_csv(RAW / 'screener' / 'stock_list.csv', dtype=str).drop_duplicates('code').set_index('code')
_DL = pd.read_csv(RAW / 'agents' / 'delist_official.csv', dtype=str); DL_NAME = dict(zip(_DL.code, _DL.name))
FIN_WORDS = ('銀', '證', '金控', '證金', '產保', '壽', '票券', '產險', '人壽', '投信', '合庫', '信託')
def _is_fin(c):
    if SL['industry'].get(c) == '金融保險業' or c[:2] == '28' or c in ('5876', '5880'): return True
    n = DL_NAME.get(c, ''); n = n if isinstance(n, str) else ''
    return any(w in n for w in FIN_WORDS)
FINMASK = np.array([_is_fin(c) for c in COLS])
DRMASK = np.array([c.startswith('91') for c in COLS])
ELEC_IND = {'電子零組件業', '半導體業', '光電業', '電腦及週邊設備業', '其他電子業', '通信網路業', '資訊服務業', '電子通路業', '數位雲端'}
IND = np.array([SL['industry'].get(c, '下市/未知') if isinstance(SL['industry'].get(c, None), str) else '下市/未知' for c in COLS])
ELECMASK = np.array([i in ELEC_IND for i in IND])

# ---------------- PIT 成分 ----------------
_TW = json.loads((RAW / 'agents' / 'tw50_members.json').read_text(encoding='utf-8'))
_TW_STEPS = [(pd.Timestamp(s['eff']), s['members_before']) for s in _TW['steps_backward']]
TW50_EARLIEST = pd.Timestamp(_TW['earliest_valid_eff'])
_MP = {pd.Timestamp(k): v for k, v in json.loads((DATA / 'membership_pit.json').read_text(encoding='utf-8')).items()}
def tw50_at(day):
    """day 開盤時適用的成分（生效日收盤後換新名單：eff < day → 新名單）。"""
    day = pd.Timestamp(day)
    if day <= TW50_EARLIEST:
        ks = [k for k in sorted(_MP) if k < day]
        return list(_MP[ks[-1]]) if ks else None
    S = _TW['current_2026-10-09']
    for eff, mb in _TW_STEPS:
        if day <= eff: S = mb
    return list(S)

# ---------------- 權重 ----------------
def zrank(score_rev, score52, codes_idx):
    """回傳層內 z（缺值 → 0）。"""
    n = len(codes_idx)
    ok = np.isfinite(score_rev) & np.isfinite(score52)
    z = np.zeros(n)
    m = int(ok.sum())
    if m < 2: return z, m
    r52 = pd.Series(score52[ok]).rank(pct=True).to_numpy(); rr = pd.Series(score_rev[ok]).rank(pct=True).to_numpy()
    sc = 0.5 * r52 + 0.5 * rr
    keys = list(zip(-sc, -score_rev[ok], [COLS[j] for j in codes_idx[ok]]))
    order = sorted(range(m), key=lambda i: keys[i])          # 分數高 → rank m（z 最大）
    rk = np.empty(m); rk[order] = np.arange(m, 0, -1)
    zz = (rk - (m + 1) / 2) / math.sqrt((m * m - 1) / 12)
    z[ok] = zz
    return z, m

def tilt(w, z, lam, cap):
    """層內傾斜：w'∝w·exp(λz)，縮放回 Σw；cap：倍數夾在 [0.25, 2]。"""
    if lam == 0 or len(w) < 2: return w.copy()
    tot = w.sum(); m = np.exp(lam * z)
    if not cap:
        wp = w * m; return wp * tot / wp.sum()
    k = m.copy()
    for _ in range(100):
        s = tot / (w * k).sum(); kk = np.clip(k * s, 0.25, 2.0)
        fixed = (kk <= 0.25 + 1e-12) | (kk >= 2.0 - 1e-12)
        if fixed.all(): k = kk; break
        rem = tot - (w[fixed] * kk[fixed]).sum(); free = ~fixed
        s2 = rem / (w[free] * k[free]).sum()
        kn = np.where(fixed, kk, np.clip(k * s2, 0.25, 2.0))
        if np.allclose(kn, k, rtol=1e-12, atol=1e-14): k = kn; break
        k = kn
    wp = w * k
    return wp * tot / wp.sum()

def universe(d, e, kind):
    t = IDX.get_loc(d)
    mem = tw50_at(e)
    mem = [CI[c] for c in mem if c in CI]
    mc = mcap_row(t)
    alive = L.TRADED[max(t - 5, 0):t + 1].any(axis=0)
    if kind == 'T1':
        U = np.array(sorted(set(mem)), int)
    else:
        elig = np.where(~DRMASK & np.isfinite(mc) & alive, mc, -np.inf)
        top = np.argsort(-elig)[:100]; top = top[np.isfinite(elig[top])]
        U = np.array(sorted(set(mem) | set(top.tolist())), int)
    U = U[np.isfinite(mc[U]) & (mc[U] > 0) & L.TRADED[max(t - 5, 0):t + 1, U].any(axis=0)]
    return U, mc[U]

TIER_EDGES = [(1, 1, '2330'), (2, 5, 'R2_5'), (6, 50, 'R6_50'), (51, 10 ** 6, 'R51p')]
def target(d, e, kind, lam, cap, scores=None, rng=None, info=False):
    """回傳 (U, w_base, w_tilt[, info])。scores：可傳入 (rev, s52) 覆寫（安慰劑用）；rng：若給定則在層內隨機排列 z。"""
    t = IDX.get_loc(d)
    U, mc = universe(d, e, kind)
    wb = mc / mc.sum()
    order = np.argsort(-mc); rank = np.empty(len(U), int); rank[order] = np.arange(1, len(U) + 1)
    rev = y3_ab(d)[U] if scores is None else scores[0][U]
    s52 = S52[t, U] if scores is None else scores[1][U]
    wt = wb.copy(); tiers = {}
    fin = FINMASK[U]
    for lo, hi, nm in TIER_EDGES:
        m = (rank >= lo) & (rank <= hi) & ~fin
        if nm == '2330': m = m & (U == J2330)
        idx = np.where(m)[0]
        tiers[nm] = idx
        if nm == '2330' or len(idx) < 2: continue
        z, nok = zrank(rev[idx], s52[idx], U[idx])
        if rng is not None:
            okz = z != 0
            zz = z[okz].copy(); rng.shuffle(zz); z = z.copy(); z[okz] = zz
        wt[idx] = tilt(wb[idx], z, lam, cap)
    # 2330 不在名次 1 時（理論上不會）：2330 仍固定權重
    tiers['FIN'] = np.where(fin)[0]
    if info: return U, wb, wt, {'tiers': tiers, 'rank': rank}
    return U, wb, wt

# ---------------- 月頻引擎 ----------------
GX = L.GX
LOGG = np.log(np.where(GX > 0, GX, 1.0)); CUMG = np.vstack([np.zeros((1, N)), np.cumsum(LOGG, axis=0)])   # CUMG[t+1]-CUMG[a] = Σ_{a..t} log G
def factor_open_to_close(e, d2):
    """e 開盤 → d2 收盤：C[e]/O[e] × Π_{e+1..d2} GX。e 日沒開盤 → NaN。"""
    with np.errstate(invalid='ignore', divide='ignore'):
        f = np.where(L.TRADED[e] & np.isfinite(L.On[e]) & (L.On[e] > 0), L.Cn[e] / L.On[e], np.nan)
    return f * np.exp(CUMG[d2 + 1] - CUMG[e + 1])
def factor_close_to_open(d, e):
    """d 收盤 → e 開盤（e＝d+1）：GX[e]×O[e]/C[e]；e 沒開盤 → 1（凍結，留到之後）。"""
    with np.errstate(invalid='ignore', divide='ignore'):
        f = np.where(L.TRADED[e] & np.isfinite(L.On[e]) & (L.On[e] > 0), GX[e] * L.On[e] / L.Cn[e], np.nan)
    return f
def factor_close_to_close(a, b):
    return np.exp(CUMG[b + 1] - CUMG[a + 1])

def run(days, plans, cost='normal', record=False):
    """days：訊號日序列；plans：{d: (idx, w)}（w 和為 1）。回傳每窗報酬 Series（index＝d_t）與周轉、成本。
    凍結：e 日沒開盤的持股保留（不可交易），其餘換股；下市（LAST_TRADE < e）持股以最後收盤價值扣賣出成本轉現金。"""
    buy, sell = rates(cost)
    ti = [IDX.get_loc(d) for d in days]
    h = np.zeros(N); cash = 1.0
    out = []; tov = []; cst = []; W = {}
    for k in range(len(ti) - 1):
        t, t2 = ti[k], ti[k + 1]; e = t + 1
        V0 = h.sum() + cash
        if V0 <= 0: V0 = 1e-12
        h = h / V0; cash = cash / V0     # 每窗正規化成 1
        # d 收盤 → e 開盤
        fo = factor_close_to_open(t, e)
        tr = np.isfinite(fo)
        dead = (h > 0) & (L.LAST_TRADE < e)
        cash += (h[dead] * (1 - sell)).sum(); h[dead] = 0.0
        frz = (h > 0) & ~tr
        h = np.where(tr, h * np.nan_to_num(fo, nan=1.0), h)
        Vfz = h[frz].sum(); htr = np.where(frz, 0.0, h); Vt = htr.sum() + cash
        idx, w = plans[days[k]]
        wv = np.zeros(N); np.add.at(wv, idx, w)
        wv[frz] = 0.0; wv[~tr] = 0.0
        wv = wv / wv.sum() if wv.sum() > 0 else wv
        # 成本：T + c(T) = Vt，兩次固定點
        T = Vt
        for _ in range(3):
            tgt = wv * T
            c = buy * np.clip(tgt - htr, 0, None).sum() + sell * np.clip(htr - tgt, 0, None).sum()
            T = Vt - c
        tgt = wv * T
        tov.append(float(np.clip(htr - tgt, 0, None).sum() / max(Vt + Vfz, 1e-12))); cst.append(float(Vt - T))
        # e 開盤 → d2 收盤
        f2 = factor_open_to_close(e, t2)
        newh = tgt * np.nan_to_num(f2, nan=1.0)
        hz = np.where(frz, h * factor_close_to_close(e - 1, t2) / np.where(frz, np.nan_to_num(1.0, nan=1.0), 1.0), 0.0)
        h = newh + np.where(frz, hz, 0.0); cash = 0.0
        out.append(h.sum() + cash - 1.0)
        if record: W[days[k]] = (tgt / max(T, 1e-12))
    s = pd.Series(out, index=pd.DatetimeIndex(days[:-1]))
    return s, {'turnover': np.array(tov), 'cost': np.array(cst), 'W': W}

def bench_win(days):
    e = L.E50.to_numpy(); ti = [IDX.get_loc(d) for d in days]
    return pd.Series([e[b] / e[a] - 1 for a, b in zip(ti[:-1], ti[1:])], index=pd.DatetimeIndex(days[:-1]))

def stats(x):
    x = pd.Series(x).dropna()
    if len(x) < 3: return dict(n=len(x), mean_pp=float('nan'), t=float('nan'), TE=float('nan'), IR=float('nan'))
    te = float(x.std(ddof=1) * math.sqrt(12))
    return dict(n=int(len(x)), mean_pp=float(x.mean() * 100), t=L.tstat(x), TE=te * 100, IR=float(x.mean() * 12 / te) if te > 0 else float('nan'))

def cagr(r):
    r = pd.Series(r).dropna()
    if len(r) == 0: return float('nan')
    return float(np.prod(1 + r) ** (12 / len(r)) - 1)
def mdd(r):
    v = np.cumprod(1 + pd.Series(r).dropna().to_numpy()); return float((v / np.maximum.accumulate(v) - 1).min()) if len(v) else float('nan')

def sig_days(j, start, end):
    """j＝'off' 為官方日；整數 j 為 jscan。多取一個訊號日當最後一窗的終點。"""
    if j == 'off': ds = L.official_days(start)
    else: ds = L.jscan_days(int(j), start)
    ds = [d for d in ds if d <= pd.Timestamp(end) or True]
    a = [d for d in ds if pd.Timestamp(start) <= d <= pd.Timestamp(end)]
    nxt = [d for d in ds if d > pd.Timestamp(end)]
    return a + nxt[:1]

# ---------------- 安慰劑（隨機分數；層內排列 z；不含成本＝毛報酬，用來檢驗結構拖累）----------------
def tilt_mat(w, Z, lam, cap):
    """Z：P×n（各組排列後的 z）。回傳 P×n 權重，每列和＝w.sum()。"""
    tot = w.sum(); M = np.exp(lam * Z)
    if not cap:
        Wp = w[None, :] * M; return Wp * (tot / Wp.sum(axis=1, keepdims=True))
    K = M.copy()
    for _ in range(60):
        s = tot / (w[None, :] * K).sum(axis=1, keepdims=True)
        KK = np.clip(K * s, 0.25, 2.0)
        fixed = (KK <= 0.25 + 1e-12) | (KK >= 2.0 - 1e-12)
        rem = tot - (w[None, :] * KK * fixed).sum(axis=1, keepdims=True)
        den = (w[None, :] * K * ~fixed).sum(axis=1, keepdims=True)
        s2 = np.where(den > 0, rem / np.where(den > 0, den, 1), 1.0)
        Kn = np.where(fixed, KK, np.clip(K * s2, 0.25, 2.0))
        if np.allclose(Kn, K, rtol=1e-10, atol=1e-13): K = Kn; break
        K = Kn
    Wp = w[None, :] * K
    return Wp * (tot / Wp.sum(axis=1, keepdims=True))

_PRE = {}
def precompute(days, kind):
    key = (tuple(days), kind)
    if key in _PRE: return _PRE[key]
    out = []
    for d in days[:-1]:
        e = IDX[IDX.get_loc(d) + 1]
        U, wb, wt, inf = target(d, e, kind, 0.0, False, info=True)
        t = IDX.get_loc(d)
        rev = y3_ab(d)[U]; s52 = S52[t, U]
        tz = {}
        for nm in ('R2_5', 'R6_50', 'R51p'):
            idx = inf['tiers'][nm]
            if len(idx) < 2: continue
            z, nok = zrank(rev[idx], s52[idx], U[idx]); tz[nm] = (idx, z)
        out.append((d, U, wb, tz))
    _PRE[key] = out
    return out

def gross_matrix(days, kind, lam, cap, P=1000, seed=0, real=False):
    """回傳 (R_perm P×T, R_base T, R_real T)：毛報酬（零成本）每窗報酬。real=True 時 P 列全用真實 z（只取一列）。"""
    pre = precompute(days, kind)
    rng = np.random.default_rng(seed)
    ti = [IDX.get_loc(d) for d in days]
    Pn = 1 if real else P
    H = np.zeros((Pn, N)); cash = np.ones(Pn); Hb = np.zeros(N); cashb = 1.0
    R = np.zeros((Pn, len(ti) - 1)); Rb = np.zeros(len(ti) - 1)
    for k in range(len(ti) - 1):
        t, t2 = ti[k], ti[k + 1]; e = t + 1
        d, U, wb, tz = pre[k]
        fo = factor_close_to_open(t, e); tr = np.isfinite(fo)
        f2 = np.nan_to_num(factor_open_to_close(e, t2), nan=1.0); fcc = factor_close_to_close(t, t2)
        dead = L.LAST_TRADE < e
        # 基準
        V0 = Hb.sum() + cashb; Hb = Hb / V0; cashb = cashb / V0
        cashb += Hb[dead].sum(); Hb[dead] = 0.0
        frz = (Hb > 0) & ~tr
        Hb = np.where(tr, Hb * np.nan_to_num(fo, nan=1.0), Hb)
        Vfz = Hb[frz].sum(); Vt = Hb[~frz].sum() + cashb
        wv = np.zeros(N); wv[U] = wb; wv[frz | ~tr] = 0.0; wv /= wv.sum()
        Hb = wv * Vt * f2 + np.where(frz, Hb * fcc, 0.0); cashb = 0.0
        Rb[k] = Hb.sum() - 1.0
        # 傾斜（P 組）
        V0 = H.sum(axis=1) + cash; H = H / V0[:, None]; cash = cash / V0
        cash = cash + H[:, dead].sum(axis=1); H[:, dead] = 0.0
        FRZ = (H > 0) & ~tr[None, :]
        H = np.where(tr[None, :], H * np.nan_to_num(fo, nan=1.0)[None, :], H)
        Vfz = np.where(FRZ, H, 0.0).sum(axis=1); Vt = np.where(FRZ, 0.0, H).sum(axis=1) + cash
        Wt = np.tile(wb, (Pn, 1))
        for nm, (idx, z) in tz.items():
            okz = z != 0
            if real:
                Zm = z[None, :]
            else:
                Zm = np.tile(z, (Pn, 1))
                if okz.sum() >= 2:
                    sub = Zm[:, okz]; sub = rng.permuted(sub, axis=1); Zm[:, okz] = sub
            Wt[:, idx] = tilt_mat(wb[idx], Zm, lam, cap)
        WV = np.zeros((Pn, N)); WV[:, U] = Wt
        WV[FRZ] = 0.0; WV[:, ~tr] = 0.0
        WV = WV / WV.sum(axis=1, keepdims=True)
        H = WV * (Vt[:, None]) * f2[None, :] + np.where(FRZ, H * fcc[None, :], 0.0); cash = np.zeros(Pn)
        R[:, k] = H.sum(axis=1) - 1.0
    idx = pd.DatetimeIndex(days[:-1])
    return R, pd.Series(Rb, index=idx)
