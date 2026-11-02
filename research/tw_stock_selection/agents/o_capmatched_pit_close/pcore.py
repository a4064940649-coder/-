#!/usr/bin/env python3
"""o_capmatched_pit_close 共用核心：只換輸入、不換載體。

載體＝o7（o_capmatched_selection）的 cmlib／cmstrat，唯讀 import（python3 -I -B，不寫 pycache、不改 o7 檔案）。
本檔在 import cmstrat 之前，依 CONFIG 修補 cmlib 的模組全域變數：
  shares  : 'est'（o7 shares_est）｜'pit_i'（shares_pit，只在 shares_est 有值的格子；不加修正）｜
            'pit_ii'（同 pit_i，再加 4763／2317 修正）｜'pit_a'（全部 shares_pit＋修正；含下市股）｜'pit_b'（pit_a 落後 63 個交易日）
  delist  : False｜True（delist_prices 補 2019-05-17～2023 下市股價格，並以 V7 上市身分遮罩市值）
  memb_pre: 'o7'｜'pit'（2015-09-18 以前的臺灣50 規則重建名單：o7 membership_pre2015 或 o8 membership_pit）
快速引擎 sim_fast 與 cmlib.simulate_w 數學上相同（二分法改成分段線性精確解；換股之間用累積乘積），s1_gate.py 逐值核對。
"""
import os, sys, json, math, hashlib
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
AG = HERE.parent
O7 = AG / 'o_capmatched_selection'
O8 = AG / 'o_0050_tilt_pit'
DATA = HERE / 'data'; OUT = HERE / 'out'
def sha(f): return hashlib.sha256(Path(f).read_bytes()).hexdigest()

CONFIGS = {
    # P2 拆解的逐步替換（順序事前寫死）
    'o7':      dict(shares='est',    delist=False, memb_pre='o7'),
    'i':       dict(shares='pit_i',  delist=False, memb_pre='pit'),
    'ii':      dict(shares='pit_ii', delist=False, memb_pre='pit'),
    'a':       dict(shares='pit_a',  delist=True,  memb_pre='pit'),   # (iii) 納入下市股＝PIT 版本 (a)
    'b':       dict(shares='pit_b',  delist=True,  memb_pre='pit'),   # PIT 版本 (b)：落後 63 個交易日
}

sys.path.insert(0, str(O7))
import cmlib as L          # noqa: E402  （o7 原檔，唯讀）
sys.path.pop(0)
IDX, COLS, N, CI = L.IDX, L.COLS, L.N, L.CI

FIX4763 = dict(a0='2019-09-04', a1='2023-07-27', div_a=12.65, b1='2025-06-30', div_b=10.0)   # 同 replicate_o8/rcore.py（TDCC 2021 真值 58,575,605 股）
FIX2317 = dict(pre_until='2018-10-25', eff='2018-10-26', pre_val_date='2018-06-29')          # 2018-10-26 減資 20% 生效
LAG = 63

def shares_pit_raw():
    S = pd.read_parquet(O8 / 'data' / 'shares_pit.parquet')
    S = S.reindex(columns=COLS)
    S = S.reindex(IDX.union(S.index)).sort_index().ffill(limit=10).reindex(IDX)   # 對到 o7 日曆（只補短缺口）
    S[IDX < pd.Timestamp('2013-01-02')] = np.nan
    return S

def apply_fixes(S):
    S = S.copy()
    ix = S.index
    a = (ix >= pd.Timestamp(FIX4763['a0'])) & (ix < pd.Timestamp(FIX4763['a1']))
    b = (ix >= pd.Timestamp(FIX4763['a1'])) & (ix < pd.Timestamp(FIX4763['b1']))
    S.loc[a, '4763'] = S.loc[a, '4763'] / FIX4763['div_a']
    S.loc[b, '4763'] = S.loc[b, '4763'] / FIX4763['div_b']
    pre_val = float(S.loc[:pd.Timestamp(FIX2317['pre_val_date']), '2317'].dropna().iloc[-1])
    post_val = float(S.loc[pd.Timestamp(FIX2317['eff']):, '2317'].dropna().iloc[0])
    m = (ix > pd.Timestamp(FIX2317['pre_val_date'])) & (ix <= pd.Timestamp(FIX2317['pre_until']))
    S.loc[m, '2317'] = pre_val
    return S, dict(pre_val_2317=pre_val, post_val_2317=post_val)

def build_shares(kind):
    est = pd.read_parquet(O7 / 'data' / 'shares_est.parquet').reindex(index=IDX, columns=COLS)
    if kind == 'est': return est
    P = shares_pit_raw()
    if kind == 'pit_i': return P.where(est.notna())
    if kind == 'pit_ii': return apply_fixes(P)[0].where(est.notna())
    Pa = apply_fixes(P)[0]
    if kind == 'pit_a': return Pa
    if kind == 'pit_b': return Pa.shift(LAG)
    raise ValueError(kind)

def listed_mask():
    Lt = pd.read_parquet(AG / 'o_v7_listing_exec_engine' / 'data' / 'listed_twse.parquet')
    Lt = Lt.reindex(columns=COLS).astype('float')
    Lt = Lt.reindex(IDX.union(Lt.index)).sort_index().ffill().reindex(IDX)
    return Lt.fillna(1.0).to_numpy() > 0.5      # 無 V7 欄位者（無）視為上市

DELIST_INFO = {}
def patch_delist():
    """o_infra_v3_rebuild/delist_prices.parquet 補價（只補 o7 面板缺值格；同 o8 tlib 的作法）。"""
    dp = pd.read_parquet(AG / 'o_infra_v3_rebuild' / 'delist_prices.parquet')
    dp['date'] = pd.to_datetime(dp['date']); dp = dp[dp.code.isin(CI) & dp.date.isin(IDX)]
    filled = 0; ret_nan = 0
    for col, M in [('close', L.C), ('open', L.O), ('value', L.V), ('ret', L.RET)]:
        P = dp.pivot_table(index='date', columns='code', values=col, aggfunc='last')
        for c in P.columns:
            s = P[c].dropna(); m = M.loc[s.index, c].isna().to_numpy()
            if col == 'close': filled += int(m.sum())
            M.loc[s.index[m], c] = s[m].to_numpy()
        if col == 'ret':
            ret_nan = int(dp['ret'].isna().sum())
    DELIST_INFO.update(cells_filled=filled, codes=int(dp.code.nunique()), ret_nan_rows=ret_nan)
    L.Cn = L.C.to_numpy(); L.On = L.O.to_numpy(); L.Vn = L.V.to_numpy(); L.Rn = L.RET.to_numpy()
    L.TRADED = np.isfinite(L.Cn)
    L.GROW = np.where(L.TRADED, 1.0 + np.nan_to_num(L.Rn, nan=0.0), 1.0)
    L.LAST_TRADE = np.where(L.TRADED.any(axis=0), len(IDX) - 1 - np.argmax(L.TRADED[::-1], axis=0), -1)
    L.CFF = L.C.ffill().to_numpy(); L.PREVC = L.C.ffill().shift(1).to_numpy()
    L.NOBS = L.TRADED.sum(axis=1); L.GOOD = L.NOBS >= 300
    L._GX.clear(); L._RS.clear()

CFG = {}
T = None
def setup(name):
    """依設定修補 cmlib，之後才 import cmstrat（cmstrat 在 import 時計算滾動訊號）。每個行程只能呼叫一次。"""
    global T
    assert not CFG, 'setup 只能呼叫一次'
    cfg = CONFIGS[name]; CFG.update(cfg, name=name)
    if cfg['delist']: patch_delist()
    S = build_shares(cfg['shares']).to_numpy().copy()
    if cfg['delist']:
        S[~listed_mask()] = np.nan
    L._MC['S'] = S
    sys.path.insert(0, str(O7))
    import cmstrat as _T
    sys.path.pop(0)
    T = _T
    if cfg['memb_pre'] == 'pit':
        mp = json.loads((O8 / 'data' / 'membership_pit.json').read_text(encoding='utf-8'))
        T.MEMB_PRE = {pd.Timestamp(k): v for k, v in mp.items()}
    T._MEMC.clear(); T._CS.clear()
    # 開盤漲停快取（target 每次都呼叫）
    _lu = {}
    _orig = L.limit_up_open
    def lu(t):
        if t not in _lu: _lu[t] = _orig(t)
        return _lu[t]
    L.limit_up_open = lu
    global GX, CP
    GX = L.grow_exfix()
    CP = np.cumprod(GX, axis=0)
    return T

FROZEN = 'F1_K5_N20_sqrt'

# ---------------- 日期集合 ----------------
TX = pd.DatetimeIndex(sorted(set(pd.read_csv(O7 / 'raw' / 'bench' / 'taiex_twii.csv', parse_dates=['date'])['date'])))
SPECIAL_DL = {'2019-02': pd.Timestamp('2019-02-15'), '2023-04': pd.Timestamp('2023-04-12')}
def _first_td(ts):
    k = TX.searchsorted(ts); return TX[k] if k < len(TX) else None
def deadline_st(m):
    """s4_jscan7.py／st_core.deadline：錨定月 m 的營收法定期限（加權指數交易日）。"""
    s = str(m)
    if s in SPECIAL_DL: return _first_td(SPECIAL_DL[s])
    return _first_td(pd.Timestamp(m.year, m.month, 10))
def day_for(m, j):
    """s4_jscan7.py／st_core.day_for：D_m 之後第 j 個加權指數交易日；該日須為面板 GOOD 日且之後還有 GOOD 日。"""
    D = deadline_st(m)
    if D is None: return None
    k = TX.get_loc(D) + j
    if k < 0 or k >= len(TX): return None
    s = TX[k]
    if s not in IDX: return None
    t = IDX.get_loc(s)
    if not L.GOOD[t]: return None
    if not L.GOOD[t + 1:].any(): return None
    return s

def jdays(j, m0, m1, mend):
    """訊號日：錨定月 m0..m1（含）；終點＝錨定月 mend 的 J_j（不存在或超出面板 → 面板最後一日）。另要求 d+1 為 GOOD（同 s4_jscan7）。"""
    out = []
    for m in pd.period_range(m0, m1, freq='M'):
        d = day_for(m, j)
        if d is None: continue
        t = IDX.get_loc(d)
        if not L.GOOD[t + 1]: continue
        out.append(d)
    de = day_for(pd.Period(mend, 'M'), j)
    if de is None or de > IDX[-1]: de = IDX[-1]
    out = [d for d in out if d < de]
    return out, de

def official(m0, m1, mend):
    """o7 官方日：每月第一個日 ≥ 11 的面板交易日。終點＝mend 月的官方日。"""
    ds = L.official_days(str(pd.Period(m0, 'M').start_time.date()), str(pd.Period(m1, 'M').end_time.date()))
    de = L.official_days(str(pd.Period(mend, 'M').start_time.date()), str(pd.Period(mend, 'M').end_time.date()))
    de = de[0] if de else IDX[-1]
    return [d for d in ds if d < de], de

PERIODS = {
    'MAIN': ('2019-05', '2026-09', '2026-10'),   # P1：2019-05～2026-10
    'PRE':  ('2014-01', '2019-04', '2019-05'),   # P5：2014-01～2019-04
}
def daysets(period):
    m0, m1, me = PERIODS[period]
    out = {'OFF': official(m0, m1, me)}
    for j in range(13): out[f'J{j}'] = jdays(j, m0, m1, me)
    return out

# ---------------- 快速引擎 ----------------
def solve_T(w, h, Vt, buy, sell):
    """解 T + buy·Σ(wT−h)+ + sell·Σ(h−wT)+ ＝ Vt（cmlib.simulate_w 的二分法目標；分段線性、遞增 → 精確解）。"""
    m = w > 0
    c0 = sell * h[~m].sum()
    ww = w[m]; hh = h[m]
    b = hh / ww; o = np.argsort(b, kind='stable'); b = b[o]; ww = ww[o]; hh = hh[o]
    cw = np.concatenate([[0.0], np.cumsum(ww)]); ch = np.concatenate([[0.0], np.cumsum(hh)])
    Wt, Ht = cw[-1], ch[-1]
    slope = 1 + buy * cw - sell * (Wt - cw); icpt = c0 - buy * ch + sell * (Ht - ch)
    fb = b * slope[:-1] + icpt[:-1]
    k = int(np.searchsorted(fb, Vt, side='left'))
    Tsol = (Vt - icpt[k]) / slope[k]
    return min(max(Tsol, 0.0), Vt)

def sim_fast(plan, buy, sell, t0, t1, qidx):
    """與 cmlib.simulate_w(plan, buy, sell, t0, t1, G=grow_exfix) 相同；只回傳 qidx（遞增的面板索引）收盤淨值。"""
    Cn, On, TR, LT = L.Cn, L.On, L.TRADED, L.LAST_TRADE
    rb = sorted(t for t in plan if t0 < t <= t1)
    ev = sorted(set(rb) | set(qidx))
    val = np.zeros(N); cash = 1.0; tc = t0; out = {}
    rbs = set(rb)
    for t in ev:
        if t in rbs:
            if t - 1 > tc: val = val * (CP[t - 1] / CP[tc])
            pk, w = plan[t]
            tr = TR[t] & np.isfinite(On[t])
            gap = np.where(tr, GX[t] * On[t] / np.where(tr, Cn[t], 1.0), 1.0)
            h = val * gap
            frz = (h > 0) & ~tr
            dead = frz & (LT < t)
            if dead.any():
                cash += h[dead].sum() * (1 - sell); h = np.where(dead, 0.0, h); frz &= ~dead
            htr = np.where(frz, 0.0, h)
            Vt = htr.sum() + cash
            wv = np.zeros(N); np.add.at(wv, np.asarray(pk, int), np.asarray(w, float))
            wv[frz] = 0.0
            s = wv.sum()
            if s <= 0: wv[:] = 0.0; lo = 0.0
            else:
                wv = wv / s; lo = solve_T(wv, htr, Vt, buy, sell)
            tgt = wv * lo
            newval = np.where(frz, h, 0.0) + tgt
            oc = np.where(tr, Cn[t] / np.where(tr, On[t], 1.0), 1.0)
            val = np.where(frz, newval, newval * oc)
            cash = 0.0; tc = t
            if t in qidx: out[t] = val.sum() + cash
        else:
            if t > tc: val = val * (CP[t] / CP[tc]); tc = t
            out[t] = val.sum() + cash
    return out

def win_returns(navd, sdays, de):
    """訊號窗報酬：訊號日 d_k 收盤 → d_{k+1}（最後一窗到 de）收盤。"""
    ti = [IDX.get_loc(d) for d in sdays] + [IDX.get_loc(de)]
    v = np.array([navd[t] for t in ti])
    return v[1:] / v[:-1] - 1

E50 = L.E50
def bench_win(sdays, de):
    ti = [IDX.get_loc(d) for d in sdays] + [IDX.get_loc(de)]
    v = E50.to_numpy()[ti]
    return v[1:] / v[:-1] - 1

def tstat(x):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    if len(x) < 3: return float('nan')
    s = x.std(ddof=1)
    return float(x.mean() / (s / math.sqrt(len(x)))) if s > 0 else float('nan')

# ---------------- 一次執行（實際或隨機分數）----------------
def run_scores(sdays, de, score_fn, cost='normal', exclude=None, want_plan=False):
    """score_fn(cs) → 分數向量（None＝實際分數）。exclude：從合格集合剔除的股票索引（P3 用；實際與隨機都剔除）。"""
    plan = {}; held = set(); plist = []
    for d in sdays:
        cs = T.cross_section(d)
        c2 = dict(cs)
        if exclude is not None:
            el = cs['elig'].copy(); el[exclude] = False; c2['elig'] = el
            if score_fn is None:
                def pr(x): return pd.Series(np.where(el, x, np.nan)).rank(pct=True).to_numpy()
                sc = 0.5 * pr(cs['s52']) + 0.5 * pr(cs['y3']); sc[~el] = np.nan; c2['sc'] = sc
        if score_fn is not None: c2['sc'] = score_fn(c2)
        js, ws, _, _ = T.target(c2, FROZEN, held)
        plan[cs['e']] = (js, ws); held = set(js.tolist())
        if want_plan: plist.append((d, js, ws))
    t0 = IDX.get_loc(sdays[0]); t1 = IDX.get_loc(de)
    qidx = set(IDX.get_loc(d) for d in sdays) | {t1}
    b, s = L.rates(cost)
    navd = sim_fast(plan, b, s, t0, t1, qidx)
    r = win_returns(navd, sdays, de)
    return (r, plist) if want_plan else r

def rand_fn(rg):
    return lambda cs: np.where(cs['elig'], rg.random(N), np.nan)

def dumps(o): return json.dumps(o, ensure_ascii=False, indent=1, default=lambda z: float(z) if isinstance(z, (np.floating, np.integer)) else (z.tolist() if isinstance(z, np.ndarray) else str(z)))
