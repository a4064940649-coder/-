#!/usr/bin/env python3
"""o_tpex_pool_extension 共用：選股名單產生器（上市／上櫃／合併池）、V7X 模擬包裝、統計。
- 換股日：沿用 V7 plans_j7.csv（AB_V7_B10、seed 0）每個 j 的訊號日與成交日（同一日曆，上市版與合併版完全同日）。
- 分數：0.5×52 週高點比百分位＋0.5×近 3 月營收年增率平均百分位（池內）；同分先比營收年增率，再比代碼。
- 營收：tw_revenue.csv（上市＋上櫃，2005 起）；訊號日（≥ 每月 10 日法定期限）用 M−1、M−2、M−3 月。
  B10：2013 年起缺營收、且之後下市／下櫃的股票，營收百分位固定 1.0 放回（同 V7 B10，保守處理營收檔的存活者偏誤）。
- 股池：訊號日 60 日均成交值 ≥ floor、收盤 ≥ 10，依 V60 排名取前 K（池內市場別依 PIT：訊號日當時是上櫃就算上櫃）；
  再排除金融保險（含 28xx、5876、5880）、KY／DR／91xx、處置中（訊號日或成交日）、成交日無開盤、無 52 週值、缺營收。
- 開盤漲停（成交日開盤 ≥ 漲停價）且未持有 → 跳過遞補（M2 規則）；上櫃檔數上限 cap（None＝不設）。
- 模擬：o_exec_engine_rewrite_v7x/x_exec.py（只 import），本資料夾 x_data.py 為合併面板替身。正式設定＝F1＋F2＋F3(0.5)＋F4、K5T。
"""
import sys, json, math, os
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
AG = HERE.parent
V7X = AG / 'o_exec_engine_rewrite_v7x'
sys.path.insert(0, str(HERE))
import x_data as D            # noqa: E402（合併面板替身，必須先於 x_exec 載入）
sys.path.insert(1, str(V7X))
import x_exec as X            # noqa: E402（只讀 import）
assert X.D is D
sys.path.remove(str(V7X))

O7D = AG / 'o_v7_listing_exec_engine' / 'data'
EXT = Path('/home/user/stevenhsiehsheng-ctrl/tw-stock-screener/data')
REV = Path('/home/user/niklavsd/bottleneck-signals/data/processed/tw_revenue.csv')
FORMAL = dict(f1=True, f2=True, f3=0.5, f4=True, stress_odd_extra=True, rank_right=True)
TIERS = {'30萬': 3e5, '100萬': 1e6, '300萬': 3e6, '1000萬': 1e7}
SEL = ('2006-04-01', '2019-04-30'); VAL = ('2019-05-01', '2023-12-31'); HOLD = ('2024-01-01', '2026-10-31')
PERIODS = {'ALL': ('2006-04-01', '2026-10-31'), 'SEL': SEL, 'OOS': ('2019-05-01', '2026-10-31'), 'VAL': VAL, 'HOLD': HOLD,
           'Y19': ('2019-01-01', '2026-10-31'), 'P1': ('2006-01-01', '2012-12-31'), 'P2': ('2013-01-01', '2018-12-31'), 'P3': ('2019-01-01', '2026-10-31'),
           'S1a': ('2008-01-01', '2012-12-31')}
E50 = pd.Series(D.E50, index=D.IDX)

# ---------------- 換股日 ----------------
_P = pd.read_csv(O7D / 'plans_j7.csv', usecols=['strat', 'j', 'seed', 'signal', 'exec_idx'])
_P = _P[(_P.strat == 'AB_V7_B10') & (_P.seed == 0)]
SIGS = {int(j): [(D.TPOS[pd.Timestamp(s)], int(e)) for s, e in zip(g.signal, g.exec_idx)] for j, g in _P.groupby('j')}
ENDS = {int(k.split('|J')[1]): pd.Timestamp(v) for k, v in json.loads((O7D / 'jscan_end_days.json').read_text()).items() if k.startswith('AB_V7_B10|J')}

# ---------------- 排除名單 ----------------
_sl = pd.read_csv(EXT / 'stock_list.csv', dtype=str)
_rv = pd.read_csv(REV, dtype={'code': str})
_names = pd.concat([_sl[['code', 'name']], _rv[['code', 'name']]]).dropna().drop_duplicates('code', keep='first').set_index('code')['name'].to_dict()
FIN = set(_sl.loc[_sl['industry'] == '金融保險業', 'code']) | {c for c in D.COLS if c[:2] == '28' or c in ('5876', '5880')}
KYDR = {c for c in D.COLS if c[:2] == '91' or ('KY' in str(_names.get(c, '')).upper()) or ('DR' in str(_names.get(c, '')).upper())}
EXCL = np.array([(c in FIN) or (c in KYDR) or (not c[:1].isdigit()) or len(c) != 4 for c in D.COLS])
NOW = set(_sl.code)
LASTT = np.where(D.TR.any(0), len(D.IDX) - 1 - np.argmax(D.TR[::-1], 0), -1)
DEAD = np.array([(D.IDX[LASTT[j]] < D.IDX[-1] - pd.Timedelta(days=30)) and (c not in NOW) for j, c in enumerate(D.COLS)])

# ---------------- 營收 ----------------
def _yoy():
    d = _rv.copy()
    d['p'] = pd.PeriodIndex(pd.to_datetime(d['month']), freq='M')
    d = d.sort_values(['p', 'market']).drop_duplicates(['p', 'code'], keep='first')
    d = d[d['code'].isin(set(D.COLS))]
    Y = d.pivot_table(index='p', columns='code', values='revenue_k_twd', aggfunc='last').sort_index()
    Y = Y.reindex(pd.period_range(Y.index.min(), Y.index.max(), freq='M'))
    prev = Y.shift(12)
    yo = ((Y / prev - 1) * 100).where(prev > 0)
    return yo.rolling(3, min_periods=3).mean().reindex(columns=D.COLS)
YOY3 = _yoy()

def yoy3_at(sig_t):
    d = D.IDX[sig_t]; m = pd.Period(d, 'M') - (1 if d.day >= 10 else 2)
    if m not in YOY3.index: return np.full(D.N, np.nan)
    return YOY3.loc[m].to_numpy(float)


# ---------------- 名單 ----------------
def gen_plan(v, j, return_diag=False):
    """v：dict(pool='TW'|'TP'|'ALL', K, n, cap=None|float, floor=1e7)。回傳 [(exec_idx, picks, sig_idx)]。"""
    pool, K, n = v['pool'], v['K'], v['n']; cap = v.get('cap'); floor = v.get('floor', 1e7)
    out = []; held = set(); diag = []
    for sig, ex in SIGS[j]:
        if D.IDX[sig] > ENDS[j]: break
        mk = D.MK[sig]
        base = np.isfinite(D.V60[sig]) & (D.V60[sig] >= floor) & D.TR[sig] & (D.C[sig] >= 10)
        if pool == 'TW': base &= ~mk
        elif pool == 'TP': base &= mk
        w = np.where(base)[0]
        if not len(w): continue
        w = w[np.argsort(-D.V60[sig, w], kind='stable')][:K]
        y3 = yoy3_at(sig)
        s52 = D.S52[sig]
        b10 = (~np.isfinite(y3)) & DEAD & (D.IDX[sig] >= pd.Timestamp('2013-01-01'))
        ok = (~EXCL[w]) & (~D.DISP[sig, w]) & (~D.DISP[ex, w]) & np.isfinite(D.O[ex, w]) & D.TR[ex, w] & np.isfinite(s52[w]) & (np.isfinite(y3[w]) | b10[w])
        p = w[ok]
        if len(p) < max(2 * n, 20): continue
        r52 = pd.Series(s52[p]).rank(pct=True).to_numpy()
        yy = y3[p]; fin = np.isfinite(yy)
        rr = np.ones(len(p)); rr[fin] = pd.Series(yy[fin]).rank(pct=True).to_numpy() * fin.sum() / len(p) + 0  # 有值者依序；B10 固定 1.0
        # 百分位口徑：有值者在「全池」中的名次百分位；B10 股固定 1.0
        if (~fin).any():
            rk = pd.Series(np.where(fin, yy, np.inf)).rank(pct=True, method='average').to_numpy()
            rr = np.where(fin, rk, 1.0)
        sc = np.round(0.5 * r52 + 0.5 * rr, 10)
        ysort = np.where(fin, yy, np.inf)
        order = sorted(range(len(p)), key=lambda i: (-sc[i], -ysort[i], D.COLS[p[i]]))
        picks = []; ntp = 0; maxtp = None if cap is None else int(math.floor(cap * n + 1e-9))
        for i in order:
            jj = p[i]
            if X.LIMUP[ex, jj] == X.LIMUP[ex, jj] and D.O[ex, jj] >= X.LIMUP[ex, jj] - 1e-6 and jj not in held:
                continue
            if maxtp is not None and mk[jj] and ntp >= maxtp: continue
            picks.append(jj); ntp += int(mk[jj])
            if len(picks) == n: break
        out.append((ex, np.array(picks, int), sig))
        held = set(picks)
        if return_diag:
            diag.append(dict(sig=str(D.IDX[sig].date()), n_pool=int(len(p)), n_pool_tp=int(mk[p].sum()), n_tp=ntp,
                             v60_med=float(np.median(D.V60[sig, picks])), picks=[D.COLS[x] for x in picks]))
    return (out, diag) if return_diag else out


# ---------------- 模擬 ----------------
def cfg(tier, mode, cost='normal', model='K5T', **kw):
    c = dict(tier=TIERS.get(tier, tier), mode=mode, model=model, cost=cost); c.update(FORMAL); c.update(kw)
    return c


def sim(pl, c, t1=None):
    s = X.Sim(pl, c); nav = s.run(t1=t1)
    pa = np.array(s.part) if s.part else np.array([np.nan])
    acc = dict(s.acc); acc['part_median'] = float(np.nanmedian(pa)); acc['part_p90'] = float(np.nanpercentile(pa, 90)) if np.isfinite(pa).any() else float('nan')
    return nav, acc


def ideal_windows(pl, j):
    """零衝擊、零成本：成交日開盤等權買進，持有到下一個成交日開盤（含息近似）。回傳以訊號日為 index 的窗報酬。"""
    AO = D.AI * D.O / np.where(np.isfinite(D.C) & (D.C > 0), D.C, np.nan)    # 含息指數換算的開盤
    AOf = pd.DataFrame(AO).ffill().to_numpy(); AIf = pd.DataFrame(np.where(D.TR, D.AI, np.nan)).ffill().to_numpy()
    out = {}
    for k in range(len(pl) - 1):
        e1, pk, sg = pl[k]; e2 = pl[k + 1][0]
        a = AO[e1, pk]; b = np.where(np.isfinite(AO[e2, pk]), AO[e2, pk], AIf[e2, pk])
        r = b / a - 1; r = r[np.isfinite(r)]
        out[D.IDX[sg]] = float(r.mean()) if len(r) else np.nan
    return pd.Series(out)


# ---------------- 統計 ----------------
def tstat(x):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    if len(x) < 3 or x.std(ddof=1) == 0: return float('nan')
    return float(x.mean() / (x.std(ddof=1) / math.sqrt(len(x))))


def nw_t(x, L=None):
    x = np.asarray(x, float); x = x[np.isfinite(x)]; n = len(x)
    if n < 6: return float('nan')
    L = L or int(math.floor(4 * (n / 100) ** (2 / 9)))
    e = x - x.mean(); s = (e * e).sum() / n
    for i in range(1, L + 1):
        s += 2 * (1 - i / (L + 1)) * (e[i:] * e[:-i]).sum() / n
    return float(x.mean() / math.sqrt(s / n))


def windows(nav, j, ref=None):
    """訊號日窗（訊號日收盤→下一訊號日收盤）報酬；ref＝對照淨值（預設 0050）。回傳 DataFrame[s, b, x]，index＝窗起點訊號日。"""
    e = ENDS[j]
    sd = pd.DatetimeIndex([D.IDX[s] for s, _ in SIGS[j] if D.IDX[s] < e]).append(pd.DatetimeIndex([e]))
    nav = nav.dropna()
    sd = sd[(sd >= nav.index[0]) & (sd <= nav.index[-1])]
    r = nav.reindex(sd); r = r.shift(-1) / r - 1
    b = (ref if ref is not None else E50).reindex(sd); b = b.shift(-1) / b - 1
    df = pd.DataFrame({'s': r, 'b': b}).iloc[:-1]
    df['x'] = df.s - df.b
    return df


HOLE_MONTHS = None
def hole_months():
    global HOLE_MONTHS
    if HOLE_MONTHS is None:
        gm = json.loads((HERE / 'data' / 'gap_map.json').read_text())
        h = {k for k, v in gm.items() if v['hole']}
        # 洞後恢復期：上櫃合格股數（含 52 週值）< 洞前 12 個月中位數 85% 的月份
        tp = D.MK; el = np.isfinite(D.V60) & np.isfinite(D.S52)        # 資料可得性（不含流動性門檻，避免空頭月被誤判）
        s = pd.Series((el & tp).sum(1), index=D.IDX).resample('ME').last()
        s.index = s.index.to_period('M')
        for m in s.index:
            if m < pd.Period('2008-06', 'M'): continue
            ref = s[(s.index < m) & (s.index >= m - 12)].median()
            if s[m] < 0.85 * ref: h.add(str(m))
        HOLE_MONTHS = sorted(h)
    return HOLE_MONTHS


def cagr(nav, a, b):
    nav = nav.dropna(); s = nav.loc[a:b]
    if len(s) < 20: return float('nan')
    yrs = (s.index[-1] - s.index[0]).days / 365.25
    return float((s.iloc[-1] / s.iloc[0]) ** (1 / yrs) - 1) * 100


def mdd(nav, a=None, b=None):
    s = nav.dropna().loc[a:b]
    return float((s / s.cummax() - 1).min() * 100)


def seg_stats(W, a, b, drop_holes=True):
    x = W.loc[a:b]
    if drop_holes:
        hm = set(hole_months()); x = x[[str(pd.Period(d, 'M')) not in hm for d in x.index]]
    xs = x.x
    if len(xs) < 3: return dict(n=int(len(xs)))
    ann_s = float(np.prod(1 + x.s) ** (12 / len(x)) - 1) * 100; ann_b = float(np.prod(1 + x.b) ** (12 / len(x)) - 1) * 100
    yrs = pd.Series(xs.values, index=x.index.year)
    tY = yrs.groupby(level=0).sum()
    drop3 = tY.sort_values().index[-3:] if len(tY) > 5 else []
    xd = xs[~x.index.year.isin(drop3)]
    return dict(n=int(len(xs)), ann=ann_s, ann_b=ann_b, ann_x=ann_s - ann_b, mean_pp=float(xs.mean() * 100), t=tstat(xs), t_nw=nw_t(xs),
                t_drop_best3y=tstat(xd), best3y=[int(y) for y in drop3])


def jsumm(vals):
    v = np.array([x for x in vals if x is not None and np.isfinite(x)], float)
    if not len(v): return dict(median=None, p10=None, min=None, max=None)
    return dict(median=float(np.median(v)), p10=float(np.percentile(v, 10)), min=float(v.min()), max=float(v.max()))
