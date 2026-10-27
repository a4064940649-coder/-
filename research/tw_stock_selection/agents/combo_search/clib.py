#!/usr/bin/env python3
"""combo_search 共用：9 個元件、通用 M2 選股（任意分數）、可持現金／月中賣出的日頻引擎、月窗統計、bootstrap。
資料、成本、股池、M2 成交規則全部沿用 revenue_longhistory/lhlib.py（直接 import，不改原檔）。"""
import itertools, math, sys
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'revenue_longhistory'))
sys.path.insert(0, str(HERE.parent / 'industry'))
sys.dont_write_bytecode = True
import lhlib as L  # noqa: E402
from ilib import PREFIX_IND  # noqa: E402  (只用代碼→產業對照表；ilib.load 不呼叫)
from btlib import tstat  # noqa: E402

IDX, COLS, N = L.IDX, L.COLS, L.N
COMP = ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9']
COMBOS = [c for k in (1, 2, 3) for c in itertools.combinations(COMP, k)]
def cname(c): return '+'.join(c)
MAIN = ('C1', 'C2')

# ---------- 有效交易日上的價格衍生量 ----------
_ag = L.adj.loc[L.GOOD]
_agf = _ag.ffill(limit=5)
_rg = _ag / _ag.shift(1) - 1
VOL252 = _rg.rolling(252, min_periods=200).std().reindex(IDX)
MAX21 = _rg.rolling(21, min_periods=15).max().reindex(IDX)
R21 = (_agf / _agf.shift(21) - 1).reindex(IDX)
_mx = _agf.rolling(252, min_periods=240).max()
NH21 = (_agf.rolling(21, min_periods=15).max() >= _mx - 1e-12).where(_mx.notna()).reindex(IDX)

def industry_map():
    sl = pd.read_csv('/home/user/stevenhsiehsheng-ctrl/tw-stock-screener/data/stock_list.csv', dtype={'code': str})
    ind = sl.drop_duplicates('code').set_index('code')['industry']
    m = {}
    for c in COLS:
        if c in ind.index and isinstance(ind[c], str): m[c] = ind[c]
        elif c[:2] in PREFIX_IND: m[c] = PREFIX_IND[c[:2]]
    return pd.Series(m)
IND = industry_map()

def rev_level():
    d = pd.read_csv(L.LONG, dtype={'code': str, 'market': str, 'name': str})
    d['p'] = pd.PeriodIndex(pd.to_datetime(d['month']), freq='M')
    d = d.sort_values(['p', 'market']).drop_duplicates(['p', 'code'], keep='first')
    d = d[d['code'].isin(COLS)]
    Lv = d.pivot_table(index='p', columns='code', values='revenue_k_twd', aggfunc='last').sort_index()
    return Lv.reindex(pd.period_range(Lv.index.min(), Lv.index.max(), freq='M'))

def rev_level_5y():
    rev = pd.read_csv(L.X / 'rev_5y.csv.gz', dtype={'code': str}); rev = rev[rev['code'].isin(COLS)]; rev['p'] = pd.PeriodIndex(rev['ym'], freq='M')
    col = [c for c in rev.columns if c in ('rev', 'revenue', 'value')]
    Lv = rev.pivot_table(index='p', columns='code', values=col[0], aggfunc='last').sort_index() if col else None
    return Lv.reindex(pd.period_range(Lv.index.min(), Lv.index.max(), freq='M')) if Lv is not None else None


def components(S, LEV):
    """S: lhlib.Setup；LEV: 月營收水準（index=Period）。回傳 {Ck: DataFrame(DATES×COLS)}（方向已調成越大越好）。"""
    D = S.DATES; mm = [pd.Period(d, 'M') - 1 for d in S.dates]
    def bym(M): return pd.DataFrame(M.reindex(mm).to_numpy(), index=D, columns=M.columns).reindex(columns=COLS)
    YOY3 = S.YOY3
    out = {'C1': S.s52, 'C2': S.yoy3, 'C3': bym(YOY3 - YOY3.shift(3))}
    LEV = LEV.reindex(columns=COLS)
    mx = LEV.rolling(12, min_periods=12).max()
    out['C4'] = bym((LEV / mx).where(mx > 0))
    out['C5'] = -VOL252.reindex(D); out['C6'] = -MAX21.reindex(D); out['C7'] = -R21.reindex(D)
    v60 = L.v60_fix.reindex(D); px = L.C.reindex(D); base = (v60 >= 1e7) & (px >= 10)
    nh = NH21.reindex(D)
    cols = [c for c in COLS if c in IND.index]
    nhb = nh[cols].where(base[cols])
    g = nhb.T.groupby(IND[cols]); mean = g.mean().T; cnt = g.count().T
    mean = mean.where(cnt >= 3)
    c8 = pd.DataFrame(np.nan, index=D, columns=COLS)
    for c in cols: c8[c] = mean[IND[c]].to_numpy()
    out['C8'] = c8
    out['C9'] = v60
    return out


def pct_ranks(comps, el):
    return {k: v.where(el).rank(axis=1, pct=True).where(el).fillna(0.5).where(el) for k, v in comps.items()}


def score(prk, combo):
    w = 1.0 / len(combo)
    out = w * prk[combo[0]]
    for k in combo[1:]: out = out + w * prk[k]
    return out

# ---------- 擇時開關 ----------
_tx = pd.read_csv('/home/user/jonatw/finlab-txd/data/raw/taiex_twii.csv', parse_dates=['date']).set_index('date')['close'].sort_index()
_ma = _tx.rolling(200, min_periods=200).mean()
TX = _tx.reindex(IDX, method='ffill'); MA200 = _ma.reindex(IDX, method='ffill')
def switch_on(d):
    m = MA200.loc[d]
    return True if not np.isfinite(m) else bool(TX.loc[d] >= m)


def select(S, sc, el, n, switch=False, gap='era'):
    """M2 通用版（邏輯同 lhlib.Setup.selection）。sc: DataFrame 分數；el: 股池。switch=True 時大盤 < 200MA 該期持現金。"""
    GU = L.GAP_UP_OPEN_ERA if gap == 'era' else L.GAP_UP_OPEN_10
    SC = sc.to_numpy(); EL = el.to_numpy()
    events, pools = {}, {}; held = set()
    for k, d in enumerate(S.dates[:-1]):
        i = L.pos(d); ex = i + 1
        trad = np.isfinite(L.OPEN_RAW[ex]) & np.isfinite(L.ADJ_RAW[ex])
        ok = EL[k] & trad & np.isfinite(SC[k])
        if ok.sum() < 50: continue
        pools[ex] = ('open', np.where(ok)[0])
        if switch and not switch_on(d):
            events[ex] = ('open', np.array([], dtype=int)); held = set(); continue
        order = SC[k].copy(); order[~ok] = np.nan
        cand = np.argsort(-np.nan_to_num(order, nan=-9))[:int(ok.sum())]
        pick = []
        for j in cand:
            if GU[ex, j] and j not in held: continue
            pick.append(j)
            if len(pick) == n: break
        events[ex] = ('open', np.array(pick, dtype=int)); held = set(pick)
    return events, pools


def sim(events, buy, sell, t0, t1, sells=None):
    """lhlib.sim_portfolio 的延伸：空清單＝全部持現金；sells={t: [j,...]} 在 t 日開盤賣出該股、持現金到下次換股。"""
    AO, AC = L.AO, L.AC
    u = np.zeros(N); cash = 1.0; nav = np.empty(t1 - t0 + 1)
    for t in range(t0, t1 + 1):
        if t in events:
            mode, pk = events[t]
            Pz = AO[t]
            h = np.where(u > 0, u * np.nan_to_num(Pz), 0.0)
            Vt = h.sum() + cash
            tgt = np.zeros(N)
            if len(pk) == 0:
                cash = Vt - sell * h.sum(); u = np.zeros(N)
            else:
                w = 1.0 / len(pk); Vn = Vt
                for _ in range(6):
                    tgt[:] = 0; tgt[pk] = w * Vn
                    cost = buy * np.clip(tgt - h, 0, None).sum() + sell * np.clip(h - tgt, 0, None).sum()
                    Vn = Vt - cost
                u = np.zeros(N); u[pk] = tgt[pk] / AO[t][pk]; cash = 0.0
        if sells and t in sells:
            for j in sells[t]:
                if u[j] > 0:
                    cash += u[j] * AO[t][j] * (1 - sell); u[j] = 0.0
        nav[t - t0] = np.nansum(u * AC[t]) + cash
    return pd.Series(nav, index=IDX[t0:t1 + 1])


def monthly(nav, DATES):
    nd = nav.reindex(DATES)
    return (nd.shift(-1) / nd - 1).iloc[:-1]


def win(s, a, b, drop=None):
    m = (s.index >= pd.Timestamp(a)) & (s.index <= pd.Timestamp(b))
    if drop is not None:
        m &= ~((s.index >= pd.Timestamp(drop[0])) & (s.index <= pd.Timestamp(drop[1])))
    return s[m]

# ---------- bootstrap ----------
def stationary_idx(T, B, mean_block, rng):
    p = 1.0 / mean_block
    idx = np.empty((B, T), dtype=int)
    idx[:, 0] = rng.integers(0, T, B)
    for t in range(1, T):
        new = rng.random(B) < p
        idx[:, t] = np.where(new, rng.integers(0, T, B), (idx[:, t - 1] + 1) % T)
    return idx


def tvec(X):
    m = X.mean(axis=-2); s = X.std(axis=-2, ddof=1)
    return m / (s / math.sqrt(X.shape[-2]))


def maxt_bootstrap(X, B=5000, mean_block=6, seed=0, spa=False):
    """X: T×K 月超額。H0: 全部平均 ≤ 0。回傳觀察 t、max-t 校正 p（單步，White RC 型）；spa=True 時 Hansen SPA_c 的重新置中。"""
    X = np.asarray(X, float); T, K = X.shape
    t_obs = tvec(X); mu = X.mean(axis=0); sd = X.std(axis=0, ddof=1)
    if spa:
        thr = -math.sqrt(2 * math.log(math.log(T)))
        cen = np.where(t_obs > thr, mu, 0.0)   # SPA_c：表現很差的不完全置中
        Xc = X - cen
        # SPA 的檢定統計量 max(0, max t)
    else:
        Xc = X - mu
    rng = np.random.default_rng(seed)
    idx = stationary_idx(T, B, mean_block, rng)
    mx = np.empty(B)
    for b0 in range(0, B, 500):
        Xb = Xc[idx[b0:b0 + 500]]          # (b,T,K)
        mb = Xb.mean(axis=1)
        tb = mb / (sd / math.sqrt(T))       # 用原樣本標準差（Hansen 2005 的作法）
        mx[b0:b0 + 500] = tb.max(axis=1)
    if spa: mx = np.maximum(mx, 0)
    t_max = float(np.max(t_obs))
    p_each = [(float((mx >= max(t, 0 if spa else t)).mean())) for t in t_obs]
    return {'t_obs_max': round(t_max, 3), 'argmax': int(np.argmax(t_obs)), 'p_maxt': round(float((mx >= t_max).mean()), 4),
            'p_each': [round(p, 4) for p in p_each], 'boot_max_t_q95': round(float(np.quantile(mx, 0.95)), 3), 'T': T, 'K': K}


def deflated_sharpe(x_best, sr_all, N_trials):
    from statistics import NormalDist
    nd = NormalDist(); x = np.asarray(x_best, float); T = len(x)
    sr = x.mean() / x.std(ddof=1)
    sk = float(((x - x.mean()) ** 3).mean() / x.std(ddof=0) ** 3); ku = float(((x - x.mean()) ** 4).mean() / x.std(ddof=0) ** 4)
    v = float(np.var(sr_all, ddof=1)); g = 0.5772156649
    sr0 = math.sqrt(v) * ((1 - g) * nd.inv_cdf(1 - 1 / N_trials) + g * nd.inv_cdf(1 - 1 / (N_trials * math.e)))
    z = (sr - sr0) * math.sqrt(T - 1) / math.sqrt(1 - sk * sr + (ku - 1) / 4 * sr ** 2)
    psr0 = (sr) * math.sqrt(T - 1) / math.sqrt(1 - sk * sr + (ku - 1) / 4 * sr ** 2)
    return {'SR_monthly': round(float(sr), 4), 'SR0_expected_max': round(sr0, 4), 'var_SR_trials': round(v, 6), 'N_trials': round(float(N_trials), 1),
            'skew': round(sk, 3), 'kurt': round(ku, 3), 'T': T, 'DSR': round(nd.cdf(z), 4), 'PSR_vs_0': round(nd.cdf(psr0), 4)}
