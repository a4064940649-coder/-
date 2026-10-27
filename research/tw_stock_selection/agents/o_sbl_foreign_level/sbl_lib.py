#!/usr/bin/env python3
"""o_sbl_foreign_level 共用：資料載入、外資持股比率回推、借券／融券面板、主策略引擎包裝。

只讀 import：main_v2/v2_core.py（→ independent_audit/ia_core.py）。不修改其他資料夾。
raw/ 下的檔案來自 GitHub 鏡像，視為不可信：只用 pandas 讀取。
"""
import sys, json, math, hashlib
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
RAW = HERE / 'raw'
AG = HERE.parent
sys.path.insert(0, str(AG / 'main_v2'))
import v2_core as V          # noqa: E402
sys.path.pop(0)
K = V.K
IDX, COLS, N = K.IDX, K.COLS, K.N
CI = {c: j for j, c in enumerate(COLS)}
X = K.X                       # tw-stock-screener/data/extras

SEL = (pd.Timestamp('2021-10-01'), pd.Timestamp('2023-12-31'))     # 挑選期（訊號日）
VAL = (pd.Timestamp('2024-01-01'), pd.Timestamp('2026-10-12'))     # 驗證期（訊號日）
VAL_A = (pd.Timestamp('2024-01-01'), pd.Timestamp('2025-09-30'))   # 驗證期前段：回推持股
HOLD = (pd.Timestamp('2025-10-01'), pd.Timestamp('2026-10-12'))    # 保留段：官方 MI_QFIIS 實值
ANCHOR = pd.Timestamp('2025-10-07')                                # 第一份 MI_QFIIS 快照


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# ---------------- 原始資料 ----------------
def load_inst5y():
    d = pd.read_csv(X / 'inst_5y.csv.gz', dtype={'code': str}, usecols=['date', 'code', 'foreign'])
    d['date'] = pd.to_datetime(d['date'])
    d['foreign'] = d['foreign'] * 1000.0      # 張 → 股
    return d


def load_flows():
    """外資（不含自營）日淨買超（股）。2021-10-08～2026-10-07 用 inst_5y；之後用 BlackBoxOP（已證實逐筆一致）。"""
    a = load_inst5y()
    b = pd.read_parquet(RAW / 'bb_inst.parquet')
    b = b.rename(columns={'Date': 'date', 'Ticker': 'code', 'Foreign_Net': 'foreign'})[['date', 'code', 'foreign']]
    b['date'] = pd.to_datetime(b['date'])
    b = b[b.date > a.date.max()]
    f = pd.concat([a, b], ignore_index=True)
    W = f.pivot_table(index='date', columns='code', values='foreign', aggfunc='last')
    return W


def load_qfiis():
    d = pd.read_parquet(RAW / 'vf_twse_foreign.parquet')
    d['date'] = pd.to_datetime(d['date'])
    d = d[d.total_shares > 0]
    R = d.pivot_table(index='date', columns='code', values='foreign_ratio', aggfunc='last') / 100.0
    S = d.pivot_table(index='date', columns='code', values='total_shares', aggfunc='last')
    Fs = d.pivot_table(index='date', columns='code', values='foreign_shares', aggfunc='last')
    return R, S, Fs


def backcast(R, S, W, anchor, start):
    """比率空間回推：r(t−1) = r(t) − net(t)/S。S 用錨點日股數（2025-10 之前沒有歷史股數）。
    R：MI_QFIIS 比率（錨點日之後是實值）。回傳 index=交易日（start..錨點）的回推比率寬表。"""
    days = W.index[(W.index > start) & (W.index <= anchor)]
    r0 = R.loc[anchor]
    s0 = S.loc[anchor]
    codes = r0.index[r0.notna() & s0.notna() & (s0 > 0)]
    Wn = W.reindex(index=days, columns=codes).fillna(0.0)
    inc = Wn.div(s0[codes], axis=1)                       # 每日比率增量
    # t 日收盤比率：r(t) = r(anchor) − Σ_{s∈(t, anchor]} inc(s) = r0 − (total − cumsum(t))
    cs = inc.cumsum()
    total = cs.iloc[-1]
    rc = (cs - total).add(r0[codes], axis=1)
    return rc


def fo_panel():
    """外資持股比率（小數）日頻寬表，對齊面板 IDX（只填上市代碼）。
    2021-10-07（回推起點）～2025-10-06：回推值；2025-10-07 起：MI_QFIIS 實值。另回傳回推旗標寬表。"""
    R, S, Fs = load_qfiis()
    W = load_flows()
    bc = backcast(R, S, W, ANCHOR, pd.Timestamp('2021-10-07'))
    act = R[R.index >= ANCHOR]
    allp = pd.concat([bc[bc.index < ANCHOR], act])
    allp = allp[~allp.index.duplicated(keep='last')]
    P = allp.reindex(columns=COLS)
    P = P.reindex(IDX.union(P.index)).sort_index()
    P = P.ffill(limit=5).reindex(IDX)
    P[IDX < pd.Timestamp('2021-10-07')] = np.nan
    flag_bc = pd.Series(IDX < ANCHOR, index=IDX)
    return P, flag_bc


# ---------------- 主策略（只讀 v2_core） ----------------
_S = {}
def strategy(exfin=True):
    if exfin not in _S:
        YL, Y5, _ = V.load_revenue()
        _S[exfin] = V.V2('gapfix', YL, Y5, exfin=exfin)
    return _S[exfin]


def period_returns(plan, t_end):
    """每期（成交日開盤 → 下一成交日開盤）每檔報酬（無成本）。回傳 list of (d, e, picks, r)。"""
    ex = [p[0] for p in plan]
    out = []
    for i, p in enumerate(plan):
        e, pk, d = p[0], p[1], p[3]
        e2 = ex[i + 1] if i + 1 < len(ex) else None
        last = e2 if e2 is not None else t_end
        r = open_to_open(e, last, pk, to_open=e2 is not None)
        out.append((d, e, pk, r))
    return out


def open_to_open(e, e2, cols, to_open=True):
    """成交日 e 開盤買進 → e2 開盤（to_open）或 e2 收盤賣出的報酬（停牌凍結＝報酬 0 期間不變）。"""
    cols = np.asarray(cols, int)
    r = K.Cn[e, cols] / K.On[e, cols]
    if e2 > e:
        r = r * np.prod(K.GROW[e + 1:e2 + 1, cols], axis=0)
    if to_open:
        tr = K.TRADED[e2, cols] & np.isfinite(K.On[e2, cols])
        r = np.where(tr, r * K.On[e2, cols] / np.where(tr, K.Cn[e2, cols], 1.0), r)
    return r - 1.0


def tstat(x):
    return K.tstat(x)


def monthly_t(nav, bench, sig_days, a, b):
    """訊號窗（訊號日收盤 → 下一訊號日收盤）月超額的均值、標準差、t（a ≤ 訊號日 ≤ b）。"""
    D = pd.DatetimeIndex([d for d in sig_days if d in nav.index])
    nd = nav.reindex(D); bd = bench.reindex(D)
    pr = (nd.shift(-1) / nd - 1).iloc[:-1]; br = (bd.shift(-1) / bd - 1).iloc[:-1]
    x = (pr - br)
    x = x[(x.index >= pd.Timestamp(a)) & (x.index <= pd.Timestamp(b))].dropna()
    return x


def summ(x):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    if len(x) < 3: return {'n': int(len(x))}
    return {'n': int(len(x)), 'mean%': round(float(x.mean()) * 100, 3), 'sd%': round(float(x.std(ddof=1)) * 100, 3),
            't': round(float(x.mean() / (x.std(ddof=1) / math.sqrt(len(x)))), 2)}


def cagr_dd(nav, a, b):
    s = nav[(nav.index >= pd.Timestamp(a)) & (nav.index <= pd.Timestamp(b))].dropna()
    if len(s) < 5: return {}
    i0 = nav.index.get_loc(s.index[0])
    base = nav.iloc[i0 - 1] if i0 > 0 else s.iloc[0]
    st = nav.index[i0 - 1] if i0 > 0 else s.index[0]
    yrs = (s.index[-1] - st).days / 365.25
    eq = s / base
    return {'CAGR%': round(float(eq.iloc[-1] ** (1 / yrs) - 1) * 100, 2), 'maxDD%': round(float((eq / eq.cummax() - 1).min()) * 100, 2),
            'years': round(yrs, 2)}


# ---------------- 外資持股訊號（訊號日 t 只用 t−1 收盤已公布的值） ----------------
def check_freeze():
    fz = json.loads((HERE / 'data_freeze.json').read_text(encoding='utf-8'))
    for f, h in fz['files'].items():
        assert sha(HERE / f) == h, f'凍結檔案被改動：{f}'
    return fz


_FO = {}
def fo_data():
    if not _FO:
        check_freeze()
        P = pd.read_parquet(HERE / 'fo_panel.parquet').reindex(index=IDX, columns=COLS).to_numpy(dtype=float)
        _FO['P'] = np.clip(P, 0.0, 1.0)          # 回推可能出現負值（約 0.4% 的格子 < −0.5pp）→ 截到 0
        sh = pd.read_csv(HERE / 'shares_anchor.csv', dtype={'code': str} if False else None, index_col=0)
        sh.index = sh.index.astype(str)
        _FO['shares'] = sh['shares'].reindex(COLS).to_numpy(dtype=float)
    return _FO


def fo_features(t):
    """t＝訊號日索引。回傳 dict：lvl（t−1）、d6（t−1 對 t−1−126）、d12（t−1 對 t−1−252）、mcap（近似：2025-10 股數 × t 收盤）。"""
    D = fo_data(); P = D['P']
    lvl = P[t - 1]
    d6 = P[t - 1] - P[t - 1 - 126] if t - 1 - 126 >= 0 else np.full(N, np.nan)
    d12 = P[t - 1] - P[t - 1 - 252] if t - 1 - 252 >= 0 else np.full(N, np.nan)
    mcap = D['shares'] * K.Cn[t]
    return {'lvl': lvl, 'd6': d6, 'd12': d12, 'mcap': mcap}


def terciles(mcap, el):
    """池內市值三分組：0 小、1 中、2 大（NaN 市值 → −1）。"""
    g = np.full(N, -1)
    ok = el & np.isfinite(mcap) & (mcap > 0)
    r = pd.Series(np.where(ok, mcap, np.nan)).rank(pct=True).to_numpy()
    g[ok] = np.minimum((r[ok] * 3 - 1e-12).astype(int), 2)
    return g


# ---------------- 版本引擎（M2：訊號日收盤算、隔日開盤成交、開盤漲停的新買進遞補） ----------------
COST = {'normal': K.rates(0.001), 'stress': (K.FEE + 0.004, K.FEE + K.TAX + 0.004)}   # 壓力＝每邊再加 0.3% 滑價


def _pick(cand, gu, held, n):
    pick = []
    for j in cand:
        if gu[j] and j not in held: continue
        pick.append(j)
        if len(pick) == n: break
    return pick


def sleeve_plan(S, mode, a, b, n_per=10, n_tot=30):
    """第二條腿名單。mode：'lvl_size'（市值三組內外資持股最高 n_per 檔）、'd6_size'（三組內 6 月增量最高）、'lvl_pool'（池內最高 n_tot）。"""
    out = []; held = set()
    for d in S.sig_days:
        if not (pd.Timestamp(a) <= d <= pd.Timestamp(b)): continue
        cs = S.cross_section(d); el = cs['el']
        if el.sum() < 50: continue
        t, e = cs['t'], cs['e']; f = fo_features(t); gu = K.gap_up_open(e)
        key = f['d6'] if mode.startswith('d6') else f['lvl']
        ok = el & np.isfinite(key)
        if ok.sum() < 30: continue
        if mode.endswith('_size'):
            g = terciles(f['mcap'], el)
            pick = []
            for gg in (0, 1, 2):
                m = ok & (g == gg)
                cand = np.where(m)[0][np.argsort(-key[m], kind='stable')]
                pick += _pick(cand, gu, held, n_per)
        else:
            cand = np.where(ok)[0][np.argsort(-key[ok], kind='stable')]
            pick = _pick(cand, gu, held, n_tot)
        out.append((e, np.array(pick, int), np.where(el)[0], d))
        held = set(pick)
    return out


def main_plan(S, a, b, n=10, third=None, w3=0.2):
    """主策略名單；third＝None（原版）、'lvl'、'd6'：分數改為 (1−w3)/2×52 週＋(1−w3)/2×營收＋w3×（市值組內外資持股或其 6 月增量百分位）。
    外資資料缺值者第三分數給 0.5（中性）。"""
    if third is None:
        return [p for p in S.plan(n) if pd.Timestamp(a) <= p[3] <= pd.Timestamp(b)]
    out = []; held = set()
    for d in S.sig_days:
        if not (pd.Timestamp(a) <= d <= pd.Timestamp(b)): continue
        cs = S.cross_section(d); el = cs['el']
        if el.sum() < 50: continue
        t, e = cs['t'], cs['e']; f = fo_features(t); gu = K.gap_up_open(e)
        key = f['d6'] if third == 'd6' else f['lvl']
        g = terciles(f['mcap'], el)
        p3 = np.full(N, 0.5)
        for gg in (0, 1, 2):
            m = el & (g == gg) & np.isfinite(key)
            if m.sum() > 3:
                p3[m] = pd.Series(key[m]).rank(pct=True).to_numpy()
        pr = lambda x: pd.Series(np.where(el, x, np.nan)).rank(pct=True).to_numpy()
        sc = (1 - w3) / 2 * pr(cs['s52']) + (1 - w3) / 2 * pr(cs['y3']) + w3 * p3
        sc[~el] = np.nan
        cs2 = dict(cs); cs2['sc'] = sc
        cand = V.V2.order(cs2, 'official')[:int(el.sum())]
        pick = _pick(cand, gu, held, n)
        out.append((e, np.array(pick, int), np.where(el)[0], d))
        held = set(pick)
    return out


def run_plan(plan, cost, t1):
    t0 = plan[0][0]
    nav, lg = K.simulate([(p[0], p[1]) for p in plan], *COST[cost], t0, t1)
    return nav, lg


def ew_nav(plan, t1):
    t0 = plan[0][0]
    nav, _ = K.simulate([(p[0], p[2]) for p in plan], 0, 0, t0, t1)
    return nav


def combine(nav_a, nav_b, plan, w=0.8):
    """每個成交日開盤前（以前一日收盤近似）把兩個子組合重設回 w：1−w。兩條 nav 需同起點。"""
    idx = nav_a.index.intersection(nav_b.index)
    A, B = nav_a.reindex(idx), nav_b.reindex(idx)
    reset = set(IDX[[p[0] - 1 for p in plan]])
    out = np.empty(len(idx)); v = 1.0; ia = A.iloc[0]; ib = B.iloc[0]; base = 1.0
    for k, d in enumerate(idx):
        val = base * (w * A.iloc[k] / ia + (1 - w) * B.iloc[k] / ib)
        out[k] = val
        if d in reset:
            base = val; ia = A.iloc[k]; ib = B.iloc[k]
    return pd.Series(out, index=idx)
