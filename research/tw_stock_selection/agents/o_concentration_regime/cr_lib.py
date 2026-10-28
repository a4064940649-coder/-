#!/usr/bin/env python3
"""o_concentration_regime 共用函式庫（權值集中狀態的持續性：0050 對同池等權的相對強弱輪動）。

只讀：scratchpad/panel（adj、close_raw、value）、本資料夾 raw/（各研究的淨值、0050、ETF、法人）。
寫：只寫本資料夾 data/ 與結果 json。

時間口徑（全部事前固定）：
* 決策日 T＝主策略官方訊號日（每月第一個日期 ≥ 11 的面板 GOOD 交易日；與 plan_V4.csv 的 signal 欄逐一核對）。
* 視窗 k＝[T_k, T_{k+1}]（收盤到收盤）。d_k＝ln(1+r0050_k) − ln(1+rEW_k)。
* T_k 收盤算訊號，主策略在 T_k+1 開盤換股；輪動腿的移轉在 T_k+1（exec 日）收盤執行（與 integrate_o6 的袖子混合同一時點）。
* 同池等權（訊號用）：T_k 當日 60 日均成交值（GOOD 日滾動，min_periods 50）≥ 1,000 萬、收盤 ≥ 10、排除 KY／DR，取前 300 名；
  收盤到收盤含息等權報酬，下市股以最後價計（adj ffill）。
"""
import json, hashlib, math
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
RAW, DATA = HERE / 'raw', HERE / 'data'
PANEL = Path('/tmp/claude-0/-home-user--/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/panel')
RESULTS = HERE / 'results.json'

SEL = ('2019-01-01', '2022-12-31')      # 挑選期（視窗起點所在月份）
VAL = ('2023-01-01', '2026-10-31')      # 驗證期
PRE = ('2006-01-01', '2018-12-31')      # 結構前對照（只檢驗 P1、P5）

FEE = 0.001425 * 0.6
COST = {   # (主策略腿 買, 主策略腿 賣, 0050 買, 0050 賣)
    'normal': (FEE + 0.001, FEE + 0.003 + 0.001, FEE + 0.0005, FEE + 0.001 + 0.0005),
    'stress': (0.005, 0.005, FEE + 0.0035, FEE + 0.001 + 0.0035),
    'zero': (0.0, 0.0, 0.0, 0.0),
    'io6': (FEE + 0.001, FEE + 0.003 + 0.001, FEE + 0.001, FEE + 0.002),   # 只用來重現 integrate_o6 的袖子混合（ETF 買＝股票買、賣＝買＋0.1%）
}


def sha(f):
    return hashlib.sha256(Path(f).read_bytes()).hexdigest()


def load_json(p=RESULTS):
    p = Path(p)
    return json.loads(p.read_text(encoding='utf-8')) if p.exists() else {}


def save_json(obj, p=RESULTS):
    Path(p).write_text(json.dumps(obj, ensure_ascii=False, indent=1, default=_js), encoding='utf-8')


def _js(o):
    if isinstance(o, (np.integer,)): return int(o)
    if isinstance(o, (np.floating,)): return None if not np.isfinite(o) else float(o)
    if isinstance(o, (np.bool_,)): return bool(o)
    if isinstance(o, (pd.Timestamp,)): return str(o.date())
    if isinstance(o, np.ndarray): return o.tolist()
    return str(o)


# ---------------- 0050 與 ETF ----------------
def e50_series():
    e = pd.read_csv(RAW / 'etf_0050.csv', parse_dates=['date']).set_index('date')['adj'].sort_index()
    e = e[e.index != pd.Timestamp('2026-09-20')]     # 週日壞資料（w3_size_rotation_28 已記錄），事前剔除
    return e


def etf_long(code):
    """etf_long 的 adj，做分割修補：日比值 < 0.5 視為分割，乘回整數倍。只當敏感度。"""
    el = pd.read_csv(RAW / 'etf_long.csv.gz', dtype={'code': str}, parse_dates=['date'])
    s = el[el.code == code].set_index('date')['adj'].sort_index()
    q = (s / s.shift(1)).to_numpy().copy()
    for i in range(1, len(q)):
        if q[i] < 0.5:
            k = round(1 / q[i]); q[i] *= k
    q[0] = 1.0
    return pd.Series(s.iloc[0] * np.cumprod(q), index=s.index)


def on_cal(s, ix):
    """把價格序列放到日曆 ix（asof ffill）。"""
    return s.reindex(ix.union(s.index)).ffill().reindex(ix)


# ---------------- 外國公司（KY／DR）----------------
def foreign_codes(cols):
    fz = json.loads((RAW / 'rev_survivorship_data_freeze.json').read_text(encoding='utf-8'))
    F = set(fz['foreign_codes']) | {c for c in cols if c.startswith('91')}
    dl = pd.read_csv(RAW / 'delist_official.csv', dtype=str)
    F |= {c for c, n in zip(dl.code, dl.name) if isinstance(n, str) and (n.startswith('F-') or 'KY' in n.upper())}
    sl = pd.read_csv(RAW / 'stock_list.csv', dtype=str)
    F |= {c for c, n in zip(sl.code, sl.name) if isinstance(n, str) and 'KY' in n.upper()}
    return F


# ---------------- 面板與訊號基礎 ----------------
class Panel:
    def __init__(self, end=None):
        adj = pd.read_parquet(PANEL / 'adj.parquet'); C = pd.read_parquet(PANEL / 'close_raw.parquet'); V = pd.read_parquet(PANEL / 'value.parquet')
        if end is not None:
            e = pd.Timestamp(end); adj, C, V = adj[adj.index <= e], C[C.index <= e], V[V.index <= e]
        self.adj, self.C, self.V = adj, C, V
        self.IDX = adj.index
        self.GOOD = adj.notna().sum(axis=1) >= 300
        F = foreign_codes(adj.columns)
        self.keep = np.array([c not in F for c in adj.columns])
        g = self.GOOD
        self.v60 = V.loc[g].rolling(60, min_periods=50).mean().reindex(self.IDX)
        self.AC = adj.ffill()

    def official_dates(self, start='2004-03-01'):
        ii = self.IDX[(self.IDX >= pd.Timestamp(start)) & self.GOOD.to_numpy()]
        s = pd.Series(ii, index=ii)
        return [g[g.dt.day >= 11].iloc[0] for _, g in s.groupby([ii.year, ii.month]) if (g.dt.day >= 11).any()]

    def asof(self, d):
        """日期 d 在面板上的最近一個 ≤ d 的 GOOD 日。"""
        gi = self.IDX[self.GOOD.to_numpy()]
        k = gi.searchsorted(pd.Timestamp(d), side='right') - 1
        return gi[k]

    def pool(self, d):
        d = self.asof(d)
        v = self.v60.loc[d].to_numpy(); px = self.C.loc[d].to_numpy()
        base = (v >= 1e7) & (px >= 10) & self.keep & np.isfinite(v)
        vv = np.where(base, v, -np.inf)
        k = int(min(300, base.sum()))
        idx = np.argsort(-vv)[:k]
        return idx, v[idx]

    def windows(self, dates):
        """對連續決策日 dates 回傳每個視窗的同池等權報酬、前 5 大成交值占比、池大小。"""
        rows = []
        for k in range(len(dates) - 1):
            a, b = self.asof(dates[k]), self.asof(dates[k + 1])
            idx, v = self.pool(dates[k])
            pa = self.AC.loc[a].to_numpy()[idx]; pb = self.AC.loc[b].to_numpy()[idx]
            r = pb / pa - 1
            ok = np.isfinite(r)
            top5 = np.sort(v)[::-1][:5].sum() / v.sum()
            top1 = np.sort(v)[::-1][:1].sum() / v.sum()
            rows.append(dict(start=pd.Timestamp(dates[k]), end=pd.Timestamp(dates[k + 1]), rEW=float(np.mean(r[ok])), n_pool=int(len(idx)),
                             n_ret=int(ok.sum()), top5=float(top5), top1=float(top1)))
        return pd.DataFrame(rows).set_index('start')


def add_e50(W, e50):
    s = on_cal(e50, pd.DatetimeIndex(sorted(set(W.index) | set(W['end']))))
    W = W.copy()
    W['r50'] = (s.reindex(W['end']).to_numpy() / s.reindex(W.index).to_numpy()) - 1
    W['d'] = np.log1p(W['r50']) - np.log1p(W['rEW'])
    return W


# ---------------- 訊號與狀態 ----------------
def signal(W, kind, L):
    """在每個決策日 T_k（W 的 index 是視窗起點＝決策日；訊號用到 T_k 為止已結束的視窗，也就是 W 的前幾列）。
    kind: 'S1'＝過去 L 個視窗 d 的累積；'S1skip'＝過去第 2～L+1 個視窗；'S2'＝前 5 大占比的 L 個月變化（以決策日橫截面）。
    回傳 Series（index＝決策日＝視窗起點）。"""
    d = W['d']
    past = d.shift(1)      # 決策日 T_k 時已知的是「結束於 T_k 的視窗」＝前一列
    if kind == 'S1':
        return past.rolling(L, min_periods=L).sum()
    if kind == 'S1skip':
        return past.shift(1).rolling(L, min_periods=L).sum()
    if kind == 'S2':
        t5 = W['top5']       # T_k 當日的橫截面占比（決策日已知）
        return t5 - t5.shift(L)
    raise ValueError(kind)


def states(S, h):
    """遲滯帶 h：S > h → C（集中期）、S < −h → D（擴散期）、其間維持前一狀態；第一個有效值在帶內時取 D。"""
    out = []; prev = None
    for v in S.to_numpy():
        if not np.isfinite(v):
            out.append(None); continue
        if v > h: st = 'C'
        elif v < -h: st = 'D'
        else: st = prev if prev is not None else 'D'
        out.append(st); prev = st
    return pd.Series(out, index=S.index)


# ---------------- 淨值模擬 ----------------
def rotate(mainnav, e50, exec_w, cost='normal'):
    """mainnav：主策略腿日頻淨值（已扣自身成本）；e50：0050 價格（任意日曆，asof 放到 mainnav 日曆）；
    exec_w：{exec 日: 0050 目標權重}（在 exec 日收盤調整；權重間漂移）。成本：主腿與 0050 各自單邊。
    回傳（日頻淨值, 交易紀錄 DataFrame）。"""
    bM, sM, bE, sE = COST[cost]
    ix = mainnav.index
    rM = mainnav.pct_change().fillna(0).to_numpy()
    rE = on_cal(e50, ix).pct_change().fillna(0).to_numpy()
    ex = {ix.get_indexer([pd.Timestamp(d)])[0]: w for d, w in exec_w.items() if pd.Timestamp(d) in ix}
    first = min(ex) if ex else 0
    a = b = 0.0; nav = np.full(len(ix), np.nan); log = []
    for i in range(first, len(ix)):
        if i == first:
            w = ex[i]; a, b = 1 - w, w
        else:
            a *= 1 + rM[i]; b *= 1 + rE[i]
            if i in ex:
                w = ex[i]; tot = a + b; tb = w * tot
                if b < tb:   # 主腿 → 0050
                    m = tb - b; c = m * (sM + bE)
                else:
                    m = b - tb; c = m * (sE + bM)
                tot -= c; a, b = (1 - w) * tot, w * tot
                log.append(dict(date=ix[i], moved=m / (tot + c), cost=c / (tot + c), w=w))
        nav[i] = a + b
    return pd.Series(nav, index=ix).dropna(), pd.DataFrame(log)


def exec_map(dec_dates, weights, ix):
    """決策日 → exec 日（決策日之後 mainnav 日曆的下一個交易日）。"""
    out = {}
    for d, w in zip(dec_dates, weights):
        k = ix.searchsorted(pd.Timestamp(d), side='right')
        if k < len(ix) and w is not None and np.isfinite(w): out[ix[k]] = float(w)
    return out


# ---------------- 統計 ----------------
def win_ret(nav, sd):
    D = pd.DatetimeIndex([pd.Timestamp(d) for d in sd if pd.Timestamp(d) in nav.index])
    nd = nav.reindex(D)
    return (nd.shift(-1) / nd - 1).iloc[:-1]


def sel(x, per):
    return x[(x.index >= pd.Timestamp(per[0])) & (x.index <= pd.Timestamp(per[1]))].dropna()


def tstat(x):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    if len(x) < 3 or x.std(ddof=1) == 0: return float('nan')
    return float(x.mean() / (x.std(ddof=1) / math.sqrt(len(x))))


def nw_t(x, lag):
    x = np.asarray(x, float); x = x[np.isfinite(x)]; n = len(x)
    if n < 5: return float('nan')
    e = x - x.mean(); s = (e @ e) / n
    for l in range(1, lag + 1):
        s += 2 * (1 - l / (lag + 1)) * (e[l:] @ e[:-l]) / n
    return float(x.mean() / math.sqrt(s / n)) if s > 0 else float('nan')


def ols_hac(y, x, lag):
    y = np.asarray(y, float); x = np.asarray(x, float); m = np.isfinite(y) & np.isfinite(x); y, x = y[m], x[m]; n = len(y)
    X = np.column_stack([np.ones(n), x]); XtX = np.linalg.inv(X.T @ X); b = XtX @ X.T @ y; u = y - X @ b
    S = (X * u[:, None]).T @ (X * u[:, None]) / n
    for l in range(1, lag + 1):
        G = (X[l:] * u[l:, None]).T @ (X[:-l] * u[:-l, None]) / n
        S += (1 - l / (lag + 1)) * (G + G.T)
    V = n * XtX @ S @ XtX
    se_iid = math.sqrt((u @ u) / (n - 2) * XtX[1, 1])
    return dict(beta=float(b[1]), alpha=float(b[0]), t_hac=float(b[1] / math.sqrt(V[1, 1])), t_iid=float(b[1] / se_iid), n=int(n),
                r2=float(1 - (u @ u) / ((y - y.mean()) @ (y - y.mean()))))


def runs(lbl):
    """連續相同標籤的區段 → list of (start_pos, end_pos_inclusive, label)。"""
    lbl = list(lbl); out = []; s = 0
    for i in range(1, len(lbl) + 1):
        if i == len(lbl) or lbl[i] != lbl[s]:
            out.append((s, i - 1, lbl[s])); s = i
    return out


def state_block_boot(x, lbl, B=5000, seed=0, stat=np.mean):
    """狀態區塊 bootstrap：以連續相同狀態的區段為區塊，有放回抽區塊直到長度 ≥ n，截到 n。回傳 2.5/5/50/95/97.5 百分位與 P(stat ≤ 0)。"""
    x = np.asarray(x, float); n = len(x); rs = runs(lbl)
    blocks = [x[a:b + 1] for a, b, _ in rs]
    rng = np.random.default_rng(seed); out = np.empty(B)
    for i in range(B):
        acc = []; L = 0
        while L < n:
            bl = blocks[rng.integers(len(blocks))]; acc.append(bl); L += len(bl)
        out[i] = stat(np.concatenate(acc)[:n])
    q = np.percentile(out, [2.5, 5, 50, 95, 97.5])
    return dict(p2_5=float(q[0]), p5=float(q[1]), p50=float(q[2]), p95=float(q[3]), p97_5=float(q[4]), p_le0=float((out <= 0).mean()), n_blocks=len(blocks))


def mbb_boot(x, block=6, B=5000, seed=1, stat=np.mean):
    x = np.asarray(x, float); n = len(x); rng = np.random.default_rng(seed); out = np.empty(B)
    for i in range(B):
        st = rng.integers(0, n - block + 1, size=math.ceil(n / block))
        out[i] = stat(np.concatenate([x[s:s + block] for s in st])[:n])
    q = np.percentile(out, [2.5, 5, 50, 95, 97.5])
    return dict(p2_5=float(q[0]), p5=float(q[1]), p50=float(q[2]), p95=float(q[3]), p97_5=float(q[4]), p_le0=float((out <= 0).mean()))


def cagr_mdd(nav, a, b):
    s = nav[(nav.index >= pd.Timestamp(a)) & (nav.index <= pd.Timestamp(b))].dropna()
    if len(s) < 2: return float('nan'), float('nan')
    yrs = (s.index[-1] - s.index[0]).days / 365.25
    cg = (s.iloc[-1] / s.iloc[0]) ** (1 / yrs) - 1
    mdd = float((s / s.cummax() - 1).min())
    return float(cg), mdd


def te_ann(r_a, r_b):
    x = (r_a - r_b).dropna(); return float(x.std(ddof=1) * math.sqrt(12))
