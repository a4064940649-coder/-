#!/usr/bin/env python3
"""o_forward_sequential_v2 共用函式庫。
- 只讀本資料夾的 raw/ 與 vendor_o7/（o_tier 的 tlib／hmethods／seqlib 副本，只 import、不改）。
- 決策序列：主策略 B10 可執行（V7 navs_exec7，K5T，一般／壓力成本，4 級距 × FIX／GROW）在 13 個 j 的
  「錨定月窗超額」（對 0050 含息，pp），每月取 13 個 j 的中位數＝j 中位數決策序列（o7 教訓 1）。
- 錨定月：m 月的法定期限 D_m＝m 月 10 日起第一個交易日；任一訊號日 d 的錨定月＝滿足 D_m ≤ d 的最大 m。
- 窗：每個 j 自己的訊號日收盤 → 下一個訊號日收盤（最後一窗到 jscan_end_days）；超額＝線窗報酬 − 0050 同窗報酬。
- PREREG 工具：results.json 的 key 與內容 sha256。
"""
import sys, json, hashlib, math, datetime
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
AG = HERE.parent
RAW, VEN, DATA = HERE / 'raw', HERE / 'vendor_o7', HERE / 'data'
RES_F = HERE / 'results.json'
DATA.mkdir(exist_ok=True)
sys.path.insert(0, str(VEN))
import tlib as T          # noqa: E402
import hmethods as H      # noqa: E402
import seqlib as SQ       # noqa: E402
sys.path.pop(0)

TIERS = ['30萬', '100萬', '300萬', '1000萬']
MODES = ['FIX', 'GROW']
JS = list(range(13))
SEL_END = pd.Period('2019-04', 'M')
VAL = (pd.Period('2019-05', 'M'), pd.Period('2023-12', 'M'))
HOLD = (pd.Period('2024-01', 'M'), pd.Period('2026-10', 'M'))
MAIN_CELL = ('100萬', 'FIX')            # 主決策線（事前指定）：B10 K5T j 中位數、100 萬、FIX、一般成本
MAIN_CLUSTER = 114


# ---------------------------------------------------------------- results.json
def _js(o):
    if isinstance(o, (np.integer,)): return int(o)
    if isinstance(o, (np.floating,)): return float(o)
    if isinstance(o, np.ndarray): return o.tolist()
    if isinstance(o, (pd.Timestamp, pd.Period)): return str(o)
    if isinstance(o, (np.bool_,)): return bool(o)
    return str(o)


def canon_sha(obj):
    return hashlib.sha256(json.dumps(obj, ensure_ascii=False, sort_keys=True, default=_js).encode()).hexdigest()


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def now_utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def load_res():
    return json.loads(RES_F.read_text(encoding='utf-8')) if RES_F.exists() else {}


def save_key(key, obj, need=()):
    res = load_res()
    for k in need: assert k in res, f'{k} 不存在：執行順序錯誤'
    res[key] = obj
    RES_F.write_text(json.dumps(res, ensure_ascii=False, indent=1, default=_js), encoding='utf-8')


def register(key, obj):
    """登記（不可覆寫）：寫入 obj 與 sha256。"""
    res = load_res()
    assert key not in res, f'{key} 已登記，不可覆寫'
    res[key] = obj; res[key + '_sha256'] = canon_sha(obj)
    RES_F.write_text(json.dumps(res, ensure_ascii=False, indent=1, default=_js), encoding='utf-8')
    return res[key + '_sha256']


def check_prereg(key='PREREG', files=True):
    res = load_res()
    assert key in res and key + '_sha256' in res, f'{key} 不存在：樣本外計算前必須先凍結'
    assert canon_sha(res[key]) == res[key + '_sha256'], f'{key} 被改過'
    if files:
        for rel, h in res[key].get('file_hashes', {}).items():
            assert sha(HERE / rel) == h, f'{rel} 與 {key} 登記的 sha256 不符'
    return res[key]


# ---------------------------------------------------------------- 價格與日曆
def e50():
    e = pd.read_csv(RAW / 'finlab/data/raw/etf_0050.csv', parse_dates=['date']).set_index('date')['adj'].sort_index()
    return e[~e.index.duplicated(keep='last')]


_EX = {}
def exec_navs():
    if 'x' not in _EX: _EX['x'] = pd.read_parquet(RAW / 'v7/navs_exec7.parquet')
    return _EX['x']


def cal():
    return exec_navs().index


def deadlines(c=None):
    c = cal() if c is None else c
    out = {}
    for p in pd.period_range(pd.Period(c[0], 'M'), pd.Period(c[-1], 'M'), freq='M'):
        d = c[(c >= pd.Timestamp(p.year, p.month, 10)) & (c.year == p.year) & (c.month == p.month)]
        if len(d): out[p] = d[0]
    return pd.Series(out)


def anchor_of(dates, Dm=None):
    Dm = deadlines() if Dm is None else Dm
    v = Dm.values; per = Dm.index
    out = []
    for d in pd.DatetimeIndex(dates):
        k = np.searchsorted(v, np.datetime64(d), side='right') - 1
        out.append(per[k] if k >= 0 else None)
    return out


def j_signals(j, strat='AB_V7_B10'):
    p = pd.read_csv(RAW / 'v7/plans_j7.csv', usecols=['strat', 'j', 'seed', 'signal'])
    p = p[(p.strat == strat) & (p.j == j) & (p.seed == 0)]
    return pd.DatetimeIndex(sorted(pd.to_datetime(p.signal).unique()))


def j_end(j, strat='AB_V7_B10'):
    d = json.loads((RAW / 'v7/jscan_end_days.json').read_text())
    return pd.Timestamp(d[f'{strat}|J{j}'])


def _at(s, d):
    s = s.dropna()
    k = s.index.searchsorted(d, side='right') - 1
    return float(s.iloc[k]) if k >= 0 else np.nan


def window_excess(nav, sig, end, E=None, Dm=None):
    """nav：日淨值；sig：訊號日；end：最後一窗終點。回傳 DataFrame（index＝錨定月）：start、end、r、r50、ex（pp）。"""
    E = e50() if E is None else E
    pts = list(sig[sig < end]) + [end]
    rows = []
    for a, b in zip(pts[:-1], pts[1:]):
        r = _at(nav, b) / _at(nav, a) - 1
        r50 = _at(E, b) / _at(E, a) - 1
        rows.append((a, b, r, r50))
    D = pd.DataFrame(rows, columns=['start', 'end', 'r', 'r50'])
    D['anchor'] = anchor_of(D.start, Dm)
    D['ex'] = (D.r - D.r50) * 100
    return D.set_index('anchor')


def cell_col(j, tier, mode, model='K5T', cost='normal', strat='B10'):
    return f'{strat}|J{j}|{model}|{cost}|{tier}|{mode}'


def cell_jmatrix(tier, mode, model='K5T', cost='normal', strat='B10', navs=None):
    """13 個 j 的錨定月窗超額矩陣（月 × j）與各 j 的窗報酬（給配置模擬）。"""
    X = exec_navs() if navs is None else navs
    E = e50(); Dm = deadlines()
    exs, rs, r50s = {}, {}, {}
    for j in JS:
        W = window_excess(X[cell_col(j, tier, mode, model, cost, strat)], j_signals(j), j_end(j), E, Dm)
        W = W[~W.index.duplicated(keep='last')]
        exs[j], rs[j], r50s[j] = W.ex, W.r, W.r50
    return pd.DataFrame(exs), pd.DataFrame(rs), pd.DataFrame(r50s)


def jmed(M, min_j=7):
    """月 × j 矩陣 → 每月 j 中位數（至少 min_j 個 j 有值）。"""
    ok = M.notna().sum(axis=1) >= min_j
    return M.median(axis=1, skipna=True).where(ok).dropna()


def official_series(col='AB_V7_B10|normal', plan='plan_V7_B10.csv', end='2026-10-12'):
    nav = pd.read_parquet(RAW / 'v7/navs_v7.parquet')[col]
    sig = pd.DatetimeIndex(pd.to_datetime(pd.read_csv(RAW / 'v7' / plan)['signal']))
    W = window_excess(nav, sig, pd.Timestamp(end))
    return W[~W.index.duplicated(keep='last')]


def per_mask(idx, a=None, b=None):
    idx = pd.PeriodIndex(idx, freq='M')
    m = np.ones(len(idx), bool)
    if a is not None: m &= idx >= a
    if b is not None: m &= idx <= b
    return m


def hac_sd(x, L=6):
    """長期 sd（HAC，Bartlett L）＝ sqrt(n·HACVar(mean))。"""
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    return float(math.sqrt(T.hac_var_mean(x, L) * len(x)))


# ---------------------------------------------------------------- 族群（V4 格點窗）的錨定月
def family():
    X, lab, med = H.load_real()
    anc = anchor_of(T.WIN)
    return X, lab, med, pd.PeriodIndex(anc, freq='M')


def align_to_family(series, fam_anchor):
    """把錨定月序列對到族群窗；缺值補 0 超額（記數回報）。"""
    s = series.copy(); s.index = pd.PeriodIndex(s.index, freq='M')
    s = s[~s.index.duplicated(keep='last')]
    v = s.reindex(fam_anchor)
    return v.fillna(0.0).to_numpy(), int(v.isna().sum())
