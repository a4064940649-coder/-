#!/usr/bin/env python3
"""ADR25（騰落レシオ）逆勢擇時：回測主程式。先核對 PREREG 雜湊，結果寫入 results.json['results']。"""
import json, hashlib, math, sys
from pathlib import Path
import numpy as np, pandas as pd
H = Path(__file__).resolve().parent; AG = H.parent
sys.path.insert(0, str(AG / 'exec_rebuild'))
import exlib  # noqa: E402  (AO、AC、OPEN_RAW、cost_rates、tstat)
from exlib import tstat, FEE  # noqa: E402
RJ = json.loads((H / 'results.json').read_text())
assert hashlib.sha256(json.dumps(RJ['PREREG'], ensure_ascii=False, sort_keys=True).encode()).hexdigest() == RJ['PREREG_sha256'], 'PREREG changed'
PR = RJ['PREREG']

COSTS = {'main_rt0.1': (0.0005, 0.0005), 'stress_rt0.3': (0.0015, 0.0015),
         'exlib_etf': (FEE + 0.001, FEE + 0.001 + 0.001), 'exlib_stress': (0.005, 0.005), 'gross': (0.0, 0.0)}
WIN = {k: (pd.Timestamp(a), pd.Timestamp(b)) for k, (a, b) in PR['periods'].items()}

# ---------------- 日曆與廣度 ----------------
RET = exlib.RET; PIDX = exlib.IDX
cnt = RET.notna().sum(axis=1)
T = PIDX[(cnt >= 300).to_numpy()]
V = T[T != pd.Timestamp('2024-02-15')]
R_V = RET.loc[V]
UP = (R_V > 0).sum(axis=1); DN = (R_V < 0).sum(axis=1)
def adr(n): return (UP.rolling(n, min_periods=n).sum() / DN.rolling(n, min_periods=n).sum() * 100)
ADR = {25: adr(25), 10: adr(10)}
QL = ADR[25].rolling(1250, min_periods=750).quantile(0.05)
QH = ADR[25].rolling(1250, min_periods=750).quantile(0.95)
NT = len(T); TP = PIDX.get_indexer(T)       # T 位置 → 面板位置
assert (TP >= 0).all()

# ---------------- 0050 與開盤代理 ----------------
e50 = pd.read_csv('/home/user/jonatw/finlab-txd/data/raw/etf_0050.csv', parse_dates=['date']).set_index('date')['adj'].sort_index()
e50 = e50.drop(pd.Timestamp('2026-09-20'), errors='ignore')
tx = pd.read_csv('/home/user/jonatw/finlab-txd/data/raw/taiex_twii.csv', parse_dates=['date']).set_index('date').sort_index()
gap = (tx['open'] / tx['close'].shift(1))
o50 = e50.shift(1) * gap.reindex(e50.index)
C50 = e50.reindex(T).ffill().to_numpy()
O50 = o50.reindex(T).to_numpy()
TRAD = np.isin(T, e50.index) & np.isfinite(O50)
GAPLOCK = np.zeros(NT, bool)
_g = O50 / np.r_[np.nan, C50[:-1]] - 1
GAPLOCK[np.isfinite(_g)] = _g[np.isfinite(_g)] >= 0.094
O50X = pd.Series(np.where(TRAD, O50, np.nan)).bfill().to_numpy()      # 出場順延到下一個可交易日開盤
last = NT - 1

# ---------------- 股池與等權 ----------------
Cr = exlib.C.loc[T]; Vv = exlib.V.loc[T]
v60 = Vv.rolling(60, min_periods=60).mean()
POOL = ((v60 >= 1e7) & (Cr >= 10)).to_numpy()
AO = exlib.AO[TP]; AC = exlib.AC[TP]; OPN = np.isfinite(exlib.OPEN_RAW[TP])
RT = RET.loc[T].to_numpy()

def ew_bh(s, e, x, pool=True, trunc=False):
    m = OPN[e] & (POOL[s] if pool else True)
    px = AC[last] if trunc else AO[x]
    r = px[m] / AO[e][m] - 1
    r = r[np.isfinite(r)]
    return float(r.mean()) if len(r) else np.nan

# 每日 EW（流動性股池，前一日條件，收對收）
ew_d = np.full(NT, np.nan)
for i in range(1, NT):
    m = POOL[i - 1] & np.isfinite(RT[i])
    if m.sum() > 0: ew_d[i] = RT[i][m].mean()
EWD = pd.Series(ew_d, index=T)
R50D = pd.Series(C50, index=T).pct_change()

# ---------------- 訊號 ----------------
def series_T(s): return s.reindex(T).to_numpy()
def make_signals(N, L, Hh, mode):
    a = ADR[N]
    if L == 'Q':
        Ls, Hs = QL, QH
    else:
        Ls = pd.Series(float(L), index=a.index); Hs = pd.Series(float(Hh), index=a.index)
    if mode == 'cross':
        ent = (a.shift(1) <= Ls.shift(1)) & (a > Ls)
    else:
        ent = a <= Ls
    ext = a >= Hs
    return series_T(ent.astype(float).where(a.notna() & Ls.notna())) == 1, series_T(ext.astype(float).where(a.notna() & Hs.notna())) == 1

def simulate(ENT, EXT, maxh=60):
    trades = []; hold = False; pend_in = None; pend_out = False; e = None; reason = None
    for i in range(NT):
        # 開盤：出場
        if hold and TRAD[i] and (pend_out or i - e >= maxh):
            trades.append(dict(s=s0, e=e, x=i, reason='ADR' if pend_out else 'time', trunc=False))
            hold = False; pend_out = False
        # 開盤：進場
        if (not hold) and pend_in is not None and TRAD[i]:
            if GAPLOCK[i]:
                pass                      # 鎖漲停買不到，順延
            else:
                hold = True; e = i; s0 = pend_in; pend_in = None
        # 收盤：訊號
        if hold and EXT[i]: pend_out = True
        if (not hold) and pend_in is None and ENT[i]: pend_in = i
    if hold: trades.append(dict(s=s0, e=e, x=last, reason='trunc', trunc=True))
    return trades

# ---------------- 無條件基準快取 ----------------
_b50, _bew = {}, {}
def base50(w, h):
    k = (w, h)
    if k not in _b50:
        a, b = WIN[w]; d = np.where(TRAD & (T >= a) & (T <= b))[0]; d = d[d + h <= last]
        _b50[k] = float(np.nanmean(O50X[d + h] / O50[d] - 1)) if len(d) else np.nan
    return _b50[k]
def baseew(w, h):
    k = (w, h)
    if k not in _bew:
        a, b = WIN[w]; d = np.where((T >= a) & (T <= b))[0]; d = d[(d + h <= last) & (d >= 1)]
        r = AO[d + h] / AO[d] - 1
        m = OPN[d] & POOL[d - 1] & np.isfinite(r)
        rowm = np.where(m, r, 0.0).sum(1) / np.where(m.sum(1) > 0, m.sum(1), np.nan)   # 修正：原版 where(...,nan).sum 會整列 NaN
        _bew[k] = float(np.nanmean(rowm))
    return _bew[k]

def trade_frame(trades):
    rows = []
    for t in trades:
        s, e, x, tr = t['s'], t['e'], t['x'], t['trunc']
        raw = (C50[last] if tr else O50X[x]) / O50[e] - 1
        h = x - e
        rows.append(dict(sig=T[s], entry=T[e], exit=T[x], h=h, reason=t['reason'], trunc=tr, raw=raw,
                         ew=ew_bh(s, e, x, True, tr), ew_all=ew_bh(s, e, x, False, tr)))
    return pd.DataFrame(rows)

def event_stats(df, w, cost):
    a, b = WIN[w]; d = df[(df.entry >= a) & (df.entry <= b)].copy()
    o = {'n_trades': int(len(d))}
    if len(d) == 0: return o
    buy, sell = COSTS[cost]
    d['net'] = (1 + d.raw) * (1 - sell) / (1 + buy) - 1
    d['abn'] = d.net - d.ew; d['abn_all'] = d.net - d.ew_all
    d['b50'] = [base50(w, int(h)) for h in d.h]; d['x50'] = d.net - d.b50
    d['m'] = d.entry.dt.to_period('M')
    g = d.groupby('m')[['net', 'abn', 'abn_all', 'x50']].mean()
    o.update({'n_months': int(len(g)), 'avg_hold_days': round(float(d.h.mean()), 1),
              'exit_ADR_share': round(float((d.reason == 'ADR').mean()), 3), 'n_trunc': int(d.trunc.sum()),
              'net_mean_pct': round(float(d.net.mean() * 100), 2), 'net_hit': round(float((d.net > 0).mean()), 3),
              'abnEW_mean_pct': round(float(d.abn.mean() * 100), 2), 'abnEW_t_month': tstat(g.abn), 'abnEW_t_trade': tstat(d.abn),
              'abnEW_hit': round(float((d.abn > 0).mean()), 3),
              'abnEWall_mean_pct': round(float(d.abn_all.mean() * 100), 2), 'abnEWall_t_month': tstat(g.abn_all),
              'vs0050uncond_mean_pct': round(float(d.x50.mean() * 100), 2), 'vs0050uncond_t_month': tstat(g.x50),
              'base0050_mean_pct': round(float(d.b50.mean() * 100), 2)})
    return o

def nav_series(trades, cost):
    buy, sell = COSTS[cost]; f = np.ones(NT); held = np.zeros(NT, bool)
    for t in trades:
        e, x, tr = t['e'], t['x'], t['trunc']
        f[e] *= C50[e] / O50[e] / (1 + buy); held[e] = True
        end = x if tr else x - 1
        for i in range(e + 1, end + 1): f[i] *= C50[i] / C50[i - 1]; held[i] = True
        if not tr: f[x] *= O50X[x] / C50[x - 1] * (1 - sell)
    return pd.Series(f - 1, index=T), pd.Series(held, index=T)

def port_stats(r, held, w):
    a, b = WIN[w]; m = (r.index >= a) & (r.index <= b)
    rr = r[m]; yrs = (rr.index[-1] - rr.index[0]).days / 365.25
    eq = (1 + rr).cumprod()
    def cagr(x): return round(float((1 + x.fillna(0)).prod() ** (1 / yrs) - 1), 4)
    def mdd(x): q = (1 + x.fillna(0)).cumprod(); return round(float((q / q.cummax() - 1).min()), 4)
    b50 = R50D[m]; bew = EWD[m]
    mo = lambda x: (1 + x.fillna(0)).groupby(x.index.to_period('M')).prod() - 1
    ms, m50, mew = mo(rr), mo(b50), mo(bew)
    return {'CAGR': cagr(rr), 'maxDD': mdd(rr), 'exposure': round(float(held[m].mean()), 3),
            '0050_CAGR': cagr(b50), '0050_maxDD': mdd(b50), 'EW_CAGR': cagr(bew), 'EW_maxDD': mdd(bew),
            't_vs_0050_m': tstat(ms - m50), 't_vs_EW_m': tstat(ms - mew),
            'yearly_pct': {int(y): round(float((1 + g).prod() - 1) * 100, 1) for y, g in rr.groupby(rr.index.year)}}

def ew_target_stats(df, w):
    """敏感度：交易標的改為流動性股池等權。"""
    a, b = WIN[w]; d = df[(df.entry >= a) & (df.entry <= b)].copy()
    if len(d) == 0: return {'n_trades': 0}
    buy, sell = COSTS['main_rt0.1']
    d['net'] = (1 + d.ew) * (1 - sell) / (1 + buy) - 1
    d['bew'] = [baseew(w, int(h)) for h in d.h]; d['x'] = d.net - d.bew
    g = d.groupby(d.entry.dt.to_period('M'))['x'].mean()
    return {'n_trades': int(len(d)), 'net_mean_pct': round(float(d.net.mean() * 100), 2),
            'vsEWuncond_mean_pct': round(float(d.x.mean() * 100), 2), 'vsEWuncond_t_month': tstat(g)}

# ---------------- 執行 ----------------
VARS = []
for N in (25, 10):
    for (L, Hh) in ((70, 120), (75, 115)):
        for mode in ('cross', 'immediate'):
            VARS.append((f'N{N}_L{L}_H{Hh}_{mode}', N, L, Hh, mode))
VARS += [('N25_Q5_Q95_cross', 25, 'Q', 'Q', 'cross'), ('N25_Q5_Q95_immediate', 25, 'Q', 'Q', 'immediate')]
MAIN = 'N25_L70_H120_cross'

res = {}; trade_lists = {}
for name, N, L, Hh, mode in VARS:
    ENT, EXT = make_signals(N, L, Hh, mode)
    tr = simulate(ENT, EXT); df = trade_frame(tr); trade_lists[name] = df
    o = {'events': {}, 'portfolio': {}, 'ew_target': {}}
    for w in WIN:
        o['events'][w] = {c: event_stats(df, w, c) for c in COSTS}
        o['ew_target'][w] = ew_target_stats(df, w)
    for c in ('main_rt0.1', 'exlib_etf', 'gross'):
        r, held = nav_series(tr, c)
        o['portfolio'][c] = {w: port_stats(r, held, w) for w in WIN}
    o['trades_per_year'] = {int(k): int(v) for k, v in df.entry.dt.year.value_counts().sort_index().items()} if len(df) else {}
    res[name] = o
    print(name, len(df), {w: (o['events'][w]['main_rt0.1'].get('n_trades'), o['events'][w]['main_rt0.1'].get('abnEW_mean_pct'),
                             o['events'][w]['main_rt0.1'].get('abnEW_t_month'), o['events'][w]['main_rt0.1'].get('vs0050uncond_mean_pct'))
                          for w in ('select_price', 'valid_price')})

# 主規格交易明細
mt = trade_lists[MAIN].copy()
buy, sell = COSTS['main_rt0.1']
mt['net'] = (1 + mt.raw) * (1 - sell) / (1 + buy) - 1; mt['abn'] = mt.net - mt.ew
main_trades = [{k: (str(v.date()) if isinstance(v, pd.Timestamp) else (round(float(v), 4) if isinstance(v, float) else (bool(v) if isinstance(v, (bool, np.bool_)) else int(v) if isinstance(v, (np.integer,)) else v)))
                for k, v in r.items()} for r in mt[['sig', 'entry', 'exit', 'h', 'reason', 'raw', 'net', 'ew', 'abn']].to_dict('records')]

# 診斷：ADR 分布逐年（不參與判定）
a25 = ADR[25]
diag = {'ADR25_pct_by_year': {int(y): {'p5': round(float(g.quantile(.05)), 1), 'p50': round(float(g.median()), 1), 'p95': round(float(g.quantile(.95)), 1),
                                       'share_le70': round(float((g <= 70).mean()), 3), 'share_ge120': round(float((g >= 120).mean()), 3)}
                              for y, g in a25.dropna().groupby(a25.dropna().index.year)},
        'n_T': int(NT), 'n_V': int(len(V)), 'dropped_days': [str(d.date()) for d in PIDX.difference(T)],
        'up_dn_mean_by_year': {int(y): [round(float(UP[g.index].mean()), 0), round(float(DN[g.index].mean()), 0)] for y, g in UP.groupby(UP.index.year)},
        'gaplock_entries_blocked': int(GAPLOCK.sum())}

# 判定
def passes(w):
    s = res[MAIN]['events'][w]['main_rt0.1']
    if s['n_trades'] < 3: return None, s
    ok1 = s['abnEW_mean_pct'] > 0 and s['n_months'] >= 3 and (s['abnEW_t_month'] == s['abnEW_t_month']) and s['abnEW_t_month'] >= 2
    ok2 = s['vs0050uncond_mean_pct'] >= 0
    return bool(ok1 and ok2), {'beat_EW_t2': bool(ok1), 'not_lose_0050': bool(ok2)}
p_sel, d_sel = passes('select_price'); p_val, d_val = passes('valid_price')
if p_sel is None and p_val is None: verdict = '無法測試'
else:
    n = sum(bool(x) for x in (p_sel, p_val))
    verdict = '可用' if n == 2 else ('待觀察' if n == 1 else '不可用')
RJ['results'] = {'variants': res, 'main': MAIN, 'main_trades': main_trades, 'diagnostics': diag,
                 'verdict': {'select_price': d_sel, 'valid_price': d_val, 'verdict': verdict}}
(H / 'results.json').write_text(json.dumps(RJ, ensure_ascii=False, indent=1, default=lambda x: None if x != x else x), encoding='utf-8')
print('VERDICT', verdict, d_sel, d_val)
