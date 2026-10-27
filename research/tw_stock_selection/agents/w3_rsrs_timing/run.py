#!/usr/bin/env python3
"""RSRS 擇時回測主程式。先核對 PREREG 雜湊，結果寫入 results.json['results']。"""
import json, hashlib, math, sys
from pathlib import Path
import numpy as np, pandas as pd
H = Path(__file__).resolve().parent; AG = H.parent
sys.path.insert(0, str(AG / 'exec_rebuild'))
import exlib  # noqa: E402  (RET、C、V、AO、OPEN_RAW、FEE、tstat)
from exlib import tstat, FEE  # noqa: E402
RJ = json.loads((H / 'results.json').read_text())
assert hashlib.sha256(json.dumps(RJ['PREREG'], ensure_ascii=False, sort_keys=True).encode()).hexdigest() == RJ['PREREG_sha256'], 'PREREG changed'
PR = RJ['PREREG']
COSTS = {'main_rt0.1': (0.0005, 0.0005), 'stress_rt0.3': (0.0015, 0.0015),
         'exlib_etf': (FEE + 0.001, FEE + 0.001 + 0.001), 'exlib_stress': (0.005, 0.005), 'gross': (0.0, 0.0)}
WIN = {k: (pd.Timestamp(a), pd.Timestamp(b)) for k, (a, b) in PR['periods'].items()}

# ---------------- TAIEX 訊號 ----------------
tx = pd.read_csv('/home/user/jonatw/finlab-txd/data/raw/taiex_twii.csv', parse_dates=['date']).set_index('date').sort_index()
TXI = tx.index
def beta_r2(n):
    lo, hi = tx['low'], tx['high']
    cov = lo.rolling(n, min_periods=n).cov(hi); vl = lo.rolling(n, min_periods=n).var(); vh = hi.rolling(n, min_periods=n).var()
    return cov / vl, cov ** 2 / (vl * vh)
BR = {n: beta_r2(n) for n in (16, 18, 20)}
def score(kind, n, m):
    b, r2 = BR[n]
    z = (b - b.rolling(m, min_periods=m).mean()) / b.rolling(m, min_periods=m).std(ddof=0)
    return {'z': z, 'zR2': z * r2, 'zR2b': z * r2 * b}[kind]
MA20 = tx['close'].rolling(20).mean(); MAUP = (MA20 > MA20.shift(3)).to_numpy()
MA200 = tx['close'].rolling(200).mean()

def state(sc, thr, filt):
    s = sc.to_numpy(); tgt = np.zeros(len(s), bool); cur = False
    for i in range(len(s)):
        if np.isfinite(s[i]):
            if s[i] > thr and (not filt or MAUP[i]): cur = True
            elif s[i] < -thr: cur = False
        tgt[i] = cur
    return tgt

# ---------------- 0050 ----------------
e50 = pd.read_csv('/home/user/jonatw/finlab-txd/data/raw/etf_0050.csv', parse_dates=['date']).set_index('date')['adj'].sort_index()
e50 = e50.drop(pd.Timestamp('2026-09-20'), errors='ignore')
D = e50.index; ND = len(D); C50 = e50.to_numpy()
txc_asof = tx['close'].reindex(D, method='ffill').to_numpy()
txo = tx['open'].reindex(D).to_numpy()
O50 = np.full(ND, np.nan); O50[1:] = C50[:-1] * txo[1:] / txc_asof[:-1]
TRAD = np.isfinite(O50)
GAP = np.full(ND, np.nan); GAP[1:] = O50[1:] / C50[:-1] - 1
UPLOCK = np.nan_to_num(GAP) >= 0.094; DNLOCK = np.nan_to_num(GAP) <= -0.094
# d 之前最後一個 TAIEX 收盤日的位置
PREV_TX = np.searchsorted(TXI.values, D.values, side='left') - 1
R50D = pd.Series(C50, index=D).pct_change()

def map_target(tgt):
    out = np.zeros(ND, bool); ok = PREV_TX >= 0; out[ok] = tgt[PREV_TX[ok]]; return out

def simulate(tgt_d, cost, close_exec=False):
    """tgt_d[k]：0050 日 k 開盤時應有部位（= k 之前最後一個 TAIEX 收盤的目標）。回傳日報酬、收盤持有、交易表。"""
    buy, sell = COSTS[cost]; f = np.ones(ND); held = np.zeros(ND, bool); pos = False; trades = []; ent = None
    for k in range(1, ND):
        want = tgt_d[k]
        if close_exec:
            fk = C50[k] / C50[k - 1] if pos else 1.0
            if want != pos:
                if want: fk = 1.0 / (1 + buy); pos = True; ent = k
                else: fk *= (1 - sell); pos = False; trades.append((ent, k, False))
            f[k] = fk; held[k] = pos; continue
        if want != pos and TRAD[k] and not (want and UPLOCK[k]) and not ((not want) and DNLOCK[k]):
            if want:
                f[k] = C50[k] / O50[k] / (1 + buy); pos = True; ent = k
            else:
                f[k] = O50[k] / C50[k - 1] * (1 - sell); pos = False; trades.append((ent, k, False))
        else:
            f[k] = C50[k] / C50[k - 1] if pos else 1.0
        held[k] = pos
    if pos: trades.append((ent, ND - 1, True))
    return pd.Series(f - 1, index=D), pd.Series(held, index=D), trades

# ---------------- 同池等權（面板） ----------------
RET = exlib.RET; PIDX = exlib.IDX
cnt = RET.notna().sum(axis=1); T = PIDX[(cnt >= 300).to_numpy()]; NT = len(T); TP = PIDX.get_indexer(T)
Cr = exlib.C.loc[T]; v60 = exlib.V.loc[T].rolling(60, min_periods=60).mean()
POOL = ((v60 >= 1e7) & (Cr >= 10)).to_numpy(); RT = RET.loc[T].to_numpy()
AOp = exlib.AO[TP]; OPN = np.isfinite(exlib.OPEN_RAW[TP]); ACp = exlib.AC[TP]
ew = np.full(NT, np.nan)
for i in range(1, NT):
    m = POOL[i - 1] & np.isfinite(RT[i])
    if m.sum(): ew[i] = RT[i][m].mean()
EWD = pd.Series(ew, index=T)

def mkey(idx):
    p = idx.to_period('M'); return p.where(p != pd.Period('2024-01', 'M'), pd.Period('2024-02', 'M'))
def monthly(r):
    r = r.dropna(); return (1 + r).groupby(mkey(r.index)).prod() - 1

def win_mask(idx, w):
    a, b = WIN[w]; return (idx >= a) & (idx <= b)

def port_stats(r, held, w, ntr_dates):
    m = win_mask(r.index, w); rr = r[m]
    if len(rr) < 20: return None
    i0 = np.argmax(m); start = r.index[i0 - 1] if i0 > 0 else rr.index[0]
    yrs = (rr.index[-1] - start).days / 365.25
    def cagr(x): return round(float((1 + x.fillna(0)).prod() ** (1 / yrs) - 1), 4)
    def mdd(x): q = (1 + x.fillna(0)).cumprod(); return round(float((q / q.cummax() - 1).min()), 4)
    b50 = R50D[m]; mew = win_mask(EWD.index, w); bew = EWD[mew]
    ms, m50, me = monthly(rr), monthly(b50), monthly(bew)
    x50 = (ms - m50).dropna(); xew = (ms - me).dropna()
    a, b = WIN[w]
    return {'CAGR': cagr(rr), 'maxDD': mdd(rr), 'vol': round(float(rr.std() * math.sqrt(252)), 4), 'exposure': round(float(held[m].mean()), 3),
            'n_entries': int(sum((d >= a) & (d <= b) for d in ntr_dates)),
            '0050_CAGR': cagr(b50), '0050_maxDD': mdd(b50), 'EW_CAGR': round(float((1 + bew.fillna(0)).prod() ** (1 / yrs) - 1), 4), 'EW_maxDD': mdd(bew),
            'n_months': int(len(xew)),
            'ex_vs_0050_pct_m': round(float(x50.mean() * 100), 3), 't_vs_0050_m': tstat(x50),
            'ex_vs_EW_pct_m': round(float(xew.mean() * 100), 3), 't_vs_EW_m': tstat(xew),
            't_vs_0050_daily': tstat(rr - b50),
            'yearly_pct': {int(y): round(float((1 + g).prod() - 1) * 100, 1) for y, g in rr.groupby(rr.index.year)}}

def passes(ps):
    if ps is None or ps['n_entries'] == 0 and ps['exposure'] == 0 or ps['n_months'] < 12: return None
    return bool(ps['ex_vs_EW_pct_m'] > 0 and ps['t_vs_EW_m'] >= 2 and ps['CAGR'] >= ps['0050_CAGR'])

# ---------------- 事件 ----------------
TPOS = np.searchsorted(T.values, D.values, side='left')   # 0050 日 → 第一個 ≥ 該日的面板日
_b50 = {}
def base50(w, h):
    k = (w, h)
    if k not in _b50:
        d = np.where(TRAD & win_mask(D, w))[0]; d = d[d + h < ND]
        _b50[k] = float(np.nanmean(np.where(TRAD[d + h], O50[d + h], np.nan) / O50[d] - 1)) if len(d) else np.nan
    return _b50[k]
def ew_bh(e, x, trunc):
    pe, px = TPOS[e], TPOS[x]
    if pe >= NT: return np.nan
    px = min(px, NT - 1); s = max(pe - 1, 0)
    m = OPN[pe] & POOL[s]
    end = ACp[NT - 1] if trunc else AOp[px]
    r = end[m] / AOp[pe][m] - 1; r = r[np.isfinite(r)]
    return float(r.mean()) if len(r) else np.nan
def event_stats(trades, w, cost, close_exec=False):
    buy, sell = COSTS[cost]; rows = []
    for e, x, tr in trades:
        if not win_mask(pd.DatetimeIndex([D[e]]), w)[0]: continue
        pe = C50[e] if close_exec else O50[e]; px = C50[x] if (tr or close_exec) else O50[x]
        raw = px / pe - 1; net = (1 + raw) * (1 - (0 if tr else sell)) / (1 + buy) - 1
        rows.append(dict(entry=D[e], h=x - e, net=net, ew=ew_bh(e, x, tr), b50=base50(w, x - e), trunc=tr))
    if not rows: return {'n_trades': 0}
    d = pd.DataFrame(rows); d['abn'] = d.net - d.ew; d['x50'] = d.net - d.b50
    g = d.groupby(d.entry.dt.to_period('M'))[['net', 'abn', 'x50']].mean()
    return {'n_trades': int(len(d)), 'n_months': int(len(g)), 'avg_hold_days': round(float(d.h.mean()), 1),
            'net_mean_pct': round(float(d.net.mean() * 100), 2), 'net_median_pct': round(float(d.net.median() * 100), 2), 'net_hit': round(float((d.net > 0).mean()), 3),
            'abnEW_mean_pct': round(float(d.abn.mean() * 100), 2), 'abnEW_t_month': tstat(g.abn),
            'vs0050uncond_mean_pct': round(float(d.x50.mean() * 100), 2), 'vs0050uncond_t_month': tstat(g.x50)}

# ---------------- 執行 ----------------
VARS = [(k, n, m, t, f) for k in ('z', 'zR2', 'zR2b') for n in (16, 18, 20) for m in (300, 600) for t in (0.7, 1.0) for f in (False, True)]
MAIN = 'zR2_N18_M600_T0.7_none'
def vname(k, n, m, t, f): return f'{k}_N{n}_M{m}_T{t}_{"ma20up" if f else "none"}'
res = {}; MAIN_OBJ = {}
for v in VARS:
    name = vname(*v); k, n, m, t, f = v
    tgt = state(score(k, n, m), t, f); tgt_d = map_target(tgt)
    o = {'portfolio': {}, 'events': {}}
    for c in COSTS:
        r, held, trades = simulate(tgt_d, c)
        ed = [D[e] for e, _, _ in trades]
        o['portfolio'][c] = {w: port_stats(r, held, w, ed) for w in WIN}
        if c in ('main_rt0.1', 'exlib_etf'):
            o['events'][c] = {w: event_stats(trades, w, c) for w in WIN}
        if c == 'main_rt0.1' and name == MAIN:
            MAIN_OBJ = dict(r=r, held=held, trades=trades, tgt=tgt, tgt_d=tgt_d)
    o['verdict_windows'] = {w: passes(o['portfolio']['main_rt0.1'][w]) for w in ('select_price', 'valid_price')}
    o['verdict_windows_exlib_etf'] = {w: passes(o['portfolio']['exlib_etf'][w]) for w in ('select_price', 'valid_price')}
    if name == MAIN or v[1:] == (18, 600, 0.7, False):
        r, held, trades = simulate(tgt_d, 'main_rt0.1', close_exec=True); ed = [D[e] for e, _, _ in trades]
        o['sens_close_exec'] = {w: port_stats(r, held, w, ed) for w in WIN}
    res[name] = o
    ps = o['portfolio']['main_rt0.1']
    print(name, {w: (ps[w]['CAGR'], ps[w]['0050_CAGR'], ps[w]['t_vs_EW_m'], ps[w]['exposure'], ps[w]['n_entries']) for w in ('select_price', 'valid_price')}, flush=True)

# ---------------- 對照：MA200、曝險相同的隨機擇時 ----------------
tg = np.zeros(len(TXI), bool); cl = tx['close'].to_numpy(); ma = MA200.to_numpy()
for i in range(len(TXI)): tg[i] = bool(np.isfinite(ma[i]) and cl[i] > ma[i])
base = {}
for c in ('main_rt0.1', 'exlib_etf'):
    r, held, trades = simulate(map_target(tg), c); ed = [D[e] for e, _, _ in trades]
    base[c] = {w: port_stats(r, held, w, ed) for w in WIN}
rng = np.random.default_rng(20261027)
rand = {}
for w in ('select_price', 'valid_price', 'y2024_2025', 'full'):
    m = win_mask(D, w); idx = np.where(m)[0]; tgw = MAIN_OBJ['tgt_d'][idx]
    act = res[MAIN]['portfolio']['main_rt0.1'][w]['CAGR']
    cg = []
    for _ in range(1000):
        sh = np.roll(tgw, rng.integers(1, len(idx)))
        tt = MAIN_OBJ['tgt_d'].copy(); tt[idx] = sh
        r, held, _ = simulate(tt, 'main_rt0.1'); rr = r[m]
        yrs = (rr.index[-1] - D[idx[0] - 1]).days / 365.25
        cg.append(float((1 + rr).prod() ** (1 / yrs) - 1))
    cg = np.array(cg)
    rand[w] = {'actual_CAGR': act, 'rand_mean_CAGR': round(float(cg.mean()), 4), 'rand_p05': round(float(np.quantile(cg, .05)), 4),
               'rand_p95': round(float(np.quantile(cg, .95)), 4), 'pctile_of_actual': round(float((cg < act).mean()), 3)}
    print('rand', w, rand[w], flush=True)

# ---------------- 主規格明細 ----------------
mt = MAIN_OBJ['trades']
trade_list = [{'entry': str(D[e].date()), 'exit': str(D[x].date()), 'hold_days': int(x - e), 'trunc': bool(tr),
               'raw_pct': round(float(((C50[x] if tr else O50[x]) / O50[e] - 1) * 100), 2)} for e, x, tr in mt]
sc = score('zR2', 18, 600)
diag = {'score_first_valid': str(sc.first_valid_index().date()),
        'main_score_quantiles': {str(q): round(float(sc.quantile(q)), 3) for q in (.05, .25, .5, .75, .95)},
        'share_days_above_0.7': round(float((sc > 0.7).mean()), 3), 'share_days_below_-0.7': round(float((sc < -0.7).mean()), 3),
        'n_uplock_days': int(UPLOCK.sum()), 'n_dnlock_days': int(DNLOCK.sum()),
        'gap_abs_median_pct': round(float(np.nanmedian(np.abs(GAP)) * 100), 3),
        'main_exposure_by_year': {int(y): round(float(g.mean()), 3) for y, g in MAIN_OBJ['held'].groupby(D.year)},
        'main_entries_by_year': {int(y): int(v) for y, v in pd.Series([D[e].year for e, _, _ in mt]).value_counts().sort_index().items()}}
verdict_main = res[MAIN]['verdict_windows']
npass = sum(bool(v) for v in verdict_main.values())
verdict = '無法測試' if all(v is None for v in verdict_main.values()) else ('可用' if npass == 2 else '待觀察' if npass == 1 else '不可用')
RJ['results'] = {'main': MAIN, 'verdict': verdict, 'verdict_windows': verdict_main,
                 'verdict_exlib_etf': res[MAIN]['verdict_windows_exlib_etf'],
                 'variants': res, 'baseline_MA200': base, 'random_timing_main': rand, 'main_trades': trade_list, 'diag': diag}
(H / 'results.json').write_text(json.dumps(RJ, ensure_ascii=False, indent=1, default=str), encoding='utf-8')
print('VERDICT', verdict, verdict_main, res[MAIN]['verdict_windows_exlib_etf'])
