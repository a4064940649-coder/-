#!/usr/bin/env python3
"""o7 獨立複製：資料、訊號、權重、引擎（不 import 研究員程式；只讀凍結 raw/ 與 data/shares_est.parquet）。"""
import json, math
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
RAW = BASE / 'raw'
OUT = HERE / 'out'; OUT.mkdir(exist_ok=True)

# ---------------- panel ----------------
def _p(n): return pd.read_parquet(RAW / 'panel' / f'{n}.parquet')
ADJ = _p('adj'); C = _p('close_raw'); O = _p('open'); V = _p('value'); RET = _p('ret'); FLAG = _p('flag')
IDX = ADJ.index; COLS = ADJ.columns; NC = len(COLS); CI = {c: j for j, c in enumerate(COLS)}
POS = {d: i for i, d in enumerate(IDX)}
SH = pd.read_parquet(BASE / 'data' / 'shares_est.parquet').reindex(index=IDX, columns=COLS)
CF = C.ffill()
CAP = SH * CF                                   # 回推市值
TRADED6 = V.notna().rolling(6, min_periods=1).max().fillna(0).astype(bool)
ISDR = np.array([c.startswith('91') for c in COLS])

# ---------------- names / financial / KY ----------------
SL = pd.read_csv(RAW / 'screener' / 'stock_list.csv', dtype=str)
DL = pd.read_csv(RAW / 'agents' / 'delist_official.csv', dtype=str).drop_duplicates('code', keep='last')
NAME = dict(zip(DL.code, DL.name)); NAME.update(dict(zip(SL.code, SL.name)))
IND = dict(zip(SL.code, SL.industry))
def _isfin(c):
    if IND.get(c) == '金融保險業' or c.startswith('28') or c in ('5876', '5880'):
        return True
    if c not in IND:   # 已下市
        nm = NAME.get(c, '')
        return any(k in nm for k in ('銀', '證', '壽', '產保', '保險', '金控', '信託', '票券'))
    return False
ISFIN = np.array([_isfin(c) for c in COLS])
ISKY = np.array([('KY' in NAME.get(c, '')) or NAME.get(c, '').startswith('F-') or c.startswith('91') for c in COLS])

# ---------------- signal dates ----------------
def signal_dates(start='2006-04-01', end='2026-10-31'):
    s = pd.Series(IDX, index=IDX); s = s[(s >= start) & (s <= end)]
    out = []
    for _, g in s.groupby([s.index.year, s.index.month]):
        g = g[g.dt.day >= 11]
        if len(g): out.append(g.iloc[0])
    return pd.DatetimeIndex(out)
DATES = signal_dates()

# ---------------- signals (gapfix) ----------------
GOOD = ADJ.notna().sum(axis=1) >= 300
_ag = ADJ.loc[GOOD]; _vg = V.loc[GOOD]
S52 = (_ag / _ag.rolling(252).max()).reindex(IDX)
V60 = _vg.rolling(60).mean().reindex(IDX)

# 營收
YR = pd.read_parquet(RAW / 'agents' / 'yoy_long_R_FIX.parquet'); YR.index = pd.PeriodIndex(YR.index, freq='M')
Y5 = pd.read_parquet(RAW / 'agents' / 'yoy_rev5y_FIX.parquet'); Y5.index = pd.PeriodIndex(Y5.index, freq='M')
YR = YR.reindex(columns=COLS); Y5 = Y5.reindex(columns=COLS)
_full = pd.period_range('2005-01', '2026-09', freq='M')
YR = YR.reindex(_full); Y5 = Y5.reindex(_full)
PD = json.load(open(RAW / 'agents' / 'p_delay.json'))
INS15 = [c for c in PD['INS15'] if c in CI]
FS = pd.read_csv(RAW / 'agents' / 'firstseen_2026-09.csv', dtype=str, index_col=0)
TX = pd.read_csv(RAW / 'bench' / 'taiex_twii.csv', parse_dates=['date']).set_index('date').sort_index()
TXD = TX.index

def rev_deadline(d):
    day = 15 if (d.year, d.month) in ((2019, 2), (2023, 4)) else 10
    t = pd.Timestamp(d.year, d.month, day)
    return TXD[TXD >= t][0]

def yoy3_at(d):
    """d 當時可用的近 3 月營收年增率平均（%）。"""
    Y = (YR if d < pd.Timestamp('2021-03-01') else Y5).copy()
    m = pd.Period(d, 'M')
    if m - 1 == pd.Period('2026-09', 'M'):   # 2026-09 營收：firstseen
        ok = [c for c in FS.index if c in CI and pd.Timestamp(FS.loc[c, 'first_seen']).tz_convert('Asia/Taipei').tz_localize(None) < d + pd.Timedelta(hours=14)]
        Y.loc[m - 1] = np.nan
        Y.loc[m - 1, ok] = YR.loc[m - 1, ok]
    lag = 2 if rev_deadline(d) >= d else 1
    def y3(l):
        mm = [m - l, m - l - 1, m - l - 2]
        YY = YR if (Y is not YR and min(mm) < Y5.first_valid_index()) else Y   # Y5 起點前（只在 j 平移時發生）退回 R_FIX
        return YY.loc[mm].mean(axis=0, skipna=False)
    out = y3(lag)
    # INS15：2026-01～08 營收一律視為未公告 → 該股往前一月
    if pd.Period('2026-01', 'M') <= m - lag <= pd.Period('2026-08', 'M'):
        out[INS15] = y3(lag + 1)[INS15]
    return out

# 處置
DSP = pd.read_csv(RAW / 'agents' / 'disposal_hist_rejudge.csv', dtype=str)
DSP = DSP[DSP.code.isin(CI)].copy()
for k in ('announce', 'start', 'end'): DSP[k] = pd.to_datetime(DSP[k], errors='coerce')
def disposed(d, e):
    x = DSP[(DSP.announce <= d) & (DSP.start <= e) & (DSP.end >= e)]
    m = np.zeros(NC, bool); m[[CI[c] for c in x.code]] = True; return m

# ---------------- membership ----------------
TW = json.load(open(RAW / 'agents' / 'tw50_members.json'))
_steps = sorted(TW['steps_backward'], key=lambda s: s['eff'])
# 由 current 往回推出每段名單：eff 之後＝下一段的 members_before 或 current
SEG = []   # (eff, members_after)
for k, s in enumerate(_steps):
    after = _steps[k + 1]['members_before'] if k + 1 < len(_steps) else TW['current_2026-10-09']
    SEG.append((pd.Timestamp(s['eff']), set(after)))
FIRST_BEFORE = set(_steps[0]['members_before'])   # 2015-09-18 收盤前

def tick(p):
    return np.select([p < 10, p < 50, p < 100, p < 500, p < 1000], [0.01, 0.05, 0.1, 0.5, 1.0], 5.0)

def rank_universe(i):
    cap = CAP.iloc[i].to_numpy()
    ok = np.isfinite(cap) & (cap > 0) & TRADED6.iloc[i].to_numpy() & ~ISDR
    return cap, ok

def cap_rank(i):
    cap, ok = rank_universe(i)
    r = np.full(NC, np.inf)
    o = np.where(ok)[0]; order = o[np.argsort(-cap[o], kind='stable')]
    r[order] = np.arange(1, len(order) + 1)
    return cap, r

# 2015-09-18 以前：規則重建
def build_pre2015():
    effs = []
    t = pd.Timestamp('2005-06-17')
    while t <= pd.Timestamp('2015-06-19'):
        # 第三個星期五 3/6/9/12
        for mo in (3, 6, 9, 12):
            pass
        break
    # 季度第三個星期五
    for y in range(2005, 2016):
        for mo in (3, 6, 9, 12):
            first = pd.Timestamp(y, mo, 1)
            fri = first + pd.Timedelta(days=(4 - first.weekday()) % 7) + pd.Timedelta(days=14)
            if pd.Timestamp('2005-03-01') <= fri < pd.Timestamp('2015-09-18'):
                effs.append(fri)
    segs = []
    mem = None
    for eff in effs:
        cut = eff - pd.Timedelta(days=25)
        i = IDX.searchsorted(cut, side='right') - 1
        cap, r = cap_rank(i)
        if mem is None:
            mem = set(np.where(r <= 50)[0])
        else:
            mem = {j for j in mem if r[j] <= 60}
            add = [j for j in np.argsort(r) if r[j] <= 40 and j not in mem]
            mem |= set(add)
            if len(mem) < 50:
                for j in np.argsort(r):
                    if len(mem) >= 50 or not np.isfinite(r[j]): break
                    mem.add(j)
            elif len(mem) > 50:
                mem = set(sorted(mem, key=lambda j: r[j])[:50])
        segs.append((eff, {COLS[j] for j in mem}))
    return segs
PRE_SEG = build_pre2015()

def members_for(e):
    """e 日開盤適用的成分：最後一個 eff < e 的名單。"""
    if e > pd.Timestamp('2015-09-18'):
        cur = None
        for eff, m in SEG:
            if eff < e: cur = m
        return cur
    cur = None
    for eff, m in PRE_SEG:
        if eff < e: cur = m
    return cur if cur is not None else PRE_SEG[0][1]

# ---------------- φ ----------------
W50 = pd.read_csv(RAW / 'screener' / 'etf0050_w.csv', dtype={'code': str})
W50 = W50[W50['asof'] == '2026-10-09'].set_index('code')['weight'] / 100
_iphi = IDX.searchsorted(pd.Timestamp('2026-10-09'), side='right') - 1
_capphi = CAP.iloc[_iphi]
_cw = _capphi[W50.index] / _capphi[W50.index].sum()
_ratio = (W50 / W50.sum()) / _cw
PHI_SNAP = _ratio / _ratio['2330']
PHI_MED = float(PHI_SNAP.median())
PHI = np.full(NC, PHI_MED)
for c, v in PHI_SNAP.items():
    if c in CI and np.isfinite(v): PHI[CI[c]] = v

FILL = json.load(open(BASE / 'data' / 'tw50_fill.json'))
MINCAP_CODES = [c for c, v in FILL.items() if v['mode'].startswith('fallback')]

def replica_weights(i_sig, e):
    """R：e 開盤適用成分，以訊號日 d 收盤市值×φ。回傳 (R 向量, 成分 idx, cap 向量)。"""
    mem = members_for(e)
    cap = CAP.iloc[i_sig].to_numpy().copy()
    idx = np.array(sorted(CI[c] for c in mem if c in CI))
    c2 = cap[idx].copy()
    # 2823／2888：同日最小成分市值
    for c in MINCAP_CODES:
        if c in mem and c in CI:
            w0 = pd.Timestamp(FILL[c]['window'][0]); w1 = pd.Timestamp(FILL[c]['window'][1])
            if w0 <= IDX[i_sig] <= w1:
                k = list(idx).index(CI[c])
                others = np.delete(c2, k); others = others[np.isfinite(others)]
                c2[k] = others.min()
    ok = np.isfinite(c2) & (c2 > 0)
    idx = idx[ok]; c2 = c2[ok]
    capm = cap.copy(); capm[idx] = c2
    R = np.zeros(NC); R[idx] = c2 * PHI[idx]; R /= R.sum()
    return R, idx, capm

# ---------------- exfix 日報酬拆解 ----------------
_Cn = C.to_numpy(); _On = O.to_numpy(); _Rn = RET.to_numpy(); _Fn = FLAG.to_numpy()
_Cprev = np.vstack([np.full((1, NC), np.nan), _Cn[:-1]])
_gap_raw = _On / _Cprev - 1
def _exfix():
    R = _Rn.copy()
    GAP = np.full_like(R, np.nan)
    with np.errstate(invalid='ignore', divide='ignore'):
        intr = _Cn / _On - 1
        GAP = (1 + _Rn) / (_Cn / _On) - 1
    for t in range(len(IDX)):
        ex = _Fn[t] == 1
        if not ex.any(): continue
        non = (_Fn[t] == 0) & np.isfinite(_gap_raw[t])
        med = np.nanmedian(_gap_raw[t][non]) if non.any() else 0.0
        R[t, ex] = (1 + intr[t, ex]) * (1 + med) - 1
        GAP[t, ex] = med
    return R, GAP, intr
RETX, GAPX, INTR = _exfix()
# 面板原口徑（敏感度）
with np.errstate(invalid='ignore', divide='ignore'):
    GAPP = (1 + _Rn) / (_Cn / _On) - 1
TRADE_OK = np.isfinite(_On) & np.isfinite(_Cn)
LAST_TRADE = np.array([np.where(TRADE_OK[:, j])[0].max() if TRADE_OK[:, j].any() else -1 for j in range(NC)])

# ---------------- benchmark ----------------
E50 = pd.read_csv(RAW / 'bench' / 'etf_0050.csv', parse_dates=['date']).set_index('date')['adj'].sort_index().reindex(IDX, method='ffill')

FEE = 0.001425 * 0.6
def costs(kind):
    if kind == 'stress': return 0.005, 0.005
    return FEE + 0.001, FEE + 0.003 + 0.001
