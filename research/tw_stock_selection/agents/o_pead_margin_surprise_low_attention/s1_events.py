#!/usr/bin/env python3
"""第 1 步：事件表（定義在探索前寫死，見 PREDEF 字典；本程式第一次執行時把 PREDEF 雜湊存進 data/predef.json）。
SUE   ＝(EPS_q − EPS_{q−4}) ÷ std(前 8 季的 (EPS − EPS_{−4}))，至少 6 季；std 用 ddof=1。
REVSUE＝同法，季營收（MOPS 單季營收；季營收在季報前已由月營收公布）。
SUE⊥ ＝PIT：SUE、REVSUE 在「前 4 個完整財報季（至少 2 季）的事件分布」中的百分位；
        以同一批前期事件擬合 pct(SUE) = a + b·pct(REVSUE)，SUE⊥ = pct(SUE) − (a + b·pct(REVSUE))。
        門檻（前 1/5）也用前期 SUE⊥ 分布的第 80 百分位 → 完全不用同季較晚公告者的資訊。
股池（以 A 日收盤、PIT）：上市＋上櫃普通股（4 碼、非 0 開頭、非 91xx）、排除金融保險（MOPS 財報格式）、
        60 日均成交值 ≥ 1,000 萬、收盤 ≥ 10。
關注度：A 日股池內 60 日均成交值三分位（1＝最低）；0050 成分（PIT，o_0050_tilt_pit membership_pit.json）。
低關注＝三分位 1 且不在 0050。
進場：j0＝A 的下一個交易日開盤；j＝0..9。開盤 ≥ 漲停價 → 當日買不到，順延至多 2 日，仍不行則該 j 放棄。
        處置期間（2021-09 起才有官方資料）進場日在處置中 → 該 j 放棄。
出場：進場後第 H 個交易日開盤（H＝20／40／60／120），遇停牌用下一個有開盤的日子；下市用最後收盤。
報酬：含息（官方漲跌價差推算），開盤到開盤。"""
import json, sys, hashlib
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
from pcore import DATA, AG, FIN_IND, deadline, load_panel

PREDEF = {'sue_min_q': 6, 'sue_win': 8, 'pool_v60': 1e7, 'pool_close': 10, 'pit_seasons': 4, 'pit_min_seasons': 2,
          'j_max': 9, 'H': [20, 40, 60, 120], 'limit_delay_max': 2, 'att_terciles': 3}
pd_path = DATA / 'predef.json'
h = hashlib.sha256(json.dumps(PREDEF, sort_keys=True).encode()).hexdigest()
if not pd_path.exists():
    json.dump({'PREDEF': PREDEF, 'sha256': h, 'utc': pd.Timestamp.utcnow().isoformat()}, open(pd_path, 'w'), indent=1)
assert json.load(open(pd_path))['sha256'] == h, 'PREDEF 被改過'

P = load_panel()
IDX = pd.DatetimeIndex(P['IDX']); COLS = list(P['COLS']); CI = {c: j for j, c in enumerate(COLS)}
T, N = P['C'].shape
O, C, R, V60, SIG, LIM, SH, MKT = P['O'], P['C'], P['R'], P['V60'], P['SIG'], P['LIMUP'], P['SH'], P['MKT']
G = np.where(np.isfinite(R), 1 + R, 1.0)
CUM = np.cumprod(G, axis=0)                     # 含息累積指數（收盤）
CF = pd.DataFrame(C).ffill().to_numpy()

# ---------- 財報 ----------
fin = pd.read_parquet(DATA / 'fin_long.parquet')
fin = fin[fin['code'].str.fullmatch(r'\d{4}') & ~fin['code'].str.startswith('0') & ~fin['code'].str.startswith('91')]
fin['k'] = fin['y'] * 4 + fin['q'] - 1
ind = fin.sort_values('k').groupby('code')['industry'].last()


def sue_of(col):
    W = fin.pivot(index='code', columns='k', values=col)
    ks = np.arange(W.columns.min(), W.columns.max() + 1); W = W.reindex(columns=ks)
    Dd = W - W.shift(4, axis=1)
    sd = Dd.T.shift(1).rolling(PREDEF['sue_win'], min_periods=PREDEF['sue_min_q']).std().T
    S = Dd / sd.where(sd > 0)
    return W, Dd, sd, S


EPS, dE, sE, SUE = sue_of('eps_s')
REV, dR, sR, RSUE = sue_of('rev_s')
GP = fin.pivot(index='code', columns='k', values='gp_s').reindex(index=REV.index, columns=REV.columns)
OPI = fin.pivot(index='code', columns='k', values='opi_s').reindex(index=REV.index, columns=REV.columns)
GM = GP / REV.where(REV > 0); OM = OPI / REV.where(REV > 0)
dGM = GM - GM.shift(4, axis=1); dOM = OM - OM.shift(4, axis=1)

AV = pd.read_parquet(DATA / 'avail_dates.parquet')
AV['k'] = AV['y'] * 4 + AV['q'] - 1

# ---------- 月營收（PIT：m 月營收在 m+1 月 10 日後可得） ----------
rv = pd.read_parquet(DATA / 'rev_long.parquet')
rv['avail'] = (pd.PeriodIndex(rv['ym'], freq='M') + 1).to_timestamp() + pd.Timedelta(days=10)
YOY = rv.pivot(index='ym', columns='code', values='yoy_pct').sort_index()
YOY.index = pd.PeriodIndex(YOY.index, freq='M')
YOY3 = YOY.rolling(3, min_periods=3).mean()

# ---------- 0050 PIT 成分 ----------
mem = json.load(open(AG / 'o_0050_tilt_pit' / 'data' / 'membership_pit.json'))
mdates = sorted(pd.Timestamp(x) for x in mem)


def in0050(code, d):
    i = np.searchsorted(np.array(mdates, dtype='datetime64[ns]'), np.datetime64(d), side='right') - 1
    if i < 0:
        return False
    return code in mem[mdates[i].strftime('%Y-%m-%d')]


# ---------- 處置（2021-09 起官方） ----------
dsp = pd.read_csv('/home/user/stevenhsiehsheng-ctrl/tw-stock-screener/data/extras/disposal_5y.csv.gz', dtype={'code': str},
                  parse_dates=['start', 'end'])
DISP = np.zeros((T, N), bool)
for _, r in dsp.iterrows():
    if r['code'] in CI and pd.notna(r['start']) and pd.notna(r['end']):
        DISP[(IDX >= r['start']) & (IDX <= r['end']), CI[r['code']]] = True

# ---------- 52 週比、股池、關注度 ----------
HI = pd.DataFrame(CUM).rolling(252, min_periods=240).max().to_numpy()
R52 = CUM / HI
POOL = (V60 >= PREDEF['pool_v60']) & (CF >= PREDEF['pool_close'])
finmask = np.array([ind.get(c, 'ci') in FIN_IND for c in COLS])
ordmask = np.array([c.isdigit() and len(c) == 4 and not c.startswith('0') and not c.startswith('91') for c in COLS])
POOL &= (~finmask & ordmask)[None, :]

# A 日股池內 V60 三分位、52 週比百分位、營收 3 月年增百分位
def xs_pct(row, mask):
    out = np.full(N, np.nan)
    w = np.where(mask & np.isfinite(row))[0]
    if len(w) > 1:
        rk = pd.Series(row[w]).rank(pct=True).to_numpy(); out[w] = rk
    return out


# ---------- 事件 ----------
ev = AV[(AV['k'] >= 2015 * 4) & AV['code'].isin(CI)].copy()
ev['SUE'] = [SUE.at[c, k] if (c in SUE.index and k in SUE.columns) else np.nan for c, k in zip(ev['code'], ev['k'])]
ev = ev[np.isfinite(ev['SUE'])].copy()
gets = lambda M: [M.at[c, k] if (c in M.index and k in M.columns) else np.nan for c, k in zip(ev['code'], ev['k'])]
ev['REVSUE'] = gets(RSUE); ev['dEPS'] = gets(dE); ev['sEPS'] = gets(sE); ev['EPS'] = gets(EPS)
ev['dGM'] = gets(dGM); ev['dOM'] = gets(dOM)
ev['ai'] = IDX.searchsorted(pd.DatetimeIndex(ev['A']))
ev = ev[ev['ai'] < T - 1].copy()
ev['j'] = ev['code'].map(CI)
ai = ev['ai'].to_numpy(); jj = ev['j'].to_numpy()
ev['pool'] = POOL[ai, jj]
ev['V60'] = V60[ai, jj]; ev['close'] = CF[ai, jj]; ev['sig60'] = SIG[ai, jj]
ev['mcap'] = CF[ai, jj] * SH[ai, jj]
ev['fin'] = finmask[jj]
ev['mkt'] = np.where(MKT[jj] == 1, 'twse', 'tpex')
# 橫斷面量（A 日）
att = np.full(len(ev), np.nan); p52 = np.full(len(ev), np.nan); prv = np.full(len(ev), np.nan)
cache = {}
for n_, (a, j) in enumerate(zip(ai, jj)):
    if a not in cache:
        pm = POOL[a]
        v = V60[a]; w = np.where(pm & np.isfinite(v))[0]
        tr = np.full(N, np.nan)
        if len(w) > 2:
            rk = pd.Series(v[w]).rank(pct=True).to_numpy()
            tr[w] = np.ceil(rk * 3).clip(1, 3)
        d = IDX[a]
        m_av = pd.Period(d - pd.Timedelta(days=10), 'M') - 1  # 已公布的最新月份
        y3 = YOY3.reindex([m_av]).reindex(columns=COLS).to_numpy()[0] if m_av in YOY3.index else np.full(N, np.nan)
        cache[a] = (tr, xs_pct(R52[a], pm), xs_pct(y3, pm))
    tr, a52, ar = cache[a]
    att[n_] = tr[j]; p52[n_] = a52[j]; prv[n_] = ar[j]
ev['att'] = att; ev['p52'] = p52; ev['prev3'] = prv
ev['in0050'] = [in0050(c, d) for c, d in zip(ev['code'], ev['A'])]
ev['lowatt'] = (ev['att'] == 1) & ~ev['in0050']
# 注意力分散：同日公告家數（全部事件，不限股池）、期限前最後 2 個交易日
cnt = ev.groupby('A')['code'].transform('count')
ev['n_same_day'] = cnt
dl_i = IDX.searchsorted(pd.DatetimeIndex(ev['deadline']), side='right') - 1
ev['last2'] = (ev['ai'] >= dl_i - 1) & (ev['ai'] <= dl_i)

# ---------- PIT 百分位與 SUE⊥ ----------
E = ev[ev['pool']].copy()
E = E.sort_values(['k', 'A']).reset_index(drop=True)
E['SUE_pct'] = np.nan; E['REV_pct'] = np.nan; E['SUEP'] = np.nan; E['SUEP_thr80'] = np.nan; E['SUEP_thr20'] = np.nan
E['SUE_pct_c'] = E.groupby('k')['SUE'].rank(pct=True)              # 同季（含較晚公告者，只作診斷）
E['REV_pct_c'] = E.groupby('k')['REVSUE'].rank(pct=True)
for k in sorted(E['k'].unique()):
    base = E[(E['k'] < k) & (E['k'] >= k - PREDEF['pit_seasons'])]
    if base['k'].nunique() < PREDEF['pit_min_seasons']:
        continue
    bs = np.sort(base['SUE'].dropna().to_numpy())
    b2 = base.dropna(subset=['REVSUE'])
    br = np.sort(b2['REVSUE'].to_numpy())
    ecdf = lambda srt, x: np.searchsorted(srt, x, side='right') / len(srt)
    # 前期事件自己的百分位（在前期分布中）→ 擬合係數
    xs = ecdf(br, b2['REVSUE'].to_numpy()); ys = ecdf(bs, b2['SUE'].to_numpy())
    b, a = np.polyfit(xs, ys, 1)
    res_base = ys - (a + b * xs)
    cur = E['k'] == k
    sp = ecdf(bs, E.loc[cur, 'SUE'].to_numpy())
    rp = np.where(E.loc[cur, 'REVSUE'].notna(), ecdf(br, E.loc[cur, 'REVSUE'].fillna(0).to_numpy()), np.nan)
    E.loc[cur, 'SUE_pct'] = sp; E.loc[cur, 'REV_pct'] = rp
    E.loc[cur, 'SUEP'] = sp - (a + b * rp)
    E.loc[cur, 'SUEP_thr80'] = np.quantile(res_base, 0.8); E.loc[cur, 'SUEP_thr20'] = np.quantile(res_base, 0.2)
    E.loc[cur, 'SUE_thr80'] = np.quantile(bs, 0.8); E.loc[cur, 'SUE_thr20'] = np.quantile(bs, 0.2)
    E.loc[cur, 'fit_b'] = b
E['top_SUEP'] = E['SUEP'] >= E['SUEP_thr80']; E['bot_SUEP'] = E['SUEP'] <= E['SUEP_thr20']
E['top_SUE'] = E['SUE'] >= E['SUE_thr80']; E['bot_SUE'] = E['SUE'] <= E['SUE_thr20']

# ---------- 事件報酬（j=0..9 × H），開盤到開盤，含息 ----------
OK_O = np.isfinite(O) & (O > 0)


def entry_day(a, j, jd):
    """回傳實際進場列 e（或 -1）：A 之後第 1+jd 個交易日起，漲停順延至多 2 日；處置中放棄。"""
    e = a + 1 + jd
    for _ in range(PREDEF['limit_delay_max'] + 1):
        if e >= T:
            return -1
        if DISP[e, j]:
            return -1
        if OK_O[e, j] and not (O[e, j] >= LIM[e, j] - 1e-9):
            return e
        if not OK_O[e, j] and not np.isfinite(C[e, j]):
            e += 1; continue
        e += 1
    return -1


def oo_ret(e, x, j):
    """e 日開盤買、x 日開盤賣（x 日沒開盤就往後找；找不到用最後收盤）。"""
    buy = O[e, j]
    # 到 x−1 收盤的含息價值
    v = C[e, j] / buy if np.isfinite(C[e, j]) else 1.0
    if x - 1 > e:
        v *= CUM[x - 1, j] / CUM[e, j]
    xx = x
    while xx < T and not OK_O[xx, j]:
        if np.isfinite(C[xx, j]):
            v *= G[xx, j]
        xx += 1
    if xx >= T:
        return v - 1, T - 1
    gx = O[xx, j] * G[xx, j] / C[xx, j] if np.isfinite(C[xx, j]) and C[xx, j] > 0 else O[xx, j] / CF[xx - 1, j]
    return v * gx - 1, xx


Hs = PREDEF['H']
ent = np.full((len(E), PREDEF['j_max'] + 1), -1, int)
RET = {H: np.full((len(E), PREDEF['j_max'] + 1), np.nan) for H in Hs}
for n_, (a, j) in enumerate(zip(E['ai'].to_numpy(), E['j'].to_numpy())):
    for jd in range(PREDEF['j_max'] + 1):
        e = entry_day(a, j, jd)
        ent[n_, jd] = e
        if e < 0:
            continue
        for H in Hs:
            x = e + H
            if x >= T:
                continue
            RET[H][n_, jd] = oo_ret(e, x, j)[0]
for H in Hs:
    for jd in range(PREDEF['j_max'] + 1):
        E[f'r{H}_j{jd}'] = RET[H][:, jd]
for jd in range(PREDEF['j_max'] + 1):
    E[f'e_j{jd}'] = ent[:, jd]
# 事件時間路徑（j0，0～120 日，累積含息報酬，起點＝j0 開盤）
PATH = np.full((len(E), 121), np.nan)
for n_, (e, j) in enumerate(zip(ent[:, 0], E['j'].to_numpy())):
    if e < 0:
        continue
    base = O[e, j]
    for h in range(0, 121):
        t = e + h
        if t >= T:
            break
        PATH[n_, h] = CUM[t, j] / CUM[e, j] * C[e, j] / base - 1 if np.isfinite(C[e, j]) else np.nan
# 同池等權基準路徑（A 日股池全部股票，j0 開盤起；用收盤累積近似）
EWP = np.full((len(E), 121), np.nan)
ewc = {}
for n_, (a, e) in enumerate(zip(E['ai'].to_numpy(), ent[:, 0])):
    if e < 0:
        continue
    if e not in ewc:
        w = np.where(POOL[a])[0]
        seg = CUM[e - 1:min(e + 120, T), w] / CUM[e - 1, w]
        ewc[e] = np.nanmean(seg, axis=1) - 1
    s = ewc[e]; EWP[n_, :len(s)] = s
E.to_parquet(DATA / 'events.parquet', index=False)
np.savez_compressed(DATA / 'event_paths.npz', PATH=PATH, EWP=EWP)
ev.to_parquet(DATA / 'events_all.parquet', index=False)   # 含非股池（覆蓋率用）
print('events (pool)', len(E), 'with SUEP', int(E['SUEP'].notna().sum()), 'seasons', E['k'].min() // 4, E['k'].max() // 4)
print(E.groupby('k').agg(n=('code', 'size'), suep=('SUEP', lambda s: s.notna().sum()), low=('lowatt', 'sum'),
                         tpex=('mkt', lambda s: (s == 'tpex').sum())).to_string())
