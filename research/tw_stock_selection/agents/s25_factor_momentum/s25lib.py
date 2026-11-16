#!/usr/bin/env python3
"""S25 共用程式庫：因子動能與「扣掉同業」的選股報酬。
在 s20lib（不改；其下 s17lib→s10lib：面板、創新板、SURGE（STD，σ>0）、F1、E0、引擎 simulate_s10）之上，本檔只新增：
- load_factors()／factor_trend()：讀 s24 monthly.csv（不重建），因子過去 12 個月報酬和（趨勢）。
- trend_features()：每個訊號列 k，取訊號日所在月份的前一個月月底已知的因子趨勢（MOM、RMOM 等）。
- apply_regime()：照 s23lib.apply_regime 的同一套做法（逐字同邏輯），只改當月買進名單 sig：'stop'＝不新買、'half'＝只買 SURGE 排序前 ceil(n/2) 檔；
  沒買的錢照引擎掃進 0050。只在換股日判斷。
- industry_matrix()：PIT 產業（同 s21lib.industry_matrix，只讀 industry_pit.csv）。
- mcap_panel()：PIT 市值（同 s24 build.py）。
- hold_ret()：持有期含息毛報酬（同 s20lib.outcomes 的公式），供個股、同業、0050 同口徑。
中間檔放 scratchpad/s25/。一律 python3 -I -B。"""
import sys, math
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
AG = HERE.parent
S25 = Path('/tmp/claude-0/-home-user--/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/s25')
sys.path.insert(0, str(AG / 's20_persistent_growth'))
import s20lib                                              # noqa: E402
from s20lib import s17lib, s10lib, s9lib, s8lib, s7lib, s6lib, s5lib, s4lib, s3lib, s2lib, stlib, cslib, COSTS   # noqa: E402,F401
s20lib.S20 = S25; s17lib.S17 = S25; s10lib.S10 = S25; s9lib.S9 = S25; s8lib.S8 = S25; s7lib.S7 = S25; s6lib.S6 = S25; s5lib.S5 = S25; s4lib.S4 = S25
s3lib.S3 = S25; s2lib.S2 = S25; stlib.S1 = S25; cslib.TMP = S25

TRAIN_LAST_MONTH = s20lib.TRAIN_LAST_MONTH                 # 營收月份 ≤ 2021-11（訊號日 2016-02～2021-12）
VALID_FIRST_MONTH = s20lib.VALID_FIRST_MONTH               # 營收月份 ≥ 2021-12（訊號日 2022-01 起）
SEGMENTS = s17lib.SEGMENTS
fresh_from = s20lib.fresh_from
MONTHLY = AG / 's24_return_attribution' / 'monthly.csv'
IND_CSV = AG / 'o_lc_sector_rotation_pit' / 'raw' / 'industry_pit.csv'
MISSING = {'未分類', '', 'nan'}
FACTORS = ['SMB', 'HML', 'EPF', 'MOM', 'RMOM']
TREND_N = 12

# 事前登記的規則族（≤ 4 條；門檻固定 0＝講義定義，不調參）
RULES = [
    dict(id='RMOM_HALF', var='RMOM', action='half',
         desc='營收動能因子 RMOM 過去 12 個月報酬和 ≤ 0（訊號日前一個月月底已知）→ 當月衛星新買減半：F1 只買 SURGE 排序前 ceil(n/2) 檔，其餘空位放 0050'),
    dict(id='RMOM_STOP', var='RMOM', action='stop',
         desc='RMOM 過去 12 個月報酬和 ≤ 0 → 當月不新買，空位放 0050；既有持股照 E0'),
    dict(id='MOM_HALF', var='MOM', action='half',
         desc='個股動能因子 MOM 過去 12 個月報酬和 ≤ 0 → 當月衛星新買減半'),
    dict(id='MOM_STOP', var='MOM', action='stop',
         desc='MOM 過去 12 個月報酬和 ≤ 0 → 當月不新買'),
]


def load():
    S25.mkdir(parents=True, exist_ok=True)
    return s20lib.load()


def surge(P):
    F = s17lib.rev_family(P, defs=['STD'])
    return dict(S=F['S']['STD'], av=F['av'])


# ---------------------------------------------------------------- 因子與趨勢
def load_factors():
    d = pd.read_csv(MONTHLY)
    d.index = pd.PeriodIndex(d['month'], freq='M')
    return d


def factor_trend(f, n=TREND_N):
    """f：月報酬 Series（PeriodIndex）。回傳 Series：月 t 月底已知的趨勢＝t−n+1～t 共 n 個月報酬和（n 個月都要有值）。"""
    return f.rolling(n, min_periods=n).sum()


def trend_features(S0, FD, n=TREND_N):
    """每個訊號列一列：month（營收月）、k、sig_date、ref_month（訊號日所在月份的前一個月）、各因子趨勢 tr_<f>（NaN＝未滿 12 個月）。"""
    TR = {f: factor_trend(FD[f], n) for f in FACTORS}
    rows = []
    for x in S0['info']:
        sd = pd.Timestamp(x['sig_date'])
        ref = pd.Period(sd, 'M') - 1
        r = dict(month=x['month'], k=int(x['k']), sig_date=str(sd.date()), ref_month=str(ref), n_tgt=int(x['n_tgt']))
        for f in FACTORS:
            v = TR[f].get(ref, np.nan)
            r['tr_' + f] = float(v) if np.isfinite(v) else np.nan
        rows.append(r)
    return pd.DataFrame(rows)


def triggered(F, rule):
    """布林陣列：該訊號列因子趨勢 ≤ 0（有值時）→ 觸發；趨勢未滿 12 個月 → 不觸發。"""
    v = F['tr_' + rule['var']].to_numpy(float)
    return np.isfinite(v) & (v <= 0)


def apply_regime(S0, F, trig, action):
    """同 s23lib.apply_regime（stop／half 兩種）：只改當月買進名單 sig／rank；X 不動。回傳 S1 與 log。"""
    sig = S0['sig'].copy(); rank = S0['rank'].copy()
    log = []
    for x, tg in zip(S0['info'], trig):
        if not tg:
            continue
        k = x['k']
        tg_idx = np.flatnonzero(sig[k])
        if action == 'stop':
            keep = []
        elif action == 'half':
            order = tg_idx[np.argsort(-rank[k, tg_idx], kind='stable')]
            keep = list(order[:int(math.ceil(len(order) / 2))])
        else:
            raise ValueError(action)
        ks = set(keep)
        drop = [j for j in tg_idx if j not in ks]
        sig[k, drop] = False; rank[k, drop] = np.nan
        log.append(dict(month=x['month'], k=int(k), n_orig=int(len(tg_idx)), n_new=int(len(keep))))
    S1 = dict(S0); S1['sig'] = sig; S1['rank'] = rank
    return S1, log


def random_trig(n_rows, eligible, n_on, rng):
    idx = np.flatnonzero(eligible)
    out = np.zeros(n_rows, bool)
    if n_on > 0 and len(idx):
        out[rng.choice(idx, size=min(n_on, len(idx)), replace=False)] = True
    return out


# ---------------------------------------------------------------- PIT 產業、市值
def load_industry_pit():
    df = pd.read_csv(IND_CSV, dtype=str, keep_default_na=False)
    df['start'] = pd.to_datetime(df['start'])
    df['end'] = pd.to_datetime(df['end'].replace('', '2100-01-01'))
    df['industry'] = df['industry'].str.strip()
    return df


def industry_matrix(P, ks):
    """同 s21lib.industry_matrix：dict(k → 長度 N 的 int 陣列，-1＝缺值) 與名稱表。"""
    df = load_industry_pit()
    names = sorted(set(df['industry']) - MISSING)
    nid = {n: i for i, n in enumerate(names)}
    rows = {c: g[['start', 'end', 'industry']].to_numpy() for c, g in df.groupby('code')}
    out = {}
    for k in ks:
        d = P.IDX[k]
        a = np.full(P.N, -1, int)
        for j, c in enumerate(P.COLS):
            r = rows.get(str(c))
            if r is None:
                continue
            for s, e, ind in r:
                if s <= d < e:
                    a[j] = nid.get(ind, -1)
                    break
        out[int(k)] = a
    return out, names


def mcap_panel(P, lag=63):
    """同 s24 build.py：股數（shares_full）× 收盤，股數落後 lag 交易日；回傳 (T, N) float32。"""
    sh = pd.read_parquet(AG / 'o_lc_sector_rotation_pit' / 'data' / 'shares_full.parquet')
    sh = sh.reindex(sh.index.union(P.IDX)).ffill().reindex(P.IDX).reindex(columns=P.COLS).to_numpy(np.float64)
    now = sh * P.CF
    MC = np.full((P.T, P.N), np.nan)
    MC[lag:] = now[:-lag] * (P.G[lag:] / P.G[:-lag])
    MC = np.where(np.isfinite(MC), MC, now)
    return MC.astype(np.float32)


# ---------------------------------------------------------------- 持有期報酬（同 s20lib.outcomes 公式）
def last_listed(P):
    """每檔最後一個 MK > 0 的列（沒有就 -1）。"""
    mk = P.MK > 0
    rev = mk[::-1].argmax(axis=0)
    out = P.T - 1 - rev
    out[~mk.any(axis=0)] = -1
    return out


def hold_ret(P, j, e, x, LL):
    """j 在 e 日開盤買、x 日開盤賣的含息毛報酬；x ≥ T 或 x 為 None → 用最後一日收盤；
    中途下市（最後掛牌日 < x）→ 最後掛牌日收盤；x 日無開盤 → x−1 收盤。e 日無開盤 → NaN。"""
    O, CF, G, REF = P.O, P.CF, P.G, P.REF
    if not (np.isfinite(O[e, j]) and O[e, j] > 0 and np.isfinite(CF[e, j]) and np.isfinite(G[e, j]) and G[e, j] > 0):
        return np.nan
    base = CF[e, j] / O[e, j]
    end = P.T - 1 if (x is None or x >= P.T) else int(x)
    ll = LL[j]
    if ll < end:                                           # 中途下市
        t = ll
        while t > e and not np.isfinite(G[t, j]):
            t -= 1
        return base * G[t, j] / G[e, j] - 1
    if x is None or x >= P.T:
        t = end
        while t > e and not np.isfinite(G[t, j]):
            t -= 1
        return base * G[t, j] / G[e, j] - 1
    t = x - 1
    while t > e and not np.isfinite(G[t, j]):
        t -= 1
    r = base * G[t, j] / G[e, j]
    if np.isfinite(O[x, j]) and O[x, j] > 0 and np.isfinite(REF[x, j]) and REF[x, j] > 0 and t == x - 1:
        r *= O[x, j] / REF[x, j]
    return r - 1


def e50_ret(P, e, x):
    if x is None or x >= P.T:
        return P.E50[P.T - 1] / P.E50O[e] - 1
    return P.E50O[x] / P.E50O[e] - 1
