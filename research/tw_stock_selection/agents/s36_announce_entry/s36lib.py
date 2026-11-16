#!/usr/bin/env python3
"""S36 共用程式庫：逐檔營收「首次可見時間」→ 公告隔天開盤進場（出場仍照 E0 在月中換股日）。
在 s20lib（不改；其下 s17lib→s10lib：面板、創新板、SURGE（STD，σ>0）、F1、E0、引擎）之上，本檔只新增：
- 開 cslib.EXRIGHT_FIX＝True（現行驗收 48.99／34.54／−32.11／381 筆已出場）。
- load_seen(P)：讀 o_rev_event_clock_ts/data/firstseen_sources.csv.gz 的 Tasi 來源（MOPS 上市營收表每日約 18:16 台北時間的 git 快照；
  T＝該檔該月營收第一次出現在快照的 commit 時間）。回傳 dict[(Period m, code)] → 面板列號 r＝「T 的台北日期（含）以前最後一個交易日」。
  r 當日收盤後（或更晚）才知道 → 最早 r＋1 開盤買。T 是上界（真正公告只會更早），所以保守、PIT。
- build_early(P, U, RT, S, rule, months, shift)：在 make_signal_s10 的 S 上，於「每檔自己的可見列 r＋shift」加 sig（rank＝SURGE），
  只在 r ＜ 官方訊號列 av[m]（＝第一個 > 10 日期限的交易日）時加；r＋shift 當日收盤判斷 F_MA60、RET20、股池 U。
  引擎 simulate_s10 原本就每天看 sig[t−1]、有空名額就買（每檔＝衛星 ÷ 10、名額 10、已持有不重買），出場 X 只在官方訊號列看 → 不改引擎。
  months 為空（或規則 None）時 S 原樣傳回 → 逐位重現現行（驗收 A2）。
中間檔放 scratchpad/s36/。規格見 PREREG.json。一律 python3 -I -B。"""
import sys
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
AG = HERE.parent
S36 = Path('/tmp/claude-0/-home-user--/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/s36')
sys.path.insert(0, str(AG / 's20_persistent_growth'))
import s20lib                                              # noqa: E402
from s20lib import s17lib, s10lib, s9lib, s8lib, s7lib, s6lib, s5lib, s4lib, s3lib, s2lib, stlib, cslib, COSTS   # noqa: E402,F401
s20lib.S20 = S36; s17lib.S17 = S36; s10lib.S10 = S36; s9lib.S9 = S36; s8lib.S8 = S36; s7lib.S7 = S36; s6lib.S6 = S36; s5lib.S5 = S36; s4lib.S4 = S36
s3lib.S3 = S36; s2lib.S2 = S36; stlib.S1 = S36; cslib.TMP = S36
cslib.EXRIGHT_FIX = True

SEEN_CSV = AG / 'o_rev_event_clock_ts' / 'data' / 'firstseen_sources.csv.gz'
ACC = dict(since2019=0.4899, full=0.3454, maxdd=-0.3211, n_trades=381)
FIRST_M, LAST_M = pd.Period('2024-06', 'M'), pd.Period('2026-09', 'M')    # Tasi 可用營收月份（2024-04、05 為補建檔，排除）
TRAIN_MONTHS = list(pd.period_range('2024-06', '2025-07', freq='M'))     # 14 個月（公告在 2024-07～2025-08）
VALID_MONTHS = list(pd.period_range('2025-08', '2026-09', freq='M'))     # 14 個月（公告在 2025-09～2026-10）
THR_GRID = (2.0, 2.5, 2.8, 3.5)

RULES = [
    dict(id='EA_MOM', thr='grid', mom=True, desc='公告可見當晚 SURGE ≥ θ、收盤站上季線（F_MA60）、RET20 > 10%、在股池 → 隔天開盤買；θ ∈ {2.0, 2.5, 2.8, 3.5} 訓練期選'),
    dict(id='EA_NOMOM', thr='grid', mom=False, desc='同 EA_MOM 但不要求 RET20 > 10%（只要 SURGE ≥ θ 且站上季線）；θ 同一組格點'),
    dict(id='EA_ADAPT', thr='prev10', mom=True, desc='θ＝上個月官方訊號列 L_MA（依 SURGE 排序）第 10 名的 SURGE（不足 10 檔取最後一名）；其餘同 EA_MOM；無自由參數'),
]


def load():
    S36.mkdir(parents=True, exist_ok=True)
    P = s20lib.load()
    assert P.exright_fix, '未開 EXRIGHT_FIX'
    return P


def surge(P):
    F = s17lib.rev_family(P, defs=['STD'])
    return dict(S=F['S']['STD'], av=F['av'])


def load_seen(P):
    d = pd.read_csv(SEEN_CSV, dtype={'code': str}, usecols=['source', 'code', 'ym', 'T', 'reason'])
    d = d[d['source'] == 'tasi'].copy()
    d['m'] = pd.PeriodIndex(d['ym'], freq='M')
    d['date'] = pd.to_datetime(d['T'].str[:10])
    d = d[(d['m'] >= FIRST_M) & (d['m'] <= LAST_M)]
    r = np.searchsorted(P.IDX.values, d['date'].values.astype('datetime64[ns]'), side='right') - 1
    d['r'] = r
    return d, {(m, c): int(x) for m, c, x in zip(d['m'], d['code'], d['r'])}


def lma_order(P, U, s, k):
    fin = U[k] & np.isfinite(s)
    ok_ma = fin & P.F_MA60[k]
    return np.flatnonzero(ok_ma)[np.argsort(-s[ok_ma], kind='stable')]


def early_events(P, U, RT, seen, months, shift=0):
    """每個 (m, j)：可見列 r＜官方 av[m] 的上市股（只回傳 SURGE 有值者）；不套規則。回傳 DataFrame。"""
    S, av = RT['S'], RT['av']
    rows = []
    ci = P.ci
    for m in months:
        if m not in S.index:
            continue
        k0 = av[m]
        if k0 >= P.T:
            continue
        s = S.loc[m].to_numpy()
        for (mm, c), r in seen.items():
            if mm != m or c not in ci:
                continue
            j = ci[c]
            if not np.isfinite(s[j]) or r >= k0:
                continue
            e = r + shift
            if e >= P.T - 1:
                continue
            rows.append((str(m), int(k0 + shift), int(r), int(e), int(j), c, float(s[j]), bool(U[e, j]), bool(P.F_MA60[e, j]), float(P.RET20[e, j])))
    return pd.DataFrame(rows, columns=['month', 'k', 'r', 'e', 'j', 'code', 'surge', 'univ', 'ma60', 'ret20'])


def prev10_thr(P, U, RT, months, shift=0):
    S, av = RT['S'], RT['av']
    out = {}
    for m in months:
        pm = m - 1
        k = av.get(pm, P.T) + shift
        if pm not in S.index or k >= P.T:
            out[m] = np.inf; continue
        s = S.loc[pm].to_numpy()
        o = lma_order(P, U, s, k)
        out[m] = float(s[o[min(9, len(o) - 1)]]) if len(o) else np.inf
    return out


def build_early(P, U, RT, S, rule, thr, months, seen, shift=0, ev=None):
    """回傳 (S'，加進的早進場清單)。rule None 或 months 空 → S 原樣。thr：數值或 dict(m→θ)。"""
    if rule is None or not months:
        return S, []
    if ev is None:
        ev = early_events(P, U, RT, seen, months, shift)
    S2 = dict(S); sig = S['sig'].copy(); rank = S['rank'].copy()
    added = []
    ms = set(str(m) for m in months)
    for x in ev.itertuples():
        if x.month not in ms:
            continue
        th = thr[pd.Period(x.month, 'M')] if isinstance(thr, dict) else thr
        if not (x.univ and x.ma60 and x.surge >= th and (x.ret20 > 0.10 if rule['mom'] else True)):
            continue
        sig[x.e, x.j] = True; rank[x.e, x.j] = x.surge
        added.append((x.month, x.code, x.e))
    S2['sig'] = sig; S2['rank'] = rank
    return S2, added
