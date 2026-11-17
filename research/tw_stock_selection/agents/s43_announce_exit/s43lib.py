#!/usr/bin/env python3
"""S43 共用程式庫：持股的下個月營收「首次公布」一出來就很差 → 公布隔天開盤提早賣出（不等月中換股日）。S36（公告即進場）的出場版。
在 s20lib（不改；其下 s17lib→s10lib：面板、創新板、SURGE（STD）、F1、E0、引擎）之上，本檔只新增：
- 開 cslib.EXRIGHT_FIX＝True（現行驗收 48.99／34.54／−32.11／381 筆已出場）。
- 首次可見時間：直接用 s36lib.load_seen 的同一份 Tasi 資料與同一個對應法（r＝T 的台北日期（含）以前最後一個交易日；r 收盤後才知道 → r＋1 開盤賣）。
- 首次公布值 SURGE：用 s39lib.surge_variant(mode='FPM')（只有當月 D 用首次值，μ、σ 用當時已知的過去值；首次值缺 → 用目前值）。
  只用來判斷提早出場；進場（F1）與月中出場（E0）仍用現行 SURGE，不改。
- build_exit(P, U, RT, S, rule, thr, months, seen, SFP, shift)：在 make_signal_s10 的 S 上，把 XV['E0'] 複製一份，於「每檔自己的可見列 e＝r＋shift」
  （只在 r ＜ 官方訊號列 av[m] 時）把符合規則的格子設 True。引擎 simulate_s10 本來就會在非訊號列看 X：持有中且 X[t,j]＝True → 標記、隔天開盤賣
  （原因 'ma_daily'；跌停／無開盤順延），賣得的錢照引擎掃進 0050；名額空著，等官方訊號列 k＋1 照現行 F1 補（已持有不重買）。不改引擎。
  規則 None、months 空或 θ＝−∞ → S 原樣傳回 → 逐位重現現行（驗收 A2）。
中間檔放 scratchpad/s43/。規格見 PREREG.json。一律 python3 -I -B。"""
import sys
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
AG = HERE.parent
S43 = Path('/tmp/claude-0/-home-user--/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/s43')
sys.path.insert(0, str(AG / 's36_announce_entry'))
sys.path.insert(0, str(AG / 's39_first_print_revenue'))
import s36lib                                              # noqa: E402  （只讀：load_seen、月份切分）
import s39lib                                              # noqa: E402  （只讀：首次值 FPM）
import s20lib                                              # noqa: E402
from s20lib import s17lib, s10lib, s9lib, s8lib, s7lib, s6lib, s5lib, s4lib, s3lib, s2lib, stlib, cslib, COSTS   # noqa: E402,F401
for _m, _a in ((s20lib, 'S20'), (s17lib, 'S17'), (s10lib, 'S10'), (s9lib, 'S9'), (s8lib, 'S8'), (s7lib, 'S7'), (s6lib, 'S6'), (s5lib, 'S5'), (s4lib, 'S4'),
               (s3lib, 'S3'), (s2lib, 'S2'), (stlib, 'S1'), (cslib, 'TMP'), (s36lib, 'S36'), (s39lib, 'S39')):
    setattr(_m, _a, S43)
cslib.EXRIGHT_FIX = True

ACC = dict(since2019=0.4899, full=0.3454, maxdd=-0.3211, n_trades=381)
TRAIN_MONTHS = s36lib.TRAIN_MONTHS                          # 營收月份 2024-06～2025-07
VALID_MONTHS = s36lib.VALID_MONTHS                          # 營收月份 2025-08～2026-09
THR_GRID = (0.5, 1.0, 1.5, 2.0)

RULES = [
    dict(id='XA_THR', thr='grid', ma20=False,
         desc='持股的新營收首次公布（Tasi 可見列 r，r＜官方訊號列）當晚，用首次值算的 SURGE（FPM）< θ → r＋1 開盤賣；θ ∈ {0.5, 1.0, 1.5, 2.0} 訓練期選'),
    dict(id='XA_NEG_MA20', thr=0.0, ma20=True,
         desc='同上，但條件＝首次值 SURGE < 0 且 r 當日還原收盤 ≤ 20 日均線（~F_MA20）；無自由參數'),
    dict(id='XA_PREV20', thr='prev20', ma20=False,
         desc='θ＝上個月官方訊號列 L_MA（依現行 SURGE 排序）第 20 名的 SURGE（＝上月 E0 的留任門檻；不足 20 檔取最後一名）；首次值 SURGE < θ → r＋1 開盤賣；無自由參數'),
]


def load():
    S43.mkdir(parents=True, exist_ok=True)
    P = s20lib.load()
    assert P.exright_fix, '未開 EXRIGHT_FIX'
    return P


def surges(P):
    """回傳 RT（現行 STD SURGE，給引擎）與 SFP（首次值 FPM SURGE，只給提早出場判斷）。另驗 s39 CUR 包裝逐位相同。"""
    F = s17lib.rev_family(P, defs=['STD'])
    fp = s39lib.load_fp()
    cur = s39lib.surge_variant(P, F, fp, 'CUR')
    ok = bool(np.array_equal(cur.to_numpy(), F['S']['STD'].to_numpy(), equal_nan=True))
    SFP = s39lib.surge_variant(P, F, fp, 'FPM')
    return dict(S=F['S']['STD'], av=F['av']), SFP, ok


def lma_order(P, U, s, k):
    fin = U[k] & np.isfinite(s)
    ok_ma = fin & P.F_MA60[k]
    return np.flatnonzero(ok_ma)[np.argsort(-s[ok_ma], kind='stable')]


def prev20_thr(P, U, RT, months, shift=0):
    S, av = RT['S'], RT['av']
    out = {}
    for m in months:
        pm = m - 1
        k = av.get(pm, P.T) + shift
        if pm not in S.index or k >= P.T:
            out[m] = -np.inf; continue
        s = S.loc[pm].to_numpy()
        o = lma_order(P, U, s, k)
        out[m] = float(s[o[min(19, len(o) - 1)]]) if len(o) else -np.inf
    return out


def exit_events(P, RT, seen, SFP, months, shift=0):
    """每個 (m, j)：Tasi 可見列 r＜官方 av[m] 的上市股（全部，不管有沒有持有；引擎只對持有者生效）。不套規則。"""
    av = RT['av']
    ci = P.ci
    by_m = {}
    for (mm, c), r in seen.items():
        by_m.setdefault(mm, []).append((c, r))
    rows = []
    for m in months:
        k0 = av.get(m, P.T)
        if k0 >= P.T:
            continue
        sf = SFP.loc[m].to_numpy() if m in SFP.index else np.full(P.N, np.nan)
        for c, r in by_m.get(m, []):
            if c not in ci or r >= k0:
                continue
            j = ci[c]; e = r + shift
            if e >= P.T - 1:
                continue
            rows.append((str(m), int(k0 + shift), int(r), int(e), int(j), c, float(sf[j]), bool(P.F_MA20[e, j])))
    return pd.DataFrame(rows, columns=['month', 'k', 'r', 'e', 'j', 'code', 'surge_fp', 'ma20'])


def build_exit(P, U, RT, S, rule, thr, months, seen, SFP, shift=0, ev=None):
    """回傳 (S'，提早出場標記清單 [(month, code, e)])。rule None 或 months 空 → S 原樣。thr：數值或 dict(m→θ)。
    SURGE_FP 為 NaN（例如 σ 不合格）→ 不觸發。"""
    if rule is None or not months:
        return S, []
    if ev is None:
        ev = exit_events(P, RT, seen, SFP, months, shift)
    X = S['XV']['E0'].copy()
    marks = []
    ms = set(str(m) for m in months)
    for x in ev.itertuples():
        if x.month not in ms:
            continue
        th = thr[pd.Period(x.month, 'M')] if isinstance(thr, dict) else thr
        if not (np.isfinite(x.surge_fp) and x.surge_fp < th):
            continue
        if rule['ma20'] and x.ma20:
            continue
        if S['sigrows'][x.e]:
            raise AssertionError('提早出場列撞到官方訊號列')
        X[x.e, x.j] = True
        marks.append((x.month, x.code, int(x.e)))
    if not marks:
        return S, []
    S2 = dict(S); XV = dict(S['XV']); XV['E0'] = X; S2['XV'] = XV; S2['X'] = X
    return S2, marks
