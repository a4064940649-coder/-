#!/usr/bin/env python3
"""S41 共用：15 日營收期限公司（保險業及具保險子公司，S40 名單 18 檔＋2884）在換股日看不到當月營收時的三種處理。
在 s20lib（→s17lib→s10lib，不改）之上的最小包裝：cslib.EXRIGHT_FIX＝True，中間檔放 scratchpad/s41/cache。
- hidden_cells(P, av, S, shift, first_month)：反事實「這群公司一直是 15 日期限」：營收月份 m 的訊號日（av[m]＋shift）早於
  15 日期限日（次月 15 日起第一個交易日）→ 該股月看不到（期限當天＝訊號日視為當晚可見，同 S40）。first_month 以前不遮（實際情境用 2026-01）。
- build(P, U, RT, treat, cells, shift)：回傳 make_signal_s10 格式的訊號 dict。
    'A'（現狀）：看不到的股月 SURGE＝NaN → 不排名、不買；持有中依 E0（不在 L_MA 前 20）賣出（＝make_list.py 現行行為）。
    'B'（持有沿用）：排名與買進同 A（看不到＝不在 L_MA、不新買）；持有中的這群股票改用「最後已知 SURGE」（上月 S[m−1]，換股日已公開）
         判斷續抱：股池內、還原收盤 > 60 日均線、S[m−1] 有值，且放進本月 L_MA（不含看不到者）後名次 ≤ 20（同分依欄位序，同 argsort stable）→ 續抱，否則賣。
    'C'（上月參與排名）：看不到的股月 SURGE 改用 S[m−1]，完整參與 L_MA 排名、F1 買進、E0 續抱（＝S40 的 PST）。
  cells 為空（這群公司一律看得到）時三者都與現行 make_signal_s10 逐位相同（驗收 A1）。
一律 python3 -I -B。"""
import sys
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
AG = HERE.parent
TMP = Path('/tmp/claude-0/-home-user--/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/s41')
CACHE = TMP / 'cache'
sys.path.insert(0, str(AG / 's20_persistent_growth'))
import s20lib                                              # noqa: E402
from s20lib import s17lib, s10lib, s9lib, s8lib, s7lib, s6lib, s5lib, s4lib, s3lib, s2lib, stlib, cslib, COSTS   # noqa: E402,F401
s20lib.S20 = CACHE; s17lib.S17 = CACHE; s10lib.S10 = CACHE; s9lib.S9 = CACHE; s8lib.S8 = CACHE; s7lib.S7 = CACHE; s6lib.S6 = CACHE; s5lib.S5 = CACHE; s4lib.S4 = CACHE
s3lib.S3 = CACHE; s2lib.S2 = CACHE; stlib.S1 = CACHE; cslib.TMP = CACHE
cslib.EXRIGHT_FIX = True

# S40 s40lib.INS15（18 檔）＋2884（S40 註明玉山金 2026-09 起實際也照 15 日）
INS15_S40 = ['2880', '2881', '2882', '2883', '2885', '2886', '2887', '2891', '2892', '5880', '2816', '2832', '2850', '2851', '2852', '2867', '2905', '2207']
GROUP = INS15_S40 + ['2884']
REAL_START = pd.Period('2026-01', 'M')
TRAIN_LAST_MONTH = s20lib.TRAIN_LAST_MONTH                 # 營收月份 ≤ 2021-11（訊號日 2016-02～2021-12）
VALID_FIRST_MONTH = s20lib.VALID_FIRST_MONTH               # 營收月份 ≥ 2021-12（訊號日 2022-01 起）
fresh_from = s20lib.fresh_from
ACC = dict(since2019=0.4899, full=0.3454, maxdd=-0.3211, n_trades=381)
TREATS = {'A': '現狀：看不到＝沒有 SURGE（不排名、不買；持有中依 E0 賣出）',
          'B': '持有中改用上月 SURGE 判斷續抱（放進本月 L_MA 名次 ≤ 20 才續抱）；不新買；排名不含這群公司',
          'C': '看不到的股月改用上月 SURGE 完整參與排名（可新買、續抱照 E0）'}


def load():
    CACHE.mkdir(parents=True, exist_ok=True)
    P = s20lib.load()
    assert P.exright_fix, '未開 EXRIGHT_FIX'
    return P


def first_td_on_or_after(P, d):
    return int(np.searchsorted(P.IDX.values, np.datetime64(pd.Timestamp(d)), side='left'))


def d15_index(P, m):
    """營收月份 m 的 15 日期限：次月 15 日（含）以後第一個交易日的列號（可能 ≥ T＝面板外）。"""
    return first_td_on_or_after(P, (m + 1).to_timestamp() + pd.Timedelta(days=14))


def hidden_cells(P, av, S, shift=0, first_month=None, codes=GROUP):
    """[(m, j)]：訊號列 av[m]＋shift 早於 15 日期限日的股月（j＝P.COLS 欄位號）。"""
    cj = [int(np.flatnonzero(P.COLS == c)[0]) for c in codes if (P.COLS == c).any()]
    out = []
    for m in S.index:
        if first_month is not None and m < first_month:
            continue
        k0 = av.get(m)
        if k0 is None or k0 >= P.T:
            continue
        k = k0 + shift
        if k >= P.T:
            continue
        if d15_index(P, m) > k:
            out.extend((m, j) for j in cj)
    return out


def _mask(S, cells, mode):
    """mode 'nan'：S[m, j]＝NaN；'stale'：S[m, j]＝S[m−1, j]（S 的前一列＝上個營收月份）。j＝P.COLS 欄位號（S 欄位與 P.COLS 同序）。"""
    if not cells:
        return S
    A = S.to_numpy(copy=True)
    A0 = S.to_numpy()
    ri = {m: i for i, m in enumerate(S.index)}
    for m, j in cells:
        i = ri[m]
        A[i, j] = np.nan if mode == 'nan' else (A0[i - 1, j] if i > 0 else np.nan)
    return pd.DataFrame(A, index=S.index, columns=S.columns)


def build(P, U, RT, treat, cells, shift=0):
    S0, av = RT['S'], RT['av']
    assert list(S0.columns) == list(P.COLS)
    if treat == 'C':
        return s10lib.make_signal_s10(P, U, dict(S=_mask(S0, cells, 'stale'), av=av), shift=shift), {}
    SN = _mask(S0, cells, 'nan')
    sg = s10lib.make_signal_s10(P, U, dict(S=SN, av=av), shift=shift)
    if treat == 'A' or not cells:
        return sg, {}
    assert treat == 'B'
    X = sg['XV']['E0'].copy()
    ri = {m: i for i, m in enumerate(S0.index)}
    A0 = S0.to_numpy()
    krow = {x['month']: x['k'] for x in sg['info']}
    by_m = {}
    for m, j in cells:
        by_m.setdefault(m, []).append(j)
    nkeep = 0; log = []
    for m, js in by_m.items():
        k = krow.get(str(m))
        if k is None:                                      # 該月沒有訊號列
            continue
        s = SN.loc[m].to_numpy()
        ok_ma = U[k] & np.isfinite(s) & P.F_MA60[k]
        idx = np.flatnonzero(ok_ma); sv = s[idx]
        i = ri[m]
        for j in js:
            v = A0[i - 1, j] if i > 0 else np.nan
            if not (U[k, j] and P.F_MA60[k, j] and np.isfinite(v)):
                continue
            ahead = int(((sv > v) | ((sv == v) & (idx < j))).sum())
            if ahead < 20:
                X[k, j] = False; nkeep += 1
                log.append((str(m), int(k), str(P.COLS[j]), ahead + 1))
    sg = dict(sg); XV = dict(sg['XV']); XV['E0'] = X; sg['XV'] = XV; sg['X'] = X
    return sg, dict(n_keepable=nkeep, keep_cells=log)


def sim(P, U, sg, cap=1e6, cost='strict', fresh=False, block=None):
    if fresh:
        sg = fresh_from(sg, VALID_FIRST_MONTH)
    S2, ekw = s10lib.variant_S(sg, 'E0')
    return s10lib.simulate_s10(P, S2, U, cost=COSTS[cost], capital=cap, block=block, sat_w=0.7, sizing='W10', lock=False, **ekw)


def prep_panel():
    """載入面板、股池、SURGE（STD）、0050 停牌修正、刪除不用的大陣列（同 S40 run.py）。"""
    import gc
    P = load()
    assert (P.T, P.N, str(P.IDX[-1].date())) == (2876, 2142, '2026-10-22'), 'P0 失敗'
    INNOV, _ = s4lib.innov_panel(P)
    U = stlib.universe(P, 'twse') & ~INNOV
    del INNOV
    F = s17lib.rev_family(P, defs=['STD'])
    RT = dict(S=F['S']['STD'], av=F['av'])
    s17lib.e50_fix_suspension(P)
    for a in ('H', 'L', 'VOL', 'CHG', 'HF', 'VOLF', 'adjH', 'HI240', 'MA60', 'nbar', 'F_HI', 'IDXC', 'AO', 'listed', 'NEARHI', 'V60', 'liq'):
        if hasattr(P, a):
            delattr(P, a)
    del F
    gc.collect()
    return P, U, RT
