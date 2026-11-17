#!/usr/bin/env python3
"""S58 共用程式庫：春節讓 1、2 月營收年增金額失真——「1、2 月合併計算 YoY」的修正（C1／C2／C3）對現行，訓練→凍結→驗證一次。
載入鏈逐行照 agents/s50_exit_candidate_validation/s50lib.py 的 setup（其現行格已逐位重現 deadline_engine_fix ALL：48.76%／381 筆）：
s20lib→s17lib→s10lib；EXRIGHT_FIX（load 前）；SURGE（STD）在 DEADLINE_FIX／INS15_MASK 關閉下建，之後兩個開關打開、rev_avail_days 重算可得日；
e50_fix_suspension；股池 twse & ~創新板；F1＝make_signal_s10(depth=10, mom_lo=0.10, nsel=10)、E0 出場＝其 X；simulate_s10(strict, sat_w=0.7, W10, lock=False)。
本檔只新增：候選規則改的是 RT['S']（SURGE 矩陣，月份 × P.COLS）的「營收月份 1 月、2 月」兩列，其餘月份逐格不動：
- C1：營收月份 1 月整列缺值 → make_signal_s10 跳過該列＝2 月換股日不換股（不買不賣、持股與名單沿用前一月）。
- C2：營收月份 1 月列 ← 前一年 12 月的 STD SURGE（1 月訊號日只有 1 月資料，改用 12 月訊號）；
      營收月份 2 月列 ← 「1＋2 月合計 YoY」SURGE：D′＝D 序列中每年 1、2 月都換成「(1＋2 月合計) − (去年 1＋2 月合計)」÷ 2（一個缺值用另一個，同 S34），
      z＝(D′(2 月) − μ24′)／σ24′，μ、σ 用 D′(m−1…m−24)（視窗內的 1、2 月也是合併值），s17lib.zscore（σ > 0）。
- C3：只在「春節月份互換年」套 C2（該年春節（正月初一）所在國曆月份 ≠ 前一年）；其他年份＝現行。
不改任何既有檔案。中間檔放 scratchpad/s58/cache。"""
import sys, gc, warnings
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
AG = HERE.parent
TMP = Path('/tmp/claude-0/-home-user/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/s58')
CACHE = TMP / 'cache'
sys.path.insert(0, str(AG / 's20_persistent_growth'))
import s20lib                                              # noqa: E402
from s20lib import s17lib, s10lib, s9lib, s8lib, s7lib, s6lib, s5lib, s4lib, s3lib, s2lib, stlib, cslib, COSTS   # noqa: E402,F401
for _m, _a in ((s20lib, 'S20'), (s17lib, 'S17'), (s10lib, 'S10'), (s9lib, 'S9'), (s8lib, 'S8'), (s7lib, 'S7'), (s6lib, 'S6'), (s5lib, 'S5'),
               (s4lib, 'S4'), (s3lib, 'S3'), (s2lib, 'S2'), (stlib, 'S1'), (cslib, 'TMP')):
    setattr(_m, _a, CACHE)

TRAIN_LAST_MONTH = s20lib.TRAIN_LAST_MONTH                 # 營收月份 ≤ 2021-11（訊號日 2016-02～2021-12）
VALID_FIRST_MONTH = s20lib.VALID_FIRST_MONTH               # 營收月份 ≥ 2021-12（訊號日 2022-01 起）
fresh_from = s20lib.fresh_from
CAPS = (6e5, 8e5, 1e6, 1.5e6, 2e6)
CELL = dict(sat_w=0.7, sizing='W10', lock=False)
TAIEX_CSV = AG / 'o_concentration_regime' / 'raw' / 'taiex_twii.csv'

# 春節（農曆正月初一）國曆日期；除夕＝前一天。同 S34 的表（agents/s34_lunar_new_year/s34lib.py LNY），另以加權指數休市空檔核對。
LNY = {2012: '2012-01-23', 2013: '2013-02-10', 2014: '2014-01-31', 2015: '2015-02-19', 2016: '2016-02-08', 2017: '2017-01-28',
       2018: '2018-02-16', 2019: '2019-02-05', 2020: '2020-01-25', 2021: '2021-02-12', 2022: '2022-02-01', 2023: '2023-01-22',
       2024: '2024-02-10', 2025: '2025-01-29', 2026: '2026-02-17'}
LNY_SOURCE = ('農曆正月初一的國曆日期：交通部中央氣象署天文站「農曆與國曆對照表」（公開曆書）、行政院人事行政總處各年「政府行政機關辦公日曆表」；'
              '除夕＝正月初一前一天。天文曆法決定，前一年即公告（無前視）。與 S34 同一張表；另以 TWSE 加權指數日資料（taiex_twii.csv）1～2 月最長無交易空檔核對。')
LNY_MONTH = {y: pd.Timestamp(d).month for y, d in LNY.items()}
SWAP = {y: bool(LNY_MONTH[y] != LNY_MONTH[y - 1]) for y in LNY if y - 1 in LNY}   # 春節月份互換年

CANDS = ('CUR', 'C1', 'C2', 'C3')
CAND_DESC = {'CUR': '現行：每月照 STD SURGE 換股',
             'C1': '1 月營收的換股（2 月）暫停：不買不賣、持股沿用前一月；其餘月份照現行',
             'C2': '1 月營收訊號改用 12 月 STD SURGE；2 月營收訊號用「1＋2 月合計 YoY」SURGE（μ24、σ24 視窗內的 1、2 月也用合併值）；其餘月份照現行',
             'C3': '只在春節月份互換年（春節所在國曆月份 ≠ 前一年）套 C2；其他年份照現行'}


def setup():
    """回傳 (P, U, RT, Z)。三個正式開關全開。Z＝dict(Rv, D, Zfull（STD 不遮 2016 前）, Zjf（合併 YoY）)。"""
    CACHE.mkdir(parents=True, exist_ok=True)
    assert cslib.DEADLINE_FIX is False and cslib.INS15_MASK is False
    cslib.EXRIGHT_FIX = True
    P = s20lib.load()
    assert P.exright_fix
    assert (P.T, P.N, str(P.IDX[-1].date())) == (2876, 2142, '2026-10-22'), '面板不符'
    INNOV, _ = s4lib.innov_panel(P)
    U = stlib.universe(P, 'twse') & ~INNOV
    del INNOV
    F = s17lib.rev_family(P, defs=['STD'])
    s17lib.e50_fix_suspension(P)
    for a in ('H', 'L', 'VOL', 'CHG', 'HF', 'VOLF', 'adjH', 'HI240', 'MA60', 'nbar', 'F_HI', 'IDXC', 'AO', 'listed', 'NEARHI', 'V60', 'liq'):
        if hasattr(P, a):
            delattr(P, a)
    S0 = F['S']['STD']; months = list(S0.index); Rv = F['Rv']
    del F; gc.collect()
    cslib.DEADLINE_FIX = True; cslib.INS15_MASK = True
    AV = cslib.rev_avail_days(P, months, True)
    Z = surge_parts(Rv)
    return P, U, dict(S=S0, av=AV), Z


def d_raw(Rv):
    D = Rv - Rv.shift(12)
    D[D.index.year == 2013] = np.nan                       # IFRS 斷點（同 s17lib）
    return D


def d_jf(D):
    """每年 1、2 月的 D 都換成兩者平均（＝(1＋2 月合計 − 去年 1＋2 月合計)÷2）；一個缺值用另一個（同 S34 'jf'）。"""
    A = D.to_numpy(float); out = A.copy()
    pos = {(p.year, p.month): i for i, p in enumerate(D.index)}
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        for y in sorted(set(D.index.year)):
            ia, ib = pos.get((y, 1)), pos.get((y, 2))
            if ia is None or ib is None:
                continue
            m = np.nanmean(np.vstack([A[ia], A[ib]]), axis=0)
            out[ia] = m; out[ib] = m
    return pd.DataFrame(out, index=D.index, columns=D.columns)


def surge_parts(Rv):
    D = d_raw(Rv)
    Zfull, _ = s17lib.zscore(D, 24, 'mean')
    Zjf, _ = s17lib.zscore(d_jf(D), 24, 'mean')
    return dict(Rv=Rv, D=D, Zfull=Zfull, Zjf=Zjf)


def variant_S(S0, Z, cand):
    """候選 → SURGE 矩陣（只改營收月份 1、2 月列，且只改 ≥ 2016-01 的列；其餘逐格＝S0）。"""
    if cand == 'CUR':
        return S0
    S = S0.copy()
    for m in S.index:
        if m < s17lib.FIRST_MONTH or m.month not in (1, 2):
            continue
        if cand == 'C3' and not SWAP.get(m.year, False):
            continue
        if cand == 'C1':
            if m.month == 1:
                S.loc[m] = np.nan
        else:                                               # C2／C3
            if m.month == 1:
                S.loc[m] = Z['Zfull'].loc[m - 1].to_numpy()
            else:
                S.loc[m] = Z['Zjf'].loc[m].to_numpy()
    return S


def signals(P, U, RT, Z, shift, cands=CANDS):
    """{cand: make_signal_s10 輸出（不含 XV）}。"""
    out = {}
    for c in cands:
        RT1 = dict(S=variant_S(RT['S'], Z, c), av=RT['av'])
        sg = s10lib.make_signal_s10(P, U, RT1, shift=shift, depth=10, mom_lo=0.10, nsel=10)
        del sg['XV']
        out[c] = sg
    return out


def sim(P, U, S, cap=1e6, cost='strict', fresh=False, block=None):
    if fresh:
        S = fresh_from(S, VALID_FIRST_MONTH)
    return s10lib.simulate_s10(P, S, U, cost=COSTS[cost], capital=cap, block=block, **CELL)


def jm_windows(P, S_cur):
    """事後描述用：每年「2、3 月換股期間」＝營收月份 1 月訊號列 k1 的收盤 → 營收月份 3 月訊號列 k3 的收盤（現行訊號列；
    涵蓋 2 月、3 月兩次換股的持有期）。回傳 {year: (k1, k3)}。"""
    ks = {pd.Period(x['month'], 'M'): int(x['k']) for x in S_cur['info']}
    out = {}
    for m, k in ks.items():
        if m.month == 1 and pd.Period(f'{m.year}-03', 'M') in ks:
            out[m.year] = (k, ks[pd.Period(f'{m.year}-03', 'M')])
    return out


def taiex_dates():
    d = pd.read_csv(TAIEX_CSV, usecols=['date'], parse_dates=['date'])['date']
    return pd.DatetimeIndex(sorted(set(d)))
