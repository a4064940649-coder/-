#!/usr/bin/env python3
"""S40 共用：營收可得日（期限延後／晚交）修正的最小包裝（不改任何共用引擎）。
在 s20lib（→s17lib→s10lib）之上：開 cslib.EXRIGHT_FIX＝True，中間檔改放 scratchpad/s40/cache。
- deadline_table(P)：營收月份 2015-01～2026-10 的法定期限、特例延長、引擎訊號日、分類。
- av_fix(P, av, ext)：把 ext＝{月份: 延後期限日} 的訊號列改成『延後期限（含）以後第一個交易日』的下一個交易日（同 rev_avail_days 規則）。
- mask_S(S, cells, mode)：cells＝[(月份, 代碼)]，mode 'nan'＝看不到就沒有、'stale'＝改用上月 SURGE。
一律 python3 -I -B。"""
import sys
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
AG = HERE.parent
TMP = Path('/tmp/claude-0/-home-user--/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/s40')
CACHE = TMP / 'cache'
sys.path.insert(0, str(AG / 's20_persistent_growth'))
import s20lib                                              # noqa: E402
from s20lib import s17lib, s10lib, s9lib, s8lib, s7lib, s6lib, s5lib, s4lib, s3lib, s2lib, stlib, cslib, COSTS   # noqa: E402,F401
s20lib.S20 = CACHE; s17lib.S17 = CACHE; s10lib.S10 = CACHE; s9lib.S9 = CACHE; s8lib.S8 = CACHE; s7lib.S7 = CACHE; s6lib.S6 = CACHE; s5lib.S5 = CACHE; s4lib.S4 = CACHE
s3lib.S3 = CACHE; s2lib.S2 = CACHE; stlib.S1 = CACHE; cslib.TMP = CACHE
cslib.EXRIGHT_FIX = True

SEEN_CSV = AG / 'o_rev_event_clock_ts' / 'data' / 'firstseen_sources.csv.gz'
RA_2609 = AG / 'rev_announce' / 'firstseen_2026-09.csv'
TASI_WIN = AG / 's39_first_print_revenue' / 'extract_check.json'      # 每月 Tasi 第一個／最後一個快照時間
ACC = dict(since2019=0.4899, full=0.3454, maxdd=-0.3211, n_trades=381)
TZ = 'Asia/Taipei'
LATE_PMS = ['LATE_SURE', 'UNSEEN_LATE', 'LATE_LEGAL']            # 確定看不到
LATE_PM = LATE_PMS + ['LATE_GAP']                              # 主分析
LATE_PMW = LATE_PM + ['LATE_GAP_DL']                           # 最壞
FIRST_M, LAST_M = pd.Period('2024-06', 'M'), pd.Period('2026-09', 'M')

# 特例延長（全體）與敏感度（2020-01 只限受疫情影響且申請的公司）
EXT_ALL = {pd.Period('2019-01', 'M'): pd.Timestamp('2019-02-15'), pd.Period('2023-03', 'M'): pd.Timestamp('2023-04-12')}
EXT_2020 = {pd.Period('2020-01', 'M'): pd.Timestamp('2020-02-15')}
EXT_NOTE = {'2019-01': '金管會特例：春節 9 天連假延至 2019-02-15（全體）',
            '2020-01': '肺炎疫情：受影響公司經申請可延至 2020-02-15（非全體，敏感度）',
            '2023-03': '金管會公告：清明連假延至 2023-04-12（全體）'}
# 2026-01 起適用 15 日期限的公司（rev_announce 依新聞整理：上市金控具保險子公司＋上市保險＋2905、2207）；2884 玉山另列待確認
INS15 = ['2880', '2881', '2882', '2883', '2885', '2886', '2887', '2891', '2892', '5880', '2816', '2832', '2850', '2851', '2852', '2867', '2905', '2207']
INS15_START = pd.Period('2026-01', 'M')


def load():
    CACHE.mkdir(parents=True, exist_ok=True)
    P = s20lib.load()
    assert P.exright_fix, '未開 EXRIGHT_FIX'
    return P


def first_td_on_or_after(P, d):
    return int(np.searchsorted(P.IDX.values, np.datetime64(pd.Timestamp(d)), side='left'))


def av_fix(P, av, ext):
    out = dict(av)
    for m, d in ext.items():
        if m in out:
            out[m] = first_td_on_or_after(P, d) + 1
    return out


def mask_S(S, cells, mode):
    S2 = S.copy()
    ci = {c: j for j, c in enumerate(S.columns)}
    ri = {m: i for i, m in enumerate(S.index)}
    A = S2.to_numpy(copy=True)
    A0 = S.to_numpy()
    n = 0
    for m, c in cells:
        if m not in ri or c not in ci:
            continue
        i, j = ri[m], ci[c]
        A[i, j] = np.nan if mode == 'nan' else A0[i - 1, j]
        n += 1
    return pd.DataFrame(A, index=S.index, columns=S.columns), n


def deadline_table(P, months):
    av = cslib.rev_avail_days(P, months, True)
    rows = []
    for m in months:
        d10 = (m + 1).to_timestamp() + pd.Timedelta(days=9)
        kD = first_td_on_or_after(P, d10)
        if kD >= P.T:
            continue
        D = P.IDX[kD]
        k = av[m]
        S = P.IDX[k] if k < P.T else None
        ext = EXT_ALL.get(m) or EXT_2020.get(m)
        Dx = P.IDX[first_td_on_or_after(P, ext)] if ext is not None else D
        r = dict(rev_month=str(m), tenth=str(d10.date()), tenth_weekday=d10.day_name(), tenth_is_trading=bool(D == d10),
                 deadline_statutory=str(D.date()), special=EXT_NOTE.get(str(m)), deadline_actual=str(Dx.date()),
                 signal_day=str(S.date()) if S is not None else None)
        if S is not None:
            r['case'] = 'deadline>signal（問題）' if Dx > S else ('deadline=signal（邊界）' if Dx == S else 'deadline<signal（安全）')
            r['days_short'] = int(np.searchsorted(P.IDX.values, np.datetime64(Dx), side='left') - k) if Dx > S else 0
        if m >= INS15_START:
            d15 = (m + 1).to_timestamp() + pd.Timedelta(days=14)
            D15 = P.IDX[first_td_on_or_after(P, d15)] if first_td_on_or_after(P, d15) < P.T else None
            r['deadline_ins15'] = str(D15.date()) if D15 is not None else str(d15.date()) + '（面板外）'
            if S is not None:
                r['ins15_case'] = '問題' if (D15 is None or D15 > S) else ('邊界' if D15 == S else '安全')
        rows.append(r)
    return pd.DataFrame(rows)


def load_seen():
    d = pd.read_csv(SEEN_CSV, dtype={'code': str}, usecols=['source', 'code', 'ym', 'T', 'T_prev', 'reason'])
    d = d[d['source'] == 'tasi'].copy()
    d['m'] = pd.PeriodIndex(d['ym'], freq='M')
    d = d[(d['m'] >= FIRST_M) & (d['m'] <= LAST_M)]
    d['T'] = pd.to_datetime(d['T'], utc=True).dt.tz_convert(TZ)
    d['T_prev'] = pd.to_datetime(d['T_prev'], utc=True).dt.tz_convert(TZ)
    return d
