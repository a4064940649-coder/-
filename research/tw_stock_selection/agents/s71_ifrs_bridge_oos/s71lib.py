#!/usr/bin/env python3
"""S71 共用：跨 IFRS 斷點的 SURGE 規則（MASK＝現行凍結、B1＝不遮罩、B2＝水準校正）。
共用引擎一律不改：重用 agents/r55_replicate_s55/r55lib.py（快取改道、月度統計）與其下的 s47lib → … → cslib。
本檔只新增
- rev_table()：月營收矩陣，讀法逐行同 s17lib.rev_family（同代碼優先上市、營收 ≤ 0 缺值、千元）；
- jump_n()／abnormal_jump()／scope_ratio()：B2 的每家公司口徑比例 k_n（n＝1、2、3 個月累計）；
- rev_rule()：依規則算標準定義 SURGE（24 個月 σ，s17lib.zscore 原函式），回傳格式同 s17lib.rev_family；
- run()：現行格（F1、E0、A10、W10_70、嚴成本、正式開關）跑一次，只換 SURGE 來源。
一律 python3 -I -B；不呼叫 git（快取先放好）。規格見 SPEC.json、PREREG.json。"""
import sys
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
AG = HERE.parent
TMP = Path('/tmp/claude-0/-home-user/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/s71')
sys.path.insert(0, str(AG / 'r55_replicate_s55'))
import r55lib as R                                           # noqa: E402
from r55lib import L, s20lib, s17lib, s10lib, s4lib, stlib, cslib   # noqa: E402
R.TMP = TMP

PRIOR_YEARS = (2008, 2009, 2010, 2011, 2012)                 # B2：「前幾年同期月增」用的年份（y−1 年 12 月 → y 年 1 月）
MIN_PRIOR = 3                                                # 至少要有幾年
BREAK = pd.Period('2013-01', 'M')                            # 合併口徑第一個月


def rev_table(cols):
    r = pd.read_csv(stlib.REV_CSV, dtype={'code': str}, parse_dates=['month'])
    r['pri'] = (r['market'] != 'sii').astype(int)
    r = r.sort_values(['month', 'code', 'pri']).drop_duplicates(['month', 'code'], keep='first')
    Rv = r.pivot(index='month', columns='code', values='revenue_k_twd').sort_index()
    Rv.index = pd.PeriodIndex(Rv.index, freq='M')
    Rv = Rv.reindex(pd.period_range(Rv.index.min(), Rv.index.max(), freq='M'))
    if cols is not None:
        Rv = Rv.reindex(columns=cols)
    return Rv.where(Rv > 0)


def jump_n(Rv, y, n):
    """ln(y 年前 n 個月營收合計 ÷ y−1 年最後 n 個月營收合計)；n 個月都要有值，否則 NaN。"""
    P_ = lambda t: pd.Period(t, 'M')
    a = Rv.loc[P_(f'{y}-01'):P_(f'{y}-{n:02d}')]; b = Rv.loc[P_(f'{y - 1}-{13 - n:02d}'):P_(f'{y - 1}-12')]
    return np.log(a.sum(min_count=n) / b.sum(min_count=n))


def abnormal_jump(Rv, year, prior_years, min_prior=MIN_PRIOR, n=1):
    """a＝jump_n(year) − 中位數{jump_n(y)，y∈prior_years}；有效的 prior 年數 < min_prior 或當年跳動缺值 → NaN。回傳 Series（index＝代號）。"""
    pri = pd.concat([jump_n(Rv, y, n) for y in prior_years], axis=1)
    a = jump_n(Rv, year, n) - pri.median(axis=1, skipna=True)
    a[pri.notna().sum(axis=1) < min_prior] = np.nan
    return a


def scope_ratio(Rv, n):
    """B2 口徑比例 k_n＝max(1, exp(a_n))，a_n＝abnormal_jump(2013；2008～2012；n 個月累計)；a 缺值 → k 缺值。只用到 2013 年第 n 個月（含）以前的營收。"""
    a = abnormal_jump(Rv, 2013, PRIOR_YEARS, n=n)
    return np.exp(a.clip(lower=0)), a


def _d_b2(Rv, k):
    pre = np.asarray(Rv.index < BREAK)
    kv = k.reindex(Rv.columns).to_numpy(float)
    A = Rv.to_numpy(float).copy()
    A[pre] = A[pre] * np.where(np.isfinite(kv), kv, 1.0)[None, :]
    Radj = pd.DataFrame(A, index=Rv.index, columns=Rv.columns)
    Dv = np.array((Radj - Radj.shift(12)).to_numpy(float), copy=True)
    Dv[np.ix_(np.asarray(Rv.index.year == 2013), ~np.isfinite(kv))] = np.nan
    return pd.DataFrame(Dv, index=Rv.index, columns=Rv.columns)


def rev_rule(P, rule, first_month='2008-01'):
    """rule：'MASK'（現行：2013 年 D 缺值）、'B1'（2013 年 D 照算）、
    'B2'（水準校正：2012-12 以前營收 × k_n 後照算，k 缺值的公司 2013 年 D 缺值；n 依訊號月份只用當時已公布的月數：
          營收月 2013-01 → n＝1、2013-02 → n＝2、2013-03 起 → n＝3；營收月 ≤ 2012-12 的列照 MASK（整段同乘 k 不改變 z 值））。"""
    Rv = rev_table(P.COLS)
    info = {}
    zs = lambda D: s17lib.zscore(D, 24, 'mean')
    if rule == 'B2':
        Dm = Rv - Rv.shift(12); Dm[Dm.index.year == 2013] = np.nan
        Z, nz = zs(Dm)
        ks = {}
        for n in (1, 2, 3):
            k, a = scope_ratio(Rv, n); ks[n] = k
            Zn, _ = zs(_d_b2(Rv, k))
            rows = (Z.index == pd.Period(f'2013-{n:02d}', 'M')) if n < 3 else (Z.index >= pd.Period('2013-03', 'M'))
            Z.loc[rows] = Zn.loc[rows]
        D = _d_b2(Rv, ks[3])
        info = dict(k=ks)
    else:
        D = Rv - Rv.shift(12)
        if rule == 'MASK':
            D[D.index.year == 2013] = np.nan
        else:
            assert rule == 'B1'
        Z, nz = zs(D)
    Z[Z.index < pd.Period(first_month, 'M')] = np.nan
    av = cslib.rev_avail_days(P, list(Z.index), True)
    return dict(S={'STD': Z}, av=av, n_sigma0={'STD': nz}, Rv=Rv, D=D, info=info)


def run(rule, start='2008-01-02', first_month='2008-01'):
    """現行格跑一次（同 r55lib.run_current，只把 s17lib.rev_family 換成 rev_rule）。"""
    stlib.START = pd.Timestamp(start)
    assert cslib.EXRIGHT_FIX and cslib.DEADLINE_FIX and cslib.INS15_MASK
    P = s20lib.load()
    INNOV, _ = s4lib.innov_panel(P)
    U = stlib.universe(P, 'twse') & ~INNOV
    F = rev_rule(P, rule, first_month)
    RT = dict(S=F['S']['STD'], av=F['av'])
    s17lib.e50_fix_suspension(P)
    sg = s10lib.make_signal_s10(P, U, RT, shift=0)
    nav, lg, st = L.sim(P, sg, U, cap=1e6)
    return P, U, F, sg, nav, lg, st
