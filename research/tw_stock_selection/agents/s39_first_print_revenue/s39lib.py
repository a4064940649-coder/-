#!/usr/bin/env python3
"""S39 共用：首次公布營收版 SURGE 的最小包裝（不改任何共用引擎）。
在 s20lib（→s17lib→s10lib）之上：開 cslib.EXRIGHT_FIX＝True，中間檔改放 scratchpad/s39/。
surge_variant(P, F, FPM, mode)：
  'CUR'：用 s17lib.rev_family 的 Rv 重算 D、s17lib.zscore → 必須與 F['S']['STD'] 逐位相同（驗收）。
  'FPM'：只有當月 D 用首次值（逐月換 D 第 m 列，μ、σ 仍用目前值）。
  'FPA'：營收矩陣 2024-06～2026-09 中 FP 與 CUR 都有值的格子全換成 FP 後照標準算。
一律 python3 -I -B。"""
import sys
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
AG = HERE.parent
S39 = Path('/tmp/claude-0/-home-user--/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/s39/cache')
sys.path.insert(0, str(AG / 's20_persistent_growth'))
import s20lib                                              # noqa: E402
from s20lib import s17lib, s10lib, s9lib, s8lib, s7lib, s6lib, s5lib, s4lib, s3lib, s2lib, stlib, cslib, COSTS   # noqa: E402,F401
s20lib.S20 = S39; s17lib.S17 = S39; s10lib.S10 = S39; s9lib.S9 = S39; s8lib.S8 = S39; s7lib.S7 = S39; s6lib.S6 = S39; s5lib.S5 = S39; s4lib.S4 = S39
s3lib.S3 = S39; s2lib.S2 = S39; stlib.S1 = S39; cslib.TMP = S39
cslib.EXRIGHT_FIX = True

FP_CSV = HERE / 'data' / 'first_print.csv.gz'
ACC = dict(since2019=0.4899, full=0.3454, maxdd=-0.3211, n_trades=381)
MONTHS = list(pd.period_range('2024-06', '2026-09', freq='M'))
IND_CSV = AG / 'o_lc_sector_rotation_pit' / 'raw' / 'industry_pit.csv'


def load():
    S39.mkdir(parents=True, exist_ok=True)
    P = s20lib.load()
    assert P.exright_fix, '未開 EXRIGHT_FIX'
    return P


def load_fp():
    d = pd.read_csv(FP_CSV, dtype={'code': str})
    d['m'] = pd.PeriodIndex(d['ym'], freq='M')
    return d


def fp_matrix(P, Rv, fp, col='FP'):
    """與 Rv 同形的矩陣：FP 與 CUR 都有值（> 0）的格子給首次值（≤ 0 → NaN），其餘 NaN；另回傳遮罩。"""
    M = pd.DataFrame(np.nan, index=Rv.index, columns=Rv.columns)
    d = fp[fp['code'].isin(set(P.COLS)) & fp['m'].isin(set(Rv.index))]
    piv = d.pivot(index='m', columns='code', values=col).reindex(index=Rv.index, columns=Rv.columns)
    has = piv.notna() & Rv.notna()
    return piv.where(piv > 0), has


def _D(Rv):
    D = Rv - Rv.shift(12)
    D[D.index.year == 2013] = np.nan
    return D


def _Z(D):
    Z, _ = s17lib.zscore(D, 24, 'mean')
    Z[Z.index < s17lib.FIRST_MONTH] = np.nan
    return Z


def surge_variant(P, F, fp, mode):
    Rv = F['Rv']
    if mode == 'CUR':
        return _Z(_D(Rv))
    FPv, has = fp_matrix(P, Rv, fp)
    if mode == 'FPA':
        R2 = Rv.where(~has, FPv)
        return _Z(_D(R2))
    if mode == 'FPM':
        D0 = _D(Rv)
        Z = _Z(D0)
        out = Z.copy()
        for m in MONTHS:
            if m not in Rv.index:
                continue
            i = Rv.index.get_loc(m)
            rowR = Rv.iloc[i].where(~has.iloc[i], FPv.iloc[i])
            D1 = D0.copy()
            D1.iloc[i] = rowR - Rv.iloc[i - 12]
            if m.year == 2013:
                D1.iloc[i] = np.nan
            out.iloc[i] = _Z(D1).iloc[i]
        return out
    raise ValueError(mode)


def industry_latest():
    df = pd.read_csv(IND_CSV, dtype=str, keep_default_na=False)
    df = df.sort_values(['code', 'start'])
    return df.groupby('code')['industry'].last().str.strip().to_dict()
