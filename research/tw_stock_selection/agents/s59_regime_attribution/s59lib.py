#!/usr/bin/env python3
"""S59 共用：在本題暫存重建 S55 的 2007～2015H1 面板，並以 S55 同一套引擎（s47lib → … → cslib，檔案不改）
跑凍結規則的樣本外（'oos'）與現行正式樣本內（'is'）。只在執行期改快取目錄與樣本窗（同 s55lib），不改任何策略參數。
一律 python3 -I -B。"""
import sys, shutil, hashlib
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
AG = HERE.parent
ROOT = AG.parent
SCR = Path('/tmp/claude-0/-home-user/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/s59')
OFFCACHE = Path('/tmp/claude-0/-home-user--/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/eng/cache')
OFF_SHA = '434f64222b8fd1f10e1e25c9ac4553e7e6f6d11305896a3ceabb3b212650dfab'
sys.path.insert(0, str(AG / 's47_allocation_family'))
import s47lib as L                                         # noqa: E402
from s47lib import s20lib, s17lib, s10lib, s9lib, s8lib, s7lib, s6lib, s5lib, s4lib, s3lib, s2lib, stlib, cslib   # noqa: E402

OOS_START = pd.Timestamp('2008-01-02')
OOS_FIRST_MONTH = pd.Period('2008-01', 'M')
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()


def build_pre_panel(out):
    """逐行照 S55 build_pre_panel.py：長表（raw/pre2015 衍生檔）→ 面板；data_panel 有值但長表沒有的格子用 data_panel 補。只寫 out。"""
    D0, D1 = pd.Timestamp('2007-01-02'), pd.Timestamp('2015-06-30')
    dp = ROOT / 'data_panel' / 'panel'
    cr = pd.read_parquet(dp / 'close_raw.parquet'); op = pd.read_parquet(dp / 'open.parquet')
    hi = pd.read_parquet(dp / 'high.parquet'); va = pd.read_parquet(dp / 'value.parquet')
    win = cr.loc[D0:D1]; IDX = win.index
    codes = [c for c in win.columns if win[c].notna().any()]
    Lg = pd.read_csv(ROOT / 'raw' / 'pre2015' / 'asoul_twse_2007_2015H1.csv.gz', dtype={'code': str}, parse_dates=['date'])
    Lg = Lg[Lg['date'].isin(IDX)]
    COLS = np.array(sorted(codes)); ci = {c: j for j, c in enumerate(COLS)}; di = {d: i for i, d in enumerate(IDX)}
    T, N = len(IDX), len(COLS)
    A = {k: np.full((T, N), np.nan) for k in ('O', 'H', 'L', 'C', 'VOL', 'AMT', 'CHG')}
    MK = np.zeros((T, N), np.int8)
    Lg = Lg[Lg['code'].isin(ci)]
    ti = Lg['date'].map(di).to_numpy(); tj = Lg['code'].map(ci).to_numpy()
    for k, col in (('O', 'open'), ('H', 'high'), ('L', 'low'), ('C', 'close'), ('VOL', 'volume'), ('AMT', 'amount'), ('CHG', 'change')):
        A[k][ti, tj] = Lg[col].to_numpy(float)
    MK[ti, tj] = 1
    dpC = win.reindex(columns=COLS).to_numpy()
    fill = np.isfinite(dpC) & ~np.isfinite(A['C'])
    A['C'][fill] = dpC[fill]
    A['O'][fill] = op.loc[IDX].reindex(columns=COLS).to_numpy()[fill]
    A['H'][fill] = hi.loc[IDX].reindex(columns=COLS).to_numpy()[fill]
    A['AMT'][fill] = va.loc[IDX].reindex(columns=COLS).to_numpy()[fill]
    MK[fill] = 1
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / 'panel_cs.npz', IDX=IDX.values.astype('datetime64[ns]'), COLS=COLS.astype(str), C=A['C'], MK=MK,
                        O=A['O'], H=A['H'], L=A['L'], VOL=A['VOL'], AMT=A['AMT'], CHG=A['CHG'])
    np.savez_compressed(out / 'innov.npz', A=np.zeros((T, N), bool), IDX=IDX.values.astype('datetime64[ns]'), COLS=COLS.astype(str),
                        codes=np.array([], dtype=str), names=np.array([], dtype=str))
    e = pd.read_csv(OFFCACHE / 'e50_ohlc.csv', parse_dates=['date']).set_index('date')
    e.reindex(IDX).rename_axis('date').reset_index().to_csv(out / 'e50_ohlc.csv', index=False)
    return dict(T=T, N=N, n_long_rows=int(len(Lg)), n_fill=int(fill.sum()))


def setup(which):
    cache = SCR / ('cache_is' if which == 'is' else 'cache_pre')
    cache.mkdir(parents=True, exist_ok=True)
    if which == 'is':
        assert sha(OFFCACHE / 'panel_cs.npz') == OFF_SHA, '官方面板快取不是預期版本'
        for f in ('panel_cs.npz', 'e50_ohlc.csv', 'innov.npz'):
            if not (cache / f).exists():
                shutil.copy2(OFFCACHE / f, cache / f)
    else:
        if not (cache / 'panel_cs.npz').exists():
            build_pre_panel(cache)
        stlib.START = OOS_START
        s17lib.FIRST_MONTH = OOS_FIRST_MONTH
    for m, a in ((s20lib, 'S20'), (s17lib, 'S17'), (s10lib, 'S10'), (s9lib, 'S9'), (s8lib, 'S8'), (s7lib, 'S7'), (s6lib, 'S6'), (s5lib, 'S5'),
                 (s4lib, 'S4'), (s3lib, 'S3'), (s2lib, 'S2'), (stlib, 'S1'), (cslib, 'TMP')):
        setattr(m, a, cache)
    L.CACHE = cache
    assert cslib.EXRIGHT_FIX and cslib.DEADLINE_FIX and cslib.INS15_MASK
    return cache


def build(which):
    setup(which)
    P = L.load()
    INNOV, _ = s4lib.innov_panel(P)
    U = stlib.universe(P, 'twse') & ~INNOV
    F = s17lib.rev_family(P, defs=['STD'])
    RT = dict(S=F['S']['STD'], av=F['av'])
    fix = s17lib.e50_fix_suspension(P)
    sg = s10lib.make_signal_s10(P, U, RT, shift=0)
    return P, U, RT, sg, fix


def monthly(nav, e50):
    """同 s55lib.monthly。"""
    nav = nav.dropna()
    b = e50.reindex(nav.index)
    me = nav.groupby(nav.index.to_period('M')).tail(1)
    s = pd.concat([nav.iloc[:1], me]); bb = pd.concat([b.iloc[:1], b.reindex(me.index)])
    return pd.DataFrame(dict(strategy=s.pct_change().iloc[1:].to_numpy(), b0050=bb.pct_change().iloc[1:].to_numpy()), index=me.index.to_period('M'))
