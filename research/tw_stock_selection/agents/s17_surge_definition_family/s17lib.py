#!/usr/bin/env python3
"""S17 共用程式庫：SURGE 定義家族（標準＋6 個新定義，每個只改一個維度）。
在 s10lib（不改）之上：資料、訊號日、創新板、引擎、統計全用 s10lib／s9lib／s4lib／s2lib／stlib；本檔只新增
- rev_family()：從同一份月營收算出 7 個定義的 SURGE 矩陣（index＝月份 Period、columns＝P.COLS），一律「σ > 0」才有值（σ＝0 當缺值）；
- e50_fix_suspension()：0050 在 2025-06-11～17 停牌（分割）日，開盤價缺值 → 用前一日收盤；
- seg_dd()、top5_share()：同 s16lib（逐字複製，不 import s16lib 以免改到它的快取路徑）。
中間檔放 scratchpad/s17tmp/。規格見 PREREG.json。一律 python3 -I -B。"""
import sys
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
AG = HERE.parent
S17 = Path('/tmp/claude-0/-home-user--/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/s17tmp')
sys.path.insert(0, str(AG / 's10_exit_details'))
import s10lib                                              # noqa: E402
from s10lib import s9lib, s8lib, s7lib, s6lib, s5lib, s4lib, s3lib, s2lib, stlib, cslib, COSTS   # noqa: E402,F401
s10lib.S10 = S17; s9lib.S9 = S17; s8lib.S8 = S17; s7lib.S7 = S17; s6lib.S6 = S17; s5lib.S5 = S17; s4lib.S4 = S17; s3lib.S3 = S17; s2lib.S2 = S17; stlib.S1 = S17; cslib.TMP = S17

DEFS = ['STD', 'W12', 'W36', 'RATE', 'SM3', 'NOMU', 'ROB']
DEF_SPEC = {
    'STD':  dict(scale='amount', smooth=1, win=24, center='mean',   desc='標準：D＝營收(m)−營收(m−12)；(D−μ)÷σ，μ、σ 用 D(m−1…m−24)'),
    'W12':  dict(scale='amount', smooth=1, win=12, center='mean',   desc='視窗：μ、σ 改用 D(m−1…m−12)'),
    'W36':  dict(scale='amount', smooth=1, win=36, center='mean',   desc='視窗：μ、σ 改用 D(m−1…m−36)'),
    'RATE': dict(scale='rate',   smooth=1, win=24, center='mean',   desc='尺度：D 改用年增率 營收(m)÷營收(m−12)−1'),
    'SM3':  dict(scale='amount', smooth=3, win=24, center='mean',   desc='平滑：D 改用近 3 個月合計的年增金額'),
    'NOMU': dict(scale='amount', smooth=1, win=24, center='none',   desc='去均值：D÷σ（不減 μ）'),
    'ROB':  dict(scale='amount', smooth=1, win=24, center='median', desc='統計量：(D−中位數)÷(1.4826×MAD)'),
}
FIRST_MONTH = pd.Period('2016-01', 'M')                     # 所有定義的訊號月份 ≥ 2016-01（同標準）
SEGMENTS = {'2018': ('2018-01-01', '2018-12-31'), '2020-03': ('2020-01-01', '2020-06-30'),
            '2022': ('2022-01-01', '2022-12-31'), '2024-07~2025-04': ('2024-07-01', '2025-04-30')}
SUSP = ('2025-06-11', '2025-06-17')                          # 0050 分割停牌


def load():
    S17.mkdir(parents=True, exist_ok=True)
    return s10lib.load()


def e50_fix_suspension(P):
    """0050 停牌日（2025-06-11～17）：官方日 K 無 0050 開盤 → 原程式以 100 元當成交參考價（只影響跳動點滑價）、還原開盤＝當日還原收盤。
    本題改成「前一日收盤」：E50O_raw＝前一個有成交日的原始收盤；E50O＝前一日還原收盤 E50[t−1]（停牌日沒有新價格）。回傳修改的列與修改前後的值。"""
    idx = pd.DatetimeIndex(P.IDX)
    rows = np.flatnonzero((idx >= pd.Timestamp(SUSP[0])) & (idx <= pd.Timestamp(SUSP[1])))
    out = []
    P.E50O_raw = np.array(P.E50O_raw, dtype=float, copy=True); P.E50O = np.array(P.E50O, dtype=float, copy=True)
    for t in rows:
        before = (float(P.E50O_raw[t]) if np.isfinite(P.E50O_raw[t]) else None, float(P.E50O[t]), float(P.E50[t]), float(P.E50[t - 1]),
                  float(P.E50C_raw[t]) if np.isfinite(P.E50C_raw[t]) else None)
        if not (np.isfinite(P.E50O_raw[t]) and P.E50O_raw[t] > 0):
            tt = t - 1
            while tt > 0 and not (np.isfinite(P.E50C_raw[tt]) and P.E50C_raw[tt] > 0):
                tt -= 1
            P.E50O_raw[t] = P.E50C_raw[tt]
            P.E50O[t] = P.E50[t - 1]
        out.append(dict(date=str(idx[t].date()), before_open_raw=before[0], before_E50O=before[1], E50=before[2], E50_prev=before[3], close_raw=before[4],
                        after_open_raw=float(P.E50O_raw[t]), after_E50O=float(P.E50O[t])))
    return out


def _windows(A, w):
    """A：M×N（月份×股票）。回傳 M×N×w 的視窗陣列 V，V[m] 是 A[m−w … m−1]（不含當月）；前 w 列為 NaN。"""
    M, N = A.shape
    V = np.full((M, N, w), np.nan)
    if M > w:
        sw = np.lib.stride_tricks.sliding_window_view(A, w, axis=0)   # (M−w+1, N, w)，sw[i] = A[i … i+w−1]
        V[w:] = sw[:M - w]
    return V


def zscore(D, w, center):
    """D：DataFrame（月份×股票）。μ／中心、σ／尺度用前 w 個月（全部要有值），逐窗兩段式計算（不用 pandas 的線上演算法，常數序列的 σ 精確為 0 附近）。
    σ > 0 條件：σ ≤ 1e-9 ×（窗內 |D| 平均 ＋ 1e-12）視為 0，當缺值。center：'mean'、'none'（不減）、'median'（MAD 尺度）。"""
    A = D.to_numpy(float)
    out = np.full(A.shape, np.nan)
    nzero = 0
    for i0 in range(0, A.shape[1], 400):                   # 分塊省記憶體
        a = A[:, i0:i0 + 400]
        V = _windows(a, w)
        full = np.isfinite(V).all(axis=2)
        with np.errstate(invalid='ignore'):
            if center == 'median':
                c = np.median(V, axis=2)
                s = 1.4826 * np.median(np.abs(V - c[:, :, None]), axis=2)
            else:
                mu = V.mean(axis=2)
                s = np.sqrt(((V - mu[:, :, None]) ** 2).sum(axis=2) / (w - 1))
                c = mu if center == 'mean' else np.zeros_like(mu)
            scale = np.abs(V).mean(axis=2)
            pos = full & np.isfinite(s) & (s > 1e-9 * (scale + 1e-12))
            nzero += int((full & np.isfinite(a) & ~pos).sum())
            z = np.where(pos & np.isfinite(a), (a - c) / np.where(pos, s, 1.0), np.nan)
        out[:, i0:i0 + 400] = z
        del V
    return pd.DataFrame(out, index=D.index, columns=D.columns), nzero


def rev_family(P, defs=DEFS):
    """7 個定義的 SURGE。營收讀法同 s2lib.rev_tables（同代碼優先上市、營收 ≤ 0 缺值、千元）。
    IFRS 斷點：用到 2012 年（含）以前營收的年增量一律缺值——單月（標準等）＝m 在 2013 年；SM3＝m ≤ 2014-02（3 個月合計的 m−14 落在 2012）。
    回傳 dict(S={id: DataFrame}, av＝訊號日列號（同 s2lib）, n_sigma0={id: σ＝0 而被當缺值的格數}, Rv＝月營收矩陣（驗收 A2 用）)。"""
    r = pd.read_csv(stlib.REV_CSV, dtype={'code': str}, parse_dates=['month'])
    r['pri'] = (r['market'] != 'sii').astype(int)
    r = r.sort_values(['month', 'code', 'pri']).drop_duplicates(['month', 'code'], keep='first')
    Rv = r.pivot(index='month', columns='code', values='revenue_k_twd').sort_index()
    Rv.index = pd.PeriodIndex(Rv.index, freq='M')
    Rv = Rv.reindex(pd.period_range(Rv.index.min(), Rv.index.max(), freq='M')).reindex(columns=P.COLS)
    Rv = Rv.where(Rv > 0)
    out, n0 = {}, {}
    for d in defs:
        sp = DEF_SPEC[d]
        if sp['smooth'] == 3:
            R3 = Rv.rolling(3, min_periods=3).sum()
            D = R3 - R3.shift(12)
            D[D.index <= pd.Period('2014-02', 'M')] = np.nan
        elif sp['scale'] == 'rate':
            D = (Rv / Rv.shift(12) - 1).replace([np.inf, -np.inf], np.nan)
            D[D.index.year == 2013] = np.nan
        else:
            D = Rv - Rv.shift(12)
            D[D.index.year == 2013] = np.nan
        Z, nz = zscore(D, sp['win'], sp['center'])
        Z[Z.index < FIRST_MONTH] = np.nan
        out[d] = Z; n0[d] = nz
    av = cslib.rev_avail_days(P, list(out[defs[0]].index), True)
    return dict(S=out, av=av, n_sigma0=n0, Rv=Rv)


def seg_dd(nav):
    """四段空頭各自的最大跌幅（高點與低點都在段內；同 s16lib.seg_dd）。"""
    o = {}
    for k, (a, b) in SEGMENTS.items():
        s = nav.loc[a:b].dropna()
        o[k] = float((s / s.cummax() - 1).min()) if len(s) > 2 else float('nan')
    return o


def top5_share(lg, nav, cap):
    """最賺 5 筆（已平倉，依損益金額）損益合計 ÷ 淨值增加（同 s16lib.top5_share）。"""
    x = lg[lg['why'] != 'open_end'].sort_values('pnl', ascending=False).head(5)
    return float(x['pnl'].sum() / (nav.iloc[-1] - cap))
