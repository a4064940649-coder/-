#!/usr/bin/env python3
"""S19 共用程式庫：XQ 門檻版回測（照 20／21 號腳本的「固定門檻」，不是研究版的「名次」）。
在 s10lib（不改）之上：面板、創新板、引擎（simulate_s10）、統計全用 s10lib／s9lib／s4lib／s2lib／stlib／cslib；本檔只新增
- rev_std()：標準 SURGE，σ > 0 才有值（逐字複製 s17lib.zscore／rev_family 的 STD 部分，不 import s17lib 以免動到它的快取路徑）；
- e50_fix_suspension()、seg_dd()、SEGMENTS：逐字複製 s17lib（0050 2025-06-11～17 停牌日開盤用前一日收盤；四段空頭）；
- b11_rows()：XQ IsRebal（第一個日期 ≥ RebalDay 的交易日），用 s9lib.signal_rows(conv='B')；
- script_universe()：照 20 號腳本字面的股池（還原日線的 Close ≥ MinPrice、Average(Close × Volume, 60) ≥ MinAvgValue）；
- lma_nth()：每月 L_MA 第 10／20 名的 SURGE（動態門檻與描述用）；
- make_signal_thr()：門檻版的訊號／出場矩陣，格式同 s4lib.make_signal（sig／rank／X／info），可直接餵 simulate_s10。
中間檔放 scratchpad/s19tmp/。規格見 PREREG.json。一律 python3 -I -B。"""
import sys
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
AG = HERE.parent
S19 = Path('/tmp/claude-0/-home-user--/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/s19tmp')
sys.path.insert(0, str(AG / 's10_exit_details'))
import s10lib                                              # noqa: E402
from s10lib import s9lib, s8lib, s7lib, s6lib, s5lib, s4lib, s3lib, s2lib, stlib, cslib, COSTS   # noqa: E402,F401
s10lib.S10 = S19; s9lib.S9 = S19; s8lib.S8 = S19; s7lib.S7 = S19; s6lib.S6 = S19; s5lib.S5 = S19; s4lib.S4 = S19; s3lib.S3 = S19; s2lib.S2 = S19; stlib.S1 = S19; cslib.TMP = S19

FIRST_MONTH = pd.Period('2016-01', 'M')
SEGMENTS = {'2018': ('2018-01-01', '2018-12-31'), '2020-03': ('2020-01-01', '2020-06-30'),
            '2022': ('2022-01-01', '2022-12-31'), '2024-07~2025-04': ('2024-07-01', '2025-04-30')}
SUSP = ('2025-06-11', '2025-06-17')


def load():
    S19.mkdir(parents=True, exist_ok=True)
    return s10lib.load()


def e50_fix_suspension(P):
    """逐字同 s17lib.e50_fix_suspension：停牌日 0050 開盤缺值 → E50O_raw＝前一個有成交日原始收盤；E50O＝E50[t−1]。"""
    idx = pd.DatetimeIndex(P.IDX)
    rows = np.flatnonzero((idx >= pd.Timestamp(SUSP[0])) & (idx <= pd.Timestamp(SUSP[1])))
    out = []
    P.E50O_raw = np.array(P.E50O_raw, dtype=float, copy=True); P.E50O = np.array(P.E50O, dtype=float, copy=True)
    for t in rows:
        before = (float(P.E50O_raw[t]) if np.isfinite(P.E50O_raw[t]) else None, float(P.E50O[t]))
        if not (np.isfinite(P.E50O_raw[t]) and P.E50O_raw[t] > 0):
            tt = t - 1
            while tt > 0 and not (np.isfinite(P.E50C_raw[tt]) and P.E50C_raw[tt] > 0):
                tt -= 1
            P.E50O_raw[t] = P.E50C_raw[tt]
            P.E50O[t] = P.E50[t - 1]
        out.append(dict(date=str(idx[t].date()), before_open_raw=before[0], before_E50O=before[1], after_open_raw=float(P.E50O_raw[t]), after_E50O=float(P.E50O[t])))
    return out


def _windows(A, w):
    M, N = A.shape
    V = np.full((M, N, w), np.nan)
    if M > w:
        sw = np.lib.stride_tricks.sliding_window_view(A, w, axis=0)
        V[w:] = sw[:M - w]
    return V


def rev_std(P):
    """標準 SURGE（逐字同 s17lib.rev_family 的 STD）：D＝營收(m)−營收(m−12)（營收 ≤ 0 當缺值；2013 年的 D 缺值）；
    μ、σ 用 D(m−1…m−24)，24 個全部要有值；σ ≤ 1e-9 ×（窗內 |D| 平均＋1e-12）當缺值；m < 2016-01 一律缺值。
    回傳 (S：DataFrame 月份 Period × P.COLS, n_sigma0)。"""
    r = pd.read_csv(stlib.REV_CSV, dtype={'code': str}, parse_dates=['month'])
    r['pri'] = (r['market'] != 'sii').astype(int)
    r = r.sort_values(['month', 'code', 'pri']).drop_duplicates(['month', 'code'], keep='first')
    Rv = r.pivot(index='month', columns='code', values='revenue_k_twd').sort_index()
    Rv.index = pd.PeriodIndex(Rv.index, freq='M')
    Rv = Rv.reindex(pd.period_range(Rv.index.min(), Rv.index.max(), freq='M')).reindex(columns=P.COLS)
    Rv = Rv.where(Rv > 0)
    D = Rv - Rv.shift(12)
    D[D.index.year == 2013] = np.nan
    A = D.to_numpy(float)
    out = np.full(A.shape, np.nan); nzero = 0; w = 24
    for i0 in range(0, A.shape[1], 400):
        a = A[:, i0:i0 + 400]
        V = _windows(a, w)
        full = np.isfinite(V).all(axis=2)
        with np.errstate(invalid='ignore'):
            mu = V.mean(axis=2)
            s = np.sqrt(((V - mu[:, :, None]) ** 2).sum(axis=2) / (w - 1))
            scale = np.abs(V).mean(axis=2)
            pos = full & np.isfinite(s) & (s > 1e-9 * (scale + 1e-12))
            nzero += int((full & np.isfinite(a) & ~pos).sum())
            z = np.where(pos & np.isfinite(a), (a - mu) / np.where(pos, s, 1.0), np.nan)
        out[:, i0:i0 + 400] = z
        del V
    Z = pd.DataFrame(out, index=D.index, columns=D.columns)
    Z[Z.index < FIRST_MONTH] = np.nan
    return Z, nzero


def a10_rows(P, months):
    return cslib.rev_avail_days(P, months, True)


def b11_rows(P, months, R=11):
    av, dl = s9lib.signal_rows(P, months, 'B', R=R)
    return av, dl


def script_universe(P, INNOV, min_price=10.0, min_val=1e7, liq_days=60):
    """照 20 號腳本字面（XQ「還原日線」）：Close ≥ MinPrice、Average(Close × Volume, 60) ≥ MinAvgValue（1,000 萬元）。
    還原日線在回測時以「最新一天」為基準往回等比例還原（假設；需在 XQ 確認）：Cadj[t]＝G[t] × 最後收盤 ÷ G[最後]（G＝含息累積指數，CF＝無成交沿用前收）。
    Volume 用官方日 K 的成交股數（無成交日 0）；不在日 K 內的日子不算（同研究版的 60 日視窗規則）。
    其餘：上市、排除創新板、排除 91 開頭（TDR）、當日有成交。回傳 (univX, Cadj 的最後一列比例 adj_ratio[t, j] 的摘要不回傳)。"""
    lastC = P.CF[-1]
    Cadj = P.G * (lastC / P.G[-1])[None, :]
    val = np.where(P.listed, Cadj * P.VOLF, np.nan)
    liqX = pd.DataFrame(val).rolling(liq_days, min_periods=liq_days).mean().to_numpy()
    with np.errstate(invalid='ignore'):
        U = (P.MK == 1) & ~INNOV & P.not91[None, :] & P.has & (Cadj >= min_price) & (liqX >= min_val)
    return U, Cadj, liqX


def lma_nth(P, U, S, av, ns=(10, 20)):
    """每個訊號月：L_MA（U 內、SURGE 有值、還原收盤 > 60 日均線，依 SURGE 由大到小）第 n 名的 SURGE；不到 n 檔為 NaN。
    月份規則同 s4lib.make_signal（k0 ≥ 61、k0 < T、U[k0] 內 SURGE 有值 ≥ 50 檔）。回傳 DataFrame(month, k, date, n_ma, s10, s20, ...)。"""
    rows = []
    for m in S.index:
        k = av[m]
        if k >= P.T or k < 61:
            continue
        s = S.loc[m].to_numpy()
        if int((U[k] & np.isfinite(s)).sum()) < 50:
            continue
        ok = U[k] & np.isfinite(s) & P.F_MA60[k]
        o = np.sort(s[ok])[::-1]
        d = dict(month=str(m), k=int(k), date=str(P.IDX[k].date()), n_ma=int(ok.sum()))
        for n in ns:
            d[f's{n}'] = float(o[n - 1]) if len(o) >= n else float('nan')
        rows.append(d)
    return pd.DataFrame(rows)


def _thr(v, year, month):
    """門檻：數字；{年: 值}（動態門檻）；或 {'YYYY-MM': 值}（驗收 A2 用：每月剛好等於名次分數）。"""
    if isinstance(v, dict):
        return float(v[month]) if month in v else float(v[year])
    return float(v)


def make_signal_thr(P, U_month, U_buy, U_exit, S, av, min_s, exit_s, mom_lo=0.10, shift=0):
    """門檻版（20＋21 號腳本）。每個訊號月 k0＝av[m]（月份規則同 s4lib.make_signal：k0 ≥ 61、k0 < T、U_month[k0] 內 SURGE 有值 ≥ 50 檔），k＝k0＋shift。
    買進候選＝U_buy[k] 內、SURGE 有值、SURGE ≥ min_s、RET20 > mom_lo、還原收盤 > 60 日均線；全部標 sig、rank＝SURGE（引擎依 SURGE 由大到小買到 10 檔滿為止，滿了不買）。
    出場（21 號）：續抱＝SURGE 有值且 ≥ exit_s 且還原收盤 > 60 日均線（U_exit 不為 None 時另要在 U_exit[k] 內）；其餘 X＝True（營收缺值也賣）。
    min_s／exit_s 可為數字、{年: 值}（動態門檻，用訊號日的年份）或 {營收月 'YYYY-MM': 值}。"""
    sig = np.zeros((P.T, P.N), bool); X = np.zeros((P.T, P.N), bool); rank = np.full((P.T, P.N), np.nan)
    info = []
    for m in S.index:
        k0 = av[m]
        if k0 >= P.T or k0 < 61:
            continue
        s = S.loc[m].to_numpy()
        if int((U_month[k0] & np.isfinite(s)).sum()) < 50:
            continue
        k = k0 + shift
        if k >= P.T:
            continue
        yr = int(pd.Timestamp(P.IDX[k]).year)
        ms, es = _thr(min_s, yr, str(m)), _thr(exit_s, yr, str(m))
        fin = np.isfinite(s)
        sv = np.where(fin, s, -np.inf)
        with np.errstate(invalid='ignore'):
            c = U_buy[k] & fin & (sv >= ms) & (P.RET20[k] > mom_lo) & P.F_MA60[k]
        cand = np.flatnonzero(c)
        cand = cand[np.argsort(-s[cand], kind='stable')]
        sig[k, cand] = True; rank[k, cand] = s[cand]
        keep = fin & (sv >= es) & P.F_MA60[k]
        if U_exit is not None:
            keep = keep & U_exit[k]
        X[k, :] = ~keep
        info.append(dict(month=str(m), k=int(k), sig_date=str(P.IDX[k].date()), n_tgt=int(len(cand)), n_keep=int(keep.sum()), min_s=ms, exit_s=es,
                         targets=[str(P.COLS[j]) for j in cand]))
    sigrows = np.zeros(P.T, bool); sigrows[[x['k'] for x in info]] = True
    return dict(sig=sig, rank=rank, X=X, SD=None, max_hold=None, info=info, sigrows=sigrows, RQ={})


def seg_dd(nav):
    o = {}
    for k, (a, b) in SEGMENTS.items():
        s = nav.loc[a:b].dropna()
        o[k] = float((s / s.cummax() - 1).min()) if len(s) > 2 else float('nan')
    return o
