#!/usr/bin/env python3
"""S85 共用：讀資料、兩種合併結構、指標。規格見 PREREG.json（開頭核對 sha256）。一律 python3 -I -B。"""
import json, hashlib
from pathlib import Path
import numpy as np, pandas as pd
H = Path(__file__).resolve().parent
AG = H.parent
_pre = (H / 'PREREG.json').read_bytes()
assert hashlib.sha256(_pre).hexdigest() == (H / 'PREREG.sha256').read_text().split()[0], 'PREREG 雜湊不符'
PRE = json.loads(_pre)
KAPPA = 0.005
START = '2016-01-30'                                    # 2016-01 最後交易日（補班日）；程式另以資料核對
TRAIN, VALID = ('2016-02', '2021-12'), ('2022-01', '2026-10')
COMPARABLE = ['PEAD_Q', 'DISP_D', 'PULLBACK_P', 'OTC_SURGE']
REFERENCE = ['MAIN_N10', 'CANSLIM_W', 'MA200_0050']
DESCRIBE = ['LOWB', 'LBQ', 'HDLV', 'MIX', 'H52_N10', 'N30', 'TURN_C', 'MINV']
SHORT = ['H52_N10', 'N30', 'TURN_C', 'MINV']             # 只到 2026-10-12
STRUCT = [('A', 0.5), ('A', 0.7), ('B', 0.5), ('B', 1.0)]


def load():
    sd = pd.read_csv(H / 'surge_daily.csv.gz', index_col=0, parse_dates=True)
    off = pd.read_csv('/home/user/-/xq_scripts/site/navs_official.csv.gz', index_col=0, parse_dates=True)
    assert sd.index.equals(off.index) and float((sd['nav'] - off['E0|W10_70']).abs().max()) < 0.01
    idx = off.index
    assert str(idx[idx < pd.Timestamp('2016-02-01')][-1].date()) == START
    C = {}
    cn = pd.read_csv(H / 'cand_navs_official_switch.csv.gz', index_col=0, parse_dates=True)
    for k in ('PEAD_Q', 'DISP_D', 'PULLBACK_P'):
        C[k] = cn[k]
    C['OTC_SURGE'] = pd.read_csv(AG / 's68_otc_universe_oos/navs.csv.gz', index_col=0, parse_dates=True)['OTC']
    m = pd.read_csv(AG / 'o_g6_stage3/sealed/v7x_navs_jmean_100w.csv.gz', index_col=0, parse_dates=True)
    C['MAIN_N10'] = m['base|REAL_opt|0|normal|100萬|std']; C['MAIN_N10_stress'] = m['base|REAL_opt|0|stress|100萬|std']
    C['CANSLIM_W'] = pd.read_csv(AG / 'o_canslim_wide_exit/navs.csv.gz', index_col=0, parse_dates=True)['T15_MA|E50']
    w = pd.read_parquet(AG / 'w4_lev2_half_rebalance/nav.parquet'); w.index = pd.to_datetime(w.index); C['MA200_0050'] = w['BENCH_MA200_0050']
    b = pd.read_csv(AG / 's30_defensive_base/basket_navs.csv.gz', index_col=0, parse_dates=True)
    for k in ('LOWB', 'LBQ', 'HDLV', 'MIX'):
        C[k] = b[k]
    mf = pd.read_parquet(AG / 'momentum_family/long_nav.parquet'); mf.index = pd.to_datetime(mf.index); C['H52_N10'] = mf['P1_52wHigh_n10_stress']
    se = pd.read_parquet(AG / 'strategy_ensemble/strategy_nav.parquet'); se.index = pd.to_datetime(se.index)
    for k in ('N30', 'TURN_C', 'MINV'):
        C[k] = se[k]
    for k in list(C):                                       # 對齊正式交易日；缺日前值補
        s = C[k].astype(float)
        s = s[~s.index.duplicated()].sort_index()
        C[k] = s.reindex(idx.union(s.index)).ffill().reindex(idx)
        if k in SHORT:
            C[k].loc[idx > pd.Timestamp('2026-10-12')] = np.nan
    cw = C['CANSLIM_W']; f0 = cw.first_valid_index()        # 偏離 1：2017-03-31 前該候選不存在，視為放 0050
    C['CANSLIM_W'] = cw.where(cw.notna(), off['0050'] / off['0050'].loc[f0] * cw.loc[f0])
    th = pd.read_csv(AG / 's83_environment_thermometer/thermometer_monthly.csv', index_col=0)['state']
    th.index = pd.PeriodIndex(th.index, freq='M')
    return sd, off['0050'], C, th


def blend_A(s_nav, c_nav, x, kappa=KAPPA, start=START):
    """分資金：x 給 SURGE 帳戶、1−x 給候選；每月最後交易日收盤拉回，成本 2κ×移動金額。回傳淨值（start＝1）。"""
    s = s_nav.loc[start:]; c = c_nav.loc[start:]
    rs = s.pct_change().fillna(0).to_numpy(); rc = c.pct_change().fillna(0).to_numpy()
    idx = s.index; per = idx.to_period('M'); last = np.r_[per[1:] != per[:-1], True]
    va, vb = x, 1 - x; out = np.empty(len(idx))
    for i in range(len(idx)):
        if i:
            va *= 1 + rs[i]; vb *= 1 + rc[i]
        if last[i] or i == 0:
            V = va + vb; mv = abs(va - x * V); V -= 2 * kappa * mv if i else 0.0
            va, vb = x * V, (1 - x) * V
        out[i] = va + vb
    return pd.Series(out, index=idx)


def blend_B(sd, b_nav, c_nav, k, kappa=KAPPA, start=START, blk=False):
    """閒置資金與底倉改放候選：r＝s＋k(1−w_{t−1})(c−b)−kκ|Δw|×[換股執行日]。"""
    nav = sd['nav_blk5pct' if blk else 'nav']; w = sd['stockfrac_blk5pct' if blk else 'stockfrac']
    rs = nav.pct_change(); rb = b_nav.pct_change(); rc = c_nav.pct_change().fillna(0)
    wl = w.shift(1); dw = (w - wl).abs() * sd['exec_day']
    r = rs + k * (1 - wl) * (rc - rb) - k * kappa * dw
    r = r.loc[start:]; r.iloc[0] = 0.0
    return (1 + r).cumprod()


def nw_t(x, lag=6):
    x = np.asarray(x, float); n = len(x); m = x.mean(); e = x - m
    v = (e @ e) / n
    for L in range(1, min(lag, n - 1) + 1):
        v += 2 * (1 - L / (lag + 1)) * (e[L:] @ e[:-L]) / n
    return float(m / np.sqrt(v / n)) if v > 0 else float('nan')


def mret(nav):
    m = nav.dropna().groupby(nav.dropna().index.to_period('M')).last()
    return m.pct_change().dropna()


def stats(nav, bench, th, win):
    """段內指標。nav、bench 為逐日淨值。"""
    a, b = pd.Period(win[0], 'M'), pd.Period(win[1], 'M')
    r = mret(nav); rb = mret(bench)
    r = r.loc[a:b]; ex = (r - rb.reindex(r.index))
    n = len(r)
    d = nav.dropna(); d0 = d.loc[:str((a - 1).end_time.date())].index[-1]; d1 = d.loc[:str(b.end_time.date())].index[-1]
    seg = d.loc[d0:d1]
    cold = ex[th.reindex(ex.index) == '冷']
    return dict(n_months=int(n), cagr=float((seg.iloc[-1] / seg.iloc[0]) ** (12 / n) - 1), maxdd=float((seg / seg.cummax() - 1).min()),
                excess=float(ex.mean() * 12), t=nw_t(ex), IR=float(ex.mean() / ex.std(ddof=1) * np.sqrt(12)),
                cold_excess=float(cold.mean() * 12), n_cold=int(len(cold)))


def passes(c, cur):
    """事前寫死的判定：年化不低於現行 2 個百分點以上，且（回撤淺 ≥3 或 冷月超額改善 ≥5 或 IR 提高 ≥0.1）。"""
    ret_ok = c['cagr'] >= cur['cagr'] - 0.02
    dd = c['maxdd'] - cur['maxdd'] >= 0.03
    cold = c['cold_excess'] - cur['cold_excess'] >= 0.05
    ir = c['IR'] - cur['IR'] >= 0.1
    return dict(ret_ok=bool(ret_ok), dd_ok=bool(dd), cold_ok=bool(cold), ir_ok=bool(ir), passed=bool(ret_ok and (dd or cold or ir)))


def all_cells(sd, bench, C, names, kappa=KAPPA, blk=False):
    out = {}
    s_nav = sd['nav_blk5pct' if blk else 'nav']
    for nm in names:
        for st, p in STRUCT:
            key = f'{nm}|{st}{int(p*100)}'
            out[key] = blend_A(s_nav, C[nm], p, kappa) if st == 'A' else blend_B(sd, bench, C[nm], p, kappa, blk=blk)
    return out
