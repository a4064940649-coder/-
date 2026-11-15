#!/usr/bin/env python3
"""S12 共用程式庫：月營收驚奇 SURGE 在不同市值層的表現。
在 s10lib（不改）之上：面板、創新板、SURGE 訊號、F1 進場、E0 出場、引擎全部沿用（s10lib.simulate_s10，不開任何出場掛鉤＝s9lib.simulate_s9 open）。
本檔只新增：PIT 市值（o_lc_sector_rotation_pit/data/shares_full.parquet 股數＝o8 shares_pit＋4763／2317 修正，落後 63 個交易日）與分層、分層版訊號（make_signal_tier）、同層等權對照、回撤分段統計。
中間檔放 scratchpad/s12tmp/。規格見 PREREG.json。一律 python3 -I -B。"""
import sys
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
AG = HERE.parent
S12 = Path('/tmp/claude-0/-home-user--/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/s12tmp')
sys.path.insert(0, str(AG / 's10_exit_details'))
import s10lib                                              # noqa: E402
from s10lib import s9lib, s8lib, s7lib, s6lib, s5lib, s4lib, s3lib, s2lib, stlib, cslib, COSTS   # noqa: E402,F401
s10lib.S10 = S12; s9lib.S9 = S12; s8lib.S8 = S12; s7lib.S7 = S12; s6lib.S6 = S12; s5lib.S5 = S12; s4lib.S4 = S12; s3lib.S3 = S12; s2lib.S2 = S12; stlib.S1 = S12; cslib.TMP = S12
SHARES = AG / 'o_lc_sector_rotation_pit' / 'data' / 'shares_full.parquet'   # o8 shares_pit（2013 起）＋o9 複製員的 4763／2317 修正（o12 lclib）
SEGMENTS = {'2018': ('2018-01-01', '2018-12-31'), '2020-03': ('2020-01-01', '2020-06-30'),
            '2022': ('2022-01-01', '2022-12-31'), '2024-07~2025-04': ('2024-07-01', '2025-04-30')}


def load():
    S12.mkdir(parents=True, exist_ok=True)
    return s10lib.load()
LAG = 63                                                   # 股數落後 63 個交易日（o8 統計審查的做法：避開季內內插用到下一季股本的前視）
TIER_EDGES = (100, 300)                                    # 大＝前 100 名、中＝101～300、小＝其餘（股池內依 PIT 市值排序）
TIER_NAMES = ('L', 'M', 'S')


def load_shares(P):
    """shares_full.parquet → 面板形狀（列＝P.IDX，欄＝P.COLS；日期以前值補到面板交易日）。沒有股數的欄位為 NaN。"""
    sh = pd.read_parquet(SHARES)
    sh = sh.reindex(sh.index.union(P.IDX)).ffill().reindex(P.IDX)
    sh = sh.reindex(columns=P.COLS).to_numpy(np.float64)
    return sh


def mcap_panel(P, SH, lag=LAG):
    """PIT 市值代理：MC[t]＝股數[t−lag] × 收盤[t−lag] × G[t] ÷ G[t−lag]（落後 lag 日的市值，乘上之後的含息漲跌，避免除權、減資造成股數與價格錯配）。
    落後值缺（新上市不滿 lag 日、股數檔沒有）→ 用當日股數 × 當日收盤（fallback，計數）。回傳 (MC, src)：src 0＝落後值、1＝當日值、2＝都沒有。"""
    now = SH * P.CF
    MC = np.full((P.T, P.N), np.nan)
    if lag > 0:
        MC[lag:] = now[:-lag] * (P.G[lag:] / P.G[:-lag])
    else:
        MC[:] = now
    src = np.where(np.isfinite(MC), 0, np.where(np.isfinite(now), 1, 2)).astype(np.int8)
    MC = np.where(np.isfinite(MC), MC, now)
    return MC, src


def tiers_at(univ_row, val_row, edges=TIER_EDGES):
    """股池內依 val 由大到小排名：0＝前 edges[0]、1＝到 edges[1]、2＝其餘；val 缺值排在最後（歸小型層）。股池外 −1。"""
    out = np.full(len(univ_row), -1, np.int8)
    idx = np.flatnonzero(univ_row)
    v = np.nan_to_num(val_row[idx], nan=-np.inf)
    order = idx[np.argsort(-v, kind='stable')]
    out[order[:edges[0]]] = 0; out[order[edges[0]:edges[1]]] = 1; out[order[edges[1]:]] = 2
    return out


def make_signal_tier(P, univ, RT, VAL, tset, shift=0, depth=10, mom_lo=0.10, nsel=10, nexit=20):
    """分層版 F1＋E0（PREREG design.signal）。月份是否有訊號的條件與 s4lib／s10lib 完全相同（未位移訊號日全股池 SURGE 有值 ≥ 50 檔）。
    訊號日 k（含位移）：tier＝tiers_at(univ[k], VAL[k])；層內 L_MA＝股池內、層在 tset、SURGE 有值、還原收盤 > 60 日均線，依 SURGE 由大到小；
    買進名單＝L_MA 前 depth 名中 RET20 > mom_lo 的（最多 nsel 檔）；
    出場（E0 層內版）：續抱＝股池內、SURGE 有值、還原收盤 > 60 日均線、SURGE ≥ 層內 L_MA 第 nexit 名的分數（不足 nexit 檔不設門檻）。
    對層內股票，這等於「在層內 L_MA 前 20 名」（驗收 A2 計數不同格數）；對因市值變動跑到別層的持股，等於「把它放進原層一起排名仍在前 20」，
    所以不會因為漲大跳層就被賣掉。tset＝None 表示全股池（＝現行 E0，驗收 A1）。
    回傳格式同 s4lib.make_signal：sig、rank、X、info（另含每月層內 L_MA 檔數與名單）。"""
    S, av = RT['S'], RT['av']
    sig = np.zeros((P.T, P.N), bool); rank = np.full((P.T, P.N), np.nan); X = np.zeros((P.T, P.N), bool)
    info = []; a2 = 0; TIER = {}
    for m in S.index:
        k0 = av[m]
        if k0 >= P.T or k0 < 61:
            continue
        s = S.loc[m].to_numpy()
        if int((univ[k0] & np.isfinite(s)).sum()) < 50:
            continue
        k = k0 + shift
        if k >= P.T:
            continue
        fin = univ[k] & np.isfinite(s)
        tr = tiers_at(univ[k], VAL[k]) if VAL is not None else np.zeros(P.N, np.int8)
        TIER[k] = tr
        intier = np.isin(tr, list(tset)) if tset is not None else (tr >= 0)
        ok_ma = fin & P.F_MA60[k] & intier
        o_ma = np.flatnonzero(ok_ma)[np.argsort(-s[ok_ma], kind='stable')]
        pool = o_ma[:depth]
        okr = P.RET20[k][pool] > mom_lo
        tgt = pool[okr][:nsel]
        sig[k, tgt] = True; rank[k, tgt] = s[tgt]
        thr = s[o_ma[nexit - 1]] if len(o_ma) >= nexit else -np.inf
        keep = fin & P.F_MA60[k] & (np.nan_to_num(s, nan=-np.inf) >= thr)
        ref = np.zeros(P.N, bool); ref[o_ma[:nexit]] = True
        a2 += int(((keep & intier) != ref).sum())
        X[k, :] = ~keep
        info.append(dict(month=str(m), k=int(k), sig_date=str(P.IDX[k].date()), n_ma=int(ok_ma.sum()), n_tgt=int(len(tgt)),
                         n_tier=int((univ[k] & intier).sum()), targets=[str(P.COLS[j]) for j in tgt]))
    sigrows = np.zeros(P.T, bool); sigrows[[x['k'] for x in info]] = True
    return dict(sig=sig, rank=rank, X=X, SD=None, max_hold=None, info=info, sigrows=sigrows, RQ={}, A2_mismatch=a2, TIER=TIER)


def as_w5(S):
    """配額格的子帳戶（5 檔、每檔＝子帳戶衛星 ÷ 5）：simulate_s10 的 sizing='WN' 用 max(當月名單檔數, 5) 當除數，
    把 info 的 n_tgt 設成 0 → 除數恆為 5（n_tgt 只用在這個除數；驗收 A3 用 n_tgt＝10、slots＝10 對 W10 逐筆相同驗證）。"""
    S2 = dict(S); S2['info'] = [dict(x, n_tgt=0) for x in S['info']]
    return S2


def seg_dd(nav):
    """四段空頭各自的最大跌幅（高點與低點都在段內；同 s11lib.seg_dd）。"""
    out = {}
    for k, (a, b) in SEGMENTS.items():
        s = nav.loc[a:b].dropna()
        out[k] = float((s / s.cummax() - 1).min()) if len(s) > 2 else float('nan')
    return out


def top5_share(lg, nav, cap):
    """最賺 5 筆（已平倉）損益合計 ÷ 淨值增加（同 s11 run.top5_share）。"""
    x = lg[lg['why'] != 'open_end'].sort_values('pnl', ascending=False).head(5)
    return float(x['pnl'].sum() / (nav.iloc[-1] - cap))
