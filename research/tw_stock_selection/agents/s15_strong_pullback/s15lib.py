#!/usr/bin/env python3
"""S15 共用程式庫：強勢股拉回買進（新訊號，純價量），單獨衛星 P 與和 SURGE 各半合併 C。
在 s11lib／s10lib（都不改）之上：面板、創新板、SURGE 訊號（S10 E0）、引擎（s10lib.simulate_s10，不開任何出場掛鉤）、回撤分段與相關係數全部沿用；
本檔只新增：拉回訊號特徵（pb_features）與事件訊號 make_signal_pb（固定持有 max_hold 個交易日）。
中間檔放 scratchpad/s15tmp/。規格見 PREREG.json。一律 python3 -I -B。"""
import sys, math
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
AG = HERE.parent
S15 = Path('/tmp/claude-0/-home-user--/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/s15tmp')
sys.path.insert(0, str(AG / 's11_pead_second_signal'))
import s11lib                                              # noqa: E402
from s11lib import s10lib, s9lib, s8lib, s7lib, s6lib, s5lib, s4lib, s3lib, s2lib, stlib, cslib, COSTS   # noqa: E402,F401
s11lib.S11 = S15; s10lib.S10 = S15; s9lib.S9 = S15; s8lib.S8 = S15; s7lib.S7 = S15; s6lib.S6 = S15; s5lib.S5 = S15
s4lib.S4 = S15; s3lib.S3 = S15; s2lib.S2 = S15; stlib.S1 = S15; cslib.TMP = S15   # 快取改放本題 scratchpad
SEGMENTS = s11lib.SEGMENTS
seg_dd, month_excess, corr_excess, surge_signal = s11lib.seg_dd, s11lib.month_excess, s11lib.corr_excess, s11lib.surge_signal

# ---------------------------------------------------------------- 事前寫死的參數（PREREG.json design.signal）
RS_DAYS = 120          # 相對強度：還原收盤 120 個交易日報酬 − 0050 含息 120 日報酬
RS_TOP = 0.20          # 強勢股池：當日股池內相對強度前 20%（名次 ≤ ceil(0.2 × 有值檔數)）
NEARHI = 0.85          # 收盤距 52 週高 ≤ 15%：還原收盤 ÷ 240 日還原最高價 ≥ 0.85（上市滿 240 根）
PB_LO, PB_HI = 0.97, 1.02   # 拉回：前一日還原收盤 ÷ 前一日 MA20 在 [0.97, 1.02]
VS, VL = 5, 20         # 量縮：前一日的近 5 日平均量 < 近 20 日平均量（股數；無成交日量記 0）
HOLD = 20              # 固定持有 20 個交易日（主格）；敏感度格 40


def load():
    S15.mkdir(parents=True, exist_ok=True)
    return s10lib.load()


def pb_features(P, U):
    """回傳 dict：sig0（T×N 布林，未位移訊號列＝訊號日 t，t 收盤後判斷、t＋1 開盤買）、rs（T×N 相對強度，同日排序用）、各條件的布林（data_check 用）。
    t 日條件（全部成立）：
      (1) 股池 U[t]（只上市、60 日均成交值 ≥ 1,000 萬、收盤 ≥ 10、有成交、非 91xx、非創新板）；
      (2) 強勢：rs[t] 在當日 U[t] 且 rs 有值的股票中名次 ≤ ceil(0.2 × 檔數)（由大到小）；
      (3) 接近新高：G[t] ÷ HI240[t] ≥ 0.85（HI240＝240 日還原最高價，上市滿 240 根）；
      (4) 拉回（t−1）：0.97 ≤ G[t−1] ÷ MA20[t−1] ≤ 1.02（MA20＝還原收盤 20 日均線）；
      (5) 量縮（t−1）：mean(量, t−5..t−1) < mean(量, t−20..t−1)；
      (6) 轉強（t）：還原收盤 G[t] > 前一日還原最高價 adjH[t−1]（＝收盤站上前一日高點，除權息已還原）。"""
    G = P.G
    E = P.E50
    with np.errstate(invalid='ignore', divide='ignore'):
        r_s = G / cslib.shift(G, RS_DAYS) - 1
        r_b = E / np.concatenate([np.full(RS_DAYS, np.nan), E[:-RS_DAYS]]) - 1
        rs = r_s - r_b[:, None]
    del r_s
    ok = U & np.isfinite(rs)
    strong = np.zeros_like(U)
    for t in range(P.T):
        idx = np.flatnonzero(ok[t])
        if len(idx) == 0:
            continue
        n = int(math.ceil(RS_TOP * len(idx)))
        o = idx[np.argsort(-rs[t, idx], kind='stable')][:n]
        strong[t, o] = True
    with np.errstate(invalid='ignore', divide='ignore'):
        near = np.nan_to_num(G / P.HI240, nan=0.0) >= NEARHI
        ma20 = stlib.rmean(G, 20)
        x = G / ma20
        pb = (x >= PB_LO) & (x <= PB_HI)
        del x, ma20
        v5 = stlib.rmean(P.VOLF, VS); v20 = stlib.rmean(P.VOLF, VL)
        vdry = v5 < v20
        del v5, v20
        adjH = P.HF * G / P.CF
        turn = G > cslib.shift(adjH, 1)
        del adjH
    pb1 = np.zeros_like(pb); pb1[1:] = pb[:-1]
    vd1 = np.zeros_like(vdry); vd1[1:] = vdry[:-1]
    sig0 = U & strong & near & pb1 & vd1 & turn & P.has
    return dict(sig0=sig0, rs=rs, strong=strong, near=near, pb1=pb1, vd1=vd1, turn=turn)


def make_signal_pb(P, univ, F, shift=0, hold=HOLD, keep_info=False):
    """拉回事件訊號（格式同 s4lib.make_signal，給 s10lib.simulate_s10 用）。
    位移 s：未位移訊號列 k 的訊號放到 k＋s（買在 k＋s＋1 開盤）；可買另要 univ[k＋s]（引擎檢查）；排序分數＝訊號日 k 的相對強度。
    出場：X 全為 False；max_hold＝hold → 持有滿 hold 個交易日（進場日開盤買，第 hold 個交易日後的開盤賣）。持有中同檔新訊號不加碼（引擎本來就不重複買）。"""
    T, N = P.T, P.N
    sig0, rs = F['sig0'], F['rs']
    sig = np.zeros((T, N), bool); rank = np.full((T, N), np.nan)
    if shift == 0:
        sig[:] = sig0; rank[sig0] = rs[sig0]
    else:
        sig[shift:] = sig0[:-shift]
        rank[shift:][sig0[:-shift]] = rs[:-shift][sig0[:-shift]]
    sig[-1] = False
    rows = np.flatnonzero(sig.any(axis=1))
    info = [dict(k=int(k), sig_date=str(P.IDX[k].date()), n_tgt=int(sig[k].sum()), **({'targets': [str(c) for c in P.COLS[sig[k]]]} if keep_info else {})) for k in rows]
    sigrows = np.zeros(T, bool); sigrows[rows] = True
    return dict(sig=sig, rank=rank, X=np.zeros((T, N), bool), SD=None, max_hold=int(hold), info=info, sigrows=sigrows, RQ={})
