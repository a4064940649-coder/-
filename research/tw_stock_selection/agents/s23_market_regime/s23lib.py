#!/usr/bin/env python3
"""S23 共用程式庫：市場環境減碼——市場轉弱時把衛星新買改放 0050（或清出衛星），能不能降低回撤而不犧牲太多報酬？
在 s20lib（不改；其下 s17lib→s10lib：面板、創新板、SURGE（STD，σ>0）、F1、E0、引擎 simulate_s10）之上，本檔只新增：
- regime_features()：每個訊號列 k（訊號日收盤可得）算三個環境變數：
    BRD  ＝股池 U[k] 且 SURGE 有值的股票中 SURGE ≥ 2 的比例；BRD3＝本月與前 2 個訊號列 BRD 的平均（平滑農曆年基期跳動）。
    LMA  ＝股池 U[k]、SURGE 有值、還原收盤 > 60 日均線（F_MA60）且 SURGE ≥ 2 的家數。
    MA200＝0050 含息還原收盤 E50[k] < 過去 200 個交易日（含當日）E50 平均 → True（弱）。
- apply_regime()：照 s10lib.make_signal_s10 的同一個訊號列，依規則只改「當月買進名單 sig」或（只有 MA_EXIT）在訊號列加賣出旗標；
  選股排序、F1 定義、E0 出場、每檔＝衛星 ÷ 10 全部不動。沒買的錢照引擎掃進 0050。
中間檔放 scratchpad/s23/。規格見 PREREG.json。一律 python3 -I -B。"""
import sys, math
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
AG = HERE.parent
S23 = Path('/tmp/claude-0/-home-user--/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/s23')
sys.path.insert(0, str(AG / 's20_persistent_growth'))
import s20lib                                              # noqa: E402
from s20lib import s17lib, s10lib, s9lib, s8lib, s7lib, s6lib, s5lib, s4lib, s3lib, s2lib, stlib, cslib, COSTS   # noqa: E402,F401
s20lib.S20 = S23; s17lib.S17 = S23; s10lib.S10 = S23; s9lib.S9 = S23; s8lib.S8 = S23; s7lib.S7 = S23; s6lib.S6 = S23; s5lib.S5 = S23; s4lib.S4 = S23
s3lib.S3 = S23; s2lib.S2 = S23; stlib.S1 = S23; cslib.TMP = S23

TRAIN_LAST_MONTH = s20lib.TRAIN_LAST_MONTH                 # 營收月份 ≤ 2021-11（訊號日 2016-02～2021-12）
VALID_FIRST_MONTH = s20lib.VALID_FIRST_MONTH               # 營收月份 ≥ 2021-12（訊號日 2022-01 起）
SEGMENTS = s17lib.SEGMENTS
fresh_from = s20lib.fresh_from
SURGE_HI = 2.0
MA_N = 200

# 事前登記的規則族（門檻 q＝訓練期訊號列（位移 0）該變數分布的分位數；數值在 prep.py 算好寫進 PREREG）
RULES = [
    dict(id='BRD_STOP_Q20', var='BRD3', q=0.20, action='stop',
         desc='全市場 SURGE 廣度（股池中 SURGE ≥ 2 比例，3 個月平均）低於訓練期第 20 百分位 → 這個月不新買，空位放 0050；既有持股照 E0'),
    dict(id='BRD_STOP_Q33', var='BRD3', q=1 / 3, action='stop',
         desc='同上，門檻＝訓練期第 33 百分位'),
    dict(id='LMA_STOP_Q20', var='LMA', q=0.20, action='stop',
         desc='L_MA 家數（股池中站上季線且 SURGE ≥ 2）低於訓練期第 20 百分位 → 這個月不新買'),
    dict(id='LMA_STOP_Q33', var='LMA', q=1 / 3, action='stop',
         desc='同上，門檻＝訓練期第 33 百分位'),
    dict(id='MA_HALF', var='MA200', q=None, action='half',
         desc='0050 含息收盤 < 200 日均線 → 衛星新買減半：當月 F1 名單只買依 SURGE 排序前 ceil(n/2) 檔，其餘空位放 0050'),
    dict(id='MA_STOP', var='MA200', q=None, action='stop',
         desc='0050 含息收盤 < 200 日均線 → 這個月不新買'),
    dict(id='MA_EXIT', var='MA200', q=None, action='exit',
         desc='0050 含息收盤 < 200 日均線 → 不新買，且訊號日把全部衛星持股標記賣出（次日開盤賣，錢進 0050）；均線回到上方後的換股日照 F1 重新買'),
]


def load():
    S23.mkdir(parents=True, exist_ok=True)
    return s20lib.load()


def surge(P):
    F = s17lib.rev_family(P, defs=['STD'])
    return dict(S=F['S']['STD'], av=F['av'])


def regime_features(P, U, RT, S0):
    """每個訊號列（S0['info'] 的順序）一列：month, k, BRD, BRD3, LMA, MA200weak, e50, ma200, n_tgt。"""
    S = RT['S']; rows = []
    E = np.asarray(P.E50, float)
    for x in S0['info']:
        m = pd.Period(x['month'], 'M'); k = x['k']
        s = S.loc[m].to_numpy()
        fin = U[k] & np.isfinite(s)
        hi = fin & (np.nan_to_num(s, nan=-np.inf) >= SURGE_HI)
        brd = float(hi.sum() / max(fin.sum(), 1))
        lma = int((hi & P.F_MA60[k]).sum())
        w = E[max(0, k - MA_N + 1):k + 1]
        ma = float(np.nanmean(w)) if np.isfinite(w).sum() >= MA_N else float('nan')
        weak = bool(np.isfinite(ma) and E[k] < ma)
        rows.append(dict(month=x['month'], k=int(k), sig_date=x['sig_date'], BRD=brd, LMA=lma, e50=float(E[k]), ma200=ma, MA200=weak, n_tgt=int(x['n_tgt'])))
    df = pd.DataFrame(rows)
    df['BRD3'] = df['BRD'].rolling(3, min_periods=1).mean()
    return df


def triggered(F, rule, thr):
    """回傳布林 Series（index 同 F）：該訊號列是否處於弱勢（規則觸發）。thr＝門檻數值（MA200 規則不用）。"""
    if rule['var'] == 'MA200':
        return F['MA200'].astype(bool)
    return F[rule['var']] < thr


def apply_regime(S0, F, trig, action):
    """S0＝make_signal_s10 的結果；F＝regime_features；trig＝布林陣列（與 F 同序）；action＝'stop'／'half'／'exit'。
    回傳 S1（sig／rank／X 可能改；X 只有 exit 會改）與 log。"""
    sig = S0['sig'].copy(); rank = S0['rank'].copy(); X = S0['X']
    if action == 'exit':
        X = X.copy()
    log = []
    for (i, x), tg in zip(enumerate(S0['info']), trig):
        if not tg:
            continue
        k = x['k']
        tg_idx = np.flatnonzero(sig[k])
        if action in ('stop', 'exit'):
            keep = []
        elif action == 'half':
            order = tg_idx[np.argsort(-rank[k, tg_idx], kind='stable')]
            keep = list(order[:int(math.ceil(len(order) / 2))])
        else:
            raise ValueError(action)
        drop = [j for j in tg_idx if j not in set(keep)]
        sig[k, drop] = False; rank[k, drop] = np.nan
        if action == 'exit':
            X[k, :] = True
        log.append(dict(month=x['month'], k=int(k), n_orig=int(len(tg_idx)), n_new=int(len(keep))))
    S1 = dict(S0); S1['sig'] = sig; S1['rank'] = rank; S1['X'] = X
    if 'XV' in S1:
        XV = dict(S1['XV']); XV['E0'] = X; S1['XV'] = XV
    return S1, log


def random_trig(n_rows, eligible, n_on, rng):
    """隨機對照：在 eligible（布林）列中隨機挑 n_on 列觸發。"""
    idx = np.flatnonzero(eligible)
    out = np.zeros(n_rows, bool)
    if n_on > 0 and len(idx):
        out[rng.choice(idx, size=min(n_on, len(idx)), replace=False)] = True
    return out
