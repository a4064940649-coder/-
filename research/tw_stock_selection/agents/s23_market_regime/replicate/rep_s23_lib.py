#!/usr/bin/env python3
"""S23 市場環境減碼：獨立複製用函式庫（複製員自寫）。
只依 s23 的 PREREG.json 規則文字與 FINDINGS.md 條件；未讀、未 import、未執行 s23lib.py／train.py／validate.py／posthoc.py。
引擎沿用研究中心共用：s20lib → s17lib → s10lib（make_signal_s10、simulate_s10、variant_S、e50_fix_suspension、seg_dd、top5_share）。
中間檔放 scratchpad/s23rep/（結束刪除）。一律 python3 -I -B。"""
import sys, math
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
AG = HERE.parent.parent
REP = Path('/tmp/claude-0/-home-user--/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/s23rep')
sys.path.insert(0, str(AG / 's20_persistent_growth'))
import s20lib                                              # noqa: E402
from s20lib import s17lib, s10lib, s9lib, s8lib, s7lib, s6lib, s5lib, s4lib, s3lib, s2lib, stlib, cslib, COSTS   # noqa: E402,F401
s20lib.S20 = REP; s17lib.S17 = REP; s10lib.S10 = REP; s9lib.S9 = REP; s8lib.S8 = REP; s7lib.S7 = REP; s6lib.S6 = REP; s5lib.S5 = REP; s4lib.S4 = REP
s3lib.S3 = REP; s2lib.S2 = REP; stlib.S1 = REP; cslib.TMP = REP
TRAIN_LAST = pd.Period('2021-11', 'M')
VALID_FIRST = pd.Period('2021-12', 'M')

# PREREG 登記的 7 條規則（門檻照 PREREG 寫死的數值；第 0 步另由本程式自行重算核對）
RULES = {
    'BRD_STOP_Q20': dict(var='BRD3', q=0.2, action='stop'),
    'BRD_STOP_Q33': dict(var='BRD3', q=1 / 3, action='stop'),
    'LMA_STOP_Q20': dict(var='LMA', q=0.2, action='stop'),
    'LMA_STOP_Q33': dict(var='LMA', q=1 / 3, action='stop'),
    'MA_HALF': dict(var='MA200', q=None, action='half'),
    'MA_STOP': dict(var='MA200', q=None, action='stop'),
    'MA_EXIT': dict(var='MA200', q=None, action='exit'),
}
PREREG_THR = {'BRD_STOP_Q20': 0.04310124711113451, 'BRD_STOP_Q33': 0.05354530121090945, 'LMA_STOP_Q20': 11.0, 'LMA_STOP_Q33': 14.0}


def load():
    REP.mkdir(parents=True, exist_ok=True)
    return s20lib.load()


def regime_rows(P, U, RT, shift, brd3_minp=1):
    """自寫訊號迴圈（同 make_signal_s10 的列選取與 F1 定義），每個訊號列算：
    BRD＝股池 U[k] 且 SURGE 有值者中 SURGE ≥ 2 的比例；BRD3＝本列與前 2 個訊號列 BRD 的平均（不足 3 列時用現有列，brd3_minp=1）；
    LMA＝股池中 F_MA60[k] 且 SURGE ≥ 2 的家數；MA200＝E50[k] < 含當日過去 200 個交易日 E50 平均。
    回傳 DataFrame（month、k、BRD、BRD3、LMA、MA200、f1（依 SURGE 排序的欄位索引 list））。"""
    S, av = RT['S'], RT['av']
    E50 = np.asarray(P.E50, float)
    ma200 = pd.Series(E50).rolling(200, min_periods=200).mean().to_numpy()
    rows = []
    for m in S.index:
        k0 = av[m]
        if k0 >= P.T or k0 < 61:
            continue
        s = S.loc[m].to_numpy()
        if int((U[k0] & np.isfinite(s)).sum()) < 50:
            continue
        k = k0 + shift
        if k >= P.T:
            continue
        fin = U[k] & np.isfinite(s)
        hi = fin & (np.nan_to_num(s, nan=-np.inf) >= 2.0)
        ok_ma = fin & P.F_MA60[k]
        o_ma = np.flatnonzero(ok_ma)[np.argsort(-s[ok_ma], kind='stable')]
        r20 = P.RET20[k]
        pool = o_ma[:10]
        f1 = pool[r20[pool] > 0.10][:10]
        rows.append(dict(month=m, k=int(k), BRD=hi.sum() / fin.sum(), LMA=int((hi & P.F_MA60[k]).sum()),
                         MA200=bool(np.isfinite(ma200[k]) and E50[k] < ma200[k]), f1=[int(j) for j in f1]))
    df = pd.DataFrame(rows)
    df['BRD3'] = df['BRD'].rolling(3, min_periods=brd3_minp).mean()
    return df


def triggered(df, rule, thr=None):
    """每列是否觸發（bool 陣列）。BRD／LMA：變數 < 門檻；MA：MA200 為真。"""
    r = RULES[rule]
    if r['var'] == 'MA200':
        return df['MA200'].to_numpy(bool)
    t = PREREG_THR[rule] if thr is None else thr
    return (df[r['var']].to_numpy(float) < t)


def apply_rule(P, S0, df, trig, action):
    """在 make_signal_s10 的 S0 上改 sig／rank（與 X）：stop＝觸發列整列不買；half＝只留 F1 前 ceil(n/2) 檔；
    exit＝不買，且觸發列 X 全部 True（所有衛星持股標記賣出，次日開盤賣）。回傳新的 S（不改 S0），與每列少買檔數。"""
    S2 = dict(S0)
    sig = S0['sig'].copy(); rank = S0['rank'].copy()
    X = S0['X'].copy() if action == 'exit' else S0['X']
    dropped = 0
    for (k, f1), tg in zip(zip(df['k'], df['f1']), trig):
        if not tg:
            continue
        drop = f1[int(math.ceil(len(f1) / 2)):] if action == 'half' else f1
        dropped += len(drop)
        if drop:
            sig[k, drop] = False; rank[k, drop] = np.nan
        if action == 'exit':
            X[k, :] = True
    S2['sig'] = sig; S2['rank'] = rank; S2['X'] = X
    return S2, dropped


def fresh(S, first=VALID_FIRST):
    """只保留營收月份 ≥ first 的 sig（重新起跑；出場 X 不動）。"""
    S2 = dict(S); sig = S['sig'].copy(); rank = S['rank'].copy()
    for x in S['info']:
        if pd.Period(x['month'], 'M') < first:
            sig[x['k'], :] = False; rank[x['k'], :] = np.nan
    S2['sig'] = sig; S2['rank'] = rank
    return S2
