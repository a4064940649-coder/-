#!/usr/bin/env python3
"""S20 共用程式庫：「持續成長」篩選（μ÷σ）——訓練期選定、驗證期只看一次。
在 s17lib（不改；含 σ > 0 條件與 0050 停牌修正）之上：面板、創新板、SURGE（STD）、F1、E0、引擎全部沿用 s10lib.make_signal_s10／simulate_s10；
本檔只新增：
- growth_features()：從與 SURGE 同一份 D＝營收(m)−營收(m−12) 算 μ、σ（D(m−1…m−24)，逐窗兩段式，同 s17lib.zscore）、MS＝μ÷σ、MU＝μ>0、NPOS12＝D(m−1…m−12) > 0 的月數；
- apply_rule()：在 make_signal_s10 的 F1 名單裡，把不符合規則的 sig 拿掉（不遞補、不改出場、不改名額；空出的錢照引擎掃進 0050）；
- fresh_from()：只保留訊號月份 ≥ 某月的 sig（驗證期從 2022-01 重新起跑）；
- outcomes()、month_cluster_diff()：逐字複製 s13lib（E0 持有期超額，事後描述用）。
中間檔放 scratchpad/s20tmp/。規格見 PREREG.json。一律 python3 -I -B。"""
import sys
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
AG = HERE.parent
S20 = Path('/tmp/claude-0/-home-user--/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/s20tmp')
sys.path.insert(0, str(AG / 's17_surge_definition_family'))
import s17lib                                              # noqa: E402
from s17lib import s10lib, s9lib, s8lib, s7lib, s6lib, s5lib, s4lib, s3lib, s2lib, stlib, cslib, COSTS   # noqa: E402,F401
s17lib.S17 = S20; s10lib.S10 = S20; s9lib.S9 = S20; s8lib.S8 = S20; s7lib.S7 = S20; s6lib.S6 = S20; s5lib.S5 = S20; s4lib.S4 = S20
s3lib.S3 = S20; s2lib.S2 = S20; stlib.S1 = S20; cslib.TMP = S20

SEGMENTS = s17lib.SEGMENTS
seg_dd = s17lib.seg_dd
top5_share = s17lib.top5_share
TRAIN_LAST_MONTH = pd.Period('2021-11', 'M')               # 訓練期：營收月份 ≤ 2021-11（訊號日 2016-02～2021-12）
VALID_FIRST_MONTH = pd.Period('2021-12', 'M')              # 驗證期：營收月份 ≥ 2021-12（訊號日 2022-01 起）
BIG = ['2408', '3653', '1325', '2603']


def load():
    S20.mkdir(parents=True, exist_ok=True)
    return s17lib.load()


def growth_features(P):
    """回傳 dict(S＝STD SURGE（s17lib.rev_family，σ>0）, av, MS, MU, SIG, NPOS12)；全部 index＝營收月份 Period、columns＝P.COLS。
    μ、σ：D(m−1…m−24) 全部有值才算，兩段式樣本標準差（同 s17lib.zscore center='mean'）；σ ≤ 1e-9 ×（窗內 |D| 平均＋1e-12）當缺值。
    MS＝μ÷σ；MU＝1.0（μ>0）／0.0（μ≤0）；NPOS12＝D(m−1…m−12) 中 > 0 的個數（12 個月都要有值）。只在 SURGE 有值的格子給值（其餘 NaN）。"""
    F = s17lib.rev_family(P, defs=['STD'])
    Rv = F['Rv']
    D = Rv - Rv.shift(12)
    D[D.index.year == 2013] = np.nan
    A = D.to_numpy(float)
    MU = np.full(A.shape, np.nan); SG = np.full(A.shape, np.nan); NP = np.full(A.shape, np.nan)
    for i0 in range(0, A.shape[1], 400):
        a = A[:, i0:i0 + 400]
        V = s17lib._windows(a, 24)
        full = np.isfinite(V).all(axis=2)
        with np.errstate(invalid='ignore'):
            mu = V.mean(axis=2)
            s = np.sqrt(((V - mu[:, :, None]) ** 2).sum(axis=2) / 23)
            scale = np.abs(V).mean(axis=2)
            pos = full & np.isfinite(s) & (s > 1e-9 * (scale + 1e-12))
            MU[:, i0:i0 + 400] = np.where(pos, mu, np.nan)
            SG[:, i0:i0 + 400] = np.where(pos, s, np.nan)
            V12 = V[:, :, 12:]                               # V[m] 的最後 12 個＝D(m−12 … m−1)
            f12 = np.isfinite(V12).all(axis=2)
            NP[:, i0:i0 + 400] = np.where(f12, (V12 > 0).sum(axis=2), np.nan)
        del V
    Sv = F['S']['STD'].to_numpy()
    ok = np.isfinite(Sv)
    mk = lambda x: pd.DataFrame(np.where(ok, x, np.nan), index=D.index, columns=D.columns)
    MSv = MU / SG
    return dict(S=F['S']['STD'], av=F['av'], MS=mk(MSv), MU=mk(np.where(np.isfinite(MU), (MU > 0).astype(float), np.nan)), SIG=mk(SG), MUAMT=mk(MU), NPOS12=mk(NP),
                n_sigma0=F['n_sigma0'])


def f1_table(P, S, G):
    """make_signal_s10 的 info → 每個 F1 目標一列：month、k、j、code、rank（L_MA 名次由 SURGE 排序；這裡只記 SURGE）、MS、MU、NPOS12。"""
    rows = []
    for x in S['info']:
        m = pd.Period(x['month'], 'M'); k = x['k']
        for c in x['targets']:
            j = P.ci[c]
            rows.append((x['month'], k, j, c, float(S['rank'][k, j]), float(G['MS'].at[m, c]), float(G['MU'].at[m, c]), float(G['NPOS12'].at[m, c]),
                         float(G['MUAMT'].at[m, c]), float(G['SIG'].at[m, c])))
    df = pd.DataFrame(rows, columns=['month', 'k', 'j', 'code', 'surge', 'MS', 'MU', 'NPOS12', 'mu', 'sigma'])
    df['sig_date'] = P.IDX[df['k'].to_numpy()] if len(df) else []
    return df


def rule_keep_value(G, feat, op, thr, month, code):
    v = G[feat].at[month, code]
    if not np.isfinite(v):
        return False
    return bool(v >= thr) if op == '>=' else (bool(v > thr) if op == '>' else bool(v < thr))


def apply_rule(P, S, G, rule):
    """rule＝None（現行）或 dict(feat, op, thr)。回傳 S 的淺拷貝，sig／rank 複製後把 F1 目標中不合規則的拿掉；另回傳 removed 清單。
    出場 X、XV、info、sigrows 都不動（名額 10、W10 不受影響；info['n_tgt'] 只在 WN 用，本題不用）。"""
    S2 = dict(S)
    if rule is None:
        return S2, []
    sig = S['sig'].copy(); rank = S['rank'].copy()
    removed = []
    for x in S['info']:
        m = pd.Period(x['month'], 'M'); k = x['k']
        for c in x['targets']:
            if not rule_keep_value(G, rule['feat'], rule['op'], rule['thr'], m, c):
                j = P.ci[c]
                sig[k, j] = False; rank[k, j] = np.nan
                removed.append((x['month'], c))
    S2['sig'] = sig; S2['rank'] = rank
    return S2, removed


def fresh_from(S, first_month):
    """只保留營收月份 ≥ first_month 的 sig（之前的訊號列整列清空）；出場 X 不動（沒有持股時不影響）。"""
    S2 = dict(S)
    sig = S['sig'].copy(); rank = S['rank'].copy()
    for x in S['info']:
        if pd.Period(x['month'], 'M') < first_month:
            sig[x['k'], :] = False; rank[x['k'], :] = np.nan
    S2['sig'] = sig; S2['rank'] = rank
    return S2


# ---------------------------------------------------------------- 以下逐字複製 s13lib（事後描述用）
def outcomes(P, X, sig_ks, cand, end_t):
    """E0 持有期報酬（同 s13lib.outcomes）。"""
    O, CF, G, REF, E50, E50O, MK, LIMDN, LIMUP = P.O, P.CF, P.G, P.REF, P.E50, P.E50O, P.MK, P.LIMDN, P.LIMUP
    sig_ks = np.array(sorted(sig_ks))
    res = []
    for k, j in zip(cand['k'].to_numpy(), cand['j'].to_numpy()):
        e = k + 1
        if e > end_t or not (np.isfinite(O[e, j]) and O[e, j] > 0):
            res.append((np.nan, np.nan, -1, 'noopen', False)); continue
        lu = bool(O[e, j] >= LIMUP[e, j] - 1e-9)
        base = CF[e, j] / O[e, j]
        nxt = sig_ks[(sig_ks > k) & (sig_ks <= end_t)]
        flag = next((kk for kk in nxt if X[kk, j]), None)
        lst = np.flatnonzero(MK[e:, j] > 0)
        last_listed = e + int(lst[-1]) if len(lst) else e
        delisted = last_listed < P.T - 1
        x = None; how = None
        if flag is not None:
            t = flag + 1
            while t <= end_t and t <= last_listed and not (np.isfinite(O[t, j]) and O[t, j] > 0 and O[t, j] > LIMDN[t, j] + 1e-9):
                t += 1
            if t <= end_t and t <= last_listed:
                x = t; how = 'rule'
        if x is None and delisted and last_listed <= end_t:
            r = base * G[last_listed, j] / G[e, j] - 1
            r50 = E50[last_listed] / E50O[e] - 1
            res.append((r, r50, last_listed, 'delist', lu)); continue
        if x is None:
            r = base * G[end_t, j] / G[e, j] - 1
            r50 = E50[end_t] / E50O[e] - 1
            res.append((r, r50, end_t, 'truncated', lu)); continue
        r = base * (G[x - 1, j] / G[e, j]) * O[x, j] / REF[x, j] - 1
        r50 = E50O[x] / E50O[e] - 1
        res.append((r, r50, x, how, lu))
    o = pd.DataFrame(res, columns=['ret', 'r50', 'exit_t', 'how', 'limup_open'], index=cand.index)
    o['excess'] = o['ret'] - o['r50']
    o['hold_days'] = np.where(o['exit_t'] >= 0, o['exit_t'] - (cand['k'] + 1), np.nan)
    return o


def month_cluster_diff(df, keep, col='excess'):
    """同 s13lib.month_cluster_diff。"""
    d = []
    for m, g in df.groupby('month'):
        g = g[np.isfinite(g[col])]
        kk = keep.reindex(g.index).fillna(False).to_numpy(bool)
        if kk.sum() == 0 or len(g) == 0:
            continue
        d.append(g[col].to_numpy()[kk].mean() - g[col].mean())
    d = np.array(d)
    if len(d) < 3:
        return dict(mean=float('nan'), t=float('nan'), n_months=int(len(d)))
    return dict(mean=float(d.mean()), t=float(d.mean() / (d.std(ddof=1) / np.sqrt(len(d)))), n_months=int(len(d)))
