#!/usr/bin/env python3
"""S13 共用程式庫：大贏家能否事前辨識（描述性研究＋乾淨樣本外）。
在 s12lib／s10lib（都不改）之上：面板、創新板、SURGE、F1、E0、引擎全部沿用；
本檔只新增：SURGE 候選（每個 A10 換股日 L_MA 前 20 名）、六個事前特徵、E0 持有期報酬（結果變數）、AUC 與月份群聚 t。
季報只讀 o_pead_margin_surprise_low_attention/data/ 的 fin_long.parquet、avail_dates.parquet（不改）。
中間檔放 scratchpad/s13tmp/。規格見 PREREG.json。一律 python3 -I -B。"""
import sys
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
AG = HERE.parent
S13 = Path('/tmp/claude-0/-home-user--/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/s13tmp')
sys.path.insert(0, str(AG / 's12_surge_size_tiers'))
import s12lib                                              # noqa: E402
from s12lib import s10lib, s9lib, s8lib, s7lib, s6lib, s5lib, s4lib, s3lib, s2lib, stlib, cslib, COSTS   # noqa: E402,F401
s12lib.S12 = S13; s10lib.S10 = S13; s9lib.S9 = S13; s8lib.S8 = S13; s7lib.S7 = S13; s6lib.S6 = S13; s5lib.S5 = S13; s4lib.S4 = S13
s3lib.S3 = S13; s2lib.S2 = S13; stlib.S1 = S13; cslib.TMP = S13
PEAD = AG / 'o_pead_margin_surprise_low_attention' / 'data'
FIN_IND = {'bd', 'basi', 'ins', 'fh', 'mim'}               # 金融保險業財報格式（同 S11）
FEATURES = ['ACC', 'REVHI12', 'GMYOY', 'NEAR52', 'RET60', 'ADVCHG']
SEGMENTS = s12lib.SEGMENTS
seg_dd = s12lib.seg_dd
top5_share = s12lib.top5_share


def load():
    S13.mkdir(parents=True, exist_ok=True)
    return s10lib.load()


def deadline(y, q):
    """證券交易法第 36 條（同 S11）：Q1 5/15、Q2 8/14、Q3 11/14、年報 次年 3/31。"""
    return {1: pd.Timestamp(y, 5, 15), 2: pd.Timestamp(y, 8, 14), 3: pd.Timestamp(y, 11, 14), 4: pd.Timestamp(y + 1, 3, 31)}[q]


# ---------------------------------------------------------------- 資料
def rev_matrix(P):
    """月營收（千元）矩陣：index＝月份 Period，columns＝P.COLS；處理方式與 s2lib.rev_tables 相同（同月同代號優先 sii、≤0 視為缺）。"""
    r = pd.read_csv(stlib.REV_CSV, dtype={'code': str}, parse_dates=['month'])
    r['pri'] = (r['market'] != 'sii').astype(int)
    r = r.sort_values(['month', 'code', 'pri']).drop_duplicates(['month', 'code'], keep='first')
    Rv = r.pivot(index='month', columns='code', values='revenue_k_twd').sort_index()
    Rv.index = pd.PeriodIndex(Rv.index, freq='M')
    Rv = Rv.reindex(pd.period_range(Rv.index.min(), Rv.index.max(), freq='M')).reindex(columns=P.COLS)
    return Rv.where(Rv > 0)


def fin_gm():
    """單季毛利率 GM＝gp_s ÷ rev_s（rev_s > 0）；回傳 (GM 字典 (code,kq)→值, A 字典 (code,kq)→市場可得日, 金融格式代號集合)。"""
    f = pd.read_parquet(PEAD / 'fin_long.parquet', columns=['code', 'y', 'q', 'industry', 'rev_s', 'gp_s'])
    f = f[f['code'].str.fullmatch(r'[1-9]\d{3}')].copy()
    f['kq'] = f['y'] * 4 + f['q'] - 1
    fin_codes = set(f.loc[f['industry'].isin(FIN_IND), 'code'])
    ok = (f['rev_s'] > 0) & np.isfinite(f['gp_s'])
    GM = {(c, k): g / r for c, k, g, r in zip(f.loc[ok, 'code'], f.loc[ok, 'kq'], f.loc[ok, 'gp_s'], f.loc[ok, 'rev_s'])}
    a = pd.read_parquet(PEAD / 'avail_dates.parquet', columns=['code', 'y', 'q', 'A'])
    a['kq'] = a['y'] * 4 + a['q'] - 1
    A = {(c, k): d for c, k, d in zip(a['code'], a['kq'], a['A']) if pd.notna(d)}
    return GM, A, fin_codes


# ---------------------------------------------------------------- 候選
def candidates(P, U, RT, shift=0):
    """每個 A10 換股日（月份條件與 s10lib.make_signal_s10 完全相同）：L_MA＝股池內、SURGE 有值、還原收盤 > 60 日均線，依 SURGE 由大到小（stable）；
    候選＝前 20 名。f1＝現行 F1 會買的（前 10 名中 RET20 > 10%）。回傳 DataFrame（每列一個候選）。"""
    S, av = RT['S'], RT['av']
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
        ok_ma = fin & P.F_MA60[k]
        o_ma = np.flatnonzero(ok_ma)[np.argsort(-s[ok_ma], kind='stable')]
        pool = o_ma[:10]
        tgt = set(int(j) for j in pool[P.RET20[k][pool] > 0.10][:10])
        for r, j in enumerate(o_ma[:20]):
            rows.append((str(m), int(k), int(j), str(P.COLS[j]), r + 1, float(s[j]), float(P.RET20[k, j]), int(j) in tgt))
    df = pd.DataFrame(rows, columns=['month', 'k', 'j', 'code', 'rank', 'surge', 'ret20', 'f1'])
    df['sig_date'] = P.IDX[df['k'].to_numpy()]
    return df


def add_features(P, RT, Rv, FG, cand):
    """六個事前特徵（全部只用訊號列 k 當天已公開的資料；定義見 PREREG.json features）。就地加欄並回傳。"""
    S = RT['S']
    GM, A, fin_codes = FG
    k = cand['k'].to_numpy(); j = cand['j'].to_numpy()
    mper = pd.PeriodIndex(cand['month'], freq='M')
    # 1 ACC：本月 SURGE − 上月 SURGE
    Sprev = S.shift(1)
    cand['ACC'] = [cand['surge'].iat[i] - Sprev.at[mper[i], cand['code'].iat[i]] if mper[i] in Sprev.index else np.nan for i in range(len(cand))]
    # 2 REVHI12：本月營收 ÷ 前 12 個月最高營收 − 1（前 12 個月至少 10 個月有值）
    Rmax = Rv.shift(1).rolling(12, min_periods=10).max()
    cand['REVHI12'] = [Rv.at[mper[i], cand['code'].iat[i]] / Rmax.at[mper[i], cand['code'].iat[i]] - 1 for i in range(len(cand))]
    # 3 GMYOY：最近一季「已過法定期限且市場可得日 ≤ 訊號日」的單季毛利率 − 去年同季（百分點）
    out = np.full(len(cand), np.nan); used = []
    for i in range(len(cand)):
        c = cand['code'].iat[i]; d = cand['sig_date'].iat[i]
        if c in fin_codes:
            used.append(''); continue
        y, q = d.year, (d.month - 1) // 3 + 1
        kq = y * 4 + q - 1
        got = None
        for kk in range(kq, kq - 8, -1):                   # 由新到舊找第一個「期限 < 訊號日」的季
            if deadline(kk // 4, kk % 4 + 1) >= d:
                continue
            a = A.get((c, kk))
            if a is not None and a > d:                    # 晚交：當時還看不到 → 改用更早一季（＝當時看得到的最近一季）
                continue
            got = kk; break
        if got is not None and (c, got) in GM and (c, got - 4) in GM:
            out[i] = (GM[(c, got)] - GM[(c, got - 4)]) * 100
        used.append('' if got is None else f'{got // 4}Q{got % 4 + 1}')
    cand['GMYOY'] = out; cand['gm_q'] = used
    # 4 NEAR52：還原收盤 ÷ 240 日還原最高價（同 s2lib.NEARHI）
    cand['NEAR52'] = P.NEARHI[k, j]
    # 5 RET60：還原收盤 ÷ 60 個交易日前 − 1
    cand['RET60'] = P.G[k, j] / P.G[k - 60, j] - 1
    # 6 ADVCHG：log（近 20 日平均成交值 ÷ 再往前 60 日平均成交值），無成交日成交值計 0
    amt = np.nan_to_num(P.AMT)
    a20 = np.array([amt[kk - 19:kk + 1, jj].mean() for kk, jj in zip(k, j)])
    a60 = np.array([amt[kk - 79:kk - 19, jj].mean() for kk, jj in zip(k, j)])
    with np.errstate(divide='ignore', invalid='ignore'):
        cand['ADVCHG'] = np.where((a20 > 0) & (a60 > 0), np.log(a20 / a60), np.nan)
    return cand


# ---------------------------------------------------------------- 結果變數
def outcomes(P, X, sig_ks, cand, end_t):
    """E0 持有期報酬：訊號列 k 的候選，k＋1 開盤進場；之後每個訊號列 k'（> k、≤ end_t）若 X[k', j]（不在 L_MA 前 20 名、跌破季線或離開股池）就標記，
    標記後第一個「有開盤價且開盤 > 跌停價」的交易日開盤出場（同引擎）；中途下市＝最後一個有日 K 的收盤；到 end_t 還沒出場＝end_t 收盤估值（truncated）。
    報酬＝(收盤[e] ÷ 開盤[e]) × G[x−1] ÷ G[e] × 開盤[x] ÷ 參考價[x] − 1（同引擎的市值演進，不扣成本）；0050＝同時窗還原開盤（或收盤）比值。
    進場日沒有開盤價 → NaN（noopen）。回傳 DataFrame：ret、r50、excess、exit_t、how、hold_days、limup_open。"""
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
        # 下市日（之後再也沒有日 K）
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


# ---------------------------------------------------------------- 統計
def auc(score, label):
    """Mann–Whitney AUC（平手給一半）：P(大贏家的分數 > 非大贏家)。NaN 先剔除。"""
    s = np.asarray(score, float); y = np.asarray(label, bool)
    ok = np.isfinite(s); s, y = s[ok], y[ok]
    n1, n0 = int(y.sum()), int((~y).sum())
    if n1 == 0 or n0 == 0:
        return float('nan')
    r = pd.Series(s).rank(method='average').to_numpy()
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def month_cluster_diff(df, keep, col='excess'):
    """每個月 d_m＝當月被選中候選的平均 col − 當月全部候選（有結果的）平均；只用有被選中的月份。回傳 mean、t（月份當群聚）、月數。"""
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


def rule_keep(df, feat, direction, thr):
    """規則：direction＝+1 → 特徵 ≥ thr 才留；−1 → 特徵 ≤ thr 才留；特徵缺值 → 不留。"""
    v = df[feat].to_numpy(float)
    with np.errstate(invalid='ignore'):
        k = (v >= thr) if direction > 0 else (v <= thr)
    return pd.Series(k & np.isfinite(v), index=df.index)
