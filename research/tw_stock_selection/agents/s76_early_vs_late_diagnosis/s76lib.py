#!/usr/bin/env python3
"""S76 共用：前段（2008–2015）對後段（2016–2026）診斷。只診斷，不設計新規則。
重用 S75 的連續面板與引擎（s75lib → s71lib／r55lib／s47lib／s10lib，全部不改）。
快取：scratchpad/s75/cache_cont（由 S75 的 build_panel.py 原樣重建；panel_check 與 S75 存檔逐欄相同）。
本檔新增：
- build_sig()：make_signal_s10 的逐行同邏輯，但可關掉 60 日線、20 日漲幅門檻、把續抱名次從前 20 改前 10（只用來拆解零件貢獻，不是新規則）；
- fwd_matrix()：每個訊號月「執行日開盤 → 之後第 h 個執行日開盤」的個股含息報酬與 0050 報酬；
- 統計小工具。一律 python3 -I -B。"""
import sys, math
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
AG = HERE.parent
sys.path.insert(0, str(AG / 's75_continuous_2008_2026'))
import s75lib as K                                           # noqa: E402
from s75lib import R, L, s10lib, s17lib, stlib, cslib         # noqa: E402
COSTS = s10lib.COSTS

SPLIT = '2016-01-01'                                          # 前段：訊號日 < 2016-01-01；後段：≥
SUBS = [('2008-02～2013-01（無 IFRS 校正）', '2008-01-01', '2013-01-31'), ('2013-02～2015-12（B2 校正期）', '2013-02-01', '2015-12-31'),
        ('2016～2018', '2016-01-01', '2018-12-31'), ('2019～2023', '2019-01-01', '2023-12-31'), ('2024～2026-10', '2024-01-01', '2026-12-31')]
ERAS = [('前段 2008–2015', '2008-01-01', '2015-12-31'), ('後段 2016–2026', '2016-01-01', '2026-12-31')]


def build_sig(P, univ, RT, use_ma=True, use_mom=True, keep_n=20, depth=10, mom_lo=0.10, nsel=10, shift=0):
    """use_ma＝True、use_mom＝True、keep_n＝20 時與 s10lib.make_signal_s10 的 sig／rank／X(E0)／info 逐格相同（run.py 驗收）。
    use_ma＝False：合格名單不看 60 日線（進場與續抱都不看）；use_mom＝False：前 depth 名全買；keep_n：不在合格名單前 keep_n 名就賣。"""
    S, av = RT['S'], RT['av']
    sig = np.zeros((P.T, P.N), bool); rank = np.full((P.T, P.N), np.nan); X = np.zeros((P.T, P.N), bool)
    info = []; RQ = {}; detail = []
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
        s = cslib.ins15_mask_row(P, m, k, s)
        fin = univ[k] & np.isfinite(s)
        ok = fin & P.F_MA60[k] if use_ma else fin
        o = np.flatnonzero(ok)[np.argsort(-s[ok], kind='stable')]
        pool = o[:depth]
        tgt = pool[P.RET20[k][pool] > mom_lo][:nsel] if use_mom else pool[:nsel]
        sig[k, tgt] = True; rank[k, tgt] = s[tgt]
        RQ[k] = set(int(j) for j in tgt)
        X[k, :] = True; X[k, o[:keep_n]] = False
        info.append(dict(month=str(m), k=int(k), sig_date=str(P.IDX[k].date()), n_ma=int(ok.sum()), n_tgt=int(len(tgt)), targets=[str(P.COLS[j]) for j in tgt]))
        detail.append(dict(m=m, k=int(k), s=s, fin=fin, order=o, tgt=tgt))
    sigrows = np.zeros(P.T, bool); sigrows[[x['k'] for x in info]] = True
    return dict(sig=sig, rank=rank, X=X, SD=None, max_hold=None, info=info, RQ=RQ, sigrows=sigrows, detail=detail)


def engine(P, U, sg, sat_w=0.7, cost='strict', capital=1e6, cap_pct=0.05, block=None):
    return s10lib.simulate_s10(P, sg, U, cost=COSTS[cost], capital=capital, cap_pct=cap_pct, sat_w=sat_w, sizing='W10', lock=False, block=block)


def fwd_matrix(P, ks, h=1):
    """ks＝訊號列（遞增）。第 i 個訊號：e＝ks[i]+1 開盤買、e2＝ks[i+h]+1 開盤賣。回傳 (F[i, j]、b[i]＝0050)；最後 h 個訊號為 NaN。
    個股：收[e]÷開[e] × Π(1＋R)（e+1…e2−1）× 開[e2]÷參考價[e2] − 1；e 日沒有開盤價 → NaN（買不到）；e2 日沒有開盤價 → 用前收（比值 1）。"""
    n = len(ks); F = np.full((n, P.N), np.nan); b = np.full(n, np.nan)
    for i in range(n - h):
        e, e2 = ks[i] + 1, ks[i + h] + 1
        if e2 >= P.T:
            continue
        with np.errstate(invalid='ignore', divide='ignore'):
            first = np.where(np.isfinite(P.O[e]) & (P.O[e] > 0), P.CF[e] / P.O[e], np.nan)
            last = np.where(np.isfinite(P.O[e2]) & (P.O[e2] > 0) & np.isfinite(P.REF[e2]) & (P.REF[e2] > 0), P.O[e2] / P.REF[e2], 1.0)
            F[i] = first * (P.G[e2 - 1] / P.G[e]) * last - 1
        b[i] = P.E50O[e2] / P.E50O[e] - 1
    return F, b


def tstat(x):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    if len(x) < 3 or x.std(ddof=1) == 0:
        return None
    return float(x.mean() / x.std(ddof=1) * math.sqrt(len(x)))


def desc(x, per_year=12):
    """月（訊號月）超額序列 → 筆數、平均（%）、年化（平均×每年次數，個百分點）、t、勝率。"""
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    if len(x) == 0:
        return dict(n=0)
    return dict(n=int(len(x)), mean_pct=float(x.mean() * 100), ann_pp=float(x.mean() * per_year * 100), t=tstat(x), win=float((x > 0).mean()), median_pct=float(np.median(x) * 100))


def welch(a, b):
    a = np.asarray(a, float); a = a[np.isfinite(a)]; b = np.asarray(b, float); b = b[np.isfinite(b)]
    se = math.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    return dict(diff_ann_pp=float((b.mean() - a.mean()) * 12 * 100), t=float((b.mean() - a.mean()) / se))


def spearman(a, b):
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 20:
        return np.nan
    ra = pd.Series(a[m]).rank().to_numpy(); rb = pd.Series(b[m]).rank().to_numpy()
    return float(np.corrcoef(ra, rb)[0, 1])


def era_mask(dates, a, b):
    d = pd.to_datetime(pd.Series(list(dates))).to_numpy()
    return (d >= np.datetime64(a)) & (d <= np.datetime64(b))


def load_industry(P):
    """PIT 產業（agents/o_lc_sector_rotation_pit/raw/industry_pit.csv，同 S21／S31／S56 的讀法）。回傳 INDD[t, j]（-1＝缺）與名稱表。
    注意：2015-05 以前是用最早快照回溯，不是真正當時的分類。"""
    df = pd.read_csv(AG / 'o_lc_sector_rotation_pit' / 'raw' / 'industry_pit.csv', dtype=str, keep_default_na=False)
    df['industry'] = df['industry'].str.strip()
    miss = {'未分類', '', 'nan'}
    names = sorted(set(df['industry']) - miss); nid = {n: i for i, n in enumerate(names)}
    col = {str(c): j for j, c in enumerate(P.COLS)}
    IND = np.full((P.T, P.N), -1, np.int16); idx = P.IDX.values
    for code, s_, e_, ind in df[['code', 'start', 'end', 'industry']].itertuples(index=False):
        j = col.get(str(code))
        if j is None or ind in miss:
            continue
        a = int(np.searchsorted(idx, np.datetime64(pd.Timestamp(s_)), side='left'))
        b = P.T if e_ == '' else int(np.searchsorted(idx, np.datetime64(pd.Timestamp(e_)), side='left'))
        IND[a:b, j] = nid[ind]
    return IND, names


ELEC = {'半導體業', '電腦及週邊設備業', '光電業', '通信網路業', '電子零組件業', '電子通路業', '資訊服務業', '其他電子業'}
