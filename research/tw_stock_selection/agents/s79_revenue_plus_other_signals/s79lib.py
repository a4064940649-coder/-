#!/usr/bin/env python3
"""S79 共用：營收訊號（SURGE，V_BRIDGE＝S71 B2 口徑）再加「本機可回溯到 2008 的其他資訊」。
共用引擎與既有研究一律不改；重用 s75lib（連續面板、B2 SURGE、s47lib.sim＝W10_70／E0／嚴成本／空位買 0050）。
本檔只新增
- features()：每日價量特徵（全部只用訊號日 k 當天收盤（含）以前的資料）與每月營收特徵（只用營收月 m（含）以前、且與 SURGE 同一個可用日）；
- make_signal()：s10lib.make_signal_s10 的一般化。spec＝{} 時與原函式逐格相同（驗收 A1）。
    名單 L＝股池 & 分數有值 & 還原收盤 > 60 日線 & 全部 list_gates；依分數由高到低排。
    進場＝L 前 depth(10) 名中，通過全部 pool_gates（預設 RET20 > 10%）者，依分數取前 10 名（ind_cap 有值時同產業最多 ind_cap 檔）。
    出場＝訊號列不在 L 前 20 名（＝現行 E0 的同一結構；list_gate 沒過也等於不在名單）。
一律 python3 -I -B；不呼叫 git。規格見 PREREG.json。"""
import sys
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
AG = HERE.parent
TMP = Path('/tmp/claude-0/-home-user/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/s79')
sys.path.insert(0, str(AG / 's75_continuous_2008_2026'))
import s75lib as K                                           # noqa: E402
from s75lib import R, L, S71, s10lib, s17lib, stlib, cslib   # noqa: E402
K.TMP = TMP; S71.TMP = TMP; R.TMP = TMP
IND_CSV = '/home/user/niklavsd/bottleneck-signals/data/processed/tw_industry.csv'
CAPS = K.CAPS; SHIFTS = K.SHIFTS
TRAIN = ('2008-01', '2015-12'); VALID = ('2016-01', '2026-10')
TRAIN_END = pd.Timestamp('2015-12-31')
TRAIN_YEARS = tuple(range(2008, 2016))


def load():
    P, U = K.load_cont()
    F, RT = K.surge(P, 'B2')
    return P, U, F, RT


def _roll(A, n, fn, minp=None):
    return getattr(pd.DataFrame(A).rolling(n, min_periods=minp or n), fn)().to_numpy()


def features(P, U, F):
    """回傳 (D, M, IND)：D＝{名稱: T×N 每日特徵}、M＝{名稱: 月份×N DataFrame（index 同 SURGE 表）}、IND＝長度 N 的產業代碼（−1＝本機分類檔沒有）。"""
    G = P.G; has = P.has
    D = {}
    with np.errstate(invalid='ignore', divide='ignore'):
        for n in (60, 120, 240):
            x = np.full_like(G, np.nan); x[n:] = G[n:] / G[:-n] - 1
            ok = np.zeros_like(has); ok[n:] = has[n:] & has[:-n]              # 兩端都要有成交
            D[f'RET{n}'] = np.where(ok, x, np.nan)
        x = np.full_like(G, np.nan); x[240:] = G[220:-20] / G[:-240] - 1     # 12−1 個月：t−240 → t−20
        ok = np.zeros_like(has); ok[240:] = has[220:-20] & has[:-240]
        D['MOM12_1'] = np.where(ok & has, x, np.nan)
        hi = _roll(np.where(has, G, np.nan), 240, 'max', 200)
        D['NH52'] = np.where(has, G / hi, np.nan)                            # 距 52 週（240 日）還原收盤最高
        r = np.where(has, P.R, np.nan)
        D['VOL60'] = _roll(r, 60, 'std', 40); D['VOL120'] = _roll(r, 120, 'std', 80)
        for k_ in ('VOL60', 'VOL120'):                                       # 股池內橫斷面百分位（0＝最低波動）
            v = np.where(U, D[k_], np.nan)
            D[k_ + '_PCT'] = (pd.DataFrame(v).rank(axis=1, pct=True)).to_numpy()
        D['V60'] = P.V60
        a20 = _roll(np.where(has, P.AMT, np.nan), 20, 'mean', 15)
        D['VRATIO'] = a20 / P.V60                                            # 近 20 日均額 ÷ 60 日均額
        D['PX'] = P.C                                                        # 原始收盤（股價位階）
        D['DIST60'] = G / P.MA60 - 1                                         # 還原收盤高於 60 日線的幅度
        D['RET20'] = P.RET20
    # ---- 月營收特徵（B2 口徑：2012-12 以前 × k_n；n＝1（2013-01）、2（2013-02）、3（其餘）；k 缺值的公司 2013 年缺值）
    Rv = F['Rv']; ks = F['info']['k']; Z = F['S']['STD']
    pre = np.asarray(Rv.index < S71.BREAK)
    def radj(n):
        kv = ks[n].reindex(Rv.columns).to_numpy(float)
        A = Rv.to_numpy(float).copy(); A[pre] = A[pre] * np.where(np.isfinite(kv), kv, 1.0)[None, :]
        return pd.DataFrame(A, index=Rv.index, columns=Rv.columns), ~np.isfinite(kv)
    def by_n(fn):
        out = None
        for n in (3, 1, 2):
            Ra, nok = radj(n); X = fn(Ra).copy()
            X.loc[np.asarray(X.index.year == 2013), nok] = np.nan
            if out is None:
                out = X
            else:
                row = pd.Period(f'2013-{n:02d}', 'M'); out.loc[row] = X.loc[row]
        return out
    M = {}
    M['YOY1'] = by_n(lambda Ra: (Ra / Ra.shift(12) - 1))
    M['YOY3'] = by_n(lambda Ra: (Ra.rolling(3, min_periods=3).sum() / Ra.rolling(3, min_periods=3).sum().shift(12) - 1))
    g = (M['YOY1'] > 0).to_numpy(); fin = M['YOY1'].notna().to_numpy()
    cons = np.zeros(g.shape)
    for i in range(g.shape[0]):                                              # 連續年增月數（遇缺值或年減歸零）
        cons[i] = np.where(g[i] & fin[i], (cons[i - 1] if i else 0) + 1, 0)
    M['CONSEC'] = pd.DataFrame(cons, index=Rv.index, columns=Rv.columns)
    M['POS12'] = pd.DataFrame(g & fin, index=Rv.index, columns=Rv.columns).astype(float).rolling(12, min_periods=12).sum()   # 近 12 個月年增為正的月數
    Ra3, _ = radj(3)
    M['REVHI12'] = (Rv >= Rv.rolling(12, min_periods=12).max()).astype(float).where(Rv.notna())
    # 創 12 個月新高跨 2013-01 斷點時新舊口徑混用 → 2013 年改用 B2 校正後營收（k=3）
    hi13 = (Ra3 >= Ra3.rolling(12, min_periods=12).max()).astype(float).where(Ra3.notna())
    M['REVHI12'].loc[np.asarray(Rv.index.year == 2013)] = hi13.loc[np.asarray(Rv.index.year == 2013)]
    M['SURGE_PREV'] = Z.shift(1)
    for k_ in M:
        M[k_] = M[k_].reindex(index=Z.index, columns=Z.columns)
    ind = pd.read_csv(IND_CSV, dtype=str).drop_duplicates('code').set_index('code')['industry']
    lab = pd.Series(P.COLS).map(ind)
    cats = {c: i for i, c in enumerate(sorted(lab.dropna().unique()))}
    IND = lab.map(cats).fillna(-1).astype(int).to_numpy()
    return D, M, IND


OPS = {'>': np.greater, '>=': np.greater_equal, '<': np.less, '<=': np.less_equal}


def _pct(x, m):
    """m 內的百分位名次（0～1，高＝大）；m 外 NaN。"""
    out = np.full(x.shape, np.nan)
    if m.sum():
        out[m] = pd.Series(x[m]).rank(pct=True).to_numpy()
    return out


def make_signal(P, U, RT, FE, spec=None, shift=0, depth=10, nsel=10):
    spec = spec or {}
    D, M, IND = FE
    S, av = RT['S'], RT['av']
    sc = spec.get('score', {'type': 'surge'})
    lg_ = spec.get('list_gates', []); pg = spec.get('pool_gates', [['RET20', '>', 0.10]]); icap = spec.get('ind_cap')
    sig = np.zeros((P.T, P.N), bool); rank = np.full((P.T, P.N), np.nan); X = np.zeros((P.T, P.N), bool)
    info = []
    def feat(name, m, k):
        return D[name][k] if name in D else M[name].loc[m].to_numpy(float)
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
        s = cslib.ins15_mask_row(P, m, k, s)
        fin = U[k] & np.isfinite(s)
        t = sc['type']
        if t == 'surge':
            score = s
        elif t == 'blend':                                   # w×SURGE 百分位 ＋ (1−w)×X 百分位（sign=−1 表示 X 越小越好）
            x = feat(sc['x'], m, k) * sc.get('sign', 1)
            ok = fin & np.isfinite(x)
            score = sc['w'] * _pct(np.nan_to_num(s), ok) + (1 - sc['w']) * _pct(np.nan_to_num(x), ok)
        elif t in ('ind_demean', 'ind_rank'):
            score = np.full(P.N, np.nan)
            for c in np.unique(IND[fin]):
                mm = fin & (IND == c)
                score[mm] = (s[mm] - np.median(s[mm])) if t == 'ind_demean' else (_pct(s, mm)[mm] if mm.sum() >= 3 else np.nan)
        else:
            raise ValueError(t)
        ok = fin & np.isfinite(score) & P.F_MA60[k]
        for name, op, thr in lg_:
            if name == 'IND_MED':                            # 所屬產業（股池內、SURGE 有值）SURGE 中位數
                x = np.full(P.N, np.nan)
                for c in np.unique(IND[fin]):
                    mm = fin & (IND == c); x[mm] = np.median(s[mm]) if mm.sum() >= 3 else np.nan
            elif name == 'IND_TOPN':                         # 產業內 SURGE 名次（1＝最高）
                x = np.full(P.N, np.nan)
                for c in np.unique(IND[fin]):
                    mm = fin & (IND == c); x[mm] = pd.Series(-s[mm]).rank(method='first').to_numpy()
            else:
                x = feat(name, m, k)
            with np.errstate(invalid='ignore'):
                ok &= np.isfinite(x) & OPS[op](x, thr)
        o = np.flatnonzero(ok)[np.argsort(-score[ok], kind='stable')]
        pool = o[:depth]
        okp = np.ones(len(pool), bool)
        for name, op, thr in pg:
            x = feat(name, m, k)[pool]
            with np.errstate(invalid='ignore'):
                okp &= np.isfinite(x) & OPS[op](x, thr)
        cand = pool[okp]
        if icap:
            cnt = {}; keep = []
            for j in cand:
                c = int(IND[j]) if IND[j] >= 0 else -1000 - int(j)   # 沒有分類的公司各自一類
                if cnt.get(c, 0) < icap:
                    keep.append(j); cnt[c] = cnt.get(c, 0) + 1
            cand = np.array(keep, dtype=int)
        tgt = cand[:nsel]
        sig[k, tgt] = True; rank[k, tgt] = score[tgt]
        X[k, :] = True; X[k, o[:20]] = False
        info.append(dict(month=str(m), k=int(k), sig_date=str(P.IDX[k].date()), n_ma=int(ok.sum()), n_tgt=int(len(tgt)), targets=[str(P.COLS[j]) for j in tgt]))
    sigrows = np.zeros(P.T, bool); sigrows[[x['k'] for x in info]] = True
    XV = {'E0': X}
    return dict(sig=sig, rank=rank, X=X, SD=None, max_hold=None, info=info, XV=XV, RQ={}, sigrows=sigrows)


def run_paths(P, U, RT, FE, spec, paths=True):
    """回傳 {(cap, shift): (nav, lg, st)}；paths=False 只跑標準路徑（100 萬、+0）。"""
    out = {}
    for s in (SHIFTS if paths else (0,)):
        sg = make_signal(P, U, RT, FE, spec, shift=s)
        for cap in (CAPS if paths else (1e6,)):
            nav, lg, st = K.sim(P, sg, U, cap)
            out[(cap, s)] = (nav, lg, dict(avg_idle=float(1 - st['stockfrac'].dropna().mean()), sf=st['stockfrac']), sg if (s == 0 and cap == 1e6) else None)
    return out


def bench(P):
    b = pd.Series(P.E50, index=P.IDX).loc[P.IDX[P.t0]:]
    return b / b.iloc[0] * 1e6


def yearly_excess(nav, b, years):
    """逐日曆年：策略年報酬 − 0050 年報酬（個百分點）；第一年以起日為基底。"""
    y1 = nav.groupby(nav.index.year).last(); y2 = b.groupby(b.index.year).last()
    out = {}
    p1, p2 = nav.iloc[0], b.iloc[0]
    for y in y1.index:
        if y in years:
            out[int(y)] = float(((y1[y] / p1 - 1) - (y2[y] / p2 - 1)) * 100)
        p1, p2 = y1[y], y2[y]
    return out


def window(nav, b, a, e):
    return K.window_block(nav, b.reindex(nav.index), a, e)


def n_trades(lg, a, e):
    x = lg[(lg['why'] != 'open_end')]
    return int(((x['ent_date'] >= pd.Timestamp(a)) & (x['ent_date'] <= pd.Timestamp(e))).sum())
