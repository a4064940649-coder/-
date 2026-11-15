#!/usr/bin/env python3
"""S11 共用程式庫：第二個獨立訊號——季報驚喜（SUE⊥），單獨衛星 Q 與和 SURGE 合併的衛星 C。
在 s10lib（不改）之上：面板、創新板、SURGE 訊號、引擎全部沿用（s10lib.simulate_s10，不開任何出場掛鉤＝s9lib.simulate_s9 open）；
本檔只新增季報資料讀取、季換股訊號（make_signal_q），以及回撤分段統計。
季報資料只讀 o_pead_margin_surprise_low_attention/data/ 的 fin_long.parquet（MOPS XBRL 單季財報）與 avail_dates.parquet（市場可得日 A），不改它們。
中間檔放 scratchpad/s11tmp/。規格見 PREREG.json。一律 python3 -I -B。"""
import sys
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
AG = HERE.parent
S11 = Path('/tmp/claude-0/-home-user--/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/s11tmp')
sys.path.insert(0, str(AG / 's10_exit_details'))
import s10lib                                              # noqa: E402
from s10lib import s9lib, s8lib, s7lib, s6lib, s5lib, s4lib, s3lib, s2lib, stlib, cslib, COSTS   # noqa: E402,F401
s10lib.S10 = S11; s9lib.S9 = S11; s8lib.S8 = S11; s7lib.S7 = S11; s6lib.S6 = S11; s5lib.S5 = S11; s4lib.S4 = S11; s3lib.S3 = S11; s2lib.S2 = S11; stlib.S1 = S11; cslib.TMP = S11
PEAD = AG / 'o_pead_margin_surprise_low_attention' / 'data'
FIN_IND = {'bd', 'basi', 'ins', 'fh', 'mim'}               # MOPS 金融保險業財報格式（同 o_pead pcore.FIN_IND）
SUE_WIN, SUE_MIN = 8, 6                                    # 同 o_pead PREDEF：前 8 季（至少 6 季）季差標準差，ddof=1
SEGMENTS = {'2018': ('2018-01-01', '2018-12-31'), '2020-03': ('2020-01-01', '2020-06-30'),
            '2022': ('2022-01-01', '2022-12-31'), '2024-07~2025-04': ('2024-07-01', '2025-04-30')}


def deadline(y, q):
    """證券交易法第 36 條（2013 年 IFRS 起）一般公司：Q1 5/15、Q2 8/14、Q3 11/14、年報（Q4）次年 3/31。"""
    return {1: pd.Timestamp(y, 5, 15), 2: pd.Timestamp(y, 8, 14), 3: pd.Timestamp(y, 11, 14), 4: pd.Timestamp(y + 1, 3, 31)}[q]


def load():
    S11.mkdir(parents=True, exist_ok=True)
    return s10lib.load()


# ---------------------------------------------------------------- 季報資料
def fin_tables():
    """回傳 dict：SUE、RSUE（index＝code，columns＝季序號 kq＝4y＋q−1）、ind（code→財報格式產業）、A（(code, kq)→市場可得日）、A_method。
    SUE＝(EPS_q − EPS_{q−4}) ÷ std(前 8 季的 EPS 季差，至少 6 季，ddof=1)；RSUE 同法用單季營收（同 o_pead s1_events.sue_of）。"""
    f = pd.read_parquet(PEAD / 'fin_long.parquet', columns=['code', 'y', 'q', 'industry', 'eps_s', 'rev_s'])
    f = f[f['code'].str.fullmatch(r'[1-9]\d{3}') & ~f['code'].str.startswith('91')].copy()
    assert not f.duplicated(['code', 'y', 'q']).any()
    f['kq'] = f['y'] * 4 + f['q'] - 1
    ind = f.sort_values('kq').groupby('code')['industry'].last()

    def sue_of(col):
        W = f.pivot(index='code', columns='kq', values=col)
        ks = np.arange(W.columns.min(), W.columns.max() + 1); W = W.reindex(columns=ks)
        D = W - W.shift(4, axis=1)
        sd = D.T.shift(1).rolling(SUE_WIN, min_periods=SUE_MIN).std(ddof=1).T
        return (D / sd.where(sd > 0)).replace([np.inf, -np.inf], np.nan)

    SUE = sue_of('eps_s'); RSUE = sue_of('rev_s').reindex(index=SUE.index, columns=SUE.columns)
    a = pd.read_parquet(PEAD / 'avail_dates.parquet', columns=['code', 'y', 'q', 'A', 'A_method', 'mkt'])
    a['kq'] = a['y'] * 4 + a['q'] - 1
    A = {(c, k): d for c, k, d in zip(a['code'], a['kq'], a['A'])}
    AM = {(c, k): m for c, k, m in zip(a['code'], a['kq'], a['A_method'])}
    return dict(SUE=SUE, RSUE=RSUE, ind=ind, A=A, A_method=AM, fin=f)


def quarter_rows(P, kq_list):
    """季序號 → 訊號列（未位移）：k＝第一個 ≥ 法定期限的交易日（期限遇假日順延），訊號列＝k＋1（同 SURGE A10 的 rev_avail_days 寫法）。"""
    out = {}
    for kq in kq_list:
        y, q = kq // 4, kq % 4 + 1
        d = deadline(y, q)
        k = int(np.searchsorted(P.IDX.values, np.datetime64(d), side='left'))
        if k + 1 < P.T:
            out[kq] = k + 1
    return out


def q_scores(P, FT, univ, kq, k, kind):
    """季序號 kq、訊號列 k（含位移）：可選股票與分數。
    可選＝univ[k]（只上市、流動性、非創新板）且非金融保險財報格式、SUE 有值、A ≤ 訊號日（A 缺→視為法定期限）；kind='perp' 另要 RSUE 有值。
    kind='perp'：SUE⊥＝pct(SUE) −（a＋b·pct(RSUE)），pct 與 OLS 都在同一季可選股票內算（訊號日時這些財報已全部公布，所以是 PIT）。
    kind='raw'：分數＝pct(SUE)。回傳 (可選欄位 index 陣列, 分數向量 N 長 NaN 補)。"""
    SUE, RSUE = FT['SUE'], FT['RSUE']
    if kq not in SUE.columns:
        return np.array([], int), np.full(P.N, np.nan), {}
    s = SUE[kq].reindex(P.COLS).to_numpy(float)
    r = RSUE[kq].reindex(P.COLS).to_numpy(float)
    ind = FT['ind'].reindex(P.COLS).to_numpy(object)
    nonfin = np.array([not (isinstance(x, str) and x in FIN_IND) for x in ind])
    y, q = kq // 4, kq % 4 + 1
    dl = deadline(y, q); dk = P.IDX[k]
    Aok = np.array([(FT['A'].get((c, kq), dl) if pd.notna(FT['A'].get((c, kq), dl)) else dl) <= dk for c in P.COLS])
    ok = univ[k] & nonfin & np.isfinite(s) & Aok
    if kind == 'perp':
        ok &= np.isfinite(r)
    w = np.flatnonzero(ok)
    sc = np.full(P.N, np.nan)
    diag = dict(n_univ=int(univ[k].sum()), n_nonfin=int((univ[k] & nonfin).sum()), n_sue=int((univ[k] & nonfin & np.isfinite(s)).sum()),
                n_late_excluded=int((univ[k] & nonfin & np.isfinite(s) & ~Aok).sum()), n_elig=int(len(w)))
    if len(w) < 50:
        return w, sc, diag
    ps = pd.Series(s[w]).rank(pct=True).to_numpy()
    if kind == 'perp':
        pr = pd.Series(r[w]).rank(pct=True).to_numpy()
        Xm = np.column_stack([np.ones(len(w)), pr])
        b = np.linalg.lstsq(Xm, ps, rcond=None)[0]
        sc[w] = ps - Xm @ b
        diag['ols_a'], diag['ols_b'] = float(b[0]), float(b[1])
        diag['corr_pct_sue_rsue'] = float(np.corrcoef(ps, pr)[0, 1])
    else:
        sc[w] = ps
    return w, sc, diag


def make_signal_q(P, univ, FT, shift=0, kind='perp', filt=None, nsel=10, mom_lo=0.10, keep_info=True):
    """季換股訊號（格式同 s4lib.make_signal，給 s10lib.simulate_s10 用）。
    每季訊號列 k＝quarter_rows（＋shift，用位移後那天的股池、價格、可得日重算）。
    filt=None（Q）：可選股票依分數由大到小（stable）取前 nsel 名＝買進名單＝續抱名單。
    filt='F1'（QF）：只看還原收盤 > 60 日均線的可選股票，依分數取前 10 名＝續抱名單；其中 RET20 > mom_lo 的＝買進名單（不遞補）。
    出場：下一個季訊號列不在續抱名單就標記，次日開盤賣（X 只在季訊號列有值；季中不出場）。
    有訊號的季：未位移訊號列可選股票 ≥ 50 檔。"""
    SUE = FT['SUE']
    kqs = [kq for kq in SUE.columns if kq >= 2015 * 4]
    rows = quarter_rows(P, kqs)
    sig = np.zeros((P.T, P.N), bool); X = np.zeros((P.T, P.N), bool); rank = np.full((P.T, P.N), np.nan)
    info = []
    for kq, k0 in sorted(rows.items()):
        if k0 < 61:
            continue
        w0, _, _ = q_scores(P, FT, univ, kq, k0, kind)
        if len(w0) < 50:
            continue
        k = k0 + shift
        if k >= P.T:
            continue
        w, sc, diag = q_scores(P, FT, univ, kq, k, kind)
        if len(w) < 50:
            continue
        if filt is None:
            cand = w
        elif filt == 'F1':
            cand = w[P.F_MA60[k, w]]
        else:
            raise ValueError(filt)
        o = cand[np.argsort(-sc[cand], kind='stable')]
        pool = o[:nsel]
        tgt = pool if filt is None else pool[P.RET20[k, pool] > mom_lo]
        sig[k, tgt] = True; rank[k, tgt] = sc[tgt]
        X[k, :] = True; X[k, pool] = False
        d = dict(kq=int(kq), fq=f'{kq // 4}Q{kq % 4 + 1}', k=int(k), sig_date=str(P.IDX[k].date()), n_tgt=int(len(tgt)), **diag)
        if keep_info:
            d['targets'] = [str(P.COLS[j]) for j in tgt]; d['keep'] = [str(P.COLS[j]) for j in pool]
        info.append(d)
    sigrows = np.zeros(P.T, bool); sigrows[[x['k'] for x in info]] = True
    return dict(sig=sig, rank=rank, X=X, SD=None, max_hold=None, info=info, sigrows=sigrows, RQ={})


def surge_signal(P, univ, RT, shift=0):
    """現行 SURGE 衛星（S10 E0）：s10lib.make_signal_s10 的 E0。"""
    S = s10lib.make_signal_s10(P, univ, RT, shift=shift)
    S2 = dict(S); S2['X'] = S['XV']['E0']; S2.pop('XV', None)
    return S2


# ---------------------------------------------------------------- 統計
def seg_dd(nav):
    """四段空頭各自的最大跌幅（高點與低點都在段內）。"""
    out = {}
    for k, (a, b) in SEGMENTS.items():
        s = nav.loc[a:b].dropna()
        out[k] = float((s / s.cummax() - 1).min()) if len(s) > 2 else float('nan')
    return out


def month_excess(nav, bench, a=None, b=None):
    s = nav.loc[a:b].dropna(); q = bench.reindex(s.index)
    m1 = s.groupby(s.index.to_period('M')).last(); m2 = q.groupby(q.index.to_period('M')).last()
    return (m1.pct_change() - m2.pct_change()).dropna()


def corr_excess(n1, n2, bench, a=None, b=None):
    x = month_excess(n1, bench, a, b); y = month_excess(n2, bench, a, b)
    z = pd.concat([x, y], axis=1).dropna()
    return float(np.corrcoef(z.iloc[:, 0], z.iloc[:, 1])[0, 1]) if len(z) > 3 else float('nan')
