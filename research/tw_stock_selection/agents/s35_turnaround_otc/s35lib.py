#!/usr/bin/env python3
"""S35 共用程式庫：「谷底翻揚」是不是 SURGE 大贏家的共同特徵？——以上櫃股票當沒看過的樣本外。
在 s20lib（不改；其下 s17lib→s10lib：面板、創新板、SURGE（STD，σ>0）、F1、E0、引擎）之上，本檔只新增：
- 開 cslib.EXRIGHT_FIX＝True（上市除權息參考價修正；上櫃沒有 TWT49U，修正不作用在上櫃，見 data_check.json）。
- 上櫃 2018-10 月營收補洞：s16 的 rev_patch_otc_201810.csv（只讀，sha256 驗證）併進 tw_revenue.csv，寫成 scratchpad/s35/tw_revenue_patched.csv，
  再把 stlib.REV_CSV 指過去（s17lib.rev_family 在呼叫時才讀 stlib.REV_CSV；共用引擎不改）。併法同 s16lib.rev_tables_s16（只補 tw_revenue.csv 沒有的代號）。
- 上櫃股池 U_OTC＝P.liq & 當日上櫃（MK＝2）& 非創新板（同 S16 OTC 格）。
- 三個谷底翻揚指標（PREREG 逐字）：PTH（同 S33）、μ（同 S20 MUAMT：D(m−1…m−24) 平均）、ROE（同 S29 Q：TTM 歸屬母公司淨利 ÷ 最近一季歸屬母公司權益，PIT）。
- 同業報酬：同產業（上市用 industry_pit.csv PIT；上櫃與其他沒有 PIT 的代號用 tw-stock-screener stock_list.csv 2026-10 快照，非 PIT）、
  上市＋上櫃合併股池（P.liq & MK>0 & 非創新板）內、排除自己與該產業當日 60 日均成交值最大者，等權平均 E0 同期間報酬（s25lib.hold_ret）；同業 < 5 檔 → 缺值。
- 月份群聚 OLS（CR1）。
中間檔放 scratchpad/s35/。規格見 PREREG.json。一律 python3 -I -B。"""
import sys, math, hashlib
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
AG = HERE.parent
S35 = Path('/tmp/claude-0/-home-user--/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/s35')
sys.path.insert(0, str(AG / 's20_persistent_growth'))
import s20lib                                              # noqa: E402
from s20lib import s17lib, s10lib, s9lib, s8lib, s7lib, s6lib, s5lib, s4lib, s3lib, s2lib, stlib, cslib, COSTS   # noqa: E402,F401
sys.path.insert(0, str(AG / 's25_factor_momentum'))
import s25lib                                              # noqa: E402  只借用 hold_ret／last_listed／load_industry_pit（不改）
sys.path.insert(0, str(AG / 's29_course_qmv'))
import s29lib                                              # noqa: E402  只借用 fund_tables（不改）
sys.path.insert(0, str(AG / 's33_52w_high'))
import s33lib                                              # noqa: E402  只借用 pth_panel（不改）


def _redirect():
    for mod, attr in ((s33lib, 'S33'), (s29lib, 'S29'), (s25lib, 'S25'), (s20lib, 'S20'), (s17lib, 'S17'), (s10lib, 'S10'), (s9lib, 'S9'), (s8lib, 'S8'),
                      (s7lib, 'S7'), (s6lib, 'S6'), (s5lib, 'S5'), (s4lib, 'S4'), (s3lib, 'S3'), (s2lib, 'S2'), (stlib, 'S1'), (cslib, 'TMP')):
        if hasattr(mod, attr):
            setattr(mod, attr, S35)


_redirect()
cslib.EXRIGHT_FIX = True

PATCH_CSV = AG / 's16_surge_otc' / 'rev_patch_otc_201810.csv'
REV_ORIG = stlib.REV_CSV
REV_PATCHED = S35 / 'tw_revenue_patched.csv'
STOCK_LIST = AG / 'o_0050_tilt_pit' / 'raw' / 'screener' / 'stock_list.csv'   # tw-stock-screener 2026-10 快照（同 /home/user/stevenhsiehsheng-ctrl/tw-stock-screener/data/stock_list.csv）
FIRST_MONTH = pd.Period('2016-01', 'M')                    # 營收月份 ≥ 2016-01（訊號日 2016-02 起）
SPLIT_LAST_MONTH = pd.Period('2020-11', 'M')               # 分段：營收月份 ≤ 2020-11（訊號日 2016-02～2020-12）＝2016～2020；其餘＝2021～2026
MIN_PEERS = 5


def sha256(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def build_patched_rev():
    """tw_revenue.csv ＋ s16 補洞檔（只補 2018-10 tw_revenue.csv 沒有的代號）→ REV_PATCHED；回傳（新增列數, sha256）。"""
    S35.mkdir(parents=True, exist_ok=True)
    r = pd.read_csv(REV_ORIG, dtype={'code': str})
    pt = pd.read_csv(PATCH_CSV, dtype={'code': str})
    have = set(r.loc[r['month'].astype(str).str[:7] == '2018-10', 'code'])
    pt = pt[~pt['code'].isin(have)][['month', 'market', 'code', 'name', 'revenue_k_twd']]
    out = pd.concat([r, pt.reindex(columns=r.columns)], ignore_index=True)
    out.to_csv(REV_PATCHED, index=False)
    return int(len(pt)), sha256(REV_PATCHED)


def load():
    S35.mkdir(parents=True, exist_ok=True)
    n_add, _ = build_patched_rev()
    stlib.REV_CSV = str(REV_PATCHED)
    P = s20lib.load()
    _redirect()
    assert P.exright_fix, '未開 EXRIGHT_FIX'
    s17lib.e50_fix_suspension(P)
    P.PTH = s33lib.pth_panel(P)
    P.n_patch_rows = n_add
    return P


def universes(P):
    INNOV, _ = s4lib.innov_panel(P)
    base = P.liq & ~INNOV
    return dict(OTC=base & (P.MK == 2), ALL=base & (P.MK > 0), TWSE=base & (P.MK == 1))


# ---------------------------------------------------------------- 產業
def industry_matrix(P, ks):
    """dict(k → 長度 N 的 int 陣列，-1＝缺值)、名稱表、來源旗標（0 缺、1 PIT、2 快照）。
    先查 industry_pit.csv（PIT，上市）；沒有涵蓋該日的代號 → stock_list.csv 快照（非 PIT；主要是上櫃）。名稱「未分類」／空白＝缺。"""
    pit = s25lib.load_industry_pit()
    sl = pd.read_csv(STOCK_LIST, dtype=str, keep_default_na=False)
    snap = dict(zip(sl['code'], sl['industry'].str.strip()))
    miss = {'未分類', '', 'nan'}
    names = sorted((set(pit['industry']) | set(snap.values())) - miss)
    nid = {n: i for i, n in enumerate(names)}
    rows = {c: g[['start', 'end', 'industry']].to_numpy() for c, g in pit.groupby('code')}
    out, src = {}, {}
    for k in ks:
        d = P.IDX[k]
        a = np.full(P.N, -1, int); s = np.zeros(P.N, np.int8)
        for j, c in enumerate(P.COLS):
            c = str(c)
            r = rows.get(c)
            hit = False
            if r is not None:
                for st, en, ind in r:
                    if st <= d < en:
                        if ind not in miss:
                            a[j] = nid[ind]; s[j] = 1
                        hit = True
                        break
            if not hit and c in snap and snap[c] not in miss:
                a[j] = nid[snap[c]]; s[j] = 2
        out[int(k)] = a; src[int(k)] = s
    return out, names, src


# ---------------------------------------------------------------- 谷底翻揚指標
class ROE:
    """S29 Q：訊號日 d 可得（avail ≤ d）的最近一季，ROE＝TTM 歸屬母公司淨利 ÷ 該季歸屬母公司權益；權益 ≤ 0 或 TTM 缺 → NaN。"""

    def __init__(self, P):
        self.P = P
        F = s29lib.fund_tables()
        self.F = F[F['code'].isin(set(map(str, P.COLS)))]

    def at(self, d):
        x = self.F[self.F['avail'] <= d]
        last = x.groupby('code').last()
        ci = self.P.ci
        js = np.array([ci[c] for c in last.index])
        ttm = np.full(self.P.N, np.nan); eq = np.full(self.P.N, np.nan); kq = np.full(self.P.N, np.nan)
        ttm[js] = last['ttm'].to_numpy(float); eq[js] = last['eq'].to_numpy(float); kq[js] = last['kq'].to_numpy(float)
        with np.errstate(invalid='ignore', divide='ignore'):
            roe = np.where(eq > 0, ttm / eq, np.nan)
        dk = d.year * 4 + (d.month - 1) // 3
        return roe, dk - kq


def below_median(v):
    """當月 F1 內：值 < 當月 F1（有值者）中位數 → True；缺值 → False；有值者 < 2 檔 → 全 False。"""
    v = np.asarray(v, float)
    ok = np.isfinite(v)
    out = np.zeros(len(v), bool)
    if ok.sum() >= 2:
        med = np.median(v[ok])
        out[ok] = v[ok] < med
    return out


def f1_table(P, U, RT, S0, G, RO, depth=10):
    """S0＝make_signal_s10（上櫃股池）。每個 F1 目標一列，含三個原始值與指標。另附 L_MA 第 11～20 名（grp＝R11_20，描述用）。"""
    S = RT['S']; rows = []
    for x in S0['info']:
        m = pd.Period(x['month'], 'M'); k = x['k']
        if m < FIRST_MONTH:
            continue
        s = S.loc[m].to_numpy()
        fin, o_ma, tgt = s33lib.row_parts(P, U, s, k)
        assert [str(P.COLS[j]) for j in tgt] == x['targets'], ('重現 F1 失敗', x['month'])
        roe, lagq = RO.at(P.IDX[k])
        for pos, j in enumerate(o_ma[:20]):
            c = str(P.COLS[j])
            grp = 'F1' if j in set(tgt.tolist()) else ('R11_20' if pos >= 10 else 'POOL_NOMOM')
            rows.append((x['month'], int(k), c, int(j), grp, pos + 1, float(s[j]), float(P.RET20[k, j]), float(P.PTH[k, j]),
                         float(G['MUAMT'].at[m, c]) if c in G['MUAMT'].columns else np.nan,
                         float(G['MS'].at[m, c]) if c in G['MS'].columns else np.nan, float(roe[j]), float(lagq[j]), int(P.MK[k, j])))
    df = pd.DataFrame(rows, columns=['month', 'k', 'code', 'j', 'grp', 'lma_rank', 'surge', 'ret20', 'PTH', 'mu', 'mu_sigma', 'ROE', 'roe_lagq', 'mk'])
    df['PTH_LOW'] = False; df['ROE_LOW'] = False
    f1 = df['grp'] == 'F1'
    for m, g in df[f1].groupby('month'):
        df.loc[g.index, 'PTH_LOW'] = below_median(g['PTH'].to_numpy())
        df.loc[g.index, 'ROE_LOW'] = below_median(g['ROE'].to_numpy())
    df['MU_LE0'] = np.isfinite(df['mu']) & (df['mu'] <= 0)
    df['n_ind'] = df[['PTH_LOW', 'MU_LE0', 'ROE_LOW']].sum(axis=1)
    df['TA'] = df['n_ind'] >= 2
    df.loc[~f1, ['PTH_LOW', 'ROE_LOW', 'TA']] = False      # 中位數只在 F1 內定義；非 F1 列不用
    df['mper'] = pd.PeriodIndex(df['month'], freq='M')
    df['half'] = np.where(df['mper'] <= SPLIT_LAST_MONTH, '2016_2020', '2021_2026')
    return df


def peer_returns(P, df, UALL, IND, V60):
    """每列（F1 目標）的同業等權報酬（同一進出場日）。回傳 peer、n_peer。"""
    LL = s25lib.last_listed(P)
    peer = np.full(len(df), np.nan); npeer = np.zeros(len(df), int)
    for i, y in enumerate(df.itertuples()):
        j, k = int(y.j), int(y.k); e = k + 1
        ind = IND[k]; g = ind[j]
        if g < 0 or not np.isfinite(y.ret):
            continue
        x = None if y.how == 'truncated' else int(y.exit_t)
        cand = np.flatnonzero(UALL[k] & (ind == g))
        cand = cand[cand != j]
        if len(cand) == 0:
            continue
        big = int(cand[np.argmax(np.nan_to_num(V60[k, cand].astype(float), nan=-1))])
        peers = [p for p in cand if p != big and np.isfinite(P.O[e, p]) and P.O[e, p] > 0]
        pr = np.array([s25lib.hold_ret(P, p, e, x, LL) for p in peers], float)
        pr = pr[np.isfinite(pr)]
        npeer[i] = len(pr)
        if len(pr) >= MIN_PEERS:
            peer[i] = pr.mean()
    return peer, npeer


# ---------------------------------------------------------------- 統計
def cluster_diff(y, x, g):
    """y＝超額、x＝0/1 組別、g＝群聚（訊號月）。OLS y＝a＋b·x，CR1 群聚標準誤；df＝群數−1。
    回傳 b（＝組 1 平均 − 組 0 平均）、se、t、單尾 p（H1：b > 0）、n、群數。"""
    from scipy import stats
    y = np.asarray(y, float); x = np.asarray(x, float); g = np.asarray(g)
    ok = np.isfinite(y)
    y, x, g = y[ok], x[ok], g[ok]
    n = len(y)
    if n < 10 or x.sum() < 2 or (1 - x).sum() < 2:
        return dict(n=int(n), n1=int(x.sum()), diff=float('nan'), t=float('nan'), p_one=float('nan'))
    X = np.column_stack([np.ones(n), x])
    XtX_inv = np.linalg.inv(X.T @ X)
    beta = XtX_inv @ X.T @ y
    e = y - X @ beta
    ug = pd.DataFrame(X * e[:, None]).groupby(g).sum().to_numpy()
    G = len(ug)
    meat = ug.T @ ug
    V = XtX_inv @ meat @ XtX_inv * (G / (G - 1)) * ((n - 1) / (n - 2))
    se = float(math.sqrt(V[1, 1]))
    t = float(beta[1] / se)
    p = float(stats.t.sf(t, G - 1))
    return dict(n=int(n), n1=int(x.sum()), n0=int(n - x.sum()), n_months=int(G), mean1=float(y[x == 1].mean()), mean0=float(y[x == 0].mean()),
                diff=float(beta[1]), se=se, t=t, p_one=p)


def within_month_diff(y, x, g):
    """只用同月兩組都有的月份：逐月（組 1 平均 − 組 0 平均），對月份做 t。"""
    d = pd.DataFrame(dict(y=y, x=np.asarray(x, int), g=g)).dropna()
    gm = d.groupby(['g', 'x'])['y'].mean().unstack()
    if 1 not in gm or 0 not in gm:
        return dict(n_months=0)
    v = (gm[1] - gm[0]).dropna().to_numpy()
    if len(v) < 3:
        return dict(n_months=int(len(v)))
    t = float(v.mean() / (v.std(ddof=1) / math.sqrt(len(v))))
    from scipy import stats
    return dict(n_months=int(len(v)), mean=float(v.mean()), t=t, p_one=float(stats.t.sf(t, len(v) - 1)))


def grp_stats(v):
    v = np.asarray(v, float); v = v[np.isfinite(v)]
    if len(v) == 0:
        return dict(n=0)
    return dict(n=int(len(v)), mean_pct=float(v.mean() * 100), median_pct=float(np.median(v) * 100), win=float((v > 0).mean()),
                mean_drop_top5_pct=float(np.sort(v)[:-5].mean() * 100) if len(v) > 5 else None)
