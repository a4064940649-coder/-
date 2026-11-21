#!/usr/bin/env python3
"""S92 共用程式庫：補完 S88 的兩個缺口——(1) 應收帳款／存貨（raw/tsd_extra/fs_xbrl）；(2) 財報實際公告日（raw/tsd_extra/filing_dates）。
引擎：正式管線（agents/official_nav/run_official.py 的 E0｜W10_70 寫法；EXRIGHT_FIX、DEADLINE_FIX、INS15_MASK、REDUCE_FIX、0050 官方含息五個開關全開）。
      模擬＝s10lib.simulate_s10（原樣呼叫）；本題只改訊號列的買進名單（sig／rank），出場 X 不動。load／signal／sim／fresh_from／order_ma 照 s90lib。
時點（兩種，pit）：
  's88'＝S88 原本的估法：法定期限 < 訊號日 d 且（市場可得日 A 缺值或 A < d）的最新一季，晚交或無資料往回 1 季。
  'fd' ＝實際公告日：該檔該季所有文件（合併／個體、中英文、重編）上傳時間的最小值 F；季 kq 可用 ⇔ F 的日期 < d（前一日以前已上傳）。
         F 缺值的季退回 S88 估法（法定期限 < d 且 A 沒晚）。由「季底 < d 的最新一季」往回最多找 5 季，取第一個可用且有資料者。
財報數字：
  S88 旗標（ACC_HI、OCF_NEG）：同 S88，讀 o_pead_margin_surprise_low_attention 的 fin_long／fin_raw（s88lib.Fin／features 的複製，只把選季抽成 pick）。
  應收／存貨（新）：fs_xbrl（一家一季一列，元）。AR＝應收帳款＋應收帳款-關係人＋應收票據（缺值當 0；三者皆缺＝缺值）；INV＝存貨；
    單季營收／營業成本＝年初至今累計相減（Q1＝累計值）。只判一般業（industry_template＝ci）；金融、證券、保險、金控不判。
    ar_g＝AR ÷ 去年同季 AR − 1；inv_g 同；rev_g＝單季營收年增率；ar_gap＝ar_g − rev_g；inv_gap＝inv_g − rev_g；
    dso＝AR ÷ 單季營收 × 91.25；dio＝INV ÷ 單季營業成本 × 91.25；dso_chg／dio_chg＝減去年同季。分母 ≤ 0 或缺值 → 缺值（→ 不擋）。
暫存：scratchpad/s92/cache（面板快取，硬連結自 offfix/cache）。一律 python3 -I -B。"""
import sys, gc
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
AG = HERE.parent
RAW = AG.parent / 'raw' / 'tsd_extra'
PEAD = AG / 'o_pead_margin_surprise_low_attention'
TMP = Path('/tmp/claude-0/-home-user/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/s92/cache')
TMP.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(AG / 's19_xq_threshold'))
import s19lib                                              # noqa: E402
from s19lib import s10lib, s9lib, s8lib, s7lib, s6lib, s5lib, s4lib, s3lib, s2lib, stlib, cslib, COSTS   # noqa: E402
for mod, attr in ((s19lib, 'S19'), (s10lib, 'S10'), (s9lib, 'S9'), (s8lib, 'S8'), (s7lib, 'S7'), (s6lib, 'S6'), (s5lib, 'S5'),
                  (s4lib, 'S4'), (s3lib, 'S3'), (s2lib, 'S2'), (stlib, 'S1'), (cslib, 'TMP')):
    setattr(mod, attr, TMP)

CAPS = (6e5, 8e5, 1e6, 1.5e6, 2e6)
SHIFTS = (0, 1, 2, 3)
SEGMENTS = {'2022': ('2022-01-01', '2022-12-31'), '2024-07~2025-04': ('2024-07-01', '2025-04-30')}
SEGV = tuple(SEGMENTS)
VALID_FIRST_MONTH = pd.Period('2021-12', 'M')
TARGET = dict(since2019=0.4886246519182116, full=0.3446294016617564, maxdd=-0.3160938700293674, n_closed=381, n_all=389, paths_median=0.4641920784536262)
FILL_DEPTH = 20
FIN_IND = ('bd', 'fh', 'basi', 'ins')
ACC_PCT = 0.80
MAX_BACK_S88 = 1
MAX_BACK_FD = 4
QDAYS = 91.25


def load():
    assert cslib.EXRIGHT_FIX is False and cslib.DEADLINE_FIX is False and cslib.INS15_MASK is False
    assert cslib.REDUCE_FIX is False and cslib.E50_SOURCE is None
    cslib.EXRIGHT_FIX = True; cslib.DEADLINE_FIX = True; cslib.INS15_MASK = True
    cslib.REDUCE_FIX = True; cslib.E50_SOURCE = cslib.E50_OFFICIAL_CSV
    P = s19lib.load()
    assert P.exright_fix and P.reduce_fix and P.e50_source == str(cslib.E50_OFFICIAL_CSV)
    s19lib.e50_fix_suspension(P)
    INNOV, _ = s4lib.innov_panel(P)
    U = stlib.universe(P, 'twse') & ~INNOV
    del INNOV
    S, _ = s19lib.rev_std(P)
    avA = s19lib.a10_rows(P, list(S.index))
    for a in ('H', 'L', 'VOL', 'CHG', 'HF', 'VOLF', 'adjH', 'HI240', 'MA60', 'nbar', 'F_HI', 'IDXC', 'listed', 'NEARHI', 'V60', 'liq', 'has'):
        if hasattr(P, a):
            delattr(P, a)
    gc.collect()
    return P, U, dict(S=S, av=avA)


def signal(P, U, RT, shift):
    return s10lib.make_signal_s10(P, U, RT, shift=shift)


def fresh_from(S, first_month=VALID_FIRST_MONTH):
    S2 = dict(S); sig = S['sig'].copy(); rank = S['rank'].copy()
    for x in S['info']:
        if pd.Period(x['month'], 'M') < first_month:
            sig[x['k'], :] = False; rank[x['k'], :] = np.nan
    S2['sig'] = sig; S2['rank'] = rank
    return S2


def sim(P, S, U, cap=1e6, cost='strict', block=None):
    return s10lib.simulate_s10(P, S, U, sat_w=0.7, sizing='W10', lock=False, cost=COSTS[cost], capital=cap, block=block)


def seg_dd(nav, keys=SEGV):
    o = {}
    for k in keys:
        a, b = SEGMENTS[k]
        s = nav.loc[a:b].dropna()
        o[k] = float((s / s.cummax() - 1).min()) if len(s) > 2 else float('nan')
    return o


def order_ma(P, U, RT, x):
    m = pd.Period(x['month'], 'M'); k = x['k']
    s = RT['S'].loc[m].to_numpy()
    s = cslib.ins15_mask_row(P, m, k, s)
    ok = U[k] & np.isfinite(s) & P.F_MA60[k]
    return np.flatnonzero(ok)[np.argsort(-s[ok], kind='stable')], s


# ---------------------------------------------------------------- 時點
def deadline(kq):
    y, q = kq // 4, kq % 4 + 1
    return {1: pd.Timestamp(y, 5, 15), 2: pd.Timestamp(y, 8, 14), 3: pd.Timestamp(y, 11, 14), 4: pd.Timestamp(y + 1, 3, 31)}[q]


def qend(kq):
    return pd.Timestamp(kq // 4, 3 * (kq % 4 + 1), 1) + pd.offsets.MonthEnd(0)


def latest_kq_by_deadline(d):
    for kq in range(4 * d.year + 3, 4 * (d.year - 2), -1):
        if deadline(kq) < d:
            return kq
    raise ValueError(d)


def latest_kq_by_qend(d):
    for kq in range(4 * d.year + 3, 4 * (d.year - 2), -1):
        if qend(kq) < d:
            return kq
    raise ValueError(d)


def tload(name, columns=None):
    sys.path.insert(0, str(RAW))
    from tsd_extra_load import load as _l
    return _l(name, columns=columns)


def pct_in(x, mask):
    o = np.full(len(x), np.nan)
    ok = mask & np.isfinite(x)
    n = int(ok.sum())
    if n >= 30:
        o[ok] = pd.Series(x[ok]).rank(method='average').to_numpy() / n
    return o


def _ytd_to_q(y, ks):
    """年初至今累計 → 單季（Q1＝累計；其餘＝本季累計 − 上一季累計）。"""
    o = np.full_like(y, np.nan)
    for i, kq in enumerate(ks):
        if kq % 4 == 0:
            o[:, i] = y[:, i]
        elif i >= 1:
            o[:, i] = y[:, i] - y[:, i - 1]
    return o


class Fin:
    """code × kq 矩陣（S88 的 fin_long／fin_raw ＋ fs_xbrl 的應收、存貨 ＋ 市場可得日 A ＋ 實際公告日 FD）。"""

    def __init__(self, P):
        codes = [str(c) for c in P.COLS]; cset = set(codes)
        f = pd.read_parquet(PEAD / 'data' / 'fin_long.parquet', columns=['code', 'period', 'y', 'q', 'industry', 'eps_s', 'rev_s', 'gp_s', 'opi_s', 'nip_s', 'assets'])
        r = pd.read_parquet(PEAD / 'replicate_o10' / 'data' / 'fin_raw.parquet', columns=['stock_id', 'period', 'ocf_ytd', 'ni_ytd'])
        assert not r.duplicated(['stock_id', 'period']).any()
        f = f.merge(r.rename(columns={'stock_id': 'code'}), on=['code', 'period'], how='left', validate='1:1')
        f = f[f['code'].str.fullmatch(r'[1-9]\d{3}')].copy()
        f['kq'] = f['y'] * 4 + f['q'] - 1
        x = tload('fs_xbrl', columns=['period', 'code', 'industry_template', 'report', 'inventories', 'accounts_receivable', 'accounts_receivable_related', 'notes_receivable',
                                      'revenue_ytd', 'cost_of_sales_ytd', 'total_assets'])
        x = x[x['code'].isin(cset)].copy()
        assert not x.duplicated(['code', 'period']).any()
        x['kq'] = x['period'].str[:4].astype(int) * 4 + x['period'].str[5].astype(int) - 1
        self.kmin = int(min(f['kq'].min(), x['kq'].min())); self.kmax = int(max(f['kq'].max(), x['kq'].max()))
        ks = np.arange(self.kmin, self.kmax + 1); self.ks = ks
        piv = lambda df, c: df.pivot(index='code', columns='kq', values=c).reindex(index=codes, columns=ks).to_numpy(float)
        # --- S88 部分（同 s88lib.Fin）
        self.rev, self.opi, self.nip, self.eps, self.ast = (piv(f, c) for c in ('rev_s', 'opi_s', 'nip_s', 'eps_s', 'assets'))
        ocf_y, ni_y = piv(f, 'ocf_ytd'), piv(f, 'ni_ytd')
        self.any88 = np.isfinite(self.rev) | np.isfinite(self.opi) | np.isfinite(self.nip) | np.isfinite(self.eps)
        qn = ks % 4 + 1

        def ttm_from_ytd(y):
            o = np.full_like(y, np.nan)
            for i, kq in enumerate(ks):
                if qn[i] == 4:
                    o[:, i] = y[:, i]
                else:
                    i_py4 = i - int(qn[i]); i_pyq = i - 4
                    if i_pyq >= 0 and i_py4 >= 0:
                        o[:, i] = y[:, i] + y[:, i_py4] - y[:, i_pyq]
            return o
        self.ocf_ttm, self.ni_ttm = ttm_from_ytd(ocf_y), ttm_from_ytd(ni_y)
        ind = f.sort_values('kq').groupby('code')['industry'].last().reindex(codes)
        self.is_fin88 = ind.isin(FIN_IND).to_numpy()
        # --- fs_xbrl 部分
        x['one'] = 1.0
        self.hasx = np.isfinite(piv(x, 'one'))
        a1, a2, a3 = piv(x, 'accounts_receivable'), piv(x, 'accounts_receivable_related'), piv(x, 'notes_receivable')
        self.ar = np.where(np.isfinite(a1) | np.isfinite(a2) | np.isfinite(a3), np.nan_to_num(a1) + np.nan_to_num(a2) + np.nan_to_num(a3), np.nan)
        self.inv = piv(x, 'inventories')
        self.xrev = _ytd_to_q(piv(x, 'revenue_ytd'), ks)
        self.xcogs = _ytd_to_q(piv(x, 'cost_of_sales_ytd'), ks)
        self.xast = piv(x, 'total_assets')
        tpl = x.sort_values('kq').groupby('code')['industry_template'].last().reindex(codes)
        self.tpl = tpl
        self.is_ci = (tpl == 'ci').to_numpy()
        # --- 時點
        a = pd.read_parquet(PEAD / 'data' / 'avail_dates.parquet', columns=['code', 'y', 'q', 'A'])
        a = a[a['code'].isin(cset)].copy(); a['kq'] = a['y'] * 4 + a['q'] - 1
        self.A = a.drop_duplicates(['code', 'kq']).pivot(index='code', columns='kq', values='A').reindex(index=codes, columns=ks).to_numpy('datetime64[D]')
        fd = tload('filing_dates', columns=['code', 'year', 'season', 'uploaded_at', 'doc_type'])
        fd = fd[fd['code'].isin(cset)].copy(); fd['kq'] = fd['year'] * 4 + fd['season'] - 1
        g = fd.groupby(['code', 'kq'])['uploaded_at'].min().reset_index()
        g['F'] = g['uploaded_at'].dt.normalize()
        self.FD = g.pivot(index='code', columns='kq', values='F').reindex(index=codes, columns=ks).to_numpy('datetime64[D]')
        self.DL = np.array([np.datetime64(deadline(int(kq)).date(), 'D') for kq in ks])
        self.QE = np.array([np.datetime64(qend(int(kq)).date(), 'D') for kq in ks])
        self._pc = {}

    def pick(self, d, pit, which):
        """回傳長度 N 的欄索引（−1＝無可用季）。which＝'s88'（fin_long 有資料）或 'x'（fs_xbrl 有列）。"""
        key = (pd.Timestamp(d), pit, which)
        if key in self._pc:
            return self._pc[key]
        d = pd.Timestamp(d); d64 = np.datetime64(d.date(), 'D')
        has = self.any88 if which == 's88' else self.hasx
        N = has.shape[0]; idx = np.full(N, -1)
        if pit == 's88':
            k0 = latest_kq_by_deadline(d); cands = range(k0, k0 - MAX_BACK_S88 - 1, -1)
        else:
            k0 = latest_kq_by_qend(d); cands = range(k0, k0 - MAX_BACK_FD - 1, -1)
        for kq in cands:
            i = kq - self.kmin
            if i < 0 or i >= has.shape[1]:
                continue
            Ai = self.A[:, i]
            notlate = np.isnat(Ai) | (Ai < d64)
            if pit == 's88':
                ok = notlate
            else:
                Fi = self.FD[:, i]
                ok = np.where(np.isnat(Fi), (self.DL[i] < d64) & notlate, Fi < d64)
            use = (idx < 0) & has[:, i] & ok
            idx[use] = i
        self._pc[key] = idx
        return idx

    def _take(self, M, idx, lag=0):
        i = idx - lag
        ok = (idx >= 0) & (i >= 0)
        o = np.full(len(idx), np.nan)
        r = np.flatnonzero(ok)
        o[r] = M[r, i[r]]
        return o

    def meta(self, d, pit, which):
        """每檔：kq、資料齡（訊號日 − 季底）、公告日距訊號日天數（F 缺值 → NaN）。"""
        idx = self.pick(d, pit, which); d64 = np.datetime64(pd.Timestamp(d).date(), 'D')
        ok = idx >= 0; r = np.flatnonzero(ok)
        kq = np.full(len(idx), np.nan); age = np.full(len(idx), np.nan); since = np.full(len(idx), np.nan)
        kq[r] = self.ks[idx[r]]
        age[r] = (d64 - self.QE[idx[r]]).astype(float)
        F = self.FD[r, idx[r]]
        since[r] = np.where(np.isnat(F), np.nan, (d64 - F).astype(float))
        return kq, age, since

    def clear(self):
        self._pc = {}


def feat88(P, U, FN, k, pit):
    """S88 的 ACC_HI／OCF_NEG（同 s88lib.features），時點 pit。"""
    d = P.IDX[k]; idx = FN.pick(d, pit, 's88')
    nf = ~FN.is_fin88
    ast = FN._take(FN.ast, idx); ocf = FN._take(FN.ocf_ttm, idx); ni = FN._take(FN.ni_ttm, idx)
    with np.errstate(invalid='ignore', divide='ignore'):
        acc = np.where(nf & (ast > 0), (ni - ocf) / ast, np.nan)
    acc_pct = pct_in(acc, U[k])
    fin = np.isfinite
    with np.errstate(invalid='ignore'):
        F = dict(acc=acc, acc_pct=acc_pct, ocf_ttm=ocf,
                 OCF_NEG=nf & fin(ocf) & (ocf < 0), ACC_HI=fin(acc_pct) & (acc_pct >= ACC_PCT))
    F['kq'], F['age_days'], F['since_filing'] = FN.meta(d, pit, 's88')
    return F


def featx(P, FN, k, pit='fd'):
    """應收／存貨特徵（fs_xbrl），時點 pit。非一般業一律缺值。"""
    d = P.IDX[k]; idx = FN.pick(d, pit, 'x')
    t = lambda M, lag=0: FN._take(M, idx, lag)
    ci = FN.is_ci
    ar, ar4, inv, inv4 = t(FN.ar), t(FN.ar, 4), t(FN.inv), t(FN.inv, 4)
    rv, rv4, cg, cg4 = t(FN.xrev), t(FN.xrev, 4), t(FN.xcogs), t(FN.xcogs, 4)
    with np.errstate(invalid='ignore', divide='ignore'):
        g = lambda a, b: np.where(ci & (a > 0) & (b > 0), a / b - 1, np.nan)
        ar_g, inv_g, rev_g = g(ar, ar4), g(inv, inv4), g(rv, rv4)
        dso = np.where(ci & (rv > 0) & np.isfinite(ar), ar / rv * QDAYS, np.nan); dso4 = np.where(ci & (rv4 > 0) & np.isfinite(ar4), ar4 / rv4 * QDAYS, np.nan)
        dio = np.where(ci & (cg > 0) & np.isfinite(inv), inv / cg * QDAYS, np.nan); dio4 = np.where(ci & (cg4 > 0) & np.isfinite(inv4), inv4 / cg4 * QDAYS, np.nan)
    F = dict(ar=np.where(ci, ar, np.nan), inv=np.where(ci, inv, np.nan), ar_g=ar_g, inv_g=inv_g, rev_g=rev_g, ar_gap=ar_g - rev_g, inv_gap=inv_g - rev_g,
             dso=dso, dio=dio, dso_chg=dso - dso4, dio_chg=dio - dio4)
    F['kq'], F['age_days'], F['since_filing'] = FN.meta(d, pit, 'x')
    return F


def apply_bad(P, U, RT, S, badfn, mode, rng=None):
    """badfn(k) → 長度 N 布林（True＝不買）。mode Z＝不遞補（錢放 0050）；F＝由 L_MA 第 11～20 名中 RET20 > 10% 且沒被擋者依 SURGE 遞補。
    rng 不為 None＝隨機對照：每月隨機剔除同檔數（F 照樣遞補、不套條件）。回傳 (S', 記錄)。"""
    rec = []
    S2 = dict(S); sig = S['sig'].copy(); rank = S['rank'].copy()
    for x in S['info']:
        k = x['k']
        tgt = np.flatnonzero(S['sig'][k])
        if not len(tgt):
            continue
        bad = np.asarray(badfn(k), bool)
        rm = [int(j) for j in tgt if bad[j]]
        if rng is not None and rm:
            rm = [int(j) for j in rng.choice(tgt, size=len(rm), replace=False)]
        added = []
        if rm:
            sig[k, rm] = False; rank[k, rm] = np.nan
            if mode == 'F':
                o, s = order_ma(P, U, RT, x)
                pool = o[:10]
                chk = pool[P.RET20[k][pool] > 0.10][:10]
                assert set(int(j) for j in chk) == set(int(j) for j in tgt), ('名單重建不符', x['month'])
                for j in o[10:FILL_DEPTH]:
                    if len(added) >= len(rm):
                        break
                    if P.RET20[k][j] > 0.10 and (rng is not None or not bad[j]):
                        added.append(int(j)); sig[k, j] = True; rank[k, j] = s[j]
        rec.append(dict(month=x['month'], k=int(k), orig=[str(P.COLS[j]) for j in tgt], removed=[str(P.COLS[j]) for j in rm], added=[str(P.COLS[j]) for j in added]))
    S2['sig'] = sig; S2['rank'] = rank
    return S2, rec
