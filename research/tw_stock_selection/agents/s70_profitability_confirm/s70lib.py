#!/usr/bin/env python3
"""S70 共用程式庫：營收驚奇名單加一道「獲利確認」（訓練→凍結→驗證一次）。
候選（事前固定，見 SPEC.json／PREREG.json；「名單」＝當月訊號名單＝L_MA 依 SURGE 前 10 名中 RET20 > 10% 者）：
  C1（OPLOSS）  ＝名單中「訊號日當時可得的最近一季，單季營業利益 < 0」者不買；不遞補，錢照引擎放 0050；缺值保留。
  C2（GPA_LOW） ＝名單中「毛利／資產（最近四季毛利合計 ÷ 最近一季總資產，Novy-Marx 2013）落在同期股池最低 30%」者不買；不遞補；缺值保留。
  C3（GPA_ORDER）＝名單不變，只把買進順序由 SURGE 改成毛利／資產同期股池百分位由高到低（缺值當 0.5；同分依 SURGE）。
  （EPS_TTM＝最近四季歸屬母公司淨利合計 ≤ 0：只用來和 S29 做重疊統計，不是候選、不模擬。）
鏈：s47lib（不改；只借它「正式開關 EXRIGHT_FIX、DEADLINE_FIX、INS15_MASK 全開」與匯入鏈）→ s20lib → s17lib → s10lib（全部不改）。
引擎直接用 s10lib.simulate_s10（原函式，不做副本）；候選只改 make_signal_s10 輸出的 sig／rank（買進名單與買進順序），出場 X（E0）不動。
財報：o_pead_margin_surprise_low_attention/data/fin_long.parquet（MOPS XBRL 單季，2013 起；只讀）＋ avail_dates.parquet（市場可得日 A）。
時點（同 S28／S29）：訊號日 d，季 kq 可用 ⇔ 法定期限 deadline(kq) < d 且（A 缺值或 A < d）；取法定期限已過的最新季，該檔該季晚交（A ≥ d）或沒有資料列時往回 1 季；
  法定期限＝證券交易法第 36 條：Q1 5/15、Q2 8/14、Q3 11/14、Q4（年報）次年 3/31（金融保險與 -KY 的 Q2 為 8/31，A10 訊號日都在 9/10 之後，不影響）。
  A10 訊號日約每月 10 日 → 5 月訊號日看不到 Q1（仍用去年 Q4）、8 月看不到 Q2、11 月看不到 Q3：資料齡偏老但保證當時可得。
中間檔放 scratchpad/s70/cache。一律 python3 -I -B。"""
import sys
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
AG = HERE.parent
TMP = Path('/tmp/claude-0/-home-user/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/s70')
CACHE = TMP / 'cache'
sys.path.insert(0, str(AG / 's47_allocation_family'))
import s47lib as L47                                       # noqa: E402
from s47lib import s20lib, s17lib, s10lib, s9lib, s8lib, s7lib, s6lib, s5lib, s4lib, s3lib, s2lib, stlib, cslib, COSTS   # noqa: E402,F401
for _m, _a in ((s20lib, 'S20'), (s17lib, 'S17'), (s10lib, 'S10'), (s9lib, 'S9'), (s8lib, 'S8'), (s7lib, 'S7'), (s6lib, 'S6'), (s5lib, 'S5'),
               (s4lib, 'S4'), (s3lib, 'S3'), (s2lib, 'S2'), (stlib, 'S1'), (cslib, 'TMP')):
    setattr(_m, _a, CACHE)
assert cslib.EXRIGHT_FIX is True and cslib.DEADLINE_FIX is True and cslib.INS15_MASK is True, '正式開關必須全開'

PEAD = AG / 'o_pead_margin_surprise_low_attention' / 'data'
CUR = 'CUR'
CANDS = ('C1', 'C2', 'C3')
RULE = dict(C1='OPLOSS', C2='GPA_LOW', C3='GPA_ORDER')
GPA_LOW_PCT = 0.30                                         # C2：同期股池最低 30%（Fama-French 30／40／30 斷點的低組）
MAX_BACK = 1                                               # 最近一季晚交或無資料時只往回 1 季
CAPS = (6e5, 8e5, 1e6, 1.5e6, 2e6)
SHIFTS = (0, 1, 2, 3)
ACC = dict(since2019=0.4875617247721724, full=0.3439657562340761, maxdd=-0.32119049020243395, n_closed=381)   # deadline_engine_fix ALL＝official_nav E0
VALID_FIRST_MONTH = s20lib.VALID_FIRST_MONTH               # 營收月份 ≥ 2021-12（訊號日 2022-01 起）
fresh_from = s20lib.fresh_from
SEGV = ('2022', '2024-07~2025-04')
DROP = ('H', 'L', 'VOL', 'CHG', 'HF', 'VOLF', 'adjH', 'HI240', 'MA60', 'nbar', 'F_HI', 'IDXC', 'AO', 'listed', 'NEARHI', 'V60', 'liq')
FIELDS = ('opi_q', 'rev_q', 'opm_q', 'gp_ttm', 'ni_ttm', 'eps_q', 'eps_ttm', 'equity', 'assets', 'roe', 'gpa', 'kq', 'age_days')


def load():
    CACHE.mkdir(parents=True, exist_ok=True)
    P = s20lib.load()
    assert P.exright_fix is True
    return P


def base(P):
    INNOV, _ = s4lib.innov_panel(P)
    U = stlib.universe(P, 'twse') & ~INNOV
    F = s17lib.rev_family(P, defs=['STD'])
    RT = dict(S=F['S']['STD'], av=F['av'])
    return U, RT


def slim(P):
    for a in DROP:
        if hasattr(P, a):
            delattr(P, a)


# ---------------------------------------------------------------- 財報（時點正確）
def deadline(kq):
    y, q = kq // 4, kq % 4 + 1
    return {1: pd.Timestamp(y, 5, 15), 2: pd.Timestamp(y, 8, 14), 3: pd.Timestamp(y, 11, 14), 4: pd.Timestamp(y + 1, 3, 31)}[q]


def latest_kq_by_deadline(d):
    """日期 d 時法定期限已過（deadline < d）的最新季序號（kq＝4×年＋季−1）。"""
    for kq in range(4 * d.year + 3, 4 * (d.year - 2), -1):
        if deadline(kq) < d:
            return kq
    raise ValueError(d)


class Fin:
    """code × kq 的單季矩陣。ttm＝連續 4 季都有值才算。"""

    def __init__(self, P):
        f = pd.read_parquet(PEAD / 'fin_long.parquet', columns=['code', 'y', 'q', 'industry', 'eps_s', 'rev_s', 'gp_s', 'opi_s', 'nip_s', 'equity_parent', 'assets'])
        f = f[f['code'].str.fullmatch(r'[1-9]\d{3}')].copy()
        assert not f.duplicated(['code', 'y', 'q']).any()
        f['kq'] = f['y'] * 4 + f['q'] - 1
        self.kmin, self.kmax = int(f['kq'].min()), int(f['kq'].max())
        ks = np.arange(self.kmin, self.kmax + 1)
        codes = [str(c) for c in P.COLS]
        piv = lambda c: f.pivot(index='code', columns='kq', values=c).reindex(index=codes, columns=ks).to_numpy(float)
        self.opi, self.rev, self.gp, self.nip, self.eps, self.eq, self.ast = (piv(c) for c in ('opi_s', 'rev_s', 'gp_s', 'nip_s', 'eps_s', 'equity_parent', 'assets'))
        f['one'] = 1.0
        self.has = np.isfinite(piv('one'))                 # fin_long 有這一列
        self.anydata = np.isfinite(self.rev) | np.isfinite(self.opi) | np.isfinite(self.nip) | np.isfinite(self.eps)

        def ttm(x):
            o = np.full_like(x, np.nan)
            o[:, 3:] = x[:, 3:] + x[:, 2:-1] + x[:, 1:-2] + x[:, :-3]
            return o
        self.gp_ttm, self.ni_ttm, self.eps_ttm = ttm(self.gp), ttm(self.nip), ttm(self.eps)
        a = pd.read_parquet(PEAD / 'avail_dates.parquet', columns=['code', 'y', 'q', 'A'])
        a = a[a['code'].isin(set(codes))].copy(); a['kq'] = a['y'] * 4 + a['q'] - 1
        A = a.drop_duplicates(['code', 'kq']).pivot(index='code', columns='kq', values='A').reindex(index=codes, columns=ks)
        self.A = A.to_numpy('datetime64[ns]')
        self.ind = f.sort_values('kq').groupby('code')['industry'].last().reindex(codes)
        self._cache = {}

    def row(self, d):
        """訊號日 d：每檔（P.COLS 順序）當時可得的最近一季各欄。回傳 dict(欄 → 長度 N 陣列)。"""
        key = pd.Timestamp(d)
        if key in self._cache:
            return self._cache[key]
        k0 = latest_kq_by_deadline(key)
        N = self.opi.shape[0]
        out = {c: np.full(N, np.nan) for c in FIELDS}
        done = np.zeros(N, bool)
        d64 = np.datetime64(key, 'ns')
        for kq in range(k0, k0 - MAX_BACK - 1, -1):
            i = kq - self.kmin
            if i < 0 or i >= self.opi.shape[1]:
                continue
            Ai = self.A[:, i]
            late = ~np.isnat(Ai) & ~(Ai < d64)             # 還沒公告
            use = ~done & self.anydata[:, i] & ~late
            qe = pd.Timestamp(kq // 4, 3 * (kq % 4 + 1), 1) + pd.offsets.MonthEnd(0)
            rev = np.where(self.rev[:, i] > 0, self.rev[:, i], np.nan)
            eq = np.where(self.eq[:, i] > 0, self.eq[:, i], np.nan)
            ast = np.where(self.ast[:, i] > 0, self.ast[:, i], np.nan)
            opm = self.opi[:, i] / rev
            opm = np.where(np.abs(opm) <= 5, opm, np.nan)  # 營收極小造成的離譜比率當缺值（同 S28）
            vals = dict(opi_q=self.opi[:, i], rev_q=self.rev[:, i], opm_q=opm, gp_ttm=self.gp_ttm[:, i], ni_ttm=self.ni_ttm[:, i], eps_q=self.eps[:, i],
                        eps_ttm=self.eps_ttm[:, i], equity=self.eq[:, i], assets=self.ast[:, i], roe=self.ni_ttm[:, i] / eq, gpa=self.gp_ttm[:, i] / ast,
                        kq=np.full(N, float(kq)), age_days=np.full(N, float((key - qe).days)))
            for c in FIELDS:
                out[c][use] = vals[c][use]
            done |= use
        self._cache[key] = out
        return out


def pct_in(x, mask):
    """x 在 mask 內有值者的百分位（0～1，平均名次 ÷ 個數；越大＝值越高）；其餘 NaN。"""
    o = np.full(len(x), np.nan)
    ok = mask & np.isfinite(x)
    n = int(ok.sum())
    if n >= 30:
        o[ok] = pd.Series(x[ok]).rank(method='average').to_numpy() / n
    return o


def features(P, U, FN, k):
    """訊號列 k：財報欄＋毛利／資產在同期股池 U[k] 的百分位 gpa_pct。"""
    r = dict(FN.row(P.IDX[k]))
    r['gpa_pct'] = pct_in(r['gpa'], U[k])
    return r


def flags(r, j):
    """單檔判定（缺值一律 False＝保留）。"""
    opi, gp, nt = r['opi_q'][j], r['gpa_pct'][j], r['ni_ttm'][j]
    return dict(OPLOSS=bool(np.isfinite(opi) and opi < 0), GPA_LOW=bool(np.isfinite(gp) and gp < GPA_LOW_PCT), EPS_TTM=bool(np.isfinite(nt) and nt <= 0))


def apply_cand(P, U, FN, S, cand):
    """回傳 (S', 記錄)。cand＝'CUR'／'C1'／'C2'／'C3'（或直接給規則名）。記錄＝每個訊號列 dict(month, k, orig, removed, order)。"""
    rule = RULE.get(cand, cand)
    rec = []
    if rule == 'CUR':
        return S, rec
    S2 = dict(S); sig = S['sig'].copy(); rank = S['rank'].copy()
    for x in S['info']:
        k = x['k']
        tgt = np.flatnonzero(S['sig'][k])
        if not len(tgt):
            continue
        r = features(P, U, FN, k)
        if rule in ('OPLOSS', 'GPA_LOW', 'EPS_TTM'):
            rm = [int(j) for j in tgt if flags(r, j)[rule]]
            sig[k, rm] = False; rank[k, rm] = np.nan
            rec.append(dict(month=x['month'], k=int(k), orig=[str(P.COLS[j]) for j in tgt], removed=[str(P.COLS[j]) for j in rm]))
        elif rule == 'GPA_ORDER':
            gp = np.where(np.isfinite(r['gpa_pct'][tgt]), r['gpa_pct'][tgt], 0.5)
            sc = S['rank'][k, tgt]
            o = np.lexsort((-sc, -gp))                         # 主鍵：毛利／資產百分位高者先；同分 SURGE 高者先
            newrank = np.empty(len(tgt)); newrank[o] = np.arange(len(tgt), 0, -1, dtype=float)
            rank[k, tgt] = newrank
            o0 = np.argsort(-sc, kind='stable')
            rec.append(dict(month=x['month'], k=int(k), orig=[str(P.COLS[j]) for j in tgt[o0]], order=[str(P.COLS[j]) for j in tgt[o]], removed=[]))
        else:
            raise ValueError(cand)
    S2['sig'] = sig; S2['rank'] = rank
    return S2, rec


def sim(P, S, U, cap=1e6, cost='strict', block=None, cap_pct=0.05, fresh=False):
    """F1＋E0、嚴成本（預設）、W10_70、空位 0050；原引擎 s10lib.simulate_s10。"""
    S1 = fresh_from(S, VALID_FIRST_MONTH) if fresh else S
    S2, ekw = s10lib.variant_S(S1, 'E0')
    return s10lib.simulate_s10(P, S2, U, cost=COSTS[cost], capital=cap, cap_pct=cap_pct, sat_w=0.7, sizing='W10', lock=False, block=block, **ekw)
