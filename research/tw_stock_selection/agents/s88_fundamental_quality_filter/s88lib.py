#!/usr/bin/env python3
"""S88 共用程式庫：營收驚奇名單加「財報品質」過濾（訓練→凍結→驗證一次）。
引擎與鏈：完全沿用 s70lib（→ s47lib → s20lib → s17lib → s10lib，全部不改；正式開關 EXRIGHT_FIX、DEADLINE_FIX、INS15_MASK 全開；
REDUCE_FIX 與 E50_SOURCE 維持預設關閉）。模擬＝s70lib.sim（F1＋E0、嚴成本、W10_70、空位 0050；原引擎 s10lib.simulate_s10）。
財報（只讀）：
  o_pead_margin_surprise_low_attention/data/fin_long.parquet（MOPS XBRL 單季：營收、毛利、營業利益、歸屬母公司淨利、EPS、總資產；2013Q1～2026Q2）
  o_pead_margin_surprise_low_attention/replicate_o10/data/fin_raw.parquet（同一批原始列：營業現金流累計 ocf_ytd、本期淨利累計 ni_ytd、負債）
  o_pead_margin_surprise_low_attention/data/avail_dates.parquet（市場可得日 A；只用來排除「訊號日還沒公告」）
  應收帳款、存貨：本機沒有（封存的 bs_hist 內容讀不到）→ 用「應計比率」「營業現金流」「總資產成長－營收成長」當替代。
時點（同 S28／S70，s70lib.latest_kq_by_deadline）：訊號日 d，季 kq 可用 ⇔ 法定期限 < d 且（A 缺值或 A < d）；晚交或無資料往回 1 季。
  年增比較用同一檔的 kq−4（比 kq 早一年，必然已公告）。
旗標（True＝品質差；任何需要的數字缺值 → False＝保留）：
  OPI_DOWN  最近一季營業利益（金額）< 去年同季
  GM_DOWN   最近一季毛利率 < 去年同季毛利率（兩季營收都 > 0、|比率| ≤ 5）
  OPM_DOWN  最近一季營業利益率 < 去年同季（＝「營業利益成長跟不上營收成長」的等價寫法；只作描述）
  LOSS      最近一季營業利益 < 0（＝S70 C1）
  NI_LOSS   最近一季歸屬母公司淨利 < 0（只作描述）
  OCF_NEG   近四季營業現金流合計 < 0（金融保險證券業不判）
  ACC_HI    應計比率＝(近四季本期淨利 − 近四季營業現金流) ÷ 最近一季總資產，落在同期股池（非金融）最高 20%（金融業不判）
  AG_GAP    總資產年增率 − 最近一季營收年增率 > 0.10（資產膨脹快於營收；應收＋存貨的粗替代；金融業不判）
規則（PREREG 的 10 條）× 兩種處理（Z＝不遞補，錢照引擎放 0050；F＝由 L_MA 第 11～20 名中 RET20 > 10% 且沒被同一規則擋掉者依 SURGE 遞補）。
一律 python3 -I -B。中間檔放 scratchpad/s88/cache。"""
import sys
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
AG = HERE.parent
TMP = Path('/tmp/claude-0/-home-user/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/s88')
CACHE = TMP / 'cache'
sys.path.insert(0, str(AG / 's70_profitability_confirm'))
import s70lib as L70                                       # noqa: E402
from s70lib import s20lib, s17lib, s10lib, s4lib, s2lib, stlib, cslib, COSTS   # noqa: E402,F401
L70.TMP = TMP; L70.CACHE = CACHE
for _m, _a in ((s20lib, 'S20'), (s17lib, 'S17'), (s10lib, 'S10'), (L70.s9lib, 'S9'), (L70.s8lib, 'S8'), (L70.s7lib, 'S7'), (L70.s6lib, 'S6'), (L70.s5lib, 'S5'),
               (s4lib, 'S4'), (L70.s3lib, 'S3'), (L70.s2lib, 'S2'), (stlib, 'S1'), (cslib, 'TMP')):
    setattr(_m, _a, CACHE)
assert cslib.EXRIGHT_FIX is True and cslib.DEADLINE_FIX is True and cslib.INS15_MASK is True, '正式開關必須全開'
assert cslib.REDUCE_FIX is False and cslib.E50_SOURCE is None, '本題不開 REDUCE_FIX／E50_SOURCE'

PEAD = AG / 'o_pead_margin_surprise_low_attention'
CAPS = L70.CAPS; SHIFTS = L70.SHIFTS; SEGV = L70.SEGV
ACC = dict(L70.ACC)                                        # 三開關口徑（2026-11-17～11-20 的正式值；封存於 official_nav/archive_2026-11-21_pre_v2）
VALID_FIRST_MONTH = L70.VALID_FIRST_MONTH
deadline = L70.deadline; latest_kq_by_deadline = L70.latest_kq_by_deadline
MAX_BACK = 1
FIN_IND = ('bd', 'fh', 'basi', 'ins')                      # 證券、金控、銀行、保險：現金流／資產類旗標不判
ACC_PCT = 0.80
AG_X = 0.10
FLAGS = ('OPI_DOWN', 'GM_DOWN', 'OPM_DOWN', 'LOSS', 'NI_LOSS', 'OCF_NEG', 'ACC_HI', 'AG_GAP')
RULES = {                                                   # 規則 id → 判定式（旗標名的布林組合）
    'R01_OPI_DOWN': lambda f: f['OPI_DOWN'],
    'R02_GM_DOWN': lambda f: f['GM_DOWN'],
    'R03_LOSS': lambda f: f['LOSS'],
    'R04_ACC_HI': lambda f: f['ACC_HI'],
    'R05_OCF_NEG': lambda f: f['OCF_NEG'],
    'R06_AG_GAP': lambda f: f['AG_GAP'],
    'R07_OPI_GM_DOWN': lambda f: f['OPI_DOWN'] & f['GM_DOWN'],
    'R08_LOSS_OCFNEG': lambda f: f['LOSS'] & f['OCF_NEG'],
    'R09_CASH_BAD': lambda f: f['ACC_HI'] | f['OCF_NEG'],
    'R10_ANY2': lambda f: (f['OPI_DOWN'].astype(int) + f['GM_DOWN'].astype(int) + f['LOSS'].astype(int) + f['ACC_HI'].astype(int) + f['OCF_NEG'].astype(int)) >= 2,
}
MODES = ('Z', 'F')                                          # Z＝放 0050（不遞補）；F＝往下遞補
CELLS = tuple(f'{r}|{m}' for r in RULES for m in MODES)     # 20 格
FILL_DEPTH = 20


def load():
    CACHE.mkdir(parents=True, exist_ok=True)
    P = s20lib.load()
    assert P.exright_fix is True and not getattr(P, 'reduce_fix', False)
    return P


base = L70.base; slim = L70.slim; sim = L70.sim; fresh_from = L70.fresh_from


class Fin:
    """code × kq 矩陣；row(d)＝訊號日 d 每檔當時可得的最近一季特徵與旗標原料。"""

    def __init__(self, P):
        f = pd.read_parquet(PEAD / 'data' / 'fin_long.parquet', columns=['code', 'period', 'y', 'q', 'industry', 'eps_s', 'rev_s', 'gp_s', 'opi_s', 'nip_s', 'assets'])
        r = pd.read_parquet(PEAD / 'replicate_o10' / 'data' / 'fin_raw.parquet', columns=['stock_id', 'period', 'ocf_ytd', 'ni_ytd', 'liabilities'])
        assert not r.duplicated(['stock_id', 'period']).any()
        f = f.merge(r.rename(columns={'stock_id': 'code'}), on=['code', 'period'], how='left', validate='1:1')
        f = f[f['code'].str.fullmatch(r'[1-9]\d{3}')].copy()
        f['kq'] = f['y'] * 4 + f['q'] - 1
        self.kmin, self.kmax = int(f['kq'].min()), int(f['kq'].max())
        ks = np.arange(self.kmin, self.kmax + 1); self.ks = ks
        codes = [str(c) for c in P.COLS]
        piv = lambda c: f.pivot(index='code', columns='kq', values=c).reindex(index=codes, columns=ks).to_numpy(float)
        self.rev, self.gp, self.opi, self.nip, self.eps, self.ast, self.liab = (piv(c) for c in ('rev_s', 'gp_s', 'opi_s', 'nip_s', 'eps_s', 'assets', 'liabilities'))
        ocf_y, ni_y = piv('ocf_ytd'), piv('ni_ytd')
        self.anydata = np.isfinite(self.rev) | np.isfinite(self.opi) | np.isfinite(self.nip) | np.isfinite(self.eps)
        qn = ks % 4 + 1

        def ttm_from_ytd(y):
            """近四季＝本年累計 ＋ 去年全年 − 去年同季累計；Q4＝本年累計。"""
            o = np.full_like(y, np.nan)
            for i, kq in enumerate(ks):
                if qn[i] == 4:
                    o[:, i] = y[:, i]
                else:
                    i_py4 = i - int(qn[i]); i_pyq = i - 4       # 去年 Q4、去年同季
                    if i_pyq >= 0 and i_py4 >= 0:
                        o[:, i] = y[:, i] + y[:, i_py4] - y[:, i_pyq]
            return o
        self.ocf_ttm, self.ni_ttm = ttm_from_ytd(ocf_y), ttm_from_ytd(ni_y)
        a = pd.read_parquet(PEAD / 'data' / 'avail_dates.parquet', columns=['code', 'y', 'q', 'A'])
        a = a[a['code'].isin(set(codes))].copy(); a['kq'] = a['y'] * 4 + a['q'] - 1
        self.A = a.drop_duplicates(['code', 'kq']).pivot(index='code', columns='kq', values='A').reindex(index=codes, columns=ks).to_numpy('datetime64[ns]')
        ind = f.sort_values('kq').groupby('code')['industry'].last().reindex(codes)
        self.is_fin = ind.isin(FIN_IND).to_numpy()
        self.ind = ind
        self._cache = {}

    def _lag4(self, x, i):
        return x[:, i - 4] if i - 4 >= 0 else np.full(x.shape[0], np.nan)

    def row(self, d):
        key = pd.Timestamp(d)
        if key in self._cache:
            return self._cache[key]
        k0 = latest_kq_by_deadline(key)
        N = self.rev.shape[0]
        cols = ('rev', 'rev4', 'gp', 'gp4', 'opi', 'opi4', 'nip', 'ast', 'ast4', 'ocf_ttm', 'ni_ttm', 'kq', 'age_days')
        out = {c: np.full(N, np.nan) for c in cols}
        done = np.zeros(N, bool)
        d64 = np.datetime64(key, 'ns')
        for kq in range(k0, k0 - MAX_BACK - 1, -1):
            i = kq - self.kmin
            if i < 0 or i >= self.rev.shape[1]:
                continue
            Ai = self.A[:, i]
            late = ~np.isnat(Ai) & ~(Ai < d64)
            use = ~done & self.anydata[:, i] & ~late
            qe = pd.Timestamp(kq // 4, 3 * (kq % 4 + 1), 1) + pd.offsets.MonthEnd(0)
            vals = dict(rev=self.rev[:, i], rev4=self._lag4(self.rev, i), gp=self.gp[:, i], gp4=self._lag4(self.gp, i), opi=self.opi[:, i], opi4=self._lag4(self.opi, i),
                        nip=self.nip[:, i], ast=self.ast[:, i], ast4=self._lag4(self.ast, i), ocf_ttm=self.ocf_ttm[:, i], ni_ttm=self.ni_ttm[:, i],
                        kq=np.full(N, float(kq)), age_days=np.full(N, float((key - qe).days)))
            for c in cols:
                out[c][use] = vals[c][use]
            done |= use
        self._cache[key] = out
        return out


def _ratio(a, b):
    with np.errstate(invalid='ignore', divide='ignore'):
        x = a / np.where(b > 0, b, np.nan)
    return np.where(np.abs(x) <= 5, x, np.nan)


def features(P, U, FN, k):
    """訊號列 k：連續特徵＋布林旗標（長度 N；缺值→旗標 False）。"""
    r = FN.row(P.IDX[k])
    nf = ~FN.is_fin
    gm, gm4 = _ratio(r['gp'], r['rev']), _ratio(r['gp4'], r['rev4'])
    opm, opm4 = _ratio(r['opi'], r['rev']), _ratio(r['opi4'], r['rev4'])
    with np.errstate(invalid='ignore', divide='ignore'):
        rev_g = np.where((r['rev'] > 0) & (r['rev4'] > 0), r['rev'] / r['rev4'] - 1, np.nan)
        ast_g = np.where((r['ast'] > 0) & (r['ast4'] > 0), r['ast'] / r['ast4'] - 1, np.nan)
        acc = np.where(nf & (r['ast'] > 0), (r['ni_ttm'] - r['ocf_ttm']) / r['ast'], np.nan)
        ocf_a = np.where(nf & (r['ast'] > 0), r['ocf_ttm'] / r['ast'], np.nan)
    acc_pct = L70.pct_in(acc, U[k])
    F = dict(gm=gm, gm_d=gm - gm4, opm=opm, opm_d=opm - opm4, rev_g=rev_g, ast_g=ast_g, ag_gap=ast_g - rev_g, acc=acc, acc_pct=acc_pct, ocf_a=ocf_a,
             opi=r['opi'], opi4=r['opi4'], nip=r['nip'], kq=r['kq'], age_days=r['age_days'])
    fin = np.isfinite
    with np.errstate(invalid='ignore'):
        F['OPI_DOWN'] = fin(r['opi']) & fin(r['opi4']) & (r['opi'] < r['opi4'])
        F['GM_DOWN'] = fin(F['gm_d']) & (F['gm_d'] < 0)
        F['OPM_DOWN'] = fin(F['opm_d']) & (F['opm_d'] < 0)
        F['LOSS'] = fin(r['opi']) & (r['opi'] < 0)
        F['NI_LOSS'] = fin(r['nip']) & (r['nip'] < 0)
        F['OCF_NEG'] = nf & fin(r['ocf_ttm']) & (r['ocf_ttm'] < 0)
        F['ACC_HI'] = fin(acc_pct) & (acc_pct >= ACC_PCT)
        F['AG_GAP'] = nf & fin(F['ag_gap']) & (F['ag_gap'] > AG_X)
    return F


def order_ma(P, U, RT, x, shift):
    """重建訊號列的 L_MA 排序（同 s10lib.make_signal_s10）。回傳 (o_ma, s)。"""
    m = pd.Period(x['month'], 'M'); k = x['k']
    s = RT['S'].loc[m].to_numpy()
    s = cslib.ins15_mask_row(P, m, k, s)
    ok = U[k] & np.isfinite(s) & P.F_MA60[k]
    o = np.flatnonzero(ok)[np.argsort(-s[ok], kind='stable')]
    return o, s


def apply_cell(P, U, RT, FN, S, cell, shift=0):
    """cell＝'CUR' 或 '<規則>|<Z/F>'。回傳 (S', 記錄 list[dict(month,k,orig,removed,added)])。"""
    rec = []
    if cell == 'CUR':
        return S, rec
    rule, mode = cell.split('|')
    fn = RULES[rule]
    S2 = dict(S); sig = S['sig'].copy(); rank = S['rank'].copy()
    for x in S['info']:
        k = x['k']
        tgt = np.flatnonzero(S['sig'][k])
        if not len(tgt):
            continue
        F = features(P, U, FN, k)
        bad = np.asarray(fn(F), bool)
        rm = [int(j) for j in tgt if bad[j]]
        added = []
        if rm:
            sig[k, rm] = False; rank[k, rm] = np.nan
            if mode == 'F':
                o, s = order_ma(P, U, RT, x, shift)
                pool = o[:10]
                chk = pool[P.RET20[k][pool] > 0.10][:10]
                assert set(int(j) for j in chk) == set(int(j) for j in tgt), ('名單重建不符', x['month'])
                for j in o[10:FILL_DEPTH]:
                    if len(added) >= len(rm):
                        break
                    if P.RET20[k][j] > 0.10 and not bad[j]:
                        added.append(int(j)); sig[k, j] = True; rank[k, j] = s[j]
        rec.append(dict(month=x['month'], k=int(k), orig=[str(P.COLS[j]) for j in tgt], removed=[str(P.COLS[j]) for j in rm], added=[str(P.COLS[j]) for j in added]))
    S2['sig'] = sig; S2['rank'] = rank
    return S2, rec


def trade_features(P, U, FN, lg):
    """交易明細每筆補上進場訊號日的特徵與旗標。"""
    rows = []
    for k, c in zip(lg['sig_t'].to_numpy(), lg['code'].astype(str)):
        F = features(P, U, FN, int(k)); j = P.ci[c]
        rows.append({n: (bool(v[j]) if v.dtype == bool else float(v[j])) for n, v in F.items()})
    return pd.concat([lg.reset_index(drop=True), pd.DataFrame(rows)], axis=1)
