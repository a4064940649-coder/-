#!/usr/bin/env python3
"""S91 共用程式庫：營收驚奇名單加「融資／融券／借券」過濾（訓練→凍結→驗證一次）。包裝方式照 S88（s88lib），引擎載入照 S89（s89lib）。
引擎：正式管線（agents/official_nav/run_official.py 的 E0 寫法；五個開關全開：EXRIGHT_FIX、DEADLINE_FIX、INS15_MASK、REDUCE_FIX、0050 官方含息）。
模擬＝s10lib.simulate_s10 原函式（W10_70、嚴成本、空位 0050），不做副本；規則只改 make_signal_s10 輸出的 sig／rank（買進名單），出場 X 不動。
資料（只讀）：raw/tsd_extra 的 margin_twse／margin_tpex（融資融券，張）、sbl_twse／sbl_tpex（借券賣出餘額，股）、daily（發行股數 shares、成交股數 volume）。
時點：訊號日 d（列 k）收盤後決定、次日開盤買。融資融券（MI_MARGN）與借券賣出餘額（TWT93U）是交易所當天晚上公布（記憶中約 21 時前後；本機無法查證）。
  LAG＝1（判定用，保守）：只用到 d−1 那天（列 k−1）的餘額；LAG＝0（敏感度）：用到 d 當天（列 k）。
特徵（長度 N；缺資料＝NaN；所有旗標在缺值時一律 False＝保留）：令 e＝k−LAG。
  m_use    融資使用率代理＝融資餘額（張）×1000 ÷ 發行股數（列 e）
  m_dv     融資餘額（股）÷ 近 20 日均量（股；列 e−19～e，至少 10 天有量）
  m_d5s／m_d20s  融資餘額近 5／20 個交易日增減（股）÷ 發行股數（列 e 對 e−5／e−20；兩端都要有值）
  m_g5／m_g20    融資餘額近 5／20 日增幅（比率）；期初餘額 < 100 張 → NaN（基期太小沒意義）
  sm_ratio 券資比＝融券餘額 ÷ 融資餘額（融資餘額 ≥ 100 張才算）
  s_sh     融券餘額（股）÷ 發行股數
  sbl_sh   借券賣出餘額（股）÷ 發行股數；sbl_d20s＝近 20 日增減 ÷ 發行股數
  short_sh ＝（融券餘額＋借券賣出餘額）÷ 發行股數；short_dv＝同分子 ÷ 近 20 日均量
  *_pct    上述各項在同期股池 U[k]（上市、非創新板）有值者中的百分位（0～1，越大值越高）
  no_margin 列 e 沒有融資融券資料列，或融資限額＝0（停資／不得融資）
規則與格子：RULES（規則 id → 布林判定式），× 兩種處理（Z＝被擋名額不遞補，錢照引擎放 0050；F＝由 L_MA 第 11～20 名中 RET20 > 10% 且沒被同一規則擋掉者依 SURGE 遞補）。
一律 python3 -I -B。暫存只有面板快取（硬連結既有檔，不占空間），放 scratchpad/s91/cache。"""
import sys, gc
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
AG = HERE.parent
RAWX = AG.parent / 'raw' / 'tsd_extra'
TMP = Path('/tmp/claude-0/-home-user/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/s91/cache')
TMP.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(AG / 's19_xq_threshold'))
import s19lib                                              # noqa: E402
from s19lib import s10lib, s9lib, s8lib, s7lib, s6lib, s5lib, s4lib, s3lib, s2lib, stlib, cslib, COSTS   # noqa: E402,F401
for _mod, _attr in ((s19lib, 'S19'), (s10lib, 'S10'), (s9lib, 'S9'), (s8lib, 'S8'), (s7lib, 'S7'), (s6lib, 'S6'), (s5lib, 'S5'),
                    (s4lib, 'S4'), (s3lib, 'S3'), (s2lib, 'S2'), (stlib, 'S1'), (cslib, 'TMP')):
    setattr(_mod, _attr, TMP)
sys.path.insert(0, str(RAWX))
import tsd_extra_load as XL                                # noqa: E402

CAPS = (6e5, 8e5, 1e6, 1.5e6, 2e6)
SHIFTS = (0, 1, 2, 3)
SEGMENTS = dict(s19lib.SEGMENTS)
SEGV = ('2022', '2024-07~2025-04')
VALID_FIRST_MONTH = pd.Period('2021-12', 'M')              # 驗證期：營收月份 ≥ 2021-12（訊號日 2022-01 起）
TARGET = dict(since2019=0.4886246519182116, full=0.3446294016617564, maxdd=-0.3160938700293674, n_closed=381, n_all=389, paths_median=0.4641920784536262)
P0 = (2876, 2142, '2026-10-22')
LAG_MAIN = 1
MIN_BASE = 100                                             # 張：增幅、券資比的基期下限
FEATS = ('m_use', 'm_dv', 'm_d5s', 'm_d20s', 'm_g5', 'm_g20', 'sm_ratio', 's_sh', 'sbl_sh', 'sbl_d20s', 'short_sh', 'short_dv')
FILL_DEPTH = 20
MODES = ('Z', 'F')

# ---------------------------------------------------------------- 規則（看完訓練期分組後、PREREG 之前定案；之後不得改）
RULES = {}
try:
    from s91rules import RULES                             # noqa: F401,E402
except ImportError:
    pass
CELLS = tuple(f'{r}|{m}' for r in RULES for m in MODES)


def load():
    """逐行照 official_nav/run_official.py 的載入（只留 E0 需要的；同 s89lib.load）。回傳 P, U, RT。"""
    assert cslib.EXRIGHT_FIX is False and cslib.DEADLINE_FIX is False and cslib.INS15_MASK is False
    assert cslib.REDUCE_FIX is False and cslib.E50_SOURCE is None
    cslib.EXRIGHT_FIX = True; cslib.DEADLINE_FIX = True; cslib.INS15_MASK = True
    cslib.REDUCE_FIX = True; cslib.E50_SOURCE = cslib.E50_OFFICIAL_CSV
    P = s19lib.load()
    assert P.exright_fix and P.reduce_fix and P.e50_source == str(cslib.E50_OFFICIAL_CSV)
    assert (P.T, P.N, str(P.IDX[-1].date())) == P0, 'P0 失敗'
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


def fresh_from(S, first_month=VALID_FIRST_MONTH):
    """同 s20lib.fresh_from：只保留營收月份 ≥ first_month 的進場訊號；出場 X 不動。"""
    S2 = dict(S)
    sig = S['sig'].copy(); rank = S['rank'].copy()
    for x in S['info']:
        if pd.Period(x['month'], 'M') < first_month:
            sig[x['k'], :] = False; rank[x['k'], :] = np.nan
    S2['sig'] = sig; S2['rank'] = rank
    return S2


def sim(P, S, U, cap=1e6, cost='strict', block=None, fresh=False):
    """W10_70、嚴成本、空位 0050；原引擎 s10lib.simulate_s10（寫法同 run_official.py 的 sim('E0', …)）。"""
    S1 = fresh_from(S) if fresh else S
    return s10lib.simulate_s10(P, S1, U, sat_w=0.7, sizing='W10', lock=False, cost=COSTS[cost], capital=cap, block=block)


def seg_dd(nav, keys=tuple(SEGMENTS)):
    o = {}
    for k in keys:
        a, b = SEGMENTS[k]
        s = nav.loc[a:b].dropna()
        o[k] = float((s / s.cummax() - 1).min()) if len(s) > 2 else float('nan')
    return o


def top5_share(lg, nav, cap):
    x = lg[lg['why'] != 'open_end'].sort_values('pnl', ascending=False).head(5)
    return float(x['pnl'].sum() / (nav.iloc[-1] - cap))


def pct_in(x, mask):
    """x 在 mask 內有值者的百分位（0～1，平均名次 ÷ 個數；越大＝值越高）；其餘 NaN。（同 s70lib.pct_in）"""
    o = np.full(len(x), np.nan)
    ok = mask & np.isfinite(x)
    n = int(ok.sum())
    if n >= 30:
        o[ok] = pd.Series(x[ok]).rank(method='average').to_numpy() / n
    return o


# ---------------------------------------------------------------- 融資融券／借券面板（T×N，float32；沒有資料列＝NaN）
class Chips:
    def __init__(self, P):
        idx = pd.DatetimeIndex(P.IDX); codes = [str(c) for c in P.COLS]
        a, b = str(idx[0].date()), str(idx[-1].date())
        self.T, self.N = len(idx), len(codes)
        dmap = {str(d.date()): i for i, d in enumerate(idx)}
        cmap = {c: j for j, c in enumerate(codes)}

        def panel(names, cols, scale=None):
            out = {c: np.full((self.T, self.N), np.nan, np.float32) for c in cols}
            for nm in names:                                 # 後讀的蓋過先讀的：tpex 先、twse 後（同日兩邊都有時以上市為準）
                for y in range(idx[0].year, idx[-1].year + 1):
                    df = XL.load(nm, max(a, f'{y}-01-01'), min(b, f'{y}-12-31'), columns=['date', 'code'] + list(cols))
                    df = df[df['code'].isin(cmap)]
                    ti = df['date'].astype(str).map(dmap); ok = ti.notna().to_numpy()
                    t = ti.to_numpy()[ok].astype(int); j = df['code'].map(cmap).to_numpy()[ok]
                    for c in cols:
                        out[c][t, j] = df[c].to_numpy()[ok].astype(np.float32)
                    del df
            return out
        m = panel(('margin_tpex', 'margin_twse'), ('m_balance', 'm_limit', 's_balance'))
        self.MB, self.ML, self.SB = m['m_balance'], m['m_limit'], m['s_balance']        # 張
        self.SBL = panel(('sbl_tpex', 'sbl_twse'), ('sbl_balance',))['sbl_balance']     # 股
        d = pd.concat([XL.load('daily', max(a, f'{y}-01-01'), min(b, f'{y}-12-31'), columns=['date', 'code', 'market', 'shares', 'volume'])
                       for y in range(idx[0].year, idx[-1].year + 1)], ignore_index=True)
        d = d[d['market'].isin(['twse', 'tpex']) & d['code'].isin(cmap)].drop_duplicates(['date', 'code'], keep='last')
        ti = d['date'].astype(str).map(dmap); ok = ti.notna().to_numpy()
        t = ti.to_numpy()[ok].astype(int); j = d['code'].map(cmap).to_numpy()[ok]
        self.SHR = np.full((self.T, self.N), np.nan, np.float64); self.SHR[t, j] = d['shares'].to_numpy()[ok].astype(float)
        V = np.full((self.T, self.N), np.nan, np.float64); V[t, j] = d['volume'].to_numpy()[ok].astype(float)
        del d
        self.SHR[self.SHR <= 0] = np.nan
        vv = pd.DataFrame(V)
        self.AV20 = np.array(vv.rolling(20, min_periods=10).mean().to_numpy(), dtype=float, copy=True)
        self.AV20[self.AV20 <= 0] = np.nan
        del V, vv; gc.collect()
        self._cache = {}

    def raw(self, k, lag):
        """訊號列 k、延遲 lag：連續特徵（長度 N）。"""
        key = (int(k), int(lag))
        if key in self._cache:
            return self._cache[key]
        e = k - lag
        f8 = lambda x: np.asarray(x, np.float64)
        mb, ml, sb, sbl, shr, av = f8(self.MB[e]), f8(self.ML[e]), f8(self.SB[e]), f8(self.SBL[e]), self.SHR[e], self.AV20[e]
        mb5, mb20, sbl20 = f8(self.MB[e - 5]), f8(self.MB[e - 20]), f8(self.SBL[e - 20])
        with np.errstate(invalid='ignore', divide='ignore'):
            F = dict(m_use=mb * 1000 / shr, m_dv=mb * 1000 / av, m_d5s=(mb - mb5) * 1000 / shr, m_d20s=(mb - mb20) * 1000 / shr,
                     m_g5=np.where(mb5 >= MIN_BASE, mb / mb5 - 1, np.nan), m_g20=np.where(mb20 >= MIN_BASE, mb / mb20 - 1, np.nan),
                     sm_ratio=np.where(mb >= MIN_BASE, sb / mb, np.nan), s_sh=sb * 1000 / shr, sbl_sh=sbl / shr, sbl_d20s=(sbl - sbl20) / shr,
                     short_sh=(sb * 1000 + sbl) / shr, short_dv=(sb * 1000 + sbl) / av)
        F = {c: np.where(np.isfinite(v), v, np.nan) for c, v in F.items()}
        F['no_margin'] = ~np.isfinite(mb) | ~(ml > 0)
        F['mb'] = mb
        self._cache[key] = F
        return F


def features(P, U, CH, k, lag=LAG_MAIN):
    """訊號列 k：連續特徵＋同期股池百分位（*_pct）。"""
    key = ('F', int(k), int(lag))
    if key in CH._cache:
        return CH._cache[key]
    F = dict(CH.raw(k, lag))
    for c in FEATS:
        F[c + '_pct'] = pct_in(F[c], U[k])
    CH._cache[key] = F
    return F


def order_ma(P, U, RT, x, shift):
    """重建訊號列的 L_MA 排序（同 s10lib.make_signal_s10）。回傳 (o_ma, s)。"""
    m = pd.Period(x['month'], 'M'); k = x['k']
    s = RT['S'].loc[m].to_numpy()
    s = cslib.ins15_mask_row(P, m, k, s)
    ok = U[k] & np.isfinite(s) & P.F_MA60[k]
    o = np.flatnonzero(ok)[np.argsort(-s[ok], kind='stable')]
    return o, s


def parse_cell(cell):
    """'<規則>|<Z/F>' 或 '<規則>|<Z/F>|L0'（延遲 0 的敏感度版）→ (rule, mode, lag)。"""
    p = cell.split('|')
    return p[0], p[1], (0 if len(p) > 2 and p[2] == 'L0' else LAG_MAIN)


def apply_cell(P, U, RT, CH, S, cell, shift=0):
    """cell＝'CUR' 或 '<規則>|<Z/F>[|L0]'。回傳 (S', 記錄 list[dict(month,k,orig,removed,added)])。"""
    rec = []
    if cell == 'CUR':
        return S, rec
    rule, mode, lag = parse_cell(cell)
    fn = RULES[rule]
    S2 = dict(S); sig = S['sig'].copy(); rank = S['rank'].copy()
    for x in S['info']:
        k = x['k']
        tgt = np.flatnonzero(S['sig'][k])
        if not len(tgt):
            continue
        F = features(P, U, CH, k, lag)
        with np.errstate(invalid='ignore'):
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


def trade_features(P, U, CH, lg, lag=LAG_MAIN, suffix=''):
    """交易明細每筆補上進場訊號日的特徵（連續值＋百分位）。"""
    rows = []
    for k, j in zip(lg['sig_t'].to_numpy(), lg['j'].to_numpy()):
        F = features(P, U, CH, int(k), lag); j = int(j)
        rows.append({n + suffix: (bool(v[j]) if v.dtype == bool else float(v[j])) for n, v in F.items()})
    return pd.DataFrame(rows)
