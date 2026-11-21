#!/usr/bin/env python3
"""S90 共用程式庫：三大法人買賣超能不能幫營收驚奇（SURGE）分辨好壞交易（過濾／確認／重排／放寬入口）。
引擎：正式管線（agents/official_nav/run_official.py 的 E0｜W10_70 寫法；EXRIGHT_FIX、DEADLINE_FIX、INS15_MASK、REDUCE_FIX、0050 官方含息 五個開關全開）。
      模擬＝s10lib.simulate_s10（原樣呼叫，不改）；本題只改「訊號列的買進名單」（sig／rank），出場 X 完全不動。
法人資料（只讀）：raw/tsd_extra/inst_twse（上市三大法人逐股逐日買賣超，單位＝股，2015-01-05 起）；
成交量與發行股數：raw/tsd_extra/daily（volume 成交股數、shares 發行股數）。某股某日沒有法人列＝當日三大法人都沒買賣 → 0。
時點：訊號列 k（訊號日收盤後）只用列 ≤ k 的法人與成交量（T86 每日收盤後公布）；次一交易日開盤成交（引擎原規則）。
特徵（列 k、每檔）：
  fvN／tvN／avN＝近 N 個交易日（含 k）外資／投信／三大法人合計買賣超股數 ÷ 同 N 日成交股數合計（N＝5、10、20）；成交量合計 ≤ 0 → 缺值
  fsN／tsN／asN＝同上 ÷ 列 k 的發行股數（流通量代理）
  TFB（投信初買）＝投信近 5 日合計買超 > 0 且 之前 20 日（k−24～k−5）合計 ≤ 0
  streak_a／streak_f／streak_t＝到 k 為止三大合計／外資／投信連續買超（> 0）天數
  pct(x)＝x 在同日股池（U[k]、有值）內的百分位（0～1）
格子（30 格，PREREG 固定）見 CELLS。Z＝被擋的名額不遞補（錢照引擎放 0050）；F＝由 L_MA 第 11～20 名中 RET20 > 10% 且通過同一條件者依 SURGE 遞補。
暫存：scratchpad/s90/cache（面板快取＋法人矩陣，用完刪）。一律 python3 -I -B。"""
import sys, gc
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
AG = HERE.parent
RAW = AG.parent / 'raw' / 'tsd_extra'
TMP = Path('/tmp/claude-0/-home-user/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/s90/cache')
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
VALID_FIRST_MONTH = pd.Period('2021-12', 'M')              # 驗證期：營收月份 ≥ 2021-12（訊號日 2022-01 起；同 s20lib／S88／S89）
TARGET = dict(since2019=0.4886246519182116, full=0.3446294016617564, maxdd=-0.3160938700293674, n_closed=381, n_all=389, paths_median=0.4641920784536262)
FILL_DEPTH = 20
PCT_LO, PCT_HI = 0.20, 0.80

# ---------------------------------------------------------------- 格子（30 格，事前固定）
BAD = {       # (a) 過濾：True＝法人大賣超 → 不買
    'FA_F20L': lambda f: f['p_fv20'] <= PCT_LO,            # 外資近 20 日買賣超占成交量，落在同日股池最低 20%
    'FA_F5L': lambda f: f['p_fv5'] <= PCT_LO,              # 外資近 5 日
    'FA_A20L': lambda f: f['p_av20'] <= PCT_LO,            # 三大合計近 20 日
    'FA_A5L': lambda f: f['p_av5'] <= PCT_LO,              # 三大合計近 5 日
    'FA_T20N': lambda f: f['ts20'] < 0,                    # 投信近 20 日合計賣超
    'FA_FT20N': lambda f: (f['fs20'] < 0) & (f['ts20'] < 0),   # 外資、投信近 20 日都賣超
}
GOOD = {      # (b) 確認：True＝法人買超 → 才買
    'CB_F20P': lambda f: f['fs20'] > 0,
    'CB_T20P': lambda f: f['ts20'] > 0,
    'CB_A20P': lambda f: f['as20'] > 0,
    'CB_A5P': lambda f: f['as5'] > 0,
    'CB_STRK3': lambda f: f['streak_a'] >= 3,              # 三大合計連買 ≥ 3 日
}
RERANK = {'RK_A20': 'av20', 'RK_F20': 'fv20', 'RK_T20': 'tv20', 'RK_A5': 'av5'}   # (c) 前 20 名內 SURGE 名次＋法人名次重排取前 10
WIDEN = {     # (d) 放寬入口：True＝動能沒過也可以進
    'WD_TFB': lambda f: f['TFB'],                          # 前 10 名、RET20 ≤ 10%、投信初買
    'WD_T20P': lambda f: f['ts20'] > 0,                    # 前 10 名、RET20 ≤ 10%、投信近 20 日買超
    'WD_A20H': lambda f: f['p_av20'] >= PCT_HI,            # 前 10 名、RET20 ≤ 10%、三大合計近 20 日占量在股池最高 20%
    'WD_TFB20': lambda f: f['TFB'],                        # WD_TFB ＋ 第 11～20 名中 RET20 > 10% 且投信初買者也進名單
}
CELLS = tuple([f'{r}|{m}' for r in BAD for m in 'ZF'] + [f'{r}|{m}' for r in GOOD for m in 'ZF'] + list(RERANK) + list(WIDEN))
assert len(CELLS) == 30


def kind(cell):
    r = cell.split('|')[0]
    return 'filter' if r in BAD else 'confirm' if r in GOOD else 'rerank' if r in RERANK else 'widen'


# ---------------------------------------------------------------- 引擎載入（逐行照 official_nav/run_official.py，只留 E0 需要的）
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
    """同 s20lib.fresh_from：只保留營收月份 ≥ first_month 的進場訊號；出場 X 不動。"""
    S2 = dict(S); sig = S['sig'].copy(); rank = S['rank'].copy()
    for x in S['info']:
        if pd.Period(x['month'], 'M') < first_month:
            sig[x['k'], :] = False; rank[x['k'], :] = np.nan
    S2['sig'] = sig; S2['rank'] = rank
    return S2


def sim(P, S, U, cap=1e6, cost='strict', block=None):
    """W10_70、嚴成本、空位 0050（原引擎）。"""
    return s10lib.simulate_s10(P, S, U, sat_w=0.7, sizing='W10', lock=False, cost=COSTS[cost], capital=cap, block=block)


def seg_dd(nav, keys=SEGV):
    o = {}
    for k in keys:
        a, b = SEGMENTS[k]
        s = nav.loc[a:b].dropna()
        o[k] = float((s / s.cummax() - 1).min()) if len(s) > 2 else float('nan')
    return o


# ---------------------------------------------------------------- 法人矩陣
FLOW_NPZ = TMP / 'flow_twse.npz'


def build_flow(P):
    """由 raw/tsd_extra 建 T×N 矩陣（對齊 P.IDX × P.COLS）：F、T、D、A（股）、VOL（成交股數）、SH（發行股數）、HAVE（該日有法人檔）。回傳 (dict, 檢查)。"""
    sys.path.insert(0, str(RAW))
    from tsd_extra_load import load as tload
    a, b = str(P.IDX[0].date()), str(P.IDX[-1].date())
    idx = pd.Index([str(d.date()) for d in P.IDX]); cols = pd.Index([str(c) for c in P.COLS])
    T, N = len(idx), len(cols)
    out = {k: np.zeros((T, N), np.float32) for k in ('F', 'T', 'D', 'A')}
    out['VOL'] = np.full((T, N), np.nan, np.float32); out['SH'] = np.full((T, N), np.nan, np.float32)
    have = np.zeros(T, bool); chk = dict(rows=0, rows_in_panel=0, days_inst=0, total_ne_sum=0, dup=0)
    for y in range(int(a[:4]), int(b[:4]) + 1):
        df = tload('inst_twse', max(a, f'{y}-01-01'), min(b, f'{y}-12-31'), columns=['date', 'code', 'foreign', 'trust', 'dealer', 'total'])
        df['date'] = df['date'].astype(str)
        chk['rows'] += len(df); chk['dup'] += int(df.duplicated(['date', 'code']).sum())
        chk['total_ne_sum'] += int((df['foreign'].astype('int64') + df['trust'] + df['dealer'] != df['total']).sum())
        ti = idx.get_indexer(df['date']); ci = cols.get_indexer(df['code'])
        have[np.unique(ti[ti >= 0])] = True
        ok = (ti >= 0) & (ci >= 0); chk['rows_in_panel'] += int(ok.sum())
        for k, c in (('F', 'foreign'), ('T', 'trust'), ('D', 'dealer'), ('A', 'total')):
            out[k][ti[ok], ci[ok]] = df[c].to_numpy()[ok]
        del df
        dk = tload('daily', max(a, f'{y}-01-01'), min(b, f'{y}-12-31'), columns=['date', 'code', 'market', 'volume', 'shares'])
        dk = dk[dk['market'] == 'twse']; dk['date'] = dk['date'].astype(str)
        ti = idx.get_indexer(dk['date']); ci = cols.get_indexer(dk['code']); ok = (ti >= 0) & (ci >= 0)
        out['VOL'][ti[ok], ci[ok]] = dk['volume'].to_numpy()[ok]; out['SH'][ti[ok], ci[ok]] = dk['shares'].to_numpy()[ok]
        del dk; gc.collect()
    chk['days_inst'] = int(have.sum()); chk['panel_days'] = int(T)
    chk['panel_days_without_inst'] = [idx[i] for i in np.flatnonzero(~have)]
    out['HAVE'] = have
    return out, chk


def load_flow(P):
    if FLOW_NPZ.exists():
        z = np.load(FLOW_NPZ)
        return {k: z[k] for k in z.files}
    out, _ = build_flow(P)
    np.savez_compressed(FLOW_NPZ, **out)
    return out


class Flow:
    def __init__(self, P, U, FL=None):
        FL = FL or load_flow(P)
        self.P, self.U = P, U
        self.have = FL['HAVE']
        self.cs = {k: np.vstack([np.zeros((1, P.N)), np.cumsum(FL[k].astype(np.float64), axis=0)]) for k in ('F', 'T', 'A')}
        self.csv = np.vstack([np.zeros((1, P.N)), np.cumsum(np.nan_to_num(FL['VOL'].astype(np.float64)), axis=0)])
        self.SH = FL['SH']
        self.pos = {k: FL[k] > 0 for k in ('F', 'T', 'A')}
        self._c = {}

    def _sum(self, cs, k, n, lag=0):
        b = k + 1 - lag; a = b - n
        return cs[b] - cs[a]

    def _streak(self, key, k):
        m = self.pos[key]; out = np.zeros(m.shape[1], int); alive = np.ones(m.shape[1], bool)
        for t in range(k, max(k - 60, -1), -1):
            alive &= m[t]
            if not alive.any():
                break
            out += alive
        return out

    def _pct(self, x, k):
        u = self.U[k] & np.isfinite(x)
        out = np.full(x.shape, np.nan)
        if u.sum() >= 30:
            xs = np.sort(x[u])
            out_all = (np.searchsorted(xs, x, side='left') + np.searchsorted(xs, x, side='right')) / 2 / len(xs)
            out = np.where(np.isfinite(x), out_all, np.nan)
        return out

    def feat(self, k):
        if k in self._c:
            return self._c[k]
        assert k >= 25 and self.have[k - 24:k + 1].all(), ('法人資料不足', k)
        f = {}
        sh = self.SH[k].astype(np.float64); sh = np.where(sh > 0, sh, np.nan)
        for n in (5, 10, 20):
            v = self._sum(self.csv, k, n); v = np.where(v > 0, v, np.nan)
            for key, nm in (('F', 'f'), ('T', 't'), ('A', 'a')):
                x = self._sum(self.cs[key], k, n)
                f[f'{nm}v{n}'] = x / v; f[f'{nm}s{n}'] = x / sh
        t5 = self._sum(self.cs['T'], k, 5); tp = self._sum(self.cs['T'], k, 20, lag=5)
        f['TFB'] = (t5 > 0) & (tp <= 0)
        for key, nm in (('A', 'a'), ('F', 'f'), ('T', 't')):
            f[f'streak_{nm}'] = self._streak(key, k)
        for nm in ('fv20', 'fv5', 'av20', 'av5'):
            f['p_' + nm] = self._pct(f[nm], k)
        self._c[k] = f
        return f


def order_ma(P, U, RT, x):
    """重建訊號列的 L_MA 排序（同 s10lib.make_signal_s10）。回傳 (o_ma, s)。"""
    m = pd.Period(x['month'], 'M'); k = x['k']
    s = RT['S'].loc[m].to_numpy()
    s = cslib.ins15_mask_row(P, m, k, s)
    ok = U[k] & np.isfinite(s) & P.F_MA60[k]
    return np.flatnonzero(ok)[np.argsort(-s[ok], kind='stable')], s


def _b(v):
    with np.errstate(invalid='ignore'):
        return np.asarray(v, bool) if np.asarray(v).dtype == bool else np.asarray(np.nan_to_num(v, nan=0.0), bool)


def apply_cell(P, U, RT, FW, S, cell, rng=None):
    """cell＝'CUR' 或 CELLS 之一。回傳 (S', 記錄 list[dict(month,k,orig,removed,added)])。
    rng 不為 None＝隨機對照：過濾／確認→每月隨機剔除同檔數（F 模式照樣遞補、不套法人條件）；重排→第二個名次改成隨機；放寬→從同一批「動能沒過」的候選隨機加同檔數。"""
    rec = []
    if cell == 'CUR':
        return S, rec
    rule = cell.split('|')[0]; mode = cell.split('|')[1] if '|' in cell else None; kd = kind(cell)
    S2 = dict(S); sig = S['sig'].copy(); rank = S['rank'].copy()
    for x in S['info']:
        k = x['k']
        tgt = np.flatnonzero(S['sig'][k])
        o, s = order_ma(P, U, RT, x)
        pool = o[:10]; r20 = P.RET20[k]
        chk = pool[r20[pool] > 0.10][:10]
        assert set(int(j) for j in chk) == set(int(j) for j in tgt), ('名單重建不符', x['month'])
        F = FW.feat(k)
        rm, added = [], []
        if kd in ('filter', 'confirm'):
            with np.errstate(invalid='ignore'):
                bad = np.asarray(BAD[rule](F), bool) if kd == 'filter' else ~np.asarray(GOOD[rule](F), bool)
            rm = [int(j) for j in tgt if bad[j]]
            if rng is not None:
                rm = [int(j) for j in rng.choice(tgt, size=len(rm), replace=False)] if rm else []
            if rm:
                sig[k, rm] = False; rank[k, rm] = np.nan
                if mode == 'F':
                    for j in o[10:FILL_DEPTH]:
                        if len(added) >= len(rm):
                            break
                        if r20[j] > 0.10 and (rng is not None or not bad[j]):
                            added.append(int(j)); sig[k, j] = True; rank[k, j] = s[j]
        elif kd == 'rerank':
            p20 = o[:FILL_DEPTH]
            if len(p20):
                v = F[RERANK[rule]][p20].astype(float)
                if rng is not None:
                    v = rng.random(len(p20))
                r2 = np.empty(len(p20)); r2[np.argsort(-np.nan_to_num(v, nan=-np.inf), kind='stable')] = np.arange(len(p20))
                comb = np.arange(len(p20)) + r2
                od = np.lexsort((np.arange(len(p20)), comb))
                top = p20[od[:10]]
                new = [int(j) for j in top if r20[j] > 0.10]
                sig[k, :] = False; rank[k, :] = np.nan
                for i, j in enumerate(new):
                    sig[k, j] = True; rank[k, j] = 1000.0 - i
                rm = [int(j) for j in tgt if j not in new]; added = [j for j in new if j not in set(int(q) for q in tgt)]
        else:
            with np.errstate(invalid='ignore'):
                good = np.asarray(WIDEN[rule](F), bool)
            cand = [int(j) for j in pool if np.isfinite(r20[j]) and not (r20[j] > 0.10)]
            pick = [j for j in cand if good[j]]
            cand2 = [int(j) for j in o[10:FILL_DEPTH] if r20[j] > 0.10] if rule == 'WD_TFB20' else []
            pick2 = [j for j in cand2 if good[j]]
            if rng is not None:
                pick = [int(j) for j in rng.choice(cand, size=len(pick), replace=False)] if pick else []
                pick2 = [int(j) for j in rng.choice(cand2, size=len(pick2), replace=False)] if pick2 else []
            for j in pick + pick2:
                added.append(j); sig[k, j] = True; rank[k, j] = s[j]
        rec.append(dict(month=x['month'], k=int(k), orig=[str(P.COLS[j]) for j in tgt], removed=[str(P.COLS[j]) for j in rm], added=[str(P.COLS[j]) for j in added]))
    S2['sig'] = sig; S2['rank'] = rank
    return S2, rec


FEATS = ('fv5', 'fv10', 'fv20', 'tv5', 'tv10', 'tv20', 'av5', 'av10', 'av20', 'fs5', 'fs10', 'fs20', 'ts5', 'ts10', 'ts20', 'as5', 'as10', 'as20',
         'TFB', 'streak_a', 'streak_f', 'streak_t', 'p_fv20', 'p_fv5', 'p_av20', 'p_av5')


def trade_features(P, FW, lg):
    """交易明細每筆補上進場訊號日（sig_t）的法人特徵。"""
    rows = []
    for k, j in zip(lg['sig_t'].to_numpy(), lg['j'].to_numpy()):
        F = FW.feat(int(k))
        rows.append({n: (bool(F[n][j]) if F[n].dtype == bool else float(F[n][j])) for n in FEATS})
    return pd.concat([lg.reset_index(drop=True), pd.DataFrame(rows)], axis=1)
