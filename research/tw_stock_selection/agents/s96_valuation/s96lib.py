#!/usr/bin/env python3
"""S96 共用程式庫：官方本益比／股價淨值比／殖利率和營收驚奇（SURGE）交易賺賠的關係。
引擎：直接 import agents/s90_institutional_flow/s90lib.py 的包裝（正式管線 E0｜W10_70、五個開關全開；模擬＝s10lib.simulate_s10 原樣），本題只改訊號列買進名單，出場不動。
估值資料（只讀）：raw/tsd_extra/per_twse_early（2012～2014）＋per_twse（2015 起）；收盤價欄不用；per／pbr／yield_pct 讀出後四捨五入到小數 2 位。
時點：訊號列 k（訊號日收盤後）只用日期 ≤ 訊號日的估值（官方每日收盤後公布）；次一交易日開盤成交（引擎原規則）。
特徵（訊號日、每檔）：
  per＝官方本益比；空值或 ≤ 0 → 視為「虧損或無」（per_ok＝False）
  pbr＝官方股價淨值比；p_pbr＝在同日股池（U[k]、有值）內的百分位（0～1）
  yld＝官方殖利率 %；p_yld＝同日股池百分位
  per3y＝今天的本益比在自己過去 756 個估值資料日（含今天、只算有本益比的日子、至少 250 筆）裡的百分位；今天虧損或歷史不足 → 缺值
暫存：scratchpad/s96/cache（用完刪）。一律 python3 -I -B。"""
import sys, gc
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
AG = HERE.parent
RAW = AG.parent / 'raw' / 'tsd_extra'
TMP = Path('/tmp/claude-0/-home-user/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/s96/cache')
TMP.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(AG / 's90_institutional_flow'))
import s90lib as E                                          # noqa: E402
from s90lib import s19lib, s10lib, s9lib, s8lib, s7lib, s6lib, s5lib, s4lib, s3lib, s2lib, stlib, cslib, COSTS   # noqa: E402
for mod, attr in ((s19lib, 'S19'), (s10lib, 'S10'), (s9lib, 'S9'), (s8lib, 'S8'), (s7lib, 'S7'), (s6lib, 'S6'), (s5lib, 'S5'),
                  (s4lib, 'S4'), (s3lib, 'S3'), (s2lib, 'S2'), (stlib, 'S1'), (cslib, 'TMP')):
    setattr(mod, attr, TMP)                                 # 快取改放 s96，不碰 s90 的暫存
E.TMP = TMP
CAPS, SHIFTS, SEGMENTS, SEGV, TARGET, VALID_FIRST_MONTH, FILL_DEPTH = E.CAPS, E.SHIFTS, E.SEGMENTS, E.SEGV, E.TARGET, E.VALID_FIRST_MONTH, E.FILL_DEPTH
load, signal, fresh_from, sim, seg_dd, order_ma = E.load, E.signal, E.fresh_from, E.sim, E.seg_dd, E.order_ma
VAL_NPZ = TMP / 'val_twse.npz'
WIN3Y, MIN3Y = 756, 250


def build_val(P):
    """T2×N 矩陣（自己的日期軸：2012-01-02 起所有估值資料日；欄＝P.COLS）：PER、PBR、YLD（float32、已四捨五入 2 位）、ROW（當日有列）。"""
    sys.path.insert(0, str(RAW))
    from tsd_extra_load import load as tload
    parts = []; chk = dict(rows=0, dup=0, per_null=0, per_le0=0, pbr_null=0, close_null=0)
    for name in ('per_twse_early', 'per_twse'):
        df = tload(name, None, str(P.IDX[-1].date()), columns=['date', 'code', 'close', 'per', 'pbr', 'yield_pct'])
        df['date'] = df['date'].astype(str)
        chk['rows'] += len(df); chk['dup'] += int(df.duplicated(['date', 'code']).sum())
        chk['per_null'] += int(df['per'].isna().sum()); chk['per_le0'] += int((df['per'] <= 0).sum()); chk['pbr_null'] += int(df['pbr'].isna().sum()); chk['close_null'] += int(df['close'].isna().sum())
        parts.append(df.drop(columns=['close']))
    df = pd.concat(parts, ignore_index=True); del parts
    dates = pd.Index(sorted(df['date'].unique())); cols = pd.Index([str(c) for c in P.COLS])
    ti = dates.get_indexer(df['date']); ci = cols.get_indexer(df['code']); ok = ci >= 0
    T2, N = len(dates), len(cols)
    out = {k: np.full((T2, N), np.nan, np.float32) for k in ('PER', 'PBR', 'YLD')}
    for k, c in (('PER', 'per'), ('PBR', 'pbr'), ('YLD', 'yield_pct')):
        out[k][ti[ok], ci[ok]] = np.round(df[c].to_numpy(np.float64)[ok], 2)
    row = np.zeros((T2, N), bool); row[ti[ok], ci[ok]] = True
    out['ROW'] = row; out['DATES'] = np.array(dates)
    chk.update(days=int(T2), first=dates[0], last=dates[-1], rows_in_panel_cols=int(ok.sum()),
               panel_days_without_val=[str(d.date()) for d in P.IDX if str(d.date()) not in set(dates)])
    del df; gc.collect()
    return out, chk


def load_val(P):
    if VAL_NPZ.exists():
        z = np.load(VAL_NPZ, allow_pickle=False)
        return {k: z[k] for k in z.files}
    out, _ = build_val(P)
    np.savez_compressed(VAL_NPZ, **out)
    return out


class Val:
    def __init__(self, P, U, VL=None):
        VL = VL or load_val(P)
        self.P, self.U = P, U
        self.PER, self.PBR, self.YLD, self.ROW = VL['PER'], VL['PBR'], VL['YLD'], VL['ROW']
        d = pd.Index([str(x) for x in VL['DATES']])
        self.k2 = d.get_indexer([str(x.date()) for x in P.IDX])      # 面板列 → 估值列（−1＝當日沒有估值檔）
        self._c = {}

    def _pct(self, x, k):
        u = self.U[k] & np.isfinite(x)
        out = np.full(x.shape, np.nan)
        if u.sum() >= 30:
            xs = np.sort(x[u])
            out = np.where(np.isfinite(x), (np.searchsorted(xs, x, side='left') + np.searchsorted(xs, x, side='right')) / 2 / len(xs), np.nan)
        return out

    def per3y(self, k, js):
        t = self.k2[k]; out = np.full(len(js), np.nan)
        for i, j in enumerate(js):
            h = np.round(self.PER[max(0, t - WIN3Y + 1):t + 1, j].astype(np.float64), 2); cur = h[-1]
            v = h[np.isfinite(h) & (h > 0)]
            if np.isfinite(cur) and cur > 0 and len(v) >= MIN3Y:
                out[i] = ((v < cur).sum() + (v <= cur).sum()) / 2 / len(v)
        return out

    def feat(self, k):
        if k in self._c:
            return self._c[k]
        t = self.k2[k]; assert t >= 0, ('訊號日沒有估值檔', k)
        per = np.round(self.PER[t].astype(np.float64), 2); pbr = np.round(self.PBR[t].astype(np.float64), 2); yld = np.round(self.YLD[t].astype(np.float64), 2)
        per_ok = np.isfinite(per) & (per > 0)
        f = dict(row=self.ROW[t], per=np.where(per_ok, per, np.nan), per_ok=per_ok, pbr=pbr, yld=yld, p_pbr=self._pct(pbr, k), p_yld=self._pct(yld, k),
                 p_per=self._pct(np.where(per_ok, per, np.nan), k))
        o = np.full(self.P.N, np.nan)
        f['per3y'] = o; f['_p3done'] = np.zeros(self.P.N, bool)
        self._c[k] = f
        return f

    def need3y(self, k, js):
        f = self.feat(k); js = [int(j) for j in js if not f['_p3done'][j]]
        if js:
            f['per3y'][js] = self.per3y(k, js); f['_p3done'][js] = True
        return f


def apply_cell(P, U, RT, VW, S, cell, CELLDEF, rng=None):
    """cell＝'CUR' 或 '規則|Z'／'規則|F'。CELLDEF[規則]＝函式(f)→布林陣列（True＝不買）。Z＝被擋名額不遞補（錢照引擎放 0050）；
    F＝由季線上名單第 11～20 名中近 20 日漲幅 > 10% 且通過同一條件者依 SURGE 遞補。rng 不為 None＝隨機對照：每月隨機剔除同檔數（F 照樣遞補、不套估值條件）。"""
    rec = []
    if cell == 'CUR':
        return S, rec
    rule, mode = cell.split('|')
    S2 = dict(S); sig = S['sig'].copy(); rank = S['rank'].copy()
    for x in S['info']:
        k = x['k']
        tgt = np.flatnonzero(S['sig'][k])
        o, s = order_ma(P, U, RT, x)
        pool = o[:10]; r20 = P.RET20[k]
        chk = pool[r20[pool] > 0.10][:10]
        assert set(int(j) for j in chk) == set(int(j) for j in tgt), ('名單重建不符', x['month'])
        F = VW.need3y(k, o[:FILL_DEPTH])
        with np.errstate(invalid='ignore'):
            bad = np.asarray(CELLDEF[rule](F), bool)
        rm = [int(j) for j in tgt if bad[j]]; added = []
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
        rec.append(dict(month=x['month'], k=int(k), orig=[str(P.COLS[j]) for j in tgt], removed=[str(P.COLS[j]) for j in rm], added=[str(P.COLS[j]) for j in added]))
    S2['sig'] = sig; S2['rank'] = rank
    return S2, rec


FEATS = ('row', 'per', 'per_ok', 'pbr', 'yld', 'p_pbr', 'p_yld', 'p_per', 'per3y')


def trade_features(P, VW, lg):
    rows = []
    for k, j in zip(lg['sig_t'].to_numpy(), lg['j'].to_numpy()):
        F = VW.need3y(int(k), [int(j)])
        rows.append({n: (bool(F[n][j]) if F[n].dtype == bool else float(F[n][j])) for n in FEATS})
    return pd.concat([lg.reset_index(drop=True), pd.DataFrame(rows)], axis=1)
