#!/usr/bin/env python3
"""S93 共用程式庫：集保股權分散（大戶／散戶／股東人數的變化）能不能幫營收驚奇（SURGE）。
引擎：重用 s90lib（正式管線 E0｜W10_70、五個開關全開；s10lib.simulate_s10 原樣呼叫）。進場格只改訊號列買進名單（sig／rank）；出場格只在訊號列把 X 多標幾檔（不新增非訊號日交易）。
集保資料（只讀）：raw/tsd_extra/tdcc_summary（每週每檔：holders 股東人數、pct_ge400 400 張以上占比 %、pct_lt10 10 張以下占比 %）。
時點：lag='cons'（主要）＝資料日 ≤ 訊號日 − 7 個日曆天的最新一筆；lag='loose'（另報）＝資料日 < 訊號日的最新一筆（至少隔一個交易日）。
「N 週前」＝資料日 ≤ 最新一筆資料日 − 7N 天的最新一筆（缺週就用再前一週）。
特徵（訊號列 k、每檔）：dB4／dB8＝pct_ge400 近 4／8 週變化（百分點）；dH4／dH8＝股東人數近 4／8 週變化率；dR4／dR8＝pct_lt10 近 4／8 週變化（百分點）；
      ndec＝到最新一筆為止 pct_ge400 連續逐筆下降的筆數（相鄰兩筆都有值且後 < 前）。缺值＝條件不成立（進場照買、出場不賣）。
規則只套用在營收月份 ≥ 2019-07（訊號日 2019-08 起）的訊號列；之前與現行逐日相同。
暫存：scratchpad/s93/cache。一律 python3 -I -B。"""
import sys, gc
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent
AG = HERE.parent
sys.path.insert(0, str(AG / 's90_institutional_flow'))
import s90lib as L90                                        # noqa: E402
from s90lib import s19lib, s10lib, s9lib, s8lib, s7lib, s6lib, s5lib, s4lib, s3lib, s2lib, stlib, cslib, COSTS, order_ma   # noqa: E402
TMP = Path('/tmp/claude-0/-home-user/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/s93/cache')
TMP.mkdir(parents=True, exist_ok=True)
for mod, attr in ((s19lib, 'S19'), (s10lib, 'S10'), (s9lib, 'S9'), (s8lib, 'S8'), (s7lib, 'S7'), (s6lib, 'S6'), (s5lib, 'S5'),
                  (s4lib, 'S4'), (s3lib, 'S3'), (s2lib, 'S2'), (stlib, 'S1'), (cslib, 'TMP')):
    setattr(mod, attr, TMP)
RAW = AG.parent / 'raw' / 'tsd_extra'
CAPS, SHIFTS, TARGET, FILL_DEPTH = L90.CAPS, L90.SHIFTS, L90.TARGET, L90.FILL_DEPTH
RULE_FIRST_MONTH = pd.Period('2019-07', 'M')               # 規則起用：營收月份 2019-07（訊號日 2019-08）
VALID_FIRST_MONTH = pd.Period('2022-12', 'M')              # 驗證：營收月份 ≥ 2022-12（訊號日 2023-01 起）
SEGMENTS = {'2024-07~2025-04': ('2024-07-01', '2025-04-30')}
load, signal, sim = L90.load, L90.signal, L90.sim


def fresh_from(S):
    return L90.fresh_from(S, VALID_FIRST_MONTH)


def seg_dd(nav):
    o = {}
    for k, (a, b) in SEGMENTS.items():
        s = nav.loc[a:b].dropna(); o[k] = float((s / s.cummax() - 1).min())
    return o


class TDCC:
    def __init__(self, P):
        sys.path.insert(0, str(RAW))
        from tsd_extra_load import load as tload
        d = tload('tdcc_summary', '2019-01-01', '2026-12-31', columns=['date', 'code', 'holders', 'pct_ge400', 'pct_lt10'])
        d['date'] = d['date'].astype(str)
        self.rows = len(d); self.dup = int(d.duplicated(['date', 'code']).sum())
        ds = np.array(sorted(d['date'].unique())); self.dates = pd.DatetimeIndex(ds)
        cols = pd.Index([str(c) for c in P.COLS])
        wi = pd.Index(ds).get_indexer(d['date']); ci = cols.get_indexer(d['code']); ok = ci >= 0
        W, N = len(ds), len(cols)
        self.B = np.full((W, N), np.nan); self.H = np.full((W, N), np.nan); self.R = np.full((W, N), np.nan)
        self.B[wi[ok], ci[ok]] = d['pct_ge400'].to_numpy(float)[ok]; self.H[wi[ok], ci[ok]] = d['holders'].to_numpy(float)[ok]; self.R[wi[ok], ci[ok]] = d['pct_lt10'].to_numpy(float)[ok]
        self.H[self.H <= 0] = np.nan
        dec = np.zeros((W, N), bool); dec[1:] = self.B[1:] < self.B[:-1]          # NaN 比較＝False
        self.ndec = np.zeros((W, N), int)
        for w in range(1, W):
            self.ndec[w] = np.where(dec[w], self.ndec[w - 1] + 1, 0)
        self.P = P; self._c = {}
        del d

    def w0(self, k, lag):
        sd = self.P.IDX[k]
        cut = sd - pd.Timedelta(days=7) if lag == 'cons' else sd - pd.Timedelta(days=1)
        return int(np.searchsorted(self.dates.values, np.datetime64(cut), side='right')) - 1

    def wback(self, w, n):
        return int(np.searchsorted(self.dates.values, np.datetime64(self.dates[w] - pd.Timedelta(days=7 * n)), side='right')) - 1

    def feat(self, k, lag='cons'):
        if (k, lag) in self._c:
            return self._c[(k, lag)]
        w = self.w0(k, lag); assert w >= 0, ('集保資料不足', k)
        f = dict(w0=w, age=int((self.P.IDX[k] - self.dates[w]).days))
        for n in (4, 8):
            wb = self.wback(w, n); f[f'wb{n}'] = wb; f[f'span{n}'] = int((self.dates[w] - self.dates[wb]).days)
            if wb < 0:                                         # 資料還不夠久 → 缺值
                f[f'dB{n}'] = f[f'dR{n}'] = f[f'dH{n}'] = np.full(self.B.shape[1], np.nan); f[f'span{n}'] = None; continue
            f[f'dB{n}'] = self.B[w] - self.B[wb]; f[f'dR{n}'] = self.R[w] - self.R[wb]; f[f'dH{n}'] = self.H[w] / self.H[wb] - 1
        f['ndec'] = self.ndec[w]; f['B'] = self.B[w]
        self._c[(k, lag)] = f
        return f


FEATS = ('dB4', 'dB8', 'dH4', 'dH8', 'dR4', 'dR8', 'ndec', 'B')


def trade_features(TD, lg, lag='cons'):
    rows = []
    for k, j in zip(lg['sig_t'].to_numpy(), lg['j'].to_numpy()):
        F = TD.feat(int(k), lag)
        rows.append(dict({n: float(F[n][j]) for n in FEATS}, age=F['age']))
    return pd.concat([lg.reset_index(drop=True), pd.DataFrame(rows)], axis=1)


def _t(v):
    with np.errstate(invalid='ignore'):
        return np.asarray(v, bool)


try:
    from s93cells import BAD, GOOD, EXIT, CELLS             # 格子在看完訓練期描述後才寫（s93cells.py），登記於 code.sha256
except ImportError:
    BAD, GOOD, EXIT, CELLS = {}, {}, {}, ()


def kind(cell):
    r = cell.split('|')[0]
    return 'filter' if r in BAD else 'confirm' if r in GOOD else 'exit'


def apply_cell(P, U, RT, TD, S, cell, lag='cons', rng=None):
    """cell＝'CUR' 或 CELLS 之一。回傳 (S', 記錄)。進場格：'規則|Z'（不遞補，錢放 0050）或 '規則|F'（L_MA 第 11～20 名中 RET20>10% 且通過同一條件者依 SURGE 遞補）。
    出場格：訊號列把「E0 本來會續抱（L_MA 前 20 名）且條件成立」的股票也標成出場（次一交易日開盤賣；當天不買回）。
    rng：隨機對照——進場格每月隨機剔除同檔數（F 照樣遞補、不套條件）；出場格每月從 L_MA 前 20 名隨機標同檔數。"""
    rec = []
    if cell == 'CUR':
        return S, rec
    rule = cell.split('|')[0]; mode = cell.split('|')[1] if '|' in cell else None; kd = kind(cell)
    S2 = dict(S); sig = S['sig'].copy(); rank = S['rank'].copy(); X = S['X'].copy()
    for x in S['info']:
        if pd.Period(x['month'], 'M') < RULE_FIRST_MONTH:
            continue
        k = x['k']; tgt = np.flatnonzero(S['sig'][k]); o, s = order_ma(P, U, RT, x); r20 = P.RET20[k]
        F = TD.feat(k, lag); rm, added, xs = [], [], []
        if kd in ('filter', 'confirm'):
            bad = _t(BAD[rule](F)) if kd == 'filter' else ~_t(GOOD[rule](F)) & np.isfinite(F['dB4']) & np.isfinite(F['dH4'])
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
        else:
            keep = o[:20]; c = _t(EXIT[rule](F))
            xs = [int(j) for j in keep if c[j]]
            if rng is not None:
                xs = [int(j) for j in rng.choice(keep, size=len(xs), replace=False)] if xs else []
            X[k, xs] = True
        rec.append(dict(month=x['month'], k=int(k), orig=[str(P.COLS[j]) for j in tgt], removed=[str(P.COLS[j]) for j in rm], added=[str(P.COLS[j]) for j in added],
                        xflag=[str(P.COLS[j]) for j in xs]))
    S2['sig'] = sig; S2['rank'] = rank; S2['X'] = X
    return S2, rec
