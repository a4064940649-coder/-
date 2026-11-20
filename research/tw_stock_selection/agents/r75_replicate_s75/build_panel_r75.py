#!/usr/bin/env python3
"""R75 第 2 步：自己把 2007-01-02～2026-10 的面板接起來（cslib.Panel 讀的 panel_cs.npz 格式）。
接縫（自訂，參數 --seam，預設 2015-01-05）：接縫日以前＝R55 主做法面板（Asoul 長表＋data_panel 補格，逐行沿用 build_panel_r55.py 主做法），
接縫日（含）以後＝官方面板原值。預設 2015-01-05＝官方日 K 第一天（官方資料有的日子一律用官方；重疊的 2015 上半年只用官方）。
敏感度：--seam 2015-07-01（重疊期改用 R55 面板那一邊）。
欄位：--cols all（pre 全部 ∪ 官方全部；主做法，不丟任何欄）／--cols twse（pre 全部 ∪ 官方曾經上市者；對照 S75 的做法）。
輸出 scratchpad/r75/<out>/panel_cs.npz、e50_ohlc.csv（官方 0050，2015-01-05 起）、innov.npz（接縫前全 False）。"""
import sys, json, time, resource, argparse
from pathlib import Path
import numpy as np, pandas as pd

ap = argparse.ArgumentParser(); ap.add_argument('--seam', default='2015-01-05'); ap.add_argument('--cols', default='all'); ap.add_argument('--out', default='cache_full')
a = ap.parse_args()
RS = Path('/home/user/-/research/tw_stock_selection')
TMP = Path('/tmp/claude-0/-home-user/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/r75')
OUT = TMP / a.out; OUT.mkdir(parents=True, exist_ok=True)
SEAM = pd.Timestamp(a.seam)
END = pd.Timestamp('2026-10-22')
t0 = time.time()

# ---- pre 面板（R55 主做法）
d = pd.read_csv(RS / 'raw/pre2015/asoul_twse_2007_2015H1.csv.gz', dtype={'code': str})
d['date'] = pd.to_datetime(d['date'])
traded = d['close'] > 0
d.loc[traded & d['change'].isna(), 'change'] = 0.0
PIDX = pd.DatetimeIndex(sorted(d['date'].unique()))
dp = RS / 'data_panel/panel'
rd = lambda f: pd.read_parquet(dp / f).loc[PIDX[0]:PIDX[-1]]
dc = rd('close_raw.parquet')
assert dc.index.equals(PIDX)
pcodes = sorted(set(d['code']) | set(dc.columns[dc.notna().any()]))
ci = {c: j for j, c in enumerate(pcodes)}; di = {x: i for i, x in enumerate(PIDX)}
T1, N1 = len(PIDX), len(pcodes)
K = ('O', 'H', 'L', 'C', 'VOL', 'AMT', 'CHG')
A = {k: np.full((T1, N1), np.nan) for k in K}
ti = d['date'].map(di).to_numpy(); tj = d['code'].map(ci).to_numpy()
for k, c in (('O', 'open'), ('H', 'high'), ('L', 'low'), ('C', 'close'), ('VOL', 'volume'), ('AMT', 'amount'), ('CHG', 'change')):
    A[k][ti, tj] = d[c].to_numpy(float)
obs = np.zeros((T1, N1), bool); obs[ti, tj] = True
dc = dc.reindex(columns=pcodes); do = rd('open.parquet').reindex(columns=pcodes); dh = rd('high.parquet').reindex(columns=pcodes)
dv = rd('value.parquet').reindex(columns=pcodes); dr = rd('ret.parquet').reindex(columns=pcodes)
fill = dc.notna().to_numpy() & ~obs
C2 = dc.to_numpy(); r2 = dr.to_numpy()
A['C'][fill] = C2[fill]; A['O'][fill] = do.to_numpy()[fill]; A['H'][fill] = dh.to_numpy()[fill]; A['AMT'][fill] = dv.to_numpy()[fill]
with np.errstate(invalid='ignore', divide='ignore'):
    chg = C2 - C2 / (1 + r2)
A['CHG'][fill] = np.where(np.isfinite(chg), np.round(chg, 4), np.nan)[fill]
obs |= fill
PMK = obs.astype(np.int8)
info = dict(seam=a.seam, cols=a.cols, pre=dict(T=T1, N=N1, n_long_rows=int(len(d)), n_x_rows=int(d['xflag'].sum()), n_dp_fill=int(fill.sum()),
                                               range=[str(PIDX[0].date()), str(PIDX[-1].date())]))
del d, dc, do, dh, dv, dr, C2, r2, chg

# ---- 官方面板
z = np.load(TMP / 'cache_is/panel_cs.npz', allow_pickle=False)
OIDX = pd.DatetimeIndex(z['IDX']); ocols = np.array(z['COLS']); OMK = z['MK']
ever_tw = (OMK == 1).any(0)
info['official'] = dict(T=len(OIDX), N=len(ocols), range=[str(OIDX[0].date()), str(OIDX[-1].date())], n_ever_twse=int(ever_tw.sum()))

# ---- 重疊期（官方第一天～pre 最後一天）逐格比對
orow = np.flatnonzero(OIDX <= PIDX[-1]); prow = np.array([di[x] for x in OIDX[orow]])
assert len(orow) == len(prow)
oc = {c: j for j, c in enumerate(ocols)}
common = [c for c in ocols if c in ci]
jo = np.array([oc[c] for c in common]); jm = np.array([ci[c] for c in common])
tw = OMK[orow][:, jo] == 1
cmp = dict(rows=int(len(orow)), first=str(OIDX[orow[0]].date()), last=str(OIDX[orow[-1]].date()))
for k in ('C', 'O', 'AMT', 'CHG'):
    x = z[k][orow][:, jo]; y = A[k][prow][:, jm]
    both = tw & np.isfinite(x) & np.isfinite(y)
    cmp[k] = dict(n_both=int(both.sum()), match=float((np.abs(x - y) < 1e-6)[both].mean()), max_abs=float(np.abs(x - y)[both].max()),
                  n_official_only=int((tw & np.isfinite(x) & ~np.isfinite(y)).sum()), n_pre_only=int((tw & ~np.isfinite(x) & np.isfinite(y)).sum()))
cmp['official_twse_codes_missing_in_pre'] = sorted(set(ocols[(OMK[orow] == 1).any(0)]) - set(pcodes))
cmp['pre_codes_in_overlap_not_official_twse'] = sorted(set(np.array(pcodes)[(PMK[prow] == 1).any(0)]) - set(ocols[(OMK[orow] == 1).any(0)]))
info['overlap'] = cmp

# ---- 接起來
pk = np.flatnonzero(PIDX < SEAM); ok = np.flatnonzero(OIDX >= SEAM)
assert len(pk) and len(ok) and (PIDX[pk[-1]] < OIDX[ok[0]])
keep_o = ocols if a.cols == 'all' else ocols[ever_tw]
codes = sorted(set(pcodes) | set(keep_o))
cj = {c: j for j, c in enumerate(codes)}
IDX = PIDX[pk].append(OIDX[ok]); T, N = len(IDX), len(codes)
jp = np.array([cj[c] for c in pcodes]); jo2 = np.array([cj[c] for c in keep_o]); osel = np.array([oc[c] for c in keep_o])
out = {}
for k in K:
    M = np.full((T, N), np.nan)
    M[np.ix_(np.arange(len(pk)), jp)] = A[k][pk]
    M[np.ix_(np.arange(len(pk), T), jo2)] = z[k][ok][:, osel]
    out[k] = M
MK = np.zeros((T, N), np.int8)
MK[np.ix_(np.arange(len(pk)), jp)] = PMK[pk]
MK[np.ix_(np.arange(len(pk), T), jo2)] = OMK[ok][:, osel]
np.savez_compressed(OUT / 'panel_cs.npz', IDX=IDX.values.astype('datetime64[ns]'), COLS=np.array(codes, dtype=str), MK=MK, **out)
e = pd.read_csv(TMP / 'cache_is/e50_ohlc.csv', parse_dates=['date'])
e.to_csv(OUT / 'e50_ohlc.csv', index=False)
# 創新板：官方快取 innov.npz（列＝官方面板到 END）→ 對到合併面板「載入後」的列（≤ END）
zi = np.load(TMP / 'cache_is/innov.npz', allow_pickle=False)
iidx = pd.DatetimeIndex(zi['IDX']); icols = np.array(zi['COLS']); IA = zi['A']
LIDX = IDX[IDX <= END]
IN = np.zeros((len(LIDX), N), bool)
ipos = {x: i for i, x in enumerate(LIDX)}
rr = np.array([ipos[x] for x in iidx if x >= SEAM]); rs = np.array([i for i, x in enumerate(iidx) if x >= SEAM])
ic = np.array([cj.get(c, -1) for c in icols])
dropped = int(IA[rs][:, ic < 0].sum())
IN[np.ix_(rr, ic[ic >= 0])] = IA[rs][:, ic >= 0]
np.savez_compressed(OUT / 'innov.npz', A=IN, IDX=LIDX.values.astype('datetime64[ns]'), COLS=np.array(codes, dtype=str), codes=zi['codes'], names=zi['names'])
info.update(T_file=T, T_loaded=int(len(LIDX)), N=N, range=[str(IDX[0].date()), str(LIDX[-1].date())], last_pre_row=str(PIDX[pk[-1]].date()), first_official_row=str(OIDX[ok[0]].date()),
            n_pre_rows=int(len(pk)), n_official_rows=int(len(ok)), n_codes_pre_only=len(set(pcodes) - set(keep_o)), n_codes_both=len(set(pcodes) & set(keep_o)),
            innov_true_kept=int(IN.sum()), innov_true_dropped=dropped, innov_true_in_overlap_before_seam=int(IA[[i for i, x in enumerate(iidx) if x < SEAM]].sum()) if SEAM > iidx[0] else 0,
            runtime_s=round(time.time() - t0, 1), maxrss_mb=round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024))
(TMP / f'panel_{a.out}.json').write_text(json.dumps(info, ensure_ascii=False, indent=1))
print(json.dumps(info, ensure_ascii=False, indent=1))
