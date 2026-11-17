#!/usr/bin/env python3
"""R55 第 2 步：自建 2007-01-02～2015-06-30 上市日 K 面板（cslib.Panel 讀的 panel_cs.npz 格式），獨立於 S55 的 build_pre_panel.py。
來源：
  (1) raw/pre2015/asoul_twse_2007_2015H1.csv.gz（S55 衍生長表；Asoul/tsec 證交所原始行情）：開高低收、成交股數、成交金額、漲跌價差；
      漲跌價差空白且有成交＝平盤 → 0；X（除權息）列 change＝0（長表已如此）；無成交列（收盤空白）→ 收盤 NaN、MK＝1。
  (2) data_panel/panel（研究中心上市面板；SPEC「上市期間依 data_panel」）：長表沒有、data_panel 有收盤的格（2010 起的 k66 來源，
      共 45 檔整段缺於 Asoul）→ 收＝close_raw、開＝open、高＝high、成交金額＝value、漲跌價差＝收 − 收÷(1＋ret)（ret 缺→NaN）；低、量＝NaN。
      變體 --no-dpfill：不補 (2)（敏感度）。
  MK：有觀測（長表列或 data_panel 補格）的格＝1，其餘 0（同官方日 K：停止交易日不在檔內；2015H1 驗證：[首,末] 之間的空檔 64 格官方全是 0）。
      變體 --gapfill：[第一筆, 最後一筆] 之間一律 MK＝1（敏感度）。
另寫：e50_ohlc.csv（官方 0050 開高低收，只有 2015-01-05 起；之前空 → 引擎退路）、innov.npz（全 False）。
並與官方面板 2015-01-05～2015-06-30 逐格比對（收、開、成交金額、漲跌價差、MK）。
python3 -I -B build_panel_r55.py [--no-dpfill]"""
import sys, json, time, resource
from pathlib import Path
import numpy as np, pandas as pd

RS = Path('/home/user/-/research/tw_stock_selection')
TMP = Path('/tmp/claude-0/-home-user/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/r55')
DPFILL = '--no-dpfill' not in sys.argv
BLANKNAN = '--blanknan' in sys.argv          # 敏感度：長表空白漲跌價差保留 NaN（不設 0）
FILLCHGNAN = '--fillchgnan' in sys.argv      # 敏感度：data_panel 補格的漲跌價差一律 NaN（引擎退路＝收盤÷前收−1，除權息日含除息跌價）
OUT = TMP / ('cache_pre' + ('' if DPFILL else '_nofill') + ('_gapfill' if '--gapfill' in sys.argv else '') + ('_blanknan' if BLANKNAN else '') + ('_fillchgnan' if FILLCHGNAN else ''))
OUT.mkdir(parents=True, exist_ok=True)
t0 = time.time()

d = pd.read_csv(RS / 'raw/pre2015/asoul_twse_2007_2015H1.csv.gz', dtype={'code': str})
d['date'] = pd.to_datetime(d['date'])
traded = d['close'] > 0
if not BLANKNAN:
    d.loc[traded & d['change'].isna(), 'change'] = 0.0
IDX = pd.DatetimeIndex(sorted(d['date'].unique()))
dp = RS / 'data_panel/panel'
rd = lambda f: pd.read_parquet(dp / f).loc[IDX[0]:IDX[-1]]
dc = rd('close_raw.parquet')
assert dc.index.equals(IDX)
codes = sorted(set(d['code']) | (set(dc.columns[dc.notna().any()]) if DPFILL else set()))
ci = {c: j for j, c in enumerate(codes)}; di = {x: i for i, x in enumerate(IDX)}
T, N = len(IDX), len(codes)
A = {k: np.full((T, N), np.nan) for k in ('O', 'H', 'L', 'C', 'VOL', 'AMT', 'CHG')}
ti = d['date'].map(di).to_numpy(); tj = d['code'].map(ci).to_numpy()
for k, c in (('O', 'open'), ('H', 'high'), ('L', 'low'), ('C', 'close'), ('VOL', 'volume'), ('AMT', 'amount'), ('CHG', 'change')):
    A[k][ti, tj] = d[c].to_numpy(float)
obs = np.zeros((T, N), bool); obs[ti, tj] = True
n_fill = 0
if DPFILL:
    dc = dc.reindex(columns=codes); do = rd('open.parquet').reindex(columns=codes); dh = rd('high.parquet').reindex(columns=codes)
    dv = rd('value.parquet').reindex(columns=codes); dr = rd('ret.parquet').reindex(columns=codes)
    fill = dc.notna().to_numpy() & ~obs
    n_fill = int(fill.sum())
    C2 = dc.to_numpy(); r2 = dr.to_numpy()
    A['C'][fill] = C2[fill]; A['O'][fill] = do.to_numpy()[fill]; A['H'][fill] = dh.to_numpy()[fill]; A['AMT'][fill] = dv.to_numpy()[fill]
    with np.errstate(invalid='ignore', divide='ignore'):
        chg = C2 - C2 / (1 + r2)
    if not FILLCHGNAN:
        A['CHG'][fill] = np.where(np.isfinite(chg), np.round(chg, 4), np.nan)[fill]
    obs |= fill
first = np.where(obs.any(0), obs.argmax(0), T); last = np.where(obs.any(0), T - 1 - obs[::-1].argmax(0), -1)
tt = np.arange(T)[:, None]
span = (tt >= first[None, :]) & (tt <= last[None, :])
n_gap = int((span & ~obs).sum())
GAPFILL = '--gapfill' in sys.argv
MK = (span if GAPFILL else obs).astype(np.int8)
np.savez_compressed(OUT / 'panel_cs.npz', IDX=IDX.values.astype('datetime64[ns]'), COLS=np.array(codes, dtype=str), MK=MK, **A)
e = pd.read_csv(TMP / 'cache_is/e50_ohlc.csv', parse_dates=['date'])
e[e['date'] <= IDX[-1]].to_csv(OUT / 'e50_ohlc.csv', index=False)
np.savez_compressed(OUT / 'innov.npz', A=np.zeros((T, N), bool), IDX=IDX.values.astype('datetime64[ns]'), COLS=np.array(codes, dtype=str),
                    codes=np.array([], dtype=str), names=np.array([], dtype=str))

# ---- 與官方面板 2015H1 比對
z = np.load(TMP / 'cache_is/panel_cs.npz', allow_pickle=False)
oidx = pd.DatetimeIndex(z['IDX']); ocols = np.array(z['COLS'])
rows = np.flatnonzero(oidx <= IDX[-1]); assert oidx[rows[0]] == pd.Timestamp('2015-01-05')
mi = np.array([di[x] for x in oidx[rows]])
OMK = z['MK'][rows]
cmp = {}
tw_o = OMK == 1
oc = {c: j for j, c in enumerate(ocols)}
common = [c for c in ocols if c in ci]
jo = np.array([oc[c] for c in common]); jm = np.array([ci[c] for c in common])
for k in ('C', 'O', 'AMT', 'CHG'):
    a = z[k][rows][:, jo]; b = A[k][mi][:, jm]; m = tw_o[:, jo]
    both = m & np.isfinite(a) & np.isfinite(b)
    cmp[k] = dict(n_both=int(both.sum()), match=float((np.abs(a - b) < 1e-6)[both].mean()), n_official_only=int((m & np.isfinite(a) & ~np.isfinite(b)).sum()),
                  n_mine_only=int((m & ~np.isfinite(a) & np.isfinite(b)).sum()))
    if k == 'CHG':
        bb = np.nan_to_num(b); aa = np.nan_to_num(a)
        cmp['CHG_nan_as_0'] = dict(n=int((m & np.isfinite(z['C'][rows][:, jo])).sum()), match=float((np.abs(aa - bb) < 1e-6)[m & np.isfinite(z['C'][rows][:, jo])].mean()))
mine_mk = np.zeros_like(tw_o[:, jo]); mine_mk = MK[mi][:, jm] == 1
cmp['MK'] = dict(official_twse_cells=int(tw_o.sum()), mine_cells=int((MK[mi] == 1).sum()), agree_on_common=float((mine_mk == tw_o[:, jo]).mean()),
                 official_twse_codes_missing_in_mine=sorted(set(ocols[tw_o.any(0)]) - set(codes)),
                 official_traded_twse_not_in_mine=int((tw_o[:, jo] & np.isfinite(z['C'][rows][:, jo]) & ~mine_mk).sum()))
info = dict(variant=OUT.name, T=T, N=N, n_long_rows=int(len(d)), n_x_rows=int(d['xflag'].sum()), n_dp_fill=n_fill, n_listed_gap_cells=n_gap,
            range=[str(IDX[0].date()), str(IDX[-1].date())], vs_official_2015H1=cmp, runtime_s=round(time.time() - t0, 1),
            maxrss_mb=round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024))
(TMP / f'panel_{info["variant"]}.json').write_text(json.dumps(info, ensure_ascii=False, indent=1))
print(json.dumps(info, ensure_ascii=False, indent=1))
