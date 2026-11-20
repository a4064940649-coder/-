#!/usr/bin/env python3
"""S75 連續面板 2007-01-02～2026-10-22（cslib.Panel 讀的 panel_cs.npz 格式）＋ e50_ohlc.csv ＋ innov.npz。不算任何報酬、不跑引擎。
(1) 2015-01-05 以前（2007-01-02～2014-12-31）：R55 主做法面板——以下「pre 段」逐行沿用 agents/r55_replicate_s55/build_panel_r55.py 主做法（不帶參數）：
    raw/pre2015/asoul_twse_2007_2015H1.csv.gz（空白漲跌價差且有成交→0）＋ data_panel 補格（漲跌價差＝收−收÷(1＋ret)，使 2014 前除權息日報酬記 0）；MK＝有觀測的格＝1。
(2) 2015-01-05 起：官方面板（樣本內快取 eng/cache/panel_cs.npz，＝正式引擎逐位重現 48.76%／381 筆用的那一份），原值不動。
接縫：2015 上半年兩邊重疊，一律用官方面板那一邊；pre 面板的 2015-01-05～06-30 只拿來做重疊期逐格比對（收盤、開盤、成交金額、漲跌價差、MK）。
欄位：pre 面板全部代號（都是上市）∪ 官方面板中「曾經 MK＝1（上市）」的代號，依代號字串排序（與官方面板相同的排序規則）。
    從未上市的純上櫃代號不放（股池＝上市，這些欄永遠不在股池；只為了省記憶體，峰值上限 3GB）。驗收：2016-02 起每月入選名單要與正式引擎逐月相同。
e50_ohlc.csv：官方 0050 開高低收原檔（2015-01-05 起；之前空 → 引擎退路：開盤÷收盤＝1）。innov.npz：官方創新板旗標搬到新欄位，2015 前全 False。
另把樣本內快取複製到 scratchpad/s75/cache_is（正式引擎單獨跑用）。
python3 -I -B build_panel.py → scratchpad/s75/cache_cont/、panel_check.json"""
import json, time, resource, shutil
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
RS = Path('/home/user/-/research/tw_stock_selection')
TMP = Path('/tmp/claude-0/-home-user/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/s75')
SRC_IS = Path('/tmp/claude-0/-home-user--/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/eng/cache')
SEAM = pd.Timestamp('2015-01-05')
OUT = TMP / 'cache_cont'; OUT.mkdir(parents=True, exist_ok=True)
IS = TMP / 'cache_is'; IS.mkdir(parents=True, exist_ok=True)
t0 = time.time()
for f in ('panel_cs.npz', 'e50_ohlc.csv', 'innov.npz'):
    if not (IS / f).exists():
        shutil.copy2(SRC_IS / f, IS / f)

# ---------------------------------------------------------------- pre 段（R55 主做法，逐行沿用）
d = pd.read_csv(RS / 'raw/pre2015/asoul_twse_2007_2015H1.csv.gz', dtype={'code': str})
d['date'] = pd.to_datetime(d['date'])
traded = d['close'] > 0
d.loc[traded & d['change'].isna(), 'change'] = 0.0
IDX = pd.DatetimeIndex(sorted(d['date'].unique()))
dp = RS / 'data_panel/panel'
rd = lambda f: pd.read_parquet(dp / f).loc[IDX[0]:IDX[-1]]
dc = rd('close_raw.parquet')
assert dc.index.equals(IDX)
codes = sorted(set(d['code']) | set(dc.columns[dc.notna().any()]))
ci = {c: j for j, c in enumerate(codes)}; di = {x: i for i, x in enumerate(IDX)}
T, N = len(IDX), len(codes)
KEYS = ('O', 'H', 'L', 'C', 'VOL', 'AMT', 'CHG')
A = {k: np.full((T, N), np.nan) for k in KEYS}
ti = d['date'].map(di).to_numpy(); tj = d['code'].map(ci).to_numpy()
for k, c in (('O', 'open'), ('H', 'high'), ('L', 'low'), ('C', 'close'), ('VOL', 'volume'), ('AMT', 'amount'), ('CHG', 'change')):
    A[k][ti, tj] = d[c].to_numpy(float)
obs = np.zeros((T, N), bool); obs[ti, tj] = True
n_long = int(len(d)); n_x = int(d['xflag'].sum())
del d
dc = dc.reindex(columns=codes); do = rd('open.parquet').reindex(columns=codes); dh = rd('high.parquet').reindex(columns=codes)
dv = rd('value.parquet').reindex(columns=codes); dr = rd('ret.parquet').reindex(columns=codes)
fill = dc.notna().to_numpy() & ~obs
n_fill = int(fill.sum())
C2 = dc.to_numpy(); r2 = dr.to_numpy()
A['C'][fill] = C2[fill]; A['O'][fill] = do.to_numpy()[fill]; A['H'][fill] = dh.to_numpy()[fill]; A['AMT'][fill] = dv.to_numpy()[fill]
with np.errstate(invalid='ignore', divide='ignore'):
    chg = C2 - C2 / (1 + r2)
A['CHG'][fill] = np.where(np.isfinite(chg), np.round(chg, 4), np.nan)[fill]
obs |= fill
MKpre = obs.astype(np.int8)
del dc, do, dh, dv, dr, C2, r2, chg, fill

# ---------------------------------------------------------------- 官方面板
z = np.load(IS / 'panel_cs.npz', allow_pickle=False)
oidx = pd.DatetimeIndex(z['IDX']); ocols = np.array(z['COLS']); OMK = z['MK']
assert oidx[0] == SEAM and str(oidx[-1].date()) >= '2026-10-22'
npre = int((IDX < SEAM).sum())
assert IDX[npre] == SEAM and str(IDX[npre - 1].date()) == '2014-12-31' and str(IDX[-1].date()) == '2015-06-30'
ov_o = np.flatnonzero(oidx <= IDX[-1]); ov_p = np.arange(npre, T)
assert len(ov_o) == len(ov_p) and (oidx[ov_o] == IDX[ov_p]).all(), '重疊期交易日不一致'

# ---- 重疊期逐格比對（官方上市格）
oc = {c: j for j, c in enumerate(ocols)}
common = [c for c in ocols if c in ci]
jo = np.array([oc[c] for c in common]); jm = np.array([ci[c] for c in common])
tw_o = OMK[ov_o] == 1
cmp = {}
for k in ('C', 'O', 'AMT', 'CHG'):
    a = z[k][ov_o][:, jo]; b = A[k][ov_p][:, jm]; m = tw_o[:, jo]
    both = m & np.isfinite(a) & np.isfinite(b)
    dif = np.abs(a - b)
    cmp[k] = dict(n_both=int(both.sum()), match=float((dif < 1e-6)[both].mean()), max_abs_diff=float(dif[both].max()),
                  n_official_only=int((m & np.isfinite(a) & ~np.isfinite(b)).sum()), n_pre_only=int((m & ~np.isfinite(a) & np.isfinite(b)).sum()))
mine_mk = MKpre[ov_p][:, jm] == 1
cmp['MK'] = dict(official_twse_cells=int(tw_o.sum()), pre_cells=int((MKpre[ov_p] == 1).sum()), agree_on_common=float((mine_mk == tw_o[:, jo]).mean()),
                 official_twse_codes_missing_in_pre=sorted(set(ocols[tw_o.any(0)]) - set(codes)),
                 official_traded_twse_not_in_pre=int((tw_o[:, jo] & np.isfinite(z['C'][ov_o][:, jo]) & ~mine_mk).sum()))

# ---------------------------------------------------------------- 接起來
ever_tw = (OMK == 1).any(0)
ALL = np.array(sorted(set(codes) | set(ocols[ever_tw])), dtype=str)
ai = {c: j for j, c in enumerate(ALL)}
TT = npre + len(oidx); NN = len(ALL)
FIDX = IDX[:npre].append(oidx)
assert FIDX.is_monotonic_increasing and FIDX.is_unique
jp = np.array([ai[c] for c in codes])                       # pre 欄 → 新欄
osel = np.flatnonzero(np.array([c in ai for c in ocols])); jq = np.array([ai[ocols[j]] for j in osel])
B = {}
for k in KEYS:
    X = np.full((TT, NN), np.nan)
    X[:npre, jp] = A[k][:npre]
    X[npre:, jq] = z[k][:, osel]
    B[k] = X
    del A[k]
MK = np.zeros((TT, NN), np.int8)
MK[:npre, jp] = MKpre[:npre]; MK[npre:, jq] = OMK[:, osel]
np.savez_compressed(OUT / 'panel_cs.npz', IDX=FIDX.values.astype('datetime64[ns]'), COLS=ALL, MK=MK, **B)
shutil.copy2(IS / 'e50_ohlc.csv', OUT / 'e50_ohlc.csv')
zi = np.load(IS / 'innov.npz', allow_pickle=False)
END = pd.Timestamp('2026-10-22')                             # cslib.END：Panel 只留到這一天；官方面板檔多出的列（到 2026-10-30）照樣存、載入時被截掉
nis = int((oidx <= END).sum())                              # innov.npz 的列數要等於 Panel 的 T（截到 END）
assert zi['A'].shape == (nis, len(ocols)) and np.array_equal(zi['COLS'], ocols.astype(str)) and (pd.DatetimeIndex(zi['IDX']) == oidx[:nis]).all()
IN = np.zeros((npre + nis, NN), bool); IN[npre:, jq] = zi['A'][:, osel]
dropped_innov = int(zi['A'][:, ~np.isin(np.arange(len(ocols)), osel)].sum())
np.savez_compressed(OUT / 'innov.npz', A=IN, IDX=FIDX[:npre + nis].values.astype('datetime64[ns]'), COLS=ALL, codes=zi['codes'], names=zi['names'])

e = pd.read_csv(IS / 'e50_ohlc.csv', parse_dates=['date'])
info = dict(T_file=int(TT), T_loaded=int(npre + nis), N=int(NN), range=[str(FIDX[0].date()), str(FIDX[npre + nis - 1].date())], seam=str(SEAM.date()), last_pre_row=str(FIDX[npre - 1].date()), n_pre_rows=int(npre),
            n_official_rows=int(len(oidx)), n_pre_codes=int(N), n_official_codes=int(len(ocols)), n_official_codes_ever_twse=int(ever_tw.sum()),
            n_official_codes_dropped_never_twse=int(len(ocols) - len(osel)), n_codes_pre_only=int(len(set(codes) - set(ocols))),
            n_codes_both=int(len(set(codes) & set(ocols[ever_tw]))), innov_true_cells_kept=int(IN.sum()), innov_true_cells_dropped=dropped_innov,
            pre=dict(n_long_rows=n_long, n_x_rows=n_x, n_dp_fill=n_fill, T=int(T), ref_R55=dict(T=2105, N=926, n_long_rows=1531951, n_x_rows=9214, n_dp_fill=72224)),
            overlap_2015H1=dict(rows=int(len(ov_o)), side_used='官方面板', **cmp),
            e50_ohlc=dict(first=str(e['date'].min().date()), last=str(e['date'].max().date()), rows=int(len(e))),
            runtime_s=round(time.time() - t0, 1), maxrss_mb=round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024))
(HERE / 'panel_check.json').write_text(json.dumps(info, ensure_ascii=False, indent=1))
print(json.dumps(info, ensure_ascii=False, indent=1))
