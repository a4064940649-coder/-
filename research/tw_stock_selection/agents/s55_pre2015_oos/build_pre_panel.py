#!/usr/bin/env python3
"""S55 建 2007-01～2015-06 上市日 K 面板（格式同 cslib.build_panel 的 panel_cs.npz），不算任何報酬。
來源：
  - 個股日 K：Asoul/tsec 證交所原始行情（本機 scratchpad tsec_all/{code}.csv，無表頭：日期(民國),成交股數,成交金額,開,高,低,收,漲跌價差,筆數），含已下市股。
  - 上市期間（MK＝1）：data_panel/panel/close_raw.parquet（build_twse_panel.py 已去掉轉上市前的櫃買期、只留上市普通股＋已下市上市股）
    每檔「data_panel 第一個有值日～最後一個有值日」之間的 Asoul 列（含無成交列）記為上市。
  - 漲跌價差 'X…'（除權息日）→ 0（同官方 2015 起日 K 的格式：除權息日漲跌價差＝0；EXRIGHT_FIX 用 TWT49U 修正，2014-01 起才有）。
  - data_panel 有值但 Asoul 沒有的格子：用 data_panel 的收／開／高／成交金額補，低＝NaN、量＝NaN、漲跌價差＝NaN（引擎改用收盤對前收）。
輸出：
  - raw/pre2015/asoul_twse_2007_2015H1.csv.gz（衍生檔：上市期間的 Asoul 列，長表）＋ raw/pre2015/asoul_source_manifest.json（用到的來源檔 sha256）
  - scratchpad/s55/cache_pre/panel_cs.npz、e50_ohlc.csv（0050 開高低收：2015-01-05 起取自官方快取，之前無資料＝NaN）、innov.npz（全 False；創新板 2021 才有）
  - agents/s55_pre2015_oos/panel_check.json（覆蓋、與 data_panel 及官方 2015 上半年日 K 的一致率）
用法：python3 -I -B build_pre_panel.py"""
import json, hashlib, gzip, io, resource, time
from pathlib import Path
import numpy as np, pandas as pd

t0 = time.time()
HERE = Path(__file__).resolve().parent
ROOT = Path('/home/user/-/research/tw_stock_selection')
TSEC = Path('/tmp/claude-0/-home-user--/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/tsec_all')
OFF = Path('/tmp/claude-0/-home-user--/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/eng/cache')   # 官方面板快取（sha 434f6422…，與 s5 快取相同）
OUT = Path('/tmp/claude-0/-home-user/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/s55/cache_pre')
RAWP = ROOT / 'raw' / 'pre2015'
D0, D1 = pd.Timestamp('2007-01-02'), pd.Timestamp('2015-06-30')
OUT.mkdir(parents=True, exist_ok=True)
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()

cr = pd.read_parquet(ROOT / 'data_panel/panel/close_raw.parquet')
op = pd.read_parquet(ROOT / 'data_panel/panel/open.parquet')
hi = pd.read_parquet(ROOT / 'data_panel/panel/high.parquet')
va = pd.read_parquet(ROOT / 'data_panel/panel/value.parquet')
first = cr.apply(lambda x: x.first_valid_index()); last = cr.apply(lambda x: x.last_valid_index())
win = cr.loc[D0:D1]
IDX = win.index
codes = [c for c in win.columns if win[c].notna().any()]
print('上市期間有值的代號', len(codes), '交易日', len(IDX), flush=True)


def num(s):
    return pd.to_numeric(s.astype(str).str.replace(',', '').str.strip(), errors='coerce')


rows, manifest, no_file = [], {}, []
for c in codes:
    f = TSEC / f'{c}.csv'
    if not f.exists():
        no_file.append(c); continue
    manifest[c] = sha(f)
    d = pd.read_csv(f, header=None, names=['d', 'vol', 'val', 'o', 'h', 'l', 'c', 'chg', 'n'], dtype=str)
    p = d['d'].str.strip().str.split('/', expand=True)
    d['date'] = pd.to_datetime(dict(year=pd.to_numeric(p[0], errors='coerce') + 1911, month=pd.to_numeric(p[1], errors='coerce'),
                                    day=pd.to_numeric(p[2], errors='coerce')), errors='coerce')
    lo, hi_ = max(first[c], D0), min(last[c], D1)
    d = d[(d['date'] >= lo) & (d['date'] <= hi_)].drop_duplicates('date', keep='last')
    if not len(d):
        continue
    ch = d['chg'].fillna('').astype(str).str.strip()
    x = ch.str.startswith('X')
    out = pd.DataFrame(dict(date=d['date'], code=c, open=num(d['o']), high=num(d['h']), low=num(d['l']), close=num(d['c']),
                            volume=num(d['vol']), amount=num(d['val']),
                            change=np.where(x, 0.0, pd.to_numeric(ch.str.replace('+', '', regex=False).str.replace(',', ''), errors='coerce')),
                            xflag=x.astype(int)))
    for k in ('open', 'high', 'low', 'close'):
        out.loc[~(out[k] > 0), k] = np.nan
    rows.append(out)
L = pd.concat(rows, ignore_index=True)
L = L[L['date'].isin(IDX)]                                   # 只留 data_panel 的交易日（已去掉全市場錯置日）
print('Asoul 列', len(L), '缺檔', len(no_file), flush=True)

COLS = np.array(sorted(codes))
ci = {c: j for j, c in enumerate(COLS)}; di = {d: i for i, d in enumerate(IDX)}
T, N = len(IDX), len(COLS)
A = {k: np.full((T, N), np.nan) for k in ('O', 'H', 'L', 'C', 'VOL', 'AMT', 'CHG')}
MK = np.zeros((T, N), np.int8)
ti = L['date'].map(di).to_numpy(); tj = L['code'].map(ci).to_numpy()
for k, col in (('O', 'open'), ('H', 'high'), ('L', 'low'), ('C', 'close'), ('VOL', 'volume'), ('AMT', 'amount'), ('CHG', 'change')):
    A[k][ti, tj] = L[col].to_numpy(float)
MK[ti, tj] = 1
# data_panel 有值、Asoul 沒有的格子
dpC = win.reindex(columns=COLS).to_numpy()
fill = np.isfinite(dpC) & ~np.isfinite(A['C'])
A['C'][fill] = dpC[fill]
A['O'][fill] = op.loc[IDX].reindex(columns=COLS).to_numpy()[fill]
A['H'][fill] = hi.loc[IDX].reindex(columns=COLS).to_numpy()[fill]
A['AMT'][fill] = va.loc[IDX].reindex(columns=COLS).to_numpy()[fill]
MK[fill] = 1
both = np.isfinite(dpC) & np.isfinite(A['C']) & ~fill
close_match = float((np.abs(A['C'][both] - dpC[both]) < 1e-6).mean())
np.savez_compressed(OUT / 'panel_cs.npz', IDX=IDX.values.astype('datetime64[ns]'), COLS=COLS.astype(str), C=A['C'], MK=MK,
                    O=A['O'], H=A['H'], L=A['L'], VOL=A['VOL'], AMT=A['AMT'], CHG=A['CHG'])
np.savez_compressed(OUT / 'innov.npz', A=np.zeros((T, N), bool), IDX=IDX.values.astype('datetime64[ns]'), COLS=COLS.astype(str),
                    codes=np.array([], dtype=str), names=np.array([], dtype=str))
e = pd.read_csv(OFF / 'e50_ohlc.csv', parse_dates=['date']).set_index('date')
e.reindex(IDX).rename_axis('date').reset_index().to_csv(OUT / 'e50_ohlc.csv', index=False)

# 衍生長表封存
RAWP.mkdir(parents=True, exist_ok=True)
buf = io.BytesIO()
with gzip.GzipFile(fileobj=buf, mode='wb', mtime=0) as g:
    g.write(L.assign(date=L['date'].dt.strftime('%Y-%m-%d')).to_csv(index=False).encode())
(RAWP / 'asoul_twse_2007_2015H1.csv.gz').write_bytes(buf.getvalue())
(RAWP / 'asoul_source_manifest.json').write_text(json.dumps(dict(
    source='Asoul/tsec 證交所原始行情的本機副本（研究中心先前下載，build_twse_panel.py 的來源 1）', path=str(TSEC),
    note='S55 只讀不改；本檔列出用到的來源檔 sha256；衍生長表＝上市期間（依 data_panel）2007-01-02～2015-06-30 的列，X 漲跌價差另記 xflag、change＝0',
    derived_sha256=sha(RAWP / 'asoul_twse_2007_2015H1.csv.gz'), files=manifest), ensure_ascii=False, indent=1))

# 檢查：與官方 2015-01-05～2015-06-30 日 K 的一致率（官方快取同期）
z = np.load(OFF / 'panel_cs.npz', allow_pickle=False)
oi = pd.DatetimeIndex(z['IDX']); oc = list(z['COLS']); om = (oi >= pd.Timestamp('2015-01-05')) & (oi <= D1)
common_d = oi[om].intersection(IDX); common_c = [c for c in COLS if c in set(oc)]
ii = [di[d] for d in common_d]; jj = [ci[c] for c in common_c]
oii = np.flatnonzero(np.isin(oi, common_d)); ojj = [oc.index(c) for c in common_c]
chk = {}
for k in ('C', 'O', 'AMT', 'CHG'):
    a = A[k][np.ix_(ii, jj)]; b = z[k][np.ix_(oii, ojj)]
    m = np.isfinite(a) & np.isfinite(b)
    chk[k] = dict(n=int(m.sum()), match=float((np.abs(a[m] - b[m]) <= 1e-6 * np.maximum(1, np.abs(b[m]))).mean()))
omk = z['MK'][np.ix_(oii, ojj)] == 1
mine = MK[np.ix_(ii, jj)] == 1
chk['MK_twse_cells_official'] = int(omk.sum()); chk['MK_twse_cells_mine'] = int(mine.sum()); chk['MK_agree'] = float((omk == mine).mean())
off_twse_codes = set(np.array(oc)[(z['MK'][oii] == 1).any(axis=0)])
chk['official_twse_codes_2015H1_missing_in_mine'] = sorted(off_twse_codes - set(COLS))
per_year = {str(y): dict(days=int((IDX.year == y).sum()), median_twse_listed=int(np.median((MK[IDX.year == y] == 1).sum(axis=1))),
                         median_traded=int(np.median(np.isfinite(A['C'][IDX.year == y]).sum(axis=1)))) for y in range(2007, 2016)}
lastv = pd.DataFrame(np.isfinite(A['C']), index=IDX, columns=COLS).apply(lambda s: s[s].index.max())
info = dict(range=[str(IDX[0].date()), str(IDX[-1].date())], T=T, N=N, n_asoul_rows=int(len(L)), n_x_rows=int(L['xflag'].sum()),
            n_codes_without_asoul_file=len(no_file), codes_without_asoul_file=no_file, n_cells_filled_from_data_panel=int(fill.sum()),
            close_match_vs_data_panel=close_match, n_cells_compared=int(both.sum()), per_year=per_year,
            n_codes_last_trade_before_2015_06=int((lastv < pd.Timestamp('2015-06-01')).sum()),
            vs_official_2015H1=chk, runtime_s=round(time.time() - t0, 1), maxrss_mb=round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024))
(HERE / 'panel_check.json').write_text(json.dumps(info, ensure_ascii=False, indent=1))
print(json.dumps({k: v for k, v in info.items() if k not in ('codes_without_asoul_file',)}, ensure_ascii=False, indent=1))
