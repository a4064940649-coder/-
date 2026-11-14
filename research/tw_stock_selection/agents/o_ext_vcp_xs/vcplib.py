#!/usr/bin/env python3
"""vcplib.py — 「VCP 突破警示」XS 腳本（raw/vcp_alert.xs）的 Python 忠實重做。
內容：資料層（最低價、成交股數的拼接與閘門）、訊號、事件研究、可執行組合引擎、統計。
只讀研究中心面板（scratchpad/panel）與公開原始檔；不 import 研究中心任何引擎。
執行一律 python3 -I -B run.py <stage>。規格全文見 FINDINGS 第 2 節（另一位研究員只看規格即可重做）。"""
import json, math, hashlib
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
S = Path('/tmp/claude-0/-home-user--/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad')
PANEL = S / 'panel'
SCR = S / 'ext_vcp'
DATA = HERE / 'data'
HIST = Path('/home/user/stevenhsiehsheng-ctrl/tw-stock-screener/data/history.csv.gz')
TAIEX = Path('/home/user/jonatw/finlab-txd/data/raw/taiex_twii.csv')
E50F = Path('/home/user/jonatw/finlab-txd/data/raw/etf_0050.csv')
DISPF = HERE.parent / 'o_disposal_list_pit_audit_pkl18' / 'data' / 'universe_announce.csv'
MAINNAV = HERE.parent / 'integrate_o20' / 'data' / 'v7x_navs_jmean_100w.csv.gz'
ODD_SWITCH = pd.Timestamp('2020-10-26')      # 盤中零股上線；之前只能整股（1,000 股）
LIM_SWITCH = pd.Timestamp('2015-06-01')      # 漲跌幅 7% → 10%
COLS9 = ['d', 'vol', 'val', 'o', 'h', 'l', 'c', 'chg', 'n']


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def tick(p):
    p = np.asarray(p, float)
    return np.select([p < 10, p < 50, p < 100, p < 500, p < 1000], [0.01, 0.05, 0.1, 0.5, 1.0], 5.0)


# =====================================================================
# 1. 資料層：最低價、成交股數（面板沒有）
# =====================================================================
def _num(s):
    return pd.to_numeric(s.astype(str).str.replace(',', '', regex=False).str.strip(), errors='coerce')


def _read_exchange(path, has_header):
    """證交所 STOCK_DAY 格式（Asoul/tsec 無表頭、民國日期；k66 有表頭、西元日期）。與 build_twse_panel.read_exchange 同樣的去重規則。"""
    if has_header:
        df = pd.read_csv(path, dtype=str); df.columns = COLS9[:len(df.columns)]
    else:
        df = pd.read_csv(path, header=None, names=COLS9, dtype=str)
    d = df['d'].astype(str).str.strip()
    if d.str.contains('/').all():
        p = d.str.split('/', expand=True).astype(int); p[0] += 1911
        df['date'] = pd.to_datetime(dict(year=p[0], month=p[1], day=p[2]), errors='coerce')
    else:
        df['date'] = pd.to_datetime(d, errors='coerce')
    for c in ['vol', 'val', 'o', 'h', 'l', 'c']:
        df[c] = _num(df[c])
    df = df.dropna(subset=['date', 'c']); df = df[df['c'] > 0]
    df = df.drop_duplicates('date').set_index('date').sort_index()
    return df[['vol', 'val', 'o', 'h', 'l', 'c']]


def build_lowvol(verbose=True):
    """依面板 src（1 Asoul/tsec、2 k66、3 Yahoo 5 分 K、4 官方日 K）逐格取同一來源的最低價與成交股數。
    src 3 沒有官方成交股數：成交股數 = 面板成交值 ÷ 面板收盤（等於 Yahoo 量 × 個股放大倍數），屬近似。
    輸出 data/low.parquet、data/volume.parquet（股）與閘門統計 dict。"""
    C = pd.read_parquet(PANEL / 'close_raw.parquet'); SRC = pd.read_parquet(PANEL / 'src.parquet')
    V = pd.read_parquet(PANEL / 'value.parquet'); Hh = pd.read_parquet(PANEL / 'high.parquet'); Oo = pd.read_parquet(PANEL / 'open.parquet')
    IDX, COLS = C.index, list(C.columns)
    hist = pd.read_csv(HIST, dtype={'code': str}, parse_dates=['date'])
    hist = hist[(hist['volume'] > 0) & (hist['close'] > 0)]
    hist_g = {c: g.drop_duplicates('date').set_index('date').sort_index() for c, g in hist.groupby('code')}
    LOW = np.full(C.shape, np.nan); VOL = np.full(C.shape, np.nan); SCL = np.full(C.shape, np.nan)
    yah_chk = []     # src 3 的品質檢查：Yahoo 與官方重疊期
    miss_file = {1: 0, 2: 0, 3: 0, 4: 0}
    fb = {'low_from_yahoo': 0, 'low_from_min_open_close': 0, 'vol_from_value_div_close': 0}
    for j, code in enumerate(COLS):
        s = SRC[code].to_numpy(); c = C[code].to_numpy()
        srcs = {}
        if (s == 1).any():
            p = S / 'tsec_all' / f'{code}.csv'
            if p.exists(): srcs[1] = _read_exchange(p, False)
            else: miss_file[1] += 1
        if (s == 2).any():
            p = SCR / 'k66' / f'{code}.csv'
            if p.exists(): srcs[2] = _read_exchange(p, True)
            else: miss_file[2] += 1
        y0 = None
        if (s == 3).any() or code in hist_g:
            p = S / 'tw_stocker_daily' / f'{code}.csv'
            if p.exists():
                y0 = pd.read_csv(p, parse_dates=['date']).drop_duplicates('date').set_index('date').sort_index()
                y0 = y0[(y0['volume'] > 0) & (y0['close'] > 0)]
            elif (s == 3).any(): miss_file[3] += 1
        if (s == 4).any():
            if code in hist_g: srcs[4] = hist_g[code].rename(columns={'low': 'l', 'volume': 'vol', 'close': 'c'})
            else: miss_file[4] += 1
        for k, df in srcs.items():
            m = np.where(s == k)[0]
            r = df.reindex(IDX[m])
            LOW[m, j] = r['l'].to_numpy(); VOL[m, j] = r['vol'].to_numpy(); SCL[m, j] = r['c'].to_numpy()
        if y0 is not None:
            m = np.where(s == 3)[0]
            if len(m):
                r = y0.reindex(IDX[m])
                LOW[m, j] = r['low'].to_numpy(); SCL[m, j] = r['close'].to_numpy()
                VOL[m, j] = V[code].to_numpy()[m] / c[m]
            if code in hist_g:      # Yahoo 對官方（重疊期 2025-09-24 起）
                h = hist_g[code]; jn = h[['low', 'volume', 'close']].join(y0[['low', 'volume', 'close']], rsuffix='_y', how='inner')
                jn = jn[jn.index >= pd.Timestamp('2025-09-24')]
                if len(jn) >= 20:
                    sc = float(np.clip((jn['volume'] / jn['volume_y']).median(), 1.0, 3.0))
                    yah_chk.append(pd.DataFrame({'code': code, 'vol_ratio': jn['volume_y'] * sc / jn['volume'],
                                                 'low_diff': jn['low_y'] / jn['low'] - 1, 'close_diff': jn['close_y'] / jn['close'] - 1,
                                                 'tick_low': tick(jn['low'].to_numpy()) / jn['low'].to_numpy()}))
        # 補缺：面板有收盤但該來源已取不到列（history.csv.gz 是滾動檔，2025-09-24 已滾出）→ 最低價先用 Yahoo、再用 min(開, 收)；成交股數用 成交值÷收盤
        miss = np.where(np.isfinite(c) & ~np.isfinite(LOW[:, j]))[0]
        if len(miss):
            oc = np.fmin(Oo[code].to_numpy()[miss], c[miss])
            if y0 is not None:
                yl = y0['low'].reindex(IDX[miss]).to_numpy()
                LOW[miss, j] = np.fmin(yl, oc); fb['low_from_yahoo'] += int(np.isfinite(yl).sum())
            st = miss[~np.isfinite(LOW[miss, j])]
            LOW[st, j] = np.fmin(Oo[code].to_numpy()[st], c[st]); fb['low_from_min_open_close'] += len(st)
        missv = np.where(np.isfinite(c) & ~np.isfinite(VOL[:, j]))[0]
        if len(missv):
            VOL[missv, j] = V[code].to_numpy()[missv] / c[missv]; fb['vol_from_value_div_close'] += len(missv)
        if verbose and j % 300 == 0: print('lowvol', j, code, flush=True)
    Cn = C.to_numpy(); Hn = Hh.to_numpy(); On = Oo.to_numpy(); Vn = V.to_numpy(); Sn = SRC.to_numpy()
    has = np.isfinite(Cn)
    gate = {'cells_with_close': int(has.sum()), 'missing_source_files': miss_file, 'fallback_cells': fb}
    gate['low_missing_share'] = float((has & ~np.isfinite(LOW)).sum() / has.sum())
    gate['vol_missing_share'] = float((has & ~np.isfinite(VOL)).sum() / has.sum())
    yrs = IDX.year.to_numpy()
    gate['by_year'] = {}
    for y in sorted(set(yrs)):
        m = yrs == y; hh = has[m]
        gate['by_year'][int(y)] = {'cells': int(hh.sum()), 'low_missing': round(float((hh & ~np.isfinite(LOW[m])).sum() / max(hh.sum(), 1)), 5),
                                   'vol_missing': round(float((hh & ~np.isfinite(VOL[m])).sum() / max(hh.sum(), 1)), 5)}
    ok = has & np.isfinite(LOW)
    with np.errstate(invalid='ignore', divide='ignore'):
        gate['source_close_mismatch_share'] = float((ok & (np.abs(SCL - Cn) > 0.011)).sum() / ok.sum())
        gate['low_gt_close_share'] = float((ok & (LOW > Cn + 1e-9)).sum() / ok.sum())
        gate['low_gt_open_share'] = float((ok & np.isfinite(On) & (LOW > On + 1e-9)).sum() / ok.sum())
        gate['low_gt_high_share'] = float((ok & np.isfinite(Hn) & (LOW > Hn + 1e-9)).sum() / ok.sum())
        gate['low_le_zero'] = int((ok & (LOW <= 0)).sum())
        # 成交值÷收盤 對 真實成交股數 的誤差（官方來源 1、2、4）
        off = has & np.isfinite(VOL) & (VOL > 0) & np.isin(Sn, [1, 2, 4]) & np.isfinite(Vn) & (Vn > 0)
        rel = (Vn[off] / Cn[off]) / VOL[off] - 1
    q = np.nanpercentile(rel, [1, 5, 25, 50, 75, 95, 99])
    gate['value_div_close_vs_shares_relerr_pct'] = {k: round(float(v) * 100, 3) for k, v in zip(['p1', 'p5', 'p25', 'p50', 'p75', 'p95', 'p99'], q)}
    gate['value_div_close_vs_shares_abs_gt5pct_share'] = float((np.abs(rel) > 0.05).mean())
    if yah_chk:
        yc = pd.concat(yah_chk)
        vr = yc['vol_ratio'].to_numpy()
        gate['src3_check_2025-09-24_on'] = {
            'rows': int(len(yc)), 'stocks': int(yc['code'].nunique()),
            'scaled_yahoo_vol_over_official_pct': {k: round(float(v), 3) for k, v in zip(['p5', 'p25', 'p50', 'p75', 'p95'], np.nanpercentile(vr, [5, 25, 50, 75, 95]))},
            'abs_err_gt20pct_share': float((np.abs(vr - 1) > 0.2).mean()),
            'low_exact_share': float((yc['low_diff'].abs() < 1e-4).mean()),
            'low_within_1tick_share': float((yc['low_diff'].abs() <= yc['tick_low'] + 1e-4).mean()),
            'yahoo_low_higher_share': float((yc['low_diff'] > 1e-4).mean()),
            'yahoo_low_lower_share': float((yc['low_diff'] < -1e-4).mean()),
            'close_exact_share': float((yc['close_diff'].abs() < 1e-4).mean())}
    DATA.mkdir(exist_ok=True)
    pd.DataFrame(LOW, index=IDX, columns=COLS).to_parquet(DATA / 'low.parquet')
    pd.DataFrame(VOL, index=IDX, columns=COLS).astype('float32').to_parquet(DATA / 'volume.parquet')
    gate['files'] = {f: {'bytes': (DATA / f).stat().st_size, 'sha256': sha(DATA / f)} for f in ['low.parquet', 'volume.parquet']}
    return gate


# =====================================================================
# 2. 載入與衍生陣列（日期座標 [T×N] 與 K 棒座標 [B×N]）
# =====================================================================
class _D:
    pass


G = _D()


def load():
    if getattr(G, 'ok', False): return G
    rd = lambda k: pd.read_parquet(PANEL / f'{k}.parquet')
    C = rd('close_raw'); G.IDX = C.index; G.COLS = list(C.columns); G.T, G.N = C.shape
    al = lambda df: df.reindex(index=G.IDX, columns=G.COLS).to_numpy().astype(float)
    G.C = C.to_numpy().astype(float); G.O = al(rd('open')); G.H = al(rd('high')); G.VAL = al(rd('value'))
    G.ADJ = al(rd('adj')); G.RET = al(rd('ret')); G.FLAG = al(rd('flag')); G.SRC = al(rd('src'))
    G.L = al(pd.read_parquet(DATA / 'low.parquet')); G.VOL = al(pd.read_parquet(DATA / 'volume.parquet'))   # 股
    G.TR = np.isfinite(G.C)
    with np.errstate(invalid='ignore', divide='ignore'):
        G.ADJF = G.ADJ / G.C
    G.AC = G.ADJ.copy(); G.ACF = pd.DataFrame(G.AC).ffill().to_numpy()
    G.AO = G.O * G.ADJF; G.AH = G.H * G.ADJF; G.AL = G.L * G.ADJF
    PC = pd.DataFrame(G.C).ffill().shift(1).to_numpy()
    with np.errstate(invalid='ignore', divide='ignore'):
        ref = G.C / (1.0 + G.RET)
    G.REF = np.where((G.FLAG == 1) | ~np.isfinite(ref), PC, ref)       # 漲跌停參考價
    G.LIM = np.where(G.IDX < LIM_SWITCH, 0.07, 0.10)[:, None]
    raw = G.REF * (1 + G.LIM); tk = tick(raw)
    G.UPLIM = np.floor(raw / tk + 1e-6) * tk
    with np.errstate(invalid='ignore'):
        G.OPENLIM = np.isfinite(G.O) & (G.O >= G.UPLIM - 1e-6)                       # 開盤即漲停
        G.LOCKUP = G.OPENLIM & (G.L >= G.UPLIM - 1e-6)                               # 全日鎖漲停（最低價也在漲停）
        G.LOCKDN = G.TR & (G.H == G.L) & (G.RET <= -(G.LIM - 0.006))                 # 全日鎖跌停
    G.BARNO = np.where(G.TR, np.cumsum(G.TR, axis=0), 0)
    e50 = pd.read_csv(E50F, parse_dates=['date']).set_index('date')['adj'].sort_index()
    G.E50 = e50.reindex(G.IDX.union(e50.index)).ffill().reindex(G.IDX).to_numpy()
    tw = pd.read_csv(TAIEX, parse_dates=['date']).set_index('date')['close'].sort_index()
    G.TWII_raw_missing = int(tw.reindex(G.IDX).isna().sum())
    G.TWII = tw.reindex(G.IDX.union(tw.index)).ffill().reindex(G.IDX).to_numpy()
    # K 棒座標
    order = np.argsort(~G.TR, axis=0, kind='stable')
    G.NB = G.TR.sum(0); G.B = int(G.NB.max())
    G.POS = order[:G.B].copy(); G.BMASK = np.arange(G.B)[:, None] < G.NB[None, :]
    G.POS[~G.BMASK] = -1
    # 處置：DISP[e, j] = e 在處置期間，且公告日 < e（即 ≤ 前一交易日）
    dv = pd.read_csv(DISPF, dtype={'code': str}, parse_dates=['announce', 'start', 'end'])
    CI = {c: j for j, c in enumerate(G.COLS)}; G.CI = CI
    G.DISP = np.zeros((G.T, G.N), bool); days = G.IDX.values; n_ev = 0
    for c, an, s, e in zip(dv.code, dv.announce, dv.start, dv.end):
        j = CI.get(c)
        if j is None or pd.isna(s) or pd.isna(e) or pd.isna(an): continue
        a = max(np.searchsorted(days, np.datetime64(s)), np.searchsorted(days, np.datetime64(an), side='right'))
        b = np.searchsorted(days, np.datetime64(e), side='right')
        if b > a: G.DISP[a:b, j] = True; n_ev += 1
    G.N_DISP = n_ev; G.DISP_RANGE = (str(dv.announce.min().date()), str(dv.announce.max().date()))
    # E3 用：還原收盤 < 還原收盤 50 根均線（K 棒座標算好再放回日期座標）
    ac = comp(G.ADJ)
    below = ac < pd.DataFrame(ac).rolling(50, min_periods=50).mean().to_numpy()
    G.BELOW50 = decomp(below, False).astype(bool)
    G.ok = True
    return G


def comp(X):
    """日期座標 → K 棒座標（每檔股票把有成交的日子往上擠緊）。"""
    out = np.take_along_axis(X, np.where(G.POS >= 0, G.POS, 0), axis=0).astype(float)
    out[~G.BMASK] = np.nan
    return out


def decomp(Xc, fill=np.nan):
    out = np.full((G.T, G.N), fill, dtype=float)
    b, j = np.where(G.BMASK)
    out[G.POS[b, j], j] = Xc[b, j]
    return out


def _sh(X, k):
    """X[k]：k 根之前的值。"""
    if k == 0: return X
    out = np.full_like(X, np.nan); out[k:] = X[:-k]
    return out


def _rmean(X, n): return pd.DataFrame(X).rolling(n, min_periods=n).mean().to_numpy()
def _rmax(X, n): return pd.DataFrame(X).rolling(n, min_periods=n).max().to_numpy()
def _rmin(X, n): return pd.DataFrame(X).rolling(n, min_periods=n).min().to_numpy()


# =====================================================================
# 3. 訊號（逐行對應 raw/vcp_alert.xs）
# =====================================================================
DEFAULT = dict(SegLen=25, MaxLastDepth=10.0, MinFirstDepth=10.0, VolDryRatio=0.7, VolBreak=1.5, En_RS=1, RSMinDiff=0.0,
               MinVol=500.0, LowMult=1.3, HighPct=25.0, MA200UpBars=20, StartBar=260)
_CACHE = {}


def _base(basis):
    """與 SegLen／量能參數無關的部分，快取。"""
    if basis in _CACHE: return _CACHE[basis]
    load()
    if basis == 'raw': Cc, Hc, Lc = comp(G.C), comp(G.H), comp(G.L)
    else: Cc, Hc, Lc = comp(G.ADJ), comp(G.AH), comp(G.AL)
    Vc = comp(G.VOL) / 1000.0                                   # 張
    Ic = comp(np.repeat(G.TWII[:, None], G.N, axis=1))          # 加權指數收盤，對齊到該股 K 棒日期
    o = dict(C=Cc, H=Hc, L=Lc, V=Vc, I=Ic,
             ma50=_rmean(Cc, 50), ma150=_rmean(Cc, 150), ma200=_rmean(Cc, 200),
             low52=_rmin(Lc, 250), high52=_rmax(Hc, 250),
             av10=_rmean(_sh(Vc, 1), 10), av50=_rmean(_sh(Vc, 1), 50))
    with np.errstate(invalid='ignore', divide='ignore'):
        o['stkroc'] = (Cc / _sh(Cc, 250) - 1) * 100
        o['idxroc'] = (Ic / _sh(Ic, 250) - 1) * 100
    _CACHE[basis] = o
    return o


def compute_signal(params=None, basis='raw', drop=()):
    """回傳 dict：sig（K 棒座標 bool）與各條件、各中間值。drop：要拿掉的條件名稱集合（消融用）。"""
    P = dict(DEFAULT); P.update(params or {})
    b = _base(basis); C, H, L, V = b['C'], b['H'], b['L'], b['V']
    S = int(P['SegLen']); sb = int(P['StartBar'])
    barno = np.arange(1, G.B + 1)[:, None] * np.ones((1, G.N))
    okbar = (barno >= sb) & (barno > 3 * S + 1) & G.BMASK                      # if CurrentBar >= 260 and CurrentBar > 3*SegLen+1
    ma50, ma150, ma200 = b['ma50'], b['ma150'], b['ma200']
    k = int(P['MA200UpBars'])
    ma200_prev = np.where(barno - k >= sb, _sh(ma200, k), 0.0)                  # _ma200[20]；第 260 根之前變數未賦值＝初值 0
    with np.errstate(invalid='ignore', divide='ignore'):
        trend = ((C > ma150) & (C > ma200) & (ma150 > ma200) & (ma200 > ma200_prev) & (ma50 > ma150) & (ma50 > ma200) & (C > ma50)
                 & (b['low52'] > 0) & (C >= b['low52'] * P['LowMult']) & (b['high52'] > 0) & (C >= b['high52'] * (1 - P['HighPct'] / 100)))
        h3 = _rmax(_sh(H, 1), S); l3 = _rmin(_sh(L, 1), S)                      # Highest(High[1], SegLen)
        h2 = _rmax(_sh(H, S + 1), S); l2 = _rmin(_sh(L, S + 1), S)              # Highest(High[SegLen+1], SegLen)
        h1 = _rmax(_sh(H, 2 * S + 1), S); l1 = _rmin(_sh(L, 2 * S + 1), S)      # Highest(High[2*SegLen+1], SegLen)
        d1 = (h1 - l1) / h1 * 100; d2 = (h2 - l2) / h2 * 100; d3 = (h3 - l3) / h3 * 100
        contract = (d1 >= P['MinFirstDepth']) & (d1 > d2) & (d2 > d3) & (d3 <= P['MaxLastDepth'])
        av10, av50 = b['av10'], b['av50']
        vr = av10 / av50
        voldry = (av50 > 0) & (vr < P['VolDryRatio'])
        pivot = h3
        rsok = ~(b['stkroc'] < b['idxroc'] + P['RSMinDiff']) if P['En_RS'] == 1 else np.ones_like(trend)
        pivot_prev = np.where(barno - 1 >= sb, _sh(pivot, 1), 0.0)             # _pivot[1]
        cross = (C > pivot) & (_sh(C, 1) <= pivot_prev)                         # Close crosses over _pivot
        volbrk = V > av50 * P['VolBreak']
        minvol = (V >= P['MinVol']) if P['MinVol'] != 0 else np.ones_like(trend)
    conds = dict(trend=trend, contract=contract, voldry=voldry, rs=rsok, volbrk=volbrk, minvol=minvol)
    sig = okbar & (pivot > 0) & cross
    for name, m in conds.items():
        if name not in drop: sig = sig & m
    return dict(sig=sig, okbar=okbar, cross=cross, barno=barno, pivot=pivot, h1=h1, l1=l1, h2=h2, l2=l2, h3=h3, l3=l3, d1=d1, d2=d2, d3=d3,
                vr=vr, ma200_prev=ma200_prev, pivot_prev=pivot_prev, **conds, **{k_: b[k_] for k_ in ('C', 'V', 'ma50', 'ma150', 'ma200', 'low52', 'high52', 'av10', 'av50', 'stkroc', 'idxroc')})


def signal_table(R, full=True, basis='raw'):
    """K 棒座標訊號 → 逐筆表（含 d+1 開盤的可成交性欄位）。"""
    b, j = np.where(R['sig']); t = G.POS[b, j]
    o = pd.DataFrame({'date': G.IDX[t], 'code': np.array(G.COLS)[j], 't': t, 'j': j, 'barno': (b + 1)})
    g = lambda k: R[k][b, j]
    o['close'] = g('C'); o['pivot'] = g('pivot'); o['vbr'] = g('V') / g('av50'); o['vol_lots'] = g('V')
    if full:
        for k in ('pivot_prev', 'h1', 'l1', 'h2', 'l2', 'h3', 'l3', 'd1', 'd2', 'd3', 'av10', 'av50', 'vr', 'ma50', 'ma150', 'ma200', 'ma200_prev',
                  'low52', 'high52', 'stkroc', 'idxroc'):
            o[k] = g(k)
        o['prev_close'] = _sh(R['C'], 1)[b, j]
    o['raw_close'] = G.C[t, j]; o['flag_d'] = G.FLAG[t, j]; o['src_d'] = G.SRC[t, j]
    o['pivot_adj'] = o['pivot'] * (G.ADJF[t, j] if basis == 'raw' else 1.0)      # 樞紐換成還原價座標（E5 出場用）
    e = t + 1; okE = e < G.T; ee = np.where(okE, e, G.T - 1)
    o['e'] = np.where(okE, e, -1)
    o['open_e'] = np.where(okE, G.O[ee, j], np.nan)
    o['noopen'] = ~(okE & np.isfinite(G.O[ee, j]) & G.TR[ee, j])
    o['openlim'] = okE & G.OPENLIM[ee, j]; o['lockup'] = okE & G.LOCKUP[ee, j]
    o['disp_e'] = okE & G.DISP[ee, j]
    with np.errstate(invalid='ignore', divide='ignore'):
        o['gap_ref'] = np.where(okE, G.O[ee, j] / G.REF[ee, j] - 1, np.nan)      # 隔日開盤相對參考價（除權息日已換成參考價）
    o['flag_e'] = np.where(okE, G.FLAG[ee, j], np.nan)
    return o.sort_values(['t', 'code']).reset_index(drop=True)


# =====================================================================
# 4. 統計工具
# =====================================================================
def tstat(x):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    if len(x) < 3 or x.std(ddof=1) == 0: return float('nan')
    return float(x.mean() / (x.std(ddof=1) / math.sqrt(len(x))))


def cluster_t(x, g):
    """cluster-robust t（群＝訊號月份），含 G/(G−1)。回傳 mean, t, G, deff, neff。"""
    x = np.asarray(x, float); ok = np.isfinite(x); x = x[ok]; g = np.asarray(g)[ok]; n = len(x)
    if n < 5: return dict(mean=float('nan'), t=float('nan'), G=0, n=n, neff=float('nan'))
    m = x.mean(); e = x - m
    s = pd.Series(e).groupby(g).sum(); Gn = len(s)
    if Gn < 3: return dict(mean=float(m), t=float('nan'), G=int(Gn), n=n, neff=float('nan'))
    var = float((s ** 2).sum()) / n ** 2 * Gn / (Gn - 1)
    naive = x.var(ddof=1) / n
    mm = pd.Series(x).groupby(g).mean()
    return dict(mean=float(m), t=float(m / math.sqrt(var)), G=int(Gn), n=int(n), neff=float(n * naive / var), t_month_mean=tstat(mm.to_numpy()), mean_month_mean=float(mm.mean()))


def block_boot_p(x, g, block=3, B=4000, seed=7):
    """月份區塊 bootstrap：把事件依月份加總，連續 block 個月為一塊（循環），重抽塊；回傳平均的雙尾 p 與 95% 區間。"""
    x = np.asarray(x, float); ok = np.isfinite(x); x = x[ok]; g = np.asarray(g)[ok]
    df = pd.DataFrame({'x': x, 'g': g}); ms = df.groupby('g')['x'].agg(['sum', 'count']).sort_index()
    allm = pd.period_range(ms.index.min(), ms.index.max(), freq='M'); ms = ms.reindex(allm).fillna(0.0)
    s = ms['sum'].to_numpy(); c = ms['count'].to_numpy(); M = len(s)
    rng = np.random.default_rng(seed); nb = int(math.ceil(M / block)); means = np.empty(B)
    for i in range(B):
        st = rng.integers(0, M, nb); ix = (st[:, None] + np.arange(block)[None, :]).ravel() % M
        cc = c[ix].sum(); means[i] = s[ix].sum() / cc if cc > 0 else np.nan
    m0 = x.mean(); means = means[np.isfinite(means)]
    p = 2 * min((means <= 0).mean(), (means >= 0).mean())
    return dict(mean=float(m0), p_two_sided=float(min(p, 1.0)), ci95=[float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))], months=int(M))


def cagr(nav, a, b):
    yrs = (b - a).days / 365.25
    return float((nav[b] / nav[a]) ** (1 / yrs) - 1)


def month_ret(nav):
    m = nav.groupby([nav.index.year, nav.index.month]).last()
    m.index = pd.PeriodIndex([pd.Period(year=y, month=mo, freq='M') for y, mo in m.index], freq='M')
    return m.pct_change().dropna()


def period_stats(nav, bench, a, b):
    """nav、bench：日頻 Series（同索引）。區間＝[第一個 ≥ a 的交易日收盤, 最後一個 ≤ b 的交易日收盤]。"""
    ix = nav.index; ia = ix[ix >= pd.Timestamp(a)]; ib = ix[ix <= pd.Timestamp(b)]
    if len(ia) == 0 or len(ib) == 0 or ia[0] >= ib[-1]: return None
    a0, b0 = ia[0], ib[-1]
    n = nav.loc[a0:b0]; e = bench.loc[a0:b0]
    mr = month_ret(nav); me = month_ret(bench).reindex(mr.index)
    sel = (mr.index >= pd.Period(a0, 'M')) & (mr.index <= pd.Period(b0, 'M'))
    x = (mr - me)[sel]
    dr = n.pct_change().dropna()
    out = dict(start=str(a0.date()), end=str(b0.date()), cagr=cagr(nav, a0, b0), bench=cagr(bench, a0, b0), n_months=int(len(x)),
               t=tstat(x.to_numpy()), mean_m_excess=float(x.mean()), maxdd=float((n / n.cummax() - 1).min()), bench_maxdd=float((e / e.cummax() - 1).min()),
               vol=float(dr.std() * math.sqrt(252)))
    out['excess'] = out['cagr'] - out['bench']
    return out


def ex_best3(nav, bench, a, b):
    ix = nav.index; a0 = ix[ix >= pd.Timestamp(a)][0]; b0 = ix[ix <= pd.Timestamp(b)][-1]
    mr = month_ret(nav); me = month_ret(bench).reindex(mr.index)
    sel = (mr.index >= pd.Period(a0, 'M')) & (mr.index <= pd.Period(b0, 'M'))
    x = (mr - me)[sel]
    yr = {}
    for y in sorted(set(x.index.year)):
        m = x.index.year == y
        yr[int(y)] = float((1 + mr[sel][m]).prod() - (1 + me[sel][m]).prod())
    best = sorted(yr, key=lambda y: -yr[y])[:3]
    rest = x[~np.isin(x.index.year, best)]
    return dict(best3_years=best, t_ex_best3=tstat(rest.to_numpy()), n_months=int(len(rest)), mean_m_excess=float(rest.mean()), yearly_excess={k: round(v, 4) for k, v in yr.items()})


def spa_test(X, B=5000, q=1 / 6, seed=11):
    """Hansen (2005) SPA（consistent p）。X：[n × k] 月超額。定態 bootstrap，平均區塊 1/q 個月。"""
    X = np.asarray(X, float); n, k = X.shape
    mu = X.mean(0); rng = np.random.default_rng(seed)
    idx = np.empty((B, n), int)
    for b_ in range(B):
        i = rng.integers(0, n); row = np.empty(n, int)
        for t in range(n):
            row[t] = i
            i = rng.integers(0, n) if rng.random() < q else (i + 1) % n
        idx[b_] = row
    bm = X[idx].mean(1)                                   # [B × k]
    om = np.sqrt(n) * bm.std(0, ddof=1)                   # ω_k
    om = np.where(om > 0, om, np.inf)
    T = max(0.0, float(np.max(np.sqrt(n) * mu / om)))
    thr = -np.sqrt(2 * np.log(np.log(n)))
    # Hansen：Z*_k = sqrt(n)(X̄*_k − g_c(X̄_k))/ω_k，g_c(x)=x·1{sqrt(n)x/ω ≥ −sqrt(2 ln ln n)}；明顯較差的模型不拉回 0
    Zc = np.sqrt(n) * (bm - mu) / om + np.where(np.sqrt(n) * mu / om >= thr, 0.0, np.sqrt(n) * mu / om)
    Tb = np.maximum(0.0, Zc.max(1))
    Tl = np.maximum(0.0, (np.sqrt(n) * (bm - mu) / om).max(1))          # White reality check（上界 p）
    return dict(T=T, p_consistent=float((Tb >= T).mean()), p_upper=float((Tl >= T).mean()), n=int(n), k=int(k))


def dsr(x, N, var_sr):
    """Deflated Sharpe Ratio（Bailey & López de Prado 2014）。x：月超額；N：試驗數；var_sr：試驗間月 SR 的變異數。"""
    from statistics import NormalDist
    nd = NormalDist(); x = np.asarray(x, float); n = len(x)
    sr = x.mean() / x.std(ddof=1)
    g = 0.5772156649
    sr0 = math.sqrt(var_sr) * ((1 - g) * nd.inv_cdf(1 - 1 / N) + g * nd.inv_cdf(1 - 1 / (N * math.e)))
    z = (x - x.mean()) / x.std(ddof=0); sk = float((z ** 3).mean()); ku = float((z ** 4).mean())
    den = math.sqrt(max(1e-12, 1 - sk * sr + (ku - 1) / 4 * sr ** 2))
    return dict(sr_m=float(sr), sr0_m=float(sr0), dsr=float(nd.cdf((sr - sr0) * math.sqrt(n - 1) / den)), skew=sk, kurt=ku, n=int(n))


# =====================================================================
# 5. 事件研究
# =====================================================================
POOL_MIN_LOTS = 500.0


def event_study(sig, horizons=(5, 10, 20, 60), with_pool=True):
    """sig：signal_table 的結果（全部訊號，含買不到的）。回傳逐筆報酬表（只含可成交者）與排除統計。
    個股：ACF[e+h−1] ÷ AO[e] − 1；0050：E50[d+h] ÷ E50[d] − 1；同池：d 日成交量 ≥ 500 張、K 棒 ≥ 260、e 有開盤者，同式等權。"""
    load()
    ex = dict(n_signals=int(len(sig)), noopen=int(sig['noopen'].sum()), openlim=int((~sig['noopen'] & sig['openlim']).sum()),
              openlim_locked_all_day=int((~sig['noopen'] & sig['lockup']).sum()), disposal_on_e=int((~sig['noopen'] & ~sig['openlim'] & sig['disp_e']).sum()))
    ev = sig[~sig['noopen'] & ~sig['openlim']].copy()
    t = ev['t'].to_numpy(); j = ev['j'].to_numpy(); e = t + 1
    ev['month'] = pd.PeriodIndex(ev['date'], freq='M')
    px = G.O[e, j]
    ev['cost_rt'] = 2 * 0.001425 * 0.28 + 0.003 + 2 * 0.5 * tick(px) / px        # 來回比例成本（手續費×2＋稅＋半跳×2）
    pool_cache = {}
    for h in horizons:
        x = e + h - 1; ok = x < G.T; xx = np.where(ok, x, G.T - 1)
        r = np.where(ok, G.ACF[xx, j] / G.AO[e, j] - 1, np.nan)
        ev[f'r{h}'] = r; ev[f'rn{h}'] = (1 + r) * (1 - ev['cost_rt'].to_numpy()) - 1
        ev[f'b{h}'] = np.where(ok, G.E50[xx] / G.E50[t] - 1, np.nan)
        if with_pool:
            pm = np.full(len(ev), np.nan); pn = np.zeros(len(ev))
            for i, (ti, oki) in enumerate(zip(t, ok)):
                if not oki: continue
                key = (ti, h)
                if key not in pool_cache:
                    ei = ti + 1
                    m = (G.VOL[ti] >= POOL_MIN_LOTS * 1000) & (G.BARNO[ti] >= 260) & np.isfinite(G.AO[ei]) & G.TR[ei]
                    with np.errstate(invalid='ignore', divide='ignore'):
                        rr = G.ACF[ei + h - 1, m] / G.AO[ei, m] - 1
                    pool_cache[key] = (float(np.nanmean(rr)), int(m.sum()))
                pm[i], pn[i] = pool_cache[key]
            ev[f'p{h}'] = pm; ev['pool_n'] = pn
    return ev, ex


def event_summary(ev, horizons=(5, 10, 20, 60), boot=True):
    out = {}
    for h in horizons:
        d = ev[np.isfinite(ev[f'r{h}'])]
        if len(d) < 5: out[str(h)] = None; continue
        r = d[f'r{h}'].to_numpy(); g = d['month'].astype(str).to_numpy()
        o = dict(n=int(len(d)), mean=float(r.mean()), median=float(np.median(r)), win=float((r > 0).mean()),
                 mean_net=float(d[f'rn{h}'].mean()), median_net=float(d[f'rn{h}'].median()), win_net=float((d[f'rn{h}'] > 0).mean()),
                 bench0050_mean=float(d[f'b{h}'].mean()), bench0050_median=float(d[f'b{h}'].median()), bench0050_win=float((d[f'b{h}'] > 0).mean()))
        o['x0050'] = cluster_t(r - d[f'b{h}'].to_numpy(), g); o['x0050_net'] = cluster_t(d[f'rn{h}'].to_numpy() - d[f'b{h}'].to_numpy(), g)
        o['x0050']['median'] = float(np.median(r - d[f'b{h}'].to_numpy())); o['x0050']['win'] = float((r > d[f'b{h}'].to_numpy()).mean())
        if f'p{h}' in d:
            p = d[f'p{h}'].to_numpy()
            o['pool_mean'] = float(np.nanmean(p)); o['xpool'] = cluster_t(r - p, g); o['xpool_net'] = cluster_t(d[f'rn{h}'].to_numpy() - p, g)
            o['xpool']['median'] = float(np.nanmedian(r - p)); o['xpool']['win'] = float((r > p).mean())
            if boot:
                o['xpool_boot3m'] = block_boot_p(r - p, d['month'].to_numpy()); o['x0050_boot3m'] = block_boot_p(r - d[f'b{h}'].to_numpy(), d['month'].to_numpy())
        out[str(h)] = o
    return out


# =====================================================================
# 6. 可執行組合引擎
# =====================================================================
FEE = 0.001425 * 0.28
TAX = 0.003
MINFEE = 20.0
EXITS = {'E1_hold20': dict(hold=20), 'E2_hold60': dict(hold=60), 'E3_stop8_ma50': dict(stop=0.08, ma50=True, hold=250),
         'E4_stop8_tp20': dict(stop=0.08, tp=0.20, hold=250), 'E5_close_below_pivot': dict(pivot=True, hold=250), 'E6_trail10': dict(trail=0.10, hold=250)}


def simulate(sig, rule, K=10, capital=1_000_000.0, cost='normal', idle='CASH', t0=0, t1=None, lag=0, buy_disp=False, buy_openlim=False,
             fixed_amt=None, cap_share=0.05, log=True):
    """日頻事件驅動模擬。sig 需要欄位 t, j, code, vbr, pivot_adj。回傳 (nav Series, trades DataFrame, info dict)。
    每日順序：(0) 空位 0050 計息 → (1) 開盤賣出前一日收盤已判定者 → (2) 開盤買進 → (3) 盤中停損／停利 → (4) 收盤：到期賣出、判定次日賣出、結算淨值。"""
    load(); X = EXITS[rule]; t1 = G.T - 1 if t1 is None else t1
    stress = 0.003 if cost == 'stress' else 0.0; zero = cost == 'zero'
    hold_n = X.get('hold'); stop = X.get('stop'); tp = X.get('tp'); trail = X.get('trail')
    cand = {}
    for t, j, code, vbr, pa in zip(sig['t'].to_numpy(), sig['j'].to_numpy(), sig['code'].to_numpy(), sig['vbr'].to_numpy(), sig['pivot_adj'].to_numpy()):
        cand.setdefault(int(t) + 1 + lag, []).append((-float(vbr), str(code), int(j), int(t), float(pa)))
    r50 = np.full(G.T, 0.0); r50[1:] = G.E50[1:] / G.E50[:-1] - 1; r50[~np.isfinite(r50)] = 0.0
    cash = float(capital); pos = {}; nav = np.empty(t1 - t0 + 1); trades = []
    info = dict(n_cand=0, skip_held=0, skip_noopen=0, skip_openlim=0, skip_disp=0, skip_full=0, skip_lot=0, skip_cash=0, n_buy=0, cap_bound=0,
                minfee_orders=0, n_orders=0, fee_paid=0.0, tax_paid=0.0, slip_paid=0.0, turnover_buy=0.0, max_pos=0, sum_pos=0, sum_invested_share=0.0, days=0)
    nav_prev = cash

    def sell(j, t, fill_adj, reason):
        nonlocal cash
        p = pos.pop(j)
        raw = fill_adj / G.ADJF[t, j]
        srate = 0.0 if zero else 0.5 * float(tick(raw)) / raw + stress
        gross = p['u'] * fill_adj; amt = gross * (1 - srate)
        fee = 0.0 if zero else max(MINFEE, amt * FEE); tax = 0.0 if zero else amt * TAX
        net = amt - fee - tax
        if idle == 'P50': net *= (1 - 0.0007)
        cash += net
        info['n_orders'] += 1; info['minfee_orders'] += int(amt * FEE < MINFEE and not zero); info['fee_paid'] += fee; info['tax_paid'] += tax; info['slip_paid'] += gross - amt
        if log:
            trades.append(dict(code=G.COLS[j], signal_date=G.IDX[p['sig_t']], entry_date=G.IDX[p['e']], exit_date=G.IDX[t], hold_days=t - p['e'] + 1, reason=reason,
                               entry_open=p['raw_open'], shares=p['shares'], cost=p['cost'], proceeds=net, ret_net=net / p['cost'] - 1,
                               ret_gross=fill_adj / p['entry_adj'] - 1, exit_px_raw=raw, vbr=p['vbr']))

    for t in range(t0, t1 + 1):
        day = G.IDX[t]
        if idle == 'P50' and cash > 0: cash *= (1 + r50[t])
        # (1) 開盤賣出
        for j in [j for j, p in pos.items() if p['sell_open']]:
            if G.TR[t, j] and np.isfinite(G.AO[t, j]) and not G.LOCKDN[t, j]:
                sell(j, t, G.AO[t, j], pos[j]['why'])
        # (2) 開盤買進
        if t in cand:
            unit = 1000 if day < ODD_SWITCH else 1
            for nv, code, j, st, pa in sorted(cand[t]):
                info['n_cand'] += 1
                if j in pos: info['skip_held'] += 1; continue
                if not (G.TR[t, j] and np.isfinite(G.O[t, j])): info['skip_noopen'] += 1; continue
                if G.OPENLIM[t, j] and not buy_openlim: info['skip_openlim'] += 1; continue
                if G.DISP[t, j] and not buy_disp: info['skip_disp'] += 1; continue
                if fixed_amt is None:
                    if len(pos) >= K: info['skip_full'] += 1; continue
                    budget = min(cash, nav_prev / K)
                else:
                    budget = fixed_amt
                capv = cap_share * G.VAL[t, j] if np.isfinite(G.VAL[t, j]) else 0.0
                if budget > capv: budget = capv; info['cap_bound'] += 1
                raw = G.O[t, j]
                px = raw if zero else (raw + 0.5 * float(tick(raw))) * (1 + stress)
                sh = math.floor(budget / (px * (1 + (0 if zero else FEE))) / unit) * unit
                while sh > 0:
                    amt = sh * px; fee = 0.0 if zero else max(MINFEE, amt * FEE)
                    tot = (amt + fee) * (1.0017 if idle == 'P50' else 1.0)
                    if tot <= budget + 1e-9: break
                    sh -= unit
                if sh <= 0:
                    target = nav_prev / K if fixed_amt is None else fixed_amt
                    info['skip_cash' if (fixed_amt is None and cash < 0.5 * target) else 'skip_lot'] += 1; continue
                cash -= tot
                info['n_buy'] += 1; info['n_orders'] += 1; info['minfee_orders'] += int(amt * FEE < MINFEE and not zero); info['fee_paid'] += fee
                info['slip_paid'] += sh * (px - raw); info['turnover_buy'] += amt
                pos[j] = dict(u=sh / G.ADJF[t, j], e=t, entry_adj=G.AO[t, j], peak=G.AO[t, j], pivot_adj=pa, sell_open=False, why='', cost=tot, shares=sh,
                              raw_open=raw, sig_t=st, vbr=-nv)
        # (3) 盤中停損／停利
        if stop is not None or tp is not None:
            for j in list(pos):
                p = pos[j]
                if not G.TR[t, j] or G.LOCKDN[t, j]: continue
                f = G.ADJF[t, j]
                if stop is not None:
                    Sraw = p['entry_adj'] * (1 - stop) / f; tk = float(tick(Sraw)); S = math.floor(Sraw / tk + 1e-9) * tk * f
                    if t > p['e'] and G.AO[t, j] <= S: sell(j, t, G.AO[t, j], 'stop_gap'); continue
                    if G.AL[t, j] <= S: sell(j, t, S, 'stop'); continue
                if tp is not None:
                    Traw = p['entry_adj'] * (1 + tp) / f; tk = float(tick(Traw)); Tg = math.ceil(Traw / tk - 1e-9) * tk * f
                    if t > p['e'] and G.AO[t, j] >= Tg: sell(j, t, G.AO[t, j], 'tp_gap'); continue
                    if G.AH[t, j] >= Tg: sell(j, t, Tg, 'tp'); continue
        # (4) 收盤
        for j in list(pos):
            p = pos[j]
            if p['sell_open']: continue                       # 已判定、等下一根開盤
            if hold_n is not None and t - p['e'] + 1 >= hold_n:
                if G.TR[t, j] and not G.LOCKDN[t, j]: sell(j, t, G.AC[t, j], 'time'); continue
                p['sell_open'] = True; p['why'] = 'time_deferred'; continue
            if not G.TR[t, j]: continue
            ac = G.AC[t, j]
            if X.get('ma50') and G.BELOW50[t, j]: p['sell_open'] = True; p['why'] = 'ma50'
            elif X.get('pivot') and ac < p['pivot_adj']: p['sell_open'] = True; p['why'] = 'pivot'
            elif trail is not None:
                p['peak'] = max(p['peak'], ac)
                if ac <= (1 - trail) * p['peak']: p['sell_open'] = True; p['why'] = 'trail'
        hv = sum(p['u'] * G.ACF[t, j] for j, p in pos.items())
        nav_prev = cash + hv; nav[t - t0] = nav_prev
        info['max_pos'] = max(info['max_pos'], len(pos)); info['sum_pos'] += len(pos); info['days'] += 1
        info['sum_invested_share'] += hv / nav_prev if nav_prev > 0 else 0.0
    info['avg_pos'] = info['sum_pos'] / info['days']; info['avg_invested'] = info['sum_invested_share'] / info['days']
    info['open_positions_at_end'] = len(pos); info['min_cash_note'] = None
    return pd.Series(nav, index=G.IDX[t0:t1 + 1]), pd.DataFrame(trades), info
