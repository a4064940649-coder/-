#!/usr/bin/env python3
"""o_ext_canslim_xs 共用程式庫：資料（官方日 K、季報 EPS PIT、月營收 PIT、加權指數、0050）、
XS 腳本（raw/canslim_entry.xs、raw/canslim_exit.xs）的逐行 Python 重寫、貼文口徑交易清單、可執行組合引擎、統計。
一律以 python3 -I -B 執行。不改任何既有檔案；中間檔放 scratchpad/ext_canslim/。"""
import io, json, hashlib, subprocess, os, math
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
AG = HERE.parent
SCR = Path('/tmp/claude-0/-home-user--/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad')
TMP = SCR / 'ext_canslim'
RAW = Path('/home/user/-/research/tw_stock_selection/raw/jb7k6295r7-bot_tw-stock-data')   # 公開封存（blobless clone，只讀）
PEAD = AG / 'o_pead_margin_surprise_low_attention' / 'data'
E50_CSV = '/home/user/jonatw/finlab-txd/data/raw/etf_0050.csv'
TWII_CSV = '/home/user/jonatw/finlab-txd/data/raw/taiex_twii.csv'
DISP_CSV = AG / 'o_disposal_list_pit_audit_pkl18' / 'data' / 'universe_announce.csv'
END = pd.Timestamp('2026-10-22')        # 0050 與加權指數檔的最後一天
ODD_DATE = pd.Timestamp('2020-10-26')   # 盤中零股上路；之前只能整股
MISSING = -999.0


def sha_bytes(b):
    return hashlib.sha256(b).hexdigest()


def sha_file(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


# ---------------------------------------------------------------- 價格面板
def _git_blobs(sub):
    env = dict(os.environ, GIT_NO_LAZY_FETCH='1')
    ls = subprocess.run(['git', '-C', str(RAW), 'ls-tree', 'HEAD', sub + '/'], capture_output=True, text=True, check=True, env=env).stdout
    items = [(l.split()[2], l.split('\t')[1]) for l in ls.strip().split('\n') if l.endswith('.csv')]
    p = subprocess.Popen(['git', '-C', str(RAW), 'cat-file', '--batch'], stdin=subprocess.PIPE, stdout=subprocess.PIPE, env=env)
    for sha, path in items:
        p.stdin.write((sha + '\n').encode()); p.stdin.flush()
        hdr = p.stdout.readline().decode().split()
        if len(hdr) < 3 or hdr[1] != 'blob':
            continue
        n = int(hdr[2]); data = p.stdout.read(n); p.stdout.read(1)
        yield path, io.BytesIO(data)
    p.stdin.close(); p.wait()


def build_panel(force=False):
    """官方日 K（data/universe/daily/YYYY-MM-DD.csv，2015-01-05 起，上市＋上櫃，含成交股數與逐日市場別）→ TMP/panel_cs.npz。
    只留四碼數字、非 0 開頭的代號（普通股、KY、TDR；不含 ETF、權證、特別股）。"""
    out = TMP / 'panel_cs.npz'
    if out.exists() and not force:
        return out
    TMP.mkdir(parents=True, exist_ok=True)
    frames = []
    for path, fh in _git_blobs('data/universe/daily'):
        try:
            d = pd.read_csv(fh, dtype={'stock_id': str}, usecols=['date', 'stock_id', 'market', 'open', 'high', 'low', 'close', 'volume', 'amount', 'change'])
        except Exception:
            continue
        d = d[d['stock_id'].str.fullmatch(r'[1-9]\d{3}') & d['market'].isin(['twse', 'tpex'])]
        frames.append(d)
    d = pd.concat(frames, ignore_index=True)
    d['date'] = pd.to_datetime(d['date'])
    for c in ['open', 'high', 'low', 'close', 'volume', 'amount', 'change']:
        d[c] = pd.to_numeric(d[c], errors='coerce')
    d = d.drop_duplicates(['date', 'stock_id'], keep='last')
    d['mk'] = np.where(d['market'] == 'twse', 1.0, 2.0)
    piv = lambda c: d.pivot(index='date', columns='stock_id', values=c).sort_index()
    C = piv('close'); IDX = C.index; COLS = C.columns
    arr = {k: piv(c).reindex(index=IDX, columns=COLS).to_numpy() for k, c in
           [('O', 'open'), ('H', 'high'), ('L', 'low'), ('VOL', 'volume'), ('AMT', 'amount'), ('CHG', 'change'), ('MK', 'mk')]}
    np.savez_compressed(out, IDX=IDX.values.astype('datetime64[ns]'), COLS=np.array(COLS, dtype=str), C=C.to_numpy(),
                        MK=np.nan_to_num(arr.pop('MK')).astype(np.int8), **arr)
    return out


def tick(p):
    p = np.asarray(p, float)
    return np.select([p < 10, p < 50, p < 100, p < 500, p < 1000], [0.01, 0.05, 0.1, 0.5, 1.0], 5.0)


class Panel:
    """載入面板並算出訊號與成交需要的全部陣列（列＝交易日，欄＝股票）。"""

    def __init__(self):
        z = np.load(build_panel(), allow_pickle=False)
        idx = pd.DatetimeIndex(z['IDX']); keep = idx <= END
        self.IDX = idx[keep]; self.COLS = np.array(z['COLS']); self.T, self.N = int(keep.sum()), len(self.COLS)
        g = lambda k: z[k][keep]
        self.O, self.H, self.L, self.C = g('O'), g('H'), g('L'), g('C')
        self.VOL, self.AMT, self.CHG, self.MK = g('VOL'), g('AMT'), g('CHG'), g('MK')
        C, O = self.C, self.O
        self.has = np.isfinite(C) & (C > 0)                               # 當日有成交
        self.listed = self.MK > 0                                          # 當日在官方日 K 內（含無成交）
        # 參考價＝收盤−漲跌價差（除權息日為除權息參考價）；含息日報酬＝收盤÷參考價−1
        ref = C - self.CHG
        ref[~(ref > 0)] = np.nan
        R = C / ref - 1
        self.bad_ret = int(np.nansum(np.abs(R) > 0.5))
        R[np.abs(R) > 0.5] = np.nan                                        # 資料錯誤或無漲跌幅限制日；改用收盤對前收
        CF = pd.DataFrame(C).ffill().to_numpy()                            # 無成交日沿用前一收盤（XQ 無成交日如何出 K 棒：需查證）
        prevC = np.vstack([np.full((1, self.N), np.nan), CF[:-1]])
        R = np.where(np.isfinite(R), R, np.where(self.has, C / prevC - 1, 0.0))
        R = np.where(np.isfinite(R), R, 0.0)
        self.R = R; self.CF = CF
        self.REF = np.where(np.isfinite(ref), ref, prevC)
        t = tick(self.REF * 1.10); self.LIMUP = np.floor(self.REF * 1.10 / t + 1e-9) * t
        t = tick(self.REF * 0.90); self.LIMDN = np.ceil(self.REF * 0.90 / t - 1e-9) * t
        self.G = np.cumprod(1 + R, axis=0)                                 # 含息累積指數（還原價訊號變體用）
        self.nbar = np.cumsum(self.listed, axis=0)                         # 已有幾根日 K
        # 無成交日的高點＝沿用收盤、量＝0
        self.HF = np.where(self.has & np.isfinite(self.H), self.H, CF)
        self.VOLF = np.where(self.has & np.isfinite(self.VOL), self.VOL, 0.0)
        tw = pd.read_csv(TWII_CSV, parse_dates=['date']).set_index('date')['close'].sort_index()
        self.IDXC = tw.reindex(self.IDX).ffill().to_numpy()
        self.twii_missing = int(tw.reindex(self.IDX).isna().sum())
        e = pd.read_csv(E50_CSV, parse_dates=['date']).set_index('date')['adj'].sort_index()
        self.E50 = e.reindex(self.IDX.union(e.index)).ffill().reindex(self.IDX).to_numpy()
        self.e50_missing = int(e.reindex(self.IDX).isna().sum())
        self.ci = {c: j for j, c in enumerate(self.COLS)}


def roll_max(a, n, minp=None):
    return pd.DataFrame(a).rolling(n, min_periods=minp or n).max().to_numpy()


def roll_mean(a, n, minp=None):
    return pd.DataFrame(a).rolling(n, min_periods=minp or n).mean().to_numpy()


def shift(a, n):
    out = np.full(a.shape, np.nan)
    if n < a.shape[0]:
        out[n:] = a[:-n]
    return out


# ---------------------------------------------------------------- 基本面（PIT／非 PIT）
def deadline(y, q):
    return {1: pd.Timestamp(y, 5, 15), 2: pd.Timestamp(y, 8, 14), 3: pd.Timestamp(y, 11, 14), 4: pd.Timestamp(y + 1, 3, 31)}[q]


def qend(y, q):
    return pd.Timestamp(y, 3 * q, 1) + pd.offsets.MonthEnd(0)


def _step_panel(P, df, dcol, vcols):
    """df：每列一個（code, 生效日 dcol, 值…），同一 code 依 seq 排序；回傳各值在面板上的階梯陣列
    （第 t 日取『生效日 ≤ 第 t 日』之中 seq 最大的一列）。"""
    out = {v: np.full((P.T, P.N), np.nan) for v in vcols}
    df = df[df['code'].isin(P.ci)].copy()
    df['j'] = df['code'].map(P.ci)
    df['t0'] = np.searchsorted(P.IDX.values, df[dcol].values.astype('datetime64[ns]'), side='left')   # 第一個 ≥ 生效日的交易日
    df = df[df['t0'] < P.T].sort_values(['j', 'seq'])
    for j, g in df.groupby('j'):
        t0 = g['t0'].to_numpy(); seq = g['seq'].to_numpy()
        # 依 seq 由小到大寫入；較新的期別覆蓋較舊的（即使較新的期別較早生效也不會被舊期別蓋回）
        for k in range(len(g)):
            sl = slice(t0[k], P.T)
            for v in vcols:
                out[v][sl, j] = g[v].iloc[k]
    return out


def load_eps(P, pit=True):
    """回傳 dict：EpsQ0, EpsQ4, EpsY0, EpsY3（面板形狀；缺值 NaN）。
    PIT：季別 (y,q) 自公告日 A（o_pead avail_dates；缺者用法定期限）起可得；非 PIT：季末日起可得。
    Q0＝可得的最新一季單季 EPS、Q4＝去年同季；Y0＝可得的最新年度 EPS（第四季累計 EPS，缺則四季單季加總）、Y3＝往前第 3 個年度。"""
    f = pd.read_parquet(PEAD / 'fin_long.parquet', columns=['code', 'y', 'q', 'eps_s', 'eps_ytd'])
    f = f[f['code'].str.fullmatch(r'[1-9]\d{3}')].copy()
    a = pd.read_parquet(PEAD / 'avail_dates.parquet', columns=['code', 'y', 'q', 'A'])
    f = f.merge(a, on=['code', 'y', 'q'], how='left')
    f['dl'] = [deadline(y, q) for y, q in zip(f['y'], f['q'])]
    f['qe'] = [qend(y, q) for y, q in zip(f['y'], f['q'])]
    f['n_noA'] = f['A'].isna()
    f['avail'] = (f['A'].fillna(f['dl']) if pit else f['qe'])
    f['seq'] = f['y'] * 4 + f['q']
    # 單季
    key = f.set_index(['code', 'y', 'q'])['eps_s']
    f['eps_q4'] = key.reindex(pd.MultiIndex.from_arrays([f['code'], f['y'] - 1, f['q']])).to_numpy()
    q = f[f['eps_s'].notna()][['code', 'seq', 'avail', 'eps_s', 'eps_q4']]
    sq = _step_panel(P, q, 'avail', ['eps_s', 'eps_q4'])
    # 年度
    s4 = f.pivot(index=['code', 'y'], columns='q', values='eps_s').sum(axis=1, skipna=False).rename('sum4')
    y4 = f[f['q'] == 4].set_index(['code', 'y'])
    ann = y4[['eps_ytd', 'avail']].join(s4, how='left')
    ann['eps_y'] = ann['eps_ytd'].fillna(ann['sum4'])
    ann = ann[ann['eps_y'].notna()].reset_index()
    ky = ann.set_index(['code', 'y'])['eps_y']
    ann['eps_y3'] = ky.reindex(pd.MultiIndex.from_arrays([ann['code'], ann['y'] - 3])).to_numpy()
    ann['seq'] = ann['y']
    sy = _step_panel(P, ann[['code', 'seq', 'avail', 'eps_y', 'eps_y3']], 'avail', ['eps_y', 'eps_y3'])
    info = dict(n_quarter_rows=int(len(q)), n_quarter_no_A_use_deadline=int(f.loc[f['eps_s'].notna(), 'n_noA'].sum()),
                n_annual_rows=int(len(ann)), annual_from_sum4=int(ann['eps_ytd'].isna().sum()) if 'eps_ytd' in ann else None)
    return dict(EpsQ0=sq['eps_s'], EpsQ4=sq['eps_q4'], EpsY0=sy['eps_y'], EpsY3=sy['eps_y3'], info=info)


def rev_avail_days(P, months, pit=True):
    """月營收 m 的可用日（面板列號）：PIT＝次月 10 日（非交易日順延到下一個交易日）之後的第一個交易日；
    非 PIT＝當月最後一個日曆日之後（含當日）的第一個交易日。"""
    out = {}
    for m in months:
        if pit:
            d10 = (m + 1).to_timestamp() + pd.Timedelta(days=9)
            k = int(np.searchsorted(P.IDX.values, np.datetime64(d10), side='left'))   # 期限日（順延後）
            out[m] = k + 1
        else:
            out[m] = int(np.searchsorted(P.IDX.values, np.datetime64(m.to_timestamp(how='end').normalize()), side='left'))
    return out


def load_rev3(P, pit=True):
    """RevYoY3＝訊號日當時已公告的最近 3 個月『月營收年增率』平均（缺任一個月＝-999，對應 XS 的 RevOK=false）。"""
    r = pd.read_parquet(PEAD / 'rev_long.parquet', columns=['code', 'ym', 'yoy_pct'])
    Y = r.pivot(index='ym', columns='code', values='yoy_pct').sort_index()
    Y.index = pd.PeriodIndex(Y.index, freq='M')
    Y = Y.reindex(pd.period_range(Y.index.min(), Y.index.max(), freq='M'))
    Y = Y.replace([np.inf, -np.inf], np.nan)
    Y3 = Y.rolling(3, min_periods=3).mean().reindex(columns=P.COLS)
    av = rev_avail_days(P, list(Y3.index), pit)
    out = np.full((P.T, P.N), np.nan)
    for m in Y3.index:                       # 月份由舊到新，後者覆蓋前者
        k = av[m]
        if k < P.T:
            out[k:] = Y3.loc[m].to_numpy()[None, :]
    return out


# ---------------------------------------------------------------- XS 進場腳本
DEF = dict(MinEpsQYoY=18.0, MinEpsCAGR=15.0, MinRevYoY3=20.0, MaxFromHigh=10.0, MinAvgQty=500.0, RSDays=120, UseMarket=1, MarketMA=60,
           TrailPct=6.0, TrailDays=20, FastMA=8, SlowMA=20)


def entry_fields(P, eps, rev3, prm=DEF, adj=False):
    """逐行對應 canslim_entry.xs。回傳各條件的布林陣列與輸出欄位。adj=True 時價格改用含息還原價（變體）。"""
    if adj:
        fac = P.G / P.CF
        C = P.G; H = P.HF * fac
    else:
        C = P.CF; H = P.HF
    with np.errstate(divide='ignore', invalid='ignore'):
        q0, q4 = eps['EpsQ0'], eps['EpsQ4']
        EpsQYoY = np.where(np.isfinite(q0) & np.isfinite(q4) & (q4 > 0), (q0 / q4 - 1) * 100, MISSING)
        y0, y3 = eps['EpsY0'], eps['EpsY3']
        EpsCAGR = np.where((y0 > 0) & (y3 > 0), (np.power(np.where((y0 > 0) & (y3 > 0), y0 / y3, 1.0), 1 / 3) - 1) * 100, MISSING)
        RevYoY3 = np.where(np.isfinite(rev3), rev3, MISSING)
        High240 = roll_max(H, 240)
        FromHigh = np.where((High240 > 0) & (P.nbar >= 240), (C / High240 - 1) * 100, MISSING)
        AvgQty = roll_mean(P.VOLF / 1000.0, 20)                           # 張
        n = int(prm['RSDays'])
        Cn = shift(C, n)
        StockRet = np.where(Cn > 0, (C / Cn - 1) * 100, MISSING)
        ic = P.IDXC; ico = np.concatenate([np.full(n, np.nan), ic[:-n]])
        IndexRet = np.where(ico > 0, (ic / ico - 1) * 100, MISSING)
        IdxMA = pd.Series(ic).rolling(int(prm['MarketMA'])).mean().to_numpy()
    MarketOK = np.ones(P.T, bool)
    if prm['UseMarket'] == 1:
        MarketOK = ~(~(IdxMA > 0) | (ic < IdxMA))
    cond = dict(
        C=EpsQYoY >= prm['MinEpsQYoY'],
        A_cagr=EpsCAGR >= prm['MinEpsCAGR'],
        A_rev=RevYoY3 >= prm['MinRevYoY3'],
        N=FromHigh >= -prm['MaxFromHigh'],
        S=np.nan_to_num(AvgQty, nan=-1) >= prm['MinAvgQty'],
        L=(StockRet > IndexRet[:, None]) & (StockRet > 0),
        M=np.repeat(MarketOK[:, None], P.N, axis=1),
        base=(C > 0) & P.listed,
    )
    fld = dict(EpsQYoY=EpsQYoY, EpsCAGR=EpsCAGR, RevYoY3=RevYoY3, FromHigh=FromHigh, RS=StockRet - IndexRet[:, None], AvgQty=AvgQty)
    return cond, fld


def entry_signal(cond, drop=()):
    s = cond['base'].copy()
    for k in ('C', 'A_cagr', 'A_rev', 'N', 'S', 'L', 'M'):
        if k not in drop:
            s &= cond[k]
    return s


def exit_signal(P, prm=DEF, adj=False):
    """逐行對應 canslim_exit.xs。回傳 (trail, cross)：
    trail：Close <= Highest(Close, TrailDays) * (1 - TrailPct/100)
    cross：Average(Close,FastMA) crosses under Average(Close,SlowMA)＝今日快線 < 慢線 且 昨日快線 >= 慢線（XS crosses under 的定義）。"""
    C = P.G if adj else P.CF
    hc = roll_max(C, int(prm['TrailDays']))
    trail = C <= hc * (1 - prm['TrailPct'] / 100.0)
    f = roll_mean(C, int(prm['FastMA'])); s = roll_mean(C, int(prm['SlowMA']))
    fp = shift(f, 1); sp = shift(s, 1)
    cross = (f < s) & (fp >= sp)
    return np.nan_to_num(trail, nan=0).astype(bool), cross


# ---------------------------------------------------------------- 貼文口徑：逐筆交易（不限檔數、每筆等額、下期開盤進出）
def post_trades(P, sig, trail, cross, t_start, univ):
    """sig[t,j]：第 t 日盤後入選；未持有才買，t+1 開盤買（無開盤價就放棄該次訊號）。
    持有期間（含進場當日收盤起）出場訊號成立 → 次日開盤賣（無開盤價則順延到下一個有開盤價的日子）。
    期末仍持有者以最後一日收盤結算（open_end=1）。回傳 DataFrame。"""
    T, N = P.T, P.N
    O, C, R, REF = P.O, P.CF, P.R, P.REF
    rows = []
    exs = trail | cross
    for j in np.where(univ.any(axis=0) & sig[t_start:].any(axis=0))[0]:
        s = sig[:, j] & univ[:, j]
        t = t_start
        while t < T - 1:
            if not s[t]:
                nz = np.flatnonzero(s[t + 1:T - 1])
                if len(nz) == 0:
                    break
                t = t + 1 + nz[0]
            e = t + 1
            if not (np.isfinite(O[e, j]) and O[e, j] > 0):
                t += 1
                continue
            # 找出場訊號日 k ≥ e
            nz = np.flatnonzero(exs[e:T, j])
            if len(nz) == 0:
                x = None
            else:
                k = e + nz[0]; x = k + 1
                while x < T and not (np.isfinite(O[x, j]) and O[x, j] > 0):
                    x += 1
                if x >= T:
                    x = None
            if x is None:
                # 期末結算：最後一個收盤
                gross_tr = (C[e, j] / O[e, j]) * np.prod(1 + R[e + 1:T, j])
                rows.append((j, t, e, T - 1, -1, gross_tr - 1, C[T - 1, j] / O[e, j] - 1, 0, 0, 1))
                break
            gross_tr = (C[e, j] / O[e, j]) * np.prod(1 + R[e + 1:x, j]) * (O[x, j] / REF[x, j])
            rows.append((j, t, e, x, k, gross_tr - 1, O[x, j] / O[e, j] - 1, int(trail[k, j]), int(cross[k, j]), 0))
            t = x          # 賣出當日收盤即可再度入選
    d = pd.DataFrame(rows, columns=['j', 'sig_t', 'ent_t', 'exit_t', 'xsig_t', 'ret_tr', 'ret_px', 'x_trail', 'x_cross', 'open_end'])
    d['code'] = P.COLS[d['j'].to_numpy()] if len(d) else []
    for k in ('sig_t', 'ent_t', 'exit_t'):
        d[k.replace('_t', '_date')] = P.IDX[d[k].to_numpy()] if len(d) else []
    return d


def post_curve(P, tr, t_start, cost=None):
    """貼文口徑權益曲線。每筆交易投入 1 單位（不複利、不限檔數）。
    回傳 dict：n_open（每日持有筆數）、pnl（每日損益，單位＝每筆金額）、twr（每日『持有中各筆等權平均報酬』的連乘）。
    cost=(每筆買進費率陣列, 每筆賣出費率陣列)：在進場日與出場日扣。"""
    T = P.T
    pnl = np.zeros(T); nopen = np.zeros(T); sret = np.zeros(T)
    O, C, R, REF = P.O, P.CF, P.R, P.REF
    n = len(tr)
    cba, csa = cost if cost else (np.zeros(n), np.zeros(n))
    for j, e, x, oe, cb, cs in zip(tr['j'].to_numpy(), tr['ent_t'].to_numpy(), tr['exit_t'].to_numpy(), tr['open_end'].to_numpy(), cba, csa):
        # 每日報酬序列：進場日（開盤→收盤）、中間日（含息日報酬）、出場日（前收→開盤）
        if oe:
            r = np.concatenate([[C[e, j] / O[e, j] / (1 + cb) - 1], R[e + 1:x + 1, j]]); days = np.arange(e, x + 1)
        elif x == e:
            r = np.array([0.0]); days = np.array([e])
        else:
            r = np.concatenate([[C[e, j] / O[e, j] / (1 + cb) - 1], R[e + 1:x, j], [O[x, j] / REF[x, j] * (1 - cs) - 1]]); days = np.arange(e, x + 1)
        v = np.cumprod(1 + r); vprev = np.concatenate([[1.0], v[:-1]])
        pnl[days] += v - vprev                 # 以進場金額為 1 的每日損益
        sret[days] += r; nopen[days] += 1
    with np.errstate(divide='ignore', invalid='ignore'):
        dr = np.where(nopen > 0, sret / np.maximum(nopen, 1), 0.0)
    twr = np.cumprod(1 + dr[t_start:])
    return dict(n_open=nopen, pnl=pnl, twr=twr, dr=dr)


# ---------------------------------------------------------------- 可執行組合引擎
FEE = 0.001425 * 0.28
TAX = 0.003
MINFEE = 20.0


def _fee(shares, px, rate, lot_only):
    """整股與零股各算一筆手續費（各自最低 20 元）。"""
    if rate == 0:
        return 0.0
    rnd = (shares // 1000) * 1000; odd = shares - rnd
    f = 0.0
    if rnd > 0:
        f += max(MINFEE, rnd * px * rate)
    if odd > 0:
        f += max(MINFEE, odd * px * rate)
    return f


def simulate(P, sig, rank, trail, cross, univ, t_start, capital=1e6, slots=10, cost='normal', idle='cash',
             cap_pct=0.05, disp=None, t_end=None):
    """日頻可執行模擬。訊號日 t（盤後）→ t+1 開盤成交。
    cost：'normal'（手續費 0.1425%×0.28、最低 20 元；賣出證交稅 0.3%；每邊滑價半個跳動點）、
          'stress'（每邊再加 0.3%）、'zero'（無成本、無滑價；回測上緣）。
    idle：'cash'（空位放現金，利率 0）或 'e50'（閒置資金放 0050 含息，換手成本近似）。
    回傳 nav（pd.Series，自 t_start 日收盤＝capital 起）、成交紀錄 DataFrame、統計 dict。"""
    T = t_end if t_end is not None else P.T
    O, C, R, REF, AMT = P.O, P.CF, P.R, P.REF, P.AMT
    fee = 0.0 if cost == 'zero' else FEE
    tax = 0.0 if cost == 'zero' else TAX
    extra = 0.003 if cost == 'stress' else 0.0
    exs = trail | cross
    cash = float(capital); pos = {}                      # j -> dict(v=市值, sh=股數, e=進場日, sig=訊號日, cost=買進總成本, flag=待賣, why)
    nav = np.full(T, np.nan); nav[t_start] = capital
    log = []; st = dict(n_sig=0, n_buy=0, blk_limup=0, blk_noopen=0, blk_lot=0, blk_cap=0, blk_nocash=0, noslot=0,
                        sell_defer_limdn=0, sell_defer_noopen=0, sell_partial_cap=0, n_delist_liq=0, buy_val=0.0, sell_val=0.0,
                        fee=0.0, tax=0.0, slip=0.0, n_disp_buy=0, minfee_orders=0, orders=0, e50_cost=0.0)
    r50 = np.zeros(P.T); r50[1:] = P.E50[1:] / P.E50[:-1] - 1
    odd_t = int(np.searchsorted(P.IDX.values, np.datetime64(ODD_DATE), side='left'))
    for t in range(t_start + 1, T):
        if idle == 'e50':
            cash *= (1 + r50[t])                           # 閒置資金隔夜放 0050（近似：整日報酬）
        cash0 = cash; sold_today = set()
        # ---- 開盤：賣
        for j in list(pos):
            p = pos[j]
            if P.MK[t, j] == 0 and not P.MK[t:, j].any():   # 已自官方日 K 消失（下市／合併）：以最後收盤價結清
                cash += p['v']; st['n_delist_liq'] += 1
                log.append((j, p['sig'], p['e'], t, p['cost'], p['proc'] + p['v'], 'delist', p['sh'], p.get('disp', 0)))
                del pos[j]; sold_today.add(j); continue
            if not p['flag']:
                continue
            o = O[t, j]
            if not (np.isfinite(o) and o > 0):
                st['sell_defer_noopen'] += 1; continue
            if o <= P.LIMDN[t, j] + 1e-9:
                st['sell_defer_limdn'] += 1; continue
            vopen = p['v'] * o / REF[t, j]                  # 開盤時市值（含除權息調整）
            frac = 1.0
            if cap_pct and np.isfinite(AMT[t, j]) and vopen > cap_pct * AMT[t, j]:
                frac = cap_pct * AMT[t, j] / vopen; st['sell_partial_cap'] += 1
            sl = (0.0 if cost == 'zero' else 0.5 * float(tick(o)) / o) + extra
            gross = vopen * frac * (1 - sl)
            sh = int(round(p['sh'] * frac))
            f = _fee(max(sh, 1), gross / max(sh, 1), fee, False); tx = gross * tax
            st['orders'] += 1; st['minfee_orders'] += int(fee > 0 and f <= 2 * MINFEE and gross * fee < MINFEE)
            cash += gross - f - tx; sold_today.add(j)
            st['sell_val'] += vopen * frac; st['fee'] += f; st['tax'] += tx; st['slip'] += vopen * frac * sl
            if frac >= 1.0:
                log.append((j, p['sig'], p['e'], t, p['cost'], p['proc'] + gross - f - tx, p['why'], p['sh'], p.get('disp', 0)))
                del pos[j]
            else:
                p['proc'] += gross - f - tx; p['v'] *= (1 - frac); p['sh'] = max(p['sh'] - sh, 0); p['part'] = True
        # ---- 開盤：買（訊號日 t-1 盤後入選、當時未持有）
        cand = np.flatnonzero(sig[t - 1] & univ[t - 1])
        if len(cand):
            held_prev = set(pos) | sold_today             # t-1 收盤時持有的（含今日開盤剛賣出的）
            cand = [j for j in cand if j not in held_prev]
            st['n_sig'] += len(cand)
            free = slots - len(pos)
            if free > 0 and cand:
                rk = rank[t - 1, cand]
                order = [cand[i] for i in np.lexsort((np.arange(len(cand)), -np.nan_to_num(rk, nan=-1e18)))]
                # 部位估值：持股以開盤價估
                eq = cash + sum(p['v'] * (O[t, k] / REF[t, k] if np.isfinite(O[t, k]) and O[t, k] > 0 else 1.0) for k, p in pos.items())
                target = eq / slots
                for j in order:
                    if free <= 0:
                        st['noslot'] += 1; continue
                    o = O[t, j]
                    if not (np.isfinite(o) and o > 0):
                        st['blk_noopen'] += 1; continue
                    if o >= P.LIMUP[t, j] - 1e-9:
                        st['blk_limup'] += 1; continue
                    sl = (0.0 if cost == 'zero' else 0.5 * float(tick(o)) / o) + extra
                    px = o * (1 + sl)
                    budget = min(target, cash)
                    sh = int(budget / (px * (1 + fee)))
                    if t < odd_t:
                        sh = (sh // 1000) * 1000
                    if sh <= 0:
                        st['blk_lot' if budget >= 0.5 * target else 'blk_nocash'] += 1; continue
                    if cap_pct and np.isfinite(AMT[t, j]):
                        mx = int(cap_pct * AMT[t, j] / o)
                        if t < odd_t:
                            mx = (mx // 1000) * 1000
                        if sh > mx:
                            sh = mx; st['blk_cap'] += 1
                            if sh <= 0:
                                continue
                    f = _fee(sh, px, fee, t < odd_t)
                    while sh > 0 and sh * px + f > cash + 1e-6:
                        sh -= 1000 if t < odd_t else 1
                        f = _fee(sh, px, fee, t < odd_t) if sh > 0 else 0.0
                    if sh <= 0:
                        st['blk_nocash'] += 1; continue
                    cst = sh * px + f
                    cash -= cst
                    st['orders'] += 1; st['minfee_orders'] += int(fee > 0 and sh * px * fee < MINFEE)
                    st['buy_val'] += sh * o; st['fee'] += f; st['slip'] += sh * o * sl; st['n_buy'] += 1
                    dz = int(disp[t, j]) if disp is not None else 0
                    st['n_disp_buy'] += dz
                    pos[j] = dict(v=sh * o, sh=sh, e=t, sig=t - 1, cost=cst, flag=False, why='', proc=0.0, new=True, disp=dz)
                    free -= 1
        # ---- 收盤：估值
        tot = 0.0
        for j, p in pos.items():
            if p.get('new'):
                p['v'] = p['v'] * (C[t, j] / O[t, j]); p['new'] = False
            else:
                p['v'] *= (1 + R[t, j])
            tot += p['v']
        if idle == 'e50':
            # 閒置資金的增減視為 0050 買賣：買 0.0399%＋滑價 0.03%；賣 0.0399%＋證交稅 0.1%＋滑價 0.03%
            d = cash - cash0
            c50 = (0.000399 + 0.0003) * max(d, 0) + (0.000399 + 0.001 + 0.0003) * max(-d, 0)
            if cost != 'zero':
                cash -= c50; st['e50_cost'] += c50
        nav[t] = cash + tot
        # ---- 盤後：出場訊號（含進場當日）
        for j, p in pos.items():
            if not p['flag'] and exs[t, j]:
                p['flag'] = True
                p['why'] = 'both' if (trail[t, j] and cross[t, j]) else ('trail' if trail[t, j] else 'cross')
        st.setdefault('npos_sum', 0); st['npos_sum'] += len(pos)
        st.setdefault('cash_frac_sum', 0.0); st['cash_frac_sum'] += cash / nav[t]
    for j, p in pos.items():   # 期末未平倉：以最後收盤市值列入
        log.append((j, p['sig'], p['e'], T - 1, p['cost'], p['proc'] + p['v'], 'open_end', p['sh'], p.get('disp', 0)))
    navs = pd.Series(nav[t_start:T], index=P.IDX[t_start:T])
    lg = pd.DataFrame(log, columns=['j', 'sig_t', 'ent_t', 'exit_t', 'cost', 'proceeds', 'why', 'shares', 'disp'])
    if len(lg):
        lg['code'] = P.COLS[lg['j'].to_numpy()]; lg['ret'] = lg['proceeds'] / lg['cost'] - 1
        lg['ent_date'] = P.IDX[lg['ent_t'].to_numpy()]; lg['exit_date'] = P.IDX[lg['exit_t'].to_numpy()]
    nd = T - 1 - t_start
    st['avg_npos'] = st.pop('npos_sum', 0) / nd; st['avg_cash_frac'] = st.pop('cash_frac_sum', 0.0) / nd
    return navs, lg, st


# ---------------------------------------------------------------- 統計
def tstat(x):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    if len(x) < 3 or x.std(ddof=1) == 0:
        return float('nan')
    return float(x.mean() / (x.std(ddof=1) / np.sqrt(len(x))))


def cagr(nav, a=None, b=None):
    s = nav.dropna()
    if a is not None:
        s0 = s[s.index < pd.Timestamp(a)]
        s = pd.concat([s0.iloc[-1:], s[s.index >= pd.Timestamp(a)]]) if len(s0) else s[s.index >= pd.Timestamp(a)]
    if b is not None:
        s = s[s.index <= pd.Timestamp(b)]
    if len(s) < 2:
        return None
    yrs = (s.index[-1] - s.index[0]).days / 365.25
    return dict(cagr=float((s.iloc[-1] / s.iloc[0]) ** (1 / yrs) - 1), total=float(s.iloc[-1] / s.iloc[0] - 1), years=yrs,
                start=str(s.index[0].date()), end=str(s.index[-1].date()), maxdd=float((s / s.cummax() - 1).min()))


def monthly(nav):
    """日曆月報酬：每月最後一個交易日收盤對上月最後一個交易日收盤；第一個月自起始日起算。"""
    s = nav.dropna()
    m = s.groupby([s.index.year, s.index.month]).last()
    first = s.iloc[0]
    v = np.concatenate([[first], m.to_numpy()])
    r = v[1:] / v[:-1] - 1
    idx = pd.PeriodIndex([pd.Period(year=y, month=mo, freq='M') for y, mo in m.index], freq='M')
    return pd.Series(r, index=idx)


def window_stats(nav, bench, a=None, b=None):
    """nav、bench：同一日曆的日頻淨值（pd.Series）。回傳年化、0050 年化、超額（pp）、月超額 t、最大回撤等。"""
    s = nav.dropna(); e = bench.reindex(s.index)
    c1 = cagr(s, a, b); c2 = cagr(e, a, b)
    if c1 is None:
        return None
    ms = monthly(s); mb = monthly(e)
    if a is not None:
        ms = ms[ms.index >= pd.Period(pd.Timestamp(a), 'M')]
    if b is not None:
        ms = ms[ms.index <= pd.Period(pd.Timestamp(b), 'M')]
    ex = ms - mb.reindex(ms.index)
    return dict(cagr=c1['cagr'], cagr_0050=c2['cagr'], excess_pp=(c1['cagr'] - c2['cagr']) * 100, total=c1['total'], total_0050=c2['total'],
                maxdd=c1['maxdd'], maxdd_0050=c2['maxdd'], years=c1['years'], start=c1['start'], end=c1['end'],
                n_months=int(len(ex)), mean_ex_m=float(ex.mean()), sd_ex_m=float(ex.std(ddof=1)), t_ex_m=tstat(ex),
                vol=float(s.pct_change().dropna().std() * math.sqrt(252)))


def drop_best3_t(nav, bench, a=None, b=None):
    """拿掉『年度超額（策略年報酬−0050 年報酬）最高的 3 個日曆年』之後的月超額 t。"""
    s = nav.dropna(); e = bench.reindex(s.index)
    ms = monthly(s); mb = monthly(e)
    if a is not None:
        ms = ms[ms.index >= pd.Period(pd.Timestamp(a), 'M')]
    ex = (ms - mb.reindex(ms.index))
    ys = (1 + ms).groupby(ms.index.year).prod() - 1; yb = (1 + mb.reindex(ms.index)).groupby(ms.index.year).prod() - 1
    yex = (ys - yb).sort_values(ascending=False)
    best = list(yex.index[:3])
    rest = ex[~np.isin(ex.index.year, best)]
    return dict(best3=[int(x) for x in best], t=tstat(rest), n=int(len(rest)),
                yearly={int(y): dict(strat=float(ys[y]), e50=float(yb[y]), ex_pp=float((ys[y] - yb[y]) * 100)) for y in ys.index})


def disposal_mask(P):
    """上市官方處置事件表 → 面板布林（處置起日 ≤ 當日 ≤ 迄日）。上櫃無資料。"""
    d = pd.read_csv(DISP_CSV, dtype={'code': str}, parse_dates=['announce', 'start', 'end'])
    m = np.zeros((P.T, P.N), bool)
    for c, s, e in zip(d['code'], d['start'], d['end']):
        j = P.ci.get(c)
        if j is None or pd.isna(s) or pd.isna(e):
            continue
        a = int(np.searchsorted(P.IDX.values, np.datetime64(s), side='left')); b = int(np.searchsorted(P.IDX.values, np.datetime64(e), side='right'))
        m[a:b, j] = True
    return m
