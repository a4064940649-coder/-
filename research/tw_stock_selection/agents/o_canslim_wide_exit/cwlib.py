#!/usr/bin/env python3
"""o_canslim_wide_exit 共用程式庫：CAN SLIM 進場（沿用 o_ext_canslim_xs/cslib.py，唯讀 import）＋本題自寫的
出場規則、逐部位出場的可執行引擎、空位放 0050（開盤價精確換手）、路徑擾動、安慰劑、統計（NW、SPA、DSR、β）。
一律 python3 -I -B。中間檔放 scratchpad/cs_wide/。不改任何既有檔案。"""
import sys, os, io, csv, re, math, json, hashlib, subprocess
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
AG = HERE.parent
sys.path.insert(0, str(AG / 'o_ext_canslim_xs'))
import cslib as cs                                    # 唯讀沿用：面板、季報／月營收 PIT、進場訊號
SCR = Path('/tmp/claude-0/-home-user--/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad')
TMP = SCR / 'cs_wide'
cs.TMP = TMP                                          # 面板快取改放本題暫存夾（不寫進別人的資料夾）
MAINNAV = AG / 'integrate_o20' / 'data' / 'v7x_navs_jmean_100w.csv.gz'
CLOSE_OLD = Path('/home/user/-/research/tw_stock_selection/data_panel/panel/close_raw.parquet')
FEE, TAX, MINFEE = cs.FEE, cs.TAX, cs.MINFEE
ETF_TAX = 0.001


def sha_bytes(b):
    return hashlib.sha256(b).hexdigest()


def sha_file(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


# ---------------------------------------------------------------- 省記憶體：分塊算訊號 → 只留曾進池的股票
def load_chunk(c0, c1):
    """與 cs.Panel.__init__ 逐行相同，但只載入第 c0～c1 欄（各欄運算彼此獨立，結果與整個面板逐格相同）。"""
    P = object.__new__(cs.Panel)
    z = np.load(cs.build_panel(), allow_pickle=False)
    idx = pd.DatetimeIndex(z['IDX']); keep = idx <= cs.END
    P.IDX = idx[keep]; P.COLS = np.array(z['COLS'])[c0:c1]; P.T, P.N = int(keep.sum()), len(P.COLS)
    g = lambda k: z[k][keep][:, c0:c1]
    P.O, P.H, P.L, P.C = g('O'), g('H'), g('L'), g('C')
    P.VOL, P.AMT, P.CHG, P.MK = g('VOL'), g('AMT'), g('CHG'), g('MK')
    C = P.C
    P.has = np.isfinite(C) & (C > 0); P.listed = P.MK > 0
    ref = C - P.CHG; ref[~(ref > 0)] = np.nan
    R = C / ref - 1
    P.bad_ret = int(np.nansum(np.abs(R) > 0.5)); R[np.abs(R) > 0.5] = np.nan
    CF = pd.DataFrame(C).ffill().to_numpy()
    prevC = np.vstack([np.full((1, P.N), np.nan), CF[:-1]])
    R = np.where(np.isfinite(R), R, np.where(P.has, C / prevC - 1, 0.0)); R = np.where(np.isfinite(R), R, 0.0)
    P.R = R; P.CF = CF
    P.REF = np.where(np.isfinite(ref), ref, prevC)
    t = cs.tick(P.REF * 1.10); P.LIMUP = np.floor(P.REF * 1.10 / t + 1e-9) * t
    t = cs.tick(P.REF * 0.90); P.LIMDN = np.ceil(P.REF * 0.90 / t - 1e-9) * t
    P.G = np.cumprod(1 + R, axis=0); P.nbar = np.cumsum(P.listed, axis=0)
    P.HF = np.where(P.has & np.isfinite(P.H), P.H, CF); P.VOLF = np.where(P.has & np.isfinite(P.VOL), P.VOL, 0.0)
    tw = pd.read_csv(cs.TWII_CSV, parse_dates=['date']).set_index('date')['close'].sort_index()
    P.IDXC = tw.reindex(P.IDX).ffill().to_numpy()
    e = pd.read_csv(cs.E50_CSV, parse_dates=['date']).set_index('date')['adj'].sort_index()
    P.E50 = e.reindex(P.IDX.union(e.index)).ffill().reindex(P.IDX).to_numpy()
    P.ci = {c: j for j, c in enumerate(P.COLS)}
    return P


FLOATS = ('O', 'CF', 'R', 'REF', 'AMT', 'LIMUP', 'LIMDN', 'RS')
BOOLS = ('sig', 'sig_ext', 'pool', 'trail6', 'trail12', 'trail15', 'trail20', 'cross', 'ma50')


def build_compact(force=False, chunk=300):
    """分塊（每塊 chunk 檔）算出進場訊號、池子、出場布林，再只保留『自延伸段起點以後曾進池（N、S、L、M、base 全過且在榜）』的股票欄。
    池子包含全部真實訊號，所以對引擎而言與整個面板等價（欄序仍是代號由小到大）。"""
    out = TMP / 'compact.npz'
    if out.exists() and not force:
        return out
    TMP.mkdir(parents=True, exist_ok=True)
    z = np.load(cs.build_panel(), allow_pickle=False); ncol = len(z['COLS']); del z
    extra, val = implied_annual_eps()
    parts = {k: [] for k in FLOATS + BOOLS + ('MK',)}; cols = []; info = dict(ext_validation=val, n_extra=len(extra), bad_ret=0, n_codes=ncol, counts={})
    nbmax = None; tot = dict(sig_all=0, sig_twse=0, pool_all=0, pool_twse=0, y3_filled=0)
    t0 = None
    for c0 in range(0, ncol, chunk):
        P = load_chunk(c0, min(c0 + chunk, ncol))
        if t0 is None:
            t0 = int(P.IDX.get_loc(pd.Timestamp('2017-03-31')))
        eps = cs.load_eps(P, True); rev = cs.load_rev3(P, True)
        cond, fld = cs.entry_fields(P, eps, rev); sig = cs.entry_signal(cond)
        epx = load_eps_ext(P, extra); c2, _ = cs.entry_fields(P, epx, rev); sig_ext = cs.entry_signal(c2)
        tot['y3_filled'] += epx['info']['n_y3_filled_from_per']
        pool = cs.entry_signal(cond, drop=('C', 'A_cagr', 'A_rev'))
        X = exit_arrays(P)
        tr6, cr6 = cs.exit_signal(P); assert (X['trail6'] == tr6).all() and (X['cross'] == cr6).all()
        nb = P.nbar.max(axis=1); nbmax = nb if nbmax is None else np.maximum(nbmax, nb)
        info['bad_ret'] += P.bad_ret
        tot['sig_all'] += int(sig[t0:].sum()); tot['sig_twse'] += int((sig & (P.MK == 1))[t0:].sum())
        tot['pool_all'] += int(pool[t0:].sum()); tot['pool_twse'] += int((pool & (P.MK == 1))[t0:].sum())
        keep = pool.any(axis=0) | sig.any(axis=0) | sig_ext.any(axis=0)
        src = dict(O=P.O, CF=P.CF, R=P.R, REF=P.REF, AMT=P.AMT, LIMUP=P.LIMUP, LIMDN=P.LIMDN, RS=fld['RS'], MK=P.MK, sig=sig, sig_ext=sig_ext, pool=pool, **X)
        for k in parts:
            parts[k].append(np.ascontiguousarray(src[k][:, keep]))
        cols.extend(P.COLS[keep].tolist())
        IDX, IDXC, E50 = P.IDX, P.IDXC, P.E50
        del P, eps, rev, cond, fld, epx, c2, sig, sig_ext, pool, X, src
    ts_ext = int(np.flatnonzero(nbmax >= 240)[0])
    info['counts'] = tot; info['ts_ext'] = ts_ext; info['n_kept'] = len(cols)
    np.savez_compressed(out, IDX=IDX.values.astype('datetime64[ns]'), COLS=np.array(cols, dtype=str), IDXC=IDXC, E50=E50,
                        info=np.array(json.dumps(info)), **{k: np.concatenate(v, axis=1) for k, v in parts.items()})
    return out


class Compact:
    """精簡面板：屬性與 cs.Panel 中引擎用到的相同（O、CF、R、REF、AMT、LIMUP、LIMDN、MK、IDX、COLS、E50、IDXC、ci），另有訊號與出場布林。"""

    def __init__(self):
        z = np.load(build_compact(), allow_pickle=False)
        self.IDX = pd.DatetimeIndex(z['IDX']); self.COLS = np.array(z['COLS']); self.T, self.N = len(self.IDX), len(self.COLS)
        for k in FLOATS + BOOLS + ('MK',):
            setattr(self, k, z[k])
        self.IDXC = z['IDXC']; self.E50 = z['E50']; self.info = json.loads(str(z['info']))
        self.ci = {c: j for j, c in enumerate(self.COLS)}
        self.X = {k: getattr(self, k) for k in ('trail6', 'trail12', 'trail15', 'trail20', 'cross', 'ma50')}
        self.ts_ext = int(self.info['ts_ext'])


# ---------------------------------------------------------------- 0050 開盤／收盤（官方日 K）
def load_e50(P):
    """回傳 dict(ov, intr, slip)：ov[t]＝0050 前收→t 日開盤的含息報酬、intr[t]＝t 日開盤→收盤、slip[t]＝半個跳動點÷開盤價。
    (1+ov)(1+intr)＝etf_0050.csv adj 的日報酬。官方日 K 沒有 0050 開盤價的日子：intr＝0、ov＝整日報酬。"""
    cache = TMP / 'e0050_ohlc.csv'
    if not cache.exists():
        TMP.mkdir(parents=True, exist_ok=True)
        env = dict(os.environ, GIT_NO_LAZY_FETCH='1')
        ls = subprocess.run(['git', '-C', str(cs.RAW), 'ls-tree', 'HEAD', 'data/universe/daily/'], capture_output=True, text=True, check=True, env=env).stdout
        files = []
        for line in ls.splitlines():
            meta, path = line.split('\t'); m = re.search(r'(\d{4}-\d{2}-\d{2})\.csv$', path)
            if m:
                files.append((m.group(1), meta.split()[2]))
        files.sort()
        p = subprocess.Popen(['git', '-C', str(cs.RAW), 'cat-file', '--batch'], stdin=subprocess.PIPE, stdout=subprocess.PIPE, env=env)
        rows = []
        for d, h in files:
            p.stdin.write((h + '\n').encode()); p.stdin.flush()
            hdr = p.stdout.readline().split(); data = p.stdout.read(int(hdr[2])); p.stdout.read(1)
            txt = data.decode('utf-8'); k = txt.find('\n%s_0050,' % d)
            if k < 0:
                rows.append((d, np.nan, np.nan)); continue
            line = txt[k + 1: txt.find('\n', k + 1)]; head = txt[:txt.find('\n')].split(',')
            rec = dict(zip(head, next(csv.reader([line]))))
            f = lambda x: float(x) if x not in ('', '--') else np.nan
            rows.append((d, f(rec['open']), f(rec['close'])))
        p.stdin.close(); p.wait()
        pd.DataFrame(rows, columns=['date', 'open', 'close']).to_csv(cache, index=False)
    e = pd.read_csv(cache, parse_dates=['date']).set_index('date').reindex(P.IDX)
    r_tot = np.zeros(P.T); r_tot[1:] = P.E50[1:] / P.E50[:-1] - 1
    ok = (e['open'] > 0) & (e['close'] > 0)
    intr = np.where(ok, e['close'] / e['open'] - 1, 0.0)
    ov = (1 + r_tot) / (1 + intr) - 1
    o = e['open'].to_numpy()
    slip = np.where(ok, 0.5 * np.where(o < 50, 0.01, 0.05) / np.where(ok, o, 1.0), 0.0003)     # ETF 跳動點：< 50 元 0.01、≥ 50 元 0.05
    return dict(ov=ov, intr=intr, slip=slip, n_missing=int((~ok).sum()), open=o, close=e['close'].to_numpy())


# ---------------------------------------------------------------- 年度 EPS 延伸（偽樣本外 a）
def implied_annual_eps(years=(2011, 2012)):
    """上市：FY y 年度 EPS ≈ (y+1) 年 4/10～4/30 每日『收盤÷本益比』的中位數（本益比 > 0 的日子；收盤取 data_panel/panel/close_raw.parquet）。
    回傳 {(code, y): eps} 與驗證表（用 FY2013、FY2014 對 fin_long 的年度 EPS）。"""
    import pyarrow.parquet as pq, pyarrow.compute as pc, pyarrow as pa, datetime as _dt
    cr = pd.read_parquet(CLOSE_OLD); cr.index = pd.to_datetime(cr.index); cr.columns = cr.columns.astype(str)
    cr = cr.loc['2012-04-01':'2015-05-31']
    fl = pd.read_parquet(cs.PEAD / 'fin_long.parquet', columns=['code', 'y', 'q', 'eps_ytd'])

    def imp(y):
        a, b = '%d-04-10' % (y + 1), '%d-04-30' % (y + 1)
        d = pq.read_table(cs.PEAD / 'per_long.parquet', columns=['date', 'code', 'mkt', 'per'],
                          filters=[('date', '>=', pd.Timestamp(a)), ('date', '<=', pd.Timestamp(b)), ('mkt', '==', 'twse'), ('per', '>', 0.0)]).to_pandas()
        d['date'] = pd.to_datetime(d['date'])
        cl = cr.loc[a:b].stack().rename('cl').reset_index(); cl.columns = ['date', 'code', 'cl']
        d = d.merge(cl, on=['date', 'code']); d['e'] = d['cl'] / d['per']
        return d.groupby('code')['e'].median()
    val = {}
    for y in (2013, 2014):
        im = imp(y); act = fl[(fl['y'] == y) & (fl['q'] == 4)].set_index('code')['eps_ytd']
        j = pd.concat([im, act], axis=1, keys=['im', 'act']).dropna(); j = j[j['act'] > 0]; rr = j['im'] / j['act'] - 1
        val[str(y)] = dict(n=int(len(j)), within2pct=float((rr.abs() < 0.02).mean()), within5pct=float((rr.abs() < 0.05).mean()))
    extra = {}
    for y in years:
        for c, v in imp(y).items():
            if np.isfinite(v):
                extra[(c, y)] = float(v)
    return extra, val


def load_eps_ext(P, extra=None):
    """與 cs.load_eps(P, pit=True) 相同；extra＝{(code, y): 年度 EPS} 只用來補 EpsY3（往前第 3 年）找不到的格子。"""
    f = pd.read_parquet(cs.PEAD / 'fin_long.parquet', columns=['code', 'y', 'q', 'eps_s', 'eps_ytd'])
    f = f[f['code'].str.fullmatch(r'[1-9]\d{3}')].copy()
    a = pd.read_parquet(cs.PEAD / 'avail_dates.parquet', columns=['code', 'y', 'q', 'A'])
    f = f.merge(a, on=['code', 'y', 'q'], how='left')
    f['dl'] = [cs.deadline(y, q) for y, q in zip(f['y'], f['q'])]
    f['avail'] = f['A'].fillna(f['dl'])
    f['seq'] = f['y'] * 4 + f['q']
    key = f.set_index(['code', 'y', 'q'])['eps_s']
    f['eps_q4'] = key.reindex(pd.MultiIndex.from_arrays([f['code'], f['y'] - 1, f['q']])).to_numpy()
    q = f[f['eps_s'].notna()][['code', 'seq', 'avail', 'eps_s', 'eps_q4']]
    sq = cs._step_panel(P, q, 'avail', ['eps_s', 'eps_q4'])
    s4 = f.pivot(index=['code', 'y'], columns='q', values='eps_s').sum(axis=1, skipna=False).rename('sum4')
    y4 = f[f['q'] == 4].set_index(['code', 'y'])
    ann = y4[['eps_ytd', 'avail']].join(s4, how='left')
    ann['eps_y'] = ann['eps_ytd'].fillna(ann['sum4'])
    ann = ann[ann['eps_y'].notna()].reset_index()
    ky = ann.set_index(['code', 'y'])['eps_y']
    ann['eps_y3'] = ky.reindex(pd.MultiIndex.from_arrays([ann['code'], ann['y'] - 3])).to_numpy()
    n_fill = 0
    if extra:
        ex = np.array([extra.get((c, y - 3), np.nan) for c, y in zip(ann['code'], ann['y'])])
        m = ann['eps_y3'].isna().to_numpy() & np.isfinite(ex)
        ann.loc[m, 'eps_y3'] = ex[m]; n_fill = int(m.sum())
    ann['seq'] = ann['y']
    sy = cs._step_panel(P, ann[['code', 'seq', 'avail', 'eps_y', 'eps_y3']], 'avail', ['eps_y', 'eps_y3'])
    return dict(EpsQ0=sq['eps_s'], EpsQ4=sq['eps_q4'], EpsY0=sy['eps_y'], EpsY3=sy['eps_y3'], info=dict(n_y3_filled_from_per=n_fill))


# ---------------------------------------------------------------- 出場規則
def exit_arrays(P):
    """股票層級的出場布林（原始價、無成交日沿用前收）。浮點比較不加容差，運算順序如下所寫。
    trail[p]：CF[t] <= max(CF[t-19..t]) * (1 - p/100)（滿 20 根才算）
    cross：SMA8[t] < SMA20[t] 且 SMA8[t-1] >= SMA20[t-1]
    ma50：CF[t] < SMA50[t]（滿 50 根才算）"""
    C = P.CF
    hc = cs.roll_max(C, 20)
    X = {'trail%d' % p: np.nan_to_num(C <= hc * (1 - p / 100.0), nan=0).astype(bool) for p in (6, 12, 15, 20)}
    f = cs.roll_mean(C, 8); s = cs.roll_mean(C, 20)
    X['cross'] = (f < s) & (cs.shift(f, 1) >= cs.shift(s, 1))
    m50 = cs.roll_mean(C, 50)
    X['ma50'] = np.nan_to_num(C < m50, nan=0).astype(bool)
    return X


def spec(trail=None, cross=False, fstop=None, ma50=False, hold=None):
    return dict(trail=trail, cross=cross, fstop=fstop, ma50=ma50, hold=hold)


CANDS = {                                               # 事前登記的 10 個候選（順序固定）
    'T15_MA':     spec(trail=15, cross=True),           # 主規格
    'T12_MA':     spec(trail=12, cross=True),
    'T20_MA':     spec(trail=20, cross=True),
    'T12':        spec(trail=12),
    'T15':        spec(trail=15),
    'T20':        spec(trail=20),
    'FS8_T15':    spec(trail=15, fstop=8),
    'FS8_T15_MA': spec(trail=15, cross=True, fstop=8),
    'MA50_FS8':   spec(ma50=True, fstop=8),
    'HOLD60':     spec(hold=60),
}
PRIMARY = 'T15_MA'
ORIG = spec(trail=6, cross=True)                        # 腳本原樣（只用於引擎等價檢查）


def _fee(shares, px, rate):
    if rate == 0:
        return 0.0
    rnd = (shares // 1000) * 1000; odd = shares - rnd
    f = 0.0
    if rnd > 0:
        f += max(MINFEE, rnd * px * rate)
    if odd > 0:
        f += max(MINFEE, odd * px * rate)
    return f


# ---------------------------------------------------------------- 可執行引擎（逐部位出場）
def simulate2(P, sig, rank, X, sp, univ, t_start, capital=1e6, slots=10, cost='normal', idle='cash', cap_pct=0.05,
              disp=None, t_end=None, lot_rule=True, same_day_slot=True, e50=None, keep_log=True):
    """日頻可執行模擬。訊號日 t-1 盤後 → t 開盤成交。與 cslib.simulate 的規則、同日先後順序、取整完全相同，另加：
    sp：出場規格（trail／cross／ma50 為股票層級；fstop＝自進場開盤價起的含息累積報酬 <= -fstop%；hold＝持有滿 hold 個交易日）
    idle：'cash' 或 'e50x'（空位資金放 0050：前收→開盤吃隔夜報酬，開盤以淨額換手並付成本，再吃開盤→收盤）
    lot_rule=False：不套 2020-10-26 前的整股限制；same_day_slot=False：當日賣出騰出的空位隔日才可買。"""
    T = t_end if t_end is not None else P.T
    O, C, R, REF, AMT = P.O, P.CF, P.R, P.REF, P.AMT
    fee = 0.0 if cost == 'zero' else FEE
    tax = 0.0 if cost == 'zero' else TAX
    extra = 0.003 if cost == 'stress' else 0.0
    minfee50 = 0.0 if cost == 'zero' else MINFEE
    TR = X['trail%d' % sp['trail']] if sp['trail'] else None
    CR = X['cross'] if sp['cross'] else None
    M5 = X['ma50'] if sp['ma50'] else None
    fs_level = (1 - sp['fstop'] / 100.0) if sp['fstop'] else None
    hold = sp['hold']
    cash = float(capital); pos = {}
    nav = np.full(T, np.nan); nav[t_start] = capital
    log = []
    st = dict(n_sig=0, n_buy=0, blk_limup=0, blk_noopen=0, blk_lot=0, blk_cap=0, blk_nocash=0, noslot=0, sell_defer_limdn=0, sell_defer_noopen=0,
              sell_partial_cap=0, n_delist_liq=0, buy_val=0.0, sell_val=0.0, fee=0.0, tax=0.0, slip=0.0, n_disp_buy=0, e50_cost=0.0, e50_traded=0.0,
              npos_sum=0, cash_frac_sum=0.0)
    odd_t = int(np.searchsorted(P.IDX.values, np.datetime64(cs.ODD_DATE), side='left')) if lot_rule else 0
    n_prev = 0
    for t in range(t_start + 1, T):
        if idle == 'e50x':
            cash *= (1 + e50['ov'][t])
            c_s50 = 0.0 if cost == 'zero' else (FEE + ETF_TAX + e50['slip'][t] + extra)
            c_b50 = 0.0 if cost == 'zero' else (FEE + e50['slip'][t] + extra)
        cash0 = cash; sold_today = set()
        # ---- 開盤：賣（依進場先後）
        for j in list(pos):
            p = pos[j]
            if P.MK[t, j] == 0 and not P.MK[t:, j].any():
                cash += p['v']; st['n_delist_liq'] += 1
                log.append((j, p['sig'], p['e'], t, p['cost'], p['proc'] + p['v'], 'delist', p['sh'], p['disp']))
                del pos[j]; sold_today.add(j); continue
            if not p['flag']:
                continue
            o = O[t, j]
            if not (np.isfinite(o) and o > 0):
                st['sell_defer_noopen'] += 1; continue
            if o <= P.LIMDN[t, j] + 1e-9:
                st['sell_defer_limdn'] += 1; continue
            vopen = p['v'] * o / REF[t, j]
            frac = 1.0
            if cap_pct and np.isfinite(AMT[t, j]) and vopen > cap_pct * AMT[t, j]:
                frac = cap_pct * AMT[t, j] / vopen; st['sell_partial_cap'] += 1
            sl = (0.0 if cost == 'zero' else 0.5 * float(cs.tick(o)) / o) + extra
            gross = vopen * frac * (1 - sl)
            sh = int(round(p['sh'] * frac))
            f = _fee(max(sh, 1), gross / max(sh, 1), fee); tx = gross * tax
            cash += gross - f - tx; sold_today.add(j)
            st['sell_val'] += vopen * frac; st['fee'] += f; st['tax'] += tx; st['slip'] += vopen * frac * sl
            if frac >= 1.0:
                log.append((j, p['sig'], p['e'], t, p['cost'], p['proc'] + gross - f - tx, p['why'], p['sh'], p['disp']))
                del pos[j]
            else:
                p['proc'] += gross - f - tx; p['v'] *= (1 - frac); p['sh'] = max(p['sh'] - sh, 0)
        # ---- 開盤：買
        cand = np.flatnonzero(sig[t - 1] & univ[t - 1])
        if len(cand):
            held_prev = set(pos) | sold_today
            cand = [j for j in cand if j not in held_prev]
            st['n_sig'] += len(cand)
            free = slots - len(pos)
            if not same_day_slot:
                free = min(free, slots - n_prev)
            if free <= 0:
                st['noslot'] += len(cand)
            elif cand:
                rk = rank[t - 1, cand]
                order = [cand[i] for i in np.lexsort((np.arange(len(cand)), -np.nan_to_num(rk, nan=-1e18)))]
                eq = cash + sum(p['v'] * (O[t, k] / REF[t, k] if np.isfinite(O[t, k]) and O[t, k] > 0 else 1.0) for k, p in pos.items())
                target = eq / slots
                res = (cash0 - max(cash0 - minfee50, 0.0) * (1 - c_s50)) if idle == 'e50x' else 0.0     # 0050 賣出成本的保留款
                for j in order:
                    if free <= 0:
                        st['noslot'] += 1; continue
                    o = O[t, j]
                    if not (np.isfinite(o) and o > 0):
                        st['blk_noopen'] += 1; continue
                    if o >= P.LIMUP[t, j] - 1e-9:
                        st['blk_limup'] += 1; continue
                    sl = (0.0 if cost == 'zero' else 0.5 * float(cs.tick(o)) / o) + extra
                    px = o * (1 + sl)
                    avail = cash - res
                    budget = min(target, avail)
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
                    f = _fee(sh, px, fee)
                    while sh > 0 and sh * px + f > avail + 1e-6:
                        sh -= 1000 if t < odd_t else 1
                        f = _fee(sh, px, fee) if sh > 0 else 0.0
                    if sh <= 0:
                        st['blk_nocash'] += 1; continue
                    cst = sh * px + f
                    cash -= cst
                    st['buy_val'] += sh * o; st['fee'] += f; st['slip'] += sh * o * sl; st['n_buy'] += 1
                    dz = int(disp[t, j]) if disp is not None else 0
                    st['n_disp_buy'] += dz
                    pos[j] = dict(v=sh * o, sh=sh, e=t, sig=t - 1, cost=cst, flag=False, why='', proc=0.0, new=True, disp=dz, g=1.0)
                    free -= 1
        # ---- 0050 淨額換手（開盤）
        if idle == 'e50x':
            d = cash - cash0
            if cost != 'zero' and abs(d) > 1e-9:
                if d > 0:                                   # 淨買進 0050
                    c50 = d * (c_b50 - FEE) + max(MINFEE, d * FEE)
                    st['e50_traded'] += d
                else:                                       # 淨賣出 0050，毛額 g 滿足 g×(1−c_s50)＝|d|
                    g = -d / (1 - c_s50)
                    c50 = g * (c_s50 - FEE) + max(MINFEE, g * FEE)
                    st['e50_traded'] += g
                cash -= c50; st['e50_cost'] += c50
            cash *= (1 + e50['intr'][t])
        # ---- 收盤：估值
        tot = 0.0
        for j, p in pos.items():
            if p['new']:
                rr = C[t, j] / O[t, j]; p['v'] = p['v'] * rr; p['g'] = rr; p['new'] = False
            else:
                p['v'] *= (1 + R[t, j]); p['g'] *= (1 + R[t, j])
            tot += p['v']
        nav[t] = cash + tot
        # ---- 盤後：出場訊號（含進場當日）
        for j, p in pos.items():
            if p['flag']:
                continue
            w = []
            if TR is not None and TR[t, j]:
                w.append('trail')
            if CR is not None and CR[t, j]:
                w.append('cross')
            if M5 is not None and M5[t, j]:
                w.append('ma50')
            if fs_level is not None and p['g'] <= fs_level:
                w.append('fstop')
            if hold is not None and t - p['e'] + 1 >= hold:
                w.append('hold')
            if w:
                p['flag'] = True; p['why'] = '+'.join(w)
        n_prev = len(pos)
        st['npos_sum'] += len(pos); st['cash_frac_sum'] += cash / nav[t]
    for j, p in pos.items():
        log.append((j, p['sig'], p['e'], T - 1, p['cost'], p['proc'] + p['v'], 'open_end', p['sh'], p['disp']))
    navs = pd.Series(nav[t_start:T], index=P.IDX[t_start:T])
    nd = T - 1 - t_start
    st['avg_npos'] = st.pop('npos_sum') / nd; st['avg_cash_frac'] = st.pop('cash_frac_sum') / nd
    yrs = (navs.index[-1] - navs.index[0]).days / 365.25; an = float(navs.mean())
    st['turnover_oneway_per_year'] = st['buy_val'] / an / yrs
    st['cost_acct_pct_per_year'] = (st['fee'] + st['tax'] + st['slip'] + st['e50_cost']) / an / yrs * 100
    st['fee_pct'] = st['fee'] / an / yrs * 100; st['tax_pct'] = st['tax'] / an / yrs * 100; st['slip_pct'] = st['slip'] / an / yrs * 100
    lg = None
    if keep_log:
        lg = pd.DataFrame(log, columns=['j', 'sig_t', 'ent_t', 'exit_t', 'cost', 'proceeds', 'why', 'shares', 'disp'])
        if len(lg):
            lg['code'] = P.COLS[lg['j'].to_numpy()]; lg['ret'] = lg['proceeds'] / lg['cost'] - 1
            lg['sig_date'] = P.IDX[lg['sig_t'].to_numpy()]; lg['ent_date'] = P.IDX[lg['ent_t'].to_numpy()]; lg['exit_date'] = P.IDX[lg['exit_t'].to_numpy()]
    return navs, lg, st


# ---------------------------------------------------------------- 貼文口徑（不限檔數）逐筆：每持有日毛報酬
def post_trades2(P, sig, X, sp, univ, t_start):
    """訊號日 t 盤後入選且未持有 → t+1 開盤買（無開盤價放棄）；出場訊號日 k（自進場日收盤起檢查）→ k+1 起第一個有開盤價的日子開盤賣；
    賣出當日收盤即可再入選；期末未平倉以最後收盤結算。回傳 DataFrame（含息毛報酬 ret_tr、持有交易日 hold）。"""
    T = P.T; O, C, R, REF = P.O, P.CF, P.R, P.REF
    TR = X['trail%d' % sp['trail']] if sp['trail'] else None
    st_lvl = np.zeros((T, P.N), bool)
    if TR is not None:
        st_lvl |= TR
    if sp['cross']:
        st_lvl |= X['cross']
    if sp['ma50']:
        st_lvl |= X['ma50']
    fs_level = (1 - sp['fstop'] / 100.0) if sp['fstop'] else None
    hold = sp['hold']
    rows = []
    for j in np.flatnonzero((sig[t_start:] & univ[t_start:]).any(axis=0)):
        s = sig[:, j] & univ[:, j]; ex = st_lvl[:, j]
        t = t_start
        while t < T - 1:
            if not s[t]:
                nz = np.flatnonzero(s[t + 1:T - 1])
                if len(nz) == 0:
                    break
                t = t + 1 + nz[0]
            e = t + 1
            if not (np.isfinite(O[e, j]) and O[e, j] > 0):
                t += 1; continue
            g = np.concatenate([[C[e, j] / O[e, j]], 1 + R[e + 1:T, j]]).cumprod()       # g[i]＝e+i 日收盤的含息累積倍數
            k = None
            nz = np.flatnonzero(ex[e:T])
            if len(nz):
                k = e + nz[0]
            if fs_level is not None:
                nz = np.flatnonzero(g <= fs_level)
                if len(nz) and (k is None or e + nz[0] < k):
                    k = e + nz[0]
            if hold is not None and e + hold - 1 < T and (k is None or e + hold - 1 < k):
                k = e + hold - 1
            x = None
            if k is not None:
                x = k + 1
                while x < T and not (np.isfinite(O[x, j]) and O[x, j] > 0):
                    x += 1
                if x >= T:
                    x = None
            if x is None:
                rows.append((j, t, e, T - 1, g[T - 1 - e] - 1, T - 1 - e, 1)); break
            rows.append((j, t, e, x, g[x - 1 - e] * (O[x, j] / REF[x, j]) - 1, x - e, 0))
            t = x
    d = pd.DataFrame(rows, columns=['j', 'sig_t', 'ent_t', 'exit_t', 'ret_tr', 'hold', 'open_end'])
    return d


# ---------------------------------------------------------------- 路徑擾動與安慰劑
def zrank(sig_u, RS):
    """每個訊號日，在當日全部訊號（sig & univ）之中把 RS 標準化（母體標準差；只有 1 檔或標準差 0 → 0）。非訊號格 NaN。"""
    Z = np.full(RS.shape, np.nan)
    for t in np.flatnonzero(sig_u.any(axis=1)):
        jj = np.flatnonzero(sig_u[t]); v = RS[t, jj]; sd = v.std()
        Z[t, jj] = (v - v.mean()) / sd if (len(jj) > 1 and sd > 0) else 0.0
    return Z


def placebo_signal(P, sig_u, pool_u, seed):
    """安慰劑訊號：每個交易日 t，從當日池子（通過 N、S、L、M 與 base、在股池內；不看 C、A）抽 k_t 檔，k_t＝當日真實訊號檔數。
    抽法：每個（日曆月, 股票）一個均勻亂數 u（np.random.default_rng(seed).random((月數, N))，月份自面板第一個月起依序），
    取當日池內 u 最小的 k_t 檔（同月內名單因此有持續性）。"""
    rng = np.random.default_rng(seed)
    mon = (P.IDX.year - P.IDX.year[0]) * 12 + (P.IDX.month - P.IDX.month[0])
    U = rng.random((int(mon.max()) + 1, P.N))
    out = np.zeros(sig_u.shape, bool)
    k = sig_u.sum(axis=1)
    for t in np.flatnonzero(k > 0):
        jj = np.flatnonzero(pool_u[t])
        u = U[mon[t], jj]
        out[t, jj[np.argsort(u, kind='stable')[:k[t]]]] = True
    return out


# ---------------------------------------------------------------- 統計
def mret(nav, a=None, b=None):
    """日曆月報酬。視窗 [a, b]：基期＝a 之前最後一個交易日的淨值（沒有就用第一筆）；若基期與第一個月底是同一天（零長度月）則丟掉該月。"""
    s = nav.dropna()
    if b is not None:
        s = s[s.index <= pd.Timestamp(b)]
    if a is not None:
        s0 = s[s.index < pd.Timestamp(a)]
        s = pd.concat([s0.iloc[-1:], s[s.index >= pd.Timestamp(a)]]) if len(s0) else s
    base = s.iloc[0]; base_d = s.index[0]
    body = s.iloc[1:]
    m = body.groupby(body.index.to_period('M')).last()
    v = np.concatenate([[base], m.to_numpy()])
    return pd.Series(v[1:] / v[:-1] - 1, index=m.index)


def tstat(x):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    if len(x) < 3 or x.std(ddof=1) == 0:
        return float('nan')
    return float(x.mean() / (x.std(ddof=1) / math.sqrt(len(x))))


def nw_t(x, lags=6):
    x = np.asarray(x, float); n = len(x); m = x.mean(); d = x - m
    s = (d * d).sum() / n
    for l in range(1, lags + 1):
        s += 2 * (1 - l / (lags + 1)) * (d[l:] * d[:-l]).sum() / n
    return float(m / math.sqrt(max(s, 1e-18) / n))


def wstats(nav, bench, a=None, b=None):
    """年化（日曆年化：(終值/基期值)^(365.25/日曆天數)−1）、0050 同期、超額 pp、月超額 t 與 NW(6) t、最大回撤。"""
    s = nav.dropna(); e = bench.reindex(s.index)
    c1 = cs.cagr(s, a, b); c2 = cs.cagr(e, a, b)
    if c1 is None:
        return None
    ms = mret(s, a, b); mb = mret(e, a, b); ex = ms - mb
    return dict(cagr=c1['cagr'], cagr_0050=c2['cagr'], excess_pp=(c1['cagr'] - c2['cagr']) * 100, maxdd=c1['maxdd'], maxdd_0050=c2['maxdd'],
                years=c1['years'], start=c1['start'], end=c1['end'], n_months=int(len(ex)), mean_ex_m=float(ex.mean()), sd_ex_m=float(ex.std(ddof=1)),
                t_ex_m=tstat(ex), nw6_t=nw_t(ex.to_numpy(), 6), pos_month_share=float((ex > 0).mean()))


def drop_best3(nav, bench, a=None, b=None):
    """拿掉『年度超額（策略年報酬−0050 年報酬，由月報酬連乘）最高的 3 個日曆年』後的月超額 t。"""
    s = nav.dropna(); e = bench.reindex(s.index)
    ms = mret(s, a, b); mb = mret(e, a, b); ex = ms - mb
    ys = (1 + ms).groupby(ms.index.year).prod() - 1; yb = (1 + mb).groupby(mb.index.year).prod() - 1
    yex = (ys - yb).sort_values(ascending=False); best = list(yex.index[:3])
    rest = ex[~np.isin(ex.index.year, best)]
    return dict(best3=[int(x) for x in best], t=tstat(rest), n=int(len(rest)),
                yearly={int(y): dict(strat=float(ys[y]), e50=float(yb[y]), ex_pp=float((ys[y] - yb[y]) * 100)) for y in ys.index})


def beta_alpha(ms, mb):
    """月報酬對 0050 月報酬的 OLS（含截距，無風險利率視為 0）。回傳 β、月 α、α 的 OLS t 與 NW(6) t、β 相同的『0050＋現金』年化。"""
    y = np.asarray(ms, float); x = np.asarray(mb, float); n = len(y)
    Xm = np.column_stack([np.ones(n), x]); coef, *_ = np.linalg.lstsq(Xm, y, rcond=None)
    res = y - Xm @ coef; s2 = (res ** 2).sum() / (n - 2); cov = s2 * np.linalg.inv(Xm.T @ Xm)
    a, b = float(coef[0]), float(coef[1])
    yrs = n / 12.0
    return dict(beta=b, alpha_m=a, alpha_t=float(a / math.sqrt(cov[0, 0])), alpha_nw6_t=nw_t(y - b * x, 6), alpha_ann_pp=a * 12 * 100,
                corr=float(np.corrcoef(x, y)[0, 1]), n=int(n),
                cagr_strat_m=float(np.prod(1 + y) ** (1 / yrs) - 1), cagr_beta_mix_m=float(np.prod(1 + b * x) ** (1 / yrs) - 1),
                cagr_0050_m=float(np.prod(1 + x) ** (1 / yrs) - 1))


def block_boot_ci(x, block=6, B=5000, seed=20261114):
    """循環移動區塊拔靴（區塊 6 個月）：平均數的 2.5%／97.5% 分位與 P(平均 <= 0)。"""
    x = np.asarray(x, float); n = len(x); rng = np.random.default_rng(seed)
    nb = int(math.ceil(n / block)); out = np.empty(B)
    for i in range(B):
        st = rng.integers(0, n, nb)
        idx = ((st[:, None] + np.arange(block)[None, :]) % n).ravel()[:n]
        out[i] = x[idx].mean()
    return dict(mean=float(x.mean()), lo=float(np.quantile(out, 0.025)), hi=float(np.quantile(out, 0.975)), p_le0=float((out <= 0).mean()))


def spa_test(Xm, B=5000, q=1 / 6, seed=11):
    """Hansen (2005) SPA。Xm：[n × k] 月超額（對 0050）。定態拔靴（平均區塊 6 個月）。回傳 consistent p 與 upper p（White RC）。"""
    Xm = np.asarray(Xm, float); n, k = Xm.shape
    mu = Xm.mean(0); rng = np.random.default_rng(seed)
    idx = np.empty((B, n), int)
    for b_ in range(B):
        i = rng.integers(0, n); row = np.empty(n, int)
        for t in range(n):
            row[t] = i
            i = rng.integers(0, n) if rng.random() < q else (i + 1) % n
        idx[b_] = row
    bm = Xm[idx].mean(1)
    om = np.sqrt(n) * bm.std(0, ddof=1); om = np.where(om > 0, om, np.inf)
    Tn = max(0.0, float(np.max(np.sqrt(n) * mu / om)))
    thr = -np.sqrt(2 * np.log(np.log(n)))
    Zc = np.sqrt(n) * (bm - mu) / om + np.where(np.sqrt(n) * mu / om >= thr, 0.0, np.sqrt(n) * mu / om)
    Tb = np.maximum(0.0, Zc.max(1)); Tl = np.maximum(0.0, (np.sqrt(n) * (bm - mu) / om).max(1))
    return dict(T=Tn, p_consistent=float((Tb >= Tn).mean()), p_upper=float((Tl >= Tn).mean()), n=int(n), k=int(k), best_col=int(np.argmax(np.sqrt(n) * mu / om)))


def dsr(x, N, var_sr):
    """Deflated Sharpe Ratio（Bailey & López de Prado 2014）。x：月超額；N：試驗數；var_sr：試驗間月 SR 的變異數。"""
    from statistics import NormalDist
    nd = NormalDist(); x = np.asarray(x, float); n = len(x)
    sr = x.mean() / x.std(ddof=1); g = 0.5772156649
    sr0 = math.sqrt(var_sr) * ((1 - g) * nd.inv_cdf(1 - 1 / N) + g * nd.inv_cdf(1 - 1 / (N * math.e)))
    z = (x - x.mean()) / x.std(ddof=0); sk = float((z ** 3).mean()); ku = float((z ** 4).mean())
    den = math.sqrt(max(1e-12, 1 - sk * sr + (ku - 1) / 4 * sr ** 2))
    return dict(sr_m=float(sr), sr0_m=float(sr0), dsr=float(nd.cdf((sr - sr0) * math.sqrt(n - 1) / den)), skew=sk, kurt=ku, n=int(n))


def main_strategy_nav(a='2017-03-31'):
    """主策略 V7X 可執行（100 萬、一般成本）13 個 j 的淨值：各 j 在 a 日（含）之前最後一個值歸一後等權平均。只用來算相關、配對差與混合。"""
    d = pd.read_csv(MAINNAV, index_col=0, parse_dates=True)
    cols = ['base|REAL_opt|%d|normal|100萬|std' % j for j in range(13)]
    x = d[cols].ffill()
    base = x[x.index <= pd.Timestamp(a)].iloc[-1]
    return (x / base).mean(axis=1)[lambda s: s.index >= x[x.index <= pd.Timestamp(a)].index[-1]]


def rnd(o, k=6):
    if isinstance(o, dict):
        return {str(kk): rnd(v, k) for kk, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [rnd(v, k) for v in o]
    if isinstance(o, (float, np.floating)):
        return None if not np.isfinite(o) else round(float(o), k)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o
