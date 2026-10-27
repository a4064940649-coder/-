"""disposal_reconcile 共用：標準事件表＋可切換的事件定義。

所有開關見 FINDINGS.md 第 0 節（事前登記）。資料唯讀：面板（上市）、disposal_5y、0050。
用法：
    import rlib as L
    T = L.trades(L.CFG_OFFICIAL)          # 每筆交易一列：date(進場日)、code、i0、i1、raw、ab、ab_lag、ab50、cost、net...
    L.stats(T, col='net')
"""
import math
from pathlib import Path
import numpy as np, pandas as pd

S = Path('/tmp/claude-0/-home-user--/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad'); P = S / 'panel'
DD = Path('/home/user/stevenhsiehsheng-ctrl/tw-stock-screener/data'); X = DD / 'extras'
HERE = Path(__file__).resolve().parent
GAP = (pd.Timestamp('2024-01-02'), pd.Timestamp('2024-02-14'))
SEL = ('2021-10-01', '2023-12-31'); VAL = ('2024-01-01', '2026-10-31')
PER = {'sel_2021-10_2023': SEL, 'val_2024_2026-10': VAL, 'val_a_2024_2025': ('2024-01-01', '2025-12-31'),
       'val_b_2026': ('2026-01-01', '2026-10-31'), 'all_2021-10_2026-10': ('2021-10-01', '2026-10-31')}

# ---------------- 面板 ----------------
adj = pd.read_parquet(P / 'adj.parquet'); C = pd.read_parquet(P / 'close_raw.parquet'); H = pd.read_parquet(P / 'high.parquet')
V = pd.read_parquet(P / 'value.parquet'); R = pd.read_parquet(P / 'ret.parquet'); FL = pd.read_parquet(P / 'flag.parquet')
SRC = pd.read_parquet(P / 'src.parquet')
IDX = adj.index; COLS = list(adj.columns); CI = {c: j for j, c in enumerate(COLS)}; T = len(IDX)
liq_now = (V.rolling(60, min_periods=60).mean() >= 1e7) & (C >= 10)
liq_lag = liq_now.shift(1, fill_value=False)
mkt_now = R.where(liq_now).mean(axis=1).fillna(0)
mkt_lag = R.where(liq_lag).mean(axis=1).fillna(0)
e50 = pd.read_csv('/home/user/jonatw/finlab-txd/data/raw/etf_0050.csv', parse_dates=['date']).set_index('date')['adj'].sort_index()
E50 = e50.reindex(IDX.union(e50.index)).ffill().reindex(IDX)
r50 = E50.pct_change().fillna(0)
LA = np.log1p(R.fillna(0).clip(lower=-0.99)).cumsum().to_numpy()
LMN = np.log1p(mkt_now.clip(lower=-0.99)).cumsum().to_numpy()
LML = np.log1p(mkt_lag.clip(lower=-0.99)).cumsum().to_numpy()
LE = np.log1p(r50).cumsum().to_numpy()
ADJN, CN, HN, VN, RN, FN, SN = (x.to_numpy() for x in (adj, C, H, V, R, FL, SRC))
LIQL = liq_lag.to_numpy()
LASTV = np.array([np.max(np.where(np.isfinite(ADJN[:, j]))[0]) if np.isfinite(ADJN[:, j]).any() else -1 for j in range(len(COLS))])
DATES = IDX.to_numpy()


def in_gap(d):
    return GAP[0] <= d <= GAP[1]


# ---------------- 處置記錄 ----------------
DP_ALL = pd.read_csv(X / 'disposal_5y.csv.gz', dtype={'code': str}, parse_dates=['date', 'start', 'end'])
DP_ALL['ordr'] = np.arange(len(DP_ALL))


def records(dp=None, idx=IDX, cols=None):
    """標準事件表（每筆處置記錄一列）：ann_i, s_i, e_i, 缺口旗標, 再處置旗標。"""
    dp = DP_ALL if dp is None else dp
    cols = CI if cols is None else cols
    d = dp[dp['code'].isin(cols)].copy()
    d['ann_i'] = idx.searchsorted(d['date'].values, 'right') - 1
    d['s_i'] = idx.searchsorted(d['start'].values)
    d['e_i'] = idx.searchsorted(d['end'].values, 'right') - 1
    d['start_in_gap'] = d['start'].map(in_gap); d['end_in_gap'] = d['end'].map(in_gap)
    d['start_is_tday'] = d['start'].isin(idx)
    d = d.sort_values(['code', 'start', 'date', 'ordr']).reset_index(drop=True)
    # 再處置（exec/ev5 定義）：本筆 start 前 30 個交易日內，同檔有先前記錄結束（且結束 < 本筆 start）
    sec = []; pe = {}
    for r in d.itertuples():
        p = pe.get(r.code, -10**9)
        sec.append(bool(p >= r.s_i - 30 and p < r.s_i))
        pe[r.code] = max(p, r.e_i)
    d['second'] = sec
    return d


REC = records()


# ---------------- 開關 ----------------
CFG_EV_BASE = dict(gap='keep', overlap='none', entry='lock', exit='none', cost='0', bench='now', liq='none', start_filter='entry')
CFG_EV_STRICT = dict(CFG_EV_BASE, overlap='first_undelayed', exit='lock')
CFG_EV_B = dict(CFG_EV_BASE, overlap='merge_all')
CFG_AT_R0 = dict(gap='drop', overlap='chain', entry='ret9', exit='ret9_max20', cost='0.8', bench='lag', liq='none', start_filter='entry')
CFG_BEAR = dict(gap='keep', overlap='first', entry='ret9', exit='ret94', cost='0.8', bench='now', liq='none', start_filter='entry')
CFG_EXEC = dict(gap='keep', overlap='first', entry='ret9_nan', exit='ret9', cost='0.8/1.2', bench='raw', liq='none', start_filter='pre_oct_ignored')
CFG_OFFICIAL = dict(gap='drop', overlap='merge_known', entry='ret9', exit='ret9', cost='0.8/1.2', bench='lag', liq='lag', start_filter='entry')


def tradable(cfg, i, j):
    if not np.isfinite(ADJN[i, j]): return False
    r = RN[i, j]; e = cfg['entry']
    if e == 'none': return True
    if e == 'lock': return not (np.isfinite(r) and r >= 0.095 and CN[i, j] >= HN[i, j] * 0.999)
    if e == 'ret9': return not (np.isfinite(r) and r >= 0.09)
    if e == 'ret9_nan': return bool(np.isfinite(r) and r < 0.09)
    raise ValueError(e)


def delay_exit(cfg, k, j):
    x = cfg['exit']
    if x == 'none': return k
    if x == 'lock':
        while k < T - 1 and FN[k, j] == 0 and np.isfinite(CN[k, j]) and np.isfinite(CN[k - 1, j]) and CN[k, j] / CN[k - 1, j] - 1 <= -0.095: k += 1
        return k
    if x == 'ret9_max20':
        n = 0
        while k + 1 < T and n < 20 and np.isfinite(RN[k, j]) and RN[k, j] <= -0.09: k += 1; n += 1
        return k
    thr = -0.09 if x == 'ret9' else -0.094
    lim = min(T - 1, LASTV[j])
    while k < lim and not (np.isfinite(RN[k, j]) and RN[k, j] > thr): k += 1
    return k


def _cost(cfg, second):
    c = cfg['cost']
    if c == '0.8/1.2': return 0.012 if second else 0.008
    return float(c) / 100


def positions(cfg, rec=REC):
    """依重疊規則產生部位 list of dict(code, j, i0, i1(延後後), plan_e, second, ann_i, n_links, ord)。"""
    out = []; stat = {'records': 0, 'invalid': 0, 'gap': 0, 'busy_skip': 0, 'untradable': 0, 'illiquid': 0, 'extended': 0, 'pre_oct_ignored': 0}
    s0 = IDX.searchsorted(pd.Timestamp('2021-10-01'))
    for code, g in rec.groupby('code', sort=False):
        j = CI[code]; g = g.sort_values(['s_i', 'ann_i', 'ordr'])
        rows = list(g.itertuples())
        if cfg['overlap'] == 'chain':
            # 鏈：重疊或首尾相接（下一段 s ≤ 上一段 e + 1）
            ch = []
            for r in rows:
                if r.s_i >= T or r.e_i < r.s_i: continue
                if ch and r.s_i <= ch[-1]['e'] + 1:
                    ch[-1]['e'] = max(ch[-1]['e'], r.e_i); ch[-1]['n'] += 1
                else:
                    ch.append({'s': r.s_i, 'e': r.e_i, 'n': 1, 'r': r})
            busy = -1
            for k in ch:
                stat['records'] += 1; r = k['r']
                if k['e'] <= k['s']: stat['invalid'] += 1; continue
                if k['s'] <= busy: stat['busy_skip'] += 1; continue
                if cfg['gap'] == 'drop' and (r.start_in_gap or (IDX[k['s']] <= GAP[0] and IDX[k['e']] >= GAP[1])): stat['gap'] += 1; continue
                if not tradable(cfg, k['s'], j): stat['untradable'] += 1; continue
                if cfg['liq'] == 'lag' and not LIQL[k['s'], j]: stat['illiquid'] += 1; continue
                i1 = delay_exit(cfg, k['e'], j)
                if cfg['gap'] == 'drop' and IDX[k['s']] <= GAP[0] and IDX[i1] >= GAP[1]: stat['gap'] += 1; continue
                busy = i1
                out.append({'code': code, 'j': j, 'i0': k['s'], 'plan_e': k['e'], 'i1': i1, 'second': r.second, 'ann_i': r.ann_i,
                            'n_links': k['n'], 'ord': r.ordr, 'ann_date': r.date})
            continue
        cur = None; busy = -1
        def close(cur):
            cur['i1'] = delay_exit(cfg, cur['plan_e'], j); return cur
        for r in rows:
            stat['records'] += 1
            if r.s_i >= T or r.e_i <= r.s_i: stat['invalid'] += 1; continue
            if cfg['start_filter'] == 'pre_oct_ignored' and r.s_i < s0: stat['pre_oct_ignored'] += 1; continue
            if cur is not None and cfg['overlap'] == 'none':
                out.append(cur); cur = None
            if cur is not None:
                hold_e = cur['plan_e'] if cfg['overlap'] in ('first_undelayed', 'merge_all', 'merge_known') else cur['i1']
                if r.s_i <= hold_e or (cfg['overlap'] in ('merge_all', 'merge_known') and r.s_i <= cur.get('i1', cur['plan_e'])):
                    ext = False
                    if cfg['overlap'] == 'merge_all' and r.e_i > cur['plan_e']: ext = True
                    if cfg['overlap'] == 'merge_known' and r.e_i > cur['plan_e'] and r.ann_i < cur['i1']: ext = True
                    if ext:
                        cur['plan_e'] = r.e_i; cur['n_links'] += 1; stat['extended'] += 1
                        close(cur)
                    else:
                        stat['busy_skip'] += 1
                    continue
                out.append(cur); cur = None
            if cfg['gap'] == 'drop' and (r.start_in_gap or r.end_in_gap or (r.start < GAP[0] and r.end > GAP[1])): stat['gap'] += 1; continue
            if not tradable(cfg, r.s_i, j): stat['untradable'] += 1; continue
            if cfg['liq'] == 'lag' and not LIQL[r.s_i, j]: stat['illiquid'] += 1; continue
            cur = {'code': code, 'j': j, 'i0': r.s_i, 'plan_e': r.e_i, 'second': r.second, 'ann_i': r.ann_i, 'n_links': 1, 'ord': r.ordr, 'ann_date': r.date}
            close(cur)
        if cur is not None: out.append(cur)
    if cfg['gap'] == 'drop':
        keep = []
        for p in out:
            if IDX[p['i0']] <= GAP[0] and IDX[p['i1']] >= GAP[1]: stat['gap'] += 1; continue
            keep.append(p)
        out = keep
    return out, stat


def trades(cfg, rec=REC, with_stat=False):
    pos, stat = positions(cfg, rec)
    rows = []
    for p in pos:
        i0, i1, j = p['i0'], p['i1'], p['j']
        raw = math.expm1(LA[i1, j] - LA[i0, j])
        mn = math.expm1(LMN[i1] - LMN[i0]); ml = math.expm1(LML[i1] - LML[i0]); m5 = math.expm1(LE[i1] - LE[i0])
        b = {'now': mn, 'lag': ml, 'raw': 0.0, '0050': m5}[cfg['bench']]
        cst = _cost(cfg, p['second'])
        rows.append(dict(p, date=IDX[i0], exit=IDX[i1], hold=i1 - i0, raw=raw, ab_now=raw - mn, ab_lag=raw - ml, ab50=raw - m5,
                         gross=raw - b, cost=cst, net=raw - b - cst, raw_net=raw - cst, delayed=i1 > p['plan_e'],
                         src3=bool(SN[i0, j] == 3 or SN[i1, j] == 3)))
    df = pd.DataFrame(rows)
    if len(df): df = df.sort_values(['date', 'code']).reset_index(drop=True)
    return (df, stat) if with_stat else df


# ---------------- 統計 ----------------
def tstat(x):
    x = pd.Series(x).dropna()
    return round(float(x.mean() / (x.std(ddof=1) / math.sqrt(len(x)))), 2) if len(x) > 2 and x.std() > 0 else None


def stats(df, col='net'):
    df = df.dropna(subset=[col]) if len(df) else df
    o = {'n': int(len(df))}
    if len(df) < 5: return o
    x = df[col]; m = x.groupby(df['date'].dt.to_period('M')).mean()
    g = df['gross']; mg = g.groupby(df['date'].dt.to_period('M')).mean()
    xs = x.sort_values(); k5 = max(1, int(math.ceil(len(xs) * 0.05)))
    ex5 = df.loc[xs.index[:-k5]]
    lo, hi = x.quantile([0.05, 0.95])
    o.update({'mean_pct': round(x.mean() * 100, 2), 'median_pct': round(x.median() * 100, 2), 'hit': round(float((x > 0).mean()), 3),
              't_month': tstat(m), 't_month_gross': tstat(mg), 't_date': tstat(x.groupby(df['date']).mean()), 't_event': tstat(x),
              'n_months': int(len(m)), 'share_pos_months': round(float((m > 0).mean()), 3),
              'gross_mean_pct': round(g.mean() * 100, 2), 'mean_raw_net_pct': round(df['raw_net'].mean() * 100, 2),
              'ex_top5pct_mean_pct': round(ex5[col].mean() * 100, 2), 'ex_top5pct_median_pct': round(ex5[col].median() * 100, 2),
              'ex_top5pct_t_month': tstat(ex5[col].groupby(ex5['date'].dt.to_period('M')).mean()),
              'winsor5_mean_pct': round(x.clip(lo, hi).mean() * 100, 2), 'mean_hold_days': round(float(df['hold'].mean()), 2)})
    return o


def by_per(df, col='net', per=PER):
    out = {}
    for k, (a, b) in per.items():
        out[k] = stats(df[(df['date'] >= pd.Timestamp(a)) & (df['date'] <= pd.Timestamp(b))] if len(df) else df, col)
    return out


def yearly(df, col='net'):
    return {int(y): stats(g, col) for y, g in df.groupby(df['date'].dt.year)}


# ---------------- 市場切換（上櫃用 history.csv.gz，對齊到面板日曆） ----------------
_KEYS = ['ADJN', 'CN', 'HN', 'VN', 'RN', 'FN', 'SN', 'LIQL', 'LASTV', 'LA', 'LMN', 'LML', 'CI', 'COLS']
TWSE = {k: globals()[k] for k in _KEYS}
TWSE['name'] = 'TWSE'


def set_market(M):
    globals().update({k: M[k] for k in _KEYS})


def load_hist_market(market):
    """history.csv.gz（2025-09-24 起官方日 K）→ 與面板同日曆的陣列。報酬：收盤/前收−1；除權息日用 exdiv 參考價；|ret|≥60% 視為缺值。"""
    h = pd.read_csv(DD / 'history.csv.gz', dtype={'code': str}, parse_dates=['date'])
    sl = pd.read_csv(DD / 'stock_list.csv', dtype={'code': str})
    codes = set(sl.loc[sl['market'] == market, 'code'])
    h = h[h['code'].isin(codes) & (h['close'] > 0) & (h['volume'] > 0)]
    W = {k: h.pivot(index='date', columns='code', values=k).sort_index() for k in ('high', 'close', 'volume')}
    hidx = W['close'].index
    Cc = W['close']; prev = Cc.ffill(limit=5).shift(1)
    ex = pd.read_csv(X / 'exdiv_5y.csv.gz', dtype={'code': str}, parse_dates=['date'])
    ex = ex[(ex['date'] >= hidx[0]) & (ex['ref_price'] > 0) & ex['code'].isin(Cc.columns)]
    ref = ex.pivot_table(index='date', columns='code', values='ref_price', aggfunc='last').reindex(index=hidx, columns=Cc.columns)
    Rr = (Cc / ref.where(ref.notna(), prev) - 1)
    Rr = Rr.where(Rr.abs() < 0.6)
    Vv = W['volume'] * Cc
    lq = ((Vv.rolling(60, min_periods=20).mean() >= 1e7) & (Cc >= 10))
    lql = lq.shift(1, fill_value=False)
    mn = Rr.where(lq).mean(axis=1).fillna(0); ml = Rr.where(lql).mean(axis=1).fillna(0)
    adjh = (1 + Rr.fillna(0)).cumprod().where(Cc.notna())
    rx = lambda df: df.reindex(IDX)
    A_ = rx(adjh); R_ = rx(Rr)
    M = {'name': market, 'ADJN': A_.to_numpy(), 'CN': rx(Cc).to_numpy(), 'HN': rx(W['high']).to_numpy(), 'VN': rx(Vv).to_numpy(),
         'RN': R_.to_numpy(), 'FN': np.zeros(A_.shape), 'SN': np.full(A_.shape, 4.0), 'LIQL': rx(lql).fillna(False).astype(bool).to_numpy(),
         'LA': np.log1p(R_.fillna(0).clip(lower=-0.99)).cumsum().to_numpy(),
         'LMN': np.log1p(mn.reindex(IDX).fillna(0).clip(lower=-0.99)).cumsum().to_numpy(),
         'LML': np.log1p(ml.reindex(IDX).fillna(0).clip(lower=-0.99)).cumsum().to_numpy(),
         'CI': {c: j for j, c in enumerate(Cc.columns)}, 'COLS': list(Cc.columns), 'first_day': hidx[0],
         'ADJ_FFILL': A_.ffill().to_numpy()}
    a = M['ADJN']
    M['LASTV'] = np.array([np.max(np.where(np.isfinite(a[:, j]))[0]) if np.isfinite(a[:, j]).any() else -1 for j in range(a.shape[1])])
    return M


def records_market(M, dp=None):
    """某市場的標準事件表；再處置旗標用全部記錄（含 2025-09 之前）在面板日曆上計算。只保留 start > 該市場資料第一天的記錄。"""
    rec = records(dp, IDX, M['CI'])
    if 'first_day' in M: rec = rec[rec['start'] > M['first_day']].reset_index(drop=True)
    return rec
