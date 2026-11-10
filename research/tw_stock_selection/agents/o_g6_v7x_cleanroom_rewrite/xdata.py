# 資料層（clean-room）：V7 凍結面板＋P 下市價、GOOD、V60、σ60、vrank、跳動點、漲跌停、處置名單、0050、CPI
import numpy as np, pandas as pd, json, os, hashlib
AG = '/home/user/-/research/tw_stock_selection/agents/'
D7 = AG + 'o_v7_listing_exec_engine/data/'
SCR = '/tmp/claude-0/-home-user--/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/g6A/'
CACHE = SCR + 'xdata_v1.npz'
EXEMPT = [pd.Timestamp('2017-08-03'), pd.Timestamp('2018-07-12')] + list(pd.date_range('2024-02-15', '2024-03-15'))

def rpq(p):
    with open(p, 'rb') as f: return pd.read_parquet(f)

def tick_of(p):
    p = np.asarray(p, dtype=float)
    return np.where(p < 10, 0.01, np.where(p < 50, 0.05, np.where(p < 100, 0.1, np.where(p < 500, 0.5, np.where(p < 1000, 1.0, 5.0)))))

def build():
    C = rpq(D7 + 'close_raw.parquet'); O = rpq(D7 + 'open.parquet'); V = rpq(D7 + 'value.parquet')
    R = rpq(D7 + 'ret.parquet'); FL = rpq(D7 + 'flag.parquet')
    dates = C.index; codes = list(C.columns)
    C = C.values.copy(); O = O[codes].values.copy(); V = V[codes].values.copy(); R = R[codes].values.copy(); FL = FL[codes].values.copy()
    P = rpq(AG + 'o_infra_v3_rebuild/delist_prices.parquet'); P['date'] = pd.to_datetime(P['date'])
    di = {d: i for i, d in enumerate(dates)}; ci = {c: j for j, c in enumerate(codes)}
    nP = 0
    for r in P.itertuples():
        if r.date in di and r.code in ci:
            i, j = di[r.date], ci[r.code]
            if np.isnan(C[i, j]):
                C[i, j] = r.close; O[i, j] = r.open; V[i, j] = r.value; R[i, j] = r.ret; nP += 1
    T, N = C.shape
    TR = ~np.isnan(C)
    G = np.where(TR, np.nan_to_num(1 + R, nan=1.0), 1.0)
    CF = pd.DataFrame(C).ffill().values
    PC = np.vstack([np.full((1, N), np.nan), CF[:-1]])
    with np.errstate(invalid='ignore', divide='ignore'):
        ADJ = np.where(TR, G * PC / C, 1.0)
    ADJ = np.where(np.isfinite(ADJ), ADJ, 1.0)
    cnt = TR.sum(1); GOOD = cnt >= 300
    gi = np.where(GOOD)[0]
    ex = np.isin(dates[gi], pd.DatetimeIndex(EXEMPT))
    Vg = V[gi]; nanv = np.isnan(Vg); bad = nanv & (~ex[:, None])
    s = pd.DataFrame(np.where(nanv, 0, Vg)).rolling(60, min_periods=60).sum().values
    n = pd.DataFrame((~nanv).astype(float)).rolling(60, min_periods=60).sum().values
    b = pd.DataFrame(bad.astype(float)).rolling(60, min_periods=60).sum().values
    with np.errstate(invalid='ignore', divide='ignore'):
        v60g = np.where((b == 0) & (n > 0), s / n, np.nan)
    Rg = np.where(TR[gi], R[gi], np.nan)
    sd = pd.DataFrame(Rg).rolling(60, min_periods=40).std(ddof=1)
    sdg = sd.ffill().values
    V60 = np.full((T, N), np.nan); SIG = np.full((T, N), np.nan)
    V60[gi] = v60g; SIG[gi] = sdg
    V60 = pd.DataFrame(V60).ffill().values  # 非 GOOD 日沿用前一個 GOOD 日
    SIG = pd.DataFrame(SIG).ffill().values
    elig = (V60 >= 1e7) & (C >= 10)
    key = np.where(elig, V60, -np.inf)
    order = np.argsort(-key, axis=1, kind='stable')
    VR = np.full((T, N), np.nan)
    rows = np.arange(T)[:, None]
    rk = np.empty_like(order); rk[rows, order] = np.arange(1, N + 1)[None, :]
    VR = np.where(elig, rk, np.nan).astype(float)
    # 漲跌停
    with np.errstate(invalid='ignore', divide='ignore'):
        ref = np.where(FL == 1, PC, C / G)
    L = np.where((dates < pd.Timestamp('2015-06-01'))[:, None], 0.07, 0.10)
    up_raw = ref * (1 + L); dn_raw = ref * (1 - L)
    tu = tick_of(np.nan_to_num(up_raw, nan=1)); td = tick_of(np.nan_to_num(dn_raw, nan=1))
    UP = np.floor(up_raw / tu + 1e-9) * tu; DN = np.ceil(dn_raw / td - 1e-9) * td
    # 0050
    e = pd.read_csv(open('/home/user/jonatw/finlab-txd/data/raw/etf_0050.csv'), parse_dates=['date']).set_index('date')['adj']
    E = e.reindex(e.index.union(dates)).ffill().reindex(dates).values
    np.savez(CACHE, dates=dates.values.astype('datetime64[ns]'), codes=np.array(codes), C=C, O=O, V=V, R=R, FL=FL, TR=TR, G=G, CF=CF, PC=PC, ADJ=ADJ,
             GOOD=GOOD, V60=V60, SIG=SIG, VR=VR, UP=UP, DN=DN, E=E, nP=nP)
    return nP

def disposal_v4():
    """V4 凍結官方處置名單：disposal_hist_rejudge in_panel ∪ disposal_5y 獨有 (code,start) 列"""
    a = pd.read_csv(open(AG + 'o_disposal_rejudge/disposal_hist_rejudge.csv'), dtype={'code': str})
    a = a[a['in_panel'].astype(str) == 'True'][['code', 'announce', 'start', 'end']]
    b = pd.read_csv(open('/home/user/stevenhsiehsheng-ctrl/tw-stock-screener/data/extras/disposal_5y.csv.gz', 'rb'), compression='gzip', dtype={'code': str})
    b = b.rename(columns={'date': 'announce'})[['code', 'announce', 'start', 'end']]
    k = set(zip(a.code, a.start))
    b = b[[(c, s) not in k for c, s in zip(b.code, b.start)]]
    u = pd.concat([a, b], ignore_index=True)
    for c in ['announce', 'start', 'end']: u[c] = pd.to_datetime(u[c])
    return u.dropna(subset=['start', 'end'])

def cpi():
    d = json.load(open(AG + 'o_v7_listing_exec_engine/raw/exec_inputs/cpi_tw.json'))['index_2006_100']
    return {int(k): v for k, v in d.items()}

def _v60_strict(X):
    """凍結後修正 F1：V60 GOOD 日窗內缺值（非豁免日）→ NaN；只有非 GOOD 日沿用前一日；vrank 隨之重算（收盤條件不變：當日 close_raw）。"""
    V = X['V']; GOOD = X['GOOD']; dates = X['dates']; T, N = V.shape; gi = np.where(GOOD)[0]
    ex = np.isin(dates[gi], pd.DatetimeIndex(EXEMPT)); Vg = V[gi]; nanv = np.isnan(Vg); bad = nanv & (~ex[:, None])
    s = pd.DataFrame(np.where(nanv, 0, Vg)).rolling(60, min_periods=60).sum().values
    n = pd.DataFrame((~nanv).astype(float)).rolling(60, min_periods=60).sum().values
    b = pd.DataFrame(bad.astype(float)).rolling(60, min_periods=60).sum().values
    with np.errstate(invalid='ignore', divide='ignore'):
        g = np.where((b == 0) & (n > 0), s / n, np.nan)
    V60 = np.full((T, N), np.nan); V60[gi] = g
    for t in np.where(~GOOD)[0]:
        if t > 0: V60[t] = V60[t - 1]
    elig = (V60 >= 1e7) & (X['C'] >= 10)
    key = np.where(elig, V60, -np.inf)
    order = np.argsort(-key, axis=1, kind='stable')
    rows = np.arange(T)[:, None]
    rk = np.empty_like(order); rk[rows, order] = np.arange(1, N + 1)[None, :]
    X['V60'] = V60; X['VR'] = np.where(elig, rk, np.nan).astype(float)
    X['FIX_F1'] = True

_X = None
def load():
    global _X
    if _X is None:
        if not os.path.exists(CACHE): build()
        z = np.load(CACHE, allow_pickle=False)
        _X = {k: z[k] for k in z.files}
        _X['dates'] = pd.DatetimeIndex(_X['dates']); _X['codes'] = [str(c) for c in _X['codes']]
        _v60_strict(_X)  # 凍結後修正 F1
    return _X

if __name__ == '__main__':
    n = build(); X = load()
    print('P cells', n, 'GOOD days', int(X['GOOD'].sum()), 'shape', X['C'].shape)
