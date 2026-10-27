"""o_index_deletion_flow 共用函式：事件載入、賣壓強度、事件報酬（對上市等權）、分群 t。"""
import os, json, math, hashlib
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
P = '/tmp/claude-0/-home-user--/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/panel/'
SRC3 = (pd.Timestamp('2024-02-15'), pd.Timestamp('2025-09-23'))
SEL_END = pd.Timestamp('2018-12-31')
EVENTS_SHA = 'b19b78f2d36d2a7de29e78d99aa8ced1fed24fc5eae7c309b72cf6a6aa705759'
# 事前設定：各類別「典型被動賣出金額」（NT$）。依指數成分股市值落在門檻附近、被動持有比例的粗估（見 explore.json 說明）。
K_CLASS = {'MSCI_STD_DEL': 5e8, 'TW50_DEL': 3e8, 'HDETF_DEL': 3e8, 'MSCI_SC_DEL': 5e7}
HD = {'0056', '00878', '00919', '00713', '00918', '00929', '00940'}

_D = {}
def panel():
    if not _D:
        for k in ['adj', 'close_raw', 'open', 'value', 'ret', 'src']:
            _D[k] = pd.read_parquet(P + f'{k}.parquet')
        A = _D['adj']; C = _D['close_raw']
        _D['adj_open'] = _D['open'] * (A / C)
        r = _D['ret'].clip(-0.5, 0.5)
        _D['ew'] = r.mean(axis=1)                       # 上市等權日報酬
        _D['v60'] = _D['value'].rolling(60, min_periods=40).mean()
        _D['v20'] = _D['value'].rolling(20, min_periods=15).mean()
    return _D

def load_events(check=True):
    f = os.path.join(HERE, 'events_v2.csv')
    if check:
        assert hashlib.sha256(open(f, 'rb').read()).hexdigest() == EVENTS_SHA, 'events_v2.csv 與凍結版不同'
    ev = pd.read_csv(f, dtype={'code': str})
    ev['eff'] = pd.to_datetime(ev.eff)
    ev['cls'] = np.where(ev.type.isin(['MSCI_STD_DEL', 'MSCI_SC_DEL', 'TW50_DEL']), ev.type,
                np.where(ev.type.str.endswith('_DEL') & ev['index'].isin(HD), 'HDETF_DEL', ev.type))
    ev['is_del'] = ev.cls.isin(K_CLASS.keys())
    return ev

def limit_pct(d):
    return 0.065 if d < pd.Timestamp('2015-06-01') else 0.095

def event_table(ev, holds=(5, 10, 20), entries=('closeE', 'openE1')):
    """每筆事件：進場可行性、流動性、賣壓強度、各持有期與進場方式的報酬與上市等權報酬。"""
    D = panel(); A = D['adj'].ffill(limit=30); AO = D['adj_open']; C = D['close_raw']; O = D['open']
    idx = A.index; out = []
    allA = A.to_numpy(); allAO = AO.to_numpy()
    for r in ev.itertuples():
        rec = {'event_id': r.event_id}
        if r.code not in A.columns or r.eff not in idx: rec['ok'] = False; rec['why'] = 'not_in_panel'; out.append(rec); continue
        j = A.columns.get_loc(r.code); i = idx.get_loc(r.eff)
        c_e, c_p = C.iat[i, j], C.iat[i - 1, j]
        v60 = D['v60'].iat[i - 1, j]; v20 = D['v20'].iat[i - 1, j]
        rec.update(v60=v60, v20=v20, px=c_p, src=D['src'].iat[i, j],
                   ret_E=(A.iat[i, j] / A.iat[i - 1, j] - 1) if np.isfinite(A.iat[i, j]) else np.nan,
                   ew_E=D['ew'].iat[i], vol_ratio=D['value'].iat[i, j] / D['value'].iloc[max(0, i - 25):i - 5, j].median())
        liq = np.isfinite(v60) and v60 >= 1e7 and np.isfinite(c_p) and c_p >= 10
        traded = np.isfinite(c_e) and D['value'].iat[i, j] > 0
        lockup = traded and np.isfinite(c_p) and c_e >= c_p * (1 + limit_pct(r.eff))
        rec['ok'] = bool(liq and traded); rec['why'] = '' if rec['ok'] else ('illiquid' if not liq else 'no_trade')
        rec['lockup_E'] = bool(lockup)
        # 賣壓強度（事前）：類別典型被動賣出額 ÷ 20 日均成交值
        rec['pressure'] = K_CLASS.get(r.cls, np.nan) / v20 if (np.isfinite(v20) and v20 > 0) else np.nan
        for h in holds:
            k = i + h
            if k >= len(idx): continue
            # 上市等權買進持有（同窗、同進場方式）
            for en in entries:
                if en == 'closeE':
                    p0 = allA[i]; p1 = allA[k]; s0 = allA[i, j]
                else:
                    if i + 1 > k: continue
                    p0 = allAO[i + 1]; p1 = allA[k]; s0 = allAO[i + 1, j]
                s1 = allA[k, j]
                with np.errstate(invalid='ignore', divide='ignore'):
                    br = p1 / p0 - 1
                bm = np.nanmean(np.clip(br[np.isfinite(br)], -0.9, 5)) if np.isfinite(br).any() else np.nan
                rr = s1 / s0 - 1 if (np.isfinite(s0) and np.isfinite(s1) and s0 > 0) else np.nan
                if en == 'openE1':
                    o1, cp1 = O.iat[i + 1, j], C.iat[i, j]
                    if np.isfinite(o1) and np.isfinite(cp1) and o1 >= cp1 * (1 + limit_pct(r.eff)): rr = np.nan  # 開盤漲停買不到
                elif lockup: rr = np.nan
                rec[f'r_{en}_{h}'] = rr; rec[f'b_{en}_{h}'] = bm; rec[f'x_{en}_{h}'] = rr - bm if np.isfinite(rr) else np.nan
        out.append(rec)
    T = pd.DataFrame(out)
    return ev.merge(T, on='event_id', how='left')

def clus_t(x, g):
    """按事件月分群的 t：先算每個事件月的平均，再對月平均做 t。"""
    s = pd.Series(np.asarray(x, float), index=np.asarray(g)).dropna()
    m = s.groupby(level=0).mean()
    if len(m) < 3 or m.std(ddof=1) == 0: return float('nan'), int(len(m))
    return round(float(m.mean() / (m.std(ddof=1) / math.sqrt(len(m)))), 2), int(len(m))

def stats(x, g, cost=0.0):
    x = pd.Series(np.asarray(x, float) - cost); g = pd.Series(np.asarray(g))
    ok = x.notna(); x, g = x[ok], g[ok]
    if len(x) == 0: return {'n': 0}
    t, nm = clus_t(x.values, g.values)
    q = x.quantile(0.95); xt = x[x <= q]
    return {'n': int(len(x)), 'n_months': nm, 'mean': round(float(x.mean()) * 100, 2), 'median': round(float(x.median()) * 100, 2),
            't_month': t, 'win': round(float((x > 0).mean()), 3), 'mean_ex_top5pct': round(float(xt.mean()) * 100, 2)}
