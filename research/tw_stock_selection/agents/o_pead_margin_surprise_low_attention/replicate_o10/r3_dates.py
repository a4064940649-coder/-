"""市場可得日 A：BWIBBU 財報季標記 → ttm/bv 切換（取較晚）→ 單一 → 法定期限。"""
from pathlib import Path
import json
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
D = HERE / 'data'
P = np.load(D / 'panel.npz', allow_pickle=True)
dates = P['dates']; codes = list(P['codes']); close = P['close']
cidx = {c: i for i, c in enumerate(codes)}
dt = pd.to_datetime(dates)

per = pd.read_parquet(D / 'per.parquet')
per = per[per.stock_id.isin(cidx)]
per = per.drop_duplicates(['date', 'stock_id'], keep='last')
fin = pd.read_parquet(D / 'fin.parquet')
fin = fin[fin.y >= 2014]

DL = {1: (5, 15), 2: (8, 14), 3: (11, 14)}


def deadline(y, q):
    if q == 4:
        return pd.Timestamp(y + 1, 3, 31)
    m, d = DL[q]
    return pd.Timestamp(y, m, d)


def qend(y, q):
    return pd.Timestamp(y, 3 * q, 1) + pd.offsets.MonthEnd(0)


def regime_switch(lo, hi, start_ok=None):
    """逐日比較：第 t 日區間與前一有效日區間不重疊，且下一有效日也不與前一日重疊 → 切換日 t。"""
    v = np.where(np.isfinite(lo) & np.isfinite(hi))[0]
    if len(v) < 2:
        return None, None
    for k in range(1, len(v)):
        p, t = v[k - 1], v[k]
        if hi[t] < lo[p] or lo[t] > hi[p]:
            if k + 1 < len(v):
                u = v[k + 1]
                if not (hi[u] < lo[p] or lo[u] > hi[p]):
                    continue
            return t, v[0]
    return None, None


rows = []
fin = fin.sort_values(['stock_id', 'qi'])
eps_map = fin.set_index(['stock_id', 'qi']).eps.to_dict()
for sid, x in per.groupby('stock_id'):
    j = cidx[sid]
    x = x.set_index(pd.to_datetime(x.date)).sort_index()
    x = x[~x.index.duplicated()]
    cl = pd.Series(close[:, j], index=dt).reindex(x.index)
    c = x.close.where(x.close > 0, cl).values
    pe = x.per.values; pb = x.pbr.values
    with np.errstate(divide='ignore', invalid='ignore'):
        Elo = np.where(pe > 0.01, c / (pe + 0.005), np.nan); Ehi = np.where(pe > 0.01, c / (pe - 0.005), np.nan)
        Blo = np.where(pb > 0.01, c / (pb + 0.005), np.nan); Bhi = np.where(pb > 0.01, c / (pb - 0.005), np.nan)
    fq = x.fs_quarter.values; mk = x.mkt.values[-1]
    idx = x.index
    for _, r in fin[fin.stock_id == sid].iterrows():
        y, q = int(r.y), int(r.q)
        dl = deadline(y, q)
        w0, w1 = qend(y, q) + pd.Timedelta(days=14), dl + pd.Timedelta(days=30)
        a, b = idx.searchsorted(w0), idx.searchsorted(w1, side='right')
        if b - a < 3:
            continue
        rec = dict(stock_id=sid, qi=int(r.qi), period=r.period, mkt=mk)
        # 1. BWIBBU 財報季標記（上市，2017-04 起）
        tag = f'{y - 1911}/{q}'
        d_flag = None
        if mk == 'twse' and w0 >= pd.Timestamp('2017-04-01') or (mk == 'twse' and idx[a] >= pd.Timestamp('2017-04-13')):
            hit = np.where(fq[a:b] == tag)[0]
            if len(hit) and hit[0] > 0:
                d_flag = idx[a + hit[0]]
        rec['d_flag'] = d_flag
        # 2. ttm 切換
        t, t0 = regime_switch(Elo[a:b], Ehi[a:b])
        d_ttm = idx[a + t] if t is not None else None
        # 開窗即已切換：起始區間吻合新 ttm、不吻合舊 ttm
        if d_ttm is not None or True:
            e = [eps_map.get((sid, r.qi - k)) for k in range(5)]
            if all(v is not None and np.isfinite(v) for v in e):
                new, old = sum(e[:4]), sum(e[1:5])
                if t0 is not None and abs(new - old) > 1e-6:
                    lo0, hi0 = Elo[a + t0], Ehi[a + t0]
                    if lo0 - 0.006 <= new <= hi0 + 0.006 and not (lo0 - 0.006 <= old <= hi0 + 0.006):
                        d_ttm = None
        t, _ = regime_switch(Blo[a:b], Bhi[a:b])
        d_bv = idx[a + t] if t is not None else None
        rec['d_ttm'] = d_ttm; rec['d_bv'] = d_bv
        if d_flag is not None:
            A, src = d_flag, 'flag'
        elif d_ttm is not None and d_bv is not None:
            A, src = max(d_ttm, d_bv), 'both'
        elif d_ttm is not None:
            A, src = d_ttm, 'ttm'
        elif d_bv is not None:
            A, src = d_bv, 'bv'
        else:
            A, src = dl, 'deadline'
        rec['A_raw'] = A; rec['src'] = src; rec['deadline'] = dl
        rows.append(rec)
av = pd.DataFrame(rows)
# A 對齊到面板交易日（≤ A 的最後一個交易日）
pos = dt.searchsorted(pd.to_datetime(av.A_raw), side='right') - 1
av['A'] = dt[pos]; av['A_pos'] = pos
av.to_parquet(D / 'avail.parquet', index=False)
# 閘門：ttm/bv 與 flag 的交易日差
tpos = lambda s: dt.searchsorted(pd.to_datetime(s))
res = {}
for m in ['d_ttm', 'd_bv']:
    z = av[av.d_flag.notna() & av[m].notna()]
    diff = tpos(z[m]) - tpos(z.d_flag)
    res[m] = dict(n=int(len(z)), within1=float(np.mean(np.abs(diff) <= 1)), early=float(np.mean(diff <= -1)))
res['src'] = av.src.value_counts().to_dict()
print(json.dumps(res, indent=1, default=str))
json.dump(res, open(D / 'avail_gate.json', 'w'), indent=1, default=str)
