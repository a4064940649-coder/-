import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import json, math
import numpy as np, pandas as pd
from scipy.stats import spearmanr, pearsonr
import r_lib as L
K, IDX, COLS, N = L.K, L.IDX, L.COLS, L.N
D = L.build()
SD = D['sig_days']
SB = pd.read_parquet(L.HERE / 'my_sbl.parquet').reindex(columns=COLS)
SH = pd.read_parquet(L.HERE / 'my_short.parquet').reindex(columns=COLS) * 1000.0   # 張→股
ANC = L.ANC
# 2026-10-12 訊號日的成交日（作為 2026-09 窗的終點開盤）
ALLSIG = K.rebalance_days('2021-10-01')
def exec_of(d): return IDX.get_loc(d) + 1

def incr(P, d, lag=20):
    i = P.index.searchsorted(d) - 1
    if i - lag < 0: return None
    a, b = P.iloc[i].to_numpy(), P.iloc[i - lag].to_numpy()
    return (a - b) / ANC, (P.index[i], P.index[i - lag])

def oo(t, t2):
    r = K.Cn[t] / K.On[t] * np.prod(K.GROW[t + 1:t2 + 1], axis=0)
    tr = K.TRADED[t2] & np.isfinite(K.On[t2])
    return np.where(tr, r * K.On[t2] / np.where(tr, K.Cn[t2], 1.0), np.nan) - 1

def top_minus_rest(x, r, m):
    idx = np.where(m & np.isfinite(x) & np.isfinite(r))[0]
    rk = pd.Series(x[idx]).rank(pct=True, method='first').to_numpy()
    top = rk > 0.8
    return float(r[idx][top].mean() - r[idx][~top].mean()), int(top.sum()), len(idx)

R = {}
sigs = [d for d in ALLSIG if d >= pd.Timestamp('2024-01-01')]
rows = []
for k, d in enumerate(sigs[:-1]):
    if d not in D['cs'] and d < pd.Timestamp('2026-10-01'): continue
    c = D['cs'].get(d)
    if c is None: continue
    t, t2 = c['e'], exec_of(sigs[k + 1])
    if t2 >= len(IDX): continue
    r = oo(t, t2)
    o = dict(d=str(d.date()))
    sh = incr(SH, d)
    if sh is not None:
        o['short'], o['n_top'], o['n'] = top_minus_rest(sh[0], r, c['el']); o['short_dates'] = [str(x.date()) for x in sh[1]]
    sb = incr(SB, d)
    if sb is not None and sb[1][1] >= SB.index[0]:
        o['sbl'], _, o['n_sbl'] = top_minus_rest(sb[0], r, c['el']); o['sbl_dates'] = [str(x.date()) for x in sb[1]]
        m = c['el'] & np.isfinite(sb[0]) & np.isfinite(sh[0])
        o['corr_pearson'] = float(pearsonr(sb[0][m], sh[0][m])[0]); o['corr_spearman'] = float(spearmanr(sb[0][m], sh[0][m]).correlation)
    rows.append(o)
T = pd.DataFrame(rows).set_index('d')
def tstat(x):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    return float(x.mean() / (x.std(ddof=1) / math.sqrt(len(x))))
R['rows'] = T.to_dict(orient='index')
for lab, (a, b) in {'2024-02~2026-09': ('2024-01-01', '2026-09-30'), '2024-03~2026-09': ('2024-03-01', '2026-09-30'),
                    '2024-02~2026-08': ('2024-01-01', '2026-08-31'), '2024-04~2026-09': ('2024-04-01', '2026-09-30')}.items():
    x = T.loc[a:b, 'short'].dropna()
    R[f'short|{lab}'] = dict(n=len(x), mean=round(x.mean() * 100, 3), t=round(tstat(x), 2))
(L.HERE / 'my_h3a.json').write_text(json.dumps(R, ensure_ascii=False, indent=1, default=str))
print(T.drop(columns=[c for c in T.columns if c.endswith('dates')]).to_string())
for k, v in R.items():
    if k != 'rows': print(k, v)
