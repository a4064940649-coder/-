#!/usr/bin/env python3
"""P3 挑選期版：安聯月超額 對 M3 因子（EW−0050、ELEC、C1−EW、C2−EW）＋第二腿（對同池等權）回歸，2006-08～2013-11。
因子與基金月報酬取自 winners 研究的 allianz_monthly.csv（訊號日窗）；第二腿用本資料夾自算淨值。"""
import sys, json, pickle
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import numpy as np, pandas as pd
AM = Path('/tmp/claude-0/-home-user--/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/winners_attr/allianz_monthly.csv')
a = pd.read_csv(AM, index_col=0, parse_dates=True)
LAG = int(sys.argv[1]) if len(sys.argv) > 1 else 3   # Newey-West（Bartlett）落後 3 期；-1＝一般 OLS
def ols(y, X, lag=None):
    lag = LAG if lag is None else lag
    X = np.column_stack([np.ones(len(y)), X]); b, *_ = np.linalg.lstsq(X, y, rcond=None)
    e = y - X @ b; XtXi = np.linalg.inv(X.T @ X)
    if lag < 0:
        s2 = e @ e / (len(y) - X.shape[1]); V = s2 * XtXi
    else:
        u = X * e[:, None]; S = u.T @ u
        for l in range(1, lag + 1):
            w = 1 - l / (lag + 1); G = u[l:].T @ u[:-l]; S += w * (G + G.T)
        V = XtXi @ S @ XtXi
    return b, b / np.sqrt(np.diag(V))
def m3(df):
    y = (df['FUND'] - df['0050']).to_numpy()
    X = np.column_stack([df['EW'] - df['0050'], df['ELEC'], df['C1'] - df['EW'], df['C2'] - df['EW']])
    return y, X
out = {}
# 檢查 M3 口徑：2006-04～2013-12（93 個月）
s = a[(a.index >= '2006-04-01') & (a.index <= '2013-12-31')]
y, X = m3(s); b, t = ols(y, X); out['check_M3_2006_2013'] = {'n': len(y), 'alpha_m': b[0] * 100, 'alpha_t': t[0], 'betas': list(zip(b[1:].round(3), t[1:].round(2)))}
import rlib as L
z = pickle.load(open(HERE / 'rep_core_both.pkl', 'rb'))
days = L.reb_days('2006-07-01', '2013-12-31'); END = pd.Timestamp('2013-12-31')
WD = days[1:] + [END]; idx = pd.DatetimeIndex(WD[:-1])
we = pd.Series(L.win_ret(z['ew'], WD), idx)
s = a[(a.index >= '2006-08-01') & (a.index <= '2013-11-30')]
y, X = m3(s); b, t = ols(y, X); out['M3_88m'] = {'n': len(y), 'alpha_m': b[0] * 100, 'alpha_t': t[0]}
for k, key in [('n30', ('leg2', 'normal')), ('n10', ('leg2_n10', 'normal'))]:
    f = pd.Series(L.win_ret(z['navs'][key], WD), idx) - we
    f = f.reindex(s.index)
    assert f.notna().all(), 'window dates misaligned'
    b, t = ols(y, np.column_stack([X, f.to_numpy()]))
    out[f'M3+leg2_{k}'] = {'loading': b[-1], 't': t[-1], 'alpha_m': b[0] * 100, 'alpha_t': t[0]}
json.dump(out, open(HERE / 'rep_p3.json', 'w'), indent=1, default=float)
for k, v in out.items(): print(k, v)
