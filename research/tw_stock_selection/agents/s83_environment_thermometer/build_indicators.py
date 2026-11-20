#!/usr/bin/env python3
"""S83 第 1 步：算 15 個環境指標（每個日曆月底）與結果變數（現行帳戶對 0050 的超額），並驗收現行格。
python3 -I -B build_indicators.py → indicators_monthly.csv、outcomes_monthly.csv、accept.json
注意：本步只「算」不「看關係」；指標定義寫死在 s83lib.INDICATORS。"""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
import s83lib as E
import s76lib as Q
from s76lib import K, cslib

HERE = Path(__file__).resolve().parent
P, U = K.load_cont()
F_, RT = K.surge(P, 'B2')
sg = K.signal(P, U, RT, 0)
nav, lg, st = K.sim(P, sg, U, 1e6)
D0, DE = P.IDX[P.t0], P.IDX[-1]
s75 = json.load(open(E.AG / 's75_continuous_2008_2026' / 'results.json'))['versions']['V_BRIDGE']
acc = dict(cagr=K.cagr(nav, D0, DE), s75_cagr=s75['full']['cagr'], final_nav=float(nav.iloc[-1]), s75_final_nav=s75['full']['final_nav'], start=str(D0.date()), end=str(DE.date()))
assert abs(acc['final_nav'] - acc['s75_final_nav']) < 1e-6 and abs(acc['cagr'] - 0.2352) < 5e-4
json.dump(acc, open(HERE / 'accept.json', 'w'), ensure_ascii=False, indent=1)
print('驗收', acc, flush=True)

T, N = P.T, P.N
ser = pd.Series(np.arange(T), index=P.IDX)
me = ser.groupby(P.IDX.to_period('M')).last()
me = me[me.index >= pd.Period('2008-01', 'M')]
G, Rd = P.G, P.R


def ret_n(k, n):
    with np.errstate(invalid='ignore', divide='ignore'):
        return G[k] / G[k - n] - 1

# 每日：大漲家數比例、上市成交值合計
with np.errstate(invalid='ignore'):
    big = (Rd >= 0.065) & U
    bigshare = big.sum(1) / np.maximum(U.sum(1), 1)
    amt = np.nansum(np.where(P.MK == 1, P.AMT, np.nan), axis=1)
amt_s = pd.Series(amt)
turn = (amt_s.rolling(20, min_periods=20).mean() / amt_s.rolling(250, min_periods=200).mean()).to_numpy()
e50 = pd.Series(P.E50)
mkt12 = (e50 / e50.shift(252) - 1).to_numpy()
mktvol = (e50.pct_change().rolling(63, min_periods=63).std(ddof=1) * np.sqrt(252)).to_numpy()

# 動能因子月報酬：月底 k 分組 → 下一個月底
kk = ser.groupby(P.IDX.to_period('M')).last()
wml = {}
for a in range(len(kk) - 1):
    k, k2 = int(kk.iloc[a]), int(kk.iloc[a + 1])
    if k < 127:
        continue
    with np.errstate(invalid='ignore', divide='ignore'):
        f = G[k - 21] / G[k - 126] - 1
        r = G[k2] / G[k] - 1
    ok = U[k] & np.isfinite(f) & np.isfinite(r)
    if ok.sum() < 100:
        continue
    q = np.quantile(f[ok], [0.2, 0.8])
    wml[kk.index[a + 1]] = float(r[ok & (f >= q[1])].mean() - r[ok & (f <= q[0])].mean())
wml = pd.Series(wml)

S, av, Rv = RT['S'], RT['av'], F_['Rv']
yoy = (Rv / Rv.shift(12) - 1).replace([np.inf, -np.inf], np.nan)
yoy[yoy.index.year == 2013] = np.nan
avs = sorted([(int(v), m) for m, v in av.items() if int(v) < T and m in S.index and np.isfinite(S.loc[m].to_numpy()).sum() > 0])

rows = []
for m, k in me.items():
    k = int(k); u = U[k]; r = dict(month=str(m), date=str(P.IDX[k].date()), k=k, n_univ=int(u.sum()))
    r3, r6, r1 = ret_n(k, 63), ret_n(k, 126), ret_n(k, 21)
    v = u & np.isfinite(r3); r['SPURT3'] = float((r3[v] > 0.5).mean())
    v = u & np.isfinite(r6); r['SPURT6'] = float((r6[v] > 0.5).mean())
    v = u & np.isfinite(r1); r['XSD21'] = float(r1[v].std(ddof=1)); r['TAIL21'] = float(np.quantile(r1[v], 0.95) - np.median(r1[v]))
    r['BIGUP21'] = float(bigshare[k - 20:k + 1].mean())
    v = u & np.isfinite(r3) & np.isfinite(P.V60[k]); vv = P.V60[k][v]; q50, q90 = np.quantile(vv, [0.5, 0.9])
    r['SMALLREL63'] = float(r3[v][vv <= q50].mean() - r3[v][vv >= q90].mean())
    r['TURN20_250'] = float(turn[k]); r['BREADTH60'] = float(P.F_MA60[k][u].mean())
    r['MKT12'] = float(mkt12[k]); r['MKTVOL63'] = float(mktvol[k])
    cand = [(k0, mm) for k0, mm in avs if k0 <= k]
    r['NSURGE3'] = r['TOP10YOY'] = r['REVBREADTH'] = np.nan; r['n_surge_ge3'] = np.nan; r['rev_month'] = ''
    if cand:
        k0, mm = cand[-1]
        s = cslib.ins15_mask_row(P, mm, k0, S.loc[mm].to_numpy())
        fin = u & np.isfinite(s)
        if fin.sum() >= 50:
            r['rev_month'] = str(mm); r['n_surge_ge3'] = int((s[fin] >= 3).sum()); r['NSURGE3'] = float((s[fin] >= 3).mean())
            y = yoy.loc[mm].to_numpy()
            top = np.flatnonzero(fin)[np.argsort(-s[fin], kind='stable')][:10]
            if np.isfinite(y[top]).sum() >= 5:
                r['TOP10YOY'] = float(np.nanmedian(y[top]))
            vy = u & np.isfinite(y)
            if vy.sum() >= 50:
                r['REVBREADTH'] = float((y[vy] > 0.2).mean())
    w = wml[wml.index <= m].tail(3)
    r['WML3'] = float(w.sum()) if len(w) == 3 and w.index[-1] == m else np.nan
    rows.append(r)
I = pd.DataFrame(rows).set_index('month')

# 結果變數：日曆月（月底對月底；第一個月以 2008-01-02 為基底）
b = pd.Series(P.E50, index=P.IDX).loc[D0:]
nm = nav.groupby(nav.index.to_period('M')).last(); bm = b.groupby(b.index.to_period('M')).last()
ln = np.log(pd.concat([pd.Series([nav.iloc[0]]), nm]).to_numpy()); lb = np.log(pd.concat([pd.Series([b.iloc[0]]), bm]).to_numpy())
n = len(nm)
def cum(a, i, j):   # 第 i 個月初到第 j 個月底（含）的累積報酬；月份編號 0…n−1
    return np.exp(a[j + 1] - a[i]) - 1
O = pd.DataFrame(index=[str(x) for x in nm.index])
O['strat'] = [cum(ln, i, i) for i in range(n)]; O['e0050'] = [cum(lb, i, i) for i in range(n)]
O['sync1'] = O['strat'] - O['e0050']
O['sync3'] = [cum(ln, i - 2, i) - cum(lb, i - 2, i) if i >= 2 else np.nan for i in range(n)]
for h in (1, 3, 6):
    O[f'lead{h}'] = [cum(ln, i + 1, i + h) - cum(lb, i + 1, i + h) if i + h < n else np.nan for i in range(n)]
I['EX6'] = [cum(ln, i - 5, i) - cum(lb, i - 5, i) if i >= 5 else np.nan for i in range(n)]
# 次要結果：月底後第一個訊號期的 raw10 − 0050（S76 signal_power_monthly）
M = pd.read_csv(E.AG / 's76_early_vs_late_diagnosis' / 'signal_power_monthly.csv')
ksig = M['k'].to_numpy()
s1, s3 = [], []
for k in I['k'].to_numpy():
    j = np.searchsorted(ksig, k, side='left')    # 訊號日 ≥ 月底日
    if j < len(M):
        s1.append(M['raw10'].iloc[j] - M['b0050'].iloc[j]); s3.append(M['raw10_3'].iloc[j] - M['b0050_3'].iloc[j])
    else:
        s1.append(np.nan); s3.append(np.nan)
O['sig_lead1'] = s1; O['sig_lead3'] = s3
O['stockfrac'] = st['stockfrac'].groupby(st['stockfrac'].index.to_period('M')).mean().reindex(nm.index).to_numpy()
assert list(I.index) == list(O.index)
I.to_csv(HERE / 'indicators_monthly.csv'); O.to_csv(HERE / 'outcomes_monthly.csv')
print(I.shape, O.shape, I[E.IND].notna().sum().to_dict())
