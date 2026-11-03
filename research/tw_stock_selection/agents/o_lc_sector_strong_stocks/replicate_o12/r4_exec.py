#!/usr/bin/env python3
"""簡化可執行引擎（自寫近似，非移植 o8 s7_exec）：
  - e 開盤成交；整股（1000 股）＋零股（餘數），各算一筆委託；手續費 0.1425%×0.28、每筆最低 20 元；賣出證交稅 0.3%。
  - 衝擊（每邊）：max(0.1%＋0.5·σ60·√(Q/V60), 一跳/開盤價)；零股部分再加一跳/開盤價（近似零股價差）。
  - 參與上限 Q ≤ 20%×V60（V60＝60 日均成交值，近似 OA）；不交易帶 0.25%（目標與現有權重差 < 0.25% 且兩者皆 >0 時不動）。
  - 壓力：每邊再加 0.5%。
  - GROW：資金從第一個窗起以淨值滾動；FIX：每窗資金重設（只影響規模相關成本）。
"""
import sys, pickle
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
import rlib as R
OUT = Path(__file__).resolve().parent / 'out'
FEE = 0.001425 * 0.28; TAX = 0.003; BAND = 0.0025; PART = 0.2
# σ60、V60
with np.errstate(invalid='ignore', divide='ignore'):
    _g = R.AFF[1:] / R.AFF[:-1] - 1
_g = np.vstack([np.full((1, R.N), np.nan), _g]); _g[~R.TRADED] = np.nan
SIG60 = pd.DataFrame(_g).rolling(60, min_periods=40).std().ffill().to_numpy()
V60 = pd.DataFrame(np.where(R.TRADED, R.VAL, np.nan)).rolling(60, min_periods=1).mean().ffill().to_numpy()
LAST_TRADE = np.where(R.TRADED.any(axis=0), len(R.IDX) - 1 - np.argmax(R.TRADED[::-1], axis=0), -1)
def tick(p):
    return np.select([p < 10, p < 50, p < 100, p < 500, p < 1000], [0.01, 0.05, 0.1, 0.5, 1.0], 5.0)

def costs(Q, P, i_idx, e, extra):
    """Q：交易金額（>0），P：開盤價。回傳每檔成本（不含稅）。"""
    if len(Q) == 0: return np.zeros(0)
    sh = Q / P
    lots = np.floor(sh / 1000) * 1000; odd = np.floor(sh - lots)
    qb = lots * P; qo = odd * P
    sig = np.nan_to_num(SIG60[e - 1, i_idx], nan=0.02); v = V60[e - 1, i_idx]
    v = np.where(np.isfinite(v) & (v > 0), v, 1e9)
    tk = tick(P) / P
    imp = np.maximum(0.001 + 0.5 * sig * np.sqrt(Q / v), tk) + extra
    c = imp * (qb + qo) + tk * qo
    c += np.where(qb > 0, np.maximum(20, FEE * qb), 0) + np.where(qo > 0, np.maximum(20, FEE * qo), 0)
    return c

def run(days, W, cap0, mode='GROW', stress=False):
    extra = 0.005 if stress else 0.0
    tend = len(R.IDX) - 1
    H = np.zeros(R.N); cash = cap0
    rets = []; cost_frac = []; odd_share = []
    for i, t in enumerate(days):
        t1 = days[i + 1] if i + 1 < len(days) else tend; e = t + 1
        V0 = H.sum() + cash
        if mode == 'FIX' and V0 > 0:
            H *= cap0 / V0; cash *= cap0 / V0; V0 = cap0
        with np.errstate(invalid='ignore', divide='ignore'):
            g = R.AOPEN[e] / R.AFF[t]
        g = np.where(np.isfinite(g), g, 1.0)
        H = H * g
        # 下市（之後再無成交）→ 以最後價格轉現金
        dead = (LAST_TRADE < e) & (H > 0)
        if dead.any(): cash += H[dead].sum(); H[dead] = 0.0
        V1 = H.sum() + cash
        w = W[i]
        T = w * V1
        P = R.On[e]
        can = np.isfinite(P) & (P > 0) & np.isfinite(R.AOPEN[e])
        dlt = T - H
        band = (np.abs(dlt) < BAND * V1) & (T > 0) & (H > 0)
        dlt = np.where(can & ~band, dlt, 0.0)
        # 參與上限
        lim = PART * np.nan_to_num(V60[e - 1], nan=np.inf)   # 無成交值歷史（新上市）→ 不設上限
        dlt = np.clip(dlt, -np.maximum(lim, 0), np.maximum(lim, 0))
        dlt = np.where(H + dlt < 1e-6, -H, dlt)
        sells = np.where(dlt < 0)[0]; buys = np.where(dlt > 0)[0]
        cs = costs(-dlt[sells], P[sells], sells, e, extra); cs_tax = TAX * (-dlt[sells])
        cash += -dlt[sells].sum() - cs.sum() - cs_tax.sum()
        # 買進以現金為限（先算成本，按比例縮）
        qb = dlt[buys]; cb = costs(qb, P[buys], buys, e, extra)
        need = qb.sum() + cb.sum()
        if need > cash and need > 0:
            sc = max(cash, 0) / need; qb = qb * sc; cb = costs(qb, P[buys], buys, e, extra) if len(qb) else cb
        cash -= qb.sum() + cb.sum()
        H[sells] += dlt[sells]; H[buys] += qb
        H = np.maximum(H, 0)
        tot_c = cs.sum() + cs_tax.sum() + cb.sum()
        cost_frac.append(tot_c / V1)
        # 零股占成交額
        Qall = np.concatenate([-dlt[sells], qb]); Pall = np.concatenate([P[sells], P[buys]])
        if Qall.sum() > 0:
            sh = Qall / Pall; lots = np.floor(sh / 1000) * 1000
            odd_share.append(((sh - lots) * Pall).sum() / Qall.sum())
        else:
            odd_share.append(np.nan)
        with np.errstate(invalid='ignore', divide='ignore'):
            base = np.where(np.isfinite(R.AOPEN[e]), R.AOPEN[e], R.AFF[e]); G = R.AFF[t1] / base
        G = np.where(np.isfinite(G), G, 1.0)
        H = H * G
        V3 = H.sum() + cash
        rets.append(V3 / V0 - 1)
    return np.array(rets), np.array(cost_frac), np.array(odd_share)

def dense(P):
    w = np.zeros(R.N)
    for k, v in P.items(): w[k] = v
    return w

def main(j, start='2006-04-01'):
    D = pickle.load(open(OUT / f'plans_J{j}.pkl', 'rb'))
    days = D['days']; plans = [p for _, p in D['plans']]
    keep = [k for k, t in enumerate(days) if R.IDX[t] >= pd.Timestamp(start)]
    days = [days[k] for k in keep]; plans = [plans[k] for k in keep]
    B = [dense(p['B']) for p in plans]
    out = {}
    specs = {'B': B}
    for key, nm in (('IND_n2_cap_c15', 'SEL'), ('CTRL_n2_cap_c15', 'CTRL')):
        X = [dense(p[key]) for p in plans]
        specs[nm] = [0.7 * b + 0.3 * x if x.sum() > 0 else b for b, x in zip(B, X)]
        specs[nm + '_PURE'] = X
    runs = [('100萬', 1e6, 'GROW', False), ('100萬', 1e6, 'GROW', True), ('30萬', 3e5, 'FIX', False), ('300萬', 3e6, 'GROW', False), ('1000萬', 1e7, 'GROW', False)]
    for nm, Wl in specs.items():
        for lab, c0, mode, st in runs:
            if nm != 'SEL' and lab != '100萬': continue
            r, cf, od = run(days, Wl, c0, mode, st)
            k = f'{nm}|{lab}|{mode}|{"stress" if st else "normal"}'
            out[k] = r
            if not st: out[k + '|cost'] = cf; out[k + '|odd'] = od
    df = pd.DataFrame(out, index=pd.DatetimeIndex([R.IDX[t] for t in days], name='d'))
    df.to_parquet(OUT / f'exec_J{j}{"" if start.startswith("2006") else "_s" + start[:4]}.parquet')
    return df

if __name__ == '__main__':
    j = int(sys.argv[1]); start = sys.argv[2] if len(sys.argv) > 2 else '2006-04-01'
    df = main(j, start); print(j, start, df.shape, flush=True)
