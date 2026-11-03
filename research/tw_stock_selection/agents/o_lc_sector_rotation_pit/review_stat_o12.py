#!/usr/bin/env python3
"""反方審查（資料與統計）o12：只讀 data/oos_windows.parquet 與 results.json，不改研究員檔案。python3 -I -B。"""
import json, math
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
W = pd.read_parquet(HERE / 'data' / 'oos_windows.parquet')
SEL = 'H52|k3|A15|M'; DEF = 'COMP|k5|A30|M'

def t(x):
    x = pd.Series(x).dropna(); return float(x.mean() / (x.std(ddof=1) / math.sqrt(len(x)))) if len(x) > 2 else np.nan
def nw(x, lag):
    x = np.asarray(pd.Series(x).dropna(), float); n = len(x); e = x - x.mean(); s = (e * e).sum() / n
    for i in range(1, lag + 1): s += 2 * (1 - i / (lag + 1)) * (e[i:] * e[:-i]).sum() / n
    return float(x.mean() / math.sqrt(s / n))

def ex(cfg, cost, bench, j):
    a = W[f'{cfg}|{cost}|J{j}'].dropna()
    b = W[f'0050|J{j}'] if bench == '0050' else W[f'B|{cost}|J{j}']
    x = (a - b.reindex(a.index)).dropna()
    # 以錨定月對齊（訊號日 → 錨定月：訊號日在 D_m 之後，D_m 為當月 10 日，j≤12 交易日仍在同月或次月初）
    return x

def anchor_frame(cfg, cost, bench):
    cols = {}
    for j in range(13):
        x = ex(cfg, cost, bench, j)
        # 訊號日屬於 D_m 所在月；j 大時可能跨月，用（日期−10 日）的月份近似錨定月，並去重
        m = (x.index - pd.Timedelta(days=9)).to_period('M')
        s = pd.Series(x.values, index=m); s = s[~s.index.duplicated()]
        cols[j] = s
    return pd.DataFrame(cols)

out = {}
P = dict(VAL=('2017-01', '2026-10'), FORMAL=('2014-01', '2026-10'), Y19=('2019-01', '2026-10'), SEL=('2014-01', '2016-12'),
         OOS1718=('2017-01', '2018-12'), SEG2=('2014-01', '2018-12'))
for cfg in (SEL, DEF):
    for cost in ('normal', 'zero'):
        for bench in ('0050', 'B'):
            F = anchor_frame(cfg, cost, bench)
            for pn, (a, b) in P.items():
                G = F.loc[pd.Period(a, 'M'):pd.Period(b, 'M')]
                avg = G.mean(axis=1)                  # j 平均路徑（13 條高度重疊，等於一條平滑序列）
                tj = [t(G[j]) for j in G.columns]
                mj = [G[j].mean() * 100 for j in G.columns]
                yr = avg.groupby(avg.index.year).sum() * 100
                drop = {}
                for kk in (1, 2, 3):
                    best = yr.sort_values(ascending=False).index[:kk]
                    drop[f'drop_best{kk}yr_t'] = t(avg[~avg.index.year.isin(best)])
                # 區塊 bootstrap（12 個月一塊）單尾 p(mean ≤ 0)
                rng = np.random.default_rng(12); v = avg.dropna().to_numpy(); n = len(v); bl = 12; bs = []
                for _ in range(4000):
                    st = rng.integers(0, n - bl + 1, size=math.ceil(n / bl)); idx = np.concatenate([np.arange(s, s + bl) for s in st])[:n]
                    bs.append(v[idx].mean())
                bs = np.array(bs)
                out[f'{cfg}|{cost}|vs{bench}|{pn}'] = dict(
                    n=int(G.shape[0]), mean_pp_jmed=float(np.median(mj)), t_jmed=float(np.median(tj)), t_min=float(np.min(tj)), t_max=float(np.max(tj)),
                    n_j_pos=int(sum(m > 0 for m in mj)), mean_pp_javg=float(avg.mean() * 100), t_javg=t(avg), t_javg_NW6=nw(avg, 6),
                    p_boot_le0=float((bs <= 0).mean() if v.mean() > 0 else (bs >= 0).mean()),
                    yearly_pp=yr.round(2).to_dict(), **drop,
                    share_of_total_from_best_yr=float(yr.max() / yr.sum()) if yr.sum() > 0 else None)

# B vs 0050 結構差：每 j
bj = []
for j in range(13):
    x = (W[f'B|normal|J{j}'] - W[f'0050|J{j}']).dropna(); x = x[x.index >= '2017-01-01']; bj.append((x.mean() * 100, t(x)))
out['B_vs0050_VAL_jmed'] = dict(mean_pp=float(np.median([a for a, _ in bj])), t=float(np.median([b for _, b in bj])))

# 選定格 vs 0050 的 beta、MDD 比較（j 平均月窗報酬複利；不同期間）
def mdd(r):
    v = np.cumprod(1 + r.to_numpy()); return float((v / np.maximum.accumulate(v) - 1).min())
def fr(col):
    cols = {}
    for j in range(13):
        x = W[f'{col}|J{j}'].dropna(); m = (x.index - pd.Timedelta(days=9)).to_period('M'); s = pd.Series(x.values, index=m); cols[j] = s[~s.index.duplicated()]
    return pd.DataFrame(cols).mean(axis=1)
S = fr(f'{SEL}|normal'); B = fr('B|normal'); Z = fr('0050')
for pn, (a, b) in P.items():
    sl = slice(pd.Period(a, 'M'), pd.Period(b, 'M'))
    s, bb, z = S.loc[sl], B.loc[sl], Z.loc[sl]
    beta = float(np.cov(s, z)[0, 1] / np.var(z, ddof=1)); betaB = float(np.cov(bb, z)[0, 1] / np.var(z, ddof=1))
    out[f'risk|{pn}'] = dict(beta_sel_on_0050=beta, beta_B_on_0050=betaB, mdd_sel_monthly=mdd(s), mdd_B=mdd(bb), mdd_0050=mdd(z),
                             vol_sel=float(s.std() * math.sqrt(12)), vol_0050=float(z.std() * math.sqrt(12)),
                             alpha_vs0050_beta_adj_pp=float((s - beta * z).mean() * 100))

R = json.loads((HERE / 'results.json').read_text())
out['check_reported'] = dict(VAL_vs0050_t=R['OOS']['main'][f'{SEL}|normal|vs0050|VAL']['jmed']['t'])
print(json.dumps(out, ensure_ascii=False, indent=1, default=float))

# 附加：beta 調整後對 0050 的 alpha 與 t（j 平均路徑）
for pn in ('VAL', 'Y19', 'SEL', 'FORMAL'):
    a, b = P[pn]; sl = slice(pd.Period(a, 'M'), pd.Period(b, 'M'))
    s, bb, z = S.loc[sl], B.loc[sl], Z.loc[sl]
    X1 = np.column_stack([np.ones(len(z)), z.to_numpy()]); coef, *_ = np.linalg.lstsq(X1, s.to_numpy(), rcond=None)
    res = s.to_numpy() - X1 @ coef; se = math.sqrt((res @ res) / (len(z) - 2) * np.linalg.inv(X1.T @ X1)[0, 0])
    coefB, *_ = np.linalg.lstsq(X1, bb.to_numpy(), rcond=None); resB = bb.to_numpy() - X1 @ coefB
    seB = math.sqrt((resB @ resB) / (len(z) - 2) * np.linalg.inv(X1.T @ X1)[0, 0])
    print('alpha', pn, 'sel', round(coef[0] * 100, 3), 't', round(coef[0] / se, 2), 'beta', round(coef[1], 3), '| B', round(coefB[0] * 100, 3), 't', round(coefB[0] / seB, 2))
