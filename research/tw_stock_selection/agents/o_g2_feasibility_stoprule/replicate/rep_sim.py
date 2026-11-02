#!/usr/bin/env python3
"""Clean-room 預複製（研究員自寫的第二份實作；不 import g2lib／s4_sim／s3_inputs／spa_lib／fam_common）。
只依 PREREG 文字，從原始檔重建輸入：族群超額（o_main_spa_breaks/data/family_excess.parquet＋上櫃 15 條）、0050、V7X R4 13 j、鄰居。
不同的 bootstrap 實作、不同種子（777...）、B=5,000、每個 IR 2,000 次。輸出 replicate/rep_results.json。
用法：python3 -I -B replicate/rep_sim.py"""
import json, math, time
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import norm, skew, kurtosis

AG = Path(__file__).resolve().parents[2]
OUTF = Path(__file__).resolve().parent / 'rep_results.json'
t0 = time.time()
# ---------- 格點與窗報酬 ----------
plan = pd.read_csv(AG / 'o_v4_revenue_patch_official' / 'plan_V4.csv')
pts = pd.DatetimeIndex(list(pd.to_datetime(plan['signal'])) + [pd.Timestamp('2026-10-12')])
def wret(nav):
    s = nav.dropna(); s = s[~s.index.duplicated(keep='last')].sort_index()
    v = pd.merge_asof(pd.DataFrame({'d': pts.astype('datetime64[ns]')}), pd.DataFrame({'d': s.index.astype('datetime64[ns]'), 'v': s.values}), on='d')['v'].to_numpy().copy()
    v[pts < s.index[0]] = np.nan; v[pts > s.index[-1] + pd.Timedelta(days=7)] = np.nan
    return v[1:] / v[:-1] - 1
e = pd.read_csv('/home/user/jonatw/finlab-txd/data/raw/etf_0050.csv', parse_dates=['date']).set_index('date')['adj'].sort_index()
b = wret(e); n = len(b); lab = pts[:-1]
# ---------- 族群 ----------
F = pd.read_parquet(AG / 'o_main_spa_breaks' / 'data' / 'family_excess.parquet')
mm = pd.read_csv(AG / 'o_main_spa_breaks' / 'data' / 'family_members.csv')
mm = mm[mm.F_long & mm.cov_W_long]
fam = F[mm.member_id].to_numpy()
tp = pd.read_parquet(AG / 'o_tpex_pool_extension' / 'out' / 'navs_oos.parquet')
extra = []
for nm in ['M', 'R1', 'R2', 'BASE', 'IT_v1_ALL300_n10', 'IT_v2_ALL400_n10', 'IT_v5_ALL400_n10_floor2k', 'IT_v6_ALL400_n15', 'IT_v7_TP200_n10', 'NB_K300', 'NB_K500', 'NB_floor0', 'NB_floor2k', 'NB_n15', 'NB_n5']:
    rr = np.nanmean(np.column_stack([tp[f'{nm}|J{j}|100萬|GROW|normal'].pct_change().to_numpy() for j in range(13)]), axis=1)
    extra.append(wret(pd.Series(np.cumprod(1 + np.nan_to_num(rr)), index=tp.index)) - b)
fam = np.nan_to_num(np.column_stack([fam] + extra))
fmain = np.concatenate([mm.F_main.to_numpy().astype(bool), np.zeros(15, bool)])
assert fam.shape == (245, 2334)
ab = np.nan_to_num(F['o_v4_revenue_patch_official/navs_v4.parquet::AB_V4|normal'].to_numpy())
v7 = pd.read_parquet(AG / 'o_exec_engine_rewrite_v7x' / 'out' / 'navs_oos_R4.parquet')
def jm(cell, cost): return np.nan_to_num(np.column_stack([wret(v7[f'B10|J{j}|K5T|{cost}|{cell}|GROW']) - b for j in range(13)]))
J1, JS, J3, J10 = jm('100萬', 'normal'), jm('100萬', 'stress'), jm('300萬', 'normal'), jm('1000萬', 'normal')
dev = J1 - J1.mean(1, keepdims=True)
dS = (J1 - JS).mean(1); d3 = (J1 - J3).mean(1); d10 = (J1 - J10).mean(1)
nbn = ['::AB_V4_B05|normal', '::AB_V4_B10|normal', '::ABW_V4|normal', '::EXFIN|n30|normal', '::EW5', '::EW15']
nb = np.column_stack([np.nan_to_num(F[[c for c in F.columns if c.endswith(s)][0]].to_numpy()) - ab for s in nbn]); nb = nb - nb.mean(0)
yr = lab.year.to_numpy(); s1 = lab <= '2012-12-31'; s2 = (lab >= '2013-01-01') & (lab <= '2018-12-31'); s3 = lab >= '2019-01-01'
# ---------- SPA_c 族群最大值（自寫平穩 bootstrap：區塊長度幾何分布，逐區塊接續）----------
rs = np.random.default_rng(777001)
def sbi(rng, T, Lb):
    out = []
    while len(out) < T:
        st = rng.integers(0, n); ln = rng.geometric(1 / Lb)
        out.extend(((st + np.arange(ln)) % n).tolist())
    return np.array(out[:T])
B = 5000
IDX = np.stack([sbi(rs, n, 6) for _ in range(B)])
def omega2(X):
    e_ = X - X.mean(0); T = X.shape[0]; s = (e_ ** 2).mean(0); p = 1 / 6
    for i in range(1, T):
        k = (T - i) / T * (1 - p) ** i + i / T * (1 - p) ** (T - i)
        if k < 1e-8 and i > 30: break
        s = s + 2 * k * (e_[i:] * e_[:-i]).sum(0) / T
    return s
def maxdist(X):
    om = np.sqrt(omega2(X)); mu = X.mean(0)
    keep = mu >= -np.sqrt(om ** 2 / n * 2 * math.log(math.log(n)))
    g = np.where(keep, mu, 0)
    out = np.empty(B)
    for i in range(0, B, 500):
        bm = X[IDX[i:i + 500]].mean(1)        # (500,N)
        out[i:i + 500] = np.maximum(((bm - g) * math.sqrt(n) / om).max(1), 0)
    return out
TB = maxdist(fam); TBm = maxdist(fam[:, fmain])
srF = fam.mean(0) / fam.std(0, ddof=1)
print('TB q95', np.quantile(TB, .95), 'TBmain q95', np.quantile(TBm, .95), round(time.time() - t0), flush=True)
# ---------- 生成與門檻 ----------
sig = ab.std(ddof=1); z = (ab - ab.mean()) / sig
def tt(x): return x.mean(-1) / x.std(-1, ddof=1) * math.sqrt(x.shape[-1])
def gex(x, r):  # x (...,T) 超額, r (T,)
    T = x.shape[-1]
    return (np.prod(1 + r + x, -1) ** (12 / T) - np.prod(1 + r) ** (12 / T)) * 100
def one(ir, rng):
    ii = sbi(rng, n, 6)
    x = ir / math.sqrt(12) * sig + sig * z[ii]; r = b[ii]
    xj = x[None, :] + dev[ii].T                 # (13,T)
    gj = gex(xj, r)
    g1 = np.median(gj) > 0 and np.percentile(gj, 10) > 0 and np.median(gex(xj - dS[None, :], r)) > 0
    tA = np.median(tt(xj)); tY = np.median(tt(xj[:, s3]))
    g2a = tA >= 3; g2b = tY >= 2
    lv = [(np.median(tt(xj - c[None, :])), np.median(tt((xj - c[None, :])[:, s3]))) for c in (d3, d10)]
    g7 = all(a >= 3 and y >= 2 for a, y in lv)
    g4a = all(np.median(gex(xj[:, m], r[m])) > 0 for m in (s1, s2, s3))
    # 拿掉最好 3 年
    td = []
    for j in range(13):
        ye = pd.Series(xj[j]).groupby(yr).apply(lambda s_: 0) if False else None
        yy = {}
        for y in np.unique(yr):
            m = yr == y; yy[y] = np.prod(1 + r[m] + xj[j, m]) - np.prod(1 + r[m])
        top = sorted(yy, key=lambda k: -yy[k])[:3]
        td.append(tt(xj[j, ~np.isin(yr, top)]))
    g4b = np.median(td) >= 2
    g4c = all(gex(x + nb[ii, k], r) > 0 for k in range(6))
    # SPA（候選自己的重新置中統計量也併入最大值）
    om = math.sqrt(omega2(x[:, None])[0]); t = math.sqrt(n) * x.mean() / om
    gg = x.mean() if x.mean() >= -math.sqrt(om ** 2 / n * 2 * math.log(math.log(n))) else 0.0
    own = (x[IDX].mean(1) - gg) * math.sqrt(n) / om
    p = float((np.maximum(np.maximum(TB, own), 0) >= t).mean()) if t > 0 else 1.0
    pm = float((np.maximum(np.maximum(TBm, own), 0) >= t).mean()) if t > 0 else 1.0
    srs = x.mean() / x.std(ddof=1); allsr = np.append(srF, srs); Nn = len(allsr)
    sr0 = math.sqrt(np.var(allsr, ddof=1)) * ((1 - 0.5772156649) * norm.ppf(1 - 1 / Nn) + 0.5772156649 * norm.ppf(1 - 1 / (Nn * math.e)))
    dsr = norm.cdf((srs - sr0) * math.sqrt(n - 1) / math.sqrt(max(1 - skew(x) * srs + (kurtosis(x, fisher=False) - 1) / 4 * srs ** 2, 1e-12)))
    g3a = p < 0.05; g3b = dsr >= 0.95
    gY = np.median(gex(xj[:, s3], r[s3]))
    A = tA >= 2 and gY > 0 and pm < 0.10
    return dict(G1=g1, G2a=g2a, G2b=g2b, G3a=g3a, G3b=g3b, G4a=g4a, G4b=g4b, G4c=g4c, G7=g7, A=A)
res = {}
for k, ir in enumerate([0.0, 0.2, 0.3, 0.4, 0.5, 0.66, 0.8, 1.0]):
    rng = np.random.default_rng(777100 + k)
    rows = pd.DataFrame([one(ir, rng) for _ in range(2000)])
    gates = ['G1', 'G2a', 'G2b', 'G3a', 'G3b', 'G4a', 'G4b', 'G4c', 'G7']
    d = {g: float(rows[g].mean()) for g in gates + ['A']}
    d['joint'] = float(rows[gates].all(1).mean()); d['joint_without_G3b'] = float(rows[[g for g in gates if g != 'G3b']].all(1).mean())
    d['binding_single'] = min(gates, key=lambda g: d[g])
    res[str(ir)] = d
    print(ir, d, round(time.time() - t0), flush=True)
OUTF.write_text(json.dumps({'nsim': 2000, 'B': 5000, 'by_IR': res}, ensure_ascii=False, indent=1))
