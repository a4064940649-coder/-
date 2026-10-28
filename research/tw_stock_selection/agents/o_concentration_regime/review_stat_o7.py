#!/usr/bin/env python3
"""反方審查（資料與統計）o7：只讀研究員檔案，不寫任何研究員檔案；輸出印在 stdout，另存 scratchpad。
python3 -I -B review_stat_o7.py"""
import sys, json, math
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import numpy as np, pandas as pd
import cr_lib as L, cr_eval as E

OUTF = Path('/tmp/claude-0/-home-user--/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/review_stat_o7.json')
R = {}
W = pd.read_parquet(L.DATA / 'W_full.parquet')
NV4 = pd.read_parquet(L.RAW / 'navs_v4.parquet'); main = NV4['AB_V4|normal'].dropna(); END = main.index[-1]
Wv = W[W['end'] <= END]
e50 = L.e50_series()
spec = dict(id='V5', sig='S2', L=6, xC=1.0, h=0.0)

# ---- 0. 重現主結果 ----
o, B = None, None
def ev(spec, per, Wx=Wv, mainnav=main, wbar=None, sim0='2018-03-01'):
    sd_end = min(Wx['end'][(Wx.index >= pd.Timestamp(per[0])) & (Wx.index <= pd.Timestamp(per[1]))].max(), mainnav.index[-1])
    B = E.build(Wx, spec, mainnav, e50, sim0, 'normal', wbar=wbar, per=per)
    return E.evaluate(B, Wx, mainnav, e50, per, sd_end, label=spec['id']), B
oV, BV = ev(spec, L.VAL); oS, BS = ev(spec, L.SEL)
R['repro'] = dict(VAL_vs_static=oV['vs_static']['mean_pp'], VAL_t=oV['vs_static']['t'], SEL_vs_static=oS['vs_static']['mean_pp'], SEL_t=oS['vs_static']['t'],
                  VAL_vs0050=oV['vs_0050']['mean_pp'], VAL_vs0050_t=oV['vs_0050']['t'], VAL_vs_main=oV['vs_main']['mean_pp'], VAL_vs_main_t=oV['vs_main']['t'])

# ---- 1. 訊號鄰域（事後，只看脆弱度）：S2 的 L＝3..12、多延遲一期 ----
nb = {}
for Lk in (3, 4, 5, 6, 7, 8, 9, 12):
    sp = dict(spec, id=f'S2L{Lk}', L=Lk)
    a, _ = ev(sp, L.SEL); b, _ = ev(sp, L.VAL)
    pre, _ = ev(sp, L.PRE, sim0='2006-04-01')
    nb[f'L{Lk}'] = dict(PRE=round(pre['vs_static']['mean_pp'], 3), SEL=round(a['vs_static']['mean_pp'], 3), VAL=round(b['vs_static']['mean_pp'], 3), VAL_t=round(b['vs_static']['t'], 2),
                        VAL_sw=b['switches'])
R['S2_L_neighborhood_vs_static_pp'] = nb
# 延遲一期：把 W 的 top5 往後移一列（訊號晚一個月才用）
Wlag = Wv.copy(); Wlag['top5'] = Wlag['top5'].shift(1)
lagres = {}
for pn, per in (('PRE', L.PRE), ('SEL', L.SEL), ('VAL', L.VAL)):
    x, _ = ev(spec, per, Wx=Wlag, sim0='2006-04-01' if pn == 'PRE' else '2018-03-01')
    lagres[pn] = dict(mean_pp=round(x['vs_static']['mean_pp'], 3), t=round(x['vs_static']['t'], 2), switches=x['switches'])
R['S2_L6_lag1_vs_static'] = lagres

# ---- 2. 擇時運氣基準：VAL 的 x_k = r50 − rM，隨機狀態序列 ----
sd = list(Wv.index) + [END]
rM = L.win_ret(main, sd); r50 = L.win_ret(L.on_cal(e50, main.index), sd)
df = pd.DataFrame(dict(rM=rM, r50=r50, rEW=Wv['rEW'].reindex(rM.index))).dropna()
df['x'] = df.r50 - df.rM; df['alpha'] = df.rM - df.rEW; df['dsim'] = df.r50 - df.rEW
stV = BV['st'].reindex(df.index)
g = L.sel(df.assign(st=stV), L.VAL)
w = (g.st == 'C').astype(float); wb = w.mean()
obs = float(((w - wb) * g.x).mean() * 100)
rng = np.random.default_rng(7)
# (a) 圓形平移狀態序列（保留連續長度結構）
lbl = (g.st == 'C').astype(float).to_numpy(); xs = g.x.to_numpy(); n = len(xs)
circ = np.array([((np.roll(lbl, k) - np.roll(lbl, k).mean()) * xs).mean() * 100 for k in range(1, n)])
# (b) 區段順序隨機（同區段長度集合）
runs = L.runs(list(lbl)); lens = [b - a + 1 for a, b, _ in runs]; labs = [l for _, _, l in runs]
perm = []
for _ in range(20000):
    o_ = rng.permutation(len(lens)); seq = np.concatenate([[labs[i]] * lens[i] for i in o_]) if False else None
    # 保持狀態交替：隨機打亂 C 區段長度與 D 區段長度，起點狀態隨機
    cL = [lens[i] for i in range(len(lens)) if labs[i] == 1.0]; dL = [lens[i] for i in range(len(lens)) if labs[i] == 0.0]
    rng.shuffle(cL); rng.shuffle(dL)
    start = rng.integers(2); seq = []; ci = di = 0; cur = start
    while ci < len(cL) or di < len(dL):
        if cur == 1 and ci < len(cL): seq += [1.0] * cL[ci]; ci += 1
        elif cur == 0 and di < len(dL): seq += [0.0] * dL[di]; di += 1
        cur = 1 - cur
    seq = np.array(seq[:n]); perm.append(((seq - seq.mean()) * xs).mean() * 100)
perm = np.array(perm)
R['luck_benchmark_VAL'] = dict(observed_timing_pp=obs, n=n, n_runs=len(runs), run_lengths=lens,
                               circular_shift_p_ge_obs=float((circ >= obs - 1e-12).mean()), circular_n=len(circ),
                               run_shuffle_p_ge_obs=float((perm >= obs - 1e-12).mean()), run_shuffle_p95=float(np.percentile(perm, 95)))

# ---- 3. 主策略 α 通道（研究員事後候選）：S2 對下一窗 α 的預測力，分期 ----
S2 = L.signal(Wv, 'S2', 6).reindex(df.index)
ach = {}
for pn, per in (('PRE', L.PRE), ('SEL', L.SEL), ('VAL', L.VAL), ('ALL', ('2006-01-01', '2026-10-31'))):
    gg = L.sel(df.assign(S2=S2), per).dropna()
    r1 = L.ols_hac(gg.alpha.to_numpy(), gg.S2.to_numpy(), 3)
    r2 = L.ols_hac(gg.alpha.to_numpy(), (gg.S2 > 0).astype(float).to_numpy(), 3)
    r3 = L.ols_hac(gg.dsim.to_numpy(), (gg.S2 > 0).astype(float).to_numpy(), 3)
    ach[pn] = dict(n=r1['n'], alpha_on_S2_beta=r1['beta'], alpha_on_S2_t_hac=r1['t_hac'], alpha_C_minus_D_pp=r2['beta'] * 100, alpha_C_minus_D_t=r2['t_hac'],
                   d_C_minus_D_pp=r3['beta'] * 100, d_C_minus_D_t=r3['t_hac'])
R['alpha_channel'] = ach

# ---- 4. P1：重疊迴歸 vs 不重疊、HAC 延遲敏感度 ----
p1 = {}
for pn, per in (('PRE', L.PRE), ('SEL', L.SEL), ('VAL', L.VAL)):
    S = L.signal(Wv, 'S1', 6); y = Wv['d']
    m = (Wv.index >= pd.Timestamp(per[0])) & (Wv.index <= pd.Timestamp(per[1])) & S.notna().to_numpy()
    yy, xx = y[m].to_numpy(), S[m].to_numpy()
    res = {f'hac{lag}': round(L.ols_hac(yy, xx, lag)['t_hac'], 2) for lag in (0, 3, 6, 12)}
    res['beta'] = round(L.ols_hac(yy, xx, 6)['beta'], 3)
    # 不重疊：每 6 個月取一點，下 6 個月累積 d 對過去 6 個月累積 d
    fwd = y.rolling(6).sum().shift(-5)  # 本窗起 6 窗累積
    nn = [(xx_, ff) for xx_, ff in zip(S[m].iloc[::6], fwd[m].iloc[::6]) if np.isfinite(xx_) and np.isfinite(ff)]
    if len(nn) >= 5:
        a_ = np.array(nn); r = L.ols_hac(a_[:, 1], a_[:, 0], 0); res['nonoverlap6_beta'] = round(r['beta'], 3); res['nonoverlap6_t'] = round(r['t_iid'], 2); res['nonoverlap6_n'] = r['n']
    p1[pn] = res
R['P1_robustness'] = p1

# ---- 5. 存活者偏誤：2019-05-16 後消失、之後才正式下市的股票，在決策日前 300 名池中的數量 ----
P = L.Panel()
lv = P.adj.apply(lambda s: s.last_valid_index())
dl = pd.read_csv(L.RAW / 'delist_official.csv', dtype=str); dl['dt'] = pd.to_datetime(dl['date'])
ghost = [c for c, dt in zip(dl.code, dl.dt) if c in P.adj.columns and dt > pd.Timestamp('2019-06-01') and lv[c] is not None and lv[c] <= pd.Timestamp('2019-05-20')]
idx, v = P.pool('2019-05-16'); inpool = [P.adj.columns[i] for i in idx if P.adj.columns[i] in set(ghost)]
R['survivorship'] = dict(n_ghost=len(ghost), ghost_in_top300_on_2019_05_16=inpool, n_in_pool=len(inpool),
                         note='這些股在 2019-05-16 後無資料，2019-06～2023 的同池等權只剩存活者')

# ---- 6. 逐年與單窗貢獻、對 0050 的 2023 依賴 ----
c = ((w - wb) * g.x) * 100
R['VAL_contrib_by_year_pp'] = {int(k): round(float(v), 2) for k, v in c.groupby(c.index.year).sum().items()}
R['VAL_contrib_excl_2023_mean_pp'] = round(float(c[c.index.year >= 2024].mean()), 3)
x50 = L.sel(oV['_series']['r_rot'] - oV['_series']['r50'], L.VAL) * 100
R['VAL_vs0050_by_year_sum_pp'] = {int(k): round(float(v), 2) for k, v in x50.groupby(x50.index.year).sum().items()}
R['VAL_vs0050_excl_2023'] = dict(mean_pp=round(float(x50[x50.index.year >= 2024].mean()), 3), t=round(L.tstat(x50[x50.index.year >= 2024]), 2))
xs_ = L.sel(oV['_series']['r_rot'] - oV['_series']['r_sta'], L.VAL) * 100
R['VAL_vs_static_2024_2026'] = dict(mean_pp=round(float(xs_[xs_.index.year >= 2024].mean()), 3), t=round(L.tstat(xs_[xs_.index.year >= 2024]), 2), n=int((xs_.index.year >= 2024).sum()))
R['VAL_vs_static_2025_2026'] = dict(mean_pp=round(float(xs_[xs_.index.year >= 2025].mean()), 3), t=round(L.tstat(xs_[xs_.index.year >= 2025]), 2), n=int((xs_.index.year >= 2025).sum()))

# ---- 7. 全期（PRE+SEL+VAL）單一檢定 ----
oA, _ = ev(spec, ('2006-01-01', '2026-10-31'), sim0='2006-04-01')
xa = oA['_series']['r_rot'] - oA['_series']['r_sta']
R['ALL_2006_2026'] = dict(mean_pp=oA['vs_static']['mean_pp'], t=oA['vs_static']['t'], n=oA['n'], vs0050_t=oA['vs_0050']['t'], CAGR_rot=oA['CAGR_rot'], CAGR_main=oA['CAGR_main'])
R['ALL_excl_VAL'] = dict(mean_pp=float(xa[xa.index < pd.Timestamp('2023-01-01')].mean() * 100), t=L.tstat(xa[xa.index < pd.Timestamp('2023-01-01')]))

OUTF.write_text(json.dumps(R, ensure_ascii=False, indent=1, default=L._js), encoding='utf-8')
print(json.dumps(R, ensure_ascii=False, indent=1, default=L._js))
