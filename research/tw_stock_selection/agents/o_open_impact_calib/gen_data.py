#!/usr/bin/env python3
"""步驟 0（資料凍結前）：用 V4 凍結版（o_v4_revenue_patch_official/v4lib.py，只讀 import）產生本題需要的名單與面板欄位。
  1) AB_V4_B10 的 40 seed 名單（V4(gap=GAP, pfix=1.0, unavail=make_unavail('mc', seed))），零衝擊一般成本重現 navs_v4 的 AB_V4_B10|normal（斷言 < 1e-9）。
     AB_V4 名單沿用 o_exec_capacity_tiers/data/plans_ab.csv（已在 o5 重現 AB_V4|normal 逐日差 0），本程式再驗一次。
  2) 執行面板：AB_V4 ∪ V4 ∪ AB_B10 ∪ j 版名單（data/plans_j.csv，先跑 gen_jplans.py）曾入選股，欄位同 o5 build_exec_data（px_open/close/grow/traded/prevclose、feat_daily、xrules），
     與 o5 的重疊欄位斷言相同。
  3) 校準用全市場特徵（2023-06～2026-10）：上市股的 V60（gapfix）、σ60、vrank（V60≥1e7 且收盤≥10 的名次），存成長表 data/feat_all_2023.parquet。
  4) 綜合分數：V4（非 AB）每個訊號日 2023-12～2026-10 的 0.5×52 週百分位＋0.5×營收百分位（只在合格股），data/scores_v4.parquet。
用法：python3 -I -B gen_data.py
"""
import sys, json, time
from pathlib import Path
HERE = Path(__file__).resolve().parent
V4D = HERE.parent / 'o_v4_revenue_patch_official'
O5 = HERE.parent / 'o_exec_capacity_tiers'
sys.path.insert(0, str(V4D))
import numpy as np, pandas as pd
import v4lib as L
sys.path.pop(0)
K, V = L.K, L.V
T0 = time.time()
def log(*a): print(f'[{time.time() - T0:6.0f}s]', *a, flush=True)

XR = L.setup_px()
YFX, Y5F = L.load_rfix()
DMT = L.disp_mask_fn('trade')
GAP = L.gap_dict()
NV = pd.read_parquet(V4D / 'navs_v4.parquet')
IDX, COLS = K.IDX, K.COLS

S0 = L.V4(YFX, Y5F, DMT)
t0, t1, sd = S0.bounds()

# ---- 1) AB_V4 再驗、AB_B10 產生 ----
P_ab = pd.read_csv(O5 / 'data' / 'plans_ab.csv', dtype={'picks': str})
def to_pl(g): return [(int(e), [K.CI[c] for c in pk.split(',')]) for e, pk in zip(g.exec_idx, g.picks)]
rets = []
for s, g in P_ab.groupby('seed'):
    nv, _ = L.O3.simulate(to_pl(g), *L.cost_rates('normal'), t0, t1, xrules=XR); rets.append(nv.pct_change())
m = (1 + pd.concat(rets, axis=1).mean(axis=1).fillna(0)).cumprod()
d_ab = float((m / NV['AB_V4|normal'].reindex(m.index) - 1).abs().max()); log('AB_V4 重現', d_ab); assert d_ab < 1e-9

rows = []; rets = []
for seed in range(40):
    S = L.V4(YFX, Y5F, DMT, gap=GAP, pfix=1.0, unavail=V.make_unavail('mc', seed))
    pl = S.plan(10, 'official')
    assert S.bounds()[:2] == (t0, t1)
    nv, _ = L.sim_plan(pl, t0, t1, 'normal', XR); rets.append(nv.pct_change())
    for p in pl:
        rows.append((seed, str(p[3].date()), str(IDX[p[0]].date()), int(p[0]), ','.join(COLS[j] for j in p[1])))
m = (1 + pd.concat(rets, axis=1).mean(axis=1).fillna(0)).cumprod()
d_b10 = float((m / NV['AB_V4_B10|normal'].reindex(m.index) - 1).abs().max()); log('AB_V4_B10 重現', d_b10); assert d_b10 < 1e-9
PB = pd.DataFrame(rows, columns=['seed', 'signal', 'exec', 'exec_idx', 'picks'])
PB.to_csv(HERE / 'data' / 'plans_b10.csv', index=False)

# ---- 2) 執行面板 ----
PJ = pd.read_csv(HERE / 'data' / 'plans_j.csv', dtype=str) if (HERE / 'data' / 'plans_j.csv').exists() else pd.DataFrame(columns=['picks'])   # gen_jplans.py 的 j 版名單（第二次執行時納入）
p = pd.concat([P_ab.astype(str), pd.read_csv(O5 / 'data' / 'plan_v4.csv', dtype=str), PB.astype(str), PJ[['picks']]])
codes = sorted({c for x in p.picks for c in x.split(',')})
J = np.array([K.CI[c] for c in codes])
def wide(a): return pd.DataFrame(a[:, J], index=IDX, columns=codes)
W = {'px_open': wide(K.On), 'px_close': wide(K.Cn), 'px_grow': wide(K.GROW), 'px_traded': wide(K.TRADED.astype(np.int8)), 'prevclose': wide(K.PREVC)}
for k, w in W.items():
    o = pd.read_parquet(O5 / 'data' / f'{k}.parquet')
    a, b = w[o.columns].to_numpy(dtype=float), o.to_numpy(dtype=float)
    assert np.array_equal(np.isnan(a), np.isnan(b)) and np.allclose(np.nan_to_num(a), np.nan_to_num(b), rtol=0, atol=0), k
    w.to_parquet(HERE / 'data' / f'{k}.parquet')
_, V60 = V.rolling_signals('gapfix'); V60 = np.asarray(V60)
G = np.asarray(K.GOOD)
Rr = pd.DataFrame(K.Rn[G], index=IDX[G], columns=COLS)
SIG = Rr.rolling(60, min_periods=40).std().reindex(IDX).ffill().to_numpy()
base = np.isfinite(V60) & (V60 >= 1e7) & np.isfinite(K.Cn) & (K.Cn >= 10)
vv = np.where(base, V60, -np.inf)
order = np.argsort(-vv, axis=1, kind='stable')
ranks = np.empty_like(order); ar = np.arange(V60.shape[1])
for t in range(len(IDX)): ranks[t, order[t]] = ar
RK = np.where(base, ranks + 1.0, np.nan)
F = pd.DataFrame({'V60': V60[:, J].ravel(), 'sig60': SIG[:, J].ravel(), 'vrank': RK[:, J].ravel()},
                 index=pd.MultiIndex.from_product([IDX, codes], names=['date', 'code'])).reset_index()
F = F[np.isfinite(F.V60) | np.isfinite(F.sig60)]
fo = pd.read_parquet(O5 / 'data' / 'feat_daily.parquet')
chk = fo.merge(F, on=['date', 'code'], suffixes=('_o', '_n'))
for c in ('V60', 'sig60', 'vrank'):
    a, b = chk[c + '_o'].to_numpy(), chk[c + '_n'].to_numpy()
    assert np.array_equal(np.isnan(a), np.isnan(b)) and np.nanmax(np.abs(a - b)) == 0, c
assert len(chk) == len(fo)
F.to_parquet(HERE / 'data' / 'feat_daily.parquet', index=False)
xr = {COLS[j]: [float(v[0]), str(pd.Timestamp(v[1]).date()), v[2], (str(pd.Timestamp(v[3]).date()) if len(v) > 3 else None)] for j, v in XR.items() if COLS[j] in set(codes)}
json.dump(xr, open(HERE / 'data' / 'xrules.json', 'w'), ensure_ascii=False, indent=1)
B = pd.DataFrame({'E50': K.E50}).reindex(IDX)
for c in ('AB_V4|normal', 'AB_V4|stress', 'AB_V4|EW', 'V4|normal', 'AB_V4_B10|normal', 'AB_V4_B10|stress', 'AB_V4_B10|EW'): B[c] = NV[c].reindex(IDX)
B.to_parquet(HERE / 'data' / 'bench.parquet')
log('codes', len(codes), 'xrules', len(xr))

# ---- 3) 全市場特徵（2023-06 起）----
sel = IDX >= pd.Timestamp('2023-06-01')
FA = pd.DataFrame({'V60': V60[sel].ravel(), 'sig60': SIG[sel].ravel(), 'vrank': RK[sel].ravel(), 'close': K.Cn[sel].ravel(), 'open': K.On[sel].ravel(),
                   'prevc': K.PREVC[sel].ravel(), 'traded': K.TRADED[sel].ravel()},
                  index=pd.MultiIndex.from_product([IDX[sel], COLS], names=['date', 'code'])).reset_index()
FA = FA[np.isfinite(FA.V60) | FA.traded]
FA['fin'] = FA.code.map(dict(zip(COLS, L.FIN2MASK)))
FA.to_parquet(HERE / 'data' / 'feat_all_2023.parquet', index=False)

# ---- 4) 綜合分數（非 AB V4）----
sr = []
for d in S0.sig_days:
    if d < pd.Timestamp('2023-12-01'): continue
    cs = S0.cross_section(d)
    e = cs['e']
    for j in np.where(cs['el'])[0]:
        sr.append((str(d.date()), str(IDX[e].date()) if e < len(IDX) else None, COLS[j], float(cs['sc'][j]), bool(cs['top'][j])))
SC = pd.DataFrame(sr, columns=['signal', 'exec', 'code', 'score', 'top300'])
SC['pct'] = SC.groupby('signal')['score'].rank(pct=True)
SC.to_parquet(HERE / 'data' / 'scores_v4.parquet', index=False)
json.dump({'AB_V4_repro_maxrel': d_ab, 'AB_V4_B10_repro_maxrel': d_b10, 'n_codes_exec': len(codes), 't0': str(IDX[t0].date()), 't1': str(IDX[t1].date()),
           'n_score_rows': len(SC), 'n_signal_scored': int(SC.signal.nunique())},
          open(HERE / 'data' / 'gen_data_check.json', 'w'), ensure_ascii=False, indent=1)
log('done')
