#!/usr/bin/env python3
"""步驟 0～1：清點與探索（只用校準期 2024-02-15～2025-06-30 的 5 分 K；MIS vol_curve 與零股是另外的時段，只當真值與描述）。
不看任何策略報酬。輸出 explore.json、data/stockdays.parquet（股日表，PREREG 涵蓋）、data/bars.parquet。
用法：python3 -I -B explore.py
"""
import sys, json
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import numpy as np, pandas as pd
import calib_lib as C

OUT = {}
X, R = C.stockdays()
X.to_parquet(C.DATA / 'stockdays.parquet', index=False); R.to_parquet(C.DATA / 'bars.parquet', index=False)

# ---------------- 步驟 0：清點 ----------------
lg = json.load(open(C.RAW / 'tw5m_download_log.json'))
oc = json.load(open(C.DATA / 'official_check.json'))
P = X[X.vrank_l1 <= 300].copy(); P['ratio'] = P.vol_day / P.vol_off
dr = P.groupby('date').ratio.median()
d5 = pd.DatetimeIndex(sorted(X.date.unique()))
O = pd.read_parquet(C.DATA / 'official_daily.parquet'); od = pd.DatetimeIndex(sorted(pd.to_datetime(O.date).unique()))
od_w = od[(od >= d5[0]) & (od <= d5[-1])]
OUT['step0_inventory'] = {
    '5m_download': lg['summary'],
    '5m_dates': dict(first=str(d5[0].date()), last=str(d5[-1].date()), n=len(d5), official_days_missing_in_5m=[str(d.date()) for d in od_w.difference(d5)],
                     gaps_gt5_busdays=[(str(a.date()), str(b.date())) for a, b in zip(d5[:-1], d5[1:]) if np.busday_count(a.date(), b.date()) - 1 > 5]),
    '5m_vs_official_volume_ratio_by_day': dict(median=float(dr.median()), p05=float(dr.quantile(.05)), p95=float(dr.quantile(.95)),
                                               anomalous_days_lt075_or_gt1=[str(d.date()) for d, v in dr.items() if v < 0.75 or v > 1.0]),
    'official': oc,
    'yahoo_open_vs_official_open_mismatch_gt0.5pct_pool': float((np.abs(P.y_open / P.o_off - 1) > 0.005).mean()),
    'full12_share_pool': float(P.full12.mean()),
}
BAD = set(OUT['step0_inventory']['5m_vs_official_volume_ratio_by_day']['anomalous_days_lt075_or_gt1'])
U = C.universe(X) & ~X.date.dt.strftime('%Y-%m-%d').isin(BAD)
X['U'] = U
CALm = U & (X.period == 'CAL')
OUT['universe'] = dict(rule='vrank_{t-1}≤300、非金融、12 根 K 棒齊全、根數≥50、σ/V60 有值、Yahoo 開盤＝官方開盤±0.5%、排除量比異常日',
                       n_CAL=int(CALm.sum()), n_VAL=int((U & (X.period == 'VAL')).sum()),
                       n_CAL_by_dec={int(k): int(v) for k, v in X[CALm].groupby('dec').size().items()},
                       n_VAL_by_dec={int(k): int(v) for k, v in X[U & (X.period == 'VAL')].groupby('dec').size().items()})

# ---------------- 步驟 1a：開盤量估計 ----------------
V = C.mis()
v = V[V.volume > 0].copy()
OUT['mis_truth'] = dict(n_stockdays=len(v), n_in_pool=int((v.vrank_l1 <= 300).sum()), days=sorted({str(d.date()) for d in v.date}),
                        oa_share=dict(median=float(v.oa_share.median()), p25=float(v.oa_share.quantile(.25)), p75=float(v.oa_share.quantile(.75))),
                        ca_share=dict(median=float(v.ca_share.median()), p25=float(v.ca_share.quantile(.25)), p75=float(v.ca_share.quantile(.75))),
                        f5_share_median=float(v.f5_share.median()), rho_F5_over_bar0905_median=float(v.rho.median()),
                        vol_surprise_ln_median=float(np.log(v.volume / v.V60sh).median()),
                        note='MIS 股日多為警示名單（爆量日）；只有少數在前 300 池')
c = X[CALm].copy()
rho_y = float((c['q09:05'] / c['q09:10']).replace([np.inf, -np.inf], np.nan).median())
nzd = c.groupby('date').apply(lambda g: (g.q0900 > 0).mean())
NZ = set(nzd[nzd > 0.8].index)
cz = c[c.date.isin(NZ) & (c.q0900 > 0)].copy()
cz['ca_y'] = (cz.vol_off - cz.vol_day) / cz.vol_off
ca_dec = cz.groupby('dec').ca_y.median()
OUT['yahoo_open_volume'] = dict(rho_y_q0905_over_q0910=rho_y, n_nonzero_0900_days=len(NZ), nonzero_days=sorted(str(d.date()) for d in NZ),
                                s0900_share_on_nonzero_days=float((cz.q0900 / cz.vol_off).median()),
                                ca_y_share_by_dec={int(k): float(x) for k, x in ca_dec.items()},
                                miss_share_by_dec_normal_days={int(k): float(x) for k, x in c[~c.date.isin(NZ)].assign(s=lambda d: d.miss / d.vol_off).groupby('dec').s.median().items()})
# 三種估計
c['ca_hat'] = c.dec.map(ca_dec) * c.vol_off
c['bar0900_b'] = np.where(c.date.isin(NZ), np.nan, c.miss - c.ca_hat)          # 正常日：官方 − Yahoo − 收盤集合競價
c['OA_b'] = c.bar0900_b - rho_y * c['q09:05']
c['OA_a'] = np.where(c.date.isin(NZ) & (c.q0900 > 0), c.q0900 - rho_y * c['q09:05'], np.nan)
c['OA_c'] = c.bar0900_b                                                         # 上界：09:00～09:05 全部
FEAT = ['lnrank', 'lnprice', 'lnsig', 'lnsurp', 'absgap']
def feats(d):
    rk = d.vrank_l1.fillna(d['vrank_all']) if 'vrank_all' in d else d.vrank_l1
    return pd.DataFrame({'const': 1.0, 'lnrank': np.log(rk.clip(lower=1)), 'lnprice': np.log(d.prevc), 'lnsig': np.log(d.sig60_l1.clip(lower=1e-4)),
                         'lnsurp': np.log(d.vol_off / d.V60sh), 'absgap': np.abs(d.gap)}, index=d.index)
vm = v[np.isfinite(v.V60sh) & np.isfinite(v.sig60_l1) & (v.OA > 0)].copy()
vm = vm.rename(columns={'volume': 'vol_off'})
fv = feats(vm)
meth = {}
for k in ('OA_a', 'OA_b', 'OA_c'):
    d = c[np.isfinite(c[k])]
    pos = d[d[k] > 0]
    Xf = feats(pos); y = np.log(pos[k] / pos.V60sh)
    ok = np.isfinite(Xf).all(axis=1) & np.isfinite(y)
    b, *_ = np.linalg.lstsq(Xf[ok].to_numpy(), y[ok].to_numpy(), rcond=None)
    pr = fv.to_numpy() @ b
    tr = np.log(vm.OA / vm.V60sh)
    e = pr - tr
    meth[k] = dict(n_fit=int(ok.sum()), share_nonpos=float((d[k] <= 0).mean()), coef=dict(zip(['const'] + FEAT, map(float, b))),
                   share_of_day_median_CAL=float((pos[k] / pos.vol_off).median()),
                   share_by_dec_CAL={int(kk): float(x) for kk, x in (pos[k] / pos.vol_off).groupby(pos.dec).median().items()},
                   mis_corr_log=float(np.corrcoef(pr, tr)[0, 1]), mis_median_abs_log_err=float(np.median(np.abs(e))), mis_median_log_bias=float(np.median(e)),
                   mis_corr_share=float(np.corrcoef(np.exp(pr) * vm.V60sh / vm.vol_off, vm.OA / vm.vol_off)[0, 1]),
                   mis_in_pool_median_abs_log_err=float(np.median(np.abs(e[(vm.vrank_l1 <= 300).to_numpy()]))) if (vm.vrank_l1 <= 300).sum() else None)
OUT['open_volume_methods'] = meth
best = min(meth, key=lambda k: meth[k]['mis_median_abs_log_err'])
OUT['open_volume_best_method'] = best
c['OA'] = c[best] if best != 'OA_a' else c['OA_b']
X.loc[c.index, 'OA_b'] = c['OA_b']; X.loc[c.index, 'ca_hat'] = c['ca_hat']

# ---------------- 步驟 1b：衝擊可觀測量 ----------------
Rb = R.loc[c.index]
rows = []
for b in C.BARS:
    q = c['q' + b]; r = Rb[b].abs()
    x2 = c.sig60_l1 * np.sqrt(q / c.V60sh)
    oa = c.OA.where(c.OA > 0)
    x1 = c.sig60_l1 * np.sqrt(q / oa)
    ok = np.isfinite(x2) & np.isfinite(r) & (q > 0)
    ok1 = ok & np.isfinite(x1)
    rows.append(dict(bar=b, n=int(ok.sum()), mean_absr=float(r[ok].mean()), Y_ls0=float((x2[ok] * r[ok]).sum() / (x2[ok] ** 2).sum()),
                     Y_ratio=float(r[ok].sum() / x2[ok].sum()), kappa_open_ls0=float((x1[ok1] * r[ok1]).sum() / (x1[ok1] ** 2).sum()),
                     kappa_open_ratio=float(r[ok1].sum() / x1[ok1].sum()), q_over_V60_median=float((q / c.V60sh)[ok].median())))
OUT['bar_sqrt_law_CAL'] = rows
# 分 x 十分位的形狀（全部 bar 合併；M2 型）
L = []
for b in C.BARS:
    L.append(pd.DataFrame({'r': Rb[b].abs(), 'x2': c.sig60_l1 * np.sqrt(c['q' + b] / c.V60sh), 'qv': c['q' + b] / c.V60sh, 'sig': c.sig60_l1, 'dec': c.dec,
                           'x1': c.sig60_l1 * np.sqrt(c['q' + b] / c.OA.where(c.OA > 0)), 'bar': b}))
Lb = pd.concat(L); Lb = Lb[np.isfinite(Lb.r) & np.isfinite(Lb.x2) & (Lb.qv > 0)]
Lb['xb'] = pd.qcut(Lb.x2, 10, labels=False)
OUT['bar_shape_by_x2_decile'] = Lb.groupby('xb').apply(lambda g: dict(x2=float(g.x2.mean()), absr=float(g.r.mean()), ratio=float(g.r.mean() / g.x2.mean()))).tolist()
# log-log 斜率（δ）：E|r|/σ 對 q/V60
gq = Lb.assign(qb=pd.qcut(np.log(Lb.qv), 20, labels=False)).groupby('qb').apply(lambda g: (np.log(g.qv).mean(), np.log((g.r / g.sig).mean())))
xs = np.array([a for a, _ in gq]); ys = np.array([b_ for _, b_ in gq])
OUT['bar_loglog_delta_q_over_V60'] = float(np.polyfit(xs, ys, 1)[0])
OUT['bar_Y_by_dec'] = {int(k): float(g.r.sum() / g.x2.sum()) for k, g in Lb.groupby('dec')}
OUT['bar_kappa_open_by_dec'] = {int(k): float(g.r[np.isfinite(g.x1)].sum() / g.x1[np.isfinite(g.x1)].sum()) for k, g in Lb.groupby('dec')}
# (ii) 開盤溢價
c['absprem'] = c.prem.abs()
c['oa_sh'] = (c.OA / c.V60sh).where(c.OA > 0)
gb = pd.cut(c.gap, [-1, -0.03, -0.01, -0.0025, 0.0025, 0.01, 0.03, 1])
OUT['open_premium_CAL'] = dict(mean=float(c.prem.mean()), median=float(c.prem.median()), mean_abs=float(c.absprem.mean()),
                               by_gap={str(k): dict(n=int(len(g)), mean_prem=float(g.prem.mean()), mean_abs=float(g.absprem.mean())) for k, g in c.groupby(gb)},
                               abs_over_sigma_by_oa_share_tercile={str(k): float((g.absprem / g.sig60_l1).mean()) for k, g in c.groupby(pd.qcut(c.oa_sh, 3))})
# 迴歸 |prem| = a·σ + b·σ√(OA/V60) + c·σ√(vol_0905_0925/V60) + d·|gap|
d = c[np.isfinite(c.oa_sh) & np.isfinite(c.absprem)]
A = np.column_stack([d.sig60_l1, d.sig60_l1 * np.sqrt(d.oa_sh), d.sig60_l1 * np.sqrt(d.vol_0905_0925 / d.V60sh), d.gap.abs()])
cf, *_ = np.linalg.lstsq(A, d.absprem.to_numpy(), rcond=None)
OUT['open_premium_regression_CAL'] = dict(vars=['σ', 'σ√(OA/V60)', 'σ√(V0905_0930/V60)', '|gap|'], coef=list(map(float, cf)), n=len(d))
# (iii) 30 分鐘回吐
c['rev30'] = -np.sign(c.gap) * np.log(c.c0925 / c.o_off)
OUT['reversal30_CAL'] = {str(k): dict(n=int(len(g)), mean_rev=float(g.rev30.mean()), median=float(g.rev30.median())) for k, g in c.groupby(gb)}
OUT['reversal30_by_oa_share_tercile_gap_gt1pct'] = {str(k): float(g.rev30.mean()) for k, g in c[c.gap.abs() > 0.01].groupby(pd.qcut(c[c.gap.abs() > 0.01].oa_sh, 3))}

# ---------------- 步驟 1c：擁擠（校準期換股日）----------------
pick, new, drop, days = C.exec_flags()
ex_days = [d for d in days if C.CAL[0] <= d <= C.CAL[1]]
allX = X[U & (X.period == 'CAL')].copy()
allX['near_exec'] = False
for e in ex_days:
    allX.loc[(allX.date >= e - pd.Timedelta(days=2)) & (allX.date <= e + pd.Timedelta(days=2)), 'near_exec'] = True
base = allX[~allX.near_exec]
res = {}
for grp, lo, hi in (('TOP10', 0.9, 1.01), ('BOT50', -1, 0.5), ('MID', 0.5, 0.9)):
    per_day = []
    for e in ex_days:
        g = allX[(allX.date == e) & (allX.pct > lo) & (allX.pct <= hi)]
        if len(g) == 0: continue
        bb = base[base.code.isin(g.code) & (base.date >= e - pd.Timedelta(days=60)) & (base.date <= e + pd.Timedelta(days=60))].groupby('code').prem.mean()
        dlt = (g.set_index('code').prem - bb).dropna()
        per_day.append(dict(date=str(e.date()), n=int(len(dlt)), mean=float(dlt.mean()), gap_mean=float(g.gap.mean())))
    m = np.array([p['mean'] for p in per_day])
    res[grp] = dict(n_days=len(m), mean_pp=float(m.mean() * 100), t_day=float(m.mean() / (m.std(ddof=1) / np.sqrt(len(m)))) if len(m) > 2 else None, per_day=per_day)
res['TOP10_minus_BOT50_pp'] = res['TOP10']['mean_pp'] - res['BOT50']['mean_pp']
dd = np.array([a['mean'] - b['mean'] for a, b in zip(res['TOP10']['per_day'], res['BOT50']['per_day'])])
res['TOP10_minus_BOT50_t_day'] = float(dd.mean() / (dd.std(ddof=1) / np.sqrt(len(dd))))
newX = allX[[(d_, c_) in new for d_, c_ in zip(allX.date, allX.code)]]
res['new_entries_exec_day'] = dict(n=len(newX), mean_prem_pp=float(newX.prem.mean() * 100), mean_gap_pp=float(newX.gap.mean() * 100),
                                   mean_rev30_pp=float((-np.log(newX.c0925 / newX.o_off)).mean() * 100) if len(newX) else None)
OUT['crowding_CAL'] = res

# ---------------- 步驟 1d：零股（2026-07～10）----------------
o = C.oddlot(); p = o[o.vrank_l1 <= 300].copy(); p['dec'] = np.ceil(p.vrank_l1 / 30)
pb = pd.cut(p.close, [0, 50, 100, 300, 1e9])
OUT['oddlot_2026'] = dict(n=len(p), n_days=int(p.date.nunique()), range=[str(p.date.min().date()), str(p.date.max().date())],
                          daily_odd_value_median=float(p.amount.median()), daily_odd_value_by_dec={int(k): float(x) for k, x in p.groupby('dec').amount.median().items()},
                          odd_share_of_day_value_median=float(p.odd_share_of_day.median()),
                          buy_cost_median_by_price={str(k): float(x) for k, x in p.groupby(pb).buy_cost.median().items()},
                          sell_cost_median_by_price={str(k): float(x) for k, x in p.groupby(pb).sell_cost.median().items()},
                          share_daily_odd_value_lt={str(int(x)): float((p.amount < x).mean()) for x in (3e4, 1e5, 1.5e5, 3e5, 1e6)},
                          note='oddlot_hist 為收盤時點的當日盤中零股累計量與最後 bid/ask；9:10 首次撮合的量沒有任何資料（覆蓋 0%）')
json.dump(OUT, open(HERE / 'explore.json', 'w'), ensure_ascii=False, indent=1, default=str)
print(json.dumps({k: OUT[k] for k in ('universe', 'open_volume_methods', 'open_volume_best_method', 'bar_loglog_delta_q_over_V60', 'open_premium_regression_CAL')}, ensure_ascii=False, indent=1, default=str)[:6000])
