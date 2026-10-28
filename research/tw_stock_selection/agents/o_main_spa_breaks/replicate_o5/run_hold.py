import sys, json; sys.path.insert(0, '.')
from breaks import *
from scipy import stats as st
res = {}
v4 = pd.read_parquet(AG / 'o_v4_revenue_patch_official/navs_v4.parquet')
E = e0050(); nav = v4['AB_V4|normal']
rm = win_ret(nav); r50 = win_ret(E); rew = win_ret(v4['AB_V4|EW'])
x = rm - r50; lx = np.log((1 + rm) / (1 + r50))
pre = slice(0, 213)
res['hold_mean_pp'] = x[HOLD].mean() * 100; res['pre_mean_pp'] = x[pre].mean() * 100
obs = lx[HOLD].sum(); res['hold_cumlog'] = obs
for L in (3, 6, 12):
    rng = np.random.default_rng(5150 + L)
    B = 10000; n0 = 213; p = 1 / L
    idx = np.empty((B, 32), int); idx[:, 0] = rng.integers(0, n0, B)
    nb = rng.random((B, 32)) < p; rr = rng.integers(0, n0, (B, 32))
    for t in range(1, 32): idx[:, t] = np.where(nb[:, t], rr[:, t], (idx[:, t - 1] + 1) % n0)
    sims = lx[pre][idx].sum(1); simm = x[pre][idx].mean(1)
    res[f'pred_L{L}'] = dict(median=float(np.median(sims)), q10=float(np.quantile(sims, .1)), q90=float(np.quantile(sims, .9)),
                             p_lower=float((sims <= obs).mean()), p_lower_mean=float((simm <= x[HOLD].mean()).mean()),
                             mde_boot_pp=float((np.quantile(simm, .8) - np.quantile(simm, .05)) * 100))
    print(L, {k: round(v, 4) for k, v in res[f'pred_L{L}'].items()})
z = st.norm.ppf(0.95) + st.norm.ppf(0.8)
e2 = np.concatenate([x[pre] - x[pre].mean(), x[HOLD] - x[HOLD].mean()])
res['mde_2s_hac_pp'] = z * math.sqrt(hac_lr(e2) * (1 / 213 + 1 / 32)) * 100
res['mde_hold_hac_pp'] = z * math.sqrt(hac_lr(x[HOLD] - x[HOLD].mean()) / 32) * 100
res['mde_hold_iid_pp'] = z * x[HOLD].std(ddof=1) / math.sqrt(32) * 100
print({k: round(v, 3) for k, v in res.items() if not isinstance(v, dict)})
for nm, y in (('xEW', rm - rew),):
    pass
# 對同池等權、五分位價差：保留期平均與 p_lower（L6，以平均算術超額）
q = pd.read_parquet(OUT / 'q_navs.parquet')
for nm, y in (('xEW', rm - rew), ('Q5mQ1_gross', win_ret(q['Q5|zero']) - win_ret(q['Q1|zero']))):
    rng = np.random.default_rng(5156); n0 = 213
    idx = np.empty((10000, 32), int); idx[:, 0] = rng.integers(0, n0, 10000)
    nb = rng.random((10000, 32)) < 1 / 6; rr = rng.integers(0, n0, (10000, 32))
    for t in range(1, 32): idx[:, t] = np.where(nb[:, t], rr[:, t], (idx[:, t - 1] + 1) % n0)
    sm = y[pre][idx].mean(1)
    res[f'hold_{nm}'] = dict(mean_pp=y[HOLD].mean() * 100, p_lower=float((sm <= y[HOLD].mean()).mean()))
    print(nm, res[f'hold_{nm}'])
# ---- 落後拆解（窗格點）----
P = PANEL
adj = pd.read_parquet(P / 'adj.parquet', columns=['2330']); cr = pd.read_parquet(P / 'close_raw.parquet')
r2330 = win_ret(adj['2330'])
ex = pd.read_csv('/home/user/stevenhsiehsheng-ctrl/tw-stock-screener/data/extras/ex_tsmc.csv', parse_dates=['date']).set_index('date')
sh = pd.read_csv('/home/user/stevenhsiehsheng-ctrl/tw-stock-screener/data/extras/shares_hist.csv', dtype={'code': str})
sh0 = sh.sort_values('ym').groupby('code').shares.first()
cols = [c for c in cr.columns if c in sh0.index]
def w_top50(d):
    px = cr[cols].loc[:d].iloc[-1]
    mc = (px * sh0[cols]).dropna().sort_values(ascending=False)
    top = mc.iloc[:50]
    return top.get('2330', 0) / top.sum()
wa = np.array([w_top50(G[i]) for i in range(NW)])
wact = np.full(NW, np.nan)
for i in range(NW):
    s = ex['w2330'].loc[:G[i]]
    if len(s) and (G[i] - s.index[-1]).days <= 7 and G[i] >= pd.Timestamp('2025-10-01'): wact[i] = s.iloc[-1] / 100
ov = np.isfinite(wact)
adjw = np.nanmean(wact[ov] - wa[ov])
w = np.where(ov, wact, wa + adjw)
res['decomp'] = dict(n_overlap=int(ov.sum()), adj_pp=adjw * 100)
rex = (r50 - w * r2330) / (1 - w)
H_ = HOLD
res['decomp'].update(sum_x50_pp=x[H_].sum() * 100, sum_xexT_pp=(rm - rex)[H_].sum() * 100,
    gap_pp=((rm - rex)[H_].sum() - x[H_].sum()) * 100,
    logW_main=float(np.log1p(rm[H_]).sum()), logW_0050=float(np.log1p(r50[H_]).sum()), logW_exT=float(np.log1p(rex[H_]).sum()),
    w_mean=float(w[H_].mean()), t2330_all=tstat(r2330 - r50), t2330_hold=tstat((r2330 - r50)[H_]))
res['decomp']['gap_share'] = (res['decomp']['logW_0050'] - res['decomp']['logW_exT']) / (res['decomp']['logW_0050'] - res['decomp']['logW_main'])
print('decomp', {k: round(v, 4) for k, v in res['decomp'].items()})
# ---- 日曆口徑 2024-01-01 起 ----
d0 = pd.Timestamp('2023-12-29')
def cal(s, a, b): s = s.dropna(); return float(s.asof(b) / s.asof(a) - 1)
for nm, s in (('main', nav), ('0050', E)):
    res[f'cal_{nm}'] = dict(all=cal(s, d0, END), gap=cal(s, d0, pd.Timestamp('2024-02-15')))
print('cal', res['cal_main'], res['cal_0050'], 'loglag', math.log((1 + res['cal_0050']['all']) / (1 + res['cal_main']['all'])))
# 日曆月 t、壓力成本 t
me = lambda s: s.dropna().groupby(s.dropna().index.to_period('M')).last()
mm = (me(nav).pct_change() - me(E.reindex(nav.index)).pct_change()).dropna()
res['t_calmonth'] = tstat(mm); print('cal-month t', res['t_calmonth'], len(mm))
xs = win_ret(v4['AB_V4|stress']) - r50
res['t_stress_all'] = tstat(xs); res['t_stress_post2019'] = tstat(xs[157:]); print('stress t', res['t_stress_all'], res['t_stress_post2019'])
# ---- 集中度因子 ----
adjall = pd.read_parquet(P / 'adj.parquet')
R = {}
for i in range(NW):
    px = cr[cols].loc[:G[i]].iloc[-1]; mc = (px * sh0[cols]).dropna().sort_values(ascending=False).iloc[:10]
    a0 = adjall[mc.index].loc[:G[i]].iloc[-1]; a1 = adjall[mc.index].loc[:G[i + 1]].iloc[-1]
    R[i] = float(((a1 / a0 - 1) * mc / mc.sum()).sum())
cf = np.array([R[i] for i in range(NW)]) - rew
def hac_reg(y, X):
    X = np.column_stack([np.ones(len(y)), X]); b = np.linalg.lstsq(X, y, rcond=None)[0]; e = y - X @ b
    n = len(y); lag = int(np.floor(4 * (n / 100) ** (2 / 9)))
    U = X * e[:, None]; S = U.T @ U / n
    for j in range(1, lag + 1):
        G_ = U[j:].T @ U[:-j] / n; S += (1 - j / (lag + 1)) * (G_ + G_.T)
    Q = np.linalg.inv(X.T @ X / n); V = Q @ S @ Q / n
    r2 = 1 - e.var() / y.var()
    return b, b / np.sqrt(np.diag(V)), r2
for lab, sl in (('all', slice(0, NW)), ('pre', slice(0, 213)), ('hold', HOLD)):
    b, t, r2 = hac_reg(x[sl], cf[sl])
    res[f'cf_{lab}'] = dict(alpha_pp=b[0] * 100, beta=b[1], t_alpha=t[0], t_beta=t[1], r2=r2, cf_mean_pp=cf[sl].mean() * 100)
    print('CF', lab, {k: round(v, 3) for k, v in res[f'cf_{lab}'].items()})
json.dump(res, open(OUT / 'hold.json', 'w'), indent=1, default=float)
