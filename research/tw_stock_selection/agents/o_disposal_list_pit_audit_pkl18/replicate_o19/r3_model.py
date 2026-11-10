"""獨立複製 o19 S2：近似注意旗標→觸發事件數 K_reg，NB2 月模型，2019 預測與區間、逐年 z。"""
import json, math
import numpy as np, pandas as pd
from scipy import optimize, stats, special

BASE = '/home/user/-/research/tw_stock_selection/agents/o_disposal_list_pit_audit_pkl18/'
OUT = BASE + 'replicate_o19/'
P = '/tmp/claude-0/-home-user--/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/panel/'

adj = pd.read_parquet(P + 'adj.parquet')
cnt = adj.notna().sum(axis=1)
keep = cnt >= 300
adj = adj[keep]
val = pd.read_parquet(P + 'value.parquet')[keep]
cr = pd.read_parquet(P + 'close_raw.parquet')[keep]
cal = adj.index
vol = val / cr
tx = pd.read_csv('/home/user/jonatw/finlab-txd/data/raw/taiex_twii.csv', parse_dates=['date']).set_index('date').close
tx = tx.reindex(cal, method='ffill')

r6 = adj / adj.shift(6) - 1
m6 = tx / tx.shift(6) - 1
theta = pd.Series(np.where(cal < pd.Timestamp('2015-06-01'), 0.25, 0.32), index=cal)
diff = r6.sub(m6, axis=0)
fa = (r6.abs().gt(theta, axis=0)) & ((np.sign(r6) * diff) > 0.20)
import sys
MP = int(sys.argv[1]) if len(sys.argv) > 1 else 60
TAG = '' if MP == 60 else '_mp%d' % MP
v60 = vol.shift(1).rolling(60, min_periods=MP).mean()
fb = (vol / v60 >= 5) & (r6.abs() > 0.10)
F = (fa | fb) & adj.notna()
Fv = F.values.astype(np.int8)
n_t, n_s = Fv.shape
early = (cal < pd.Timestamp('2014-04-01'))
# 連續天數
run = np.zeros_like(Fv, dtype=np.int32)
for t in range(n_t):
    run[t] = (run[t - 1] + 1) * Fv[t] if t else Fv[t]
c10 = pd.DataFrame(Fv).rolling(10, min_periods=1).sum().values
need = np.where(early, 5, 3)[:, None]
trig = (run >= need) | (c10 >= 6)
# 同股 10 日內只算第一次
ev = []
for j in range(n_s):
    last = -10 ** 9
    for t in np.where(trig[:, j])[0]:
        if t - last >= 10:
            # 只算新事件的第一天（觸發條件剛成立）
            ev.append((t, j)); last = t
ev_t = np.array([t for t, j in ev])
st_idx = ev_t + 1
st_idx = st_idx[st_idx < n_t]
st_dates = cal[st_idx]
K = pd.Series(1, index=st_dates).groupby(st_dates.to_period('M')).sum()

# Y_m
H = pd.read_csv(BASE + 'raw/lineage/disposal_hist_rejudge.csv', dtype={'code': str, 'name': str}, keep_default_na=False)
C = pd.read_csv(OUT + 'r1_universe.csv', dtype={'code': str})
C = C.merge(H[['code', 'start', 'in_panel']], on=['code', 'start'], how='left')
assert C.in_panel.notna().all()
Cp = C[C.in_panel.astype(str) == 'True']
Y = Cp.groupby(pd.to_datetime(Cp.start).dt.to_period('M')).size()

months = pd.period_range('2004-01', '2026-10', freq='M')
D = pd.DataFrame(index=months)
D['K'] = K.reindex(months).fillna(0)
D['S'] = F.sum(axis=1).groupby(cal.to_period('M')).sum().reindex(months).fillna(0)
D['Y'] = Y.reindex(months).fillna(0)
ts = D.index.to_timestamp()
D['R14'] = (ts >= '2014-04-01').astype(float)
D['R15'] = (ts >= '2015-06-01').astype(float)
D['R17'] = (ts >= '2017-05-01').astype(float)
D['COVID'] = ((ts >= '2020-03-01') & (ts <= '2020-12-01')).astype(float)
D['lK'] = np.log1p(D.K)
fit = ((ts >= '2006-01-01') & (ts <= '2018-12-01')) | ((ts >= '2020-01-01') & (ts <= '2023-12-01'))
cols = ['const', 'lK', 'R14', 'R15', 'R17', 'COVID']
D['const'] = 1.0
X = D.loc[fit, cols].values; y = D.loc[fit, 'Y'].values


def nb_ll(p, X, y):
    b, la = p[:-1], p[-1]
    a = math.exp(la); mu = np.exp(X @ b); r = 1 / a
    return -np.sum(special.gammaln(y + r) - special.gammaln(r) - special.gammaln(y + 1)
                   + r * np.log(r / (r + mu)) + y * np.log(mu / (r + mu)))


def pois_ll(b, X, y):
    mu = np.exp(X @ b)
    return -np.sum(y * np.log(mu) - mu - special.gammaln(y + 1))


def num_hess(f, p, h=1e-4):
    k = len(p); Hm = np.zeros((k, k))
    for i in range(k):
        for j in range(k):
            pp = p.copy(); pp[i] += h; pp[j] += h; f1 = f(pp)
            pp = p.copy(); pp[i] += h; pp[j] -= h; f2 = f(pp)
            pp = p.copy(); pp[i] -= h; pp[j] += h; f3 = f(pp)
            pp = p.copy(); pp[i] -= h; pp[j] -= h; f4 = f(pp)
            Hm[i, j] = (f1 - f2 - f3 + f4) / (4 * h * h)
    return Hm

b0 = np.linalg.lstsq(X, np.log(y + 0.5), rcond=None)[0]
rp = optimize.minimize(pois_ll, b0, args=(X, y), method='BFGS')
r = optimize.minimize(nb_ll, np.r_[rp.x, math.log(0.2)], args=(X, y), method='BFGS', options={'gtol': 1e-8})
r = optimize.minimize(nb_ll, r.x, args=(X, y), method='Nelder-Mead', options={'xatol': 1e-10, 'fatol': 1e-10, 'maxiter': 40000})
p = r.x
Hs = num_hess(lambda q: nb_ll(q, X, y), p)
V = np.linalg.inv(Hs)
se = np.sqrt(np.diag(V))
aic = 2 * r.fun + 2 * len(p)
alpha = math.exp(p[-1])
print('coef', dict(zip(cols + ['logalpha'], np.round(p, 3))), 'se', np.round(se, 3), 'alpha', round(alpha, 3), 'AIC', round(aic, 1))
print('Poisson AIC', round(2 * rp.fun + 2 * len(rp.x), 1))

# 逐年預測與 PI（多元常態抽樣 × NB 月抽樣）
rng = np.random.default_rng(20261109)
NS = 10000
draws = rng.multivariate_normal(p, V, NS)
res_year = {}
for yr in range(2004, 2027):
    m = (ts.year == yr)
    if yr == 2026: m &= (ts <= '2026-10-01')
    if yr == 2004: m &= (ts >= '2004-07-01')  # 研究員 2004 只用 6 個月（特徵暖機）
    Xy = D.loc[m, cols].values
    obs = int(D.loc[m, 'Y'].sum())
    mu_pt = np.exp(Xy @ p[:-1])
    E = float(mu_pt.sum())
    mus = np.exp(draws[:, :-1] @ Xy.T)          # NS × months
    al = np.exp(draws[:, -1])[:, None]
    rr = 1 / al
    lam = rng.gamma(rr, mus / rr)
    sim = rng.poisson(lam).sum(axis=1)
    lo, hi = np.percentile(sim, [2.5, 97.5])
    sd = sim.std()
    z = (obs - sim.mean()) / sd
    q = float((sim <= obs).mean())
    # 年層級變異（點估計下）：sum(mu + a mu^2)
    var_y = float(np.sum(mu_pt + alpha * mu_pt ** 2))
    res_year[yr] = dict(obs=obs, E=round(E, 2), sim_mean=round(float(sim.mean()), 2), PI=[float(lo), float(hi)],
                        z=round(float(z), 2), q=q, var_y=var_y, z_pt=(obs - E) / math.sqrt(var_y),
                        fit=bool(fit[m].all()), K=int(D.loc[m, 'K'].sum()))
fy = [y_ for y_ in res_year if res_year[y_]['fit'] or y_ in (2018,)]
fit_years = [y_ for y_ in range(2006, 2024) if y_ != 2019]
phi = float(np.mean([res_year[y_]['z_pt'] ** 2 for y_ in fit_years]))
phi_sim = float(np.mean([res_year[y_]['z'] ** 2 for y_ in fit_years]))
r19 = res_year[2019]
cal_sd = math.sqrt(r19['var_y'] * phi)
cal_int = [r19['E'] - 1.96 * cal_sd, r19['E'] + 1.96 * cal_sd]
z_cal = (r19['obs'] - r19['E']) / cal_sd
outside = [y_ for y_ in fit_years if not (res_year[y_]['PI'][0] <= res_year[y_]['obs'] <= res_year[y_]['PI'][1])]
# 2019-06 P(0)
m6_ = D.loc[pd.Period('2019-06', 'M')]
mu6 = float(np.exp(m6_[cols].values.astype(float) @ p[:-1]))
p0 = (1 / (1 + alpha * mu6)) ** (1 / alpha)
# 月殘差自相關
mu_fit = np.exp(X @ p[:-1]); pr = (y - mu_fit) / np.sqrt(mu_fit + alpha * mu_fit ** 2)
ac1 = float(np.corrcoef(pr[:-1], pr[1:])[0, 1])

out = dict(coef=dict(zip(cols, p[:-1].round(4).tolist())), se=dict(zip(cols, se[:-1].round(4).tolist())),
           alpha=alpha, aic=aic, n_fit=int(fit.sum()), phi=phi, phi_sim=phi_sim, cal_int_2019=cal_int, z_cal_2019=z_cal,
           y2019=r19, outside_fit=outside, jun2019=dict(obs=int(m6_.Y), mu=mu6, P0=p0), ac1=ac1,
           yearly={y_: dict(obs=v['obs'], E=v['E'], PI=v['PI'], z=v['z'], q=v['q'], K=v['K']) for y_, v in res_year.items()})
json.dump(out, open(OUT + 'r3_model%s.json' % TAG, 'w'), ensure_ascii=False, indent=1, default=float)
D.to_csv(OUT + 'r3_monthly%s.csv' % TAG)
print(json.dumps({k: out[k] for k in ('alpha', 'aic', 'n_fit', 'phi', 'phi_sim', 'cal_int_2019', 'z_cal_2019', 'y2019', 'outside_fit', 'jun2019', 'ac1')}, indent=1, default=float))
for y_, v in res_year.items():
    print(y_, v['obs'], v['E'], v['PI'], v['z'], round(v['q'], 4), v['K'])

# 事件輸出（給 r4 精確率／強觸發用）
COLS = list(adj.columns)
pd.DataFrame([(COLS[j], str(cal[t + 1].date())) for t, j in ev if t + 1 < n_t], columns=['code', 'astart']).to_csv(OUT + 'r3_events%s.csv' % TAG, index=False)
Fa = fa.values.astype(np.int8)
runa = np.zeros_like(Fa, dtype=np.int32)
for t in range(n_t):
    runa[t] = (runa[t - 1] + 1) * Fa[t] if t else Fa[t]
sv = []
for j in range(n_s):
    last = -10 ** 9
    for t in np.where(runa[:, j] >= 3)[0]:
        if t - last >= 10:
            sv.append((COLS[j], str(cal[t].date()), str(cal[t + 1].date()) if t + 1 < n_t else '')); last = t
pd.DataFrame(sv, columns=['code', 'tday', 'astart']).to_csv(OUT + 'r3_strong%s.csv' % TAG, index=False)
