# R98 獨立重算：V_BRIDGE 交易統計、相依性、厚尾。執行：python3 -I -B r98_stats.py
import json, os, numpy as np, pandas as pd
from scipy import stats
HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, '..', 's75_continuous_2008_2026', 'v3_2026-11-25_limit7')
SEED = 98; NB = 10000
R = {}
tr = pd.read_csv(os.path.join(SRC, 'trades_v3.csv.gz'))
tr = tr[tr.version == 'V_BRIDGE'].reset_index(drop=True)
nav = pd.read_csv(os.path.join(SRC, 'nav_2008_2026_v3.csv.gz'))
nav['date'] = pd.to_datetime(nav['date']); nav = nav.set_index('date')
for c in ('sig_date', 'ent_date', 'exit_date'): tr[c] = pd.to_datetime(tr[c])
b = nav['0050']
def tstat(x):
    x = np.asarray(x, float); n = len(x); m = x.mean(); s = x.std(ddof=1)
    return dict(n=int(n), mean=float(m), sd=float(s), t=float(m / s * np.sqrt(n)))
# ---------- 1. 逐筆超額 ----------
tr['bench'] = b.reindex(tr.exit_date).to_numpy() / b.reindex(tr.ent_date).to_numpy() - 1      # 總監定義：進場日收盤→出場日收盤
tr['bench_sig'] = b.reindex(tr.exit_date).to_numpy() / b.reindex(tr.sig_date).to_numpy() - 1  # 另版：訊號日收盤→出場日收盤（含進場日當天 0050 漲跌）
tr['ex'] = tr.ret - tr.bench; tr['ex_sig'] = tr.ret - tr.bench_sig
tr['ex_log'] = np.log1p(tr.ret) - np.log1p(tr.bench_sig)
tr['ym'] = tr.ent_date.dt.to_period('M'); tr['yr'] = tr.ent_date.dt.year
C1, C2 = pd.Timestamp('2015-07-01'), pd.Timestamp('2019-01-01')
seg = {'all': tr.index == tr.index, 'pre_2015H2': tr.ent_date < C1, '2015H2_2018': (tr.ent_date >= C1) & (tr.ent_date < C2), '2019_on': tr.ent_date >= C2}
R['per_trade_excess'] = {k: tstat(tr.ex[m]) for k, m in seg.items()}
R['per_trade_excess_alt_sigclose'] = {k: tstat(tr.ex_sig[m]) for k, m in seg.items()}
R['per_trade_excess_alt_log'] = {k: tstat(tr.ex_log[m]) for k, m in seg.items()}
closed = tr.why != 'open_end'
R['per_trade_excess_closed_only'] = tstat(tr.ex[closed]); R['n_open'] = int((~closed).sum())
R['raw_ret'] = dict(tstat(tr.ret), median=float(tr.ret.median()), median_excess=float(tr.ex.median()), frac_excess_pos=float((tr.ex > 0).mean()))
# 金額加權（各筆部位大小不同）
w = tr.cost / tr.cost.sum()
R['cost_weighted'] = dict(mean_excess=float((w * tr.ex).sum()), mean_ret=float((w * tr.ret).sum()),
                          excess_pnl_sum=float((tr.cost * tr.ex).sum()), pnl_sum=float(tr.pnl.sum()))
# 去掉 ret 最高 5%
def trim(k, by='ret'):
    keep = tr.sort_values(by, ascending=False, kind='stable').index[k:]
    return dict(k_removed=int(k), by=by, excess=tstat(tr.ex[keep]), raw=tstat(tr.ret[keep]))
n = len(tr); k5 = int(np.ceil(n * 0.05))
R['trim_top5pct_by_ret'] = trim(k5)
R['trim'] = {f'{p}%_by_{by}': trim(int(np.ceil(n * p / 100)), by) for p in (1, 2, 5, 10) for by in ('ret', 'ex')}
R['trim_symmetric'] = {}
for p in (1, 2, 5, 10):
    k = int(np.ceil(n * p / 100)); o = tr.sort_values('ex').index[k:n - k]
    R['trim_symmetric'][f'{p}%'] = dict(k_each_side=k, excess=tstat(tr.ex[o]))
def k_to_zero(by):
    o = tr.sort_values(by, ascending=False, kind='stable'); x = o.ex.to_numpy(); tot = x.sum()
    for k in range(n):
        if (tot - x[:k].sum()) / (n - k) <= 0: return k
R['k_remove_until_mean_excess_le0'] = {by: int(k_to_zero(by)) for by in ('ret', 'ex')}
def k_to_t(by, thr):
    o = tr.sort_values(by, ascending=False, kind='stable').ex.to_numpy()
    for k in range(n):
        if tstat(o[k:])['t'] < thr: return k
R['k_remove_until_t_below_2'] = {by: int(k_to_t(by, 2.0)) for by in ('ret', 'ex')}
# 進場月份分組
g = tr.groupby('ym').ex.mean()
R['entry_month_means'] = tstat(g.to_numpy())
# ---------- 2a. 群聚穩健 ----------
def cluster_t(x, cl):
    x = np.asarray(x, float); n = len(x); m = x.mean(); e = x - m
    s = pd.Series(e).groupby(np.asarray(cl)).sum().to_numpy(); G = len(s)
    v0 = (s ** 2).sum() / n ** 2; v1 = v0 * G / (G - 1)   # CR1：G/(G-1) 小樣本校正
    se = np.sqrt(v1); t = m / se
    return dict(n=int(n), G=int(G), mean=float(m), se_iid=float(x.std(ddof=1) / np.sqrt(n)), se_cluster=float(se), t=float(t),
                p_two_sided_tG1=float(2 * stats.t.sf(abs(t), G - 1)), ci95=[float(m - stats.t.ppf(.975, G - 1) * se), float(m + stats.t.ppf(.975, G - 1) * se)],
                design_effect=float(v1 / (x.var(ddof=1) / n)))
tr['yq'] = tr.ent_date.dt.to_period('Q')
R['cluster'] = {}
for nm, col in (('entry_month', 'ym'), ('entry_quarter', 'yq'), ('entry_year', 'yr')):
    R['cluster'][nm] = {k: cluster_t(tr.ex[m], tr[col][m].astype(str)) for k, m in seg.items()}
R['cluster']['entry_month_alt_sigclose'] = cluster_t(tr.ex_sig, tr.ym.astype(str))
# 群聚拔靴（對進場月份整月重抽）
rng = np.random.default_rng(SEED)
gs = tr.groupby('ym').ex.agg(['sum', 'count']).to_numpy(); G = len(gs)
idx = rng.integers(0, G, size=(NB, G)); bm = gs[idx, 0].sum(1) / gs[idx, 1].sum(1)
R['cluster_bootstrap_entry_month'] = dict(B=NB, seed=SEED, ci95=[float(np.percentile(bm, 2.5)), float(np.percentile(bm, 97.5))], p_le0=float((bm <= 0).mean()), se=float(bm.std(ddof=1)))
# ---------- 月報酬 ----------
me = nav.groupby(nav.index.to_period('M')).last()
mr = me.pct_change().dropna()
mx = (mr['V_BRIDGE'] - mr['0050'])
lx = np.log(me['V_BRIDGE'] / me['0050']).diff().dropna()
def ir_stats(x):
    x = np.asarray(x, float); T = len(x); m = x.mean(); s = x.std(ddof=1)
    return dict(months=int(T), ann_excess_pp=float(m * 12 * 100), ann_te_pct=float(s * np.sqrt(12) * 100), ir=float(m / s * np.sqrt(12)), t=float(m / s * np.sqrt(T)))
P1, P2 = pd.Period('2015-07', 'M'), pd.Period('2019-01', 'M')
mseg = {'all': mx.index >= mx.index[0], '2008_2015H1': mx.index < P1, '2015H2_2018': (mx.index >= P1) & (mx.index < P2), '2019_on': mx.index >= P2}
R['monthly_arith'] = {k: ir_stats(mx[m]) for k, m in mseg.items()}
R['monthly_log'] = {k: ir_stats(lx[m]) for k, m in mseg.items()}
# 含 2008-01（由期初 100 萬起算）→ 226 個月
me0 = pd.concat([pd.DataFrame([nav.iloc[0]], index=[pd.Period('2007-12', 'M')]), me]); mr0 = me0.pct_change().dropna()
R['monthly_arith_incl_2008_01'] = ir_stats(mr0['V_BRIDGE'] - mr0['0050'])
yrs = (nav.index[-1] - nav.index[0]).days / 365.25
cg = lambda s: (s.iloc[-1] / s.iloc[0]) ** (1 / yrs) - 1
R['cagr'] = dict(years=float(yrs), V_BRIDGE=float(cg(nav['V_BRIDGE'])), e0050=float(cg(nav['0050'])), diff_pp=float((cg(nav['V_BRIDGE']) - cg(nav['0050'])) * 100),
                 log_excess_ann_pp=float(lx.mean() * 12 * 100))
# 年度超額（19 年）
ye = nav.groupby(nav.index.year).last(); y0 = pd.concat([pd.DataFrame([nav.iloc[0]], index=[2007]), ye]).pct_change().dropna()
yx = y0['V_BRIDGE'] - y0['0050']
R['yearly_excess'] = dict(tstat(yx.to_numpy()), wins=int((yx > 0).sum()), sign_test_p_two_sided=float(stats.binomtest(int((yx > 0).sum()), len(yx), 0.5).pvalue), by_year={int(k): float(v) for k, v in yx.items()})
# ---------- 2b. Newey-West ----------
def nw(x, L):
    x = np.asarray(x, float); T = len(x); m = x.mean(); e = x - m
    v = (e @ e) / T
    for l in range(1, L + 1): v += 2 * (1 - l / (L + 1)) * (e[l:] @ e[:-l]) / T
    se = np.sqrt(v / T * T / (T - 1)); return dict(lag=L, se_monthly=float(se), t=float(m / se), p_two_sided_normal=float(2 * stats.norm.sf(abs(m / se))),
                                                    ci95_ann_pp=[float((m - 1.96 * se) * 1200), float((m + 1.96 * se) * 1200)])
R['newey_west'] = {nm: {k: {f'L{L}': nw(x[m], L) for L in (0, 3, 6, 12)} for k, m in mseg.items()} for nm, x in (('arith', mx), ('log', lx))}
ac = lambda x, l: float(np.corrcoef(x[l:], x[:-l])[0, 1])
R['autocorr_monthly_excess'] = {f'lag{l}': ac(mx.to_numpy(), l) for l in (1, 2, 3, 6, 12)}
import statsmodels.api as sm
chk = sm.OLS(mx.to_numpy(), np.ones(len(mx))).fit(cov_type='HAC', cov_kwds=dict(maxlags=6, use_correction=True))
R['newey_west_statsmodels_check_L6_t'] = float(chk.tvalues[0])
# CAPM alpha（帳戶對 0050 的 beta 不一定是 1）
X = sm.add_constant(mr['0050'].to_numpy()); R['alpha_beta'] = {}
for L in (0, 3, 6, 12):
    f = sm.OLS(mr['V_BRIDGE'].to_numpy(), X).fit(cov_type='HAC', cov_kwds=dict(maxlags=L, use_correction=True)) if L else sm.OLS(mr['V_BRIDGE'].to_numpy(), X).fit()
    R['alpha_beta'][f'L{L}'] = dict(alpha_ann_pp=float(f.params[0] * 1200), beta=float(f.params[1]), t_alpha=float(f.tvalues[0]))
# ---------- 2c. 區塊拔靴 ----------
def block_boot(x, bl, seed, circular=True):
    x = np.asarray(x, float); T = len(x); rng = np.random.default_rng(seed); nb = int(np.ceil(T / bl))
    st = rng.integers(0, T if circular else T - bl + 1, size=(NB, nb))
    ix = (st[:, :, None] + np.arange(bl)[None, None, :]).reshape(NB, -1)[:, :T] % T
    s = x[ix]; m = s.mean(1); sd = s.std(1, ddof=1)
    ann = m * 1200; ir = m / sd * np.sqrt(12)
    q = lambda a: [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))]
    z = (m.mean() - 0) / m.std(ddof=1)
    return dict(block=bl, B=NB, seed=seed, circular=circular, ann_excess_pp_ci95=q(ann), ir_ci95=q(ir), p_excess_le0=float((m <= 0).mean()),
                boot_se_monthly=float(m.std(ddof=1)), implied_t=float(x.mean() / m.std(ddof=1)))
R['block_bootstrap'] = {nm: {k: {f'B{bl}': block_boot(x[m], bl, SEED + bl) for bl in (1, 6, 12)} for k, m in mseg.items()} for nm, x in (('arith', mx), ('log', lx))}
R['block_bootstrap_noncircular_all'] = {f'B{bl}': block_boot(mx, bl, SEED + bl, False) for bl in (6, 12)}
# ---------- 3. 厚尾 ----------
x = tr.ex.to_numpy()
R['shape'] = dict(skew=float(stats.skew(x, bias=False)), kurt_excess=float(stats.kurtosis(x, bias=False)), kurt_pearson=float(stats.kurtosis(x, fisher=False, bias=False)),
                  jarque_bera_p=float(stats.jarque_bera(x).pvalue), min=float(x.min()), max=float(x.max()),
                  pcts={str(p): float(np.percentile(x, p)) for p in (1, 5, 25, 50, 75, 95, 99)},
                  ret_skew=float(stats.skew(tr.ret, bias=False)), ret_kurt_excess=float(stats.kurtosis(tr.ret, bias=False)),
                  wilcoxon_p_two_sided=float(stats.wilcoxon(x).pvalue), sign_test_p=float(stats.binomtest(int((x > 0).sum()), n, 0.5).pvalue))
rng = np.random.default_rng(SEED + 1); ix = rng.integers(0, n, size=(NB, n)); s = x[ix]; bm = s.mean(1); bs = s.std(1, ddof=1)
tb = (bm - x.mean()) / (bs / np.sqrt(n)); se = x.std(ddof=1) / np.sqrt(n)
R['trade_bootstrap_iid'] = dict(B=NB, seed=SEED + 1, mean=float(x.mean()), ci95_percentile=[float(np.percentile(bm, 2.5)), float(np.percentile(bm, 97.5))],
                                ci95_bootstrap_t=[float(x.mean() - np.percentile(tb, 97.5) * se), float(x.mean() - np.percentile(tb, 2.5) * se)], p_le0=float((bm <= 0).mean()))
# ---------- 4. 逐年 ----------
TH = 0.576
yt = tr.groupby('yr').agg(n=('ex', 'size'), mean_excess=('ex', 'mean'), median_excess=('ex', 'median'), mean_ret=('ret', 'mean'), big=('ret', lambda r: int((r >= TH).sum())),
                          max_ret=('ret', 'max'), pnl=('pnl', 'sum')).reindex(range(2008, 2027)).fillna(0)
yt['n'] = yt['n'].astype(int); yt['big'] = yt['big'].astype(int); yt['big_rate'] = yt.big / yt.n
yt['acct_excess'] = yx.reindex(yt.index).to_numpy()
yt.to_csv(os.path.join(HERE, 'yearly_trades.csv'))
bigN = int(yt.big.sum()); lam = bigN / 19
R['yearly'] = dict(threshold=TH, big_total=bigN, years_zero=[int(y) for y in yt.index[yt.big == 0]], n_years_zero=int((yt.big == 0).sum()),
                   dist={int(k): int(v) for k, v in yt.big.value_counts().sort_index().items()},
                   big_by_seg={k: int((tr.ret[m] >= TH).sum()) for k, m in seg.items()}, big_rate_by_seg={k: float((tr.ret[m] >= TH).mean()) for k, m in seg.items()},
                   years_pos_mean_excess=int((yt.mean_excess > 0).sum()), poisson_expected_zero_years=float(19 * np.exp(-lam)),
                   var_over_mean=float(yt.big.var(ddof=1) / yt.big.mean()),
                   fisher_p_pre2019_vs_2019on=float(stats.fisher_exact([[int((tr.ret[tr.ent_date < C2] >= TH).sum()), int((tr.ent_date < C2).sum() - (tr.ret[tr.ent_date < C2] >= TH).sum())],
                                                                         [int((tr.ret[tr.ent_date >= C2] >= TH).sum()), int((tr.ent_date >= C2).sum() - (tr.ret[tr.ent_date >= C2] >= TH).sum())]])[1]),
                   table=json.loads(yt.reset_index().to_json(orient='records')))
# ---------- 需要筆數／年數 ----------
A = R['per_trade_excess']
need = lambda t, d: float((t * d['sd'] / d['mean']) ** 2)
R['need_trades'] = dict(all={f't{t}': need(t, A['all']) for t in (2, 3, 4.1)}, pre_2015H2_t2=need(2, A['pre_2015H2']),
                        all_cluster_adj={f't{t}': need(t, A['all']) * R['cluster']['entry_month']['all']['design_effect'] for t in (2, 3, 4.1)})
R['need_years'] = {f'IR{ir}': {f't{t}': float((t / ir) ** 2) for t in (1.65, 2)} for ir in (0.72, 0.5, 0.28)}
R['need_years']['IR_own'] = {k: {f't{t}': float((t / v['ir']) ** 2) for t in (1.65, 2, 3)} for k, v in R['monthly_arith'].items()}
# ---------- 補充：帳戶層級對單一年份／少數月份的敏感度 ----------
mxa = mx.to_numpy(); yrs_m = np.array([q.year for q in mx.index])
R['monthly_leave_one_year_out'] = {int(y): ir_stats(mxa[yrs_m != y]) for y in range(2008, 2027)}
o = np.sort(mxa)[::-1]
R['monthly_drop_best_k'] = {f'k{k}': ir_stats(o[k:]) for k in (1, 2, 3, 5, 10)}
R['monthly_k_best_removed_until_t_below_2'] = int(next(k for k in range(len(o)) if ir_stats(o[k:])['t'] < 2))
R['monthly_k_best_removed_until_mean_le0'] = int(next(k for k in range(len(o)) if o[k:].mean() <= 0))
R['monthly_shape'] = dict(skew=float(stats.skew(mxa, bias=False)), kurt_excess=float(stats.kurtosis(mxa, bias=False)), frac_pos=float((mxa > 0).mean()),
                          wilcoxon_p=float(stats.wilcoxon(mxa).pvalue), best_month=str(mx.idxmax()), best=float(mx.max()), worst_month=str(mx.idxmin()), worst=float(mx.min()))
# 以各段波動標準化後的 t（三段追蹤誤差差 4 倍，等權 t 被高波動段主導）
z = np.concatenate([mxa[m] / mxa[m].std(ddof=1) for k, m in mseg.items() if k != 'all'])
R['monthly_vol_standardized_t'] = float(z.mean() / z.std(ddof=1) * np.sqrt(len(z)))
tr.drop(columns=['ym', 'yq']).to_csv(os.path.join(HERE, 'trades_excess.csv'), index=False)
pd.DataFrame({'V_BRIDGE': mr['V_BRIDGE'], 'e0050': mr['0050'], 'excess': mx, 'log_excess': lx}).to_csv(os.path.join(HERE, 'monthly_excess.csv'))
json.dump(R, open(os.path.join(HERE, 'results.json'), 'w'), ensure_ascii=False, indent=1, default=float)
print(json.dumps(R, ensure_ascii=False, indent=1, default=float))
