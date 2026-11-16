#!/usr/bin/env python3
"""S25 A 部分（描述，不改策略）：規格見 SPEC.json（sha256 必須相符）。
A1 台股因子動能（s24 monthly.csv）；A2 產業動能 vs 因子動能（PIT 產業）；A3 扣同業籃子的逐筆選股報酬（現行 E0／W10_70 主跑）。
輸出 results_describe.json、peer_trades.csv.gz。python3 -I -B describe.py"""
import sys, json, hashlib, time, resource, gc
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
import s25lib
from s25lib import s17lib, s10lib, s4lib, stlib, COSTS

H = Path(__file__).resolve().parent
assert hashlib.sha256((H / 'SPEC.json').read_bytes()).hexdigest() == (H / 'SPEC.sha256').read_text().split()[0], 'SPEC 不符'
t_all = time.time()
rss = lambda: round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
R = {'SPEC_sha256': (H / 'SPEC.sha256').read_text().split()[0]}


# ---------------------------------------------------------------- 統計工具
def tstat(x):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    if len(x) < 3:
        return float('nan')
    return float(x.mean() / (x.std(ddof=1) / np.sqrt(len(x))))


def welch(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    if len(a) < 3 or len(b) < 3:
        return float('nan')
    return float((a.mean() - b.mean()) / np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b)))


def nw_ols(y, X, lag=4):
    """OLS＋Newey-West 標準誤。X 不含常數。回傳 dict(coef, t, r2, n)；coef[0]＝常數。"""
    y = np.asarray(y, float); X = np.column_stack([np.ones(len(y))] + [np.asarray(c, float) for c in X])
    ok = np.isfinite(y) & np.isfinite(X).all(axis=1)
    y, X = y[ok], X[ok]
    n, k = X.shape
    XtX_inv = np.linalg.inv(X.T @ X)
    b = XtX_inv @ X.T @ y
    e = y - X @ b
    S = (X * e[:, None]).T @ (X * e[:, None])
    for L in range(1, lag + 1):
        w = 1 - L / (lag + 1)
        G = (X[L:] * e[L:, None]).T @ (X[:-L] * e[:-L, None])
        S += w * (G + G.T)
    V = XtX_inv @ S @ XtX_inv
    se = np.sqrt(np.diag(V))
    r2 = 1 - (e @ e) / (((y - y.mean()) ** 2).sum())
    return dict(coef=[float(v) for v in b], t=[float(v) for v in b / se], r2=float(r2), n=int(n))


def sharpe(x):
    x = np.asarray(x, float)
    return float(x.mean() / x.std(ddof=1) * np.sqrt(12))


def boot_sharpe_diff(a, b, n=2000, block=6, seed=20261116):
    a = np.asarray(a, float); b = np.asarray(b, float); T = len(a)
    rng = np.random.default_rng(seed); out = []
    nb = int(np.ceil(T / block))
    for _ in range(n):
        st = rng.integers(0, T - block + 1, nb)
        idx = (st[:, None] + np.arange(block)[None, :]).ravel()[:T]
        out.append(sharpe(a[idx]) - sharpe(b[idx]))
    out = np.array(out)
    return dict(diff=sharpe(a) - sharpe(b), ci90=[float(np.quantile(out, 0.05)), float(np.quantile(out, 0.95))], p_le0=float((out <= 0).mean()))


def ann(x):
    x = np.asarray(x, float)
    return dict(mean_ann_pct=float(x.mean() * 1200), vol_ann_pct=float(x.std(ddof=1) * np.sqrt(12) * 100), sharpe=sharpe(x), t_mean=tstat(x), n=int(len(x)))


# ---------------------------------------------------------------- A1 因子動能
FD = s25lib.load_factors()
FAC = s25lib.FACTORS


def a1(FD, n=12, mask=None):
    out = {'per_factor': {}}
    rows = []
    for f in FAC:
        r = FD[f]; tr = s25lib.factor_trend(r, n).shift(1)          # 月 t 用到 t−1 為止的趨勢
        d = pd.DataFrame({'r': r, 'tr': tr}).dropna()
        if mask is not None:
            d = d[mask(d.index)]
        w = d.loc[d['tr'] > 0, 'r']; l = d.loc[d['tr'] <= 0, 'r']
        out['per_factor'][f] = dict(win_mean_pct=float(w.mean() * 100), win_t=tstat(w), win_n=int(len(w)),
                                    lose_mean_pct=float(l.mean() * 100) if len(l) else None, lose_t=tstat(l), lose_n=int(len(l)),
                                    diff_pct=float((w.mean() - l.mean()) * 100) if len(l) else None, diff_welch_t=welch(w, l),
                                    uncond_mean_pct=float(d['r'].mean() * 100), uncond_t=tstat(d['r']), first=str(d.index[0]), last=str(d.index[-1]))
        rows.append(d.assign(f=f))
    A = pd.concat(rows)
    w = A[A['tr'] > 0]; l = A[A['tr'] <= 0]
    g = A.groupby([A.index, A['tr'] > 0])['r'].mean().unstack()
    g = g.dropna()
    out['pooled'] = dict(win_mean_pct=float(w['r'].mean() * 100), win_n=int(len(w)), lose_mean_pct=float(l['r'].mean() * 100), lose_n=int(len(l)),
                         diff_pct=float((w['r'].mean() - l['r'].mean()) * 100), diff_t_monthly_ts=tstat(g[True] - g[False]), n_months_both=int(len(g)),
                         note='講義（美國 1964～2019）：贏家 0.51%／月、輸家 0.06%／月')
    # TSFM
    R_ = pd.DataFrame({f: FD[f] for f in FAC}); TR = pd.DataFrame({f: s25lib.factor_trend(FD[f], n).shift(1) for f in FAC})
    ok = R_.notna().all(axis=1) & TR.notna().all(axis=1)
    if mask is not None:
        ok &= mask(R_.index)
    R_, TR = R_[ok], TR[ok]
    pos = TR > 0
    win = (R_.where(pos)).mean(axis=1).fillna(0.0)
    los = (R_.where(~pos)).mean(axis=1)
    wml = np.where(los.notna() & pos.any(axis=1), win - los.fillna(0), np.where(pos.any(axis=1), win, -los))
    wml = pd.Series(wml, index=R_.index)
    ew = R_.mean(axis=1)
    out['tsfm'] = dict(months=[str(R_.index[0]), str(R_.index[-1])], n=int(len(R_)), avg_n_winners=float(pos.sum(axis=1).mean()),
                       WIN=ann(win), EW=ann(ew), WML=ann(wml), LOSERS_only=ann(los.dropna()),
                       WIN_minus_EW_sharpe=boot_sharpe_diff(win.to_numpy(), ew.to_numpy()),
                       WIN_on_EW=nw_ols(win.to_numpy(), [ew.to_numpy()]),
                       share_months_factor_winner={f: float(pos[f].mean()) for f in FAC})
    return out, dict(WIN=win, WML=wml, EW=ew)


A1, TS = a1(FD, 12)
R['A1'] = A1
R['A1_robust'] = {'trend6': a1(FD, 6)[0], 'ex2020': a1(FD, 12, lambda ix: ix.year != 2020)[0], 'since2019': a1(FD, 12, lambda ix: ix >= pd.Period('2019-01', 'M'))[0]}
for k in R['A1_robust']:
    R['A1_robust'][k].pop('per_factor', None) if k != 'trend6' else None
# 策略月超額依因子趨勢（描述；PREREG 已寫完才跑）
st = {}
for f in ('MOM', 'RMOM'):
    tr = s25lib.factor_trend(FD[f]).shift(1)
    d = pd.DataFrame({'ex': FD['R_FULL'] - FD['R_0050'], 'full': FD['R_FULL'], 'tr': tr}).dropna()
    w = d[d['tr'] > 0]; l = d[d['tr'] <= 0]
    st[f] = dict(win_ex_mean_pct=float(w['ex'].mean() * 100), win_n=int(len(w)), lose_ex_mean_pct=float(l['ex'].mean() * 100), lose_n=int(len(l)),
                 diff_welch_t=welch(w['ex'], l['ex']), win_full_mean_pct=float(w['full'].mean() * 100), lose_full_mean_pct=float(l['full'].mean() * 100),
                 note='月報酬依「上月底已知的因子 12 月趨勢」分組（R_FULL＝現行 W10_70 月報酬，月份＝日曆月，不是換股月）')
R['A1_strategy_by_trend'] = st
print('A1 完成', f'{time.time()-t_all:.0f}s', flush=True)

# ---------------------------------------------------------------- 面板
P = s25lib.load()
assert (P.T, P.N, str(P.IDX[-1].date())) == (2876, 2142, '2026-10-22'), 'P0 失敗'
INNOV, _ = s4lib.innov_panel(P)
U = stlib.universe(P, 'twse') & ~INNOV
RT = s25lib.surge(P)
s17lib.e50_fix_suspension(P)
MC = s25lib.mcap_panel(P)
IDX = pd.DatetimeIndex(P.IDX)
print('載入完成', f'{time.time()-t_all:.0f}s', rss(), 'MB', flush=True)

# ---------------------------------------------------------------- A2 產業動能
me_rows = pd.Series(np.arange(P.T), index=IDX).groupby(IDX.to_period('M')).max()
me_rows = me_rows[me_rows.index >= pd.Period('2015-01', 'M')]
months = list(me_rows.index)
INDm, NAMES = s25lib.industry_matrix(P, list(me_rows.values))
Gme = P.G[me_rows.values]
IR = {'VW': {}, 'EW': {}}; NIND = []
for i in range(len(months) - 1):
    t = int(me_rows.iloc[i]); m1 = months[i + 1]
    ret = Gme[i + 1] / Gme[i] - 1
    mc = MC[t].astype(float)
    elig = (P.MK[t] == 1) & P.has[t] & P.not91 & ~INNOV[t] & np.isfinite(mc) & (mc > 0) & np.isfinite(ret)
    ind = INDm[t]
    vw = {}; ew = {}
    for g in np.unique(ind[elig & (ind >= 0)]):
        msk = elig & (ind == g)
        if msk.sum() < 5:
            continue
        vw[NAMES[g]] = float(np.sum(mc[msk] * ret[msk]) / np.sum(mc[msk])); ew[NAMES[g]] = float(ret[msk].mean())
    IR['VW'][m1] = vw; IR['EW'][m1] = ew; NIND.append(len(vw))


def indmom(IRw, form=6, nlong=3):
    D = pd.DataFrame(IRw).T.sort_index()
    D.index = pd.PeriodIndex(D.index, freq='M')
    cum = (1 + D).rolling(form, min_periods=form).apply(np.prod, raw=True) - 1
    out = {}
    for i in range(len(D.index) - 1):
        m, m1 = D.index[i], D.index[i + 1]
        c = cum.loc[m].dropna(); nxt = D.loc[m1]
        c = c[nxt.reindex(c.index).notna()]
        if len(c) < 2 * nlong + 2:
            continue
        o = c.sort_values()
        out[m1] = float(nxt[o.index[-nlong:]].mean() - nxt[o.index[:nlong]].mean())
    return pd.Series(out)


A2 = dict(n_industries=dict(min=int(min(NIND)), median=float(np.median(NIND)), max=int(max(NIND))))
FDm = FD.copy()
for wt in ('VW', 'EW'):
    im = indmom(IR[wt])
    df = pd.DataFrame({'IND': im, 'WML': TS['WML'], 'WIN': TS['WIN'], 'MKT': FD['MKT_VW'], 'MOM': FD['MOM']}).dropna()
    o = dict(months=[str(df.index[0]), str(df.index[-1])], n=int(len(df)), INDMOM=ann(df['IND']), FMOM_WML=ann(df['WML']),
             corr_IND_WML=float(df['IND'].corr(df['WML'])), corr_IND_MOM=float(df['IND'].corr(df['MOM'])), corr_WML_MOM=float(df['WML'].corr(df['MOM'])))
    reg = {}
    reg['IND_on_const'] = nw_ols(df['IND'], [])
    reg['IND_on_WML'] = nw_ols(df['IND'], [df['WML']])
    reg['IND_on_WML_MKT'] = nw_ols(df['IND'], [df['WML'], df['MKT']])
    reg['IND_on_WML_MKT_MOM'] = nw_ols(df['IND'], [df['WML'], df['MKT'], df['MOM']])
    reg['IND_on_WIN_MKT'] = nw_ols(df['IND'], [df['WIN'], df['MKT']])
    reg['WML_on_const'] = nw_ols(df['WML'], [])
    reg['WML_on_IND'] = nw_ols(df['WML'], [df['IND']])
    reg['WML_on_IND_MKT'] = nw_ols(df['WML'], [df['IND'], df['MKT']])
    reg['WML_on_IND_MKT_MOM'] = nw_ols(df['WML'], [df['IND'], df['MKT'], df['MOM']])
    reg['IND_on_MOM_MKT'] = nw_ols(df['IND'], [df['MOM'], df['MKT']])
    for v in reg.values():
        v['alpha_ann_pct'] = v['coef'][0] * 1200; v['alpha_t'] = v['t'][0]
    o['regs'] = reg
    o['regs_cols'] = '每個回歸 coef／t 依序：常數、再依名稱中的變數順序'
    A2[wt] = o
R['A2'] = A2
del INDm, Gme
gc.collect()
print('A2 完成', f'{time.time()-t_all:.0f}s', rss(), 'MB', flush=True)

# ---------------------------------------------------------------- A3 扣同業
for a in ('H', 'L', 'VOL', 'CHG', 'HF', 'VOLF', 'adjH', 'HI240', 'MA60', 'nbar', 'F_HI', 'IDXC', 'AO', 'listed', 'NEARHI', 'V60', 'liq'):
    if hasattr(P, a):
        delattr(P, a)
gc.collect()
S0 = s10lib.make_signal_s10(P, U, RT, shift=0)
S2, ekw = s10lib.variant_S(S0, 'E0')
nav, lg, _ = s10lib.simulate_s10(P, S2, U, sat_w=0.7, sizing='W10', lock=False, cost=COSTS['strict'], capital=1e6, **ekw)
lastb = lambda d: P.IDX[P.IDX < pd.Timestamp(d)][-1]
a0 = dict(since2019=stlib.cagr(nav, lastb('2019-01-01'), P.IDX[-1]), full=stlib.cagr(nav, P.IDX[P.t0], P.IDX[-1]), maxdd=stlib.maxdd(nav, P.IDX[P.t0], P.IDX[-1]), n_trades=int(len(lg)))
a0['ok'] = bool(abs(a0['since2019'] - 0.4813) < 0.00025 and abs(a0['full'] - 0.3408) < 0.00025 and abs(a0['maxdd'] + 0.3340) < 0.00025)
assert a0['ok'], a0
R['A3_acceptance'] = a0
LL = s25lib.last_listed(P)
sigks = sorted(set(int(k) for k in lg['sig_t']))
INDk, _ = s25lib.industry_matrix(P, sigks)
BROAD = (P.MK == 1) & ~INNOV & P.not91[None, :]
rows = []
for y in lg.itertuples():
    j, k, e = int(y.j), int(y.sig_t), int(y.ent_t)
    x = None if y.why == 'open_end' else int(y.exit_t)
    r = s25lib.hold_ret(P, j, e, x, LL); r50 = s25lib.e50_ret(P, e, x)
    ind = INDk[k]; g = ind[j]
    out = dict(code=str(P.COLS[j]), sig_date=str(IDX[k].date()), ent_date=str(IDX[e].date()), exit_date=str(IDX[int(y.exit_t)].date()), why=y.why,
               ret_engine=float(y.ret), ret=r, r50=r50, industry=NAMES[g] if g >= 0 else '')
    for nm, pool in (('U', U[k]), ('BROAD', BROAD[k])):
        if g < 0:
            out.update({f'n_{nm}': 0, f'peer_{nm}': np.nan, f'peerVW_{nm}': np.nan, f'big_{nm}': ''}); continue
        cand = np.flatnonzero(pool & (ind == g))
        cand_all = np.union1d(cand, [j])
        mcs = np.nan_to_num(MC[k, cand_all].astype(float), nan=-1)
        big = int(cand_all[np.argmax(mcs)])
        peers = [p for p in cand if p != j and p != big and np.isfinite(P.O[e, p]) and P.O[e, p] > 0]
        pr = np.array([s25lib.hold_ret(P, p, e, x, LL) for p in peers], float)
        okp = np.isfinite(pr); pr = pr[okp]; pw = np.nan_to_num(MC[k, np.array(peers, int)[okp]].astype(float), nan=0.0) if len(peers) else np.array([])
        out[f'n_{nm}'] = int(len(pr)); out[f'big_{nm}'] = str(P.COLS[big])
        out[f'peer_{nm}'] = float(pr.mean()) if len(pr) >= 5 else np.nan
        out[f'peerVW_{nm}'] = float(np.sum(pw * pr) / np.sum(pw)) if len(pr) >= 5 and pw.sum() > 0 else np.nan
    rows.append(out)
T3 = pd.DataFrame(rows)
T3['month'] = T3['sig_date'].str[:7]
T3['ex50'] = T3['ret'] - T3['r50']
for nm in ('U', 'BROAD'):
    T3[f'exP_{nm}'] = T3['ret'] - T3[f'peer_{nm}']
    T3[f'exPVW_{nm}'] = T3['ret'] - T3[f'peerVW_{nm}']
    T3[f'peer50_{nm}'] = T3[f'peer_{nm}'] - T3['r50']


def stats(df, col):
    d = df[np.isfinite(df[col])]
    v = d[col].to_numpy()
    if len(v) < 3:
        return dict(n=int(len(v)))
    mm = d.groupby('month')[col].mean()
    top = d.nlargest(5, col)
    dd = d.drop(top.index)
    mm2 = dd.groupby('month')[col].mean()
    return dict(n=int(len(v)), n_months=int(len(mm)), mean_pct=float(v.mean() * 100), median_pct=float(np.median(v) * 100), win_rate=float((v > 0).mean()),
                t=tstat(v), t_cluster_month=tstat(mm.to_numpy()), mean_month_avg_pct=float(mm.mean() * 100),
                drop_top5_mean_pct=float(dd[col].mean() * 100), drop_top5_median_pct=float(dd[col].median() * 100), drop_top5_t_cluster=tstat(mm2.to_numpy()),
                top5=[dict(code=a.code, sig=a.sig_date, val_pct=float(getattr(a, col) * 100)) for a in top.itertuples()])


A3 = {'n_trades': int(len(T3)), 'gross_vs_engine_ret_corr': float(np.corrcoef(T3['ret'], T3['ret_engine'])[0, 1]),
      'gross_minus_engine_mean_pct': float((T3['ret'] - T3['ret_engine']).mean() * 100), 'industry_missing': int((T3['industry'] == '').sum())}
for per, msk in (('all', np.ones(len(T3), bool)), ('train', (T3['sig_date'] < '2022-01-01').to_numpy()), ('valid', (T3['sig_date'] >= '2022-01-01').to_numpy()),
                 ('since2019', (T3['sig_date'] >= '2019-01-01').to_numpy())):
    d = T3[msk]
    o = {}
    okU = np.isfinite(d['peer_U'])
    o['n_peer_ge5'] = int(okU.sum()); o['n_peer_lt5'] = int((~okU).sum())
    o['main（同業≥5，U 股池等權）'] = {c: stats(d[okU], c) for c in ('ret', 'r50', 'peer_U', 'ex50', 'exP_U', 'peer50_U')}
    o['peer_lt5'] = {c: stats(d[~okU], c) for c in ('ret', 'ex50')}
    o['all_trades_ex50'] = stats(d, 'ex50')
    okB = np.isfinite(d['peer_BROAD'])
    o['robust_BROAD'] = dict(n=int(okB.sum()), exP=stats(d[okB], 'exP_BROAD'), ex50_same=stats(d[okB], 'ex50'))
    o['robust_VW_U'] = stats(d[okU], 'exPVW_U')
    A3[per] = o
# 產業別
g = T3[np.isfinite(T3['peer_U'])].groupby('industry')
A3['by_industry'] = {k: dict(n=int(len(v)), ex50_mean_pct=float(v['ex50'].mean() * 100), exP_mean_pct=float(v['exP_U'].mean() * 100),
                             peer50_mean_pct=float(v['peer50_U'].mean() * 100), n_peers_median=float(v['n_U'].median()))
                     for k, v in sorted(g, key=lambda kv: -len(kv[1]))}
A3['peer_count_U'] = dict(median=float(T3['n_U'].median()), q10=float(T3['n_U'].quantile(0.1)), q90=float(T3['n_U'].quantile(0.9)))
A3['biggest_excluded_top'] = T3['big_U'].value_counts().head(10).to_dict()
R['A3'] = A3
T3.drop(columns=[c for c in T3.columns if c.startswith('big_BROAD')]).to_csv(H / 'peer_trades.csv.gz', index=False, float_format='%.6f')
R['meta'] = dict(runtime_s=round(time.time() - t_all, 1), maxrss_mb=rss())
(H / 'results_describe.json').write_text(json.dumps(R, ensure_ascii=False, indent=1, default=str))
print(json.dumps(R['meta']))
