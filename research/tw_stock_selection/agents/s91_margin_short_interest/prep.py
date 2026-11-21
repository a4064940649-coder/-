#!/usr/bin/env python3
"""S91 第 0 步（PREREG 之前）：驗收（現行標準路徑、20 條路徑、與網站 navs_official.csv.gz 逐日差）＋資料驗收（不看報酬）＋訓練期（訊號 2016-02～2021-12）報酬分組。
驗證期（2022 起）的交易報酬在本步驟完全不計算分組、不輸出。輸出 data_check.json。python3 -I -B prep.py"""
import sys, json, time, resource, gc
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
import s91lib as L
from s91lib import s10lib, stlib

H = Path(__file__).resolve().parent
t0 = time.time(); rss = lambda: round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
P, U, RT = L.load()
lastb = lambda d: P.IDX[P.IDX < pd.Timestamp(d)][-1]
D0, DE, D19 = P.IDX[P.t0], P.IDX[-1], lastb('2019-01-01')
res = dict(flags=dict(EXRIGHT_FIX=L.cslib.EXRIGHT_FIX, DEADLINE_FIX=L.cslib.DEADLINE_FIX, INS15_MASK=L.cslib.INS15_MASK, REDUCE_FIX=L.cslib.REDUCE_FIX, E50_SOURCE=str(L.cslib.E50_SOURCE)))

# ---- 驗收
S0 = s10lib.make_signal_s10(P, U, RT, shift=0)
nav, lg, st = L.sim(P, S0, U, cap=1e6)
lg['code'] = lg['code'].astype(str)
got = dict(since2019=stlib.cagr(nav, D19, DE), full=stlib.cagr(nav, D0, DE), maxdd=stlib.maxdd(nav, D0, DE), n_closed=int((lg['why'] != 'open_end').sum()), n_all=int(len(lg)))
site = pd.read_csv('/home/user/-/xq_scripts/site/navs_official.csv.gz', parse_dates=['date']).set_index('date')
mine = nav.loc[D0:DE].round(2)
got['nav_vs_site_same_dates'] = bool(list(mine.index) == list(site.index))
got['nav_vs_site_max_abs_diff'] = float((mine - site['E0|W10_70']).abs().max())
paths = {}
for s in L.SHIFTS:
    S = S0 if s == 0 else s10lib.make_signal_s10(P, U, RT, shift=s)
    for cap in L.CAPS:
        paths[f'{int(cap/1e4)}萬|+{s}'] = stlib.cagr(L.sim(P, S, U, cap=cap)[0], D19, DE)
got['paths_median'] = float(np.median(list(paths.values())))
T_ = L.TARGET
res['acceptance'] = dict(got=got, target=T_, ok=bool(all(abs(got[k] - T_[k]) < 1e-9 for k in ('since2019', 'full', 'maxdd', 'paths_median')) and got['n_closed'] == T_['n_closed']
                                                      and got['n_all'] == T_['n_all'] and got['nav_vs_site_same_dates'] and got['nav_vs_site_max_abs_diff'] == 0))
print('驗收', res['acceptance'], f'{time.time()-t0:.0f}s', rss(), 'MB', flush=True)
assert res['acceptance']['ok']

# ---- 資料驗收（不看報酬）
CH = L.Chips(P)
print('籌碼面板', f'{time.time()-t0:.0f}s', rss(), 'MB', flush=True)
ks = np.array([x['k'] for x in S0['info']])
dc = {}
anyrow = np.isfinite(CH.MB).any(axis=1); anysbl = np.isfinite(CH.SBL).any(axis=1)
dc['panel_days'] = int(P.T); dc['days_without_any_margin_row'] = [str(P.IDX[t].date()) for t in np.flatnonzero(~anyrow)]
dc['days_without_any_sbl_row'] = [str(P.IDX[t].date()) for t in np.flatnonzero(~anysbl)]
dc['first_panel_day'] = str(P.IDX[0].date())
Uall = U[P.t0:]
dc['universe_cell_coverage_all_days'] = dict(margin_row=float(np.isfinite(CH.MB[P.t0:])[Uall].mean()), margin_limit_gt0=float((CH.ML[P.t0:] > 0)[Uall].mean()),
                                              margin_row_but_limit0=float((np.isfinite(CH.MB[P.t0:]) & ~(CH.ML[P.t0:] > 0))[Uall].mean()),
                                              limit0_but_balance_gt0=float((~(CH.ML[P.t0:] > 0) & (CH.MB[P.t0:] > 0))[Uall].mean()),
                                              margin_balance_eq0=float((CH.MB[P.t0:] == 0)[Uall].mean()), short_balance_eq0=float((CH.SB[P.t0:] == 0)[Uall].mean()),
                                              sbl_row=float(np.isfinite(CH.SBL[P.t0:])[Uall].mean()), sbl_balance_eq0=float((CH.SBL[P.t0:] == 0)[Uall].mean()),
                                              shares=float(np.isfinite(CH.SHR[P.t0:])[Uall].mean()))
lr = CH.ML[P.t0:].astype(float) * 1000 / CH.SHR[P.t0:]
z = lr[Uall & (CH.ML[P.t0:] > 0)]
dc['limit_over_shares_quantiles'] = {q: float(np.nanquantile(z, q)) for q in (0.01, 0.25, 0.5, 0.75, 0.99)}
del lr, z
qs = (0.05, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99)
dist = {}
for lag in (1, 0):
    pool = {c: [] for c in L.FEATS}; nm = []
    for k in ks:
        F = L.features(P, U, CH, int(k), lag)
        for c in L.FEATS:
            pool[c].append(F[c][U[k]])
        nm.append(float(F['no_margin'][U[k]].mean()))
    dist[f'lag{lag}'] = dict(no_margin_rate_mean=float(np.mean(nm)),
                             **{c: dict(avail=float(np.isfinite(np.concatenate(v)).mean()), **{f'q{int(q*100)}': float(np.nanquantile(np.concatenate(v), q)) for q in qs}) for c, v in pool.items()})
dc['universe_feature_dist_signal_days'] = dist
res['data_check'] = dc
print(json.dumps(dc, ensure_ascii=False, indent=1), flush=True)

# ---- 交易特徵（全部交易都算特徵；報酬分組只用訓練期）
T = pd.concat([lg.reset_index(drop=True), L.trade_features(P, U, CH, lg, 1), L.trade_features(P, U, CH, lg, 0, '_L0')], axis=1)
T['ret20'] = P.RET20[T['sig_t'].to_numpy(), T['j'].to_numpy()]
per = lambda d: ('pre' if d < pd.Timestamp('2016-02-01') else 'train' if d < pd.Timestamp('2022-01-01') else 'valid')
T['per'] = [per(d) for d in T['sig_date']]
cov = {}
for nm_, zz in (('all', T), ('pre', T[T['per'] == 'pre']), ('train', T[T['per'] == 'train']), ('valid', T[T['per'] == 'valid'])):
    cov[nm_] = dict(n=int(len(zz)), no_margin=float(zz['no_margin'].mean()), avail={c: float(np.isfinite(zz[c]).mean()) for c in L.FEATS},
                    pct_median={c: float(zz[c + '_pct'].median()) for c in L.FEATS},
                    lag0_vs_lag1_pct_corr={c: float(zz[[c + '_pct', c + '_pct_L0']].corr(method='spearman').iloc[0, 1]) for c in L.FEATS})
res['bought_trades_feature_coverage'] = cov

Z = T[T['per'] == 'train'].copy()
Z['ex'] = Z['ret'].to_numpy() - (P.E50[Z['exit_t'].to_numpy()] / P.E50O[Z['ent_t'].to_numpy()] - 1)
hi, lo = Z['ret'].quantile(0.95), Z['ret'].quantile(0.05)
Z['pnl_rank'] = Z['pnl'].rank(ascending=False, method='first').astype(int)
tp = float(Z['pnl'].sum())


def gstat(z):
    if not len(z):
        return dict(n=0)
    return dict(n=int(len(z)), win=float((z['ret'] > 0).mean()), mean=float(z['ret'].mean()), median=float(z['ret'].median()), mean_ex=float(z['ex'].mean()),
                n_big_win=int((z['ret'] >= hi).sum()), n_big_lose=int((z['ret'] <= lo).sum()), n_top10_pnl=int((z['pnl_rank'] <= 10).sum()),
                n_bottom10_pnl=int((z['pnl_rank'] > len(Z) - 10).sum()), pnl_share=float(z['pnl'].sum() / tp))


BINS = (('p00_33', 0, 1 / 3), ('p33_67', 1 / 3, 2 / 3), ('p67_100', 2 / 3, 1.0001), ('top20', 0.8, 1.0001), ('top10', 0.9, 1.0001), ('bot20', 0, 0.2))
G = {}
for sfx in ('', '_L0'):
    g = dict(ALL=gstat(Z), no_margin=gstat(Z[Z['no_margin' + sfx]]))
    for c in L.FEATS:
        p = Z[c + '_pct' + sfx]
        g[c] = {b: gstat(Z[(p >= a) & (p < e)]) for b, a, e in BINS}
        g[c]['NaN'] = gstat(Z[~np.isfinite(p)])
    # 絕對門檻（描述）
    v = Z['m_g20' + sfx]; g['m_g20_abs'] = {'<=0': gstat(Z[v <= 0]), '0~30%': gstat(Z[(v > 0) & (v <= 0.3)]), '30~100%': gstat(Z[(v > 0.3) & (v <= 1)]), '>100%': gstat(Z[v > 1])}
    v = Z['m_g5' + sfx]; g['m_g5_abs'] = {'<=0': gstat(Z[v <= 0]), '0~20%': gstat(Z[(v > 0) & (v <= 0.2)]), '>20%': gstat(Z[v > 0.2])}
    v = Z['sm_ratio' + sfx]; g['sm_ratio_abs'] = {'<1%': gstat(Z[v < 0.01]), '1~5%': gstat(Z[(v >= 0.01) & (v < 0.05)]), '5~20%': gstat(Z[(v >= 0.05) & (v < 0.2)]), '>=20%': gstat(Z[v >= 0.2])}
    v = Z['sbl_sh' + sfx]; g['sbl_sh_abs'] = {'=0': gstat(Z[v == 0]), '0~0.5%': gstat(Z[(v > 0) & (v < 0.005)]), '0.5~2%': gstat(Z[(v >= 0.005) & (v < 0.02)]), '>=2%': gstat(Z[v >= 0.02])}
    G['lag1' if sfx == '' else 'lag0'] = g
# 與 S82 追高單（進場前 20 日已漲 > 50%）的交叉
ch = Z['ret20'] > 0.5
G['chase50'] = dict(chase=gstat(Z[ch]), chase_and_m_d20s_top20=gstat(Z[ch & (Z['m_d20s_pct'] >= 0.8)]), chase_and_not=gstat(Z[ch & ~(Z['m_d20s_pct'] >= 0.8)]),
                    notchase_and_m_d20s_top20=gstat(Z[~ch & (Z['m_d20s_pct'] >= 0.8)]))
G['thresholds'] = dict(big_win_ret_ge=float(hi), big_lose_ret_le=float(lo))
G['spearman_ret_vs_pct_lag1'] = {c: float(Z[[c + '_pct', 'ret']].corr(method='spearman').iloc[0, 1]) for c in L.FEATS}
G['top10_pnl_trades'] = [dict(code=y.code, sig=str(y.sig_date.date()), ret=float(y.ret), pnl=float(y.pnl), **{c: (None if not np.isfinite(getattr(y, c + '_pct')) else round(float(getattr(y, c + '_pct')), 3)) for c in L.FEATS})
                         for y in Z.sort_values('pnl', ascending=False).head(10).itertuples()]
G['bottom10_pnl_trades'] = [dict(code=y.code, sig=str(y.sig_date.date()), ret=float(y.ret), pnl=float(y.pnl), **{c: (None if not np.isfinite(getattr(y, c + '_pct')) else round(float(getattr(y, c + '_pct')), 3)) for c in L.FEATS})
                            for y in Z.sort_values('pnl').head(10).itertuples()]
res['train_returns_by_group'] = G
res['meta'] = dict(runtime_s=round(time.time() - t0, 1), maxrss_mb=rss())
(H / 'data_check.json').write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str))
print(json.dumps(cov, ensure_ascii=False))
fmt = lambda v: {a: (round(b, 3) if isinstance(b, float) else b) for a, b in v.items()}
for lagk in ('lag1', 'lag0'):
    print('=====', lagk)
    for k, v in G[lagk].items():
        if 'n' in v:
            print(k, fmt(v))
        else:
            for b, w in v.items():
                print(' ', k, b, fmt(w))
print('chase', {k: fmt(v) for k, v in G['chase50'].items()})
print(G['spearman_ret_vs_pct_lag1'])
for y in G['top10_pnl_trades'] + G['bottom10_pnl_trades']:
    print(y)
print(res['meta'])
