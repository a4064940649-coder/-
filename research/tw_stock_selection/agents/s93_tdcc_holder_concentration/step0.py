#!/usr/bin/env python3
"""S93 第 0 步（登記前）：①重現正式標準路徑；②集保資料驗收（data_check.json）；③只看訓練期（訊號日 2019-08～2022-12）現行已平倉交易的集保分組描述。
任何加了集保條件的模擬都不在這裡跑。用法：python3 -I -B step0.py"""
import sys, json, time, resource
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import s93lib as L
from s93lib import stlib
t0 = time.time(); rss = lambda: round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
P, U, RT = L.load()
lastb = lambda d: P.IDX[P.IDX < pd.Timestamp(d)][-1]
D0, DE, D19 = P.IDX[P.t0], P.IDX[-1], lastb('2019-01-01')
S0 = L.signal(P, U, RT, 0)
nav, lg, st = L.sim(P, S0, U)
site = pd.read_csv('/home/user/-/xq_scripts/site/navs_official.csv.gz', index_col=0, parse_dates=True)['E0|W10_70']
A0 = dict(since2019=stlib.cagr(nav, D19, DE), full=stlib.cagr(nav, D0, DE), maxdd=stlib.maxdd(nav, D0, DE), n_closed=int((lg['why'] != 'open_end').sum()), n_all=int(len(lg)),
          site_same_index=bool(site.index.equals(nav.index)), max_abs_diff_vs_site_nav=float((nav.round(2) - site).abs().max()), panel=[int(P.T), int(P.N), str(DE.date())])
A0['pass'] = bool(abs(A0['since2019'] - L.TARGET['since2019']) < 1e-12 and abs(A0['full'] - L.TARGET['full']) < 1e-12 and A0['n_closed'] == 381 and A0['n_all'] == 389
                  and A0['site_same_index'] and A0['max_abs_diff_vs_site_nav'] == 0)
print('A0', A0, f'{time.time()-t0:.0f}s', rss(), 'MB', flush=True)
assert A0['pass']
TD = L.TDCC(P)
gap = pd.Series(TD.dates).diff().dt.days
i0 = int(np.flatnonzero(TD.dates >= pd.Timestamp('2019-06-28'))[0])
info = [x for x in S0['info'] if pd.Period(x['month'], 'M') >= L.RULE_FIRST_MONTH]
dc = dict(rows_2019on=TD.rows, dup=TD.dup, weeks=len(TD.dates), first=str(TD.dates[0].date()), last=str(TD.dates[-1].date()), continuous_from='2019-06-28',
          gaps_ge10d_since_2019_06_28=[dict(date=str(TD.dates[i].date()), prev=str(TD.dates[i - 1].date()), gap_days=int(gap[i])) for i in range(i0 + 1, len(TD.dates)) if gap[i] >= 10],
          weekday_counts={int(k): int(v) for k, v in pd.Series(TD.dates[i0:].weekday).value_counts().sort_index().items()},
          missing_week_rule='「N 週前」＝資料日 ≤ 最新一筆資料日 − 7N 天的最新一筆；缺週自動用再前一週。連續下降用相鄰兩筆可得資料比。', signal_days={})
for lag in ('cons', 'loose'):
    ages = [TD.feat(x['k'], lag)['age'] for x in info]; sp4 = [TD.feat(x['k'], lag)['span4'] for x in info]; sp8 = [TD.feat(x['k'], lag)['span8'] for x in info]
    tg = [(x['k'], j) for x in info for j in np.flatnonzero(S0['sig'][x['k']])]
    dc['signal_days'][lag] = dict(n=len(info), first=info[0]['sig_date'], last=info[-1]['sig_date'], age_days_min=int(min(ages)), age_days_median=float(np.median(ages)), age_days_max=int(max(ages)),
                                  span4_days=sorted(set(v for v in sp4 if v is not None)), span8_days=sorted(set(v for v in sp8 if v is not None)), n_sigdays_no_8w_history=int(sum(v is None for v in sp8)), targets=len(tg),
                                  targets_with_dB4=int(sum(np.isfinite(TD.feat(k, lag)['dB4'][j]) for k, j in tg)), targets_with_dB8=int(sum(np.isfinite(TD.feat(k, lag)['dB8'][j]) for k, j in tg)),
                                  targets_with_dH4=int(sum(np.isfinite(TD.feat(k, lag)['dH4'][j]) for k, j in tg)))
T19 = pd.Timestamp('2019-08-01')
z = lg[lg['sig_date'] >= T19]; zf = L.trade_features(TD, z)
dc['trades_since_2019_08'] = dict(n_all=int(len(z)), n_closed=int((z['why'] != 'open_end').sum()), with_dB4=int(np.isfinite(zf['dB4']).sum()), with_dB8=int(np.isfinite(zf['dB8']).sum()), with_dH4=int(np.isfinite(zf['dH4']).sum()),
                                  missing_dB8=[dict(code=str(r.code), sig=str(r.sig_date.date())) for r in zf[~np.isfinite(zf['dB8'])].itertuples()], missing=[dict(code=str(r.code), sig=str(r.sig_date.date())) for r in zf[~np.isfinite(zf['dB4'])].itertuples()])
print(json.dumps(dc, ensure_ascii=False, default=str))
(HERE / 'data_check.json').write_text(json.dumps(dc, ensure_ascii=False, indent=1, default=str))
# ---- 訓練期描述（訊號日 2019-08-01～2022-12-31、已平倉）
x = zf[(zf['why'] != 'open_end') & (zf['sig_date'] <= pd.Timestamp('2022-12-31'))].copy()
n5 = int(round(len(x) * 0.05))
x['bigwin'] = x['ret'].rank(ascending=False, method='first') <= n5; x['biglose'] = x['ret'].rank(ascending=True, method='first') <= n5


def grp(lab):
    o = {}
    for g, y in x.groupby(lab, observed=True):
        o[str(g)] = dict(n=int(len(y)), win=float((y['ret'] > 0).mean()), mean=float(y['ret'].mean()), median=float(y['ret'].median()), pnl_sum=float(y['pnl'].sum()), bigwin=int(y['bigwin'].sum()), biglose=int(y['biglose'].sum()))
    return o


D = {}; Q = {}
for c in ('dB4', 'dB8', 'dH4', 'dH8', 'dR4', 'dR8'):
    q = x[c].quantile([1 / 3, 2 / 3]).to_numpy(); Q[c] = dict(q33=float(q[0]), q67=float(q[1]), p10=float(x[c].quantile(0.1)), p90=float(x[c].quantile(0.9)), min=float(x[c].min()), max=float(x[c].max()))
    x['g'] = pd.cut(x[c], [-np.inf, q[0], q[1], np.inf], labels=['低', '中', '高']); D[c] = grp('g')
x['g'] = np.where((x['dB4'] > 0) & (x['dH4'] < 0), '集中(大戶增、人數減)', np.where((x['dB4'] < 0) & (x['dH4'] > 0), '發散(大戶減、人數增)', '其他')); D['conc4'] = grp('g')
x['g'] = np.where((x['dB8'] > 0) & (x['dH8'] < 0), '集中(大戶增、人數減)', np.where((x['dB8'] < 0) & (x['dH8'] > 0), '發散(大戶減、人數增)', '其他')); D['conc8'] = grp('g')
x['g'] = pd.cut(x['dB4'], [-np.inf, -2, -1, 0, 1, np.inf], labels=['≤−2', '−2~−1', '−1~0', '0~+1', '>+1']); D['dB4_fixed'] = grp('g')
x['g'] = pd.cut(x['dH4'], [-np.inf, -0.05, 0, 0.1, 0.3, np.inf], labels=['≤−5%', '−5~0%', '0~10%', '10~30%', '>30%']); D['dH4_fixed'] = grp('g')
x['g'] = pd.cut(x['ndec'], [-1, 0, 1, 2, 100], labels=['0', '1', '2', '≥3']); D['ndec'] = grp('g')
res = dict(A0=A0, train_desc=dict(window=['2019-08-01', '2022-12-31'], n=int(len(x)), n5=n5, tercile_cuts=Q, groups=D,
                                  rank_corr_ret={c: float(x[[c, 'ret']].corr(method='spearman').iloc[0, 1]) for c in ('dB4', 'dB8', 'dH4', 'dH8', 'dR4', 'dR8', 'ndec')}),
           meta=dict(runtime_s=round(time.time() - t0, 1), maxrss_mb=rss()))
(HERE / 'step0.json').write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str))
print(len(x), n5, {k: {a: round(b, 4) for a, b in v.items()} for k, v in Q.items()}); print(res['train_desc']['rank_corr_ret'])
for k, v in D.items():
    print(k, {g: (o['n'], round(o['win'], 2), round(o['mean'], 3), round(o['median'], 3), o['bigwin'], o['biglose']) for g, o in v.items()})
print(res['meta'])
