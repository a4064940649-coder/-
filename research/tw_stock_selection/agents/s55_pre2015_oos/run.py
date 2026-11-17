#!/usr/bin/env python3
"""S55 主跑（只跑一次）：凍結的現行規則在 2008-01～2015-06 的真正樣本外。
先驗 SPEC.json／PREREG.json／code.sha256／面板 sha256；results.json 已存在就拒跑。
輸出：results.json、oos_nav.csv.gz（日淨值＋0050 含息）、oos_monthly.csv、oos_trades.csv.gz。用法：python3 -I -B run.py"""
import sys, json, math, time, resource, hashlib
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert not (HERE / 'results.json').exists(), 'results.json 已存在：S55 只跑一次'
for f in ('SPEC', 'PREREG'):
    assert sha(HERE / f'{f}.json') == (HERE / f'{f}.sha256').read_text().split()[0], f'{f}.json 與登記的 sha256 不符'
for line in (HERE / 'code.sha256').read_text().strip().split('\n'):
    h, f = line.split()
    assert sha(HERE / f) == h, f'{f} 與登記的 sha256 不符'
SPEC = json.loads((HERE / 'SPEC.json').read_text()); PRE = json.loads((HERE / 'PREREG.json').read_text())
import s55lib as X                                          # noqa: E402
cache = X.setup('oos')
for f, h in SPEC['data_sha256'].items():
    assert sha(cache / f) == h, f'{f} 與 SPEC 登記的 sha256 不符'
t0 = time.time()
rss = lambda: round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)

P, U, RT, sg, fix = X.build('oos')
nav, lg, st = X.L.sim(P, sg, U, cap=1e6)
e50 = pd.Series(P.E50, index=P.IDX)
D0 = P.IDX[P.t0]
nav = nav.loc[D0:]
pd.DataFrame(dict(nav=nav, e50=e50.reindex(nav.index))).to_csv(HERE / 'oos_nav.csv.gz')
M = X.monthly(nav, e50)
M['excess'] = M['strategy'] - M['b0050']
M.to_csv(HERE / 'oos_monthly.csv')
ISd = pd.read_csv(X.SCR / 'is_nav.csv.gz', index_col=0, parse_dates=True)
ISd = ISd.loc[pd.Timestamp('2015-07-01'):]
MI = X.monthly(ISd['nav'], ISd['e50']); MI['excess'] = MI['strategy'] - MI['b0050']
assert len(MI) == 136


def nw_t(x, L=6):
    x = np.asarray(x, float); T = len(x); mu = x.mean(); e = x - mu; s = (e @ e) / T
    for i in range(1, L + 1):
        s += 2 * (1 - i / (L + 1)) * (e[i:] @ e[:-i]) / T
    return float(mu / math.sqrt(s / T))


def desc(x):
    x = np.asarray(x, float); T = len(x); mu, sd = x.mean(), x.std(ddof=1)
    return dict(T=T, mean_m=float(mu), sd_m=float(sd), ann_excess_pp=float(mu * 12 * 100), ann_te_pp=float(sd * math.sqrt(12) * 100),
                IR_ann=float(mu / sd * math.sqrt(12)), t_iid=float(mu / sd * math.sqrt(T)), t_nw6=nw_t(x, 6),
                skew=float(stats.skew(x)), kurt=float(stats.kurtosis(x, fisher=False)), rho1=float(np.corrcoef(x[1:], x[:-1])[0, 1]),
                hit_rate=float((x > 0).mean()))


def window(Mw, dnav=None, db=None):
    yrs = len(Mw) / 12
    cg = lambda r: float(np.prod(1 + r) ** (1 / yrs) - 1)
    o = dict(months=int(len(Mw)), period=f'{Mw.index[0]}～{Mw.index[-1]}', cagr_strategy=cg(Mw['strategy']), cagr_0050=cg(Mw['b0050']),
             excess=desc(Mw['excess']), beta=float(np.polyfit(Mw['b0050'], Mw['strategy'], 1)[0]), corr=float(np.corrcoef(Mw['b0050'], Mw['strategy'])[0, 1]))
    o['cagr_diff_pp'] = (o['cagr_strategy'] - o['cagr_0050']) * 100
    if dnav is not None:
        o['maxdd_strategy'] = float((dnav / dnav.cummax() - 1).min()); o['maxdd_0050'] = float((db / db.cummax() - 1).min())
    return o


W = PRE['windows']
res = dict(spec_sha256=sha(HERE / 'SPEC.json'), prereg_sha256=sha(HERE / 'PREREG.json'), windows={})
for key in ('primary', 'secondary'):
    a, b = W[key]['from'], W[key]['to']
    Mw = M.loc[a:b]
    end = Mw.index[-1].to_timestamp(how='end')
    dn = nav.loc[:end]; dbb = e50.reindex(dn.index)
    res['windows'][key] = window(Mw, dn, dbb)
res['windows']['stuck_2013_02_to_2015_06'] = window(M.loc['2013-02':'2015-06'])
res['windows']['in_sample_2015_07_to_2026_10'] = window(MI)
for key, oos in (('merged_primary_plus_IS', 'primary'), ('merged_secondary_plus_IS', 'secondary')):
    Mw = pd.concat([M.loc[W[oos]['from']:W[oos]['to']], MI])
    res['windows'][key] = window(Mw)
res['by_year'] = {str(y): dict(months=int((M.index.year == y).sum()), strategy=float(np.prod(1 + M.loc[M.index.year == y, 'strategy']) - 1),
                               b0050=float(np.prod(1 + M.loc[M.index.year == y, 'b0050']) - 1)) for y in sorted(set(M.index.year))}
for y in res['by_year']:
    res['by_year'][y]['diff_pp'] = (res['by_year'][y]['strategy'] - res['by_year'][y]['b0050']) * 100

# 判定（PREREG）
pw = res['windows']['primary']['excess']
thr = PRE['decision']['t_threshold']
verdict = '支持' if (pw['ann_excess_pp'] > 0 and pw['t_nw6'] > thr) else ('方向一致但不顯著' if pw['ann_excess_pp'] > 0 else '不支持')
sw = res['windows']['secondary']['excess']
res['decision'] = dict(rule=PRE['decision']['rule'], primary_ann_excess_pp=pw['ann_excess_pp'], primary_t_nw6=pw['t_nw6'], verdict=verdict,
                       secondary_ann_excess_pp=sw['ann_excess_pp'], secondary_t_nw6=sw['t_nw6'],
                       secondary_same_verdict=bool(('支持' if (sw['ann_excess_pp'] > 0 and sw['t_nw6'] > thr) else ('方向一致但不顯著' if sw['ann_excess_pp'] > 0 else '不支持')) == verdict))

# 交易與部位
x = lg[lg['why'] != 'open_end']
res['trades'] = dict(n_signal_months=len(sg['info']), first_signal=sg['info'][0]['sig_date'], last_signal=sg['info'][-1]['sig_date'],
                     avg_n_tgt=float(np.mean([i['n_tgt'] for i in sg['info']])), n_trades=int(len(lg)), n_closed=int(len(x)),
                     why_counts={str(k): int(v) for k, v in lg['why'].value_counts().items()},
                     win_rate_closed=float((x['pnl'] > 0).mean()) if len(x) else None, avg_npos=float(st['avg_npos']), avg_idle_frac=float(st['avg_idle_frac']),
                     top5_share_of_gain=float(X.s17lib.top5_share(lg, nav, 1e6)) if nav.iloc[-1] > 1e6 else None,
                     open_end_positions=int((lg['why'] == 'open_end').sum()))
keep = [c for c in ('j', 'ent_t', 'exit_t', 'ent_date', 'why', 'shares', 'cost', 'proceeds', 'pnl') if c in lg.columns]
T_ = lg[keep].copy(); T_['code'] = [str(P.COLS[j]) for j in T_['j']]
T_['exit_date'] = [str(P.IDX[int(t)].date()) if np.isfinite(t) and int(t) < P.T else None for t in T_['exit_t']]
T_.to_csv(HERE / 'oos_trades.csv.gz', index=False)

# 事前登記的描述性診斷
tick = X.cslib.tick
REF = P.REF; O = P.O
lim7 = P.IDX < pd.Timestamp('2015-06-01')
n_b7 = n_s7 = 0
for r in lg.itertuples():
    t, j = int(r.ent_t), int(r.j)
    if lim7[t] and np.isfinite(O[t, j]):
        u = np.floor(REF[t, j] * 1.07 / tick(REF[t, j] * 1.07) + 1e-9) * tick(REF[t, j] * 1.07)
        n_b7 += int(O[t, j] >= u - 1e-9)
    if r.why != 'open_end' and np.isfinite(r.exit_t) and int(r.exit_t) < P.T:
        te = int(r.exit_t)
        if lim7[te] and np.isfinite(O[te, j]):
            d = np.ceil(REF[te, j] * 0.93 / tick(REF[te, j] * 0.93) - 1e-9) * tick(REF[te, j] * 0.93)
            n_s7 += int(O[te, j] <= d + 1e-9)
xr = pd.read_csv(X.AG.parent / 'raw' / 'pre2015' / 'asoul_twse_2007_2015H1.csv.gz', usecols=['date', 'code', 'xflag'], dtype={'code': str})
xr = xr[(xr['xflag'] == 1) & (pd.to_datetime(xr['date']) < pd.Timestamp('2014-01-01'))]
di = {d: i for i, d in enumerate(P.IDX.strftime('%Y-%m-%d'))}
xs = {}
for d, c in zip(xr['date'], xr['code']):
    if d in di and c in P.ci:
        xs.setdefault(P.ci[c], []).append(di[d])
n_x_held = 0
for r in lg.itertuples():
    te = int(r.exit_t) if np.isfinite(r.exit_t) else P.T
    n_x_held += sum(1 for t in xs.get(int(r.j), []) if int(r.ent_t) < t <= min(te, P.T - 1))
lastp = np.array([np.flatnonzero(P.has[:, j]).max() if P.has[:, j].any() else -1 for j in range(P.N)])
surv = []
for i in sg['info']:
    k = i['k']; m = pd.Period(i['month'], 'M'); s = RT['S'].loc[m].to_numpy()
    miss = U[k] & ~np.isfinite(s) & P.F_MA60[k]
    dl15 = miss & (lastp <= min(k + 315, P.T - 1)) & (lastp < P.T - 5)
    surv.append(dict(month=str(m), n_univ_ma60_surge_nan=int(miss.sum()), of_which_last_trade_within_15m=int(dl15.sum())))
res['diagnostics'] = dict(
    buys_open_at_or_above_7pct_limit_pre_2015_06=n_b7, sells_open_at_or_below_7pct_limit_pre_2015_06=n_s7,
    held_exright_days_without_twt49u_pre2014=n_x_held,
    survivorship=dict(total_univ_ma60_surge_nan=int(sum(s['n_univ_ma60_surge_nan'] for s in surv)),
                      total_of_which_delisted_within_15m=int(sum(s['of_which_last_trade_within_15m'] for s in surv)), per_month=surv),
    exright_info=P.exright_info, e50_open_missing=int(P.e50_open_missing))
res['runtime_s'] = round(time.time() - t0, 1); res['maxrss_mb'] = rss()
(HERE / 'results.json').write_text(json.dumps(res, ensure_ascii=False, indent=1, default=lambda o: o.item() if hasattr(o, 'item') else str(o)))
print(json.dumps(dict(decision=res['decision'], windows={k: dict(v['excess'], cagr_s=v['cagr_strategy'], cagr_b=v['cagr_0050']) for k, v in res['windows'].items()},
                      trades=res['trades'], diag={k: v for k, v in res['diagnostics'].items() if k != 'survivorship'},
                      surv_total=res['diagnostics']['survivorship']['total_of_which_delisted_within_15m']), ensure_ascii=False, indent=1))
