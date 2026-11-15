#!/usr/bin/env python3
"""S20 事後描述（驗證期判定之後才跑；未登記，只描述，不影響判定、不換門檻）：
(1) F1 名單（位移 0）逐筆 E0 持有期超額（s13lib.outcomes），依凍結規則留下／篩掉、依 μ÷σ 三分組，訓練期與驗證期分開；
(2) 大贏家（2408、3653、1325、2603）的 μ÷σ、NPOS12、是否被凍結規則保留；
(3) 「隨機篩掉同樣比例」對照（主跑 100 萬、位移 0、全期連續，各 20 次）：區分「少買幾檔、多放 0050」與「選股」；
(4) 逐年差（讀 results.json）。輸出 results_posthoc.json。python3 -I -B posthoc.py"""
import sys, json, time, resource, gc, random
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
import s20lib
from s20lib import s17lib, s10lib, s4lib, stlib, COSTS

H = Path(__file__).resolve().parent
FZ = json.loads((H / 'FROZEN.json').read_text())
R0 = json.loads((H / 'results.json').read_text())
RULE = {k: FZ['frozen_rule'][k] for k in ('feat', 'op', 'thr')}
PLR = {k: FZ['placebo'][k] for k in ('feat', 'op', 'thr')}
t_all = time.time()
rss = lambda: round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
P = s20lib.load()
INNOV, _ = s4lib.innov_panel(P)
U = stlib.universe(P, 'twse') & ~INNOV
del INNOV
G = s20lib.growth_features(P)
RT = dict(S=G['S'], av=G['av'])
s17lib.e50_fix_suspension(P)
for a in ('H', 'L', 'VOL', 'CHG', 'HF', 'VOLF', 'adjH', 'HI240', 'MA60', 'nbar', 'F_HI', 'IDXC', 'AO', 'listed', 'NEARHI', 'V60', 'liq'):
    if hasattr(P, a):
        delattr(P, a)
gc.collect()
lastb = lambda d: P.IDX[P.IDX < pd.Timestamp(d)][-1]
TA, TB, D19, DE = lastb('2016-02-01'), lastb('2022-01-01'), lastb('2019-01-01'), P.IDX[-1]
iTB = int(np.searchsorted(P.IDX.values, TB.to_datetime64()))
S0 = s10lib.make_signal_s10(P, U, RT, shift=0)
f1 = s20lib.f1_table(P, S0, G)
f1['mper'] = pd.PeriodIndex(f1['month'], freq='M')
f1['period'] = np.where(f1['mper'] <= s20lib.TRAIN_LAST_MONTH, 'train', 'valid')
sig_ks = [x['k'] for x in S0['info']]
X = S0['XV']['E0']
out = []
for per, end_t in (('train', iTB), ('valid', P.T - 1)):
    d = f1[f1['period'] == per]
    out.append(s20lib.outcomes(P, X, sig_ks, d, end_t))
f1 = f1.join(pd.concat(out))
f1['keep_FZ'] = [s20lib.rule_keep_value(G, RULE['feat'], RULE['op'], RULE['thr'], m, c) for m, c in zip(f1['mper'], f1['code'])]
f1['keep_LOW'] = [s20lib.rule_keep_value(G, PLR['feat'], PLR['op'], PLR['thr'], m, c) for m, c in zip(f1['mper'], f1['code'])]
qtr = np.quantile(f1.loc[f1['period'] == 'train', 'MS'], [1 / 3, 2 / 3])
f1['ms_grp'] = np.where(f1['MS'] < qtr[0], '低', np.where(f1['MS'] < qtr[1], '中', '高'))
res = {'note': '事後描述，未登記；不影響判定、不換門檻。', 'ms_tercile_cut_train': [float(x) for x in qtr]}


def grp(d):
    e = d['excess'].dropna()
    if len(e) == 0:
        return dict(n=0)
    top5 = e.sort_values(ascending=False)
    return dict(n=int(len(e)), mean=float(e.mean()), median=float(e.median()), win=float((e > 0).mean()), big=float((e >= 0.1953).mean()),
                mean_ex_top5=float(top5.iloc[5:].mean()) if len(e) > 5 else float('nan'), sum=float(e.sum()),
                top5_share_of_sum=float(top5.iloc[:5].sum() / e.sum()) if e.sum() > 0 else float('nan'), hold_days=float(d['hold_days'].mean()))


ev = {}
for per in ('train', 'valid'):
    d = f1[f1['period'] == per]
    o = dict(all=grp(d), kept=grp(d[d['keep_FZ']]), removed=grp(d[~d['keep_FZ']]))
    o['MS_tercile'] = {g: grp(d[d['ms_grp'] == g]) for g in ('高', '中', '低')}
    o['MU'] = {'μ>0': grp(d[d['MU'] > 0]), 'μ≤0': grp(d[d['MU'] <= 0])}
    o['month_cluster_kept_minus_all'] = s20lib.month_cluster_diff(d, d['keep_FZ'])
    o['month_cluster_highMS_minus_all'] = s20lib.month_cluster_diff(d, d['ms_grp'] == '高')
    o['removed_top'] = [dict(month=r.month, code=r.code, MS=float(r.MS), excess=float(r.excess)) for r in d[~d['keep_FZ']].sort_values('excess', ascending=False).head(8).itertuples()]
    o['removed_worst'] = [dict(month=r.month, code=r.code, MS=float(r.MS), excess=float(r.excess)) for r in d[~d['keep_FZ']].sort_values('excess').head(5).itertuples()]
    ev[per] = o
res['events_f1'] = ev
# 逐年（事件層，依訊號年）：留下 vs 篩掉 平均超額
f1['year'] = f1['sig_date'].dt.year
res['events_by_year'] = {int(y): dict(kept=grp(d[d['keep_FZ']]).get('mean'), removed=grp(d[~d['keep_FZ']]).get('mean'),
                                      n_kept=int(d['keep_FZ'].sum()), n_removed=int((~d['keep_FZ']).sum()))
                         for y, d in f1.groupby('year')}
# (2) 大贏家
bw = f1[f1['code'].isin(s20lib.BIG)].sort_values(['code', 'month'])
res['big_winners'] = [dict(code=r.code, month=r.month, sig=str(r.sig_date.date()), MS=round(float(r.MS), 3), mu_k=float(r.mu), sigma_k=float(r.sigma), NPOS12=int(r.NPOS12),
                           keep_FZ=bool(r.keep_FZ), keep_LOW=bool(r.keep_LOW), excess=float(r.excess) if np.isfinite(r.excess) else None) for r in bw.itertuples()]
# 實際交易（results 的 trades.csv.gz）裡大贏家是否被凍結規則買到
tr = pd.read_csv(H / 'trades.csv.gz', dtype={'code': str})
res['big_winner_trades'] = {nm: [dict(code=r.code, ent=r.ent_date, exit=r.exit_date, ret=float(r.ret)) for r in tr[(tr['run'] == nm) & tr['code'].isin(s20lib.BIG)].itertuples()]
                            for nm in ('CUR|cont', 'FZ|cont', 'LOW|cont')}
# (3) 隨機篩掉同樣比例（全期連續主跑，20 次）
CELL = dict(sat_w=0.7, sizing='W10', lock=False)


def run(S):
    S2, ekw = s10lib.variant_S(S, 'E0')
    nav = s10lib.simulate_s10(P, S2, U, cost=COSTS['strict'], capital=1e6, **CELL, **ekw)[0]
    return dict(train=stlib.cagr(nav, TA, TB), valid=stlib.cagr(nav, TB, DE), since2019=stlib.cagr(nav, D19, DE))


base = run(S0)
fz = run(s20lib.apply_rule(P, S0, G, RULE)[0]); low = run(s20lib.apply_rule(P, S0, G, PLR)[0])
rnd = {}
for frac_name, keep_p in (('留 2/3（對照 FZ）', float(f1['keep_FZ'].mean())), ('留 1/2（對照 LOW）', float(f1['keep_LOW'].mean()))):
    vals = []
    for seed in range(20):
        rng = random.Random(1000 + seed)
        sig = S0['sig'].copy(); rank = S0['rank'].copy()
        for x in S0['info']:
            for c in x['targets']:
                if rng.random() > keep_p:
                    j = P.ci[c]; sig[x['k'], j] = False; rank[x['k'], j] = np.nan
        S2 = dict(S0); S2['sig'] = sig; S2['rank'] = rank
        vals.append(run(S2))
    rnd[frac_name] = dict(keep_p=keep_p, **{k: dict(median=float(np.median([v[k] for v in vals])), p10=float(np.percentile([v[k] for v in vals], 10)),
                                                    p90=float(np.percentile([v[k] for v in vals], 90))) for k in ('train', 'valid', 'since2019')})
res['random_filter'] = dict(CUR=base, FZ=fz, LOW=low, random=rnd, note='主跑 100 萬、位移 0、全期連續；train＝2016-01-29→2021-12-30，valid＝2021-12-30→2026-10-22（連續，非重新起跑）。')
# (4) 逐年（主跑）
yr = {}
for mode in ('cont', 'fresh'):
    for c in ('CUR', 'FZ', 'LOW'):
        yr[f'{c}|{mode}'] = {y: v['strat'] for y, v in R0['main'][f'{c}|{mode}']['yearly'].items()}
    yr[f'0050|{mode}'] = {y: v['e0050'] for y, v in R0['main'][f'CUR|{mode}']['yearly'].items()}
res['yearly'] = yr
res['meta'] = dict(runtime_s=round(time.time() - t_all, 1), maxrss_mb=rss())
(H / 'results_posthoc.json').write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str))
print(json.dumps({k: res[k] for k in ('events_f1', 'big_winners', 'random_filter', 'meta')}, ensure_ascii=False, indent=0, default=str)[:9000])
