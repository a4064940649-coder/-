#!/usr/bin/env python3
"""S44 事後描述（驗證期判定之後才跑；兩項已在 PREREG 登記為「只描述」，不影響判定、不換門檻）：
(1) 隨機剔除同檔數對照（PREREG random_control）：凍結規則每個訊號月剔除的檔數，隨機從當月 F1 剔除同樣檔數，200 次；
    主跑 100 萬、位移 0；訓練窗（全期連續切出）與驗證窗（重新起跑）年化差（− CUR），報凍結規則的百分位；
(2) 被剔除股票逐筆超額（PREREG removed_trade_level）：F1（位移 0）逐筆 E0 持有期報酬 − 0050（扣成本前），5 條規則各自留下／剔除，訓練／驗證分開，同月群聚差；
(3) 未登記的補充：逐年差（讀 results.json）、1、2 月與其他月份的剔除組超額。輸出 results_posthoc.json。python3 -I -B posthoc.py"""
import sys, json, time, resource, gc
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
import s44lib
from s44lib import s17lib, s10lib, stlib, COSTS

H = Path(__file__).resolve().parent
FZ = json.loads((H / 'FROZEN.json').read_text())
R0 = json.loads((H / 'results.json').read_text())
RULE = FZ['frozen_rule']
t_all = time.time()
rss = lambda: round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
P = s44lib.load()
U, RT = s44lib.base(P)
s17lib.e50_fix_suspension(P)
lastb = lambda d: P.IDX[P.IDX < pd.Timestamp(d)][-1]
TA, TB, D19, DE = lastb('2016-02-01'), lastb('2022-01-01'), lastb('2019-01-01'), P.IDX[-1]
iTB = int(np.searchsorted(P.IDX.values, TB.to_datetime64()))
S0 = s10lib.make_signal_s10(P, U, RT, shift=0)
F1, _ = s44lib.f1_rows(P, U, RT, S0)
F1['period'] = np.where(F1['mper'] <= s44lib.TRAIN_LAST_MONTH, 'train', 'valid')
CELL = dict(sat_w=0.7, sizing='W10', lock=False)
FCAP = 1e6 / R0['meta']['fresh_capital_scale']


def run(S):
    S2, ekw = s10lib.variant_S(S, 'E0')
    nav = s10lib.simulate_s10(P, S2, U, cost=COSTS['strict'], capital=1e6, **CELL, **ekw)[0]
    Sf = s44lib.fresh_from(S, s44lib.VALID_FIRST_MONTH)
    S3, ekw = s10lib.variant_S(Sf, 'E0')
    navf = s10lib.simulate_s10(P, S3, U, cost=COSTS['strict'], capital=FCAP, **CELL, **ekw)[0]
    return dict(train=stlib.cagr(nav, TA, TB), valid_fresh=stlib.cagr(navf, TB, DE), since2019=stlib.cagr(nav, D19, DE))


# ---------------------------------------------------------------- (1) 隨機剔除同檔數
res = {'note': '(1)(2) 為 PREREG 登記的描述項；(3) 未登記。皆不影響判定、不換門檻。'}
base = run(S0)
Sfz, rm = s44lib.apply_rule(P, S0, F1, RULE)
fz = run(Sfz)
nbm = pd.Series([m for m, _ in rm]).value_counts().to_dict()
vals = []
for i in range(200):
    vals.append(run(s44lib.random_drop(P, S0, F1, nbm, 20261117 + i)))
    if i % 50 == 49:
        print('隨機', i + 1, f'{time.time()-t_all:.0f}s', rss(), 'MB', flush=True)
rc = {'CUR': base, 'FZ': fz, 'n_removed_by_rule': len(rm), 'n_months': len(nbm)}
for k in ('train', 'valid_fresh', 'since2019'):
    d = np.array([v[k] - base[k] for v in vals]) * 100
    f = (fz[k] - base[k]) * 100
    rc[k] = dict(FZ_minus_CUR_pp=f, random_median_pp=float(np.median(d)), random_p5_pp=float(np.percentile(d, 5)), random_p95_pp=float(np.percentile(d, 95)),
                 FZ_percentile=float((d < f).mean() * 100), random_frac_ge_FZ=float((d >= f).mean()))
res['random_control'] = rc
gc.collect()

# ---------------------------------------------------------------- (2) 被剔除股票逐筆超額（5 條規則）
sig_ks = [x['k'] for x in S0['info']]
X = S0['XV']['E0']
out = []
for per, end_t in (('train', iTB), ('valid', P.T - 1)):
    out.append(s44lib.outcomes(P, X, sig_ks, F1[F1['period'] == per], end_t))
F1 = F1.join(pd.concat(out))


def grp(d):
    e = d['excess'].dropna()
    if len(e) == 0:
        return dict(n=0)
    return dict(n=int(len(e)), mean=float(e.mean()), median=float(e.median()), win=float((e > 0).mean()), max=float(e.max()), min=float(e.min()),
                t=float(e.mean() / (e.std(ddof=1) / np.sqrt(len(e)))) if len(e) > 2 else float('nan'), hold_days=float(d['hold_days'].mean()))


tl = {}
for r in s44lib.RULES:
    kp = s44lib.keep(F1, r)
    o = {}
    for per in ('train', 'valid', 'all'):
        m = np.ones(len(F1), bool) if per == 'all' else (F1['period'] == per).to_numpy()
        d = F1[m]; k = kp[m]
        o[per] = dict(all=grp(d), kept=grp(d[k]), removed=grp(d[~k]))
        if (~k).sum() >= 1:
            o[per]['month_cluster_removed_minus_all'] = s44lib.month_cluster_diff(d, ~k)
            o[per]['removed_top5'] = [dict(month=x.month, code=x.code, surge=round(float(x.surge), 3), pct=round(float(x.pct), 4), excess=float(x.excess))
                                      for x in d[~k].sort_values('excess', ascending=False).head(5).itertuples()]
    tl[r['id']] = o
res['removed_trade_level'] = tl

# ---------------------------------------------------------------- (3) 未登記補充
kp = s44lib.keep(F1, RULE)
F1['cm'] = F1['mper'].dt.month; F1['year'] = F1['mper'].dt.year
res['frozen_removed_by_calmonth_group'] = {g: dict(removed=grp(F1[~kp & sel]), kept=grp(F1[kp & sel]))
                                           for g, sel in (('1、2 月營收', F1['cm'].isin([1, 2])), ('其他月份', ~F1['cm'].isin([1, 2])))}
res['frozen_removed_by_year'] = {int(y): dict(n_removed=int((~kp[g.index]).sum()), removed_mean=grp(g[~kp[g.index]]).get('mean'), kept_mean=grp(g[kp[g.index]]).get('mean'))
                                 for y, g in F1.groupby('year')}
yr = {}
for mode in ('cont', 'fresh'):
    for c in ('CUR', 'FZ'):
        yr[f'{c}|{mode}'] = {y: v['strat'] for y, v in R0['main'][f'{c}|{mode}']['yearly'].items()}
    yr[f'0050|{mode}'] = {y: v['e0050'] for y, v in R0['main'][f'CUR|{mode}']['yearly'].items()}
res['yearly'] = yr
res['meta'] = dict(runtime_s=round(time.time() - t_all, 1), maxrss_mb=rss())
(H / 'results_posthoc.json').write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str))
print(json.dumps({k: res[k] for k in ('random_control', 'frozen_removed_by_calmonth_group', 'meta')}, ensure_ascii=False, indent=0, default=str)[:5000])
for r, o in tl.items():
    print(r, {p: (o[p]['removed'].get('n'), round(o[p]['removed'].get('mean', np.nan), 4), round(o[p]['kept'].get('mean', np.nan), 4),
                  o[p].get('month_cluster_removed_minus_all')) for p in ('train', 'valid')})
