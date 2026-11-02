import sys, json; from pathlib import Path; HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
import numpy as np, pandas as pd, rb_core as B
OUT = HERE / 'out'; AG = B.AG; NB = pd.read_parquet(OUT / 'navs_b.parquet'); NA = pd.read_parquet(OUT / 'navs_a.parquet')
nav = {k: B.avg_nav([NB[f'AB_{k}|normal|s{s}'].dropna() for s in range(40)]) for k in ('V7', 'V7_B10')}
V4N = pd.read_parquet(AG / 'o_v4_revenue_patch_official' / 'navs_v4.parquet'); V6N = pd.read_parquet(AG / 'o_v6_price_panel_official' / 'navs_v6.parquet')
SD4 = list(pd.to_datetime(pd.read_csv(AG / 'o_v4_revenue_patch_official' / 'plan_V4.csv')['signal'])) + [B.END]
res = {}
def paired(na, nb, sd):
    D = pd.DatetimeIndex([d for d in sd if d in na.index and d in nb.index])
    ra = (na.reindex(D).shift(-1) / na.reindex(D) - 1).iloc[:-1]; rb = (nb.reindex(D).shift(-1) / nb.reindex(D) - 1).iloc[:-1]
    d = (ra - rb); d = d[d.index >= '2006-04-01']; nz = d[d.abs() > 1e-12]
    return dict(n=len(d), mean_pp=round(d.mean() * 100, 5), t=round(B.tstat(d), 4), n_nonzero=len(nz), first=str(nz.index.min().date()), last=str(nz.index.max().date()),
                max_month=str(d.idxmax().date()), t_drop_max=round(B.tstat(d.drop(d.idxmax())), 4))
res['AB_V7-V4|V4grid'] = paired(nav['V7'], V4N['AB_V4|normal'].dropna(), SD4)
res['AB_V7_B10-V4_B10|V4grid'] = paired(nav['V7_B10'], V4N['AB_V4_B10|normal'].dropna(), SD4)
# 2019 起年化（以 2019 第一個訊號日收盤為基期）
SD = B.official_days() + [B.END]
for k in ('V7', 'V7_B10'):
    n = nav[k]; a0 = [d for d in SD if d >= pd.Timestamp('2019-01-01')][0]
    yrs = (B.END - a0).days / 365.25
    e = B.E50.reindex(n.index)
    res[f'AB_{k}|POST2019|sigbase'] = dict(base=str(a0.date()), CAGR=round(((n[B.END] / n[a0]) ** (1 / yrs) - 1) * 100, 3), E50=round(((e[B.END] / e[a0]) ** (1 / yrs) - 1) * 100, 3))
# 周轉
pl, nb = B.plan(B.official_days(), unavail=B.unavail_mc(0))
_, to = B.simulate(pl, *B.rates('normal'), B.TPOS[SD[0]], B.TPOS[B.END])
res['turnover_per_rebalance_s0'] = round(to, 4); res['annual_turnover_s0'] = round(to * 12, 2); res['nblk_s0'] = nb
pl0, nb0 = B.plan(B.official_days()); res['nblk_V7_official'] = nb0
json.dump(res, open(OUT / 'stats2.json', 'w'), indent=1); print(json.dumps(res, indent=1))
