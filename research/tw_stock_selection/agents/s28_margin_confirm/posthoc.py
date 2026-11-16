#!/usr/bin/env python3
"""S28 事後描述（未登記，驗證期判定之後才跑；不影響判定）。輸出 results_posthoc.json。python3 -I -B posthoc.py
(1) 逐筆：F1 在 E0 持有期對 0050 的超額（s20lib.outcomes，不扣成本），分組：
    DET＝毛利率惡化（RY > 0 且 DGM < −2）、TROUGH＝谷底季（RY ≤ 0 且 DGM < −2）、OTHER＝其他（含缺值）、FZ＝凍結規則判定（RY > 0、DGM < −2、DOPM < −2）；
    平均、中位數、去最賺 5 筆、勝率、月份群聚差 t（組 − 當月 F1 全部）。訓練／驗證期分開。
(2) 隨機剔除對照：每月從 F1 隨機剔除與凍結規則相同檔數，20 次；驗證窗（重新起跑）與 2019 起（全期連續），位移 0、100 萬。
(3) 凍結規則擋掉了哪些現行交易（全期連續主跑），被擋的贏家是不是由衰轉盛（μ ≤ 0）。"""
import sys, json, time, resource, gc
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
import s28lib
from s28lib import s20lib, s17lib, s10lib, s4lib, stlib, COSTS

H = Path(__file__).resolve().parent
assert (H / 'results.json').exists(), '先跑 validate.py'
FZ = json.loads((H / 'FROZEN.json').read_text())
RULE = FZ['frozen_rule']
t0 = time.time()
rss = lambda: round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
P = s28lib.load()
INNOV, _ = s4lib.innov_panel(P)
U = stlib.universe(P, 'twse') & ~INNOV
del INNOV
G = s20lib.growth_features(P)
RT = dict(S=G['S'], av=G['av'])
FQ = s28lib.fin_quarters()
s17lib.e50_fix_suspension(P)
S0 = s10lib.make_signal_s10(P, U, RT, shift=0)
ks = [x['k'] for x in S0['info']]
lastb = lambda d: P.IDX[P.IDX < pd.Timestamp(d)][-1]
TB, D19, DE = lastb('2022-01-01'), lastb('2019-01-01'), P.IDX[-1]
CELL = dict(sat_w=0.7, sizing='W10', lock=False)
out = {'note': '事後描述，未登記；判定之後才跑，不影響判定'}

# ---------------------------------------------------------------- (1) 逐筆
df = s28lib.f1_margin_table(P, U, RT, S0, FQ, G)
df = df[df['grp'] == 'F1'].copy()
o = s20lib.outcomes(P, S0['X'], ks, df, P.T - 1)
df = df.join(o)
df['mper'] = pd.PeriodIndex(df['month'], freq='M')
df = df[df['mper'] >= pd.Period('2016-01', 'M')]
df['period'] = np.where(df['mper'] <= s28lib.TRAIN_LAST_MONTH, 'train', 'valid')
df['DET'] = (df['RY'] > 0) & (df['DGM'] < -2)
df['TROUGH'] = (df['RY'] <= 0) & (df['DGM'] < -2)
df['FZ'] = df['DET'] & (df['DOPM'] < -2)
df['OTHER'] = ~(df['DET'] | df['TROUGH'])


def desc(x):
    x = np.sort(x[np.isfinite(x)])
    if not len(x):
        return dict(n=0)
    return dict(n=int(len(x)), mean=float(x.mean()), median=float(np.median(x)), mean_drop_top5=float(x[:-5].mean()) if len(x) > 5 else None,
                hit=float((x > 0).mean()), max=float(x[-1]))


ev = {}
for per in ('train', 'valid', 'all'):
    d = df if per == 'all' else df[df['period'] == per]
    d = d[np.isfinite(d['excess'])]
    r = {lab: desc(d.loc[d[lab], 'excess'].to_numpy()) for lab in ('DET', 'TROUGH', 'FZ', 'OTHER')}
    r['ALL_F1'] = desc(d['excess'].to_numpy())
    r['NOT_DET'] = desc(d.loc[~d['DET'], 'excess'].to_numpy())
    for lab in ('DET', 'FZ', 'TROUGH'):
        r[f'month_cluster_{lab}_vs_F1'] = s20lib.month_cluster_diff(d, d[lab])
    r['DET_minus_NOT_DET_mean'] = (r['DET']['mean'] - r['NOT_DET']['mean']) if r['DET']['n'] else None
    r['DET_minus_NOT_DET_median'] = (r['DET']['median'] - r['NOT_DET']['median']) if r['DET']['n'] else None
    ev[per] = r
out['event_level'] = ev
# 最賺的 F1 交易（全期）在判定上的位置
top = df[np.isfinite(df['excess'])].sort_values('excess', ascending=False).head(15)
out['top15_F1_excess'] = [dict(code=r.code, month=r.month, excess=round(float(r.excess), 3), RY=round(float(r.RY), 3) if np.isfinite(r.RY) else None,
                               DGM=round(float(r.DGM), 2) if np.isfinite(r.DGM) else None, DOPM=round(float(r.DOPM), 2) if np.isfinite(r.DOPM) else None,
                               mu_le0=bool(r.MU <= 0) if np.isfinite(r.MU) else None, DET=bool(r.DET), FZ=bool(r.FZ)) for r in top.itertuples()]
out['FZ_flagged_events'] = [dict(code=r.code, month=r.month, excess=round(float(r.excess), 3) if np.isfinite(r.excess) else None, RY=round(float(r.RY), 3),
                                 DGM=round(float(r.DGM), 2), DOPM=round(float(r.DOPM), 2), mu_le0=bool(r.MU <= 0) if np.isfinite(r.MU) else None)
                            for r in df[df['FZ']].sort_values('month').itertuples()]
print('(1) 完成', f'{time.time()-t0:.0f}s', flush=True)


# ---------------------------------------------------------------- (2) 隨機剔除
def sim(S1, cap=1e6, fresh=False):
    if fresh:
        S1 = s28lib.fresh_from(S1, s28lib.VALID_FIRST_MONTH)
    S2, ekw = s10lib.variant_S(S1, 'E0')
    return s10lib.simulate_s10(P, S2, U, cost=COSTS['strict'], capital=cap, **CELL, **ekw)


ratio = json.loads((H / 'results.json').read_text())['meta']['fresh_capital_scale']
SF, lgf = s28lib.build_variant(P, U, RT, S0, FQ, RULE, {})
navF = sim(SF, 1e6 / ratio, fresh=True)[0]; navFc, lgFc, _ = sim(SF)
navC = sim(S0, 1e6 / ratio, fresh=True)[0]; navCc, lgCc, _ = sim(S0)
base = dict(CUR=dict(valid=stlib.cagr(navC, TB, DE), since2019=stlib.cagr(navCc, D19, DE)), FZ=dict(valid=stlib.cagr(navF, TB, DE), since2019=stlib.cagr(navFc, D19, DE)))
nrem = {x['month']: len(set(x['orig']) - set(x['new'])) for x in lgf}
rng = np.random.default_rng(28)
R = []
for it in range(20):
    sig = S0['sig'].copy(); rank = S0['rank'].copy()
    for x in S0['info']:
        nr = nrem[x['month']]
        if nr:
            k = x['k']
            drop = rng.choice(len(x['targets']), nr, replace=False)
            for i in drop:
                j = P.ci[x['targets'][i]]
                sig[k, j] = False; rank[k, j] = np.nan
    SR = dict(S0); SR['sig'] = sig; SR['rank'] = rank
    R.append(dict(valid=stlib.cagr(sim(SR, 1e6 / ratio, fresh=True)[0], TB, DE), since2019=stlib.cagr(sim(SR)[0], D19, DE)))
    del SR, sig, rank
    gc.collect()
rv = np.array([r['valid'] for r in R]); rs = np.array([r['since2019'] for r in R])
out['random_removal'] = dict(base=base, n=20, n_removed_total=int(sum(nrem.values())),
                             n_removed_valid=int(sum(v for m, v in nrem.items() if pd.Period(m, 'M') >= s28lib.VALID_FIRST_MONTH)),
                             valid=dict(median=float(np.median(rv)), p10=float(np.quantile(rv, .1)), p90=float(np.quantile(rv, .9)), min=float(rv.min()), max=float(rv.max()),
                                        n_random_below_FZ=int((rv < base['FZ']['valid']).sum())),
                             since2019=dict(median=float(np.median(rs)), p10=float(np.quantile(rs, .1)), p90=float(np.quantile(rs, .9)), min=float(rs.min()), max=float(rs.max()),
                                            n_random_below_FZ=int((rs < base['FZ']['since2019']).sum())))
print('(2) 完成', f'{time.time()-t0:.0f}s', flush=True)

# ---------------------------------------------------------------- (3) 擋掉的交易
kc = lambda lg: set(zip(lg['code'], lg['ent_t']))
cset, fset = kc(lgCc), kc(lgFc)
lost = lgCc[[(c, e) not in fset for c, e in zip(lgCc['code'], lgCc['ent_t'])]].sort_values('pnl', ascending=False)
gain = lgFc[[(c, e) not in cset for c, e in zip(lgFc['code'], lgFc['ent_t'])]].sort_values('pnl', ascending=False)


def mu_at(code, sig_date):
    m = [x['month'] for x in S0['info'] if x['sig_date'] <= str(pd.Timestamp(sig_date).date())]
    if not m:
        return None
    v = G['MUAMT'].at[pd.Period(m[-1], 'M'), code] if code in G['MUAMT'].columns else np.nan
    return bool(v <= 0) if np.isfinite(v) else None


tl = lambda d, n=10: [dict(code=str(r.code), ent=str(pd.Timestamp(r.ent_date).date()), ret=round(float(r.ret), 3), pnl=round(float(r.pnl)),
                          mu_le0=mu_at(str(r.code), r.sig_date)) for r in d.head(n).itertuples()]
out['trades_diff_cont'] = dict(n_only_CUR=int(len(lost)), pnl_only_CUR=float(lost['pnl'].sum()), top_only_CUR=tl(lost), bottom_only_CUR=tl(lost.sort_values('pnl'), 5),
                               n_only_FZ=int(len(gain)), pnl_only_FZ=float(gain['pnl'].sum()), top_only_FZ=tl(gain, 5))
out['meta'] = dict(runtime_s=round(time.time() - t0, 1), maxrss_mb=rss())
(H / 'results_posthoc.json').write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str))
print(json.dumps(out, ensure_ascii=False, indent=1, default=str)[:12000])
