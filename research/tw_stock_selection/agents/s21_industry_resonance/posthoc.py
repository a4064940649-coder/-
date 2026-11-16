#!/usr/bin/env python3
"""S21 事後描述（未登記，驗證期判定之後才跑；不影響判定）。輸出 results_posthoc.json。python3 -I -B posthoc.py
(1) 逐筆：F1 與 L_MA 第 11～20 名（RET20 > 10%）在 E0 持有期對 0050 的超額（s20lib.outcomes，不扣成本），依「同業 SURGE ≥ 2 至少 2 家」分組，訓練／驗證期。
(2) 隨機換股對照：每月剔除與 SWAP_P2 相同檔數的 F1（隨機挑）、從第 11～20 名 RET20 > 10% 隨機遞補相同上限，20 次；比較凍結規則是否落在隨機範圍。
(3) 凍結規則剔除了哪些現行的賺錢交易、加進了哪些（全期連續主跑，位移 0、100 萬）。"""
import sys, json, time, resource, gc
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
import s21lib
from s21lib import s20lib, s17lib, s10lib, s4lib, stlib, COSTS

H = Path(__file__).resolve().parent
assert (H / 'results.json').exists(), '先跑 validate.py'
FZ = json.loads((H / 'FROZEN.json').read_text())
fixr = lambda r: {k: (tuple(v) if isinstance(v, list) else v) for k, v in r.items()}
RULE = fixr(FZ['frozen_rule'])
t0 = time.time()
rss = lambda: round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
P = s21lib.load()
INNOV, _ = s4lib.innov_panel(P)
U = stlib.universe(P, 'twse') & ~INNOV
del INNOV
RT = s21lib.surge(P)
s17lib.e50_fix_suspension(P)
S0 = s10lib.make_signal_s10(P, U, RT, shift=0)
ks = [x['k'] for x in S0['info']]
IND, inames = s21lib.industry_matrix(P, ks)
lastb = lambda d: P.IDX[P.IDX < pd.Timestamp(d)][-1]
TA, TB, D19, DE = lastb('2016-02-01'), lastb('2022-01-01'), lastb('2019-01-01'), P.IDX[-1]
CELL = dict(sat_w=0.7, sizing='W10', lock=False)
out = {'note': '事後描述，未登記；不影響判定'}

# ---------------------------------------------------------------- (1) 逐筆
df = s21lib.f1_breadth_table(P, U, RT, S0, IND)
df = df[(df['grp'] == 'F1') | ((df['grp'] == 'R11_20') & (df['ret20'] > 0.10))].copy()
o = s20lib.outcomes(P, S0['X'], ks, df, P.T - 1)
df = df.join(o)
df['mper'] = pd.PeriodIndex(df['month'], freq='M')
df = df[df['mper'] >= pd.Period('2016-01', 'M')]
df['period'] = np.where(df['mper'] <= s21lib.TRAIN_LAST_MONTH, 'train', 'valid')
df['res'] = df['PEER'] >= 2
ev = {}
for per in ('train', 'valid'):
    for grp in ('F1', 'R11_20'):
        d = df[(df['period'] == per) & (df['grp'] == grp) & np.isfinite(df['excess'])]
        r = {}
        for lab, g in (('共振', d[d['res']]), ('不共振', d[~d['res']])):
            x = np.sort(g['excess'].to_numpy())
            r[lab] = dict(n=int(len(x)), mean=float(x.mean()) if len(x) else None, median=float(np.median(x)) if len(x) else None,
                          mean_drop_top5=float(x[:-5].mean()) if len(x) > 5 else None, hit=float((x > 0).mean()) if len(x) else None)
        r['month_cluster_共振_vs_全部'] = s20lib.month_cluster_diff(d.assign(month=d['month']), d['res'])
        ev[f'{per}|{grp}'] = r
out['event_level'] = ev
# 產業別（F1，全期）：共振與否的超額
f1 = df[(df['grp'] == 'F1') & np.isfinite(df['excess'])]
ind_tab = f1.groupby('ind').agg(n=('excess', 'size'), mean=('excess', 'mean'), res_share=('res', 'mean')).sort_values('n', ascending=False).head(12)
out['f1_by_industry'] = {inames[i] if i >= 0 else '缺值': dict(n=int(r.n), mean_excess=float(r['mean']), res_share=float(r.res_share)) for i, r in ind_tab.iterrows()}
print('(1) 完成', f'{time.time()-t0:.0f}s', flush=True)


# ---------------------------------------------------------------- (2) 隨機換股
def sim(S1, cap=1e6, fresh=False):
    if fresh:
        S1 = s21lib.fresh_from(S1, s21lib.VALID_FIRST_MONTH)
    S2, ekw = s10lib.variant_S(S1, 'E0')
    return s10lib.simulate_s10(P, S2, U, cost=COSTS['strict'], capital=cap, **CELL, **ekw)


ratio = json.loads((H / 'results.json').read_text())['meta']['fresh_capital_scale']
SF, lgf = s21lib.build_variant(P, U, RT, S0, IND, RULE)
navF = sim(SF, 1e6 / ratio, fresh=True)[0]; navFc, lgFc, _ = sim(SF)
navC = sim(S0, 1e6 / ratio, fresh=True)[0]; navCc, lgCc, _ = sim(S0)
base = dict(CUR=dict(valid=stlib.cagr(navC, TB, DE), since2019=stlib.cagr(navCc, D19, DE)), FZ=dict(valid=stlib.cagr(navF, TB, DE), since2019=stlib.cagr(navFc, D19, DE)))
nrem = {x['month']: len(set(x['orig']) - set(x['new'])) for x in lgf}
nadd = {x['month']: len(set(x['new']) - set(x['orig'])) for x in lgf}
rng = np.random.default_rng(21)
R = []
Sm = RT['S']
for it in range(20):
    sig = np.zeros_like(S0['sig']); rank = np.full_like(S0['rank'], np.nan)
    for x in S0['info']:
        k = x['k']; s = Sm.loc[pd.Period(x['month'], 'M')].to_numpy()
        _, o_ma, tgt = s21lib.row_parts(P, U, s, k)
        keep = list(tgt)
        nr = nrem[x['month']]
        if nr:
            drop = set(rng.choice(len(tgt), nr, replace=False).tolist())
            keep = [j for i, j in enumerate(tgt) if i not in drop]
        elig = [j for j in o_ma[10:20] if P.RET20[k, j] > 0.10]
        na = min(nadd[x['month']], len(elig), 10 - len(keep))
        if na > 0:
            add = rng.choice(len(elig), na, replace=False)
            keep += [elig[i] for i in sorted(add)]
        for j in keep:
            sig[k, j] = True; rank[k, j] = s[j]
    SR = dict(S0); SR['sig'] = sig; SR['rank'] = rank
    R.append(dict(valid=stlib.cagr(sim(SR, 1e6 / ratio, fresh=True)[0], TB, DE), since2019=stlib.cagr(sim(SR)[0], D19, DE)))
    del SR, sig, rank
    gc.collect()
rv = np.array([r['valid'] for r in R]); rs = np.array([r['since2019'] for r in R])
out['random_swap'] = dict(base=base, n=20, valid=dict(median=float(np.median(rv)), p10=float(np.quantile(rv, .1)), p90=float(np.quantile(rv, .9)), rank_FZ=int((rv < base['FZ']['valid']).sum())),
                          since2019=dict(median=float(np.median(rs)), p10=float(np.quantile(rs, .1)), p90=float(np.quantile(rs, .9)), rank_FZ=int((rs < base['FZ']['since2019']).sum())))
print('(2) 完成', f'{time.time()-t0:.0f}s', flush=True)

# ---------------------------------------------------------------- (3) 剔除／加入的交易
kc = lambda lg: set(zip(lg['code'], lg['ent_t']))
cset, fset = kc(lgCc), kc(lgFc)
lost = lgCc[[(c, e) not in fset for c, e in zip(lgCc['code'], lgCc['ent_t'])]].sort_values('pnl', ascending=False)
gain = lgFc[[(c, e) not in cset for c, e in zip(lgFc['code'], lgFc['ent_t'])]].sort_values('pnl', ascending=False)
tl = lambda d: [dict(code=str(r.code), ent=str(pd.Timestamp(r.ent_date).date()), ret=round(float(r.ret), 3), pnl=round(float(r.pnl))) for r in d.head(8).itertuples()]
out['trades_diff_cont'] = dict(n_only_CUR=int(len(lost)), pnl_only_CUR=float(lost['pnl'].sum()), top_only_CUR=tl(lost),
                               n_only_FZ=int(len(gain)), pnl_only_FZ=float(gain['pnl'].sum()), top_only_FZ=tl(gain),
                               bottom_only_FZ=tl(gain.sort_values('pnl')))
out['meta'] = dict(runtime_s=round(time.time() - t0, 1), maxrss_mb=rss())
(H / 'results_posthoc.json').write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str))
print(json.dumps(out, ensure_ascii=False, indent=1, default=str)[:6000])
