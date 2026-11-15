#!/usr/bin/env python3
"""S13 訓練期：只算訓練期（訊號日 ≤ 2021-12-31）候選的 E0 持有期超額（截在 2021-12-30），做單變量、18 條規則，依 PREREG 機械式凍結一條規則。
python3 -I -B train.py → results_train.json、FROZEN.json、FROZEN.sha256、cand_train.csv.gz。PREREG.json 的 sha256 必須相符；FROZEN.json 已存在就拒絕重跑。"""
import sys, json, hashlib, time, resource
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
import s13lib
from s13lib import s10lib, s4lib, s2lib, stlib, COSTS, FEATURES

H = Path(__file__).resolve().parent
pre = (H / 'PREREG.json').read_bytes()
sha = hashlib.sha256(pre).hexdigest()
assert sha == (H / 'PREREG.sha256').read_text().split()[0], 'PREREG.json 與 PREREG.sha256 不符'
assert not (H / 'FROZEN.json').exists(), 'FROZEN.json 已存在：訓練期只做一次'
t0 = time.time()
P = s13lib.load()
assert (P.T, P.N, str(P.IDX[-1].date())) == (2876, 2142, '2026-10-22'), 'P0 失敗'
U0 = stlib.universe(P, 'twse'); INNOV, _ = s4lib.innov_panel(P); U = U0 & ~INNOV; del U0, INNOV
RT = s2lib.rev_tables(P)
Rv = s13lib.rev_matrix(P)
FG = s13lib.fin_gm()
S = s10lib.make_signal_s10(P, U, RT, shift=0)
cand = s13lib.candidates(P, U, RT, 0)
cand = s13lib.add_features(P, RT, Rv, FG, cand)
END_TR = int(np.searchsorted(P.IDX.values, np.datetime64('2022-01-01'), side='left')) - 1     # 2021-12-30
tr = cand[cand['sig_date'] <= pd.Timestamp('2021-12-31')].copy()
sig_ks = [x['k'] for x in S['info']]
o = s13lib.outcomes(P, S['X'], sig_ks, tr, END_TR)
tr = tr.join(o)
tr = tr[np.isfinite(tr['excess'])].copy()

# 驗收 A2：W10_70 主跑實際交易（訓練期進場、規則出場、出場 ≤ END_TR）的出場日
nav, lg, _ = s10lib.simulate_s10(P, S, U, sat_w=0.7, sizing='W10', lock=False, cost=COSTS['strict'])
lg2 = lg[(lg['why'] == 'rule') & (lg['exit_t'] <= END_TR)]
mp = {(int(a), int(b)): int(c) for a, b, c in zip(tr['k'], tr['j'], tr['exit_t'])}
hits = [mp.get((int(s_), int(j))) == int(x) for s_, j, x in zip(lg2['sig_t'], lg2['j'], lg2['exit_t'])]
A2 = dict(n=len(hits), same=int(sum(hits)), frac=float(np.mean(hits)) if hits else float('nan'))
assert A2['frac'] >= 0.99, ('驗收 A2 失敗', A2)
del nav, lg

thr_bw = float(np.quantile(tr['excess'], 0.9))
tr['bw'] = tr['excess'] >= thr_bw
res = {'PREREG_sha256': sha, 'acceptance': {'A2_exit_match': A2},
       'sample': dict(n=int(len(tr)), n_months=int(tr['month'].nunique()), n_bw=int(tr['bw'].sum()), noopen_dropped=int((o['how'] == 'noopen').sum()),
                      how=tr['how'].value_counts().to_dict(), limup_open=int(tr['limup_open'].sum()), end_trunc=str(P.IDX[END_TR].date()),
                      excess_mean=float(tr['excess'].mean()), excess_median=float(tr['excess'].median()), ret_mean=float(tr['ret'].mean()),
                      f1_excess_mean=float(tr.loc[tr['f1'], 'excess'].mean()), nonf1_excess_mean=float(tr.loc[~tr['f1'], 'excess'].mean())),
       'big_winner_threshold_excess': thr_bw, 'univariate': {}, 'rules': []}
res['sample']['bw_share_f1'] = float(tr.loc[tr['f1'], 'bw'].mean()); res['sample']['bw_share_nonf1'] = float(tr.loc[~tr['f1'], 'bw'].mean())
res['sample']['top_bw'] = tr.sort_values('excess', ascending=False).head(15)[['sig_date', 'code', 'rank', 'f1', 'excess', 'hold_days']].astype(str).values.tolist()
# 現行策略最賺 5 筆的影響：最大 5 筆超額占全部超額合計
res['sample']['top5_excess_share_of_sum'] = float(tr['excess'].nlargest(5).sum() / tr['excess'].sum())

for f in FEATURES:
    g = tr[np.isfinite(tr[f])]
    q = pd.qcut(g[f], 5, labels=False, duplicates='drop')
    tab = []
    for b in sorted(q.unique()):
        x = g[q == b]
        tab.append(dict(q=int(b) + 1, n=int(len(x)), lo=float(x[f].min()), hi=float(x[f].max()), bw_share=float(x['bw'].mean()),
                        mean_excess=float(x['excess'].mean()), median_excess=float(x['excess'].median())))
    a = s13lib.auc(g[f], g['bw'])
    res['univariate'][f] = dict(n=int(len(g)), auc=a, direction=1 if a >= 0.5 else -1, auc_dir=max(a, 1 - a), quintiles=tab,
                                spearman_excess=float(g[[f, 'excess']].corr(method='spearman').iloc[0, 1]))

for f in FEATURES:
    u = res['univariate'][f]; d = u['direction']
    vals = tr[f].dropna().to_numpy()
    for qq, keep_frac in ((1 / 3, '2/3'), (1 / 2, '1/2'), (2 / 3, '1/3')):
        thr = float(np.quantile(vals, qq if d > 0 else 1 - qq))
        kp = s13lib.rule_keep(tr, f, d, thr)
        mc = s13lib.month_cluster_diff(tr, kp)
        res['rules'].append(dict(feature=f, direction=d, keep_approx=keep_frac, threshold=thr, keep_share=float(kp.mean()),
                                 bw_share_kept=float(tr.loc[kp, 'bw'].mean()), bw_share_all=float(tr['bw'].mean()),
                                 mean_excess_kept=float(tr.loc[kp, 'excess'].mean()), mean_excess_all=float(tr['excess'].mean()),
                                 d_mean=mc['mean'], t=mc['t'], n_months=mc['n_months'], auc_dir=u['auc_dir']))
elig = [r for r in res['rules'] if r['auc_dir'] >= 0.55]
weak = len(elig) == 0
if weak:
    elig = res['rules']
best = sorted(elig, key=lambda r: (-r['t'], -r['auc_dir'], r['keep_share']))[0]
FROZEN = dict(PREREG_sha256=sha, feature=best['feature'], direction=best['direction'], threshold=best['threshold'], keep_approx=best['keep_approx'],
              big_winner_threshold_excess=thr_bw, training_weak=weak, training_t=best['t'], training_auc_dir=best['auc_dir'],
              rule_text=f"候選中 {best['feature']} {'≥' if best['direction'] > 0 else '≤'} {best['threshold']:.6g} 才通過",
              frozen_at=time.strftime('%Y-%m-%dT%H:%M:%S'))
res['frozen'] = FROZEN
res['N'] = dict(N_before=2779, added_training=24, N_after_training=2803)
res['runtime_s'] = round(time.time() - t0, 1); res['maxrss_mb'] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
txt = json.dumps(FROZEN, ensure_ascii=False, indent=1)
(H / 'FROZEN.json').write_text(txt)
(H / 'FROZEN.sha256').write_text(hashlib.sha256(txt.encode()).hexdigest() + '  FROZEN.json\n')
(H / 'results_train.json').write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str))
cols = ['month', 'sig_date', 'code', 'rank', 'f1', 'surge', 'ret20'] + FEATURES + ['gm_q', 'ret', 'r50', 'excess', 'bw', 'how', 'hold_days', 'limup_open']
tr[cols].to_csv(H / 'cand_train.csv.gz', index=False, float_format='%.6g')
print(json.dumps({k: res[k] for k in ('acceptance', 'sample', 'big_winner_threshold_excess', 'frozen', 'runtime_s', 'maxrss_mb')}, ensure_ascii=False, indent=1, default=str))
for f in FEATURES:
    u = res['univariate'][f]
    print(f, 'AUC', round(u['auc'], 3), 'ρ', round(u['spearman_excess'], 3), [(t['q'], t['n'], round(t['bw_share'] * 100, 1), round(t['mean_excess'] * 100, 2)) for t in u['quintiles']])
for r in res['rules']:
    print(r['feature'], r['direction'], r['keep_approx'], round(r['threshold'], 4), 'keep', round(r['keep_share'], 3), 'bw', round(r['bw_share_kept'] * 100, 1),
          'mean', round(r['mean_excess_kept'] * 100, 2), 'd', round(r['d_mean'] * 100, 2), 't', round(r['t'], 2))
