#!/usr/bin/env python3
"""S23 第 0 步（不算任何報酬）：建面板快取（scratchpad/s23/）、每個訊號列的環境變數（BRD、BRD3、LMA、0050 對 200 日均線）、
訓練期門檻（分位數）、各規則在訓練／驗證期觸發幾個月、會少買幾檔（只數檔數）。輸出 data_check.json、regime_features.csv。python3 -I -B prep.py"""
import sys, json, time, resource
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
import s23lib
from s23lib import s10lib, s4lib, stlib

H = Path(__file__).resolve().parent
t = time.time()
rss = lambda: round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
P = s23lib.load()
A, _ = s4lib.innov_panel(P)
assert (P.T, P.N, str(P.IDX[-1].date())) == (2876, 2142, '2026-10-22'), 'P0 失敗'
U = stlib.universe(P, 'twse') & ~A
del A
RT = s23lib.surge(P)
S0 = s10lib.make_signal_s10(P, U, RT, shift=0)
F = s23lib.regime_features(P, U, RT, S0)
F['mper'] = pd.PeriodIndex(F['month'], freq='M')
F['period'] = np.where(F['mper'] < pd.Period('2016-01', 'M'), 'pre', np.where(F['mper'] <= s23lib.TRAIN_LAST_MONTH, 'train', 'valid'))
tr = F[F['period'] == 'train']; va = F[F['period'] == 'valid']
thr = {}
for r in s23lib.RULES:
    if r['q'] is not None:
        thr[r['id']] = float(np.quantile(tr[r['var']].to_numpy(), r['q']))
eff = {}
for r in s23lib.RULES:
    trig = s23lib.triggered(F, r, thr.get(r['id'])).to_numpy()
    S1, lg = s23lib.apply_regime(S0, F, trig, r['action'])
    o = {}
    for per in ('train', 'valid'):
        msk = (F['period'] == per).to_numpy()
        ks = set(F['k'][msk])
        L = [x for x in lg if x['k'] in ks]
        o[per] = dict(n_months=int(msk.sum()), months_triggered=int(trig[msk].sum()),
                      f1_orig_in_trig=int(sum(x['n_orig'] for x in L)), f1_kept_in_trig=int(sum(x['n_new'] for x in L)),
                      f1_total=int(F['n_tgt'][msk].sum()),
                      trig_months=[str(m) for m, g in zip(F['month'][msk], trig[msk]) if g])
    eff[r['id']] = o
corr = F[F['period'] != 'pre'][['BRD', 'BRD3', 'LMA', 'n_tgt']].assign(MA200=F['MA200'].astype(int)).corr(method='spearman').round(3).to_dict()
chk = dict(panel=dict(T=P.T, N=P.N, last=str(P.IDX[-1].date())),
           n_signal_rows=dict(pre=int((F['period'] == 'pre').sum()), train=int(len(tr)), valid=int(len(va))),
           train_window_signal=[tr['sig_date'].iloc[0], tr['sig_date'].iloc[-1]], valid_window_signal=[va['sig_date'].iloc[0], va['sig_date'].iloc[-1]],
           dist={v: {per: dict(q=[float(np.quantile(d[v], q)) for q in (0.1, 0.2, 1 / 3, 0.5, 0.9)], min=float(d[v].min()), max=float(d[v].max()))
                     for per, d in (('train', tr), ('valid', va))} for v in ('BRD', 'BRD3', 'LMA', 'n_tgt')},
           ma200_weak_share=dict(train=float(tr['MA200'].mean()), valid=float(va['MA200'].mean())),
           f1_lt10_share=dict(train=float((tr['n_tgt'] < 10).mean()), valid=float((va['n_tgt'] < 10).mean())),
           f1_zero_months=dict(train=int((tr['n_tgt'] == 0).sum()), valid=int((va['n_tgt'] == 0).sum())),
           spearman=corr, thresholds=thr, rule_effect_preview=eff)
chk['runtime_s'] = round(time.time() - t, 1); chk['maxrss_mb'] = rss()
F.drop(columns=['mper']).to_csv(H / 'regime_features.csv', index=False)
(H / 'data_check.json').write_text(json.dumps(chk, ensure_ascii=False, indent=1))
print(json.dumps({k: v for k, v in chk.items() if k != 'rule_effect_preview'}, ensure_ascii=False, indent=1))
for r, o in eff.items():
    print(r, {p: {k: v for k, v in d.items() if k != 'trig_months'} for p, d in o.items()})
