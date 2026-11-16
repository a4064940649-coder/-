#!/usr/bin/env python3
"""S25 第 0 步（不算任何策略報酬）：建面板快取（scratchpad/s25/）、每個訊號列的因子 12 個月趨勢（s24 monthly.csv）、
各規則在訓練／驗證期觸發幾個月、會少買幾檔（只數檔數）。輸出 data_check.json、trend_features.csv。python3 -I -B prep.py"""
import sys, json, time, resource
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
import s25lib
from s25lib import s10lib, s4lib, stlib

H = Path(__file__).resolve().parent
t = time.time()
rss = lambda: round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
P = s25lib.load()
A, _ = s4lib.innov_panel(P)
assert (P.T, P.N, str(P.IDX[-1].date())) == (2876, 2142, '2026-10-22'), 'P0 失敗'
U = stlib.universe(P, 'twse') & ~A
del A
RT = s25lib.surge(P)
S0 = s10lib.make_signal_s10(P, U, RT, shift=0)
FD = s25lib.load_factors()
F = s25lib.trend_features(S0, FD)
F['mper'] = pd.PeriodIndex(F['month'], freq='M')
F['period'] = np.where(F['mper'] < pd.Period('2016-01', 'M'), 'pre', np.where(F['mper'] <= s25lib.TRAIN_LAST_MONTH, 'train', 'valid'))
eff = {}
for r in s25lib.RULES:
    trig = s25lib.triggered(F, r)
    S1, lg = s25lib.apply_regime(S0, F, trig, r['action'])
    o = {}
    for per in ('train', 'valid'):
        msk = (F['period'] == per).to_numpy()
        ks = set(F['k'][msk])
        L = [x for x in lg if x['k'] in ks]
        o[per] = dict(n_months=int(msk.sum()), n_trend_known=int(np.isfinite(F['tr_' + r['var']][msk]).sum()), months_triggered=int(trig[msk].sum()),
                      f1_orig_in_trig=int(sum(x['n_orig'] for x in L)), f1_kept_in_trig=int(sum(x['n_new'] for x in L)),
                      f1_total=int(F['n_tgt'][msk].sum()),
                      trig_months=[str(m) for m, g in zip(F['month'][msk], trig[msk]) if g])
    eff[r['id']] = o
chk = dict(panel=dict(T=P.T, N=P.N, last=str(P.IDX[-1].date())),
           n_signal_rows={p: int((F['period'] == p).sum()) for p in ('pre', 'train', 'valid')},
           trend_first_known={f: (str(F.loc[np.isfinite(F['tr_' + f]), 'sig_date'].iloc[0]) if np.isfinite(F['tr_' + f]).any() else None) for f in s25lib.FACTORS},
           share_trend_le0={f: {p: float((F.loc[F['period'] == p, 'tr_' + f] <= 0).sum() / max(np.isfinite(F.loc[F['period'] == p, 'tr_' + f]).sum(), 1))
                                for p in ('train', 'valid')} for f in s25lib.FACTORS},
           corr_trend_MOM_RMOM=float(F[['tr_MOM', 'tr_RMOM']].dropna().corr().iloc[0, 1]),
           rule_effect_preview=eff,
           note='只數月份與檔數；沒有算任何策略報酬。因子趨勢本身是因子報酬（s24 已公開的 monthly.csv），門檻固定 0、不調參。')
chk['runtime_s'] = round(time.time() - t, 1); chk['maxrss_mb'] = rss()
F.drop(columns=['mper']).to_csv(H / 'trend_features.csv', index=False, float_format='%.6f')
(H / 'data_check.json').write_text(json.dumps(chk, ensure_ascii=False, indent=1))
print(json.dumps({k: v for k, v in chk.items() if k != 'rule_effect_preview'}, ensure_ascii=False, indent=1))
for r, o in eff.items():
    print(r, {p: {k: v for k, v in d.items() if k != 'trig_months'} for p, d in o.items()})
    print('  valid trig', o['valid']['trig_months'])
