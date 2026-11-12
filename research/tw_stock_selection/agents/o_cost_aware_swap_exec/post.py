#!/usr/bin/env python3
"""凍結後補充（只報不選、描述）：
(1) 參數鄰域：CA τ75／85／95、MT F*×0.5／×2、DC5C 3／10 日（100 萬一般，seeds 0～4，全期）對同 seeds 的 BASE 配對差（挑選期、驗證期、保留期、全期）；
(2) P3 子樣本：探索 E4 遞補配對（挑選期）依被換出候選股價 < 50 元、跳動點占價 > 0.5%、當日新進檔數 ≤ 2 分組；
(3) 預測 P1～P7 的逐條核對（讀 iterations.json 與 results.json['OOS']）。寫入 results.json['POST']。
python3 -I -B post.py"""
import sys, json
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); import cas_lib as CL; import run_lib as RL; sys.path.pop(0)
RES = HERE / 'results.json'
R = json.loads(RES.read_text(encoding='utf-8'))
for k in ('PREREG', 'PREREG_POWER', 'PREREG_FREEZE'): assert CL.canon(R[k]) == R[k + '_sha256']
assert 'OOS' in R
IT = json.loads((HERE / 'iterations.json').read_text(encoding='utf-8'))
NB = {'CA75': {'ca': 0.75}, 'CA85': {'ca': 0.85}, 'CA95': {'ca': 0.95}, 'MTx0.5': {'mt': 0.5}, 'MTx2': {'mt': 2.0}, 'DC3C': {'dc': 3, 'dcmode': 'C'}, 'DC10C': {'dc': 10, 'dcmode': 'C'},
      'BASE': {}, 'CA80': {'ca': 0.8}, 'CA90': {'ca': 0.9}, 'MT': {'mt': 1.0}, 'DC5C': {'dc': 5, 'dcmode': 'C'}}
out = {}
res = {}
for nm, o in NB.items():
    res.update(RL.run_batch(RL.jobs_for(f'NB|{nm}', '100萬', 'normal', o, seeds=range(5)), nproc=2, log=f'nb:{nm}', ckpt=f'post_nb_{nm}.pkl'))
P = {}
for nm in NB:
    if nm == 'BASE': continue
    P[nm] = {per: RL.paired(res, f'NB|{nm}', 'NB|BASE', a, b)['worst'] for per, (a, b) in
             dict(selection=('2006-01-01', '2014-12-31'), validation=('2015-01-01', '2020-12-31'), holdout=('2021-01-01', '2026-12-31'), full=('2006-01-01', '2026-12-31')).items()}
out['neighbors_seeds0_4'] = P
NV = pd.DataFrame({f'{k[0]}|{k[1]}|{k[2]}': v['nav'] for k, v in res.items()}).astype('float32')
NV.to_parquet(HERE / 'data/post_neighbor_navs.parquet', compression='zstd')
# P3 子樣本
sub = {}
for q in (80, 90):
    A = pd.read_csv(HERE / f'data/explore_ca_pairs_tau{q}.csv.gz')
    tk = lambda p: np.select([p < 10, p < 50, p < 100, p < 500, p < 1000], [0.01, 0.05, 0.1, 0.5, 1.0], 5.0) / p
    A['tk_x'] = tk(A.px_x.values); nn = A.groupby(['j', 'signal']).x.transform('size')
    g = {}
    for lab, m in (('all', A.index == A.index), ('px_x<50', A.px_x < 50), ('px_x>=50', A.px_x >= 50), ('tick_x>0.5%', A.tk_x > 0.005), ('tick_x<=0.5%', A.tk_x <= 0.005),
                   ('n_repl_day<=2', nn <= 2), ('n_repl_day>2', nn > 2)):
        s = A[m]; bym = s.groupby('signal')[['d_gross', 'd_net']].mean()
        g[lab] = dict(n=int(len(s)), d_gross=float(s.d_gross.mean()), d_net=float(s.d_net.mean()), t_net_bydate=CL.tstat(bym.d_net), t_gross_bydate=CL.tstat(bym.d_gross))
    sub[f'tau{q}'] = g
out['P3_subsamples_selection'] = sub
r = json.loads(RES.read_text(encoding='utf-8')); r['POST'] = out
RES.write_text(CL.dumps(r), encoding='utf-8')
print(CL.dumps(out)[:5000])
