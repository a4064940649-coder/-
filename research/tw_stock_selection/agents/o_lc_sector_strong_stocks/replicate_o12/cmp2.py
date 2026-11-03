import sys, pickle; sys.path.insert(0, '.')
import pandas as pd, numpy as np
import rlib as R
p = pd.read_csv('../out/plans_SEL_CTRL_J0_J6_J12.csv.gz', dtype={'code': str, 'top_sectors': str})
j = int(sys.argv[1]); spec = sys.argv[2]; key = 'IND_n2_cap_c15' if spec == 'SEL' else 'CTRL_n2_cap_c15'
import os; D = pickle.load(open(os.environ.get('PD','out') + f'/plans_J{j}.pkl', 'rb'))
q = p[(p.spec == spec) & (p.j == j)]
nsec = 0; nB = 0; nX = 0; tot = 0
for (t, P), info in zip(D['plans'], D['diag']):
    ds = str(R.IDX[t].date())
    if ds < '2014-01-01': continue
    g = q[q.signal == ds]; th = dict(zip(g.code, g.weight)); tot += 1
    B = {R.COLS[k]: v for k, v in P['B'].items()}
    # 研究員 PURE 部分＝(SAT − 0.7B)/0.3（用我的 B）
    pure_th = {k: (th.get(k, 0) - 0.7 * B.get(k, 0)) / 0.3 for k in set(th) | set(B)}
    pk_th = sorted(k for k, v in pure_th.items() if v > 0.02)
    pk_me = sorted(R.COLS[k] for k in P[key])
    resid = sum(abs(v) for k, v in pure_th.items() if k not in pk_th)
    if spec == 'SEL' and set(g.top_sectors.iloc[0].split('|')) != set(info['sel']):
        nsec += 1; print('SEC', ds, g.top_sectors.iloc[0], info['sel'])
    if resid > 0.01: nB += 1; print('Bdiff', ds, round(resid, 4))
    if pk_th != pk_me:
        nX += 1; print('PICK', ds, 'th-only', sorted(set(pk_th) - set(pk_me)), 'me-only', sorted(set(pk_me) - set(pk_th)))
print('tot', tot, 'sec', nsec, 'B', nB, 'pick', nX)
