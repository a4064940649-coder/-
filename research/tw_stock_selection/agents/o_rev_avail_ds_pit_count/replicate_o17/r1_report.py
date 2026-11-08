import pandas as pd, numpy as np, json
D = pd.read_csv('rep_slot_avail.csv', dtype={'code': str})
def per(s):
    if s < '2014': return '2006-2013'
    if s < '2019': return '2014-2018'
    if s < '2024-06': return '2019-2024-05'
    return '2024-06+'
D['per'] = D.sig.map(per)
F = D[D.formal]
print(pd.crosstab(F.per, F.state, margins=True))
print('V5 AVAIL', (F.state_V5 == 'AVAIL').sum())
A = D[~D.formal]; print('appendix', A.state.value_counts().to_dict())
up = F.state != 'AVAIL'; lo = F.state == 'UNAVAIL'
def eff(mask, den, name):
    s = F[mask]; n = len(s)
    top = s.code.value_counts()
    if n == 0: print(f'{name}: 0/{den}'); return
    print(f'{name}: {n}/{den}={n/den:.4%} days={s.sig.nunique()} issuers={s.code.nunique()} top={top.index[0]}({top.iloc[0]}) drop={(n-top.iloc[0])/den:.4%}')
eff(up & (F.sig < '2014'), 930, 'r1a upper'); eff(lo & (F.sig < '2014'), 930, 'r1a lower')
eff(up, 2450, 'r1b upper'); eff(lo, 2450, 'r1b lower')
eff(up & (F.sig >= '2019'), 920, 'r1c upper'); eff(lo & (F.sig >= '2019'), 920, 'r1c lower')
u = F[(F.per == '2024-06+')]
print('2024-06+ UNDET by day', u[u.state == 'UNDET'].sig.value_counts().to_dict())
na = u[u.state != 'AVAIL']; print('2nd cutoff non-AVAIL', len(na), 'seen24', na.seen24.fillna(False).astype(bool).sum())
print('2019+ DS/DgtS days', F[F.sig >= '2019'].groupby('case').sig.nunique().to_dict())
# 對研究員
R = pd.read_csv('../data/slot_avail.csv', dtype={'code': str})
m = D.merge(R, on=['sig', 'rank'], suffixes=('', '_r'))
print('merged', len(m), 'code diff', (m.code != m.code_r).sum(), 'state diff', (m.state != m.state_r).sum(), 'V5 diff', (m.state_V5 != m.state_V5_r).sum())
print(m[m.state != m.state_r][['sig', 'code', 'state', 'state_r', 'src']].to_string())
