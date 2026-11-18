#!/usr/bin/env python3
"""S71 事前檢查（不跑引擎、不算任何報酬）：三個規則的 SURGE 矩陣在自建面板上的涵蓋與一致性。
python3 -I -B prep.py → prep_check.json"""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
import s71lib as S
from s71lib import R, s20lib, s4lib, stlib

R.point_cache(S.TMP / 'cache_pre')
stlib.START = pd.Timestamp('2008-01-02')
P = s20lib.load()
INNOV, _ = s4lib.innov_panel(P)
U = stlib.universe(P, 'twse') & ~INNOV
Z = {r: S.rev_rule(P, r) for r in ('MASK', 'B1', 'B2')}
zm, z1, z2 = (Z[r]['S']['STD'] for r in ('MASK', 'B1', 'B2'))
pre = zm.index <= pd.Period('2012-12', 'M')
eq = lambda a, b: bool(np.array_equal(a.to_numpy(), b.to_numpy(), equal_nan=True))
out = dict(pre2013_rows_identical=dict(B1=eq(z1[pre], zm[pre]), B2=eq(z2[pre], zm[pre])),
           panel=dict(T=P.T, N=P.N, first=str(P.IDX[0].date()), last=str(P.IDX[-1].date())))
av = Z['B2']['av']
rows = []
for m in pd.period_range('2012-10', '2015-05', freq='M'):
    t = av[m]
    u = U[t - 1] if t - 1 < P.T else np.zeros(P.N, bool)          # 訊號列（可得日的前一列＝期限日）當天的股池
    rows.append(dict(month=str(m), avail_row_date=str(P.IDX[t].date()) if t < P.T else None, n_universe=int(u.sum()),
                     **{f'n_surge_{r}': int(np.isfinite(Z[r]['S']['STD'].loc[m].to_numpy()).sum()) for r in Z},
                     **{f'n_surge_in_univ_{r}': int((np.isfinite(Z[r]['S']['STD'].loc[m].to_numpy()) & u).sum()) for r in Z},
                     **{f'p90_in_univ_{r}': (float(np.nanquantile(Z[r]['S']['STD'].loc[m].to_numpy()[u], .9)) if np.isfinite(Z[r]['S']['STD'].loc[m].to_numpy()[u]).any() else None) for r in ('B1', 'B2')}))
out['coverage'] = rows
ks = Z['B2']['info']['k']
out['k'] = {f'n{n}': dict(n_est=int(k.notna().sum()), n_missing=int(k.isna().sum()), share_gt1=float((k.dropna() > 1).mean()), median_gt1=float(k[k > 1].median()),
                          share_gt2=float((k.dropna() > 2).mean())) for n, k in ks.items()}
out['k_corr_log'] = dict(n1_n3=float(np.corrcoef(np.log(ks[1]).fillna(0)[ks[1].notna() & ks[3].notna()], np.log(ks[3])[ks[1].notna() & ks[3].notna()])[0, 1]),
                         n2_n3=float(np.corrcoef(np.log(ks[2])[ks[2].notna() & ks[3].notna()], np.log(ks[3])[ks[2].notna() & ks[3].notna()])[0, 1]))
(S.HERE / 'prep_check.json').write_text(json.dumps(out, ensure_ascii=False, indent=1))
print(json.dumps({k: v for k, v in out.items() if k != 'coverage'}, ensure_ascii=False))
print(pd.DataFrame(rows).to_string())
