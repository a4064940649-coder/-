#!/usr/bin/env python3
"""S95 機制檢查（描述）：每個訊號日，季線上 SURGE 前 50 名等權（無成本、訊號日收盤→下一訊號日收盤、含息累積指數 P.G）對 0050 的超額，及 0050 同期報酬；季報月 vs 其他月。
面板＝S75 連續面板、v2 口徑（同 run_v2.py v2 模式的開關）、V_BRIDGE（B2）的 SURGE。python3 -I -B mechanism.py → mechanism.json、mechanism_monthly.csv"""
import sys, json, resource
from pathlib import Path
H = Path(__file__).resolve().parent; S75 = H.parent / 's75_continuous_2008_2026'
sys.path.insert(0, str(S75))
import numpy as np, pandas as pd
import s75lib as K
from s75lib import cslib
cslib.E50_SOURCE = S75 / 'v2_2026-11-21' / 'e50_spliced.csv'; cslib.REDUCE_FIX = True
P, U = K.load_cont()
F, RT = K.surge(P, 'B2'); S, av = RT['S'], RT['av']
rows = []
for m in S.index:
    k = av[m]
    if k >= P.T or k < 61: continue
    s = S.loc[m].to_numpy()
    if int((U[k] & np.isfinite(s)).sum()) < 50: continue
    s = cslib.ins15_mask_row(P, m, k, s)
    ok = U[k] & np.isfinite(s) & P.F_MA60[k]
    o = np.flatnonzero(ok)[np.argsort(-s[ok], kind='stable')]
    rows.append((str(m), k, o))
out = []
for i, (m, k, o) in enumerate(rows[:-1]):
    k2 = rows[i + 1][1]; r = P.G[k2] / P.G[k] - 1; b = P.E50[k2] / P.E50[k] - 1
    d = dict(month=m, sig_date=str(P.IDX[k].date()), next_sig=str(P.IDX[k2].date()), n_ma=len(o), e50=float(b))
    for n in (10, 50):
        x = r[o[:n]]; d[f'top{n}_n'] = int(np.isfinite(x).sum()); d[f'top{n}_ex'] = float(np.nanmean(x) - b) if len(x) else np.nan
    out.append(d)
D = pd.DataFrame(out); D['d'] = pd.to_datetime(D.sig_date); D['y'] = D.d.dt.year; D['mo'] = D.d.dt.month
sg = pd.read_csv(S75 / 'signals.csv'); sg = sg[sg.version == 'V_BRIDGE']
chk = D.merge(sg[['month', 'sig_date']], on='month', how='left', suffixes=('', '_s75'))
agree = int((chk.sig_date == chk.sig_date_s75).sum()), int(chk.sig_date_s75.notna().sum())
D['season'] = np.where(D.y <= 2012, D.mo.isin([5, 9, 11]), D.mo.isin([3, 5, 8, 11]))
D['season_new_def'] = D.mo.isin([3, 5, 8, 11])
R = dict(sig_date_agree_with_s75=agree, maxrss_mb=round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024), periods={})
def tt(x):
    return dict(n=int(len(x)), top50_ex=float(x.top50_ex.mean()), top50_ex_median=float(x.top50_ex.median()), top50_win=float((x.top50_ex > 0).mean()), top10_ex=float(x.top10_ex.mean()), e50=float(x.e50.mean()), e50_median=float(x.e50.median()))
for nm, (a, b) in {'2008_2012': (2008, 2012), '2013_2015': (2013, 2015), '2016_2021': (2016, 2021), '2022_2026': (2022, 2026), '2008_2015': (2008, 2015), '2016_2026': (2016, 2026)}.items():
    x = D[(D.y >= a) & (D.y <= b)]
    R['periods'][nm] = {'季報月': tt(x[x.season]), '其他月': tt(x[~x.season])}
    if a <= 2012: R['periods'][nm + '_硬套3581'] = {'季報月': tt(x[x.season_new_def]), '其他月': tt(x[~x.season_new_def])}
R['by_month_2022_2026'] = {int(k): tt(v) for k, v in D[D.y >= 2022].groupby('mo')}
(H / 'mechanism.json').write_text(json.dumps(R, ensure_ascii=False, indent=1), encoding='utf-8')
D.drop(columns=['d']).round(6).to_csv(H / 'mechanism_monthly.csv', index=False)
print(agree, R['maxrss_mb'])
for k, v in R['periods'].items():
    for g, z in v.items(): print(k, g, {a: round(b, 4) for a, b in z.items()})
for k, v in R['by_month_2022_2026'].items(): print(k, {a: round(b, 4) for a, b in v.items()})
