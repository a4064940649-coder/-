#!/usr/bin/env python3
"""S55 寫 PREREG 之前的準備（不算任何 2015-07 以前的報酬）：
  python3 -I -B prep.py is   → 用官方面板重現現行正式數字（2019 起 48.76%、全期 34.40%、回撤 −32.12%、381 筆），
                               月淨值與 S52 family_monthly_nav.csv.gz 的 OFFICIAL:E0|W10_70／0050 比對；存 scratchpad/s55/is_nav.csv.gz（已知樣本內）
  python3 -I -B prep.py oos  → 2007～2015-06 面板只算訊號層的計數（訊號月數、訊號日、股池檔數、SURGE 有值檔數、入選檔數），不跑淨值
輸出 prep_check.json（兩段各自寫入自己的鍵）。"""
import sys, json, time, resource
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import s55lib as X                                          # noqa: E402

which = sys.argv[1]
t0 = time.time()
rss = lambda: round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
out_f = HERE / 'prep_check.json'
out = json.loads(out_f.read_text()) if out_f.exists() else {}
P, U, RT, sg, fix = X.build(which)
stl = X.stlib
if which == 'is':
    assert (P.T, P.N, str(P.IDX[-1].date())) == (2876, 2142, '2026-10-22')
    nav, lg, st = X.L.sim(P, sg, U, cap=1e6)
    D0, DE = P.IDX[P.t0], P.IDX[-1]; D19 = P.IDX[P.IDX < pd.Timestamp('2019-01-01')][-1]
    x = lg[lg['why'] != 'open_end']
    got = dict(since2019=stl.cagr(nav, D19, DE), full=stl.cagr(nav, D0, DE), maxdd=stl.maxdd(nav, D0, DE), n_closed=int(len(x)))
    ok = all(abs(got[k] - X.L.ACC[k]) < 1e-12 for k in ('since2019', 'full', 'maxdd')) and got['n_closed'] == X.L.ACC['n_closed']
    e50 = pd.Series(P.E50, index=P.IDX)
    pd.DataFrame(dict(nav=nav, e50=e50)).to_csv(X.SCR / 'is_nav.csv.gz')
    m = X.monthly(nav.loc[D0:], e50)
    s52 = pd.read_csv(X.AG / 's52_snooping_adjusted_significance' / 'family_monthly_nav.csv.gz', index_col=0, parse_dates=True)
    r52 = s52[['OFFICIAL:E0|W10_70', '0050']].pct_change().iloc[1:]
    r52.index = r52.index.to_period('M'); r52 = r52.loc['2015-07':]
    j = m.index.intersection(r52.index)
    d_s = float((m.loc[j, 'strategy'] - r52.loc[j, 'OFFICIAL:E0|W10_70']).abs().max())
    d_b = float((m.loc[j, 'b0050'] - r52.loc[j, '0050']).abs().max())
    out['is'] = dict(reproduced=got, acc=X.L.ACC, bitwise_ok=bool(ok), months=int(len(m)), first=str(m.index[0]), last=str(m.index[-1]),
                     vs_s52_monthly=dict(n=int(len(j)), max_abs_diff_strategy=d_s, max_abs_diff_0050=d_b),
                     runtime_s=round(time.time() - t0, 1), maxrss_mb=rss())
    print(json.dumps(out['is'], ensure_ascii=False, indent=1))
    assert ok, 'IS 重現失敗'
else:
    info = sg['info']
    S = RT['S']; av = RT['av']
    per = []
    for mth in S.index:
        k = av[mth]
        if k >= P.T or k < 61:
            continue
        s = S.loc[mth].to_numpy()
        per.append(dict(month=str(mth), sig_row=int(k), sig_date=str(P.IDX[k].date()), n_univ=int(U[k].sum()),
                        n_univ_surge=int((U[k] & np.isfinite(s)).sum()), n_univ_surge_ma60=int((U[k] & np.isfinite(s) & P.F_MA60[k]).sum())))
    sig_months = [x['month'] for x in info]
    out['oos'] = dict(panel=dict(T=P.T, N=P.N, first=str(P.IDX[0].date()), last=str(P.IDX[-1].date()), t0=str(P.IDX[P.t0].date()),
                                 exright_info=P.exright_info, e50_missing=P.e50_missing, e50_open_missing=P.e50_open_missing, twii_missing=P.twii_missing),
                      n_signal_months=len(info), first_signal=info[0]['sig_date'] if info else None, last_signal=info[-1]['sig_date'] if info else None,
                      signal_months=sig_months, n_tgt_by_month={x['month']: x['n_tgt'] for x in info},
                      avg_n_tgt=float(np.mean([x['n_tgt'] for x in info])) if info else None,
                      months_with_avail_row=per, e50_fix=fix, runtime_s=round(time.time() - t0, 1), maxrss_mb=rss())
    print(json.dumps({k: v for k, v in out['oos'].items() if k not in ('months_with_avail_row', 'n_tgt_by_month', 'signal_months')}, ensure_ascii=False, indent=1))
out_f.write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str))
