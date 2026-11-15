#!/usr/bin/env python3
"""S18 驗收第一步：用 S10 研究員程式庫（s10lib，不改）重建 E0 訊號與 W10_70 主跑，輸出對照檔。
- 每個訊號月：訊號日、L_MA 前 20 名（代號、SURGE）、F1 買進名單（s10lib.make_signal_s10 的 info.targets）；
- E0｜W10_70 100 萬主跑（s10lib.simulate_s10）的逐筆交易與期末狀態（持股、底倉 B、淨值）。
快取與輸出都放 scratchpad/s18tmp/（結束時刪）。python3 -I -B validate_ref.py
單一行程；只讀資料，不寫 repo 以外的任何東西。"""
import sys, json, time, resource
from pathlib import Path
import numpy as np, pandas as pd

H = Path(__file__).resolve().parent
AG = H.parent
TMP = Path('/tmp/claude-0/-home-user--/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/s18tmp/ref')
TMP.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(AG / 's10_exit_details'))
import s10lib                                                     # noqa: E402
from s10lib import s9lib, s8lib, s7lib, s6lib, s5lib, s4lib, s3lib, s2lib, stlib, cslib, COSTS   # noqa: E402
# 快取全部改放本題 scratchpad（不動其他題目的目錄）
s10lib.S10 = TMP; s9lib.S9 = TMP; s8lib.S8 = TMP; s7lib.S7 = TMP; s6lib.S6 = TMP; s5lib.S5 = TMP
s4lib.S4 = TMP; s3lib.S3 = TMP; s2lib.S2 = TMP; stlib.S1 = TMP; cslib.TMP = TMP

t0 = time.time()
rss = lambda: round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
P = s10lib.load()
print('面板', P.T, P.N, str(P.IDX[-1].date()), f'{time.time()-t0:.0f}s', rss(), 'MB', flush=True)
U0 = stlib.universe(P, 'twse')
RT = s2lib.rev_tables(P)
INNOV, _ = s4lib.innov_panel(P)
U = U0 & ~INNOV
del U0, INNOV
S = s10lib.make_signal_s10(P, U, RT, shift=0)
SV = RT['S']
months = []
for x in S['info']:
    k = x['k']
    m = pd.Period(x['month'], freq='M')
    s = SV.loc[m].to_numpy()
    fin = U[k] & np.isfinite(s)
    ok_ma = fin & P.F_MA60[k]
    o_ma = np.flatnonzero(ok_ma)[np.argsort(-s[ok_ma], kind='stable')]
    top20 = [dict(code=str(P.COLS[j]), surge=float(s[j]), ret20=float(P.RET20[k, j])) for j in o_ma[:20]]
    assert sorted(int(j) for j in np.flatnonzero(~S['X'][k])) == sorted(int(j) for j in o_ma[:20])
    months.append(dict(rev_month=str(m), signal_month=str(m + 1), k=int(k), sig_date=x['sig_date'], n_ma=int(ok_ma.sum()),
                       n_univ_surge=int(fin.sum()), top20=top20, f1=x['targets']))
print('訊號月數', len(months), f'{time.time()-t0:.0f}s', rss(), 'MB', flush=True)

# E0｜W10_70 主跑（100 萬、嚴成本、開盤）
S2, kw = s10lib.variant_S(S, 'E0')
nav, lg, st = s10lib.simulate_s10(P, S2, U, sat_w=0.7, sizing='W10', lock=False, cost=COSTS['strict'], **kw)
D19 = P.IDX[P.IDX < pd.Timestamp('2019-01-01')][-1]
r19 = stlib.cagr(nav, D19, P.IDX[-1]); rfull = stlib.cagr(nav, P.IDX[P.t0], P.IDX[-1]); dd = stlib.maxdd(nav, P.IDX[P.t0], P.IDX[-1])
trades = []
for r in lg.itertuples():
    trades.append(dict(code=str(r.code), sig_date=str(r.sig_date.date()), ent_date=str(r.ent_date.date()), exit_date=str(r.exit_date.date()),
                       why=r.why, shares=int(r.shares), proceeds=float(r.proceeds), cost=float(r.cost)))
openv = float(lg.loc[lg['why'] == 'open_end', 'proceeds'].sum())
end = dict(date=str(P.IDX[-1].date()), nav=float(nav.iloc[-1]), B=float(st['B'].iloc[-1]), stock_value=openv, idle=float(nav.iloc[-1]) - openv,
           rb_trim_orders=int(st['rb_trim_orders']), e50_close_adj=float(P.E50[-1]), e50_close_raw=float(P.E50C_raw[-1]))
out = dict(panel=dict(T=int(P.T), N=int(P.N), end=str(P.IDX[-1].date())), main=dict(since2019=r19, full=rfull, maxdd=dd, n_trades=len(trades)),
           months=months, trades=trades, end_state=end, runtime_s=round(time.time() - t0, 1), maxrss_mb=rss())
(TMP / 'ref_s10.json').write_text(json.dumps(out, ensure_ascii=False))
print('主跑', round(r19, 4), round(rfull, 4), round(dd, 4), len(trades), end, f'{time.time()-t0:.0f}s', rss(), 'MB', flush=True)
