#!/usr/bin/env python3
"""S30 第 0 步（寫 PREREG 之前）：只做資料與程式檢查，不算任何防禦組合或新規則的報酬。
(1) 引擎驗收 A0：EXRIGHT_FIX＝True 的現行 CUR（W10_70）全期主跑＝48.99／34.54／−32.11／381 筆。
(2) 包裝驗收 A1：經 base_swap 傳入「0050 陣列的複本、不換成本函數」→ 淨值與交易逐位相同；B＝None 也逐位相同。
(3) 防禦組合持股（每季）：股池檔數、各組檔數、組合間重疊、ROE／DY 涵蓋、平均 β（形成日事前值）、持股換手（只數檔數）。
輸出 data_check.json、holdings.csv.gz。python3 -I -B prep.py"""
import sys, json, time, resource, gc
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
import s30lib
from s30lib import s17lib, s10lib, s4lib, stlib

H = Path(__file__).resolve().parent
t_all = time.time()
rss = lambda: round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
P = s30lib.load()
INNOV, _ = s4lib.innov_panel(P)
U = stlib.universe(P, 'twse') & ~INNOV
RT = s30lib.surge(P)
fix = s17lib.e50_fix_suspension(P)
print('載入', P.T, P.N, str(P.IDX[-1].date()), P.exright_info if not isinstance(P.exright_info, dict) else {k: v for k, v in P.exright_info.items() if not isinstance(v, (list, dict))},
      f'{time.time()-t_all:.0f}s', rss(), 'MB', flush=True)
out = {'panel': dict(T=P.T, N=P.N, start=str(P.IDX[0].date()), t0=str(P.IDX[P.t0].date()), end=str(P.IDX[-1].date()), exright_fix=bool(P.exright_fix))}

# ---------------------------------------------------------------- A0／A1
D19 = P.IDX[P.IDX < pd.Timestamp('2019-01-01')][-1]
S0 = s10lib.make_signal_s10(P, U, RT, shift=0)
nav, lg, st = s30lib.sim(P, S0, U)
a0 = dict(since2019=stlib.cagr(nav, D19, P.IDX[-1]), full=stlib.cagr(nav, P.IDX[P.t0], P.IDX[-1]), maxdd=stlib.maxdd(nav, P.IDX[P.t0], P.IDX[-1]), n_trades=int((lg['why'] != 'open_end').sum()), n_trades_all=int(len(lg)))
a0['ok'] = bool(abs(a0['since2019'] - 0.4899) < 0.00005 and abs(a0['full'] - 0.3454) < 0.00005 and abs(a0['maxdd'] + 0.3211) < 0.00005 and a0['n_trades'] == 381)
Bcopy = dict(E50=P.E50.copy(), E50O=P.E50O.copy(), E50O_raw=P.E50O_raw.copy(), E50C_raw=P.E50C_raw.copy(), sb=None, tax=None)
nav2, lg2, _ = s30lib.sim(P, S0, U, B=Bcopy)
S2, ekw = s10lib.variant_S(S0, 'E0')
nav3, lg3, _ = s10lib.simulate_s10(P, S2, U, cost=s30lib.COSTS['strict'], capital=1e6, **s30lib.CELL, **ekw)
cols = ['code', 'ent_t', 'exit_t', 'why', 'cost', 'proceeds']
a1 = dict(wrapper_copy_nav_bitwise=bool(np.array_equal(nav.to_numpy(), nav2.to_numpy())), wrapper_copy_trades=bool(lg[cols].equals(lg2[cols])),
          direct_engine_nav_bitwise=bool(np.array_equal(nav.to_numpy(), nav3.to_numpy())), direct_engine_trades=bool(lg[cols].equals(lg3[cols])),
          restored=bool(stlib.etf_tick(60.0) == 0.05 and s10lib.TAX_ETF == 0.001 and P.E50 is not Bcopy['E50']))
out['acceptance'] = dict(A0=a0, A1=a1, e50_fix=fix)
assert a0['ok'] and all(a1.values()), out['acceptance']
out['cur_avg_stockfrac'] = float(np.nanmean(st['stockfrac'].to_numpy()))
del nav2, lg2, nav3, lg3, Bcopy
print('驗收', json.dumps(out['acceptance'], ensure_ascii=False, default=str), flush=True)

# ---------------------------------------------------------------- 持股
Hd, info, FR = s30lib.holdings(P, INNOV)
out['formation'] = dict(n=len(FR), first=info[0]['form'], last=info[-1]['form'])
df = pd.DataFrame(info)
out['per_formation_summary'] = {c: dict(min=float(df[c].min()), median=float(df[c].median()), max=float(df[c].max())) for c in df.columns if c not in ('form', 'trade')}
ov = {}
for a in s30lib.BASKETS:
    for b in s30lib.BASKETS:
        if a < b:
            ov[f'{a}∩{b}'] = float(np.mean([len(set(Hd[a][t]) & set(Hd[b][t])) for t, _ in FR]))
out['overlap_avg_names'] = ov
chg = {}
for b in s30lib.BASKETS:
    ts = [t for t, _ in FR]
    ch = [len(set(Hd[b][ts[i]]) - set(Hd[b][ts[i - 1]])) for i in range(1, len(ts))]
    chg[b] = dict(avg_new_names_per_quarter=float(np.mean(ch)), max=int(np.max(ch)))
out['name_turnover'] = chg
# 2026-09 最新持股（實際操作參考）
tl = FR[-1][0]
out['latest_holdings'] = {b: [str(P.COLS[j]) for j in Hd[b][tl]] for b in s30lib.BASKETS}
rows = []
for b in s30lib.BASKETS:
    for t, te in FR:
        for j in Hd[b][t]:
            rows.append((b, str(P.IDX[t].date()), str(P.IDX[te].date()), str(P.COLS[j])))
pd.DataFrame(rows, columns=['basket', 'form_date', 'trade_date', 'code']).to_csv(H / 'holdings.csv.gz', index=False)
out['meta'] = dict(runtime_s=round(time.time() - t_all, 1), maxrss_mb=rss())
(H / 'data_check.json').write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str))
print(json.dumps({k: out[k] for k in ('formation', 'per_formation_summary', 'overlap_avg_names', 'name_turnover', 'latest_holdings', 'meta')}, ensure_ascii=False, indent=0)[:5000])
