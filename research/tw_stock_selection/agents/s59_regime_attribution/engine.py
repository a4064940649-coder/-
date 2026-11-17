#!/usr/bin/env python3
"""S59 第 1 步：重跑凍結規則（不改任何參數），輸出策略層資料到暫存，並做驗收。
  python3 -I -B engine.py oos   → 2008-01～2015-06；月報酬須等於 S55 oos_monthly.csv（< 1e-9）、交易等於 oos_trades.csv.gz
  python3 -I -B engine.py is    → 2015-07～2026-10；日淨值須等於 navs_official.csv.gz 的 E0|W10_70（< 0.01 元）
輸出 scratchpad/s59/{which}_monthly.csv、{which}_trades.csv、{which}_signals.csv、{which}_daily.csv；驗收寫 engine_check.json。"""
import sys, json, time, resource
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import s59lib as X                                          # noqa: E402

which = sys.argv[1]
assert which in ('oos', 'is')
t0 = time.time()
rss = lambda: round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
P, U, RT, sg, fix = X.build(which)
nav, lg, st = X.L.sim(P, sg, U, cap=1e6)
e50 = pd.Series(P.E50, index=P.IDX)
D0 = P.IDX[P.t0]
nav = nav.loc[D0:]
M = X.monthly(nav, e50); M['excess'] = M['strategy'] - M['b0050']
chk = {}
if which == 'oos':
    ref = pd.read_csv(X.AG / 's55_pre2015_oos' / 'oos_monthly.csv', index_col=0)
    ref.index = pd.PeriodIndex(ref.index, freq='M')
    chk['monthly_max_abs_diff'] = float((M[['strategy', 'b0050']] - ref.loc[M.index, ['strategy', 'b0050']]).abs().max().max())
    chk['n_months'] = int(len(M))
    rt = pd.read_csv(X.AG / 's55_pre2015_oos' / 'oos_trades.csv.gz', dtype={'code': str})
    chk['n_trades_ref'] = int(len(rt)); chk['n_trades'] = int(len(lg))
    chk['trades_pnl_max_abs_diff'] = float(np.abs(lg['pnl'].to_numpy() - rt['pnl'].to_numpy()).max()) if len(rt) == len(lg) else None
    ok = chk['monthly_max_abs_diff'] < 1e-9 and chk['n_trades'] == chk['n_trades_ref'] and chk['trades_pnl_max_abs_diff'] < 1e-6
else:
    off = pd.read_csv('/home/user/-/xq_scripts/site/navs_official.csv.gz', index_col=0, parse_dates=True)['E0|W10_70']
    d = (nav.reindex(off.index) - off).abs()
    chk['nav_max_abs_diff_yuan'] = float(d.max()); chk['n_days'] = int(off.notna().sum()); chk['n_months'] = int(len(M))
    x = lg[lg['why'] != 'open_end']; chk['n_closed'] = int(len(x))
    ok = chk['nav_max_abs_diff_yuan'] < 0.01 and len(M) == 136 and chk['n_closed'] == X.L.ACC['n_closed']
chk['ok'] = bool(ok)

# 交易
T_ = lg.copy()
T_['code'] = [str(P.COLS[int(j)]) for j in T_['j']]
T_['ent_date'] = [str(P.IDX[int(t)].date()) for t in T_['ent_t']]
T_['exit_date'] = [str(P.IDX[int(t)].date()) if np.isfinite(t) and int(t) < P.T else None for t in T_['exit_t']]
T_['ret'] = T_['proceeds'] / T_['cost'] - 1
T_['hold_days'] = [(int(min(e, P.T - 1)) if np.isfinite(e) else P.T - 1) - int(s) for s, e in zip(T_['ent_t'], T_['exit_t'])]
keep = [c for c in ('code', 'ent_date', 'exit_date', 'why', 'shares', 'cost', 'proceeds', 'pnl', 'ret', 'hold_days', 'ent_t', 'exit_t', 'j') if c in T_.columns]
T_[keep].to_csv(X.SCR / f'{which}_trades.csv', index=False)

# 訊號層：每月選出檔數、驚奇名單（股池＋60 日均線上，SURGE 前 10 名）平均 SURGE
rows = []
for i in sg['info']:
    k = i['k']; m = pd.Period(i['month'], 'M')
    s = X.cslib.ins15_mask_row(P, m, k, RT['S'].loc[m].to_numpy())
    ok_ma = U[k] & np.isfinite(s) & P.F_MA60[k]
    o = np.flatnonzero(ok_ma)[np.argsort(-s[ok_ma], kind='stable')]
    top = o[:10]
    tg = [P.ci[c] for c in i['targets']]
    rows.append(dict(rev_month=i['month'], sig_date=i['sig_date'], n_ma=i['n_ma'], n_tgt=i['n_tgt'],
                     surge_top10=float(np.mean(s[top])) if len(top) else np.nan, surge_tgt=float(np.mean(s[tg])) if tg else np.nan,
                     n_univ_surge=int((U[k] & np.isfinite(s)).sum()), n_top10_ret20_gt10=int((P.RET20[k][top] > 0.10).sum())))
pd.DataFrame(rows).to_csv(X.SCR / f'{which}_signals.csv', index=False)

# 每日個股部位比重（Σ 股數 × 原始收盤 ÷ 淨值）與持股數
w = np.zeros(P.T); npos = np.zeros(P.T)
Craw = P.CF if hasattr(P, 'CF') else P.C
for r in lg.itertuples():
    a = int(r.ent_t); b = int(r.exit_t) if np.isfinite(r.exit_t) else P.T
    b = min(b, P.T)
    j = int(r.j)
    px = pd.Series(Craw[a:b, j]).ffill().to_numpy()
    w[a:b] += np.nan_to_num(px * r.shares)
    npos[a:b] += 1
navfull = pd.Series(np.nan, index=P.IDX); navfull.loc[nav.index] = nav
D = pd.DataFrame(dict(nav=navfull, e50=e50, mv_stock=w, npos=npos), index=P.IDX).loc[D0:]
D['w_act'] = D['mv_stock'] / D['nav']
D.to_csv(X.SCR / f'{which}_daily.csv')
M.to_csv(X.SCR / f'{which}_monthly.csv')
chk['avg_npos_engine'] = float(st['avg_npos']); chk['avg_npos_mine'] = float(D['npos'].mean())
chk['which_close_used'] = 'CF' if hasattr(P, 'CF') else 'C'
chk['runtime_s'] = round(time.time() - t0, 1); chk['maxrss_mb'] = rss()
f = HERE / 'engine_check.json'
allc = json.loads(f.read_text()) if f.exists() else {}
allc[which] = chk
f.write_text(json.dumps(allc, ensure_ascii=False, indent=1))
print(json.dumps(chk, ensure_ascii=False, indent=1))
assert ok, '驗收失敗'
