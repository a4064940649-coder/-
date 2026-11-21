#!/usr/bin/env python3
"""S85 第 0 步（不算任何合併結果）：用正式管線同一套開關重跑現行 E0｜W10_70 標準路徑（100 萬、位移 0），
取出 (1) 逐日淨值（驗收：與 xq_scripts/site/navs_official.csv.gz 逐日相同）、(2) 逐日個股占比 stockfrac、
(3) 換股執行日、(4) 逐筆交易；再跑一次「禁買報酬率前 5% 交易」（引擎 block，名額空著照規則放 0050）。
只讀共用引擎、不改任何既有檔。用法：python3 -I -B surge_sleeve.py"""
import sys, json, gc, math, resource, time
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent
AG = HERE.parent
TMP = Path('/tmp/claude-0/-home-user--/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/offfix/cache')
t0 = time.time()
sys.path.insert(0, str(AG / 's19_xq_threshold'))
import s19lib
from s19lib import s10lib, s9lib, s8lib, s7lib, s6lib, s5lib, s4lib, s3lib, s2lib, stlib, cslib, COSTS
for mod, attr in ((s19lib, 'S19'), (s10lib, 'S10'), (s9lib, 'S9'), (s8lib, 'S8'), (s7lib, 'S7'), (s6lib, 'S6'), (s5lib, 'S5'),
                  (s4lib, 'S4'), (s3lib, 'S3'), (s2lib, 'S2'), (stlib, 'S1'), (cslib, 'TMP')):
    setattr(mod, attr, TMP)
cslib.EXRIGHT_FIX = True; cslib.DEADLINE_FIX = True; cslib.INS15_MASK = True
P = s19lib.load(); s19lib.e50_fix_suspension(P)
INNOV, _ = s4lib.innov_panel(P)
U = stlib.universe(P, 'twse') & ~INNOV
S, _ = s19lib.rev_std(P)
avA = s19lib.a10_rows(P, list(S.index))
Sg = s10lib.make_signal_s10(P, U, dict(S=S, av=avA), shift=0)
kw = dict(sat_w=0.7, sizing='W10', lock=False, cost=COSTS['strict'], capital=1e6)
nav, lg, st = s10lib.simulate_s10(P, Sg, U, **kw)
off = pd.read_csv('/home/user/-/xq_scripts/site/navs_official.csv.gz', index_col=0, parse_dates=True)
a = nav.reindex(off.index)
dmax = float(np.nanmax(np.abs(a.to_numpy() - off['E0|W10_70'].to_numpy())))
assert dmax < 0.01, dmax
# 報酬率前 5% 交易（含期末未平倉，依 ret 由大到小；筆數＝ceil(0.05×N)）
n5 = math.ceil(0.05 * len(lg))
top = lg.sort_values('ret', ascending=False, kind='stable').head(n5)
blk = {}
for r in top.itertuples():
    blk.setdefault(int(r.j), []).append((int(r.ent_t), int(r.exit_t)))
nav_b, lg_b, st_b = s10lib.simulate_s10(P, Sg, U, block=blk, **kw)
e50 = pd.Series(P.E50, index=P.IDX)
ex = np.zeros(P.T, dtype=int)
for x in Sg['info']:
    if int(x['k']) + 1 < P.T: ex[int(x['k']) + 1] = 1
out = pd.DataFrame({'nav': nav, 'stockfrac': st['stockfrac'], 'nav_blk5pct': nav_b, 'stockfrac_blk5pct': st_b['stockfrac'],
                    'e50': e50, 'exec_day': pd.Series(ex, index=P.IDX)}).loc[off.index[0]:off.index[-1]]
out.index.name = 'date'
out.to_csv(HERE / 'surge_daily.csv.gz', float_format='%.6f')
cols = [c for c in ('code', 'j', 'ent_t', 'exit_t', 'ret', 'pnl', 'why', 'hold_days') if c in lg.columns]
lg2 = lg[cols].copy(); lg2['ent_date'] = P.IDX[lg['ent_t'].astype(int)].date; lg2['exit_date'] = P.IDX[lg['exit_t'].astype(int).clip(upper=P.T - 1)].date
lg2['top5pct'] = lg2.index.isin(top.index)
lg2.to_csv(HERE / 'surge_trades.csv.gz', index=False)
info = dict(nav_vs_official_max_abs_diff=dmax, n_trades_all=int(len(lg)), n_closed=int((lg['why'] != 'open_end').sum()), n_top5pct=int(n5),
            top5pct_min_ret=float(top['ret'].min()), top5pct_pnl_share_of_trade_pnl=float(top['pnl'].sum() / lg['pnl'].sum()),
            top5pct_codes=[str(c) for c in top['code']], n_trades_blocked_run=int(len(lg_b)),
            final_nav=float(nav.iloc[-1]), final_nav_blk=float(nav_b.iloc[-1]),
            peak_mb=round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024), secs=round(time.time() - t0))
(HERE / 'surge_sleeve_check.json').write_text(json.dumps(info, ensure_ascii=False, indent=1))
print(info)
