#!/usr/bin/env python3
"""S27 獨立複製（第 1 步）：用 s20lib→s17lib→s10lib 鏈重跑 E0｜W10_70（EXRIGHT_FIX=True）20 條路徑的日淨值。
不 import／不讀 s27 的 build.py／analyze.py。SURGE 用 s17lib.rev_family(['STD'])、A10 訊號日用 rev_family 回傳的 av（cslib.rev_avail_days(...,True)）、
0050 停牌修正用 s17lib.e50_fix_suspension；股池＝上市 & ~創新板；make_signal_s10 + simulate_s10（sat_w 0.7、W10、lock False、strict）。
輸出：暫存區 navs_daily.npz（日淨值，結束刪）；replicate/engine_check.json（驗收）。用法：python3 -I -B rep_engine.py"""
import sys, json, time, resource, gc
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
AG = HERE.parent.parent
TMP = Path('/tmp/claude-0/-home-user--/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/s27rep')
CACHE = TMP / 'cache'; CACHE.mkdir(parents=True, exist_ok=True)
t0 = time.time(); rss = lambda: round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)

sys.path.insert(0, str(AG / 's20_persistent_growth'))
import s20lib                                              # noqa: E402
from s20lib import s17lib, s10lib, s9lib, s8lib, s7lib, s6lib, s5lib, s4lib, s3lib, s2lib, stlib, cslib, COSTS  # noqa: E402
s20lib.S20 = CACHE; s17lib.S17 = CACHE
for mod, attr in ((s10lib, 'S10'), (s9lib, 'S9'), (s8lib, 'S8'), (s7lib, 'S7'), (s6lib, 'S6'), (s5lib, 'S5'),
                  (s4lib, 'S4'), (s3lib, 'S3'), (s2lib, 'S2'), (stlib, 'S1'), (cslib, 'TMP')):
    setattr(mod, attr, CACHE)
cslib.EXRIGHT_FIX = True

P = s20lib.load()
assert P.exright_fix is True
s17lib.e50_fix_suspension(P)
INNOV, _ = s4lib.innov_panel(P)
U = stlib.universe(P, 'twse') & ~INNOV
del INNOV
F = s17lib.rev_family(P, ['STD'])
RT = dict(S=F['S']['STD'], av=F['av']); del F
for a in ('H', 'L', 'VOL', 'CHG', 'HF', 'VOLF', 'adjH', 'HI240', 'MA60', 'nbar', 'F_HI', 'IDXC', 'AO', 'listed', 'NEARHI', 'V60', 'liq'):
    if hasattr(P, a):
        delattr(P, a)
gc.collect()
print('載入', f'{time.time()-t0:.0f}s', rss(), 'MB', flush=True)
D0, DE = P.IDX[P.t0], P.IDX[-1]
D19 = P.IDX[P.IDX < pd.Timestamp('2019-01-01')][-1]
navs = {}
for s in range(4):
    Sg = s10lib.make_signal_s10(P, U, RT, shift=s)
    for cap in (6e5, 8e5, 1e6, 1.5e6, 2e6):
        nav, lg, st = s10lib.simulate_s10(P, Sg, U, sat_w=0.7, sizing='W10', lock=False, cost=COSTS['strict'], capital=cap)
        navs[f'{int(cap/1e4)}萬|+{s}'] = nav.loc[D0:DE].astype(float)
        del nav, lg, st
    del Sg; gc.collect()
    print('shift', s, f'{time.time()-t0:.0f}s', rss(), 'MB', flush=True)
bench = pd.Series(P.E50, index=P.IDX).loc[D0:DE].astype(float)
keys = [f'{c}萬|+{s}' for s in range(4) for c in (60, 80, 100, 150, 200)]
np.savez(TMP / 'navs_daily.npz', IDX=np.array(bench.index.values.astype('datetime64[ns]')), keys=np.array(keys),
         NAV=np.column_stack([navs[k].reindex(bench.index).to_numpy() for k in keys]), E50=bench.to_numpy())
# 驗收：nav_fix1（標準路徑、0050）、res_fix1 paths（2019 起年化）、s27 monthly_paths.csv.gz（月底值）
ref = pd.read_csv(AG / 'exright_engine_fix/out/nav_fix1.csv.gz', index_col=0, parse_dates=True)
std = navs['100萬|+0']
r_std = (std.reindex(ref.index) / ref['E0|W10_70'] - 1).abs().max()
b_ref = bench.reindex(ref.index); b_ref = b_ref / b_ref.iloc[0] * 1e6
r_b = (b_ref / ref['0050'] - 1).abs().max()
rj = json.load(open(AG / 'exright_engine_fix/out/res_fix1.json'))['strats']['E0']['paths']
cg = {k: stlib.cagr(navs[k], D19, DE) for k in keys}
d_cg = max(abs(cg[k] - rj[k]) for k in keys)
mp = pd.read_csv(AG / 's27_expectation_ranges/monthly_paths.csv.gz', index_col=0)
me = pd.DataFrame({k: navs[k] for k in keys}); me['0050'] = bench
me = me.groupby(me.index.to_period('M')).last(); me.index = me.index.astype(str)
common = mp.index.intersection(me.index)
relm = ((me.loc[common, mp.columns] / mp.loc[common] - 1).abs().max())
out = dict(start=str(D0.date()), end=str(DE.date()), n_days=int(len(bench)), n_ref_days=int(len(ref)), ref_days_same=bool(ref.index.equals(bench.index)),
           std_vs_nav_fix1_max_rel=float(r_std), e50_vs_nav_fix1_max_rel=float(r_b), cagr19_vs_res_fix1_max_abs=float(d_cg),
           monthly_months_mine=int(len(me)), monthly_months_s27=int(len(mp)), monthly_same_months=bool(list(me.index) == list(mp.index)),
           monthly_vs_s27_max_rel_by_col={k: float(v) for k, v in relm.items()}, monthly_vs_s27_max_rel=float(relm.max()),
           cagr19={k: float(v) for k, v in cg.items()}, peak_mb=rss(), secs=round(time.time() - t0))
json.dump(out, open(HERE / 'engine_check.json', 'w'), ensure_ascii=False, indent=1)
print({k: v for k, v in out.items() if k not in ('cagr19', 'monthly_vs_s27_max_rel_by_col')})
