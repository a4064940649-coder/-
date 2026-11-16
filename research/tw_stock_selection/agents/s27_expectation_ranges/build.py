#!/usr/bin/env python3
"""S27 第一步：重跑修正後現行 E0｜W10_70 的 20 條路徑（cslib.EXRIGHT_FIX=True），照 exright_engine_fix/run_engine.py 的 E0 做法（不改程式庫）。
輸出：monthly_paths.csv.gz（20 條月底淨值＋0050）、build_meta.json（驗收）；日淨值放暫存區 scratchpad/s27/daily_paths.pkl 供 analyze.py 算日口徑回撤。
python3 -I -B build.py"""
import sys, json, hashlib, time, resource, gc
from pathlib import Path
import numpy as np, pandas as pd

H = Path(__file__).resolve().parent
AG = H.parent
assert hashlib.sha256((H / 'SPEC.json').read_bytes()).hexdigest() == (H / 'SPEC.sha256').read_text().split()[0], 'SPEC.json 與 SPEC.sha256 不符'
TMP = Path('/tmp/claude-0/-home-user--/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/s27')
(TMP / 'cache').mkdir(parents=True, exist_ok=True)
t0 = time.time()
rss = lambda: round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)

sys.path.insert(0, str(AG / 's19_xq_threshold'))
import s19lib                                              # noqa: E402
from s19lib import s10lib, s9lib, s8lib, s7lib, s6lib, s5lib, s4lib, s3lib, s2lib, stlib, cslib, COSTS   # noqa: E402
for mod, attr in ((s19lib, 'S19'), (s10lib, 'S10'), (s9lib, 'S9'), (s8lib, 'S8'), (s7lib, 'S7'), (s6lib, 'S6'), (s5lib, 'S5'),
                  (s4lib, 'S4'), (s3lib, 'S3'), (s2lib, 'S2'), (stlib, 'S1'), (cslib, 'TMP')):
    setattr(mod, attr, TMP / 'cache')
cslib.EXRIGHT_FIX = True

P = s19lib.load()
assert P.exright_fix
s19lib.e50_fix_suspension(P)
INNOV, _ = s4lib.innov_panel(P)
U = stlib.universe(P, 'twse') & ~INNOV
del INNOV
S, _ = s19lib.rev_std(P)
months = list(S.index)
avA = s19lib.a10_rows(P, months)
for a in ('H', 'L', 'VOL', 'CHG', 'HF', 'VOLF', 'adjH', 'HI240', 'MA60', 'nbar', 'F_HI', 'IDXC', 'listed', 'NEARHI', 'V60', 'liq', 'has'):
    if hasattr(P, a):
        delattr(P, a)
gc.collect()
print('載入', f'{time.time()-t0:.0f}s', rss(), 'MB', flush=True)
D0, DE = P.IDX[P.t0], P.IDX[-1]
D19 = P.IDX[P.IDX < pd.Timestamp('2019-01-01')][-1]

navs = {}
for s in range(4):
    Sg = s10lib.make_signal_s10(P, U, dict(S=S, av=avA), shift=s)
    for cap in (6e5, 8e5, 1e6, 1.5e6, 2e6):
        nav = s10lib.simulate_s10(P, Sg, U, sat_w=0.7, sizing='W10', lock=False, cost=COSTS['strict'], capital=cap)[0]
        navs[f'{int(cap/1e4)}萬|+{s}'] = nav.loc[D0:DE]
    del Sg; gc.collect()
    print('位移', s, f'{time.time()-t0:.0f}s', rss(), 'MB', flush=True)

bench = pd.Series(P.E50, index=P.IDX).loc[D0:DE]
daily = pd.DataFrame(navs)
daily['0050'] = bench
daily.index.name = 'date'

# 驗收
meta = dict(spec_sha256=(H / 'SPEC.sha256').read_text().split()[0], start=str(D0.date()), end=str(DE.date()))
ref = pd.read_csv(AG / 'exright_engine_fix' / 'out' / 'nav_fix1.csv.gz', parse_dates=['date']).set_index('date')
a = daily['100萬|+0'].reindex(ref.index)
meta['acc_std_vs_nav_fix1_max_rel'] = float((a / ref['E0|W10_70'] - 1).abs().max())
b = (daily['0050'] / daily['0050'].iloc[0] * 1e6).reindex(ref.index)
meta['acc_0050_vs_nav_fix1_max_rel'] = float((b / ref['0050'] - 1).abs().max())
rj = json.load(open(AG / 'exright_engine_fix' / 'out' / 'res_fix1.json'))['strats']['E0']['paths']
diffs = {k: abs(stlib.cagr(daily[k], D19, DE) - rj[k]) for k in navs}
meta['acc_paths_cagr19_max_absdiff'] = float(max(diffs.values()))
meta['acc_pass'] = bool(meta['acc_std_vs_nav_fix1_max_rel'] < 1e-6 and meta['acc_paths_cagr19_max_absdiff'] < 1e-9)
meta['n_days'] = int(len(daily))
print(meta, flush=True)
assert meta['acc_pass'], '驗收不符，停下來查'

daily.to_pickle(TMP / 'daily_paths.pkl')
mon = daily.groupby(daily.index.to_period('M')).last()
mon.index = mon.index.astype(str); mon.index.name = 'month'
mon.to_csv(H / 'monthly_paths.csv.gz', float_format='%.4f')
meta['peak_mb'] = rss(); meta['secs'] = round(time.time() - t0)
json.dump(meta, open(H / 'build_meta.json', 'w'), ensure_ascii=False, indent=1)
print('done', meta['secs'], 's', meta['peak_mb'], 'MB')
