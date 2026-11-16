#!/usr/bin/env python3
"""除權息參考價修正接進共用引擎（cslib.Panel(exright_fix=...)）後的正式重跑＋換股日兩種定義的比較。
用法：python3 -I -B run_engine.py <fix 0|1> [策略,逗號分隔，預設 E0,EB,XT,XA] [--no-paths]
- fix 0：cslib.EXRIGHT_FIX＝False（預設；應與舊結果逐位相同）；fix 1：True（TWT49U 封存 raw/twt49u/）。
- 策略（全部 W10_70、嚴成本；標準路徑＝100 萬、位移 0；20 條路徑＝60／80／100／150／200 萬 × 位移 0～3，同 s19）：
  E0＝現行（研究版名次、A10 換股日＝10 日（含）後第一個交易日的下一個交易日）
  EB＝E0 但換股日用 B11（第一個日期 ≥ 11 的交易日；＝s19 的 XR）
  XT＝XQ 門檻版（MinSurge 2.5／ExitSurge 2.0、腳本股池）用 B11（＝s19 的 XT，舊腳本）
  XA＝XQ 門檻版用 A10（＝統一換股日後的 20／21／22 號腳本）
程式沿用 s19lib（不改）：面板、SURGE（rev_std）、a10_rows、b11_rows、script_universe、make_signal_thr、s10lib.simulate_s10。
結果：out/res_fix<0|1>.json；fix 1 另寫標準路徑淨值 out/nav_fix1.csv.gz。暫存（面板快取）放 scratchpad/eng/。"""
import sys, json, time, resource, gc
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
AG = HERE.parent
TMP = Path('/tmp/claude-0/-home-user--/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/eng/cache')
TMP.mkdir(parents=True, exist_ok=True)
OUT = HERE / 'out'; OUT.mkdir(exist_ok=True)
FIX = sys.argv[1] == '1'
STRATS = (sys.argv[2] if len(sys.argv) > 2 and not sys.argv[2].startswith('--') else 'E0,EB,XT,XA').split(',')
PATHS = '--no-paths' not in sys.argv
t0 = time.time()
rss = lambda: round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)

sys.path.insert(0, str(AG / 's19_xq_threshold'))
import s19lib                                              # noqa: E402
from s19lib import s10lib, s9lib, s8lib, s7lib, s6lib, s5lib, s4lib, s3lib, s2lib, stlib, cslib, COSTS   # noqa: E402
for mod, attr in ((s19lib, 'S19'), (s10lib, 'S10'), (s9lib, 'S9'), (s8lib, 'S8'), (s7lib, 'S7'), (s6lib, 'S6'), (s5lib, 'S5'),
                  (s4lib, 'S4'), (s3lib, 'S3'), (s2lib, 'S2'), (stlib, 'S1'), (cslib, 'TMP')):
    setattr(mod, attr, TMP)
cslib.EXRIGHT_FIX = FIX

P = s19lib.load()
assert P.exright_fix == FIX
s19lib.e50_fix_suspension(P)
INNOV, _ = s4lib.innov_panel(P)
U = stlib.universe(P, 'twse') & ~INNOV
UX, Cadj, liqX = s19lib.script_universe(P, INNOV)
del Cadj, liqX, INNOV
S, _ = s19lib.rev_std(P)
months = list(S.index)
avA = s19lib.a10_rows(P, months)
avB, _dl = s19lib.b11_rows(P, months, 11)
for a in ('H', 'L', 'VOL', 'CHG', 'HF', 'VOLF', 'adjH', 'HI240', 'MA60', 'nbar', 'F_HI', 'IDXC', 'listed', 'NEARHI', 'V60', 'liq', 'has'):
    if hasattr(P, a):
        delattr(P, a)
gc.collect()
print('fix', FIX, P.exright_info, '載入', f'{time.time()-t0:.0f}s', rss(), 'MB', flush=True)

D0, DE = P.IDX[P.t0], P.IDX[-1]
D19 = P.IDX[P.IDX < pd.Timestamp('2019-01-01')][-1]
D24 = P.IDX[P.IDX < pd.Timestamp('2024-01-01')][-1]


def build(sid, s):
    if sid == 'E0':
        return s10lib.make_signal_s10(P, U, dict(S=S, av=avA), shift=s)
    if sid == 'EB':
        return s10lib.make_signal_s10(P, U, dict(S=S, av=avB), shift=s)
    av = avB if sid == 'XT' else avA
    return s19lib.make_signal_thr(P, U, UX, None, S, av, 2.5, 2.0, shift=s)


def sim(sid, Sg, cap):
    return s10lib.simulate_s10(P, Sg, UX if sid in ('XT', 'XA') else U, sat_w=0.7, sizing='W10', lock=False, cost=COSTS['strict'], capital=cap)


def yearly(nav):
    yr = nav.groupby(nav.index.year).last(); prev = nav.iloc[0]; out = {}
    for y in yr.index:
        out[int(y)] = float(yr.loc[y] / prev - 1); prev = yr.loc[y]
    return out


res = dict(exright_fix=FIX, exright_info=P.exright_info, start=str(D0.date()), end=str(DE.date()), T=int(P.T), N=int(P.N), strats={})
navs = {}
for sid in STRATS:
    R = dict(paths={}, paths_dd={})
    for s in (range(4) if PATHS else (0,)):
        Sg = build(sid, s)
        if s == 0:
            R['n_signal_months'] = len(Sg['info'])
            R['targets'] = {x['month']: x['targets'] for x in Sg['info']}
            R['sig_dates'] = {x['month']: x.get('sig_date', str(P.IDX[x['k']].date())) for x in Sg['info']}
        for cap in ((6e5, 8e5, 1e6, 1.5e6, 2e6) if PATHS else (1e6,)):
            nav, lg, st = sim(sid, Sg, cap)
            key = f'{int(cap/1e4)}萬|+{s}'
            R['paths'][key] = stlib.cagr(nav, D19, DE); R['paths_dd'][key] = stlib.maxdd(nav, D0, DE)
            if s == 0 and cap == 1e6:
                sd = nav.loc[D0:DE]; dd = sd / sd.cummax() - 1; tr = dd.idxmin(); pk = sd.loc[:tr].idxmax()
                R['std'] = dict(cagr_since2019=stlib.cagr(nav, D19, DE), cagr_full=stlib.cagr(nav, D0, DE), cagr_since2024=stlib.cagr(nav, D24, DE),
                                maxdd=stlib.maxdd(nav, D0, DE), dd_peak=str(pk.date()), dd_trough=str(tr.date()), final_nav=float(nav.iloc[-1]),
                                n_trades_closed=int((lg['why'] != 'open_end').sum()), n_trades_all=int(len(lg)), yearly=yearly(nav),
                                open_end=sorted(lg.loc[lg['why'] == 'open_end', 'code'].astype(str).tolist()))
                navs[sid] = nav
            del nav, lg, st
        del Sg; gc.collect()
    if PATHS:
        v = np.array(list(R['paths'].values())); d = np.array(list(R['paths_dd'].values()))
        R['paths_summary'] = dict(cagr19_median=float(np.median(v)), cagr19_min=float(v.min()), cagr19_max=float(v.max()),
                                  maxdd_median=float(np.median(d)), maxdd_worst=float(d.min()))
    res['strats'][sid] = R
    print(sid, {k: (round(v, 6) if isinstance(v, float) else v) for k, v in R['std'].items() if k not in ('yearly', 'open_end')},
          R.get('paths_summary'), f'{time.time()-t0:.0f}s', rss(), 'MB', flush=True)

res['peak_mb'] = rss(); res['secs'] = round(time.time() - t0)
tag = 'fix1' if FIX else 'fix0'
json.dump(res, open(OUT / f'res_{tag}.json', 'w'), ensure_ascii=False, default=str)
bench = pd.Series(P.E50, index=P.IDX).loc[D0:]
df = pd.DataFrame({'0050': bench / bench.iloc[0] * 1e6, **{f'{k}|W10_70': v for k, v in navs.items()}}).round(2)
df.index.name = 'date'
df.to_csv(OUT / f'nav_{tag}.csv.gz')
print('done', res['secs'], 's', res['peak_mb'], 'MB')
