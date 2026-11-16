#!/usr/bin/env python3
"""第 3 步：用某一版「漲跌價差」重跑現行 E0／W10_70（s20lib→s17lib→s10lib.simulate_s10，不改共用程式；
只把 cslib.Panel 讀的面板檔換成本題修正過 CHG 的版本，其餘欄位逐位元相同）。
標準路徑（100 萬、位移 0、嚴成本）＋ 20 條路徑（60／80／100／150／200 萬 × 位移 0～3，同 s19）。
用法：python3 -I -B bt_variant.py <orig|twt|main|close>；結果寫 scratchpad/exr/res_<mode>.json、nav_<mode>.csv（compare.py 合併）。"""
import sys, json, time, resource, gc
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import numpy as np, pandas as pd
from exr_common import TMP, AG, load_twt49u, corrected_chg

MODE = sys.argv[1]
SIGFROM = sys.argv[2] if len(sys.argv) > 2 else None   # 拆解用：訊號沿用另一版（只在標準路徑），報酬用本版
t0 = time.time()
rss = lambda: round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
sys.path.insert(0, str(AG / 's20_persistent_growth'))
import s20lib                                                   # noqa: E402
from s20lib import s17lib, s10lib, s9lib, s8lib, s7lib, s6lib, s5lib, s4lib, s3lib, s2lib, stlib, cslib, COSTS   # noqa: E402
for mod, attr in ((s20lib, 'S20'), (s17lib, 'S17'), (s10lib, 'S10'), (s9lib, 'S9'), (s8lib, 'S8'), (s7lib, 'S7'), (s6lib, 'S6'), (s5lib, 'S5'),
                  (s4lib, 'S4'), (s3lib, 'S3'), (s2lib, 'S2'), (stlib, 'S1'), (cslib, 'TMP')):
    setattr(mod, attr, TMP)

base = stlib.build_panel()                                      # scratchpad/exr/panel_cs.npz（原始）
vdir = TMP / f'v_{MODE}'; vdir.mkdir(exist_ok=True)
vf = vdir / 'panel_cs.npz'
z = dict(np.load(base, allow_pickle=False))
CHG, nchg = corrected_chg(z, load_twt49u(), MODE)
n_oob = getattr(corrected_chg, 'n_oob', 0)
z['CHG'] = CHG
np.savez(vf, **z)
del z, CHG; gc.collect()
cslib.build_panel = lambda force=False: vf                       # cslib.Panel 改讀修正版
P = s20lib.load()
vf.unlink(); vdir.rmdir()
INNOV, _ = s4lib.innov_panel(P)
U = stlib.universe(P, 'twse') & ~INNOV
del INNOV
G = s20lib.growth_features(P)
RT = dict(S=G['S'], av=G['av'])
susp = s17lib.e50_fix_suspension(P)
for a in ('H', 'L', 'VOL', 'CHG', 'HF', 'VOLF', 'adjH', 'HI240', 'MA60', 'nbar', 'F_HI', 'IDXC', 'listed', 'NEARHI', 'V60', 'liq'):
    if hasattr(P, a):
        delattr(P, a)
gc.collect()
print(MODE, '改動格數', nchg, '載入完成', f'{time.time()-t0:.0f}s', rss(), 'MB', flush=True)

lastb = lambda d: P.IDX[P.IDX < pd.Timestamp(d)][-1]
D0, DE, D19, D24 = P.IDX[P.t0], P.IDX[-1], lastb('2019-01-01'), lastb('2024-01-01')
res = dict(mode=MODE, n_chg_cells=nchg, n_ref_fallback_rights_issue=n_oob, end=str(DE.date()), start=str(D0.date()))
paths = {}; paths_dd = {}; lists = {}
for s in range(4):
    S0 = s10lib.make_signal_s10(P, U, RT, shift=s)
    S2, ekw = s10lib.variant_S(S0, 'E0')
    if s == 0 and MODE == 'orig' and SIGFROM is None:
        import pickle
        pickle.dump((S2, ekw), open(TMP / 'sig_orig_s0.pkl', 'wb'), protocol=5)
    if SIGFROM is not None:
        import pickle
        S2, ekw = pickle.load(open(TMP / f'sig_{SIGFROM}_s0.pkl', 'rb'))
        nav, lg, st = s10lib.simulate_s10(P, S2, U, cost=COSTS['strict'], capital=1e6, sat_w=0.7, sizing='W10', lock=False, **ekw)
        r = dict(mode=MODE, sig_from=SIGFROM, cagr_since2019=stlib.cagr(nav, D19, DE), cagr_full=stlib.cagr(nav, D0, DE), maxdd=stlib.maxdd(nav, D0, DE),
                 n_trades_closed=int((lg['why'] != 'open_end').sum()))
        tl = lg[['code', 'ent_date', 'exit_date', 'ret', 'why']].copy(); tl['ent_date'] = tl['ent_date'].astype(str); tl['exit_date'] = tl['exit_date'].astype(str)
        r['trades_list'] = tl.to_dict(orient='records'); r['yearly'] = {int(y): float(v) for y, v in (nav.groupby(nav.index.year).last() / nav.groupby(nav.index.year).last().shift(1).fillna(nav.iloc[0]) - 1).items()}
        json.dump(r, open(TMP / f'res_{MODE}_sig{SIGFROM}.json', 'w'), ensure_ascii=False)
        print({k: v for k, v in r.items() if k != 'trades_list'}); sys.exit(0)
    if s == 0:
        lists = {x['month']: x['targets'] for x in S0['info']}
        res['n_signal_months'] = len(lists)
    for cap in (6e5, 8e5, 1e6, 1.5e6, 2e6):
        nav, lg, st = s10lib.simulate_s10(P, S2, U, cost=COSTS['strict'], capital=cap, sat_w=0.7, sizing='W10', lock=False, **ekw)
        key = f'{int(cap/1e4)}萬|+{s}'
        paths[key] = stlib.cagr(nav, D19, DE); paths_dd[key] = stlib.maxdd(nav, D0, DE)
        if s == 0 and cap == 1e6:
            sd = nav.loc[D0:DE]; dd = sd / sd.cummax() - 1; tr = dd.idxmin(); pk = sd.loc[:tr].idxmax()
            res['std'] = dict(cagr_since2019=stlib.cagr(nav, D19, DE), cagr_full=stlib.cagr(nav, D0, DE), cagr_since2024=stlib.cagr(nav, D24, DE),
                              maxdd=stlib.maxdd(nav, D0, DE), dd_peak=str(pk.date()), dd_trough=str(tr.date()), final_nav=float(nav.iloc[-1]),
                              n_trades_all=int(len(lg)), n_trades_closed=int((lg['why'] != 'open_end').sum()), trades=stlib.trade_stats(lg),
                              open_end=sorted(lg.loc[lg['why'] == 'open_end', 'code'].astype(str).tolist()))
            yr = nav.groupby(nav.index.year).last(); prev = nav.iloc[0]; yy = {}
            for y in yr.index:
                yy[int(y)] = float(yr.loc[y] / prev - 1); prev = yr.loc[y]
            res['std']['yearly'] = yy
            nav.rename('nav').to_csv(TMP / f'nav_{MODE}.csv')
            tl = lg[['code', 'ent_date', 'exit_date', 'ret', 'why']].copy(); tl['ent_date'] = tl['ent_date'].astype(str); tl['exit_date'] = tl['exit_date'].astype(str)
            res['trades_list'] = tl.to_dict(orient='records')
    del S0, S2; gc.collect()
    print(MODE, '位移', s, f'{time.time()-t0:.0f}s', rss(), 'MB', flush=True)
v = np.array(list(paths.values())); d = np.array(list(paths_dd.values()))
res['paths'] = paths; res['paths_dd'] = paths_dd
res['paths_summary'] = dict(cagr19_median=float(np.median(v)), cagr19_min=float(v.min()), cagr19_max=float(v.max()), maxdd_median=float(np.median(d)), maxdd_worst=float(d.min()))
res['lists_f1'] = lists
res['peak_mb'] = rss(); res['secs'] = round(time.time() - t0)
json.dump(res, open(TMP / f'res_{MODE}.json', 'w'), ensure_ascii=False, default=str)
print(MODE, json.dumps(dict(std={k: res['std'][k] for k in ('cagr_since2019', 'cagr_full', 'maxdd', 'n_trades_closed')}, paths=res['paths_summary']), ensure_ascii=False), rss(), 'MB')
