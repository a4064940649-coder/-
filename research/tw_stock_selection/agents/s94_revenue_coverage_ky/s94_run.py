#!/usr/bin/env python3
"""S94 步驟 3：正式管線（agents/official_nav/run_official.py 的 E0｜W10_70 部分，五個開關全開）逐行照抄，只多一件事：
在載入前把 stlib.REV_CSV（s19lib.rev_std 讀的營收檔路徑）指到暫存區的 rev_<版本>.csv。不改任何既有檔、規則一字不改。
用法：python3 -I -B s94_run.py A|B|C [block.json]      （block.json＝{代號: [[進場列, 出場列], ...]}，去掉指定交易的重跑；輸出到 out/<版本>_blk/）
輸出：out/<版本>/res.json、navs20.csv.gz（20 條路徑逐日淨值）、trades.csv（標準路徑）、elig.csv.gz（每個訊號月「股池內＋季線上＋SURGE 有值」名單）。"""
import sys, json, time, resource, gc
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent
AG = HERE.parent
SC = Path('/tmp/claude-0/-home-user/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/s94')
TMP = SC / 'cache'
V = sys.argv[1]; BLK = sys.argv[2] if len(sys.argv) > 2 else None
OUT = HERE / 'out' / (V + ('_blk' if BLK else '')); OUT.mkdir(parents=True, exist_ok=True)
t0 = time.time(); rss = lambda: round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
sys.path.insert(0, str(AG / 's19_xq_threshold'))
import s19lib                                              # noqa: E402
from s19lib import s10lib, s9lib, s8lib, s7lib, s6lib, s5lib, s4lib, s3lib, s2lib, stlib, cslib, COSTS   # noqa: E402
for mod, attr in ((s19lib, 'S19'), (s10lib, 'S10'), (s9lib, 'S9'), (s8lib, 'S8'), (s7lib, 'S7'), (s6lib, 'S6'), (s5lib, 'S5'),
                  (s4lib, 'S4'), (s3lib, 'S3'), (s2lib, 'S2'), (stlib, 'S1'), (cslib, 'TMP')):
    setattr(mod, attr, TMP)
cslib.EXRIGHT_FIX = True; cslib.DEADLINE_FIX = True; cslib.INS15_MASK = True; cslib.REDUCE_FIX = True; cslib.E50_SOURCE = cslib.E50_OFFICIAL_CSV
stlib.REV_CSV = str(SC / f'rev_{V}.csv')                   # ←←← 唯一的改動：營收來源
P = s19lib.load()
s19lib.e50_fix_suspension(P)
INNOV, _ = s4lib.innov_panel(P)
U = stlib.universe(P, 'twse') & ~INNOV
S, _ = s19lib.rev_std(P)
months = list(S.index)
avA = s19lib.a10_rows(P, months)
D0, DE = P.IDX[P.t0], P.IDX[-1]
D19 = P.IDX[P.IDX < pd.Timestamp('2019-01-01')][-1]
D21 = P.IDX[P.IDX < pd.Timestamp('2022-01-01')][-1]
D15 = P.IDX[P.IDX < pd.Timestamp('2016-01-01')][-1]


def yearly(nav):
    yr = nav.groupby(nav.index.year).last(); prev = nav.iloc[0]; out = {}
    for y in yr.index:
        out[int(y)] = float(yr.loc[y] / prev - 1); prev = yr.loc[y]
    return out


res = dict(version=V, rev_csv=stlib.REV_CSV, start=str(D0.date()), end=str(DE.date()), paths={}, paths_dd={}, paths_full={}, paths_1621={}, paths_2226={})
block = None
if BLK:
    block = {int(np.flatnonzero(np.asarray(P.COLS) == c)[0]): [tuple(x) for x in v] for c, v in json.load(open(BLK)).items()}
    res['block'] = json.load(open(BLK))
navs = {}; el = []
for s in range(4):
    Sg = s10lib.make_signal_s10(P, U, dict(S=S, av=avA), shift=s)
    if s == 0:
        res['n_signal_months'] = len(Sg['info'])
        res['targets'] = {x['month']: x['targets'] for x in Sg['info']}
        res['sig_dates'] = {x['month']: x['sig_date'] for x in Sg['info']}
        res['n_ma'] = {x['month']: x['n_ma'] for x in Sg['info']}
        if not BLK:
            for x in Sg['info']:
                k = x['k']; m = pd.Period(x['month'], 'M')
                sv = cslib.ins15_mask_row(P, m, k, S.loc[m].to_numpy())
                fin = U[k] & np.isfinite(sv)
                res.setdefault('n_fin', {})[x['month']] = int(fin.sum()); res.setdefault('n_univ', {})[x['month']] = int(U[k].sum())
                res.setdefault('n_univ_ma', {})[x['month']] = int((U[k] & P.F_MA60[k]).sum())
                for j in np.flatnonzero(fin):
                    el.append((x['month'], P.COLS[j], round(float(sv[j]), 4), int(P.F_MA60[k, j]), round(float(P.V60[k, j]) / 1e6, 2), round(float(P.RET20[k, j]), 4)))
    for cap in (6e5, 8e5, 1e6, 1.5e6, 2e6):
        nav, lg, st = s10lib.simulate_s10(P, Sg, U, sat_w=0.7, sizing='W10', lock=False, cost=COSTS['strict'], capital=cap, block=block)
        key = f'{int(cap/1e4)}萬|+{s}'
        res['paths'][key] = stlib.cagr(nav, D19, DE); res['paths_dd'][key] = stlib.maxdd(nav, D0, DE); res['paths_full'][key] = stlib.cagr(nav, D0, DE)
        res['paths_1621'][key] = stlib.cagr(nav, D15, D21); res['paths_2226'][key] = stlib.cagr(nav, D21, DE)
        navs[key] = nav.loc[D0:DE]
        if s == 0 and cap == 1e6:
            res['std'] = dict(cagr_since2019=stlib.cagr(nav, D19, DE), cagr_full=stlib.cagr(nav, D0, DE), maxdd=stlib.maxdd(nav, D0, DE), final_nav=float(nav.iloc[-1]),
                              cagr_1621=stlib.cagr(nav, D15, D21), cagr_2226=stlib.cagr(nav, D21, DE),
                              n_trades_closed=int((lg['why'] != 'open_end').sum()), n_trades_all=int(len(lg)), yearly=yearly(nav.loc[D0:DE]))
            lg.to_csv(OUT / 'trades.csv', index=False)
        del nav, lg, st
    del Sg; gc.collect()
v = np.array(list(res['paths'].values())); d = np.array(list(res['paths_dd'].values()))
res['paths_summary'] = dict(cagr19_median=float(np.median(v)), cagr19_min=float(v.min()), cagr19_max=float(v.max()), maxdd_median=float(np.median(d)), maxdd_worst=float(d.min()),
                            full_median=float(np.median(list(res['paths_full'].values()))))
df = pd.DataFrame(navs).round(2); df.index.name = 'date'; df.to_csv(OUT / 'navs20.csv.gz')
if el:
    pd.DataFrame(el, columns=['month', 'code', 'surge', 'ma60', 'v60_m', 'ret20']).to_csv(OUT / 'elig.csv.gz', index=False)
if V == 'A' and not BLK:                                    # 驗收：重現正式數字＋與網站淨值檔逐日差 0
    off = pd.read_csv('/home/user/-/xq_scripts/site/navs_official.csv.gz', index_col=0, parse_dates=True)['E0|W10_70']
    mine = df['100萬|+0']
    res['acceptance'] = dict(same_index=bool(off.index.equals(mine.index)), max_abs_diff=float((off - mine).abs().max()),
                             ok=bool(abs(res['std']['cagr_since2019'] - 0.4886) <= 5e-5 and abs(res['std']['cagr_full'] - 0.3446) <= 5e-5 and res['std']['n_trades_closed'] == 381
                                     and abs(res['paths_summary']['cagr19_median'] - 0.4642) <= 5e-5))
    print('acceptance', res['acceptance'])
res['peak_mb'] = rss(); res['secs'] = round(time.time() - t0)
json.dump(res, open(OUT / 'res.json', 'w'), ensure_ascii=False, default=str, indent=1)
print(V, {k: (round(x, 6) if isinstance(x, float) else x) for k, x in res['std'].items() if k != 'yearly'}, res['paths_summary'], res['secs'], 's', res['peak_mb'], 'MB')
