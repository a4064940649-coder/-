#!/usr/bin/env python3
"""軌 B（自寫引擎）：AB 40 seeds（V7、V7_B10；一般／壓力）＋ j＝0～12（自寫 day_for／unavail_for，seeds 0～39）。
輸出 out/navs_b.parquet、out/seed_t_b.json、out/navs_jscan_b.parquet、out/jscan_seed_t_b.json"""
import sys, json, time
from pathlib import Path
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
import numpy as np, pandas as pd
import rb_core as B
from multiprocessing import Pool
OUT = HERE / 'out'
SD = B.official_days(); T0 = B.TPOS[SD[0]]; T1 = B.TPOS[B.END]; SDE = SD + [B.END]

# ---- j 日曆（自寫）：期限 D_m＝第 m 月 10 日起第一個加權指數交易日（2019-02→2/15、2023-04→4/12）；第 j 個交易日 ----
TX = pd.DatetimeIndex(sorted(set(pd.read_csv('/home/user/jonatw/finlab-txd/data/raw/taiex_twii.csv', parse_dates=['date'])['date'])))
GOODD = set(B.IDX[B.GOOD]); GP = np.where(B.GOOD)[0]
SPEC = {'2019-02': pd.Timestamp('2019-02-15'), '2023-04': pd.Timestamp('2023-04-12')}
def ftd(ts):
    k = TX.searchsorted(ts); return TX[k] if k < len(TX) else None
def dline(m): return ftd(SPEC.get(str(m), pd.Timestamp(m.year, m.month, 10)))
def jday(m, j):
    D = dline(m)
    if D is None: return None
    k = TX.get_loc(D) + j
    if k < 0 or k >= len(TX): return None
    s = TX[k]
    if s not in GOODD: return None
    t = B.TPOS[s]
    if np.searchsorted(GP, t, side='right') >= len(GP): return None
    return s
def unav_j(m, s, seed):
    D = dline(m); u = np.zeros(B.N, bool)
    if s < D: u[:] = True
    elif s == D or (str(m) == '2020-02' and s <= pd.Timestamp('2020-02-14')):
        u = np.random.default_rng([int(seed), int(s.strftime('%Y%m%d')), 0]).random(B.N) < B.PCOL
    rm = str(m - 1)
    if '2026-01' <= rm <= '2026-08':
        D15 = ftd(pd.Timestamp(m.year, m.month, 15))
        if s < D15: u |= B.IS15
        elif s == D15: u |= B.IS15 & (np.random.default_rng([int(seed), int(s.strftime('%Y%m%d')), 15]).random(B.N) < B.PCOL)
    return u
ANCH = [pd.Period(d, 'M') for d in SD]

def job_ab(arg):
    k, seed = arg
    pl, nb = B.plan(SD, B10=(k == 'V7_B10'), unavail=B.unavail_mc(seed))
    out = {}
    for c in ('normal', 'stress'):
        nav, to = B.simulate(pl, *B.rates(c), T0, T1); out[c] = nav
    ew, _ = B.simulate(pl, 0.0, 0.0, T0, T1, xr=False, pool=True); out['EW'] = ew
    return k, seed, out, nb, [(str(p['d'].date()), [B.COLS[j] for j in p['picks']]) for p in pl]

def job_j(arg):
    k, j, seed = arg
    days = [(m, jday(m, j)) for m in ANCH]; days = [(m, d) for m, d in days if d is not None]
    de = jday(pd.Period('2026-10', 'M'), j)
    if de is None or de > B.IDX[-1]: de = B.IDX[-1]
    days = [(m, d) for m, d in days if d < de]
    mof = {d: m for m, d in days}; dl = [d for m, d in days]
    U = lambda d: unav_j(mof[d], d, seed)
    pl, nb = B.plan(dl, B10=(k == 'V7_B10'), unavail=U, mof=mof)
    t0, t1 = B.TPOS[dl[0]], B.TPOS[de]
    out = {}
    for c in ('normal', 'stress'):
        nav, _ = B.simulate(pl, *B.rates(c), t0, t1); out[c] = nav
    return k, j, seed, out, [str(d.date()) for d in dl] + [str(de.date())], nb

if __name__ == '__main__':
    t = time.time(); mode = sys.argv[1]
    with Pool(int(sys.argv[2]) if len(sys.argv) > 2 else 2) as P:
        if mode == 'ab':
            res = P.map(job_ab, [(k, s) for k in ('V7', 'V7_B10') for s in range(40)], chunksize=1)
            navs = {}; st = {}; plans = {}; nbl = {}
            for k, s, out, nb, pls in res:
                for c, nav in out.items(): navs[f'AB_{k}|{c}|s{s}'] = nav
                st[f'{k}|s{s}'] = {c: B.seg(out[c], SDE, '2006-04-01', '2026-10-12')['t'] for c in ('normal', 'stress')}
                plans[f'{k}|s{s}'] = pls; nbl[f'{k}|s{s}'] = nb
            pd.DataFrame(navs).to_parquet(OUT / 'navs_b.parquet')
            json.dump(dict(t=st, nblk=nbl, plans=plans), open(OUT / 'ab_b.json', 'w'), ensure_ascii=False)
        else:
            res = P.map(job_j, [(k, j, s) for j in range(13) for k in ('V7', 'V7_B10') for s in range(40)], chunksize=4)
            navs = {}; st = {}; jd = {}; nbl = {}
            for k, j, s, out, dl, nb in res:
                for c, nav in out.items():
                    navs[f'{k}|j{j}|{c}|s{s}'] = nav
                    st[f'{k}|{c}|j{j}|s{s}'] = B.seg(nav, dl, '2006-04-01', '2026-10-12')['t']
                jd[f'j{j}'] = dl; nbl[f'{k}|j{j}|s{s}'] = nb
            pd.DataFrame(navs).to_parquet(OUT / 'navs_jscan_b.parquet')
            json.dump(dict(t=st, jdays=jd, nblk=nbl), open(OUT / 'jscan_b.json', 'w'))
    print('done', mode, time.time() - t)
