#!/usr/bin/env python3
"""S38 主程式：驗收 A0／A1 → 16 格 × 20 條路徑 → 中位數、成對差、選擇表 → results.json。
python3 -I -B run.py（先確認 SPEC.json 與 SPEC.sha256 相符）"""
import sys, json, time, resource, gc, hashlib
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
import s38lib
from s38lib import s17lib, s10lib, s4lib, stlib

H = Path(__file__).resolve().parent
SHA = hashlib.sha256((H / 'SPEC.json').read_bytes()).hexdigest()
assert SHA == (H / 'SPEC.sha256').read_text().split()[0], 'SPEC.json 與 SPEC.sha256 不符'
t_all = time.time()
rss = lambda: round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
P = s38lib.load()
INNOV, _ = s4lib.innov_panel(P)
U = stlib.universe(P, 'twse') & ~INNOV
RT = s38lib.surge(P)
s17lib.e50_fix_suspension(P)
del INNOV; gc.collect()
print('載入', f'{time.time()-t_all:.0f}s', rss(), 'MB', flush=True)
D0, DE = P.IDX[P.t0], P.IDX[-1]
D19 = P.IDX[P.IDX < pd.Timestamp('2019-01-01')][-1]
D16 = P.IDX[P.IDX < pd.Timestamp('2016-02-01')][-1]
bench = pd.Series(P.E50, index=P.IDX)
cg, md = stlib.cagr, stlib.maxdd
out = {'SPEC_sha256': SHA, 'nature': '描述性；風險偏好選擇，不是策略改進；現行預設不改；不增加 N',
       'period': dict(full_from=str(D0.date()), since2019_from=str(D19.date()), end=str(DE.date()))}

# ---------------------------------------------------------------- 驗收
S0 = s10lib.make_signal_s10(P, U, RT, shift=0)
nav, lg, _ = s38lib.sim(P, S0, U, 0.7)
a0 = dict(since2019=cg(nav, D19, DE), full=cg(nav, D0, DE), maxdd=md(nav, D0, DE), n_trades=int((lg['why'] != 'open_end').sum()))
a0['ok'] = bool(abs(a0['since2019'] - 0.4899) < 5e-5 and abs(a0['full'] - 0.3454) < 5e-5 and abs(a0['maxdd'] + 0.3211) < 5e-5 and a0['n_trades'] == 381)
Bc = dict(E50=P.E50.copy(), E50O=P.E50O.copy(), E50O_raw=P.E50O_raw.copy(), E50C_raw=P.E50C_raw.copy(), sb=None, tax=None)
nav2, lg2, _ = s38lib.sim(P, S0, U, 0.7, B=Bc)
cols = ['code', 'ent_t', 'exit_t', 'why', 'cost', 'proceeds']
a1 = dict(copy_nav_bitwise=bool(np.array_equal(nav.to_numpy(), nav2.to_numpy())), copy_trades=bool(lg[cols].equals(lg2[cols])),
          restored=bool(stlib.etf_tick(60.0) == 0.05 and s10lib.TAX_ETF == 0.001 and P.E50 is not Bc['E50']))
out['acceptance'] = dict(A0=a0, A1=a1)
print('驗收', out['acceptance'], flush=True)
assert a0['ok'] and all(a1.values()), '驗收不符，停下來查'
del nav2, lg2, Bc, nav, lg; gc.collect()

# ---------------------------------------------------------------- 網格
SATS = (0.5, 0.6, 0.7, 0.8); WS = (1.0, 0.75, 0.5, 0.25)
CAPS = (6e5, 8e5, 1e6, 1.5e6, 2e6)
cell = lambda s, w: f'衛星{int(s*100)}｜0050占{int(w*100)}'
CUR = cell(0.7, 1.0)
BASE = {w: s38lib.base_arrays(P, w) for w in WS}
ref = {'0050': dict(cagr_2016=cg(bench, D16, DE), cagr_2019=cg(bench, D19, DE), maxdd_2016=md(bench, D16, DE), seg=s17lib.seg_dd(bench))}
for w in WS[1:]:
    s = pd.Series(BASE[w]['E50'], index=P.IDX)
    ref[f'混合{int(w*100)}'] = dict(cagr_2016=cg(s, D16, DE), cagr_2019=cg(s, D19, DE), maxdd_2016=md(s, D16, DE), beta=s38lib.capm_beta(s, bench, D16, DE), seg=s17lib.seg_dd(s))
out['reference_indices'] = ref
PATH = {cell(s, w): {} for s in SATS for w in WS}
for sh in range(4):
    S = S0 if sh == 0 else s10lib.make_signal_s10(P, U, RT, shift=sh)
    for cap in CAPS:
        key = f'{int(cap/1e4)}萬|+{sh}'
        for s in SATS:
            for w in WS:
                nav, lg, st = s38lib.sim(P, S, U, s, B=BASE[w], cap=cap)
                sg = s17lib.seg_dd(nav)
                full = cg(nav, D0, DE); dd = md(nav, D0, DE)
                PATH[cell(s, w)][key] = dict(cagr_2019=cg(nav, D19, DE), cagr_full=full, maxdd=dd, sharpe=stlib.sharpe(nav, D0, DE),
                                             sharpe_rf1=s38lib.sharpe_rf(nav, D0, DE), calmar=full / abs(dd), beta=s38lib.capm_beta(nav, bench, D0, DE),
                                             roll12_worst=s38lib.roll12_worst(nav, bench), seg2022=sg['2022'], seg2425=sg['2024-07~2025-04'],
                                             n_exit=int((lg['why'] != 'open_end').sum()), stockfrac=float(np.nanmean(st['stockfrac'].to_numpy())))
                del nav, lg, st
    if sh:
        del S
    gc.collect()
    print('位移', sh, f'{time.time()-t_all:.0f}s', rss(), 'MB', flush=True)

# ---------------------------------------------------------------- 彙整
keys = list(PATH[CUR]); F = list(PATH[CUR][keys[0]])
med, dif = {}, {}
for c in PATH:
    med[c] = {f: float(np.median([PATH[c][k][f] for k in keys])) for f in F}
    med[c]['maxdd_worst_path'] = float(min(PATH[c][k]['maxdd'] for k in keys))
    d = {f: np.array([PATH[c][k][f] - PATH[CUR][k][f] for k in keys]) for f in F if f not in ('n_exit',)}
    dif[c] = {f: dict(median=float(np.median(v)), p10=float(np.percentile(v, 10)), npos=int((v > 0).sum())) for f, v in d.items()}
out['median'] = med; out['diff_vs_CUR'] = dif


def menu(crit):
    rows = {}
    for th in (-0.20, -0.25, -0.30, -0.35):
        ok = [c for c in PATH if med[c][crit] >= th - 1e-12]
        if not ok:
            rows[f'{int(th*100)}%'] = None; continue
        best = max(ok, key=lambda c: (round(med[c]['cagr_full'], 10), med[c]['sharpe']))
        m = med[best]; d = dif[best]
        rows[f'{int(th*100)}%'] = dict(cell=best, n_ok=len(ok), cagr_2019=m['cagr_2019'], cagr_full=m['cagr_full'], maxdd=m['maxdd'], maxdd_worst=m['maxdd_worst_path'],
                                       sharpe=m['sharpe'], calmar=m['calmar'], beta=m['beta'], roll12_worst=m['roll12_worst'], seg2022=m['seg2022'], seg2425=m['seg2425'],
                                       cost_full_pp_median=d['cagr_full']['median'] * 100, cost_full_pp_p10=d['cagr_full']['p10'] * 100,
                                       cost_2019_pp_median=d['cagr_2019']['median'] * 100, cost_2019_pp_p10=d['cagr_2019']['p10'] * 100)
    return rows


out['menu_by_median_dd'] = menu('maxdd')
out['menu_by_worst_path_dd'] = menu('maxdd_worst_path')
eff = {}
for c in PATH:
    dd_gain = (med[c]['maxdd'] - med[CUR]['maxdd']) * 100
    if dd_gain > 0:
        eff[c] = dict(dd_shallower_pp=dd_gain, cagr_cost_pp=(med[CUR]['cagr_full'] - med[c]['cagr_full']) * 100,
                      cost_per_pp=(med[CUR]['cagr_full'] - med[c]['cagr_full']) * 100 / dd_gain, sharpe=med[c]['sharpe'], sharpe_rf1=med[c]['sharpe_rf1'])
out['efficiency'] = eff
out['paths'] = PATH
out['meta'] = dict(runtime_s=round(time.time() - t_all, 1), maxrss_mb=rss(), n_sims=len(PATH) * len(keys) + 2)
(H / 'results.json').write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str))
pc = lambda x: round(100 * x, 2)
for c in PATH:
    m = med[c]
    print(c, '19起', pc(m['cagr_2019']), '全期', pc(m['cagr_full']), 'DD', pc(m['maxdd']), '最差', pc(m['maxdd_worst_path']), 'Sh', round(m['sharpe'], 3), round(m['sharpe_rf1'], 3),
          'Cal', round(m['calmar'], 2), 'β', round(m['beta'], 2), 'R12', pc(m['roll12_worst']), '22', pc(m['seg2022']), '2425', pc(m['seg2425']),
          '差全期', round(dif[c]['cagr_full']['median'] * 100, 2), round(dif[c]['cagr_full']['p10'] * 100, 2), 'stk', pc(m['stockfrac']))
print(json.dumps(out['menu_by_median_dd'], ensure_ascii=False, indent=0))
print(json.dumps(out['menu_by_worst_path_dd'], ensure_ascii=False, indent=0))
print(out['meta'])
