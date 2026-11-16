#!/usr/bin/env python3
"""S36 訓練期：9 個設定 × 20 條路徑，只開訓練月份（2024-06～2025-07）早進場，只看訓練窗（2024-06-28 → 2025-08-29）；依 PREREG 凍結一個。
輸出 results_train.json、FROZEN.json、FROZEN.sha256。PREREG sha256 須相符；FROZEN.json 已存在就拒絕。python3 -I -B train.py"""
import sys, json, hashlib, time, resource, gc, datetime
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
import s36lib
from s36lib import s17lib, s10lib, s4lib, stlib, COSTS

H = Path(__file__).resolve().parent
pre = (H / 'PREREG.json').read_bytes(); sha = hashlib.sha256(pre).hexdigest()
assert sha == (H / 'PREREG.sha256').read_text().split()[0], 'PREREG 不符'
assert not (H / 'FROZEN.json').exists(), 'FROZEN.json 已存在，拒絕重跑'
PRE = json.loads(pre)
assert json.loads(json.dumps(s36lib.RULES)) == PRE['rule_form']['candidates（3 條規則、9 個設定，全部計入 N）'], '規則與 PREREG 不符'
t_all = time.time()
rss = lambda: round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
P = s36lib.load()
assert (P.T, P.N, str(P.IDX[-1].date())) == (2876, 2142, '2026-10-22')
INNOV, _ = s4lib.innov_panel(P)
U = stlib.universe(P, 'twse') & ~INNOV
del INNOV
RT = s36lib.surge(P)
s17lib.e50_fix_suspension(P)
_, seen = s36lib.load_seen(P)
for a in ('H', 'L', 'VOL', 'CHG', 'HF', 'VOLF', 'adjH', 'HI240', 'MA60', 'nbar', 'F_HI', 'IDXC', 'AO', 'listed', 'NEARHI', 'V60', 'liq'):
    if hasattr(P, a):
        delattr(P, a)
gc.collect()
lastb = lambda d: P.IDX[P.IDX < pd.Timestamp(d)][-1]
TA, TB = lastb('2024-07-01'), lastb('2025-09-01')
assert (str(TA.date()), str(TB.date())) == ('2024-06-28', '2025-08-29')
CAPS = (6e5, 8e5, 1e6, 1.5e6, 2e6)
CELL = dict(sat_w=0.7, sizing='W10', lock=False)
MON = s36lib.TRAIN_MONTHS


def sim(S1, cap=1e6, cost='strict'):
    S2, ekw = s10lib.variant_S(S1, 'E0')
    return s10lib.simulate_s10(P, S2, U, cost=COSTS[cost], capital=cap, **CELL, **ekw)


CFG = {}
for r in s36lib.RULES:
    for th in (s36lib.THR_GRID if r['thr'] == 'grid' else ['prev10']):
        CFG[f"{r['id']}|{th}"] = (r, th)

# ---------------------------------------------------------------- 驗收 A2（位移 0、100 萬）
S0 = s10lib.make_signal_s10(P, U, RT, shift=0)
navc, lgc, _ = sim(S0)
acc = {}
cols = ['code', 'ent_t', 'exit_t', 'why']
for nm, (Sx, _a) in {'no_months': s36lib.build_early(P, U, RT, S0, s36lib.RULES[0], 2.0, [], seen),
                     'thr_inf': s36lib.build_early(P, U, RT, S0, s36lib.RULES[0], np.inf, MON + s36lib.VALID_MONTHS, seen)}.items():
    n2, l2, _ = sim(Sx)
    acc[nm] = dict(trades_identical=bool(len(l2) == len(lgc) and (l2[cols].reset_index(drop=True) == lgc[cols].reset_index(drop=True)).all().all()),
                   nav_identical=bool(np.array_equal(n2.to_numpy(), navc.to_numpy())))
assert all(v['trades_identical'] and v['nav_identical'] for v in acc.values()), acc
del navc, lgc
print('驗收 A2', acc, f'{time.time()-t_all:.0f}s', rss(), 'MB', flush=True)

res = {k: {} for k in ['CUR'] + list(CFG)}
nsig = {k: {} for k in CFG}
early_trades = {k: {} for k in CFG}
for s in range(4):
    S = S0 if s == 0 else s10lib.make_signal_s10(P, U, RT, shift=s)
    ev = s36lib.early_events(P, U, RT, seen, MON, s)
    pth = s36lib.prev10_thr(P, U, RT, MON, s)
    built = {}
    for c, (r, th) in CFG.items():
        built[c] = s36lib.build_early(P, U, RT, S, r, pth if th == 'prev10' else th, MON, seen, s, ev)
    for cap in CAPS:
        key = f'{int(cap/1e4)}萬|+{s}'
        nav, lg, st = sim(S, cap)
        res['CUR'][key] = dict(train=stlib.cagr(nav, TA, TB), train_dd=stlib.maxdd(nav, TA, TB), n_buy=int(st['n_buy']))
        base = nav
        for c in CFG:
            Sx, added = built[c]
            nav, lg, st = sim(Sx, cap)
            assert np.array_equal(nav.loc[:TA].to_numpy(), base.loc[:TA].to_numpy()), ('窗前不同', c, key)
            ad = {(int(e), str(code)) for _, code, e in added}
            lgw = lg[lg['ent_date'] > TA]
            ne = int(sum((int(a) - 1, str(b)) in ad for a, b in zip(lgw['ent_t'], lgw['code'])))
            res[c][key] = dict(train=stlib.cagr(nav, TA, TB), train_dd=stlib.maxdd(nav, TA, TB), n_buy=int(st['n_buy']), n_early_filled=ne)
            nsig[c][key] = len(added)
        del base, nav
    del built, ev
    if s:
        del S
    gc.collect()
    print('位移', s, f'{time.time()-t_all:.0f}s', rss(), 'MB', flush=True)

keys = list(res['CUR'])
summ = {}
for c in CFG:
    d = np.array([res[c][k]['train'] - res['CUR'][k]['train'] for k in keys]) * 100
    dd = np.array([res[c][k]['train_dd'] - res['CUR'][k]['train_dd'] for k in keys]) * 100
    summ[c] = dict(score_median_pp=float(np.median(d)), n_ge_025=int((d >= 0.25).sum()), n_pos=int((d > 0).sum()), min_pp=float(d.min()), max_pp=float(d.max()),
                   dd_diff_median_pp=float(np.median(dd)), n_signals_shift0=int(nsig[c][keys[0]]),
                   n_early_filled_median=float(np.median([res[c][k]['n_early_filled'] for k in keys])))
order = sorted(CFG, key=lambda c: (-summ[c]['score_median_pp'], -summ[c]['n_ge_025'], summ[c]['n_signals_shift0']))
best = order[0]
T0 = bool(summ[best]['score_median_pp'] >= 1.0 and summ[best]['n_ge_025'] >= 14)
out = dict(PREREG_sha256=sha, acceptance_A2=acc, windows=dict(train=[str(TA.date()), str(TB.date())]),
           CUR_median=dict(train=float(np.median([res['CUR'][k]['train'] for k in keys])), train_dd=float(np.median([res['CUR'][k]['train_dd'] for k in keys]))),
           summary=summ, ranking=order, frozen=best, T0_pass=T0, paths=res, runtime_s=round(time.time() - t_all, 1), maxrss_mb=rss())
(H / 'results_train.json').write_text(json.dumps(out, ensure_ascii=False, indent=1))
r, th = CFG[best]
FZ = dict(PREREG_sha256=sha, frozen_at=datetime.datetime.now().isoformat(timespec='seconds'), frozen_id=best, frozen_rule=r, frozen_thr=th,
          train_score_median_pp=summ[best]['score_median_pp'], train_n_ge_025=summ[best]['n_ge_025'], T0_pass=T0,
          note=('T0 通過' if T0 else 'T0 不過：驗證期只描述，判定必為不通過'))
b = json.dumps(FZ, ensure_ascii=False, indent=1).encode()
(H / 'FROZEN.json').write_bytes(b)
(H / 'FROZEN.sha256').write_text(hashlib.sha256(b).hexdigest() + '  FROZEN.json\n')
print(json.dumps(summ, ensure_ascii=False, indent=1))
print('凍結', best, 'T0', T0, hashlib.sha256(b).hexdigest())
