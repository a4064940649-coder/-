#!/usr/bin/env python3
"""S34 訓練期：4 條候選規則 × 20 條路徑，只算訓練窗（2016-01-29 → 2021-12-30）；依 PREREG 選一條凍結。
輸出 results_train.json、FROZEN.json、FROZEN.sha256、code.sha256。PREREG.json 的 sha256 必須相符；FROZEN.json 已存在就拒絕重跑。python3 -I -B train.py"""
import sys, json, hashlib, time, resource, gc, datetime
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
import s34lib
from s34lib import s17lib, s10lib, s4lib, stlib, COSTS

H = Path(__file__).resolve().parent
pre = (H / 'PREREG.json').read_bytes()
sha = hashlib.sha256(pre).hexdigest()
assert sha == (H / 'PREREG.sha256').read_text().split()[0], 'PREREG.json 與 PREREG.sha256 不符'
assert not (H / 'FROZEN.json').exists(), 'FROZEN.json 已存在，拒絕重跑'
PRE = json.loads(pre)
RULES = PRE['rule_form']['candidates（4 條，全部計入 N；參與訓練期選擇）']
assert RULES == json.loads(json.dumps(s34lib.RULES, ensure_ascii=False)), '規則與 PREREG 不符'
t_all = time.time()
rss = lambda: round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
P = s34lib.load()
assert (P.T, P.N, str(P.IDX[-1].date())) == (2876, 2142, '2026-10-22'), 'P0 失敗'
INNOV, _ = s4lib.innov_panel(P)
U = stlib.universe(P, 'twse') & ~INNOV
del INNOV
TD = s34lib.trading_days(); Rv = s34lib.revenue(P)
RT = s34lib.surge_rule(P, Rv, 'none', TD)
fix = s17lib.e50_fix_suspension(P)
print('載入完成', f'{time.time()-t_all:.0f}s', rss(), 'MB', flush=True)

lastb = lambda d: P.IDX[P.IDX < pd.Timestamp(d)][-1]
TA, TB = lastb('2016-02-01'), lastb('2022-01-01')
T18 = lastb('2019-01-01'); D19 = T18
CAPS = (6e5, 8e5, 1e6, 1.5e6, 2e6)
CELL = dict(sat_w=0.7, sizing='W10', lock=False)


def sim(S1, cap=1e6, cost='strict'):
    S2, ekw = s10lib.variant_S(S1, 'E0')
    return s10lib.simulate_s10(P, S2, U, cost=COSTS[cost], capital=cap, **CELL, **ekw)


def tr_metrics(nav):
    return dict(train=stlib.cagr(nav, TA, TB), train_dd=stlib.maxdd(nav, TA, TB), y1618=stlib.cagr(nav, TA, T18), y1921=stlib.cagr(nav, T18, TB))


# ---------------------------------------------------------------- 驗收 A0、A1
S0 = s10lib.make_signal_s10(P, U, RT, shift=0)
acc = {'P0': True, 'exright_fix': P.exright_fix, 'e50_fix': fix}
navc, lgc, _ = sim(S0)
a0 = dict(since2019=stlib.cagr(navc, D19, P.IDX[-1]), full=stlib.cagr(navc, P.IDX[P.t0], P.IDX[-1]), maxdd=stlib.maxdd(navc, P.IDX[P.t0], P.IDX[-1]),
          n_closed=int((lgc['why'] != 'open_end').sum()))
a0['ok'] = bool(abs(a0['since2019'] - s34lib.ACC['since2019']) < 0.00025 and abs(a0['full'] - s34lib.ACC['full']) < 0.00025
                and abs(a0['maxdd'] - s34lib.ACC['maxdd']) < 0.00025 and a0['n_closed'] == s34lib.ACC['n_closed'])
acc['A0'] = a0
SA, _ = s34lib.build(P, U, Rv, TD, dict(id='ID', kind='surge', mode='none'), 0)
nava, lga, _ = sim(SA)
cols = ['code', 'ent_t', 'exit_t', 'why']
acc['A1'] = dict(trades_identical=bool(len(lga) == len(lgc) and (lga[cols].reset_index(drop=True) == lgc[cols].reset_index(drop=True)).all().all()),
                 nav_identical=bool(np.allclose(nava.to_numpy(), navc.to_numpy(), rtol=0, atol=1e-6)))
assert a0['ok'] and acc['A1']['trades_identical'] and acc['A1']['nav_identical'], ('驗收失敗', acc)
del navc, lgc, nava, lga, SA
print('驗收', json.dumps({k: v for k, v in acc.items() if k != 'e50_fix'}, ensure_ascii=False), flush=True)

# ---------------------------------------------------------------- 20 條路徑 × 5 個設定（只算訓練窗）
CFG = {'CUR': None}
for r in RULES:
    CFG[r['id']] = r
M = {c: {} for c in CFG}
eff = {c: {} for c in CFG if c != 'CUR'}
for s in range(4):
    S = S0 if s == 0 else s10lib.make_signal_s10(P, U, RT, shift=s)
    VAR = {}
    for c, rule in CFG.items():
        S1, _ = s34lib.build(P, U, Rv, TD, rule, s, S_cur=S, RT_cur=RT)
        VAR[c] = S1
        if c != 'CUR':
            a = {x['month']: set(np.flatnonzero(S['sig'][x['k']])) for x in S['info'] if pd.Period(x['month'], 'M') <= s34lib.TRAIN_LAST_MONTH and P.IDX[x['k']] > TA}
            b = {x['month']: set(np.flatnonzero(S1['sig'][x['k']])) for x in S1['info']}
            eff[c][f'+{s}'] = dict(removed=sum(len(a[m] - b.get(m, set())) for m in a), added=sum(len(b.get(m, set()) - a[m]) for m in a),
                                   n_orig=sum(len(a[m]) for m in a), months_changed=sum(1 for m in a if a[m] != b.get(m, set())))
    for cap in CAPS:
        key = f'{int(cap/1e4)}萬|+{s}'
        for c in CFG:
            nav, lg, st = sim(VAR[c], cap)
            M[c][key] = tr_metrics(nav)
    del VAR
    gc.collect()
    print('位移', s, f'{time.time()-t_all:.0f}s', rss(), 'MB', flush=True)

keys = list(M['CUR'])
res = {'PREREG_sha256': sha, 'acceptance': acc, 'window': dict(train=[str(TA.date()), str(TB.date())], y1618=[str(TA.date()), str(T18.date())], y1921=[str(T18.date()), str(TB.date())]),
       'rule_effect_train': eff, 'paths': M, 'summary': {}}
bench = pd.Series(P.E50[P.t0:], index=P.IDX[P.t0:])
res['bench_0050_train'] = stlib.cagr(bench, TA, TB)
for c in CFG:
    v = np.array([M[c][k]['train'] for k in keys])
    o = dict(train_median=float(np.median(v)), train_min=float(v.min()), train_max=float(v.max()),
             train_dd_median=float(np.median([M[c][k]['train_dd'] for k in keys])))
    if c != 'CUR':
        d = np.array([M[c][k]['train'] - M['CUR'][k]['train'] for k in keys]) * 100
        dd = np.array([M[c][k]['train_dd'] - M['CUR'][k]['train_dd'] for k in keys]) * 100
        d1 = np.array([M[c][k]['y1618'] - M['CUR'][k]['y1618'] for k in keys]) * 100
        d2 = np.array([M[c][k]['y1921'] - M['CUR'][k]['y1921'] for k in keys]) * 100
        e0 = eff[c]['+0']
        o.update(score_median_pp=float(np.median(d)), npos=int((d > 0).sum()), n_ge025=int((d >= 0.25).sum()), diff_min_pp=float(d.min()), diff_max_pp=float(d.max()),
                 dd_diff_median_pp=float(np.median(dd)), y1618_median_pp=float(np.median(d1)), y1618_npos=int((d1 > 0).sum()),
                 y1921_median_pp=float(np.median(d2)), y1921_npos=int((d2 > 0).sum()), n_changed_shift0=int(e0['removed'] + e0['added']))
    res['summary'][c] = o
cand = [r['id'] for r in RULES]
best = sorted(cand, key=lambda c: (res['summary'][c]['score_median_pp'], res['summary'][c]['n_ge025'], -res['summary'][c]['n_changed_shift0']), reverse=True)[0]
bs = res['summary'][best]
t0 = bool(bs['score_median_pp'] >= 1.0 and bs['n_ge025'] >= 14)
rule = next(r for r in RULES if r['id'] == best)
res['selection'] = dict(order=sorted(cand, key=lambda c: res['summary'][c]['score_median_pp'], reverse=True), frozen=best, T0_pass=t0,
                        T0_text='訓練期找到可用的春節校正規則' if t0 else '訓練期沒有找到（仍凍結分數最高的一條，驗證期只描述）')
res['meta'] = dict(runtime_s=round(time.time() - t_all, 1), maxrss_mb=rss())
(H / 'results_train.json').write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str))
FZ = dict(written_at=datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S'), PREREG_sha256=sha, frozen_rule=rule, T0_pass=t0, T0_text=res['selection']['T0_text'],
          train_score=bs, note='凍結後 validate.py 只跑一次；不得換規則或門檻。')
txt = json.dumps(FZ, ensure_ascii=False, indent=1, default=str)
(H / 'FROZEN.json').write_text(txt)
fsha = hashlib.sha256(txt.encode()).hexdigest()
(H / 'FROZEN.sha256').write_text(f'{fsha}  FROZEN.json\n')
code = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(H.glob('*.py'))}
(H / 'code.sha256').write_text(''.join(f'{v}  {k}\n' for k, v in code.items()))
for c in CFG:
    print(c, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in res['summary'][c].items()}, flush=True)
print('凍結', best, 'T0', t0, fsha, res['meta'])
