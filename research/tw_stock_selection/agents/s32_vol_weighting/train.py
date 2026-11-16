#!/usr/bin/env python3
"""S32 訓練期：4 條候選＋1 條對照 × 20 條路徑，只算訓練窗（2016-01-29 → 2021-12-30，全期連續模擬切出）；依 PREREG 選一條凍結。
輸出 results_train.json、FROZEN.json、FROZEN.sha256。PREREG.json 的 sha256 必須相符；FROZEN.json 已存在就拒絕重跑。python3 -I -B train.py"""
import sys, json, hashlib, time, resource, gc, datetime
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
import s32lib
from s32lib import s17lib, s10lib, s4lib, stlib, COSTS

H = Path(__file__).resolve().parent
pre = (H / 'PREREG.json').read_bytes(); sha = hashlib.sha256(pre).hexdigest()
assert sha == (H / 'PREREG.sha256').read_text().split()[0], 'PREREG.json 與 PREREG.sha256 不符'
assert not (H / 'FROZEN.json').exists(), 'FROZEN.json 已存在，拒絕重跑'
PRE = json.loads(pre)
RULES = PRE['rules（4 條候選，全部計入 N；參與訓練期選擇）']; CONTROL = PRE['control（1 條，只描述，不參與選擇與判定）']
for r, r0 in zip(RULES + [CONTROL], s32lib.RULES + [s32lib.CONTROL]):
    assert r['id'] == r0['id'] and r['q'] == r0['q'], ('規則與 PREREG 不符', r, r0)
t_all = time.time()
rss = lambda: round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
P = s32lib.load()
assert (P.T, P.N, str(P.IDX[-1].date())) == (2876, 2142, '2026-10-22'), 'P0 失敗'
INNOV, _ = s4lib.innov_panel(P)
U = stlib.universe(P, 'twse') & ~INNOV
del INNOV
RT = s32lib.surge(P)
fix = s17lib.e50_fix_suspension(P)
for a in ('H', 'L', 'VOL', 'CHG', 'HF', 'VOLF', 'adjH', 'HI240', 'MA60', 'nbar', 'F_HI', 'IDXC', 'AO', 'listed', 'NEARHI', 'V60', 'liq'):
    if hasattr(P, a):
        delattr(P, a)
gc.collect()
print('載入完成', f'{time.time()-t_all:.0f}s', rss(), 'MB', flush=True)
bench = pd.Series(P.E50[P.t0:], index=P.IDX[P.t0:])
lastb = lambda d: P.IDX[P.IDX < pd.Timestamp(d)][-1]
TA, TB = lastb('2016-02-01'), lastb('2022-01-01')
D19 = lastb('2019-01-01')
CAPS = (6e5, 8e5, 1e6, 1.5e6, 2e6)
CELL = dict(sat_w=0.7, sizing='W10', lock=False)


def sim(S1, cap=1e6, cost='strict'):
    S2, ekw = s10lib.variant_S(S1, 'E0')
    return s32lib.simulate_w(P, S2, U, cost=COSTS[cost], capital=cap, **CELL, **ekw)


def tr_metrics(nav, st):
    mw = st['maxw'].loc[TA:TB]; hh = st['hhi'].loc[TA:TB]; npd = st['nposd'].loc[TA:TB]
    return dict(train=stlib.cagr(nav, TA, TB), train_dd=stlib.maxdd(nav, TA, TB), calmar=s32lib.calmar(nav, TA, TB),
                y1618=stlib.cagr(nav, TA, D19), y1921=stlib.cagr(nav, D19, TB), beta=s32lib.capm_beta(nav, bench, TA, TB)['beta'],
                maxw_max=float(mw.max()), maxw_mean=float(mw[npd > 0].mean()), effn_mean=float((1.0 / hh[npd > 0]).mean()))


# ---------------------------------------------------------------- 驗收 A0、A1（位移 0）
S0 = s10lib.make_signal_s10(P, U, RT, shift=0)
S2, ekw = s10lib.variant_S(S0, 'E0')
navc, lgc, _ = s10lib.simulate_s10(P, S2, U, cost=COSTS['strict'], capital=1e6, **CELL, **ekw)
a0 = dict(since2019=stlib.cagr(navc, D19, P.IDX[-1]), full=stlib.cagr(navc, P.IDX[P.t0], P.IDX[-1]), maxdd=stlib.maxdd(navc, P.IDX[P.t0], P.IDX[-1]),
          n_closed=int((lgc['why'] != 'open_end').sum()))
a0['ok'] = bool(abs(a0['since2019'] - 0.4899) < 0.00025 and abs(a0['full'] - 0.3454) < 0.00025 and abs(a0['maxdd'] + 0.3211) < 0.00025 and a0['n_closed'] == 381)
navw, lgw, _ = sim(s32lib.build_wmul(P, S0, None)[0])
acc = dict(exright_fix=bool(P.exright_fix), e50_fix=fix, A0=a0, A1=bool(np.array_equal(navc.to_numpy(), navw.to_numpy()) and len(lgc) == len(lgw)))
assert a0['ok'] and acc['A1'], acc
del navc, lgc, navw, lgw
print('驗收通過', flush=True)

CFG = {'CUR': None}
for r in RULES + [CONTROL]:
    CFG[r['id']] = r
M = {c: {} for c in CFG}
WD = {c: {} for c in CFG if c != 'CUR'}
for s in range(4):
    S = S0 if s == 0 else s10lib.make_signal_s10(P, U, RT, shift=s)
    VAR = {}
    for c, rule in CFG.items():
        S1, lg = s32lib.build_wmul(P, S, rule)
        VAR[c] = S1
        if c != 'CUR':
            L = [x for x in lg if pd.Period(x['month'], 'M') <= s32lib.TRAIN_LAST_MONTH]
            m = np.concatenate([x['m'] for x in L])
            WD[c][f'+{s}'] = dict(m_min=float(m.min()), m_max=float(m.max()), m_median=float(np.median(m)), sd=float(m.std()))
    for cap in CAPS:
        key = f'{int(cap/1e4)}萬|+{s}'
        for c in CFG:
            nav, lg, st = sim(VAR[c], cap)
            M[c][key] = tr_metrics(nav, st)
            if cap == 1e6 and s == 0:
                x = lg[(lg['ent_date'] > TA) & (lg['ent_date'] <= TB)]
                M[c][key]['n_trades'] = int(len(x))
            del nav, lg, st
    del VAR
    if s:
        del S
    gc.collect()
    print('位移', s, f'{time.time()-t_all:.0f}s', rss(), 'MB', flush=True)

keys = list(M['CUR'])
med = lambda v: float(np.median(v))
res = {'PREREG_sha256': sha, 'acceptance': acc, 'window': dict(train=[str(TA.date()), str(TB.date())]), 'weights_train': WD,
       'bench_0050_train': dict(cagr=stlib.cagr(bench, TA, TB), maxdd=stlib.maxdd(bench, TA, TB)), 'paths': M, 'summary': {}}
for c in CFG:
    o = {f + '_median': med([M[c][k][f] for k in keys]) for f in ('train', 'train_dd', 'calmar', 'beta', 'maxw_max', 'maxw_mean', 'effn_mean')}
    if c != 'CUR':
        d = np.array([M[c][k]['train'] - M['CUR'][k]['train'] for k in keys]) * 100
        dd = np.array([M[c][k]['train_dd'] - M['CUR'][k]['train_dd'] for k in keys]) * 100
        dc = np.array([M[c][k]['calmar'] - M['CUR'][k]['calmar'] for k in keys])
        imp = (dc >= 0.05) | (dd >= 0.5)
        d1 = np.array([M[c][k]['y1618'] - M['CUR'][k]['y1618'] for k in keys]) * 100
        d2 = np.array([M[c][k]['y1921'] - M['CUR'][k]['y1921'] for k in keys]) * 100
        o.update(n_imp=int(imp.sum()), n_calmar_imp=int((dc >= 0.05).sum()), n_dd_imp=int((dd >= 0.5).sum()),
                 ann_diff_median_pp=med(d), ann_npos=int((d > 0).sum()), ann_diff_min_pp=float(d.min()), ann_diff_max_pp=float(d.max()),
                 dd_diff_median_pp=med(dd), calmar_diff_median=med(dc), y1618_median_pp=med(d1), y1921_median_pp=med(d2))
    res['summary'][c] = o
cand = [r['id'] for r in RULES]
sm = res['summary']
order = sorted(cand, key=lambda c: (sm[c]['ann_diff_median_pp'] >= -1.0, sm[c]['n_imp'], sm[c]['calmar_diff_median'], sm[c]['ann_diff_median_pp']), reverse=True)
best = order[0]; bs = sm[best]
t0 = bool(bs['n_imp'] >= 14 and bs['ann_diff_median_pp'] >= -1.0)
rule = next(r for r in RULES if r['id'] == best)
res['selection'] = dict(order=order, frozen=best, T0_pass=t0, T0_text='訓練期找到可用的波動度權重' if t0 else '訓練期沒有找到（仍凍結排序第一的一條，驗證期只描述、判定必為不通過）')
res['meta'] = dict(runtime_s=round(time.time() - t_all, 1), maxrss_mb=rss())
(H / 'results_train.json').write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str))
FZ = dict(written_at=datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S'), PREREG_sha256=sha, frozen_rule=rule, T0_pass=t0, T0_text=res['selection']['T0_text'],
          train_score=bs, control=CONTROL, note='凍結後 validate.py 只跑一次；不得換規則或上下限。')
txt = json.dumps(FZ, ensure_ascii=False, indent=1, default=str)
(H / 'FROZEN.json').write_text(txt)
fsha = hashlib.sha256(txt.encode()).hexdigest()
(H / 'FROZEN.sha256').write_text(f'{fsha}  FROZEN.json\n')
for c in CFG:
    print(c, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in sm[c].items()}, flush=True)
print('凍結', best, 'T0', t0, fsha, res['meta'])
