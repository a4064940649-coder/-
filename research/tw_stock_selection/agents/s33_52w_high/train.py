#!/usr/bin/env python3
"""S33 訓練期：4 條候選規則＋1 條對照 × 20 條路徑，只算訓練窗（2016-01-29 → 2021-12-30）；依 PREREG 選一條凍結。
由 S29 train.py 改寫（只換分數為 PTH、規則清單與名稱）。
輸出 results_train.json、FROZEN.json、FROZEN.sha256、code.sha256。PREREG.json 的 sha256 必須相符；FROZEN.json 已存在就拒絕重跑。python3 -I -B train.py"""
import sys, json, hashlib, time, resource, gc, datetime
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
import s33lib
from s33lib import s17lib, s10lib, s4lib, stlib, COSTS

H = Path(__file__).resolve().parent
pre = (H / 'PREREG.json').read_bytes()
sha = hashlib.sha256(pre).hexdigest()
assert sha == (H / 'PREREG.sha256').read_text().split()[0], 'PREREG.json 與 PREREG.sha256 不符'
assert not (H / 'FROZEN.json').exists(), 'FROZEN.json 已存在，拒絕重跑'
PRE = json.loads(pre)
RULES = PRE['rule_form']['candidates（4 條，全部計入 N；參與訓練期選擇）']
CONTROL = PRE['rule_form']['control（1 條，只描述，不參與選擇與判定）']
for r, r0 in zip(RULES + [CONTROL], s33lib.RULES + [s33lib.CONTROL]):
    assert json.loads(json.dumps(r0)) == r, ('規則與 PREREG 不符', r, r0)
MIN_IMP = 0.25
t_all = time.time()
rss = lambda: round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
P = s33lib.load()
assert (P.T, P.N, str(P.IDX[-1].date())) == (2876, 2142, '2026-10-22'), 'P0 失敗'
INNOV, _ = s4lib.innov_panel(P)
U = stlib.universe(P, 'twse') & ~INNOV
del INNOV
RT = s33lib.surge(P)
fix = s17lib.e50_fix_suspension(P)
QM = s33lib.PTHS(P, U)
for a in ('H', 'L', 'VOL', 'CHG', 'HF', 'VOLF', 'adjH', 'HI240', 'MA60', 'nbar', 'F_HI', 'IDXC', 'AO', 'listed', 'NEARHI', 'V60', 'liq'):
    if hasattr(P, a):
        delattr(P, a)
gc.collect()
print('載入完成', f'{time.time()-t_all:.0f}s', rss(), 'MB', flush=True)

bench = pd.Series(P.E50[P.t0:], index=P.IDX[P.t0:])
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


# ---------------------------------------------------------------- 驗收 A0、A2
S0 = s10lib.make_signal_s10(P, U, RT, shift=0)
acc = {'P0': True, 'e50_fix': fix}
navc, lgc, _ = sim(S0)
a0 = dict(since2019=stlib.cagr(navc, D19, P.IDX[-1]), full=stlib.cagr(navc, P.IDX[P.t0], P.IDX[-1]), maxdd=stlib.maxdd(navc, P.IDX[P.t0], P.IDX[-1]), n_trades=int((lgc['why'] != 'open_end').sum()), n_trades_all=int(len(lgc)))   # 381＝已出場筆數（同 exright_engine_fix n_trades_closed）
A = s33lib.ACC
a0['ok'] = bool(abs(a0['since2019'] - A['since2019']) < 0.00025 and abs(a0['full'] - A['full']) < 0.00025 and abs(a0['maxdd'] - A['maxdd']) < 0.00025 and a0['n_trades'] == A['n_trades'])
acc['A0'] = a0
SA, _ = s33lib.build_variant(P, U, RT, S0, QM, dict(id='ALL', kind='filter_rel', score='PTH', frac=0.0))
nava, lga, _ = sim(SA)
cols = ['code', 'ent_t', 'exit_t', 'why']
acc['A2'] = dict(trades_identical=bool(len(lga) == len(lgc) and (lga[cols].reset_index(drop=True) == lgc[cols].reset_index(drop=True)).all().all()),
                 nav_identical=bool(np.allclose(nava.to_numpy(), navc.to_numpy(), rtol=0, atol=1e-6)))
assert a0['ok'] and acc['A2']['trades_identical'] and acc['A2']['nav_identical'], ('驗收失敗', acc)
del navc, lgc, nava, lga, SA
print('驗收', json.dumps({k: v for k, v in acc.items() if k != 'e50_fix'}, ensure_ascii=False), flush=True)

# ---------------------------------------------------------------- 20 條路徑 × 7 個設定（只算訓練窗）
CFG = {'CUR': None}
for r in RULES + [CONTROL]:
    CFG[r['id']] = r
M = {c: {} for c in CFG}
eff = {c: {} for c in CFG if c != 'CUR'}
for s in range(4):
    S = S0 if s == 0 else s10lib.make_signal_s10(P, U, RT, shift=s)
    VAR = {}
    for c, rule in CFG.items():
        S1, lg = s33lib.build_variant(P, U, RT, S, QM, rule)
        VAR[c] = S1
        if c != 'CUR':
            L = [x for x in lg if pd.Period(x['month'], 'M') <= s33lib.TRAIN_LAST_MONTH and pd.Timestamp(P.IDX[x['k']]) > TA]
            eff[c][f'+{s}'] = dict(removed=sum(len(set(x['orig']) - set(x['new'])) for x in L), added=sum(len(set(x['new']) - set(x['orig'])) for x in L),
                                   n_orig=sum(len(x['orig']) for x in L))
    for cap in CAPS:
        key = f'{int(cap/1e4)}萬|+{s}'
        for c in CFG:
            nav, lg, st = sim(VAR[c], cap)
            M[c][key] = tr_metrics(nav)
            del nav, lg, st
    del VAR
    if s:
        del S
    gc.collect()
    print('位移', s, f'{time.time()-t_all:.0f}s', rss(), 'MB', flush=True)

keys = list(M['CUR'])
res = {'PREREG_sha256': sha, 'acceptance': acc, 'window': dict(train=[str(TA.date()), str(TB.date())], y1618=[str(TA.date()), str(T18.date())], y1921=[str(T18.date()), str(TB.date())]),
       'rule_effect_train': eff, 'bench_0050_train': stlib.cagr(bench, TA, TB), 'paths': M, 'summary': {}}
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
        o.update(score_median_pp=float(np.median(d)), n_imp_ge_min=int((d >= MIN_IMP).sum()), npos=int((d > 0).sum()), diff_min_pp=float(d.min()), diff_max_pp=float(d.max()),
                 dd_diff_median_pp=float(np.median(dd)), y1618_median_pp=float(np.median(d1)), y1618_npos=int((d1 > 0).sum()),
                 y1921_median_pp=float(np.median(d2)), y1921_npos=int((d2 > 0).sum()), n_changed_shift0=int(e0['removed'] + e0['added']))
    res['summary'][c] = o
cand = [r['id'] for r in RULES]
best = sorted(cand, key=lambda c: (res['summary'][c]['score_median_pp'], res['summary'][c]['n_imp_ge_min'], -res['summary'][c]['n_changed_shift0']), reverse=True)[0]
bs = res['summary'][best]
t0 = bool(bs['score_median_pp'] >= 1.0 and bs['n_imp_ge_min'] >= 14)
rule = next(r for r in RULES if r['id'] == best)
res['selection'] = dict(order=sorted(cand, key=lambda c: res['summary'][c]['score_median_pp'], reverse=True), frozen=best, T0_pass=t0, min_improvement_pp=MIN_IMP,
                        T0_text='訓練期找到可用的 PTH 規則' if t0 else '訓練期沒有找到（仍凍結分數最高的一條，驗證期只描述）')
res['meta'] = dict(runtime_s=round(time.time() - t_all, 1), maxrss_mb=rss())
(H / 'results_train.json').write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str))
FZ = dict(written_at=datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S'), PREREG_sha256=sha, frozen_rule=rule, T0_pass=t0, T0_text=res['selection']['T0_text'],
          train_score=bs, control=CONTROL, note='凍結後 validate.py 只跑一次；不得換規則或門檻。')
txt = json.dumps(FZ, ensure_ascii=False, indent=1, default=str)
(H / 'FROZEN.json').write_text(txt)
fsha = hashlib.sha256(txt.encode()).hexdigest()
(H / 'FROZEN.sha256').write_text(f'{fsha}  FROZEN.json\n')
(H / 'code.sha256').write_text(''.join(f'{hashlib.sha256((H / p).read_bytes()).hexdigest()}  {p}\n' for p in ('s33lib.py', 'prep.py', 'prereg_write.py', 'train.py')))
for c in CFG:
    print(c, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in res['summary'][c].items()}, flush=True)
print('凍結', best, 'T0', t0, fsha, res['meta'])
