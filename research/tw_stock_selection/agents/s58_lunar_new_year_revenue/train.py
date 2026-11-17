#!/usr/bin/env python3
"""S58 驗收＋訓練期：四個設定 × 20 條路徑，只算訓練窗（2016-01-29 → 2021-12-30，全期連續模擬切出）；依 PREREG 選一個凍結。
輸出 results_train.json、FROZEN.json、FROZEN.sha256、code.sha256。SPEC／PREREG 的 sha256 必須相符；FROZEN.json 已存在就拒絕重跑。python3 -I -B train.py"""
import sys, json, hashlib, time, resource, gc, datetime
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
import s58lib as L
from s58lib import stlib, s17lib

H = Path(__file__).resolve().parent
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
for f in ('SPEC', 'PREREG'):
    assert sha(H / f'{f}.json') == (H / f'{f}.sha256').read_text().split()[0], f'{f}.json 與登記的 sha256 不符'
assert not (H / 'FROZEN.json').exists(), 'FROZEN.json 已存在，拒絕重跑'
t0 = time.time()
rss = lambda: round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
P, U, RT, Z = L.setup()
print('載入', f'{time.time()-t0:.0f}s', rss(), 'MB', flush=True)
# A0：重建 STD
Zm = Z['Zfull'].copy(); Zm[Zm.index < s17lib.FIRST_MONTH] = np.nan
A0 = bool(Zm.index.equals(RT['S'].index) and np.array_equal(Zm.to_numpy(), RT['S'].to_numpy(), equal_nan=True))
assert A0, '驗收 A0 失敗'
lastb = lambda d: P.IDX[P.IDX < pd.Timestamp(d)][-1]
TA, TB, T18, D19, DE = lastb('2016-02-01'), lastb('2022-01-01'), lastb('2019-01-01'), lastb('2019-01-01'), P.IDX[-1]
ref = json.loads((L.AG / 'deadline_engine_fix' / 'out' / 'results.json').read_text())


def scope_ok(SS):
    """A1：營收月份 1、2 月以外的列，候選與現行的 info／sig／X 逐格相同。"""
    cur = SS['CUR']; inf0 = {x['month']: x for x in cur['info']}
    ok = {}
    for c in ('C1', 'C2', 'C3'):
        s1 = SS[c]; inf1 = {x['month']: x for x in s1['info']}
        good = True
        for m, x in inf0.items():
            if pd.Period(m, 'M').month in (1, 2):
                continue
            y = inf1.get(m)
            k = x['k']
            if y is None or y['targets'] != x['targets'] or y['k'] != k or not np.array_equal(s1['sig'][k], cur['sig'][k]) or not np.array_equal(s1['X'][k], cur['X'][k]):
                good = False; break
        ok[c] = good
    return ok


M = {c: {} for c in L.CANDS}
A1 = {}; acc_paths = {}; main = None
for s in range(4):
    SS = L.signals(P, U, RT, Z, s)
    A1[s] = scope_ok(SS)
    assert all(A1[s].values()), f'驗收 A1 失敗（位移 {s}）'
    for cap in L.CAPS:
        key = f'{int(cap/1e4)}萬|+{s}'
        for c in L.CANDS:
            nav, lg, st = L.sim(P, U, SS[c], cap)
            M[c][key] = dict(train=stlib.cagr(nav, TA, TB), train_dd=stlib.maxdd(nav, TA, TB), y1618=stlib.cagr(nav, TA, T18), y1921=stlib.cagr(nav, T18, TB),
                             n_closed_train=int(((lg['why'] != 'open_end') & (lg['exit_date'] <= TB)).sum()))
            if c == 'CUR':
                acc_paths[key] = stlib.cagr(nav, D19, DE)
                if s == 0 and cap == 1e6:
                    main = dict(since2019=acc_paths[key], full=stlib.cagr(nav, P.IDX[P.t0], DE), maxdd_full=stlib.maxdd(nav, P.IDX[P.t0], DE),
                                n_closed=int((lg['why'] != 'open_end').sum()))
            del nav, lg, st
    del SS; gc.collect()
    print('位移', s, f'{time.time()-t0:.0f}s', rss(), 'MB', flush=True)

R0 = ref['standard']['ALL']
acc = dict(A0_std_rebuild=A0, A1_scope=A1, main=main,
           A2_main_bitwise=bool(main['since2019'] == R0['since2019'] and main['full'] == R0['full'] and main['maxdd_full'] == R0['maxdd_full'] and main['n_closed'] == R0['n_closed']),
           A2_paths_bitwise=bool(all(acc_paths[p] == ref['paths']['ALL'][p]['since2019'] for p in ref['paths']['ALL']) and len(ref['paths']['ALL']) == 20),
           paths_median_since2019=float(np.median(list(acc_paths.values()))))
acc['ok'] = bool(A0 and all(all(v.values()) for v in A1.values()) and acc['A2_main_bitwise'] and acc['A2_paths_bitwise'])
print('驗收', json.dumps(acc, ensure_ascii=False), flush=True)
assert acc['ok'], '驗收失敗：停止、不報結果'

keys = list(M['CUR'])
summ = {}
for c in L.CANDS:
    v = np.array([M[c][k]['train'] for k in keys])
    o = dict(train_median=float(np.median(v)), train_min=float(v.min()), train_max=float(v.max()),
             train_dd_median=float(np.median([M[c][k]['train_dd'] for k in keys])),
             y1618_median=float(np.median([M[c][k]['y1618'] for k in keys])), y1921_median=float(np.median([M[c][k]['y1921'] for k in keys])),
             n_closed_train_median=float(np.median([M[c][k]['n_closed_train'] for k in keys])))
    if c != 'CUR':
        d = np.array([M[c][k]['train'] - M['CUR'][k]['train'] for k in keys]) * 100
        dd = np.array([M[c][k]['train_dd'] - M['CUR'][k]['train_dd'] for k in keys]) * 100
        d1 = np.array([M[c][k]['y1618'] - M['CUR'][k]['y1618'] for k in keys]) * 100
        d2 = np.array([M[c][k]['y1921'] - M['CUR'][k]['y1921'] for k in keys]) * 100
        o.update(score_median_pp=float(np.median(d)), npos=int((d > 1e-9).sum()), n_ge025=int((d >= 0.25).sum()), diff_min_pp=float(d.min()), diff_max_pp=float(d.max()),
                 dd_diff_median_pp=float(np.median(dd)), n_dd_shallower=int((dd > 0).sum()), y1618_median_pp=float(np.median(d1)), y1618_npos=int((d1 > 1e-9).sum()),
                 y1921_median_pp=float(np.median(d2)), y1921_npos=int((d2 > 1e-9).sum()))
    summ[c] = o
cand = ['C1', 'C2', 'C3']
order = sorted(cand, key=lambda c: (-summ[c]['score_median_pp'], -summ[c]['n_ge025'], cand.index(c)))
top = order[0]; ts = summ[top]
pick = top if ts['score_median_pp'] >= 1.0 else 'CUR'
t0ok = bool(pick != 'CUR' and ts['score_median_pp'] >= 1.0 and ts['n_ge025'] >= 14)
res = dict(SPEC_sha256=sha(H / 'SPEC.json'), PREREG_sha256=sha(H / 'PREREG.json'), acceptance=acc,
           window=dict(train=[str(TA.date()), str(TB.date())], y1618=[str(TA.date()), str(T18.date())], y1921=[str(T18.date()), str(TB.date())]),
           paths=M, summary=summ, selection=dict(order=order, top_score=top, frozen=pick, T0_pass=t0ok,
                                                 text=(f'凍結 {top}' if pick != 'CUR' else f'最高分 {top} 未達 +1 個百分點 → 凍結「不改」')),
           meta=dict(runtime_s=round(time.time() - t0, 1), maxrss_mb=rss()))
(H / 'results_train.json').write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str))
FZ = dict(written_at=datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S'), SPEC_sha256=res['SPEC_sha256'], PREREG_sha256=res['PREREG_sha256'],
          frozen=pick, frozen_desc=L.CAND_DESC[pick], validate_target=top, validate_target_desc=L.CAND_DESC[top],
          T0_pass=t0ok, train_score=ts, order=order,
          note='凍結後 validate.py 只跑一次；不得換候選。frozen＝CUR（不改）或 T0 不過時，validate_target 只作描述，總判定必為「不改」。')
txt = json.dumps(FZ, ensure_ascii=False, indent=1, default=str)
(H / 'FROZEN.json').write_text(txt)
fsha = hashlib.sha256(txt.encode()).hexdigest()
(H / 'FROZEN.sha256').write_text(f'{fsha}  FROZEN.json\n')
code = {p.name: sha(p) for p in sorted(H.glob('*.py'))}
(H / 'code.sha256').write_text(''.join(f'{v}  {k}\n' for k, v in code.items()))
for c in L.CANDS:
    print(c, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in summ[c].items()}, flush=True)
print('選擇', json.dumps(res['selection'], ensure_ascii=False), 'FROZEN', fsha, res['meta'])
