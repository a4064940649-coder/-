#!/usr/bin/env python3
"""S41 訓練期：反事實（一直是 15 日期限）下 A、B、C × 20 條路徑，只用訓練窗（2016-01-29 → 2021-12-30）選擇；B、C 對 A。
另存（描述，不參與選擇）：同一批全期連續模擬的全期指標、對 CUR 的距離（建議規則要用；只在驗證期跑完後才讀）。
輸出 results_train.json、FROZEN.json、FROZEN.sha256。PREREG sha256 須相符；FROZEN.json 已存在就拒絕。python3 -I -B train.py"""
import sys, json, hashlib, time, resource, gc, datetime
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
import s41lib as L
from s41lib import s17lib, s10lib, stlib

H = Path(__file__).resolve().parent
pre = (H / 'PREREG.json').read_bytes(); sha = hashlib.sha256(pre).hexdigest()
assert sha == (H / 'PREREG.sha256').read_text().split()[0], 'PREREG.json 與 PREREG.sha256 不符'
assert not (H / 'FROZEN.json').exists(), 'FROZEN.json 已存在，拒絕重跑'
MIN_IMP = 0.25
t_all = time.time()
rss = lambda: round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
P, U, RT = L.prep_panel()
S0, av = RT['S'], RT['av']
lastb = lambda d: P.IDX[P.IDX < pd.Timestamp(d)][-1]
TA, TB = lastb('2016-02-01'), lastb('2022-01-01')
D19, DE, D0 = lastb('2019-01-01'), P.IDX[-1], P.IDX[P.t0]
CAPS = (6e5, 8e5, 1e6, 1.5e6, 2e6)
CFG = ['CUR', 'A', 'B', 'C']


def met(nav):
    return dict(train=stlib.cagr(nav, TA, TB), train_dd=stlib.maxdd(nav, TA, TB), since2019=stlib.cagr(nav, D19, DE), full=stlib.cagr(nav, D0, DE),
                maxdd_full=stlib.maxdd(nav, D0, DE), seg={k: v for k, v in s17lib.seg_dd(nav).items() if k in ('2018', '2020-03')})


M = {c: {} for c in CFG}
eff = {}
acc = {}
for s in range(4):
    cells = L.hidden_cells(P, av, S0, s)
    SG = {'CUR': s10lib.make_signal_s10(P, U, RT, shift=s)}
    for c in 'ABC':
        SG[c], lg = L.build(P, U, RT, c, cells, s)
        if c == 'B':
            eff[f'+{s}'] = dict(n_cells=len(cells), B_keep_events=lg['n_keepable'],
                                B_keep_events_train=sum(1 for x in lg['keep_cells'] if pd.Period(x[0], 'M') <= L.TRAIN_LAST_MONTH))
    for cap in CAPS:
        key = f'{int(cap/1e4)}萬|+{s}'
        navs = {}
        for c in CFG:
            nav, lg, st = L.sim(P, U, SG[c], cap)
            M[c][key] = met(nav)
            navs[c] = nav.to_numpy()
            del lg, st
        if s == 0 and cap == 1e6:                          # A2：B、C 第一個與 A 不同的日子
            for c in 'BC':
                d = np.flatnonzero(navs[c] != navs['A'])
                acc[f'A2_{c}_first_diff_vs_A'] = str(P.IDX[P.t0 + d[0]].date()) if len(d) else None
            d = np.flatnonzero(navs['A'] != navs['CUR'])
            acc['A_first_diff_vs_CUR'] = str(P.IDX[P.t0 + d[0]].date()) if len(d) else None
    del SG
    gc.collect()
    print('位移', s, f'{time.time()-t_all:.0f}s', rss(), 'MB', flush=True)
keys = list(M['CUR'])
res = {'PREREG_sha256': sha, 'acceptance': acc, 'window': dict(train=[str(TA.date()), str(TB.date())]), 'cells_and_B_keeps': eff, 'paths': M, 'summary': {}}
for c in CFG:
    o = dict(train_median=float(np.median([M[c][k]['train'] for k in keys])), train_dd_median=float(np.median([M[c][k]['train_dd'] for k in keys])))
    for b in ('A', 'CUR'):
        if c == b:
            continue
        d = np.array([M[c][k]['train'] - M[b][k]['train'] for k in keys]) * 100
        dd = np.array([M[c][k]['train_dd'] - M[b][k]['train_dd'] for k in keys]) * 100
        o[f'vs_{b}'] = dict(score_median_pp=float(np.median(d)), n_imp_ge_min=int((d >= MIN_IMP).sum()), npos=int((d > 1e-12).sum()), nneg=int((d < -1e-12).sum()),
                            diff_min_pp=float(d.min()), diff_max_pp=float(d.max()), dd_diff_median_pp=float(np.median(dd)),
                            by_shift_median_pp={f'+{s}': float(np.median([(M[c][k]['train'] - M[b][k]['train']) * 100 for k in keys if k.endswith(f'+{s}')])) for s in range(4)})
    res['summary'][c] = o
cand = ['B', 'C']
order = sorted(cand, key=lambda c: (res['summary'][c]['vs_A']['score_median_pp'], res['summary'][c]['vs_A']['n_imp_ge_min'], c == 'B'), reverse=True)
best = order[0]; bs = res['summary'][best]['vs_A']
t0 = bool(bs['score_median_pp'] >= 1.0 and bs['n_imp_ge_min'] >= 14)
res['selection'] = dict(order=order, frozen=best, T0_pass=t0, min_improvement_pp=MIN_IMP,
                        T0_text='訓練期找到明顯優於現狀的處理' if t0 else '訓練期沒有找到（仍凍結分數最高者，驗證期只描述、判定不通過）')
res['meta'] = dict(runtime_s=round(time.time() - t_all, 1), maxrss_mb=rss())
(H / 'results_train.json').write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str))
FZ = dict(written_at=datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S'), PREREG_sha256=sha, frozen=best, frozen_desc=L.TREATS[best], T0_pass=t0,
          T0_text=res['selection']['T0_text'], train_score=bs, note='凍結後 validate.py 只跑一次；不得換處理或門檻。')
txt = json.dumps(FZ, ensure_ascii=False, indent=1, default=str)
(H / 'FROZEN.json').write_text(txt)
fsha = hashlib.sha256(txt.encode()).hexdigest()
(H / 'FROZEN.sha256').write_text(f'{fsha}  FROZEN.json\n')
for c in CFG:
    print(c, json.dumps(res['summary'][c], ensure_ascii=False), flush=True)
print('A2', acc, 'eff', eff)
print('凍結', best, 'T0', t0, fsha, res['meta'])
