#!/usr/bin/env python3
"""S79 訓練（2008-01～2015-12）：第一階段全部試驗 × 20 路徑 → 挑選 → 第二階段合併 → FROZEN.json＋sha256。
每條淨值一出引擎就截到 2015-12-31，之後的數字不看、不存。python3 -I -B train.py → train_results.json、FROZEN.json、FROZEN.sha256"""
import sys, json, hashlib, time, itertools
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert sha(HERE / 'PREREG.json') == (HERE / 'PREREG.sha256').read_text().split()[0]
for line in (HERE / 'code.sha256').read_text().strip().split('\n'):
    h, f = line.split(); assert sha(HERE / f) == h, f
assert not (HERE / 'FROZEN.json').exists(), '已凍結'
import numpy as np, pandas as pd
import s79lib as Q, trials as TR
from s79lib import K

t0 = time.time()
P, U, F, RT = Q.load(); FE = Q.features(P, U, F)
B = Q.bench(P)
Bt = B.loc[:Q.TRAIN_END]


def evaluate(spec):
    res = Q.run_paths(P, U, RT, FE, spec)
    ey = []; cx = []
    for (cap, s), (nav, lg, st, sg) in res.items():
        nav = nav.loc[:Q.TRAIN_END]                          # 截斷：只看訓練窗
        b = Bt / Bt.iloc[0] * cap
        ey.append(Q.yearly_excess(nav, b, Q.TRAIN_YEARS))
        cx.append((K.cagr(nav) - K.cagr(b)) * 100)
        if cap == 1e6 and s == 0:
            w = Q.window(nav, b, *Q.TRAIN)
            lgt = lg[lg['ent_date'] <= Q.TRAIN_END]
            std = dict(cagr=w['cagr'], cagr_0050=w['cagr_0050'], ex_ann_pp=w['ex_ann_pp'], t_nw6=w['t_nw6'], IR=w['IR'], maxdd=w['maxdd'], maxdd_0050=w['maxdd_0050'],
                       trades=int(len(lgt)),
                       avg_idle=float(1 - st['sf'].loc[:Q.TRAIN_END].dropna().mean()),
                       mean_targets=float(np.mean([i['n_tgt'] for i in sg['info'] if i['sig_date'] <= '2015-12-31'])))
    ex_year = {y: float(np.median([e[y] for e in ey])) for y in Q.TRAIN_YEARS}
    v = np.array(list(ex_year.values()))
    return dict(ex_year=ex_year, J=float((np.median(v) + v.min()) / 2), ex_med=float(np.median(v)), ex_min=float(v.min()), cagr_ex=float(np.median(cx)),
                cagr_ex_min=float(np.min(cx)), cagr_ex_max=float(np.max(cx)), **std)


T = TR.build(); NB = TR.neighbors(T)
RES = {}
for i, t in enumerate(T):
    RES[t['id']] = evaluate(t['spec'])
    if i % 10 == 0:
        print(i, t['id'], round(RES[t['id']]['J'], 2), round(RES[t['id']]['cagr_ex'], 2), f'{time.time() - t0:.0f}s', flush=True)
base = RES['F0.base.current']
for t in T:
    r = RES[t['id']]; nb = NB[t['id']]
    r['J_adj'] = min(r['J'], float(np.mean([RES[n]['J'] for n in nb]))) if nb else r['J'] - 1.0
    if t['id'] == 'F0.base.current':
        r['J_adj'] = r['J']


def eligible(r):
    return bool(r['cagr_ex'] >= base['cagr_ex'] + 2.0 and r['trades'] >= 0.5 * base['trades'] and r['J_adj'] >= base['J'] + 1.0)


for t in T:
    RES[t['id']]['eligible'] = eligible(RES[t['id']]) and t['fam'] != 'F0'
el = [t for t in T if RES[t['id']]['eligible']]
key = lambda t: (-RES[t['id']]['J_adj'], t['id'])
best1 = sorted(el, key=key)[0] if el else None
# ---- 第二階段
fam_best = {}
for t in sorted(el, key=key):
    fam_best.setdefault(t['fam'], t)
top = sorted(fam_best.values(), key=key)[:4]
T2 = []
DEF = TR.DEF_PG
if len(top) >= 2:
    for n in (2, 3, 4):
        for comb in itertools.combinations(top, n):
            scs = [c['spec']['score'] for c in comb if 'score' in c['spec']]
            if len(scs) > 1:
                continue
            spec = {}
            lgs = [g for c in comb for g in c['spec'].get('list_gates', [])]
            if lgs:
                spec['list_gates'] = lgs
            pgs = []
            for c in comb:
                for g in c['spec'].get('pool_gates', []):
                    if 'pool_gates' in c['spec'] and g not in pgs:
                        pgs.append(g)
            if pgs:
                spec['pool_gates'] = pgs
            if scs:
                spec['score'] = scs[0]
            ic = [c['spec']['ind_cap'] for c in comb if 'ind_cap' in c['spec']]
            if ic:
                spec['ind_cap'] = min(ic)
            T2.append(dict(id='C.' + '+'.join(c['id'] for c in comb), fam='C', members=[c['id'] for c in comb], spec=spec))
for t in T2:
    r = evaluate(t['spec']); r['J_adj'] = r['J'] - 1.0; r['eligible'] = eligible(r); RES[t['id']] = r
    print('stage2', t['id'], round(r['J'], 2), round(r['cagr_ex'], 2), flush=True)
win = best1
if best1 is not None:
    c2 = sorted([t for t in T2 if RES[t['id']]['eligible'] and RES[t['id']]['J_adj'] >= RES[best1['id']]['J_adj'] + 1.0], key=key)
    if c2:
        win = c2[0]
out = dict(n_stage1=len(T), n_stage2=len(T2), n_trials=len(T) + len(T2), base=base, results=RES, stage2=[dict(id=t['id'], members=t['members'], spec=t['spec']) for t in T2],
           eligible=[t['id'] for t in sorted(el, key=key)], stage2_families=[t['id'] for t in top], winner=win['id'] if win else None, runtime_s=round(time.time() - t0, 1))
(HERE / 'train_results.json').write_text(json.dumps(out, ensure_ascii=False, indent=1))
fz = dict(frozen_rule=win['id'] if win else 'F0.base.current', is_new_rule=win is not None, spec=win['spec'] if win else {},
          train_stats=RES[win['id']] if win else base, base_train_stats=base,
          all_trial_specs={**{t['id']: t['spec'] for t in T}, **{t['id']: t['spec'] for t in T2}},
          PREREG_sha256=sha(HERE / 'PREREG.json'), train_results_sha256=sha(HERE / 'train_results.json'),
          note='凍結後才跑 validate.py（2016-01～2026-10 只跑一次，不回頭改）')
(HERE / 'FROZEN.json').write_text(json.dumps(fz, ensure_ascii=False, indent=1))
(HERE / 'FROZEN.sha256').write_text(sha(HERE / 'FROZEN.json') + '  FROZEN.json\n')
print('winner', fz['frozen_rule'], 'eligible', len(el), 'stage2', len(T2), 'base J', base['J'], 'cagr_ex', base['cagr_ex'], f'{time.time() - t0:.0f}s')
