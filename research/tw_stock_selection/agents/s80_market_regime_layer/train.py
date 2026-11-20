#!/usr/bin/env python3
"""S80 訓練（2008-01-02～2015-12-31；引擎只跑到 2015-12-31）：66 格 × 20 條路徑 → results_train.json；照 PREREG 的準則挑格 → FROZEN.json＋FROZEN.sha256。
results_train.json 或 FROZEN.json 已存在就拒跑。python3 -I -B train.py"""
import sys, json, time, hashlib, datetime
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
assert not (HERE / 'results_train.json').exists() and not (HERE / 'FROZEN.json').exists(), '訓練只跑一次'
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert sha(HERE / 'PREREG.json') == (HERE / 'PREREG.sha256').read_text().split()[0]
for line in (HERE / 'code.sha256').read_text().strip().split('\n'):
    h, f = line.split(); assert sha(HERE / f) == h, f'{f} 與登記的 sha256 不符'
import numpy as np, pandas as pd
import s80lib as S
from s80lib import K
t0 = time.time()
PRE = json.loads((HERE / 'PREREG.json').read_text())
assert PRE['search_space']['cells'] == [c['id'] for c in S.CELLS] and PRE['engine']['src_sha'] == S.SRC_SHA
P, U = K.load_cont()
F, RT = K.surge(P, 'B2')
IND, br = S.indicators(P, U); WE = S.week_ends(P)
tend = int(np.searchsorted(P.IDX.values, np.datetime64(S.TRAIN_END), side='right'))
A, B = P.IDX[P.t0], P.IDX[tend - 1]
assert str(B.date()) == S.TRAIN_END
bench = pd.Series(P.E50, index=P.IDX).loc[A:]; bench = bench / bench.iloc[0] * 1e6
ids = [S.CUR] + [c['id'] for c in S.CELLS]
PATH = {i: {} for i in ids}; STD = {}
for s in S.SHIFTS:
    sg = K.signal(P, U, RT, s)
    for cid in ids:
        if cid == S.CUR:
            sg2, cm, szm, st = sg, None, None, None
        else:
            sg2, cm, szm, st, ch = S.apply_rule(P, U, sg, S.CELL[cid], IND, WE)
        for cap in S.CAPS:
            nav, lg, sx = S.sim(P, sg2, U, cap=cap, cashmode=cm, szm=szm, tend=tend)
            c, d = S.cagr(nav), S.maxdd(nav)
            PATH[cid][f'{int(cap / 1e4)}萬|+{s}'] = dict(cagr=c, maxdd=d, calmar=c / abs(d), n_closed=int((lg['why'] != 'open_end').sum()))
            if s == 0 and cap == 1e6:
                w = S.win_stats(nav, bench, A, B); w.update(S.flat_stats(P, lg, sx, A, B))
                w['yearly'] = {int(y['year']): y['strat'] for y in K.yearly_table(nav, bench.reindex(nav.index))}
                w['bear_state_day_frac'] = float(st[P.t0:tend].mean()) if st is not None else 0.0
                STD[cid] = w
    print('shift', s, f'{time.time() - t0:.0f}s', flush=True)
keys = list(PATH[S.CUR])
g = lambda cid, f: np.array([PATH[cid][k][f] for k in keys])
SUM = {}
for cid in ids:
    o = dict(cagr_med=float(np.median(g(cid, 'cagr'))), maxdd_med=float(np.median(g(cid, 'maxdd'))), C=float(np.median(g(cid, 'calmar'))), n_closed_med=float(np.median(g(cid, 'n_closed'))))
    if cid != S.CUR:
        o['dCAGR'] = float(np.median(g(cid, 'cagr') - g(S.CUR, 'cagr')) * 100)
        o['dMDD'] = float(np.median(g(cid, 'maxdd') - g(S.CUR, 'maxdd')) * 100)
        o['nC'] = int((g(cid, 'calmar') > g(S.CUR, 'calmar') + 1e-12).sum())
        o['nCAGR_pos'] = int((g(cid, 'cagr') > g(S.CUR, 'cagr') + 1e-12).sum())
        o['eligible'] = bool(o['dCAGR'] >= -1.0 and o['dMDD'] >= 3.0 and o['nC'] >= 15)
    SUM[cid] = o
for c in S.CELLS:
    nb = S.neighbors(c['id'])
    SUM[c['id']]['neighbors'] = nb
    SUM[c['id']]['RS'] = float(min(SUM[c['id']]['C'], np.mean([SUM[n]['C'] for n in nb]))) if nb else SUM[c['id']]['C']
order = [c['id'] for c in S.CELLS]
elig = [i for i in order if SUM[i]['eligible']]
best_all = max(order, key=lambda i: (SUM[i]['RS'], -order.index(i)))
pick = max(elig, key=lambda i: (SUM[i]['RS'], -order.index(i))) if elig else None
clear = bool(pick and SUM[pick]['dMDD'] >= 10 and SUM[pick]['dCAGR'] >= 0 and SUM[pick]['nC'] >= 15)
out = dict(window=[str(A.date()), str(B.date())], n_cells=len(order), current=SUM[S.CUR], summary=SUM, std_path=STD, paths=PATH, eligible=elig, pick=pick, best_RS_overall=best_all,
           clearly_better_in_train=clear, sha=dict(PREREG=sha(HERE / 'PREREG.json')), runtime_s=round(time.time() - t0, 1))
(HERE / 'results_train.json').write_text(json.dumps(out, ensure_ascii=False, indent=1))
fz = dict(written_at=datetime.datetime.now(datetime.timezone.utc).isoformat(), frozen=(pick or S.CUR), reference_cell=(pick or best_all), n_eligible=len(elig),
          rule=(S.CELL[pick] if pick else None), reference_rule=S.CELL[pick or best_all], train=SUM[pick or best_all], train_current=SUM[S.CUR], clearly_better_in_train=clear,
          PREREG_sha256=sha(HERE / 'PREREG.json'), results_train_sha256=sha(HERE / 'results_train.json'), code_sha256=(HERE / 'code.sha256').read_text().strip().split('\n'),
          note='validate.py 只跑一次；門檻與報告項目見 PREREG.json validation')
(HERE / 'FROZEN.json').write_text(json.dumps(fz, ensure_ascii=False, indent=1))
(HERE / 'FROZEN.sha256').write_text(f"{sha(HERE / 'FROZEN.json')}  FROZEN.json\n")
print('現行', SUM[S.CUR]); print('合格', len(elig), '凍結', pick, '全體 RS 最高', best_all, SUM[pick or best_all])
