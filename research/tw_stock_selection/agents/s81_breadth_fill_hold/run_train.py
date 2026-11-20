#!/usr/bin/env python3
"""S81 訓練段（2008-01～2015-12）：192 格 × 2 版本，只算、只存訓練段統計；依 PREREG 的挑選準則機械式挑一格，寫 FROZEN.json＋sha256。
python3 -I -B run_train.py → train_grid.csv、train_monthly_excess.csv.gz、FROZEN.json、FROZEN.sha256"""
import sys, json, time, hashlib, resource
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
HERE = Path(__file__).resolve().parent
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert sha(HERE / 'PREREG.json') == (HERE / 'PREREG.sha256').read_text().split()[0]
for line in (HERE / 'code.sha256').read_text().strip().split('\n'):
    h, f = line.split(); assert sha(HERE / f) == h, f
assert not (HERE / 'FROZEN.json').exists(), 'FROZEN.json 已存在'
import numpy as np, pandas as pd
import s81lib as Q
from s81lib import K, R
t0 = time.time()
P, U = K.load_cont()
F, RT = K.surge(P, 'B2')
D0 = P.IDX[P.t0]
bench = pd.Series(P.E50, index=P.IDX).loc[D0:]; bench = bench / bench.iloc[0] * 1e6
cells = Q.grid(); nb = Q.neighbors(cells)
a, b = Q.TRAIN
rows = []; MEX = {}
sgc = {}
for i, c in enumerate(cells):
    key = (c['depth'], c['mom'], c['keep'], c['hold'])
    if key not in sgc:
        sgc = {key: Q.cell_signal(P, U, RT, c)}          # 只留一份（省記憶體）；格子順序讓同訊號的相鄰
    sg = sgc[key]
    r = dict(id=c['id'], idx=i, cap=c['cap'], depth=c['depth'], mom=c['mom'], exit=c['exit'], keep=c['keep'], hold=c['hold'])
    for ver, odd in (('LOT', False), ('ODD', True)):
        nav, lg, st = Q.simulate(P, sg, U, c['cap'], odd=odd)
        rs, rb = R.monthly_returns(nav, bench.reindex(nav.index))
        blk = Q.block(rs, rb, nav, a, b)
        ab = Q.acct_block(P, lg, st, nav, a, b)
        for k_ in ('cagr', 'ex_ann_pp', 't_nw6', 'IR', 'te_ann_pp', 'maxdd', 'worst_year_ex_pp', 'worst_year'):
            r[f'{ver}_{k_}'] = blk[k_]
        for k_ in ('avg_npos', 'stockfrac', 'n_entries', 'turnover_buy_per_year', 'avg_hold_days'):
            r[f'{ver}_{k_}'] = ab[k_]
        MEX[f"{c['id']}#{ver}"] = (Q.win(rs, a, b) - Q.win(rb, a, b))
        del nav, lg, st
    rows.append(r)
    if i % 24 == 0:
        print(i, c['id'], f'{time.time() - t0:.0f}s', flush=True)
G = pd.DataFrame(rows)
# ---- 挑選（PREREG selection）
G['IRm'] = G[['LOT_IR', 'ODD_IR']].min(axis=1); G['Wm'] = G[['LOT_worst_year_ex_pp', 'ODD_worst_year_ex_pp']].min(axis=1)
G['p_IR'] = Q.pct_rank(G['IRm'].to_numpy()); G['p_W'] = Q.pct_rank(G['Wm'].to_numpy())
G['raw'] = (G['p_IR'] + G['p_W']) / 2
raw = dict(zip(G['id'], G['raw']))
G['nb_mean'] = [float(np.mean([raw[x] for x in nb[i_]])) for i_ in G['id']]
G['n_nb'] = [len(nb[i_]) for i_ in G['id']]
G['score'] = 0.5 * G['raw'] + 0.5 * G['nb_mean']
cur = G[G['id'] == Q.CUR].iloc[0]
G['elig_spike'] = (G['raw'] - G['nb_mean']) <= 0.20
G['elig_beat_cur'] = (G['LOT_ex_ann_pp'] > cur['LOT_ex_ann_pp']) & (G['ODD_ex_ann_pp'] > cur['ODD_ex_ann_pp'])
G['eligible'] = G['elig_spike'] & G['elig_beat_cur']
E = G[G['eligible']].sort_values(['score', 'idx'], ascending=[False, True])
G.to_csv(HERE / 'train_grid.csv', index=False)
pd.DataFrame(MEX).to_csv(HERE / 'train_monthly_excess.csv.gz')
if len(E):
    w = E.iloc[0]; cid = w['id']
    froz = dict(frozen=cid, spec={k: (int(w[k]) if k in ('cap', 'depth', 'keep', 'hold') else w[k]) for k in ('cap', 'depth', 'mom', 'exit', 'keep', 'hold')},
                train={k: (float(w[k]) if not isinstance(w[k], str) else w[k]) for k in G.columns if k not in ('id', 'mom', 'exit')},
                runner_up=[dict(id=x['id'], score=float(x['score'])) for _, x in E.iloc[1:6].iterrows()])
else:
    froz = dict(frozen=None, note='沒有合格格，維持現行')
froz.update(prereg_sha256=sha(HERE / 'PREREG.json'), n_cells=len(G), n_eligible=int(len(E)), current_train={k: float(cur[k]) for k in G.columns if k.startswith(('LOT_', 'ODD_')) and not k.endswith('worst_year')} | dict(raw=float(cur['raw']), score=float(cur['score'])),
            train_grid_sha256=sha(HERE / 'train_grid.csv'), written_before_validation=True)
(HERE / 'FROZEN.json').write_text(json.dumps(froz, ensure_ascii=False, indent=1, default=str))
(HERE / 'FROZEN.sha256').write_text(f"{sha(HERE / 'FROZEN.json')}  FROZEN.json\n")
print(json.dumps(froz, ensure_ascii=False, indent=1, default=str))
print('done', f'{time.time() - t0:.0f}s', round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024), 'MB')
